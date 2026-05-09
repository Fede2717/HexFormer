import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from haa_auxiliary_loss import build_hyperbolic_prototypes
from cifar100_hierarchy import FINE_TO_SUPER, NUM_FINE, NUM_SUPER


class LorentzPrototypeClassifier(nn.Module):
    """Prototype-softmax classification head on the Lorentz hyperboloid.
    Logits = -d_L^2(cls, p_c) / T, c in fine prototypes.
    Replaces LorentzMLR for Design 2 of the Phase 2 pipeline.
    """
    def __init__(self,
                 manifold,
                 hidden_dim_lorentz: int,
                 num_classes: int,
                 K: float = 1.0,
                 proto_seed: int = 42,
                 d_s: float = 0.3,
                 d_f_low: float = 0.5,
                 d_f_high: float = 1.85,
                 T_init: float = 1.0):
        super().__init__()
        assert num_classes == NUM_FINE, \
            f"LorentzPrototypeClassifier requires num_classes == NUM_FINE; got {num_classes}"
        self.manifold = manifold
        self.K = K
        self.num_super = NUM_SUPER
        self.num_fine = NUM_FINE

        lut = torch.zeros(NUM_FINE, dtype=torch.long)
        for f, s in FINE_TO_SUPER.items():
            lut[f] = s
        protos = build_hyperbolic_prototypes(
            num_super=NUM_SUPER,
            num_fine=NUM_FINE,
            hidden_dim=hidden_dim_lorentz,
            fine_to_super_lut=lut,
            K=K,
            seed=proto_seed,
            d_s=d_s,
            d_f_low=d_f_low,
            d_f_high=d_f_high,
        )
        self.register_buffer('prototypes', protos, persistent=False)

        # Positive temperature via softplus(log_T).
        # log_T = log(exp(T_init) - 1) gives softplus(log_T) = T_init at init.
        _log_T_init = math.log(math.exp(T_init) - 1.0)
        self.log_T = nn.Parameter(torch.tensor([_log_T_init]))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_T) + 1e-3

    def forward(self, cls_lorentz: torch.Tensor) -> torch.Tensor:
        # cls_lorentz: [B, hidden_dim+1] post-final_ln Lorentz CLS.
        fine_protos = self.prototypes[self.num_super:
                                      self.num_super + self.num_fine]   # [F, D]
        # Lorentz inner product <cls, p>_L = -t_cls*t_p + s_cls·s_p
        inner = (-cls_lorentz[..., 0:1] * fine_protos[..., 0:1].T
                 + cls_lorentz[..., 1:] @ fine_protos[..., 1:].T)        # [B, F]
        arg = -inner / self.K
        u = (arg - 1.0).clamp_min(0.0)
        sqrt_term = torch.sqrt(u * (2.0 + u) + 1e-12)
        d2 = torch.log1p(u + sqrt_term).pow(2)                           # [B, F]
        T = self.temperature
        return -d2 / T
