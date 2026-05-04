"""CLS-only radial scaling residual (Stage 2.1 Path B revised).

Predicts a per-CLS scalar alpha in [0.2, ∞) via a 2-layer MLP:
    alpha = 0.2 + softplus(MLP(cls_spatial))      # FIX-ALPHA-SOFTPLUS
Range [0.2, ∞) with a smooth floor at 0.2. d(softplus)/d(raw) = sigmoid(raw)
never saturates, so the optimizer can find any alpha ≥ 0.2 freely (the prior
1 + 0.8*tanh form saturated to its [0.2, 1.8] bounds and zeroed gradients).

Scales cls_spatial by alpha, then re-projects onto the Lorentz hyperboloid
via manifold.add_time. Manifold constraint preserved exactly: <x_new, x_new>_L = -K.

Init: MLP final layer weight=0; bias = softplus_inv(0.8) = log(exp(0.8) - 1)
≈ 0.2036, giving alpha = 0.2 + softplus(0.2036) = 0.2 + 0.8 = 1.0 at every
CLS.   # FIX-ALPHA-INIT

Telemetry exposed:
    self._last_alpha       - per-CLS alpha values [B] (float, eval-only detached snapshot)
    self._last_alpha_grad  - gradient L2 norm on alpha during last backward (scalar float)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLSDepthResidual(nn.Module):
    def __init__(self, manifold, hidden_dim_lorentz: int,
                 mlp_hidden: int = 64):
        """hidden_dim_lorentz: full Lorentz dim (= hidden_dim + 1).
        We operate on the spatial slice (size hidden_dim_lorentz - 1)."""
        super().__init__()
        self.manifold = manifold
        spatial_dim = hidden_dim_lorentz - 1
        self.mlp = nn.Sequential(
            nn.Linear(spatial_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )
        # FIX-ALPHA-INIT: with softplus, softplus(0) = ln(2) ≈ 0.693, so
        # zeroing the bias would give alpha_init = 0.2 + 0.693 ≈ 0.893, not
        # 1.0. Set bias = softplus_inv(0.8) = log(exp(0.8) - 1) ≈ 0.2036 so
        # that alpha = 0.2 + softplus(0.2036) = 1.0 at every CLS at init.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias,
                          math.log(math.exp(0.8) - 1.0))
        # Hidden layer: small random init.
        nn.init.normal_(self.mlp[0].weight, std=0.01)
        nn.init.zeros_(self.mlp[0].bias)

        # Telemetry buffers (Python attrs, not Parameters/buffers).
        self._last_alpha = None       # [B] float tensor, detached
        self._last_alpha_grad = 0.0   # scalar float

    def forward(self, cls_lorentz: torch.Tensor) -> torch.Tensor:
        """cls_lorentz: [B, hidden_dim+1] Lorentz point on the manifold.
        Returns [B, hidden_dim+1] Lorentz point on the manifold."""
        cls_spatial = cls_lorentz[..., 1:]                # [B, spatial_dim]
        raw = self.mlp(cls_spatial).squeeze(-1)           # [B]
        # FIX-ALPHA-SOFTPLUS: smooth floor at 0.2, no upper saturation.
        alpha = 0.2 + F.softplus(raw)                     # [B] in [0.2, ∞)

        # Eval-time snapshot of alpha for telemetry. During training we
        # also capture the value (overwritten every step) AND register a
        # hook on alpha to capture its gradient L2 norm for the next
        # backward pass.
        self._last_alpha = alpha.detach()
        if self.training and alpha.requires_grad:
            alpha.register_hook(
                lambda g: setattr(self, '_last_alpha_grad',
                                  g.detach().norm().item()))

        cls_spatial_new = alpha.unsqueeze(-1) * cls_spatial   # [B, spatial_dim]
        cls_lorentz_new = self.manifold.add_time(cls_spatial_new)
        return cls_lorentz_new
