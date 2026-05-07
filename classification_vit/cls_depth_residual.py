"""CLS-only radial scaling residual (Stage 2.1 Path B revised).

Predicts a per-CLS scalar alpha in (0.1, 1.5) via a 2-layer MLP:
    alpha = 0.1 + 1.4 * sigmoid(MLP(cls_spatial))   # FIX-ALPHA-SIGMOID
Range (0.1, 1.5) — bounded sigmoid eliminates the unbounded softplus tail
that produced an α=80 excursion in the post-Step-8 v3.1 analysis. The
sigmoid is centred so its sensitivity peaks near α=0.8, the regime the
optimizer should explore.

Scales cls_spatial by alpha, then re-projects onto the Lorentz hyperboloid
via manifold.add_time. Manifold constraint preserved exactly: <x_new, x_new>_L = -K.

Init: MLP final layer weight=0; bias = logit(0.6) = log(0.6 / 0.4) ≈ 0.405,
giving alpha = 0.1 + 1.4 * sigmoid(0.405) = 0.1 + 1.4 * 0.6 = 1.0 at every
CLS at init.   # FIX-ALPHA-SIGMOID

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
        # FIX-ALPHA-SIGMOID: with alpha = 0.1 + 1.4 * sigmoid(raw), bias =
        # logit(0.6) = log(0.6 / 0.4) ≈ 0.405 gives alpha = 0.1 + 1.4 * 0.6
        # = 1.0 at every CLS at init.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, math.log(0.6 / 0.4))   # ≈ 0.405; gives α=1.0 at init
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
        # FIX-ALPHA-SIGMOID: bounded range (0.1, 1.5) eliminates the
        # unbounded softplus tail that produced an α=80 excursion.
        alpha = 0.1 + 1.4 * torch.sigmoid(raw)            # [B] in (0.1, 1.5)

        # Eval-time snapshot of alpha for telemetry. During training we
        # also capture the value (overwritten every step) AND register a
        # hook on alpha to capture its gradient L2 norm for the next
        # backward pass.
        self._last_alpha = alpha.detach()
        if self.training and alpha.requires_grad:
            alpha.register_hook(
                lambda g: setattr(self, '_last_alpha_grad',
                                  g.detach().abs().mean().item()))

        cls_spatial_new = alpha.unsqueeze(-1) * cls_spatial   # [B, spatial_dim]
        cls_lorentz_new = self.manifold.add_time(cls_spatial_new)
        return cls_lorentz_new
