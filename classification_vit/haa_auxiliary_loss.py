"""Auxiliary losses for breaking the HAA geometric deadlock.

Implements the Loss Factory pattern of Master_Execution_Pipeline.md Rule 2.
All auxiliary loss logic is contained in this module; the training loop
only calls ``build_aux_losses(args)`` once and iterates the returned dict.

Modules:
    RampSchedule          — warmup/ramp/plateau weight scheduler
    ConeOccupancyLoss     — band-hinge surrogate for cone occupancy
    HyperbolicHierarchyLoss — depth-stratification metric loss (HHL)
    build_aux_losses      — factory mapping CLI flags to loss instances
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from cifar100_hierarchy import FINE_TO_SUPER, NUM_FINE, NUM_SUPER


# ---------------------------------------------------------------------------
# Helper: locate the deepest HAA layer's MultiHeadAttention module.
# ---------------------------------------------------------------------------
def _get_last_haa_mha(model):
    """Return the LorentzMultiHeadAttention with use_haa=True at the
    highest layer_idx in the (possibly DataParallel-wrapped) model."""
    base = model.module if hasattr(model, 'module') else model
    last_mha = None
    last_idx = -1
    for _, m in base.named_modules():
        if getattr(m, 'use_haa', False):
            idx = getattr(m, 'layer_idx', -1)
            if idx >= last_idx:
                last_mha = m
                last_idx = idx
    return last_mha


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
class RampSchedule:
    """Warmup → linear ramp → plateau weight schedule.

    Behaviour:
        epoch < warmup                              -> 0.0
        warmup <= epoch < warmup + ramp             -> linear ramp
        epoch >= warmup + ramp                      -> plateau
    """
    def __init__(self, warmup: int, ramp: int, plateau: float):
        self.warmup = warmup
        self.ramp = ramp
        self.plateau = plateau

    def __call__(self, epoch: int) -> float:
        if epoch < self.warmup:
            return 0.0
        if epoch < self.warmup + self.ramp:
            return self.plateau * (epoch - self.warmup) / max(1, self.ramp)
        return self.plateau


# ---------------------------------------------------------------------------
# Cone Occupancy Loss
# ---------------------------------------------------------------------------
class ConeOccupancyLoss(nn.Module):
    """Smooth surrogate for cone occupancy with band-hinge target.

    s_cone = mean( σ( κ · ( -Z - m_smooth ) ) )

    Penalises when s_cone falls outside the band [s_lo, s_hi].
    """
    def __init__(self,
                 s_lo: float = 0.10,
                 s_hi: float = 0.50,
                 kappa: float = 4.0,
                 m_smooth: float = 0.05,
                 warmup: int = 15,
                 ramp: int = 25,
                 plateau: float = 0.3):
        super().__init__()
        self.s_lo = s_lo
        self.s_hi = s_hi
        self.kappa = kappa
        self.m_smooth = m_smooth
        self.schedule = RampSchedule(warmup, ramp, plateau)

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None or mha._last_B is None or mha._last_Z is None:
            return torch.zeros((), device=device)
        Z = mha._last_Z
        # REPLACED -(B+Z) WITH -Z: Breaks the cheap beta-collapse gradient path.
        # Occupancy must be satisfied by moving Z structurally, not inflating beta.
        cone_score = -Z - self.m_smooth
        s_cone = torch.sigmoid(self.kappa * cone_score).mean()
        loss_lo = F.relu(self.s_lo - s_cone).pow(2)
        loss_hi = F.relu(s_cone - self.s_hi).pow(2)
        return loss_lo + loss_hi


# ---------------------------------------------------------------------------
# Hyperbolic Hierarchy Loss (HHL)
# ---------------------------------------------------------------------------
class HyperbolicHierarchyLoss(nn.Module):
    """Enforces super-class shallower than fine-class via margin hinge.

    Depth d_L(x, O) on the Lorentz hyperboloid with curvature K is
        sqrt(K) · acosh( x_0 / sqrt(K) ).
    Computed on the CLS token's time coordinate at the final HAA layer.
    Uses EMA centroid smoothing across batches for stability.
    """
    def __init__(self,
                 K: float,
                 margin: float = 0.3,
                 ema: float = 0.9,
                 warmup: int = 5,
                 ramp: int = 25,
                 plateau: float = 0.5):
        super().__init__()
        self.K = K
        self.sqrt_K = K ** 0.5
        self.margin = margin
        self.ema = ema
        self.schedule = RampSchedule(warmup, ramp, plateau)

        # EMA centroid buffers (non-persistent — reset on checkpoint reload).
        self.register_buffer('super_centroids',
                             torch.zeros(NUM_SUPER), persistent=False)
        self.register_buffer('fine_centroids',
                             torch.zeros(NUM_FINE), persistent=False)
        # Lazy-init flag (Python attribute, not persistent).
        self._initialized = False

        # Fine -> super lookup table as a buffer so it moves with .to(device).
        _lut = torch.zeros(NUM_FINE, dtype=torch.long)
        for fine, sup in FINE_TO_SUPER.items():
            _lut[fine] = sup
        self.register_buffer('fine_to_super_lut', _lut, persistent=False)

    def _depth(self, time_coord: torch.Tensor) -> torch.Tensor:
        return self.sqrt_K * torch.acosh(
            (time_coord / self.sqrt_K).clamp_min(1.0 + 1e-3))

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None or getattr(mha, '_last_cls_time', None) is None:
            return torch.zeros((), device=device)

        cls_time = mha._last_cls_time  # [b, 1]
        if cls_time.dim() > 2:
            cls_time = cls_time.reshape(cls_time.shape[0], -1)[:, 0:1]
        depths = self._depth(cls_time).squeeze(-1)  # [b]

        super_labels = self.fine_to_super_lut[y]

        # Per-class mean depth this batch.
        super_mean = torch.zeros(NUM_SUPER, device=depths.device)
        fine_mean = torch.zeros(NUM_FINE, device=depths.device)
        super_count = torch.zeros(NUM_SUPER, device=depths.device)
        fine_count = torch.zeros(NUM_FINE, device=depths.device)
        super_mean.scatter_add_(0, super_labels, depths)
        fine_mean.scatter_add_(0, y, depths)
        super_count.scatter_add_(0, super_labels, torch.ones_like(depths))
        fine_count.scatter_add_(0, y, torch.ones_like(depths))
        super_mean = super_mean / super_count.clamp_min(1.0)
        fine_mean = fine_mean / fine_count.clamp_min(1.0)

        # EMA update — first call seeds centroids only on observed classes.
        with torch.no_grad():
            mask_super = super_count > 0
            mask_fine = fine_count > 0
            if not self._initialized:
                self.super_centroids[mask_super] = super_mean[mask_super].detach()
                self.fine_centroids[mask_fine] = fine_mean[mask_fine].detach()
                self._initialized = True
            else:
                self.super_centroids[mask_super] = (
                    self.ema * self.super_centroids[mask_super]
                    + (1.0 - self.ema) * super_mean[mask_super].detach()
                )
                self.fine_centroids[mask_fine] = (
                    self.ema * self.fine_centroids[mask_fine]
                    + (1.0 - self.ema) * fine_mean[mask_fine].detach()
                )

        # NOTE: This fix produces non-zero gradient but does NOT guarantee
        # directional depth stratification. Both tensors are derived from the
        # same batch with no external reference; the gradient is non-zero but
        # stochastic in sign across batches. Directional radial supervision
        # requires the prototype loss (L_proto) introduced in the next iteration.
        # This fix is diagnostic — it confirms whether HHL contributes anything.
        #
        # Map per-fine-class mean depths (computed above) to their superclasses,
        # then average. This is class-weighted, mathematically distinct from
        # super_mean (which is token-weighted across the whole superclass).
        observed_fine_idx = mask_fine.nonzero(as_tuple=False).squeeze(-1)
        super_of_observed_fine = self.fine_to_super_lut[observed_fine_idx]
        fine_mean_per_super  = torch.zeros(NUM_SUPER, device=depths.device)
        fine_classes_per_super = torch.zeros(NUM_SUPER, device=depths.device)
        fine_mean_per_super.scatter_add_(0, super_of_observed_fine,
                                         fine_mean[observed_fine_idx])
        fine_classes_per_super.scatter_add_(
            0, super_of_observed_fine,
            torch.ones_like(fine_mean[observed_fine_idx]))
        fine_mean_per_super = (fine_mean_per_super
                               / fine_classes_per_super.clamp_min(1.0))

        # Hinge: super centroid must be shallower than fine centroid
        # by at least margin. Active only for superclasses present in batch.
        mask = (super_count > 0) & (fine_classes_per_super > 0)
        if not mask.any():
            return torch.zeros((), device=device)

        loss = F.relu(
            super_mean[mask] - fine_mean_per_super[mask] + self.margin
        ).pow(2).mean()
        return loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_aux_losses(args):
    """Construct auxiliary loss modules controlled by CLI flags.

    Returns a dict ``{'occupancy': ..., 'hhl': ...}`` where each entry is
    either an ``nn.Module`` or ``None`` if disabled (gamma_max/eta_max == 0).
    """
    out = {'occupancy': None, 'hhl': None}

    gamma_max = float(getattr(args, 'gamma_max', 0.0))
    if gamma_max > 0:
        out['occupancy'] = ConeOccupancyLoss(
            warmup=int(getattr(args, 'gamma_warmup', 15)),
            ramp=25,
            plateau=gamma_max,
        )

    eta_max = float(getattr(args, 'eta_max', 0.0))
    if eta_max > 0:
        K = float(getattr(args, 'encoder_k', 1.0))
        out['hhl'] = HyperbolicHierarchyLoss(
            K=K,
            warmup=int(getattr(args, 'eta_warmup', 5)),
            ramp=25,
            plateau=eta_max,
        )

    return out
