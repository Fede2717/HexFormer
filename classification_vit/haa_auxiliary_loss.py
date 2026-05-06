"""Auxiliary losses for breaking the HAA geometric deadlock.

Implements the Loss Factory pattern of Master_Execution_Pipeline.md Rule 2.
All auxiliary loss logic is contained in this module; the training loop
only calls ``build_aux_losses(args)`` once and iterates the returned dict.

Modules:
    RampSchedule              — warmup/ramp/plateau weight scheduler
    DirectionalAngularLoss    — band-hinge surrogate for angular ordering
    HyperbolicHierarchyLoss   — depth-stratification metric loss (HHL)
    HyperbolicPrototypeLoss   — Lorentz-distance to fixed fine prototypes
    RadialVarianceLoss        — one-sided hinge on radial-depth variance
    BetaCapLoss               — cap β to the geometric envelope of the batch
    build_hyperbolic_prototypes — fixed-prototype constructor for L_proto
    build_aux_losses          — factory mapping CLI flags to loss instances
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from cifar100_hierarchy import FINE_TO_SUPER, NUM_FINE, NUM_SUPER


# ---------------------------------------------------------------------------
# Prototype constructor (P5 / P6)
# ---------------------------------------------------------------------------
def build_hyperbolic_prototypes(num_super: int,
                                num_fine: int,
                                hidden_dim: int,
                                fine_to_super_lut: torch.Tensor,
                                K: float = 1.0,
                                seed: int = 42,
                                d_s: float = 0.3,
                                d_f_low: float = 0.5,
                                d_f_high: float = 1.85) -> torch.Tensor:
    """Construct fixed Lorentz prototypes for L_proto.

    Returns a tensor of shape [num_super + num_fine, hidden_dim].
    Indices [0 : num_super] are super-prototypes (shallow, depth d_s).
    Indices [num_super : num_super + num_fine] are fine-prototypes
    (deep, depth d_s + d_f[c], inside parent's tan(pi/8) angular cap).

    Super angular positions: closed-form simplex ETF. C = num_super vectors
    in spatial space with all pairwise cosines = -1/(C-1). Maximally and
    uniformly separated by construction. Dataset-independent: depends only
    on (num_super, num_fine, spatial_dim, fine_to_super_lut).

    Fine angular positions: for each fine class c with super(c) = s, sample
    delta in R^spatial_dim, project onto the orthogonal complement of v_s,
    scale to tan(pi/8) magnitude, renormalise (v_s + delta_perp) to unit
    norm. Each fine prototype lies inside its parent's angular cap.

    Final spatial = unit_vec * sinh(depth); time = sqrt(K) * cosh(depth).
    """
    g = torch.Generator().manual_seed(seed)
    sqrt_K = K ** 0.5
    spatial_dim = hidden_dim - 1

    # --- super angles via closed-form simplex ETF ---
    # Mettes 2019 / neural-collapse canonical maximum-separation construction.
    C = num_super
    if spatial_dim < C - 1:
        raise ValueError(
            f"spatial_dim={spatial_dim} < num_super-1={C-1}; "
            "simplex ETF requires spatial_dim >= num_super - 1.")
    U_full = torch.randn(spatial_dim, C, generator=g)
    U, _ = torch.linalg.qr(U_full)                           # [spatial_dim, C], semi-orthogonal
    M_simplex = (math.sqrt(C / (C - 1.0)) *
                 (torch.eye(C) - (1.0 / C) * torch.ones(C, C)))
    super_angles = (U @ M_simplex).T                         # [C, spatial_dim]
    super_angles = super_angles / super_angles.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # --- fine angles within parent super's angular cap ---
    cap_tan = torch.tan(torch.tensor(math.pi / 8.0))
    fine_angles = torch.zeros(num_fine, spatial_dim)
    for c in range(num_fine):
        s = int(fine_to_super_lut[c].item())
        v_s = super_angles[s]
        delta = torch.randn(spatial_dim, generator=g)
        delta_perp = delta - (delta @ v_s) * v_s
        delta_perp = delta_perp / delta_perp.norm().clamp_min(1e-8)
        w = v_s + cap_tan.item() * delta_perp
        w = w / w.norm().clamp_min(1e-8)
        fine_angles[c] = w

    # --- depths ---
    super_depths = torch.full((num_super,), d_s)
    d_f = torch.empty(num_fine).uniform_(d_f_low, d_f_high, generator=g)
    fine_depths = d_s + d_f

    # --- assemble Lorentz points ---
    def assemble(angles, depths):
        sinh_d = torch.sinh(depths).unsqueeze(-1)
        cosh_d = torch.cosh(depths).unsqueeze(-1)
        spatial = angles * sinh_d
        time = sqrt_K * cosh_d
        return torch.cat([time, spatial], dim=-1)

    super_protos = assemble(super_angles, super_depths)
    fine_protos = assemble(fine_angles, fine_depths)
    return torch.cat([super_protos, fine_protos], dim=0)


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
class DirectionalAngularLoss(nn.Module):
    """Smooth surrogate for angular ordering with band-hinge target.

    Despite this loss being historically labelled "cone occupancy", the
    forward computation does NOT measure volumetric cone occupancy: it does
    not reference the aperture B at all. It measures angular ordering on the
    hyperboloid — the soft fraction of (Q, K) pairs in which K lies in the
    angular half-space "deeper" than Q (i.e., Z < 0 at vertex Q), regardless
    of the half-angle β subtended by Q's entailment cone.

    Surrogate:
        s_angular = mean( σ( κ · ( -Z - m_smooth ) ) )

    A pair contributes ≈ 1 when Z < 0 (K geometrically deeper than Q at
    vertex Q) and ≈ 0 when Z > 0 (sibling/shallower geometry). The earlier
    formulation cone_score = -(B + Z) - m_smooth was replaced by -Z to block
    a β-collapse loophole: the optimizer was satisfying the constraint by
    inflating β rather than moving tokens, producing trivial-looking
    occupancy with no geometric content.

    The band [s_lo, s_hi] is enforced on this angular fraction: if it falls
    outside the target window the squared hinge penalty kicks in.

    P1: Reduction is computed over geometrically valid Q-K pairs only
    (those with norm_sq_QK > 1e-6 AND norm_sq_OQ > 1e-6). Masked pairs are
    excluded entirely; they do not drag the mean toward the band-satisfying
    value σ(−0.2) ≈ 0.45 that would otherwise auto-satisfy the loss when
    the masking rate is high.
    """
    def __init__(self,
                 s_lo: float = 0.40,
                 s_hi: float = 0.70,
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
        if mha is None or getattr(mha, '_last_Z', None) is None or getattr(mha, '_last_valid_mask', None) is None:
            return torch.zeros((), device=device)
        Z = mha._last_Z
        valid = getattr(mha, '_last_valid_mask', None)
        if valid is None or not valid.any():
            return torch.zeros((), device=device)
        # REPLACED -(B+Z) WITH -Z: Breaks the cheap beta-collapse gradient path.
        # Occupancy must be satisfied by moving Z structurally, not inflating beta.
        cone_score = -Z - self.m_smooth
        sigmoid_vals = torch.sigmoid(self.kappa * cone_score)
        valid_f = valid.float()
        n_valid = valid_f.sum().clamp_min(1.0)
        s_cone = (sigmoid_vals * valid_f).sum() / n_valid
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
# Hyperbolic Prototype Loss (P5)
# ---------------------------------------------------------------------------
class HyperbolicPrototypeLoss(nn.Module):
    """Squared Lorentz distance from post-attention CLS to fixed FINE prototype.

    FIX-PROTO-DEPTH (REVISED, Alternative 1): pull CLS toward the
    fine-grained prototype at depth d_s + d_f[y], NOT the superclass
    prototype at d_s. Reasoning: the classification head needs CLS deep
    enough to discriminate ``num_fine`` classes; pulling CLS to the
    shallow superclass depth compresses the classification angular
    budget. The HAA depth-ordering requirement (Q-CLS shallower than
    K-patch) is handled separately by an in-attention alpha_q control;
    L_proto's job here is to supervise the post-encoder CLS for
    classification, not to drive HAA geometry.

    Math (validated):
      inner = -<cls, p_y>_L
      arg   = inner / K
      u     = max(arg - 1, 0)
      acosh = log1p(u + sqrt(u·(2+u) + 1e-12))   [closed-form, no torch.acosh]
      L     = mean(acosh²)

    The closed-form identity arccosh(1+u) = log1p(u + sqrt(u(2+u))) avoids
    the 1/sqrt(arg-1) singularity in the standard arccosh derivative.
    Combined with the outer ()², the gradient at coincident points is
    exactly zero (validated under static gradient checks).
    """
    def __init__(self,
                 K: float,
                 prototypes_lorentz: torch.Tensor,
                 num_super: int,
                 warmup: int = 5,
                 ramp: int = 25,
                 plateau: float = 0.5):
        super().__init__()
        self.K = K
        self.num_super = num_super
        self.register_buffer('prototypes', prototypes_lorentz, persistent=False)
        self.schedule = RampSchedule(warmup, ramp, plateau)

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None or getattr(mha, '_last_cls_lorentz', None) is None:
            return torch.zeros((), device=device)
        cls = mha._last_cls_lorentz                          # [B, hidden_dim+1]
        # FIX-PROTO-DEPTH (REVISED, Alternative 1):
        # Pull post-attention CLS toward the fine-grained prototype at
        # depth d_s + d_f[y], NOT the superclass prototype at d_s.
        # Reasoning: the classification head needs CLS deep enough to
        # discriminate 100 fine classes. Pulling CLS to the shallow
        # superclass depth (0.3) compresses the classification angular
        # budget. The HAA depth-ordering requirement (Q-CLS shallower than
        # K-patch) is handled separately by the in-attention alpha_q
        # control planned for Step 11; L_proto's job is to supervise the
        # post-encoder CLS for classification, not to drive HAA geometry.
        # Fine-prototypes occupy indices [num_super : num_super + num_fine]
        # in self.prototypes.
        target_proto = self.prototypes[self.num_super + y]   # [B, hidden_dim+1]
        inner = (-cls[..., 0:1] * target_proto[..., 0:1]
                 + (cls[..., 1:] * target_proto[..., 1:]).sum(-1, keepdim=True))
        arg = -inner / self.K
        u = (arg - 1.0).clamp_min(0.0)
        sqrt_term = torch.sqrt(u * (2.0 + u) + 1e-12)
        acosh_val = torch.log1p(u + sqrt_term)
        return acosh_val.pow(2).mean()


# ---------------------------------------------------------------------------
# Radial Variance Loss (P5)
# ---------------------------------------------------------------------------
class RadialVarianceLoss(nn.Module):
    """One-sided hinge on within-image radial-depth variance at the
    pre-MHA layer-8 input.

    Math (validated):
      c_tilde = sqrt(K) · log1p(u + sqrt(u·(2+u) + 1e-12))   with u = max(x_0/sqrt(K) - 1, 0)
      σ²_per_image = Var_n(c_tilde)         [population variance over tokens of one image]
      σ²_batch     = mean over images
      L            = ReLU(σ²_target - σ²_batch)²

    The variance is computed across tokens within each image at the
    pre-MHA layer-8 input, then averaged across the batch. HAA's score
    formula derives c_tilde from Q = W_Q . token_input, so the c_tilde
    distribution HAA actually consumes is fixed by the INPUT depth
    distribution; supervising the post-block output would let the MLP
    satisfy L_radvar without ever stratifying what HAA sees.
    Population variance (unbiased=False) avoids the 1/(n-1) blowup at
    n=2. Below target: linear-in-deficit gradient. Above target: zero
    (saturated). Token count < 2 is guarded.
    """
    def __init__(self,
                 K: float,
                 sigma2_target: float = 0.10,
                 warmup: int = 5,
                 ramp: int = 25,
                 plateau: float = 0.5):
        super().__init__()
        self.K = K
        self.sqrt_K = K ** 0.5
        self.sigma2_target = sigma2_target
        self.schedule = RampSchedule(warmup, ramp, plateau)

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None or getattr(mha, '_last_all_tokens_lorentz', None) is None:
            return torch.zeros((), device=device)
        # Pre-MHA layer-8 input tokens. HAA's score formula computes
        # c_tilde from Q = W_Q . token_input, so the c_tilde distribution
        # HAA actually consumes is determined by the INPUT depth
        # distribution. Supervising the post-block output would let the
        # MLP satisfy L_radvar without ever stratifying what HAA sees.
        tokens = mha._last_all_tokens_lorentz   # [B, n, hidden_dim+1]
        if tokens.shape[0] < 1 or tokens.shape[1] < 2:
            return torch.zeros((), device=device)
        time = tokens[..., 0]                   # [B, n]
        arg = time / self.sqrt_K
        u = (arg - 1.0).clamp_min(0.0)
        sqrt_term = torch.sqrt(u * (2.0 + u) + 1e-12)
        c_tilde = self.sqrt_K * torch.log1p(u + sqrt_term)      # [B, n]
        sigma2_per_image = c_tilde.var(dim=-1, unbiased=False)  # [B]
        return F.relu(self.sigma2_target - sigma2_per_image.mean()).pow(2)


# ---------------------------------------------------------------------------
# Beta-Cap Loss (P5)
# ---------------------------------------------------------------------------
class BetaCapLoss(nn.Module):
    """Cap β to the geometric envelope of the batch.

    Math (validated):
      target = sinh(quantile(c_tilde_batch, q))   [DETACHED — no grad through]
      L = mean over heads of ReLU(β_h - target)²

    Path A (dynamic, default): target recomputed from each batch's c̃ percentile.
    Path B (static): target frozen at value passed via static_target argument.

    Gradient flows only into β_raw via softplus parameterisation; the
    quantile is detached so the non-smooth order-statistic gradient
    never propagates. Saturated (β ≤ target): grad = 0. batch < 4 guarded.
    """
    def __init__(self,
                 K: float,
                 percentile: float = 0.25,
                 static_target: float = None,
                 warmup: int = 20,
                 ramp: int = 20,
                 plateau: float = 0.3):
        super().__init__()
        self.K = K
        self.sqrt_K = K ** 0.5
        self.percentile = percentile
        self.static_target = static_target
        self.schedule = RampSchedule(warmup, ramp, plateau)

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None or getattr(mha, '_last_all_tokens_lorentz', None) is None:
            return torch.zeros((), device=device)

        beta = F.softplus(mha.beta_raw)  # [H] after Stage 1.1, else [1]

        if self.static_target is not None:
            target = torch.sinh(torch.tensor(self.static_target, device=device))
        else:
            tokens = mha._last_all_tokens_lorentz  # [B, n, hidden_dim]
            time = tokens[..., 0:1]
            arg = time / self.sqrt_K
            u = (arg - 1.0).clamp_min(0.0)
            sqrt_term = torch.sqrt(u * (2.0 + u) + 1e-12)
            c_tilde = self.sqrt_K * torch.log1p(u + sqrt_term)
            c_flat = c_tilde.detach().flatten()
            if c_flat.numel() < 4:
                return torch.zeros((), device=device)
            target = torch.sinh(torch.quantile(c_flat, self.percentile))

        return F.relu(beta - target).pow(2).mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_aux_losses(args):
    """Construct auxiliary loss modules controlled by CLI flags.

    Returns a dict with keys 'angular', 'hhl', 'proto', 'radvar', 'betacap'
    where each entry is either an ``nn.Module`` or ``None`` if disabled
    (its corresponding *_max weight is 0). The training loop iterates the
    dict and skips None entries.
    """
    out = {'angular': None, 'hhl': None,
           'proto': None, 'radvar': None, 'betacap': None}

    gamma_max = float(getattr(args, 'gamma_angular_max', 0.0))
    if gamma_max > 0:
        out['angular'] = DirectionalAngularLoss(
            warmup=int(getattr(args, 'gamma_angular_warmup', 15)),
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

    eta_proto_max = float(getattr(args, 'eta_proto_max', 0.0))
    if eta_proto_max > 0:
        K = float(getattr(args, 'encoder_k', 1.0))
        prototypes = getattr(args, 'hyperbolic_prototypes', None)
        if prototypes is None:
            raise RuntimeError(
                "L_proto enabled but args.hyperbolic_prototypes not set. "
                "Build prototypes in the model setup before constructing aux losses.")
        out['proto'] = HyperbolicPrototypeLoss(
            K=K,
            prototypes_lorentz=prototypes,
            num_super=int(getattr(args, 'num_super', NUM_SUPER)),
            warmup=int(getattr(args, 'eta_proto_warmup', 5)),
            ramp=25,
            plateau=eta_proto_max,
        )

    zeta_radvar_max = float(getattr(args, 'zeta_radvar_max', 0.0))
    if zeta_radvar_max > 0:
        K = float(getattr(args, 'encoder_k', 1.0))
        out['radvar'] = RadialVarianceLoss(
            K=K,
            sigma2_target=float(getattr(args, 'sigma2_target', 0.10)),
            warmup=int(getattr(args, 'zeta_radvar_warmup', 5)),
            ramp=25,
            plateau=zeta_radvar_max,
        )

    xi_betacap_max = float(getattr(args, 'xi_betacap_max', 0.0))
    if xi_betacap_max > 0:
        K = float(getattr(args, 'encoder_k', 1.0))
        static_t = getattr(args, 'betacap_static_target', None)
        out['betacap'] = BetaCapLoss(
            K=K,
            percentile=float(getattr(args, 'betacap_percentile', 0.25)),
            static_target=float(static_t) if static_t is not None else None,
            warmup=int(getattr(args, 'xi_betacap_warmup', 20)),
            ramp=20,
            plateau=xi_betacap_max,
        )

    return out
