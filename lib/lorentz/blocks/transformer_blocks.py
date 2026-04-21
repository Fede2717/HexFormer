import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lib.geoopt import ManifoldParameter

from lib.utils.drop_path import DropPath

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import (
    LorentzFullyConnected,
    LorentzLayerNorm,
    LorentzProjection,
LorentzAct
)


class LorentzEmbedding(nn.Module):
    def __init__(self, manifold: CustomLorentz, hidden_dim, patch_dim, num_tokens):
        super(LorentzEmbedding, self).__init__()
        self.manifold = manifold
        self.patch_embed = LorentzFullyConnected(self.manifold, patch_dim, hidden_dim)
        self.cls_token = ManifoldParameter(self.manifold.random_normal(1, 1, hidden_dim), manifold=self.manifold)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim-1))

    def forward(self, x):
        x = self.manifold.projx(F.pad(x, pad=(1, 0)))

        x = self.patch_embed(x)

        if torch.isnan(x).sum()>0 or torch.isinf(x).sum()>0:
            print("break")

        x = torch.cat([self.cls_token.repeat(x.size(0),1,1), x], dim=1)
        x = x.narrow(-1, 1, x.shape[-1]-1) + self.pos_embed

        return self.manifold.add_time(x)


class LorentzTransformerEncoder(nn.Module):
    def __init__(self, manifold: CustomLorentz, hidden, mlp_hidden, num_patches, heads, dropout,
                 stochastic_depth=0.1, use_haa=False):
        super(LorentzTransformerEncoder, self).__init__()

        self.manifold = manifold

        self.hidden = hidden
        self.mlp_hidden = mlp_hidden
        self.num_patches = num_patches
        self.heads = heads
        self.dropout = dropout

        self.ln1 = LorentzLayerNorm(manifold, hidden)
        self.mha = LorentzMultiHeadAttention(manifold, hidden, num_patches, heads, dropout, use_haa=use_haa)
        self.ln2 = LorentzLayerNorm(manifold, hidden)
        self.mlp = nn.Sequential(
            LorentzFullyConnected(manifold, hidden, mlp_hidden, activation=nn.GELU(), dropout=dropout),
            LorentzFullyConnected(manifold, mlp_hidden, hidden, dropout=dropout),
        )

        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

    def forward(self, x):
        out = self.mha(self.ln1(x))
        out = self.drop_path(out.narrow(-1, 1, x.shape[-1]-1)) + x.narrow(-1, 1, x.shape[-1]-1)
        out = self.drop_path(self.mlp(self.ln2(self.manifold.add_time(out))).narrow(-1, 1, x.shape[-1]-1)) + out
        out = self.manifold.add_time(out)
        return out


class LorentzMultiHeadAttention(nn.Module):
    def __init__(self, manifold: CustomLorentz, num_features, num_patches, heads, dropout=0.0,
                 learn_scale=False, use_haa=False):
        super(LorentzMultiHeadAttention, self).__init__()

        self.manifold = manifold

        self.num_features = num_features
        self.num_patches = num_patches
        self.heads = heads
        self.head_dim = (num_features-1)//heads
        self.temperature = nn.Parameter(torch.ones(1))
        self.scale = nn.Parameter(self.head_dim**(-0.5)*torch.ones((1, heads, 1, 1)), requires_grad=learn_scale)

        self.softmax = nn.Softmax(dim=-1)

        self.q = LorentzFullyConnected(manifold, num_features, num_features, nheads=heads, bias=False)
        self.k = LorentzFullyConnected(manifold, num_features, num_features, nheads=heads, bias=False)
        self.v = LorentzFullyConnected(manifold, num_features, num_features, nheads=heads, bias=False)

        self.o = LorentzFullyConnected(manifold, num_features, num_features, dropout=dropout)

        self.use_haa = use_haa
        self.layer_idx = -1
        self.max_layer_idx = 8
        self._grad_norms = {}

        if use_haa:
            # τ₀=0.1: Phase 0 showed z_mean≈+0.80 everywhere; low τ lets spatial
            # penalty (which varies across pairs) dominate early training.
            self.beta_raw   = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1.0)]))
            self.tau_raw    = nn.Parameter(torch.tensor([math.log(math.exp(0.1) - 1.0)]))
            self.lambda_raw = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1.0)]))

            self.haa_alpha            = 0.0
            self.haa_tau              = 0.0
            self.haa_lambda           = 0.0
            self.haa_mean_c_tilde     = 0.0
            self.haa_mean_B           = 0.0
            self.haa_mean_Z           = 0.0
            self.haa_cone_sparsity    = 0.0
            self.haa_frac_near_origin = 0.0

    def lorentz_expmap_aggregation(self, v, score):
        v_tangent = self.manifold.logmap0(v)
        weighted_v_tangent = torch.matmul(score, v_tangent)
        sum_weights = score.sum(dim=-1, keepdim=True)
        mean_v_tangent = weighted_v_tangent / (sum_weights + 1e-8)
        mean_v = self.manifold.expmap0(mean_v_tangent)
        return mean_v

    def forward(self, x):
        b, n, l = x.size()

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        if self.use_haa:
            q_time  = q.narrow(-1, 0, 1)
            k_time  = k.narrow(-1, 0, 1)
            q_space = q.narrow(-1, 1, q.shape[-1] - 1)
            k_space = k.narrow(-1, 1, k.shape[-1] - 1)

            sqrt_k = torch.sqrt(self.manifold.k)
            K = self.manifold.k

            # --- Lorentz inner products ---
            inner_QK = (-q_time @ k_time.transpose(-1, -2)
                        + q_space @ k_space.transpose(-1, -2))
            inner_OQ = -sqrt_k * q_time                    # [b,h,n,1]
            inner_OK = -sqrt_k * k_time.transpose(-1, -2)  # [b,h,1,n]

            # --- Angular signal Z (inner-product formulation, no HLoC) ---
            norm_sq_QK = (inner_QK.pow(2) / K - K).clamp_min(0.0)
            norm_sq_OQ = (inner_OQ.pow(2) / K - K).clamp_min(0.0)
            numer_Z    = inner_OK + (inner_QK * inner_OQ) / K
            denom_Z    = torch.sqrt(norm_sq_QK * norm_sq_OQ + (5e-3)**2)
            Z_safe     = (numer_Z / denom_Z).clamp(-1.0, 1.0)

            if self.training and torch.isnan(Z_safe).any():
                import warnings
                warnings.warn(
                    f"[HAA layer {self.layer_idx}] NaN in Z_safe — "
                    f"inner_QK max: {inner_QK.abs().max().item():.2e}")
                Z_safe = torch.zeros_like(Z_safe)

            # --- Radial depth and aperture B ---
            c_tilde = torch.acosh((q_time / sqrt_k).clamp_min(1.0 + 1e-3))
            beta    = F.softplus(self.beta_raw)
            sinh_c  = torch.sinh(c_tilde)
            arg_B   = 1.0 - (beta / sinh_c).pow(2)
            B       = torch.sqrt(F.relu(arg_B) + 1e-8)

            # Gradient hooks for diagnostic layers only (first and last HAA layer)
            if self.training and self.layer_idx in (0, self.max_layer_idx):
                c_tilde.register_hook(
                    lambda g: self._grad_norms.update({'c_tilde': g.norm().item()}))
                B.register_hook(
                    lambda g: self._grad_norms.update({'B': g.norm().item()}))

            # --- Spatial penalty (soft-clamped log-cosh, δ₀=15) ---
            lam         = F.softplus(self.lambda_raw)
            safe_cosh_d = (-inner_QK / K).clamp_min(1.0 + 1e-3)
            dist_raw    = sqrt_k * torch.acosh(safe_cosh_d)
            d_soft      = 40.0 * torch.tanh(dist_raw / 40.0)
            scaled_d    = d_soft / 15.0
            H           = scaled_d + F.softplus(-2.0 * scaled_d) - math.log(2.0)
            spatial_pen = -lam * H

            # --- Entailment penalty (margin-softplus, m=0.1 fixed) ---
            tau       = F.softplus(self.tau_raw)
            _m        = 0.1
            _sp_neg_m = math.log(1.0 + math.exp(-_m))  # precomputed scalar constant
            Phi       = F.softplus((B + Z_safe) - _m) - _sp_neg_m
            entail_pen = -tau * Phi

            # --- Score and attention weights ---
            score_matrix = spatial_pen + entail_pen
            score = self.softmax(score_matrix / self.temperature)

            if not self.training:
                frac_near_origin          = (arg_B < 0).float().mean().item()
                self.haa_frac_near_origin = frac_near_origin
                self.haa_alpha            = beta.item()
                self.haa_tau              = tau.item()
                self.haa_lambda           = lam.item()
                self.haa_mean_c_tilde     = c_tilde.mean().item()
                self.haa_mean_B           = B.mean().item()
                self.haa_mean_Z           = Z_safe.mean().item()
                self.haa_cone_sparsity    = ((B + Z_safe) <= 0).float().mean().item()
        else:
            dists = -self.manifold.csqdist(q, k) * self.scale.expand((b, self.heads, 1, 1))
            score = self.softmax(dists / self.temperature)

        attn = self.lorentz_expmap_aggregation(v, score).permute(0, 2, 1, 3)

        # Lorentz direct concatenation of heads
        attn_space = attn.narrow(-1, 1, attn.shape[-1]-1).reshape(b, n, -1)
        attn_time = attn.narrow(-1, 0, 1).reshape(b, n, -1)
        time_sq_arg = torch.sum(attn_time ** 2, dim=-1, keepdim=True) - ((self.heads - 1) * self.manifold.k)
        attn_time_rescaled = torch.sqrt(time_sq_arg.clamp_min(1e-8))
        attn = torch.concat((attn_time_rescaled, attn_space), dim=-1)

        o = self.o(attn)
        return o
