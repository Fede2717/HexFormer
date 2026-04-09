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

class STE_Z_Clamp(torch.autograd.Function):
    """Straight-Through Estimator: clamps to [-1, 1] in forward, identity in backward."""
    @staticmethod
    def forward(ctx, x):
        return x.clamp(-1.0, 1.0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STE_Dist_Clamp(torch.autograd.Function):
    """Straight-Through Estimator: dtype-aware distance clamp in forward, identity in backward."""
    @staticmethod
    def forward(ctx, x):
        if x.dtype == torch.float16:
            return x.clamp(max=11.0)
        return x.clamp(max=40.0)  # bfloat16 or float32

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class LorentzEmbedding(nn.Module):
    def __init__(self, manifold: CustomLorentz, hidden_dim, patch_dim, num_tokens):
        super(LorentzEmbedding, self).__init__()
        self.manifold = manifold
        # self.patch_embed = LorentzProjection(self.manifold, patch_dim, hidden_dim)
        self.patch_embed = LorentzFullyConnected(self.manifold, patch_dim, hidden_dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.cls_token = ManifoldParameter(self.manifold.random_normal(1, 1, hidden_dim), manifold=self.manifold) # CLS token with hyperbolic randn?
        # self.pos_embed = ManifoldParameter(self.manifold.random_normal(1, num_tokens, hidden_dim-1), manifold=self.manifold)
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
    def __init__(self, manifold: CustomLorentz, hidden, mlp_hidden, num_patches, heads, dropout, stochastic_depth=0.1, is_last_layer=False):
        super(LorentzTransformerEncoder, self).__init__()

        self.manifold = manifold

        self.hidden = hidden
        self.mlp_hidden = mlp_hidden
        self.num_patches = num_patches
        self.heads = heads
        self.dropout = dropout

        self.ln1 = LorentzLayerNorm(manifold, hidden)
        self.mha = LorentzMultiHeadAttention(manifold, hidden, num_patches, heads, dropout, is_last_layer=is_last_layer)
        self.ln2 = LorentzLayerNorm(manifold, hidden)
        self.mlp = nn.Sequential(
            LorentzFullyConnected(manifold, hidden, mlp_hidden, activation=nn.GELU(), dropout=dropout), # ->internal gelu + dropout
            LorentzFullyConnected(manifold, mlp_hidden, hidden, dropout=dropout), # ->internal dropout
        )

        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

    def forward(self, x):
        out = self.mha(self.ln1(x))
        out = self.drop_path(out.narrow(-1, 1, x.shape[-1]-1)) + x.narrow(-1, 1, x.shape[-1]-1) # Residual connection
        out = self.drop_path(self.mlp(self.ln2(self.manifold.add_time(out))).narrow(-1, 1, x.shape[-1]-1)) + out
        out = self.manifold.add_time(out)
        return out

# expmap_aggregation
class LorentzMultiHeadAttention(nn.Module):
    def __init__(self, manifold: CustomLorentz, num_features, num_patches, heads, dropout=0.0, learn_scale=False, is_last_layer=False):
        super(LorentzMultiHeadAttention, self).__init__()

        self.manifold = manifold

        self.num_features = num_features
        self.num_patches = num_patches
        self.heads = heads
        self.head_dim = (num_features-1)//heads
        self.temperature = nn.Parameter(torch.ones(1))  # Initialize temperature
        self.scale = nn.Parameter(self.head_dim**(-0.5)*torch.ones((1, heads, 1, 1)), requires_grad=learn_scale)

        self.softmax = nn.Softmax(dim=-1)

        self.q = LorentzFullyConnected(manifold, num_features, num_features, nheads=heads, bias=False)
        self.k = LorentzFullyConnected(manifold, num_features, num_features, nheads=heads, bias=False)
        self.v = LorentzFullyConnected(manifold, num_features, num_features, nheads=heads, bias=False)

        self.o = LorentzFullyConnected(manifold, num_features, num_features, dropout=dropout)

        self.is_last_layer = is_last_layer
        if is_last_layer:
            self.alpha_raw  = nn.Parameter(torch.zeros(1))
            self.tau_raw    = nn.Parameter(torch.zeros(1))
            self.lambda_raw = nn.Parameter(torch.zeros(1))
            self.haa_alpha         = 0.0
            self.haa_tau           = 0.0
            self.haa_lambda        = 0.0
            self.haa_mean_c_tilde  = 0.0
            self.haa_mean_b_tilde  = 0.0
            self.haa_mean_B        = 0.0
            self.haa_mean_Z        = 0.0
            self.haa_cone_sparsity = 0.0
    
    def lorentz_expmap_aggregation(self, v, score):
        """
        Aggregate using exponential map: map to tangent space, aggregate, and map back.
        """
        
        v_tangent = self.manifold.logmap0(v)  # Shape: [128, 12, 65, 17]

        
        # Perform the weighted sum across tokens using `score` as weights
        weighted_v_tangent = torch.matmul(score, v_tangent)  # Shape: [128, 12, 65, 17]

        sum_weights = score.sum(dim=-1, keepdim=True)  # Shape: [128, 12, 65, 1] 
        # Here it doesn't matter for this case because the sum is 1 for the output of Softmax
        mean_v_tangent = weighted_v_tangent / (sum_weights + 1e-8)  # Shape: [128, 12, 65, 17]
    
        mean_v = self.manifold.expmap0(mean_v_tangent)  # Shape: [128, 12, 17]
        return mean_v

    def forward(self, x):
            b, n, l = x.size()

            # Internal Lorentz direct split
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)

            # 1. Separazione asse temporale e spaziale
            q_time = q.narrow(-1, 0, 1)
            k_time = k.narrow(-1, 0, 1)
            q_space = q.narrow(-1, 1, q.shape[-1] - 1)
            k_space = k.narrow(-1, 1, k.shape[-1] - 1)

            sqrt_k = torch.sqrt(self.manifold.k)
            K = self.manifold.k

            if self.is_last_layer:
                # ---- HAA V5 Scoring ----
                orig_dtype = q_time.dtype

                # 1. Pairwise Lorentz inner product
                inner_QK = -q_time @ k_time.transpose(-1, -2) + q_space @ k_space.transpose(-1, -2)

                # 2. Raw cosh (clamped >= 1.0 to prevent domain errors)
                raw_cosh_QK = (-inner_QK / K).clamp_min(1.0)

                # 3. Diagonal/Self-Attention Mask
                is_self_attn = (raw_cosh_QK < 1.0 + 1e-5)

                # 4. Safe acosh: feed 2.0 to diagonal to avoid infinite gradient at acosh(1.0)
                safe_cosh_QK = torch.where(is_self_attn, torch.tensor(2.0, device=raw_cosh_QK.device), raw_cosh_QK)
                dist_QK = sqrt_k * torch.acosh(safe_cosh_QK)

                # 5. Origin Distances and Scaled Depths
                cosh_OQ = (q_time / sqrt_k).clamp_min(1.0 + 1e-7)
                cosh_OK = (k_time / sqrt_k).clamp_min(1.0 + 1e-7)
                dist_OQ = (sqrt_k * torch.acosh(cosh_OQ)).clamp(max=40.0)

                c_tilde = dist_OQ / sqrt_k
                b_tilde = (sqrt_k * torch.acosh(cosh_OK)).clamp(max=40.0) / sqrt_k  # telemetry only

                # 6. Sinh terms (computed directly, not via sqrt-of-cosh²)
                sinh_QK = torch.sinh(dist_QK / sqrt_k)
                sinh_OQ = torch.sinh(c_tilde)

                # 7. HLoC Numerator — upcast to FP64 to prevent catastrophic cancellation
                numer = (
                    raw_cosh_QK.double() * cosh_OQ.double()
                    - cosh_OK.transpose(-1, -2).double()
                ).to(orig_dtype)
                denom = sinh_OQ * sinh_QK

                raw_Z = numer / denom

                # 8. STE clamp on raw_Z, then diagonal masking
                Z_clamped = STE_Z_Clamp.apply(raw_Z)
                Z_safe = torch.where(is_self_attn, torch.tensor(1.0, device=Z_clamped.device, dtype=Z_clamped.dtype), Z_clamped)

                # 9. Aperture B
                alpha = F.softplus(self.alpha_raw)
                B = (alpha * c_tilde) / (1.0 + alpha * c_tilde)

                # 10. Log-distance attraction term
                lam = F.softplus(self.lambda_raw)
                d_L = STE_Dist_Clamp.apply(dist_QK)
                log_dist_penalty = -lam * torch.log(1.0 + d_L ** 2)

                # 11. Margin-Softplus (Phi) entailment penalty — m = 0.1 (fixed)
                _m = 0.1
                _softplus_neg_m = F.softplus(torch.tensor(-_m, device=B.device, dtype=B.dtype))
                entailment_penalty = F.softplus((B + Z_safe) - _m) - _softplus_neg_m

                # 12. Final V5 score
                tau = F.softplus(self.tau_raw)
                score_matrix = log_dist_penalty - tau * entailment_penalty
                score = self.softmax(score_matrix)

                # ---- Telemetry ----
                self.haa_alpha         = alpha.item()
                self.haa_tau           = tau.item()
                self.haa_lambda        = lam.item()
                self.haa_mean_c_tilde  = c_tilde.mean().item()
                self.haa_mean_b_tilde  = b_tilde.mean().item()
                self.haa_mean_B        = B.mean().item()
                self.haa_mean_Z        = Z_safe.mean().item()
                self.haa_cone_sparsity = ((B + Z_safe) <= 0).float().mean().item()

            else:
                # ---- Existing tangent-space scoring ----

                # 2. Distanze dall'origine (Norme Iperboliche) con clamp per FP32
                norm_q = sqrt_k * torch.acosh(torch.clamp_min(q_time / sqrt_k, 1.0 + 1e-7))
                norm_k = sqrt_k * torch.acosh(torch.clamp_min(k_time / sqrt_k, 1.0 + 1e-7))

                norm_matrix = norm_q @ norm_k.transpose(-1, -2)

                # 3. Coseno dell'angolo
                dot_space = q_space @ k_space.transpose(-1, -2)
                norm_q_space = torch.norm(q_space, dim=-1, keepdim=True)
                norm_k_space = torch.norm(k_space, dim=-1, keepdim=True)

                denom_space = torch.clamp_min(norm_q_space @ norm_k_space.transpose(-1, -2), 1e-8)
                cos_theta = dot_space / denom_space

                # 4. Score Finale
                score_matrix = norm_matrix * cos_theta
                dists = score_matrix * self.scale.expand((b, self.heads, 1, 1))

                score = self.softmax(dists / self.temperature)

            attn = self.lorentz_expmap_aggregation(v, score).permute(0, 2, 1, 3)

            # Lorentz direct concatenation of heads
            attn_space = attn.narrow(-1, 1, attn.shape[-1]-1).reshape(b, n, -1)
            attn_time = attn.narrow(-1, 0, 1).reshape(b, n, -1)
            attn_time_rescaled = torch.sqrt(torch.sum(attn_time ** 2, dim=-1, keepdim=True) - ((self.heads - 1) * self.manifold.k))
            attn = torch.concat((attn_time_rescaled, attn_space), dim=-1)

            o = self.o(attn)
            return o
    
