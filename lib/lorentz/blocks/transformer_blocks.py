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
            self.tau_raw    = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1.0)]))
            self.lambda_raw = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1.0)]))
            init_val = math.log(math.exp(1.0) - 1.0)
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

                # 2. Spatial Metrics (Required for Log-Cosh Penalty and Aperture)
                raw_cosh_QK = (-inner_QK / K).clamp_min(1.0)
                dist_QK = sqrt_k * torch.acosh(raw_cosh_QK)
                
                cosh_OQ = (q_time / sqrt_k).clamp_min(1.0 + 1e-7)
                dist_OQ = sqrt_k * torch.acosh(cosh_OQ)
                c_tilde = dist_OQ / sqrt_k
                
                cosh_OK = (k_time / sqrt_k).clamp_min(1.0 + 1e-7)
                b_tilde = torch.acosh(cosh_OK) # telemetry only

                # 3. Direct Inner-Product Angle Formulation (Replaces HLoC)
                inner_OQ = -sqrt_k * q_time                              # Shape: [b,h,n,1]
                inner_OK = -sqrt_k * k_time.transpose(-1, -2)            # Shape: [b,h,1,n] (Transposed for broadcasting)

                # Squared Lorentz norms of direction vectors
                norm_sq_QK = (inner_QK.pow(2) / K - K).clamp_min(0.0)    # Shape: [b,h,n,n]
                norm_sq_OQ = (inner_OQ.pow(2) / K - K).clamp_min(0.0)    # Shape: [b,h,n,1]

                # Numerator: Lorentz inner product of the two direction vectors
                numer_Z = inner_OK + (inner_QK * inner_OQ) / K           # Shape: [b,h,n,n]

                # Denominator: product of Lorentz norms, regularised
                EPS_Z = 1e-7
                denom_Z = torch.sqrt((norm_sq_QK * norm_sq_OQ).clamp_min(EPS_Z ** 2))

                Z_raw = numer_Z / denom_Z
                
                # Z_safe naturally goes to ~0.0 for identical tokens, no mask needed
                Z_safe = Z_raw.clamp(-1.0, 1.0)

                # 4. Aperture B
                alpha = F.softplus(self.alpha_raw)
                B = (alpha * c_tilde) / (1.0 + alpha * c_tilde)
                
                # 5. Scaled Log-Cosh spatial penalty (delta_0 = 15.0)
                lam = F.softplus(self.lambda_raw)
                d_L = dist_QK.clamp(max=40.0)
                delta_0 = 15.0
                
                scaled_d = d_L / delta_0
                log_cosh_dist = scaled_d + F.softplus(-2.0 * scaled_d) - math.log(2.0)
                
                log_dist_penalty = -lam * log_cosh_dist 
                
                # 6. Margin-Softplus (Phi) entailment penalty — m = 0.1 (fixed)
                _m = 0.1
                # WARNING: If _m is ever promoted to a learnable parameter, this precomputation 
                # must be removed and replaced with F.softplus(-_m) to restore the gradient graph.
                _softplus_neg_m_val = math.log(1.0 + math.exp(-_m))
                entailment_penalty = F.softplus((B + Z_safe) - _m) - _softplus_neg_m_val

                # 7. Final V5 score
                tau = F.softplus(self.tau_raw)
                score_matrix = log_dist_penalty - tau * entailment_penalty
                score = self.softmax(score_matrix / self.temperature)
                
                # ---- Telemetry ----
                if not self.training:
                    self.haa_alpha         = alpha.item()
                    self.haa_tau           = tau.item()
                    self.haa_lambda        = lam.item()
                    self.haa_mean_c_tilde  = c_tilde.detach().mean().item()
                    self.haa_mean_b_tilde  = b_tilde.detach().mean().item()
                    self.haa_mean_B        = B.detach().mean().item()
                    self.haa_mean_Z        = Z_safe.detach().mean().item()
                    self.haa_cone_sparsity = ((B.detach() + Z_safe.detach()) <= 0).float().mean().item()                
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
    
