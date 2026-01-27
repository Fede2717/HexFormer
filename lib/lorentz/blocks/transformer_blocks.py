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
    def __init__(self, manifold: CustomLorentz, hidden, mlp_hidden, num_patches, heads, dropout, stochastic_depth=0.1):
        super(LorentzTransformerEncoder, self).__init__()

        self.manifold = manifold

        self.hidden = hidden
        self.mlp_hidden = mlp_hidden
        self.num_patches = num_patches
        self.heads = heads
        self.dropout = dropout

        self.ln1 = LorentzLayerNorm(manifold, hidden)
        self.mha = LorentzMultiHeadAttention(manifold, hidden, num_patches, heads, dropout)
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
    def __init__(self, manifold: CustomLorentz, num_features, num_patches, heads, dropout=0.0, learn_scale=False):
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

        # Internal Lorentz direct split (LFC splits into heads internally)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        dists = -self.manifold.csqdist(q, k)*self.scale.expand((b, self.heads, 1, 1))

        score = self.softmax(dists / self.temperature)

        # attn = self.manifold.centroid(v, w=score).permute(0,2,1,3)
        attn = self.lorentz_expmap_aggregation(v, score).permute(0, 2, 1, 3)
        
        # Lorentz direct concatenation of heads
        attn_space = attn.narrow(-1, 1, attn.shape[-1]-1).reshape(b, n, -1)
        attn_time = attn.narrow(-1, 0, 1).reshape(b, n, -1)
        attn_time_rescaled = torch.sqrt(torch.sum(attn_time ** 2, dim=-1, keepdim=True) - ((self.heads - 1) * self.manifold.k))
        attn = torch.concat((attn_time_rescaled, attn_space), dim=-1)

        o = self.o(attn) # internal dropout in LFC
        return o
    
