import torch
import torch.nn as nn

from lib.utils.drop_path import DropPath

class Embedding(nn.Module):
    def __init__(self, hidden_dim, patch_dim, num_tokens):
        super(Embedding, self).__init__()
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) # CLS token with hyperbolic randn?
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))

    def forward(self, x):
        x = self.patch_embed(x)
        
        x = torch.cat([self.cls_token.repeat(x.size(0),1,1), x], dim=1)
        x = x + self.pos_embed

        return x        

class TransformerEncoder(nn.Module):
    def __init__(self, hidden, mlp_hidden, num_patches, heads, dropout, stochastic_depth=0.1):
        super(TransformerEncoder, self).__init__()

        self.hidden = hidden
        self.mlp_hidden = mlp_hidden
        self.num_patches = num_patches
        self.heads = heads
        self.dropout = dropout

        self.ln1 = nn.LayerNorm(hidden)
        self.mha = MultiHeadAttention(hidden, num_patches, heads, dropout)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(), # nn.LeakyReLU(0.2), #nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden),
            nn.Dropout(dropout)
        )

        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.mha(self.ln1(x))) + x
        x = self.drop_path(self.mlp(self.ln2(x))) + x
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_features,  num_patches, heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_features = num_features
        self.num_patches = num_patches
        self.heads = heads
        self.head_dim = num_features//heads
        self.scale = self.head_dim**(-0.5)
        self.temperature = nn.Parameter(torch.ones(1))  # Initialize temperature
        self.softmax = nn.Softmax(dim=-1)

        self.q = nn.Linear(num_features, num_features, bias=False)
        self.k = nn.Linear(num_features, num_features, bias=False)
        self.v = nn.Linear(num_features, num_features, bias=False)

        self.o = nn.Linear(num_features, num_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, l = x.size()
        q = self.q(x).view(b, n, self.heads, self.head_dim).transpose(1,2)
        k = self.k(x).view(b, n, self.heads, self.head_dim).transpose(1,2)
        v = self.v(x).view(b, n, self.heads, self.head_dim).transpose(1,2)
        score = self.softmax((torch.einsum("bhif, bhjf->bhij", q, k)*self.scale)/self.temperature)
        attn = torch.einsum("bhij, bhjf->bihf", score, v)
        o = self.dropout(self.o(attn.flatten(2)))
        return o
