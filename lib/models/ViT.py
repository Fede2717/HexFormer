import torch
import torch.nn as nn

from lib.Euclidean.blocks.transformer_blocks import TransformerEncoder, Embedding

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzMLR, LorentzLayerNorm, LorentzFullyConnected
from lib.lorentz.blocks.transformer_blocks import LorentzTransformerEncoder, LorentzEmbedding


import math

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (LorentzFullyConnected)):
        # gain = nn.init.calculate_gain('leaky_relu', param=0.25)
        #nn.init.orthogonal_(m.weight.weight)
        # nn.init.kaiming_normal_(m.weight.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.xavier_normal_(m.weight.weight) # Tanh 5/3 #LR math.sqrt((2)/(1+(0.2**2)))=1.386 # math.sqrt(2) GELU
        # He initialization for LorentzFullyConnected layers
        # nn.init.kaiming_uniform_(m.weight.weight, a=math.sqrt(5))
        if m.weight.bias is not None:
            nn.init.constant_(m.weight.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class ViT(nn.Module):
    def __init__(
        self,
        manifold=None,
        num_layers=9,
        img_dim=[3,32,32],
        num_classes=100,
        patch_size=4,
        heads=12,
        hidden_dim=192,
        mlp_dim=384,
        dropout=0.0,
        remove_linear=False
    ):
        super(ViT, self).__init__()
        self.manifold = manifold

        c,h,w = img_dim

        self.patch_size = patch_size
        self.num_patches = (h//patch_size) * (w//patch_size)

        self.patch_dim = self.patch_size**2 * c
        self.hidden_dim = hidden_dim

        self.num_tokens = self.num_patches+1 # +1 because of CLS token

        self.embed = self._get_embedding()

        enc_list = [self._get_transformerEncoder(hidden_dim, mlp_dim, self.num_patches, heads, dropout) for _ in range(num_layers)]
        self.encoder = nn.Sequential(*enc_list)

        self.final_ln = self._get_layerNorm(hidden_dim)

        if remove_linear:
            self.predictor = None
        else:
            self.predictor = self._get_predictor(hidden_dim, num_classes)

        self.apply(init_weights)

    def forward(self, x):
        x = self.patchify(x)
        x = self.embed(x)
        x = self.encoder(x)[:, 0]
        x = self.final_ln(x)

        if self.predictor is not None:
            x = self.predictor(x)

        return x

    def patchify(self, x):
        """ Divide image into patches: (bs x C x H x W) -> (bs x N x L) """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.num_patches, -1)

        return out
    
    def _get_embedding(self):
        if self.manifold is None:
            return Embedding(self.hidden_dim, self.patch_dim, self.num_tokens)

        elif type(self.manifold) is CustomLorentz:
            return LorentzEmbedding(self.manifold, self.hidden_dim+1, self.patch_dim+1, self.num_tokens)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ViT.")
    
    def _get_transformerEncoder(self, hidden_dim, mlp_dim, num_patches, heads, dropout):
        if self.manifold is None:
            return TransformerEncoder(hidden_dim, mlp_dim, num_patches, heads, dropout)

        elif type(self.manifold) is CustomLorentz:
            return LorentzTransformerEncoder(self.manifold, hidden_dim+1, mlp_dim+1, num_patches, heads, dropout)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ViT.")
    
    def _get_layerNorm(self, num_features):
        if self.manifold is None:
            return nn.LayerNorm(num_features)

        elif type(self.manifold) is CustomLorentz:
            return LorentzLayerNorm(self.manifold, num_features+1)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ViT.")

    def _get_predictor(self, in_features, num_classes):
        if self.manifold is None:
            return nn.Linear(in_features, num_classes)

        elif type(self.manifold) is CustomLorentz:
            return LorentzMLR(self.manifold, in_features+1, num_classes)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ViT.")
    

def vit(**kwargs):
    " Constructs a Vision Transformer (ViT) "
    model = ViT(**kwargs)
    return model

def lorentz_vit(k=1, learn_k=False, manifold=None, **kwargs):
    " Constructs fully hyperbolic Vision Transformer (ViT) "
    if manifold is None:
        manifold = CustomLorentz(k=k, learnable=learn_k)
    model = ViT(manifold, **kwargs)
    return model