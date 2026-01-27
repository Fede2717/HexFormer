import math

import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz
from lib.geoopt.tensor import ManifoldParameter


class LorentzFullyConnected(nn.Module):
    """
        Modified Lorentz fully connected layer of Chen et al. (2022).

        Code modified from https://github.com/Graph-and-Geometric-Learning/LResNet

        args:
            manifold: Instance of Lorentz manifold
            in_features, out_features, bias: Same as nn.Linear
            init_scale: Scale parameter for internal normalization
            learn_scale: If scale parameter should be learnable
            normalize: If internal normalization should be applied
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_features,
            out_features,
            bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False,
            activation=None,
            dropout=0.0,
            nheads=1
        ):
        super(LorentzFullyConnected, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.nheads = nheads
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.init_std = 0.02
        self.reset_parameters()

        # Scale for internal normalization or Lorentz scaling
        self.scale = nn.Parameter(
            torch.ones(()) * (init_scale if init_scale is not None else 2.3), 
            requires_grad=learn_scale
        )

    def forward(self, x):
        # Apply linear transformation
        x = self.weight(x)
        
        # Separate space-like and time-like components
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)

        # Apply activation function
        if self.activation is not None:
            x_space = self.activation(x_space)
        
        x_space = self.dropout(x_space)

        # Multi-head support
        if self.nheads > 1:
            # Lorentz direct split
            x_space = x_space.view(
                x_space.size(0), x_space.size(1), self.nheads, self.out_features // self.nheads
            ).transpose(1, 2)

        # Normalization or Lorentz scaling
        if self.normalize:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            unit_length = x_space / torch.sqrt(square_norm)
            x_space = scale * unit_length
            x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(square_norm <= 1e-10, self.manifold.k.sqrt())
        else:
            # Compute time component and ensure broadcast compatibility
            time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + (self.manifold.k.sqrt().reciprocal() + 0.5)
            
            # Handle multi-head processing
            if self.nheads > 1:
                batch_size, nheads, seq_len, head_dim = x_space.shape
                x_space_norm = (x_space * x_space).sum(dim=-1, keepdim=True).clamp_min(1e-8)
                time = time.unsqueeze(1).expand(batch_size, nheads, seq_len, 1)  # Expand to match multi-head shape
                scale = (time * time - self.manifold.k.reciprocal()) / x_space_norm
                x_space = x_space * scale.clamp_min(1e-8).sqrt()
                x_time = time
            else:
                x_space_norm = (x_space * x_space).sum(dim=-1, keepdim=True).clamp_min(1e-8)
                scale = (time * time - self.manifold.k.reciprocal()) / x_space_norm
                x_space = x_space * scale.clamp_min(1e-8).sqrt()
                x_time = time


        # Combine time and space components
        x = torch.cat([x_time, x_space], dim=-1)
        return x

    def reset_parameters(self):
        # new
        # stdv = 1. / math.sqrt(self.out_features)
        # step = self.in_features
        # nn.init.uniform_(self.weight.weight, -stdv, stdv)
        # with torch.no_grad():
        #     for idx in range(0, self.in_features, step):
        #         self.weight.weight[:, idx] = 0
        # if self.bias:
        #     nn.init.constant_(self.weight.bias, 0)
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


# default
# class LorentzFullyConnected(nn.Module):
#     """
#         Modified Lorentz fully connected layer of Chen et al. (2022).

#         Code modified from https://github.com/chenweize1998/fully-hyperbolic-nn

#         args:
#             manifold: Instance of Lorentz manifold
#             in_features, out_features, bias: Same as nn.Linear
#             init_scale: Scale parameter for internal normalization
#             learn_scale: If scale parameter should be learnable
#             normalize: If internal normalization should be applied
#     """

#     def __init__(
#             self,
#             manifold: CustomLorentz,
#             in_features,
#             out_features,
#             bias=False,
#             init_scale=None,
#             learn_scale=False,
#             normalize=False,
#             activation=None,
#             dropout=0.0,
#             nheads=1
#         ):
#         super(LorentzFullyConnected, self).__init__()
#         self.manifold = manifold
#         self.in_features = in_features
#         self.out_features = out_features
#         self.bias = bias
#         self.normalize = normalize

#         self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)
#         self.nheads = nheads
#         self.dropout = nn.Dropout(dropout)
#         self.activation = activation

#         self.init_std = 0.02
#         self.reset_parameters()

#         # Scale for internal normalization
#         if init_scale is not None:
#             self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
#         else:
#             self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

#     def forward(self, x):

#         x = self.weight(x)

#         # x_space = x[..., 1:]
#         x_space = x.narrow(-1, 1, x.shape[-1] - 1)

#         # x_norm = torch.norm(x_space)
#         # x_space = x_space/x_norm * max(10, x_norm.item())

#         if self.activation is not None:
#             x_space = self.activation(x_space)
#         x_space = self.dropout(x_space)


#         if self.nheads>1:
#             # Lorentz direct split
#             x_space = x_space.view(x_space.size(0), x_space.size(1), self.nheads, self.out_features//self.nheads).transpose(1,2)

#         if self.normalize:
#             scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
#             square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)

#             mask = square_norm <= 1e-10

#             square_norm[mask] = 1
#             # square_norm = square_norm + mask * 1e-8  # Avoid division by zero
#             unit_length = x_space/torch.sqrt(square_norm)
#             x_space = scale*unit_length

#             x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
#             x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())

#             mask = mask==False
#             x_space = x_space * mask

#             x = torch.cat([x_time, x_space], dim=-1)
#         else:
#             x = self.manifold.add_time(x_space)

#         return x

#     def reset_parameters(self):
#         nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

#         if self.bias:
#             nn.init.constant_(self.weight.bias, 0)

