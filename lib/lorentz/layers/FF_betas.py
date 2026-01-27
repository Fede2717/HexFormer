import math

import torch
import torch.nn as nn

from torch.nn.utils.parametrizations import orthogonal

from lib.lorentz.manifold import CustomLorentz
from lib.geoopt.tensor import ManifoldParameter
from lib.lorentz.layers.LMLR import LorentzMLR
from lib.geoopt import Stiefel
from torch.distributions import Uniform


class LFC_hyperweight(nn.Module):
    """
        normal linear layer but the wirghts are hyperbolic
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_features,
            out_features,
            bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False
        ):
        super(LFC_hyperweight, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        self.init_std = 0.02
        initial = (-2*self.init_std)* torch.rand((self.out_features, self.in_features)) + self.init_std

        self.w = ManifoldParameter(self.manifold.projx(initial), manifold=self.manifold)
        #self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)

        self.reset_parameters()

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

    def forward(self, x):

        x = torch.nn.functional.linear(x, self.w, bias=None)
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)

        if self.normalize:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)

            mask = square_norm <= 1e-10

            square_norm[mask] = 1
            unit_length = x_space/torch.sqrt(square_norm)
            x_space = scale*unit_length

            x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())

            mask = mask==False
            x_space = x_space * mask

            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.add_time(x_space)

        return x

    def reset_parameters(self):
        return
        # nn.init.uniform_(self.w, -self.init_std, self.init_std)

        #if self.bias:
        #    nn.init.constant_(self.weight.bias, 0)


class LorentzTransform(torch.nn.Module):
    def __init__(self, manifold, dim, mode="boost", regularize=True):
        super(LorentzTransform, self).__init__()

        self.dim = dim

        self.boost = self.rotate = False

        if mode == "both" or mode == "boost":
            self.v = nn.Parameter(torch.rand((dim - 1, 1)))
            self.boost = True

        if mode == "both" or mode == "rotate":
            self.rotation_weight = nn.Parameter(torch.rand((dim - 1, dim - 1)))
            self.rotate = True

        self.eye = nn.Parameter(torch.eye(dim - 1), requires_grad=False)
        self.manifold = manifold
        self.if_regularize = True
        self.reset_parameters()

    def forward(self, x, stabalize=False):

        if self.boost:
            norm = self.v.norm(2, dim=0, keepdim=False)
            # desired = torch.clamp(norm, max=0.99)
            desired = torch.sigmoid(norm/2)
            v = self.v * (desired / norm)

            # get boost
            gamma = 1 / torch.sqrt(1 - torch.norm(v) ** 2).reshape(1, -1)
            el_1 = -gamma * v.T
            el_2 = -gamma * v
            el_3 = self.eye + (gamma - 1) * (v * v.T) / (v.norm(2, dim=0, keepdim=True) ** 2)

            upper = torch.cat([gamma, el_1], dim=1)
            lower = torch.cat([el_2, el_3], dim=1)
            boost = torch.cat([upper, lower], dim=0)

        # get rotation
        if self.rotate:
            rotation = torch.nn.functional.pad(self.rotation_weight, (1, 0, 1, 0))
            rotation[..., 0, 0] = 1


        if self.rotate and self.boost:
            output = torch.matmul(torch.matmul(rotation, boost), x.transpose(-1, -2)).transpose(-1, -2)
        elif self.rotate:
            output = torch.matmul(rotation, x.transpose(-1, -2)).transpose(-1, -2)
        elif self.boost:
            output = torch.matmul(boost, x.transpose(-1, -2)).transpose(-1, -2)

        if stabalize:
            output = self.manifold.logmap0(output)
            norm = output[..., 1:].norm(2, dim=-1, keepdim=True)
            desired = torch.clamp(norm, max=10)

            output = output[..., 1:] * (desired / norm)
            output = self.manifold.add_time(output)

            output = self.manifold.expmap0(output)

        if self.if_regularize is True:
            output = self.manifold.regularize(x)

        return output

    def reset_parameters(self):
        return
        # nn.init.kaiming_normal_(self.v)
        # nn.init.orthogonal_(self.rotation_weight)


class LorentzFullyConnected_transform(nn.Module):
    """
        Modified Lorentz fully connected layer of Chen et al. (2022).

        Code modified from https://github.com/chenweize1998/fully-hyperbolic-nn

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
            normalize=False
    ):
        super(LorentzFullyConnected_transform, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        self.weight = nn.Linear(self.in_features - 1, self.out_features - 1, bias=bias)

        self.init_std = 0.02
        self.reset_parameters()

        self.shape_matrix = nn.Parameter(torch.ones((in_features, out_features)), requires_grad=False)

        self.transform = LorentzTransform(self.manifold, out_features)
        self.transform = orthogonal(self.transform, name="rotation_weight")

    def forward(self, x):

        if self.out_features != self.in_features:
            x_space = self.weight(x[..., 1:])
            x = self.manifold.add_time(x_space)

        return self.transform(x)

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

        if self.bias:
            nn.init.constant_(self.weight.bias, 0)



class LorentzPureBoost(torch.nn.Module):
    def __init__(self, manifold, dim, regularize=True):
        super(LorentzPureBoost, self).__init__()

        self.dim = dim

        self.v = nn.Parameter(torch.rand((dim - 1, 1))*0.05)
        #self.v = nn.Parameter(torch.ones((dim - 1, 1)))

        self.eye = nn.Parameter(torch.eye(dim - 1), requires_grad=False)
        self.manifold = manifold

        self.if_regularize = True
        self.reset_parameters()

    def forward(self, x, stabalize=False):

        norm = self.v.norm(2, dim=0, keepdim=False)
        # desired = torch.clamp(norm, max=0.99)
        desired = torch.sigmoid(norm)
        v = self.v * (desired / norm)

        # get boost
        gamma = 1 / torch.sqrt(1 - torch.norm(v) ** 2).reshape(1, -1)
        el_1 = -gamma * v.T
        el_2 = -gamma * v
        el_3 = self.eye + (gamma - 1) * (v * v.T) / (desired ** 2)

        upper = torch.cat([gamma, el_1], dim=1)
        lower = torch.cat([el_2, el_3], dim=1)
        boost = torch.cat([upper, lower], dim=0)

        output = torch.matmul(boost, x.transpose(-1, -2)).transpose(-1, -2)

        #output = self.manifold.projx(output)

        return output

    def reset_parameters(self):
        return
        # nn.init.kaiming_normal_(self.v)
        # nn.init.orthogonal_(self.rotation_weight)


class LorentzBoost(nn.Module):
    """hyperbolic rotation achieved by times A = [cosh\alpha,...,sinh\alpha]
                                                [sinh\alpha,...,cosh\alpha]
    """
    def __init__(self, manifold, init_weight=1):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Parameter(torch.FloatTensor([init_weight]))

    def forward(self, x):  # x =[x_0,x_1,...,x_n]
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 2) #x_narrow = [x_1,...,x_n-1]
        x_0 = torch.cosh(self.weight) * x.narrow(-1, 0, 1) + torch.sinh(self.weight) * x.narrow(-1, x.shape[-1] - 1, 1)
        x_n = torch.sinh(self.weight) * x.narrow(-1, 0, 1) + torch.cosh(self.weight) * x.narrow(-1, x.shape[-1] - 1, 1)

        # x_0 = torch.sqrt(self.weight**2 + 1.0) * x_narrow.narrow(-1, 0, 1) + self.weight * x_narrow.narrow(-1, x_narrow.shape[-1] - 1, 1)
        # x_n = self.weight * x_narrow.narrow(-1, 0, 1) + torch.sqrt(self.weight**2 + 1.0) * x_narrow.narrow(-1, x_narrow.shape[-1] - 1, 1)
        x = torch.cat([x_0, x_narrow, x_n], dim=-1)

        return x

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))


class LorentzBoostScale(nn.Module):
    """hyperbolic rotation achieved by times A = [cosh\alpha,...,sinh\alpha]
                                                [sinh\alpha,...,cosh\alpha]
    """
    def __init__(self, manifold, init_weight=1):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Parameter(torch.FloatTensor([init_weight]))

    def forward(self, x):  # x =[x_0,x_1,...,x_n]
        return self.manifold.scale_hyperbolic_vector(x, self.weight)


class LorentzBoostScaleAlternate(nn.Module):
    """hyperbolic rotation achieved by times A = [cosh\alpha,...,sinh\alpha]
                                                [sinh\alpha,...,cosh\alpha]
    """
    def __init__(self, manifold, in_features, init_weight=1):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Linear(in_features - 1, 1, bias=False)
        self.reset_parameters()

    def forward(self, x):  # x =[x_0,x_1,...,x_n]

        #weight = nn.functional.sigmoid(self.weight(x[..., 1:]))*2
        weight = self.weight(x[..., 1:]).clamp(min=0.1, max=2)
        return self.manifold.scale_hyperbolic_vector(x, weight)

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, a=0,b=0.1)

class LorentzBoostAlternate(nn.Module):
    """hyperbolic rotation achieved by times A = [cosh\alpha,...,sinh\alpha]
                                                [sinh\alpha,...,cosh\alpha]
    """
    def __init__(self, manifold, in_features, init_weight=1):
        super().__init__()
        self.manifold = manifold
        #self.weight = nn.Parameter(torch.FloatTensor([init_weight]))
        self.weight = nn.Linear(in_features - 1, 1, bias=False)

    def forward(self, x):  # x =[x_0,x_1,...,x_n]

        weight = self.weight(x[..., 1:]).clamp(min=-4.5, max=4.5)
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 2) #x_narrow = [x_1,...,x_n-1]
        x_0 = torch.cosh(weight) * x.narrow(-1, 0, 1) + torch.sinh(weight) * x.narrow(-1, x.shape[-1] - 1, 1)
        x_n = torch.sinh(weight) * x.narrow(-1, 0, 1) + torch.cosh(weight) * x.narrow(-1, x.shape[-1] - 1, 1)

        x = torch.cat([x_0, x_narrow, x_n], dim=-1)

        return x

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight)


class LorentzRotation_Down(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 if_regularize=False,
                 if_projected=False
                 ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features-1, self.out_features-1, bias=False)
        # self.linear = orthogonal(self.linear, "weight", orthogonal_map="cayley")
        self.reset_parameters()
        self.if_regularize = if_regularize
        self.if_projected = if_projected

    def forward(self, x):

        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)

        #old_norm = x_narrow.norm(2, dim=-1, keepdim=False).unsqueeze(-1)

        #old_norm[old_norm == 0] = 1e-5

        x_ = self.linear(x_narrow)
        #x_ = x_ * x_.norm(2, dim=-1, keepdim=False).unsqueeze(-1)/old_norm

        x = self.manifold.add_time(x_)
        if self.if_regularize is True:
            x = self.manifold.rescale_to_max(x)

        if self.if_projected is True:
            x = self.manifold.projx(x)

        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.linear.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.linear.weight[:, idx] = 0


class LorentzRotation_Up(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 if_regularize=False,
                 if_projected=False
                 ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features

        #self.weight_manifold = Stiefel()

        stdv = 1. / math.sqrt(self.out_features)

        weight = torch.rand((self.out_features-1, self.in_features-1)).uniform_(-stdv, stdv)
        #self.weight = ManifoldParameter(self.weight_manifold.projx(weight), manifold=self.weight_manifold)
        self.weight = torch.nn.Parameter(weight)

        #self.reset_parameters()
        self.if_regularize = if_regularize
        self.if_projected = if_projected

    def forward(self, x):

        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)

        x_ = torch.matmul(self.weight, x_narrow.transpose(-1,-2)).transpose(-1,-2)
        x = self.manifold.add_time(x_)

        if self.if_regularize is True:
            x = self.manifold.rescale_to_max(x)

        if self.if_projected is True:
            x = self.manifold.projx(x)

        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight[:, idx] = 0


class LorentzProjection(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, dropout=False):
        super(LorentzProjection, self).__init__()

        self.down = False

        if out_features > in_features:
            self.rotation = LorentzRotation_Up(manifold, in_features, out_features, if_regularize=False, if_projected=True)
            self.rotation = orthogonal(self.rotation, "weight", orthogonal_map="cayley")
        else:
            self.rotation = LorentzRotation_Down(manifold, in_features, out_features, if_regularize=False,
                                               if_projected=True)
            self.down = True

        #self.boost = LorentzBoost(manifold, init_weight=1)
        self.boost = LorentzBoostScale(manifold)
        #self.boost = LorentzPureBoost(manifold, dim=out_features)
        self.manifold = manifold

    def forward(self, input):
        xt = self.rotation(input)
        h = self.boost(xt)
        h = self.manifold.projx(h)
        # h = self.projection(h)

        return h


class LorentzMLRFF(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, dropout=False):
        super(LorentzMLRFF, self).__init__()

        self.manifold = manifold

        self.mlr = LorentzMLR(manifold, in_features, out_features)
        self.projection = nn.Linear(out_features, out_features-1)

    def forward(self, input):
        x = self.mlr(input)
        x = self.projection(x)

        x = self.manifold.add_time(x)

        return x



#old
# class LorentzRotation_Down(nn.Module):
#     def __init__(self,
#                  manifold,
#                  in_features,
#                  out_features,
#                  if_regularize=False,
#                  if_projected=False
#                  ):
#         super().__init__()
#         self.manifold = manifold
#         self.in_features = in_features
#         self.out_features = out_features
#         self.linear = nn.Linear(self.in_features-1, self.out_features-1, bias=False)
#         # self.linear = orthogonal(self.linear, "weight", orthogonal_map="cayley")
#         self.reset_parameters()
#         self.if_regularize = if_regularize
#         self.if_projected = if_projected

#     def forward(self, x):

#         x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)

#         old_norm = x_narrow.norm(2, dim=-1, keepdim=False).unsqueeze(-1)

#         #old_norm[old_norm == 0] = 1e-5

#         x_ = self.linear(x_narrow)
#         x_ = x_ * x_.norm(2, dim=-1, keepdim=False).unsqueeze(-1)/old_norm

#         x = self.manifold.add_time(x_)

#         if self.if_regularize is True:
#             x = self.manifold.rescale_to_max(x[..., 1:])
#             x = self.manifold.add_time(x)

#         if self.if_projected is True:
#             x = self.manifold.projx(x)

#         return x

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.out_features)
#         step = self.in_features
#         nn.init.uniform_(self.linear.weight, -stdv, stdv)
#         with torch.no_grad():
#             for idx in range(0, self.in_features, step):
#                 self.linear.weight[:, idx] = 0


# class LorentzRotation_Up(nn.Module):
#     def __init__(self,
#                  manifold,
#                  in_features,
#                  out_features,
#                  if_regularize=False,
#                  if_projected=False
#                  ):
#         super().__init__()
#         self.manifold = manifold
#         self.in_features = in_features
#         self.out_features = out_features

#         self.weight_manifold = Stiefel()

#         stdv = 1. / math.sqrt(self.out_features)

#         weight = torch.rand((self.out_features-1, self.in_features-1)).uniform_(-stdv, stdv)
#         self.weight = ManifoldParameter(self.weight_manifold.projx(weight), manifold=self.weight_manifold)

#         #self.reset_parameters()
#         self.if_regularize = if_regularize
#         self.if_projected = if_projected

#     def forward(self, x):

#         x_0 = x.narrow(-1, 0, 1)
#         x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
#         x_ = torch.matmul(self.weight, x_narrow.T).T
#         x = torch.cat([x_0, x_], dim=-1)
#         if self.if_regularize is True:
#             x = self.manifold.rescale_to_max(x[...,1:])
#             x = self.manifold.add_time(x)

#         if self.if_projected is True:
#             x = self.manifold.projx(x)

#         return x

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.out_features)
#         step = self.in_features
#         nn.init.uniform_(self.weight, -stdv, stdv)
#         with torch.no_grad():
#             for idx in range(0, self.in_features, step):
#                 self.weight[:, idx] = 0


# class LorentzProjection(nn.Module):
#     """
#     Hyperbolic graph convolution layer.
#     """

#     def __init__(self, manifold, in_features, out_features, dropout=False):
#         super(LorentzProjection, self).__init__()

#         self.down = False

#         if out_features > in_features:
#             self.rotation = LorentzRotation_Up(manifold, in_features, out_features, if_regularize=False, if_projected=True)
#             # self.rotation = orthogonal(self.rotation, "weight", orthogonal_map="cayley")
#         else:
#             self.rotation = LorentzRotation_Down(manifold, in_features, out_features, if_regularize=False,
#                                                if_projected=True)
#             self.down = True
#         self.boost = LorentzPureBoost(manifold, dim=out_features)
#         #self.boost = LorentzBoost(manifold, init_weight=1)
#         self.manifold = manifold

#     def forward(self, input):
#         xt = self.rotation(input)
#         h = self.boost(xt)
#         h = self.manifold.projx(h)
#         # h = self.projection(h)

#         return h

