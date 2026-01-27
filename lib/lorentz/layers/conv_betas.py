import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils.parametrizations import orthogonal

import math
from lib.geoopt import Lorentz
from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzFullyConnected
from lib.lorentz.layers.FF_betas import LFC_hyperweight, LorentzProjection, LorentzMLRFF, LorentzBoostAlternate ,LorentzTransform, LorentzBoost
from lib.geoopt.tensor import ManifoldParameter

from lib.lorentz.layers.Kernels import get_learned_kernels
from lib.lorentz.layers.FF_betas import LorentzBoost


def cayley_map(X: torch.Tensor) -> torch.Tensor:
    n, k = X.size(-2), X.size(-1)
    transposed = n < k
    if transposed:
        X = X.mT
        n, k = k, n
    # Here n > k and X is a tall matrix

    # We just need n x k - k(k-1)/2 parameters
    X = X.tril()
    if n != k:
        # Embed into a square matrix
        X = torch.cat([X, X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)], dim=-1)
    A = X - X.mH

    # Computes the Cayley retraction (I+A/2)(I-A/2)^{-1}
    Id = torch.eye(n, dtype=A.dtype, device=A.device)
    Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
    # Q is now orthogonal (or unitary) of size (..., n, n)
    if n != k:
        Q = Q[..., :k]
    # Q is now the size of the X (albeit perhaps transposed)
    if transposed:
        Q = Q.mT
    return Q  # type: ignore[possibly-undefined]


class HyperboleIt(nn.Module):
    def forward(self, X):
        d_out, d_in, k1, k2 = X.shape

        if d_out < d_in*k1*k2:
            return X

        X = X.permute(2,3,1,0).reshape(-1, d_out)
        try:
            X = cayley_map(X)
        except:
            print("break")
        return (X.permute(1, 0)
             .reshape(d_out, k1, k2, d_in)
             .permute(0, 3, 1, 2))


class shape_to(nn.Module):
    def forward(self, X):
        d_out, d_in, k1, k2 = X.shape
        return X.permute(2, 3, 1, 0).reshape(-1, d_out)


class shape_back(nn.Module):
    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape

    def forward(self, X):
        d_out, d_in, k1, k2 = self.out_shape
        return (X.permute(1, 0)
                .reshape(d_out, k1, k2, d_in)
                .permute(0, 3, 1, 2))


class LorentzConv2d_hyperweight(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv2d_hyperweight, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1

        self.linearized_kernel = LFC_hyperweight(
            manifold,
            lin_features,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )
        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        self.linearized_kernel.w = ManifoldParameter(self.manifold.projx(self.linearized_kernel.w.data.uniform_(-stdv, stdv)))
        #if self.bias:
        #    self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  # batch_size, channels * elements/window, windows
        patches = patches.permute(0, 2, 1)

        # Now we have flattened patches with multiple time elements -> fix the concatenation to perform Lorentz direct concatenation by Qu et al. (2022)
        patches_time = torch.clamp(patches.narrow(-1, 0, self.kernel_len), min=self.manifold.k.sqrt())  # Fix zero (origin) padding
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=-1, keepdim=True) - ((self.kernel_len - 1) * self.manifold.k))

        patches_space = patches.narrow(-1, self.kernel_len, patches.shape[-1] - self.kernel_len)
        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape) # No need, but seems to improve runtime??

        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)

        out = self.linearized_kernel(patches_pre_kernel)
        out = out.view(bsz, h_out, w_out, self.out_channels)

        return out


class LorentzConv2d_kernels_old(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv2d_kernels_old, self).__init__()


        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        # kernels = get_learned_kernels(out_channels, in_channels, 200, self.manifold)
        kernels = self.manifold.projx(torch.rand((out_channels, in_channels)))
        self.kernels = ManifoldParameter(kernels, self.manifold, requires_grad=True)

        # lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1

        self.translate = LorentzFullyConnected(
            manifold,
            in_channels,
            self.out_channels * in_channels,
            bias=bias,
            normalize=LFC_normalize
        )

        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)


    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        dists = self.manifold.sqdist(x.unsqueeze(-2), self.kernels)

        x = torch.cat([x, dists], dim=-1)
        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  # batch_size, in_channels * elements/window, windows
        patches = patches.permute(0, 2, 1)
        patches = patches.reshape(bsz, -1, self.in_channels + self.out_channels, self.kernel_len).permute(0,1,3,2)
        # batch_size, windows, kernel_size, in_channels

        distances = patches[..., self.in_channels:]
        patches = patches[..., :self.in_channels]
        # batch_size, windows, kernel_size, out_channels

        # out = self.translate(patches)
        out = self.manifold.centroid(patches, distances.permute(0, 1, 3, 2))
        out = self.manifold.projx(out[..., 1:].mean(-1))

        out = out.view(bsz, h_out, w_out, self.out_channels)

        return out


class LorentzConv2d_kernels(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv2d_kernels, self).__init__()


        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        kernels = get_learned_kernels(out_channels, in_channels, 200, self.manifold)
        self.kernels = ManifoldParameter(kernels, self.manifold, requires_grad=False)

        # lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1

        self.translate = LorentzFullyConnected(
            manifold,
            in_channels,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )

        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)


    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        # dists = self.manifold.sqdist(x.unsqueeze(-2), self.kernels)

        # x = torch.cat([x, dists], dim = -1)
        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  # batch_size, in_channels * elements/window, windows
        patches = patches.permute(0, 2, 1)
        patches = patches.reshape(bsz, -1, self.in_channels, self.kernel_len).permute(0,1,3,2)
        # batch_size, windows, kernel_size, in_channels

        # distances = patches[..., self.in_channels:]
        # patches = patches[..., :self.in_channels]
        # batch_size, windows, kernel_size, out_channels

        # out = self.translate(patches)
        # out = self.manifold.centroid(out, distances.mean(dim=-1).unsqueeze(-2))
        patches = self.manifold.centroid(patches)
        patches = self.manifold.logmap0(patches)
        patches = self.manifold.transp0(self.kernels, patches.unsqueeze(-2))
        out = self.manifold.expmap(self.kernels, patches)
        out = self.manifold.centroid(out)
        out = self.translate(out)
        out = out.view(bsz, h_out, w_out, self.out_channels)

        return out


class LorentzConv2d_attention_kernels(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv2d_attention_kernels, self).__init__()


        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        kernels = get_learned_kernels(out_channels, in_channels, 200, self.manifold)
        self.kernels = ManifoldParameter(kernels, self.manifold, requires_grad=False)

        # lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1

        self.translate = LorentzFullyConnected(
            manifold,
            in_channels,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )

        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)
        self.Q = LorentzFullyConnected(manifold, in_channels, self.out_channels)
        self.K = LorentzFullyConnected(manifold, in_channels, self.out_channels)
        self.V = LorentzFullyConnected(manifold, in_channels, self.out_channels)


    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        # dists = self.manifold.sqdist(x.unsqueeze(-2), self.kernels)

        # x = torch.cat([x, dists], dim = -1)
        x = x.permute(0, 3, 1, 2)

        x_K = self.K(x)
        kernels = self.Q(self.kernels)

        dists = self.manifold.distances(x_K, kernels)
        score = self.softmax(dists)
        attn = self.manifold.centroid(x, w=score).permute(0, 2, 1, 3)

        attn_space = attn.narrow(-1, 1, attn.shape[-1] - 1)
        attn_time = attn.narrow(-1, 0, 1)
        attn_time_rescaled = torch.sqrt(
            torch.sum(attn_time ** 2, dim=-1, keepdim=True) - ((self.heads - 1) * self.manifold.k))
        out = torch.concat((attn_time_rescaled, attn_space), dim=-1)


        return out


class LorentzConv2d_kernels_modded(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv2d_kernels_modded, self).__init__()


        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        kernels = get_learned_kernels(out_channels, in_channels, 200, self.manifold)
        self.kernels = ManifoldParameter(kernels, self.manifold, requires_grad=False)

        # lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1

        self.translate = LorentzFullyConnected(
            manifold,
            in_channels,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )

        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)


    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        # dists = self.manifold.sqdist(x.unsqueeze(-2), self.kernels)

        # x = torch.cat([x, dists], dim = -1)
        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  # batch_size, in_channels * elements/window, windows
        patches = patches.permute(0, 2, 1)
        patches = patches.reshape(bsz, -1, self.in_channels, self.kernel_len).permute(0,1,3,2)
        # batch_size, windows, kernel_size, in_channels

        # distances = patches[..., self.in_channels:]
        # patches = patches[..., :self.in_channels]
        # batch_size, windows, kernel_size, out_channels

        # out = self.translate(patches)
        # out = self.manifold.centroid(out, distances.mean(dim=-1).unsqueeze(-2))
        patches = self.manifold.centroid(patches)
        patches = self.manifold.logmap0(patches)
        patches = self.manifold.transp0(self.kernels, patches.unsqueeze(-2))
        out = self.manifold.expmap(self.kernels, patches)
        out = self.manifold.centroid(out)
        out = self.translate(out)
        out = out.view(bsz, h_out, w_out, self.out_channels)

        return out


class LorentzConv2d_transform_new(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv2d_transform_new, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1

        self.linearized_kernel = LorentzMLRFF(
            manifold,
            lin_features,
            self.out_channels,
        )
        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        # self.linearized_kernel.w = ManifoldParameter(self.manifold.projx(self.linearized_kernel.w.data.uniform_(-stdv, stdv)))
        #if self.bias:
        #    self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  # batch_size, channels * elements/window, windows
        patches = patches.permute(0, 2, 1)

        # Now we have flattened patches with multiple time elements -> fix the concatenation to perform Lorentz direct concatenation by Qu et al. (2022)
        patches_time = torch.clamp(patches.narrow(-1, 0, self.kernel_len), min=self.manifold.k.sqrt())  # Fix zero (origin) padding
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=-1, keepdim=True) - ((self.kernel_len - 1) * self.manifold.k))

        patches_space = patches.narrow(-1, self.kernel_len, patches.shape[-1] - self.kernel_len)
        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape) # No need, but seems to improve runtime??

        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)

        # patches_pre_kernel = self.manifold.regularize(patches_pre_kernel)

        out = self.linearized_kernel(patches_pre_kernel)
        out = out.view(bsz, h_out, w_out, self.out_channels)

        return out


class kernel_multiplier(nn.Module):
    def __init__(self, kernel_size):
        super(kernel_multiplier, self).__init__()

        kernels = torch.rand(kernel_size)
        self.weight = nn.Parameter(kernels)
    def forward(self, x):
        x_0 = x.narrow(-1, 0, 1)
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)

        x_ = torch.einsum('abik, ikc -> abic', x_narrow, self.weight)
        x = torch.cat([x_0, x_], dim=-1)

        return x


class LorentzPureConv(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzPureConv, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        #self.linearized_kernel = LorentzProjection(
        #    manifold,
        #    self.in_channels,
        #    self.out_channels,
        #)

        self.linearized_kernel = kernel_multiplier((self.kernel_len, self.in_channels-1, self.out_channels-1))
        self.LFC_normalize = torch.nn.utils.parametrizations.orthogonal(self.linearized_kernel, name="weight")

        self.boost = LorentzBoost(self.manifold)

        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]),
                                      dilation=dilation,
                                      padding=padding,
                                      stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        return
        # self.linearized_kernel.w = ManifoldParameter(self.manifold.projx(self.linearized_kernel.w.data.uniform_(-stdv, stdv)))
        #if self.bias:
        #    self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    # @torch.compile
    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  # batch_size, channels * elements/window, windows
        patches = patches.permute(0, 2, 1)

        patches_pre_kernel = patches.reshape(bsz, -1, self.in_channels, self.kernel_len).permute(0, 1, 3, 2)
        patches_pre_kernel[:, :, :, 0].clamp_(min=1)

        # patches_pre_kernel = self.manifold.regularize(patches_pre_kernel)
        patches_pre_kernel = self.manifold.logmap0(patches_pre_kernel)
        patches_pre_kernel = self.manifold.rescale_to_max(patches_pre_kernel)
        patches_pre_kernel = self.manifold.expmap0(patches_pre_kernel)

        out = self.linearized_kernel(patches_pre_kernel).squeeze()
        # out = self.manifold.projx(out)
        #out = self.manifold.expmap0(torch.sum(self.manifold.logmap0(out), dim=-2).squeeze())
        out = self.manifold.centroid(out)
        out = self.boost(out)
        # out = self.manifold.centroid(out)
        # out = self.manifold.expmap0(self.manifold.logmap0(out)*self.kernel_len)
        out = out.view(bsz, h_out, w_out, self.out_channels)

        return out


class LorentzPureConv_transform(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzPureConv_transform, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1

        self.linearized_kernel = LorentzProjection(
            manifold,
            lin_features,
            self.out_channels,
        )

        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]),
                                      dilation=dilation,
                                      padding=padding,
                                      stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        return
        # self.linearized_kernel.w = ManifoldParameter(self.manifold.projx(self.linearized_kernel.w.data.uniform_(-stdv, stdv)))
        #if self.bias:
        #    self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    # @torch.compile
    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  # batch_size, channels * elements/window, windows
        patches = patches.permute(0, 2, 1)

        #  Now we have flattened patches with multiple time elements -> fix the concatenation to perform Lorentz direct concatenation by Qu et al. (2022)
        patches_time = torch.clamp(patches.narrow(-1, 0, self.kernel_len),
                                   min=self.manifold.k.sqrt())  # Fix zero (origin) padding
        patches_time_rescaled = torch.sqrt(
            torch.sum(patches_time ** 2, dim=-1, keepdim=True) - ((self.kernel_len - 1) * self.manifold.k))

        patches_space = patches.narrow(-1, self.kernel_len, patches.shape[-1] - self.kernel_len)
        patches_space = (patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1)
                         .transpose(-1, -2).reshape(patches_space.shape))  # No need, but seems to improve runtime??

        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)
        # patches_pre_kernel = self.manifold.logmap0(patches_pre_kernel)
        # patches_pre_kernel = self.manifold.rescale_to_max(patches_pre_kernel)
        # patches_pre_kernel = self.manifold.expmap0(patches_pre_kernel)
        #return patches_pre_kernel
        out = self.linearized_kernel(patches_pre_kernel)

        out = self.manifold.logmap0(out)
        out = self.manifold.rescale_to_max(out)
        out = self.manifold.expmap0(out)

        out = out.view(bsz, h_out, w_out, self.out_channels)

        return out


class ConvSpace(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(ConvSpace, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        self.conv = nn.Conv2d(in_channels-1,
                              out_channels-1,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

        #self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        # self.linearized_kernel.w = ManifoldParameter(self.manifold.projx(self.linearized_kernel.w.data.uniform_(-stdv, stdv)))
        #if self.bias:
        #    self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """

        out = self.conv(x[...,1:].permute(0,3,1,2))
        out = out.permute(0, 2, 3, 1)
        out = self.manifold.add_time(out)

        return out


class PureHyperbolicEfficientConv_original(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            LFC_normalize=False
    ):
        super(PureHyperbolicEfficientConv_original, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        self.rotate = nn.Conv2d(in_channels-1,
                                out_channels-1,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                #dilation=dilation,
                                bias=bias)

        self.boost = LorentzBoost(manifold)

        parametrize.register_parametrization(self.rotate, "weight", HyperboleIt())
        # old_shape = self.rotate.weight.shape
        # parametrize.register_parametrization(self.rotate, "weight", shape_to(), unsafe=True)
        # orthogonal(self.rotate, "weight", orthogonal_map="cayley")
        # parametrize.register_parametrization(self.rotate, "weight", shape_back(old_shape), unsafe=True)


        #self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        # self.linearized_kernel.w = ManifoldParameter(self.manifold.projx(self.linearized_kernel.w.data.uniform_(-stdv, stdv)))
        #if self.bias:
        #    self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        out = self.rotate(x[..., 1:].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = self.manifold.add_time(out)

        out = self.boost(out)

        #out = self.manifold.rescale_to_max(out)
        out = self.manifold.logmap0(out)
        out = self.manifold.rescale_to_max(out)
        out = self.manifold.expmap0(out)

        return out


class PureHyperbolicEfficientConvNorm(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            LFC_normalize=False
    ):
        super(PureHyperbolicEfficientConvNorm, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        self.rotate = nn.Conv2d(in_channels-1,
                                out_channels-1,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias)

        self.average_norm = nn.Conv2d(1,
                                      1,
                                      kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      bias=False)
        torch.nn.init.ones_(self.average_norm.weight)
        self.average_norm.weight.requires_grad = False

        self.boost = LorentzBoost(manifold)

        parametrize.register_parametrization(self.rotate, "weight", HyperboleIt())
        # old_shape = self.rotate.weight.shape
        # parametrize.register_parametrization(self.rotate, "weight", shape_to(), unsafe=True)
        # orthogonal(self.rotate, "weight", orthogonal_map="cayley")
        # parametrize.register_parametrization(self.rotate, "weight", shape_back(old_shape), unsafe=True)


        #self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        # self.linearized_kernel.w = ManifoldParameter(self.manifold.projx(self.linearized_kernel.w.data.uniform_(-stdv, stdv)))
        #if self.bias:
        #    self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        old_norm = self.average_norm(torch.norm(x[..., 1:], dim=-1, keepdim=True).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = self.rotate(x[..., 1:].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        out = out*(old_norm/torch.norm(out, dim=-1, keepdim=True))

        out = self.manifold.add_time(out)

        out = self.boost(out)

        out = self.manifold.logmap0(out)
        out = self.manifold.rescale_to_max(out)
        out = self.manifold.expmap0(out)

        return out


class LorentzKernelTransform(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzKernelTransform, self).__init__()


        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        # here we make kernels with inverse dimensions as well
        # i.e. the number of kernels = kernel size and dim = channels
        kernels = get_learned_kernels(self.kernel_len, out_channels, 200, self.manifold)

        # after initialization, we set them as parameters that we can choose to learn or not
        # self.kernels = ManifoldParameter(self.manifold.projx(torch.rand(self.kernel_len, out_channels)), self.manifold, requires_grad=True)
        self.kernels = ManifoldParameter(kernels, self.manifold, requires_grad=True)

        # lorentz transforms can't change dim, so we have to do it through the Lorentz Linear
        self.translate = LorentzFullyConnected(
            manifold,
            in_channels,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )

        self.distance_interpreter = nn.Linear(self.kernel_len, self.kernel_len, bias=bias)
        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / self.kernel_len)
        self.distance_interpreter.weight.data.uniform_(-stdv, stdv)

    # @torch.compile
    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  # batch_size, channels * elements/window, windows
        patches = patches.permute(0, 2, 1)

        patches_time = patches.narrow(-1, 0, self.kernel_len).unsqueeze(-1).clamp(min=self.manifold.k)
        patches_space = patches.narrow(-1, self.kernel_len, patches.shape[-1] - self.kernel_len).reshape(bsz, -1, self.in_channels-1, self.kernel_len).permute(0,1,3,2)
        patches = torch.concat((patches_time, patches_space), dim=-1)

        # patches = self.manifold.centroid(patches)
        patches_pre_kernel = self.translate(patches)


        translated = self.manifold.logmap0(patches_pre_kernel)
        translated = self.manifold.translated(self.kernels, translated)








        # weights = -self.manifold.sqdist(patches_pre_kernel.unsqueeze(-2), self.kernels)
        # weights = (weights-torch.min(weights, dim=-1)[0].unsqueeze(-1))/(torch.max(weights, dim=-1)[0].unsqueeze(-1) - torch.min(weights, dim=-1)[0].unsqueeze(-1))
        # weights = torch.nn.functional.softmax(weights, dim=-1)

        # out = self.manifold.centroid(self.kernels, w=weights)

        # distance based patch centroid
        # kernels = self.manifold.regularize(self.kernels)
        # out = self.translate(patches)

        # distances = self.manifold.sqdist(patches, kernels)
        # weighted_distances = F.softmax(distances, dim=-1)

        # pure kernel/patch centroid
        # patches_pre_kernel = torch.concat([patches.unsqueeze(-2), self.kernels.expand_as(patches).unsqueeze(-2)], dim=-2)
        # out = self.manifold.centroid(self.manifold.centroid(patches_pre_kernel))

        # parallel transport to kernel + centroid

        # patches = self.manifold.regularize(patches)
        #
        # patches = self.manifold.logmap0(patches)
        # patches = self.manifold.transp0(self.kernels, patches)
        # patches = self.manifold.expmap(self.kernels, patches)
        #
        # out = self.manifold.centroid(patches)
        # replace centroid with addition
        # out = patches.sum(dim=-2)
        # out = self.manifold.add_time(out[..., 1:])

        #out = self.translate(out)
        #out = out.view(bsz, h_out, w_out, self.out_channels)

        #return out


class LorentzKernelAttentionTransform(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzKernelAttentionTransform, self).__init__()


        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        # here we make kernels with inverse dimensions as well
        # i.e. the number of kernels = kernel size and dim = channels
        kernels = get_learned_kernels(self.kernel_len, out_channels, 200, self.manifold)

        # after initialization, we set them as parameters that we can choose to learn or not
        self.kernels = ManifoldParameter(kernels, self.manifold, requires_grad=True)

        # lorentz transforms can't change dim, so we have to do it through the Lorentz Linear
        self.K = LorentzFullyConnected(
            manifold,
            in_channels,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )
        self.Q = LorentzFullyConnected(
            manifold,
            in_channels,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )
        self.V = LorentzFullyConnected(
            manifold,
            in_channels,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )

        # self.distance_interpreter = nn.Linear(self.kernel_len, self.kernel_len, bias=bias)

        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)

    #@torch.compile
    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)
        patches = patches.permute(0, 2, 1)

        patches_time = patches.narrow(-1, 0, self.kernel_len).unsqueeze(-1)
        patches_space = patches.narrow(-1, self.kernel_len, patches.shape[-1] - self.kernel_len).reshape(bsz, -1, self.in_channels-1, self.kernel_len).permute(0,1,3,2)

        patches = torch.concat((patches_time, patches_space), dim=-1)

        distances = self.manifold.sqdist(patches, self.kernels)

        weighted_distances = F.softmax(distances, dim=-1)

        out = torch.einsum("bhij, bhjf->bihf", weighted_distances.unsqueeze(-2), self.V(patches))

        out = self.Q(out)

        return out.view(bsz, h_out, w_out, self.out_channels)


if __name__ == '__main__':

    man = CustomLorentz(1)
    test_operation = LorentzKernelTransform(man, in_channels=4, out_channels=6, kernel_size=3).to("cuda:0")

    x = man.projx(torch.rand((2, 16, 16, 4), device="cuda:0")*10)

    test_operation(x)

