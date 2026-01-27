import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

import math

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzFullyConnected
from lib.lorentz.layers.FF_betas import LorentzProjection, LorentzFullyConnected_transform, LorentzBoost, LorentzPureBoost
from lib.lorentz.layers.utils import HyperboleIt, ReshapeHyperbolicWeight, CustomWeightConv2d
from lib.geoopt.tensor import ManifoldParameter
from lib.geoopt.manifolds import Stiefel

from lib.lorentz.layers.Kernels import get_learned_kernels


class LorentzConv1d(nn.Module):
    """ Implements a fully hyperbolic 1D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, bias: Same as nn.Conv1d
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
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv1d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        lin_features = (self.in_channels - 1) * self.kernel_size + 1

        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x len x C """
        bsz = x.shape[0]

        # origin padding
        x = F.pad(x, (0, 0, self.padding, self.padding))
        x[..., 0].clamp_(min=self.manifold.k.sqrt())

        patches = x.unfold(1, self.kernel_size, self.stride)
        # Lorentz direct concatenation of features within patches
        patches_time = patches.narrow(2, 0, 1)
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=(-2,-1), keepdim=True) - ((self.kernel_size - 1) * self.manifold.k))
        patches_time_rescaled = patches_time_rescaled.view(bsz, patches.shape[1], -1)

        patches_space = patches.narrow(2, 1, patches.shape[2]-1).reshape(bsz, patches.shape[1], -1)
        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)

        out = self.linearized_kernel(patches_pre_kernel)

        return out


def unfold1d(input, kernel_size: int, stride: int, padding:int):
    *shape, length = input.shape
    n_frames = (max(length, kernel_size) - kernel_size) // stride + 1
    tgt_length = (n_frames - 1) * stride + kernel_size
    input = input[..., :tgt_length].contiguous()
    strides = list(input.stride())
    strides = strides[:-1] + [stride, 1]
    out = input.as_strided(shape + [n_frames, kernel_size], strides)
    return out.transpose(-1, -2)


class LorentzConv2d(nn.Module):
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
        super(LorentzConv2d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
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

        #self.hyperbolic_linear = LorentzBoost(manifold, init_weight=0.5)

        self.scale = torch.nn.Parameter(torch.tensor([0.1]))

        self.linearized_kernel = LorentzFullyConnected(
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
        self.linearized_kernel.weight.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

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

        patches = self.unfold(x)  # batch_size, channels * elements/window, windows
        patches = patches.permute(0, 2, 1)

        # Now we have flattened patches with multiple time elements -> fix the concatenation to perform Lorentz direct concatenation by Qu et al. (2022)
        patches_time = torch.clamp(patches.narrow(-1, 0, self.kernel_len), min=self.manifold.k.sqrt())  # Fix zero (origin) padding
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=-1, keepdim=True) - ((self.kernel_len - 1) * self.manifold.k))

        patches_space = patches.narrow(-1, self.kernel_len, patches.shape[-1] - self.kernel_len)
        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape) # No need, but seems to improve runtime??

        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)

        patches_pre_kernel = self.manifold.logmap0(patches_pre_kernel)
        patches_pre_kernel = self.manifold.rescale_to_max(patches_pre_kernel)
        patches_pre_kernel = self.manifold.expmap0(patches_pre_kernel)
        # # # patches_pre_kernel = self.manifold.boost_scale_origin(patches_pre_kernel, s=self.scale.clamp(max=2, min=1e-3))
        # patches_pre_kernel = self.hyperbolic_linear(patches_pre_kernel)
        out = self.linearized_kernel(patches_pre_kernel)
        out = out.view(bsz, h_out, w_out, self.out_channels)

        return out


class LorentzConvTranspose2d(nn.Module):
    """ Implements a fully hyperbolic 2D transposed convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, output_padding, bias: Same as nn.ConvTranspose2d
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
            output_padding=0,
            bias=True,
            LFC_normalize=False
        ):
        super(LorentzConvTranspose2d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels

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

        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding)
        else:
            self.output_padding = output_padding

        padding_implicit = [0,0]
        padding_implicit[0] = kernel_size - self.padding[0] - 1 # Ensure padding > kernel_size
        padding_implicit[1] = kernel_size - self.padding[1] - 1 # Ensure padding > kernel_size

        self.pad_weight = nn.Parameter(F.pad(torch.ones((self.in_channels,1,1,1)),(1,1,1,1)), requires_grad=False)

        self.conv = PureHyperbolicEfficientConv(
            manifold=manifold,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding_implicit,
            bias=bias,
            LFC_normalize=LFC_normalize
        )

    #@torch.compile
    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        if self.stride[0] > 1 or self.stride[1] > 1:
            # Insert hyperbolic origin vectors between features
            x = x.permute(0,3,1,2)
            # -> Insert zero vectors
            x = F.conv_transpose2d(x, self.pad_weight,stride=self.stride,padding=1, groups=self.in_channels)
            x = x.permute(0,2,3,1)
            x[..., 0].clamp_(min=self.manifold.k.sqrt())

        x = self.conv(x)

        if self.output_padding[0] > 0 or self.output_padding[1] > 0:
            x = F.pad(x, pad=(0, self.output_padding[1], 0, self.output_padding[0])) # Pad one side of each dimension (bottom+right) (see PyTorch documentation)
            x[..., 0].clamp_(min=self.manifold.k.sqrt()) # Fix origin padding

        return x


class PureHyperbolicEfficientConv(nn.Module):
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
        super(PureHyperbolicEfficientConv, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        self.rotation_manifold = Stiefel()

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]

        self.rotate = nn.Conv2d(in_channels-1,
                                out_channels-1,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                #dilation=dilation,
                                bias=False)
        d_out, d_in, k1, k2 = self.rotate.weight.shape

        self.debug_test = False

        if d_out > d_in*k1*k2:

            self.rotate = CustomWeightConv2d(self.rotation_manifold,
                                             (d_out, d_in, k1, k2),
                                             in_channels - 1,
                                                out_channels - 1,
                                                kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                # dilation=dilation,
                                                bias=False)

            parametrize.register_parametrization(self.rotate, "weight", HyperboleIt())
            self.debug_test = True


        self.boost = LorentzBoost(manifold)
        #self.boost = LorentzPureBoost(manifold, dim=out_channels)
        #self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        with torch.no_grad():
            self.rotate.weight.copy_(self.rotation_manifold.projx(self.rotate.weight.data.uniform_(-stdv, stdv)))

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        out = self.rotate(x[..., 1:].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        out = self.manifold.add_time(out)

        out = self.boost(out)
        if torch.isnan(out).sum()>0:
            print("rotate")
        out = self.manifold.logmap0(out)
        out = self.manifold.rescale_to_max(out)
        out = self.manifold.expmap0(out)

        return out


class LorentzConv1By1(nn.Module):
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
        super(LorentzConv1By1, self).__init__()

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

        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        self.linearized_kernel.weight.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    #@torch.compile
    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        patches = x.reshape(bsz, h*w, self.in_channels)

        out = self.linearized_kernel(patches)
        out = out.view(bsz, h, w, self.out_channels)

        return out


if __name__ == '__main__':

    x = torch.rand((128, 32, 32, 4))

    manifold = CustomLorentz()
    x = manifold.projx(x)

    conv = LorentzConv1By1(manifold, in_channels=4, out_channels=32, kernel_size=1)
    output = conv(x)

    print("break")
