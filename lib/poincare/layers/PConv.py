import torch
import torch.nn as nn

import math

from scipy.special import beta

from lib.poincare.manifold import CustomPoincare
from lib.poincare.layers import PoincareFullyConnected

class PoincareConv2d(nn.Module):
    """
    
    """
    def __init__(
            self,
            manifold: CustomPoincare,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True
    ):
        super(PoincareConv2d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

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

        lin_features = self.in_channels * self.kernel_size[0] * self.kernel_size[1]

        self.linearized_kernel = PoincareFullyConnected(
            manifold,
            lin_features,
            self.out_channels,
            bias
        )
        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)

        self.beta_ni = beta(self.in_channels / 2, 1 / 2)
        self.beta_n = beta(self.in_channels * self.kernel_len / 2, 1 / 2)

        #self.reset_parameters()

    def reset_parameters(self):
        #stdv = math.sqrt(2.0 / ((self.in_channels) * self.kernel_size[0] * self.kernel_size[1]))
        #self.linearized_kernel.weight.weight.data.uniform_(-stdv, stdv)
        weight = torch.eye(self.in_channels, self.out_channels).reshape(1, self.in_channels, self.out_channels).repeat(self.kernel_len, 1, 1).reshape(-1, self.out_channels)
        self.linearized_kernel.weight_g.data = weight.norm(dim=0)
        self.linearized_kernel.weight_v.data = weight
            
    def forward(self, x):
        """ """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  # batch_size, channels * elements/window, windows
        patches = patches.permute(0, 2, 1)

        # Now we have flattened patches -> fix the concatenation to perform beta concatenation
        patches = patches.reshape(bsz, patches.shape[1], self.in_channels, -1).permute(0,1,3,2)
        patches_pre_kernel = self.manifold.beta_concat(patches, self.beta_ni, self.beta_n)

        out = self.linearized_kernel(patches_pre_kernel)
        out = out.view(bsz, h_out, w_out, self.out_channels)

        return out