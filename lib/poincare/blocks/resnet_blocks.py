import torch.nn as nn

from lib.poincare.manifold import CustomPoincare
from lib.poincare.layers import (
    PoincareConv2d,
    PoincareBatchNorm2d,
    PoincareTangentReLU
)

def get_Conv2d(manifold, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    return PoincareConv2d(
        manifold=manifold, 
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        bias=bias
    )

def get_BatchNorm2d(manifold, num_channels):
    return PoincareBatchNorm2d(manifold=manifold, num_channels=num_channels)

def get_Activation(manifold):
    return PoincareTangentReLU(manifold)


class PoincareInputBlock(nn.Module):
    """ Input Block of ResNet model """

    def __init__(self, manifold: CustomPoincare, img_dim, in_channels, bias=True):
        super(PoincareInputBlock, self).__init__()

        self.manifold = manifold

        self.conv = nn.Sequential(
            get_Conv2d(
                self.manifold,
                img_dim,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, in_channels),
            get_Activation(self.manifold),
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # Make channel last (bs x H x W x C)
        x = self.manifold.projx(x)
        return self.conv(x)


class PoincareBasicBlock(nn.Module):
    """ Basic Block for Poincare ResNet-10, -18 and -34 """

    expansion = 1

    def __init__(self, manifold: CustomPoincare, in_channels, out_channels, stride=1, bias=True):
        super(PoincareBasicBlock, self).__init__()

        self.manifold = manifold

        self.activation = get_Activation(self.manifold)

        self.conv = nn.Sequential(
            get_Conv2d(
                self.manifold,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, out_channels),
            get_Activation(self.manifold),
            get_Conv2d(
                self.manifold,
                out_channels,
                out_channels * PoincareBasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(self.manifold, out_channels * PoincareBasicBlock.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * PoincareBasicBlock.expansion:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    self.manifold,
                    in_channels,
                    out_channels * PoincareBasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias
                ),
                get_BatchNorm2d(
                    self.manifold, out_channels * PoincareBasicBlock.expansion
                ),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        # Residual = Möbius addition
        out = self.manifold.mobius_add(out, res)
        #out = self.manifold.expmap0(self.manifold.logmap0(out)+self.manifold.logmap0(res))

        out = self.activation(out)

        return out
