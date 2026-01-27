import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import (
    LorentzConv2d,
    LorentzConv1By1,
    LorentzBatchNorm2d,
    LorentzAct,
    LorentzConv2d_hyperweight,
    LorentzPureConv,
    LorentzLayerNorm,
    LorentzBatchNorm2d_allvar,
    LorentzBatchNormCenterOffset2d,
    LorentzBatchNorm2dLVar,
)


class CustomLayerNorm(nn.Module):
    """Input Block of ResNet model"""

    def __init__(self, in_features):
        super(CustomLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # Make channel last (bs x H x W x C)
        x = self.layer_norm(x)

        return x.permute(0, 3, 1, 2)


CONV_TYPES = {"conv2d": LorentzConv2d, "pure_lorentz": LorentzPureConv}
BATCH_TYPES = {
    "batch2d": (LorentzBatchNorm2d, nn.BatchNorm2d),
    "batch2dvar": (LorentzBatchNorm2d_allvar, nn.BatchNorm2d),
    "batch2dLVar": (LorentzBatchNorm2dLVar, nn.BatchNorm2d),
    "batch_center_offset": (LorentzBatchNormCenterOffset2d, nn.BatchNorm2d),
    "layer2d": (LorentzLayerNorm, CustomLayerNorm),
    "syncBN": (LorentzBatchNorm2d, nn.SyncBatchNorm)
}


class ManifoldSwapper(nn.Module):
    def __init__(self, manifold, to_euclidean=False, space_only=False):
        super(ManifoldSwapper, self).__init__()

        self.manifold = manifold
        self.to_euclidean = to_euclidean
        self.space_only = space_only

    def forward(self, x):
        if self.to_euclidean:
            if self.space_only:
                return x.permute(0, 3, 1, 2)
            return self.manifold.logmap0(x)[..., 1:].permute(0, 3, 1, 2)

        x = x.permute(0, 2, 3, 1)
        return self.manifold.projx(nn.functional.pad(x, pad=(1, 0)))


class ManifoldSwapper1D(nn.Module):
    def __init__(self, manifold, to_euclidean=False, space_only=False):
        super(ManifoldSwapper1D, self).__init__()

        self.manifold = manifold
        self.to_euclidean = to_euclidean
        self.space_only = space_only

    def forward(self, x):
        if self.to_euclidean:
            if self.space_only:
                return x[..., 1:]
            return self.manifold.logmap0(x)[..., 1:]

        return self.manifold.projx(nn.functional.pad(x, pad=(1, 0)))


def get_Conv2d(
    conv_type,
    manifold,
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    bias=True,
    LFC_normalize=False,
    dilation=1,
):
    return CONV_TYPES[conv_type](
        manifold=manifold,
        in_channels=in_channels + 1,
        out_channels=out_channels + 1,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        LFC_normalize=LFC_normalize,
        dilation=dilation,
    )


def get_BatchNorm2d(batch_type, manifold, num_channels, euclid=False, norm_moment=0.1):
    if not euclid:
        return BATCH_TYPES[batch_type][0](
            manifold=manifold, num_channels=num_channels + 1, norm_moment=norm_moment
        )

    return BATCH_TYPES[batch_type][1](num_channels)


def get_Activation(manifold, activation=nn.ReLU(inplace=True)):
    return LorentzAct(activation, manifold)


class LorentzInputBlock(nn.Module):
    """Input Block of ResNet model"""

    def __init__(
        self,
        manifold: CustomLorentz,
        img_dim,
        out_channels,
        bias=True,
        padding=1,
        stride=1,
        conv_type="conv2d",
        batch_type="batch2d",
        norm_moment=0.1,
        activation=nn.ReLU(inplace=True),
        input_kernels=3,
    ):
        super(LorentzInputBlock, self).__init__()

        self.manifold = manifold

        self.conv = nn.Sequential(
            get_Conv2d(
                conv_type,
                self.manifold,
                img_dim,
                out_channels,
                kernel_size=input_kernels,
                padding=padding,
                stride=stride,
                bias=bias,
            ),
            get_BatchNorm2d(
                batch_type, self.manifold, out_channels, norm_moment=norm_moment
            ),
            get_Activation(self.manifold, activation),
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # Make channel last (bs x H x W x C)
        x = self.manifold.projx(F.pad(x, pad=(1, 0)))
        return self.conv(x)


class LorentzDilatedBasicBlock(nn.Module):
    """Basic Block for Lorentz ResNet-10, -18 and -34"""

    expansion = 1

    def __init__(
        self,
        manifold: CustomLorentz,
        in_channels,
        out_channels,
        stride=1,
        bias=True,
        conv_type="conv2d",
        batch_type="batch2d",
        norm_moment=0.1,
        activation=nn.ReLU(inplace=True),
        dilation=1,
        previous_dilation=1
    ):
        super(LorentzDilatedBasicBlock, self).__init__()

        self.manifold = manifold

        self.activation = get_Activation(self.manifold, activation)

        self.conv = nn.Sequential(
            get_Conv2d(
                conv_type,
                self.manifold,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                bias=bias,
                dilation=dilation
            ),
            get_BatchNorm2d(
                batch_type, self.manifold, out_channels, norm_moment=norm_moment
            ),
            get_Activation(self.manifold, activation),
            get_Conv2d(
                conv_type,
                self.manifold,
                out_channels,
                out_channels * LorentzDilatedBasicBlock.expansion,
                kernel_size=3,
                padding=previous_dilation,
                bias=bias,
                dilation=previous_dilation
            ),
            get_BatchNorm2d(
                batch_type,
                self.manifold,
                out_channels * LorentzDilatedBasicBlock.expansion,
                norm_moment=norm_moment,
            ),
        )

        self.shortcut = nn.Sequential()

        self.addition_weights = torch.nn.Parameter(torch.tensor([1.0, 1.0]))

        if stride != 1 or in_channels != out_channels * LorentzDilatedBasicBlock.expansion:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    conv_type,
                    self.manifold,
                    in_channels,
                    out_channels * LorentzDilatedBasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias,
                ),
                get_BatchNorm2d(
                    batch_type,
                    self.manifold,
                    out_channels * LorentzDilatedBasicBlock.expansion,
                    norm_moment=norm_moment,
                ),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        # Residual = Add space components

        out = out.narrow(-1, 1, res.shape[-1] - 1) + res.narrow(
            -1, 1, res.shape[-1] - 1
        )
        out = self.manifold.add_time(out)

        out = self.activation(out)

        return out


class LorentzDilatedBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        manifold: CustomLorentz,
        in_channels,
        out_channels,
        stride=1,
        bias=False,
        conv_type="conv2d",
        batch_type="batch2d",
        norm_moment=0.1,
        activation=nn.ReLU(inplace=True),
        dilation=1,
        previous_dilation=1
    ):
        super(LorentzDilatedBottleneck, self).__init__()

        self.manifold = manifold

        self.activation = get_Activation(self.manifold, activation)

        self.conv = nn.Sequential(
            get_Conv2d(
                conv_type,
                self.manifold,
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            get_BatchNorm2d(
                batch_type, self.manifold, out_channels, norm_moment=norm_moment
            ),
            get_Activation(self.manifold, activation),
            get_Conv2d(
                conv_type,
                self.manifold,
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                bias=bias,
                dilation=dilation
            ),
            get_BatchNorm2d(
                batch_type, self.manifold, out_channels, norm_moment=norm_moment
            ),
            get_Activation(self.manifold, activation),
            get_Conv2d(
                conv_type,
                self.manifold,
                out_channels,
                out_channels * LorentzDilatedBottleneck.expansion,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            get_BatchNorm2d(
                batch_type,
                self.manifold,
                out_channels * LorentzDilatedBottleneck.expansion,
                norm_moment=norm_moment,
            ),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzDilatedBottleneck.expansion:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    conv_type,
                    self.manifold,
                    in_channels,
                    out_channels * LorentzDilatedBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias,
                ),
                get_BatchNorm2d(
                    batch_type,
                    self.manifold,
                    out_channels * LorentzDilatedBottleneck.expansion,
                    norm_moment=norm_moment,
                ),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        # Residual = Add space components
        out = out.narrow(-1, 1, res.shape[-1] - 1) + res.narrow(
            -1, 1, res.shape[-1] - 1
        )
        out = self.manifold.add_time(out)

        out = self.activation(out)

        return out


class LorentzDilatedCoreBottleneck(nn.Module):
    """Residual block for Lorentz ResNet with > 50 layers"""

    expansion = 4

    def __init__(
        self,
        manifold: CustomLorentz,
        in_channels,
        out_channels,
        stride=1,
        bias=False,
        conv_type="conv2d",
        batch_type="batch2d",
        norm_moment=0.1,
        activation=nn.ReLU(inplace=True),
        simple_swap=False,
        dilation=1,
        previous_dilation=1
    ):
        super(LorentzDilatedCoreBottleneck, self).__init__()

        self.manifold = manifold

        self.activation = activation

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            get_BatchNorm2d(batch_type, manifold, out_channels, euclid=True),
            activation,
            ManifoldSwapper(manifold),
            get_Conv2d(
                conv_type,
                self.manifold,
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                bias=bias,
                dilation=dilation
            ),
            get_BatchNorm2d(
                batch_type, self.manifold, out_channels, norm_moment=norm_moment
            ),
            get_Activation(self.manifold, activation),
            ManifoldSwapper(manifold, to_euclidean=True, space_only=simple_swap),
            nn.Conv2d(
                out_channels,
                out_channels * LorentzDilatedCoreBottleneck.expansion,
                kernel_size=1,
                bias=bias,
            ),
            get_BatchNorm2d(
                batch_type,
                manifold,
                out_channels * LorentzDilatedCoreBottleneck.expansion,
                euclid=True,
            ),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzDilatedBottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * LorentzDilatedCoreBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=bias,
                ),
                get_BatchNorm2d(
                    batch_type,
                    manifold,
                    out_channels * LorentzDilatedCoreBottleneck.expansion,
                    euclid=True,
                ),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        # Residual = Add space components
        out = out + res

        out = self.activation(out)

        return out


class LorentzDilatedPureCoreBottleneck(nn.Module):
    """Residual block for Lorentz ResNet with > 50 layers"""

    expansion = 4

    def __init__(
        self,
        manifold: CustomLorentz,
        in_channels,
        out_channels,
        stride=1,
        bias=False,
        conv_type="conv2d",
        batch_type="batch2d",
        norm_moment=0.1,
        activation=nn.ReLU(inplace=True),
        simple_swap=False,
        dilation=1,
        previous_dilation=1
    ):
        super(LorentzDilatedPureCoreBottleneck, self).__init__()

        self.manifold = manifold

        self.activation = activation

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            get_BatchNorm2d(batch_type, manifold, out_channels, euclid=True),
            activation,
            ManifoldSwapper(manifold),
            LorentzPureConv(
                self.manifold,
                out_channels + 1,
                out_channels + 1,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                bias=bias,
                dilation=dilation
            ),
            get_BatchNorm2d(
                batch_type, self.manifold, out_channels, norm_moment=norm_moment
            ),
            get_Activation(self.manifold, activation),
            ManifoldSwapper(manifold, to_euclidean=True, space_only=simple_swap),
            nn.Conv2d(
                out_channels,
                out_channels * LorentzDilatedCoreBottleneck.expansion,
                kernel_size=1,
                bias=bias,
            ),
            get_BatchNorm2d(
                batch_type,
                manifold,
                out_channels * LorentzDilatedCoreBottleneck.expansion,
                euclid=True,
            ),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzDilatedPureCoreBottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * LorentzDilatedPureCoreBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=bias,
                ),
                get_BatchNorm2d(
                    batch_type,
                    manifold,
                    out_channels * LorentzDilatedPureCoreBottleneck.expansion,
                    euclid=True,
                ),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        # Residual = Add space components
        out = out + res

        out = self.activation(out)

        return out


class LorentzDilatedEfficientBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        manifold: CustomLorentz,
        in_channels,
        out_channels,
        stride=1,
        bias=False,
        conv_type="conv2d",
        batch_type="batch2d",
        norm_moment=0.1,
        activation=nn.ReLU(inplace=True),
        pure_core=False,
        dilation=1,
        previous_dilation=1
    ):
        super(LorentzDilatedEfficientBottleneck, self).__init__()

        self.manifold = manifold

        self.activation = get_Activation(self.manifold, activation)

        core_type = conv_type

        if pure_core:
            core_type = LorentzConv2d_hyperweight

        self.conv = nn.Sequential(
            LorentzConv1By1(
                self.manifold,
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            get_BatchNorm2d(
                batch_type, self.manifold, out_channels, norm_moment=norm_moment
            ),
            get_Activation(self.manifold, activation),
            get_Conv2d(
                core_type,
                self.manifold,
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                bias=bias,
                dilation=dilation
            ),
            get_BatchNorm2d(
                batch_type, self.manifold, out_channels, norm_moment=norm_moment
            ),
            get_Activation(self.manifold, activation),
            LorentzConv1By1(
                self.manifold,
                out_channels,
                out_channels * LorentzDilatedEfficientBottleneck.expansion,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            get_BatchNorm2d(
                batch_type,
                self.manifold,
                out_channels * LorentzDilatedEfficientBottleneck.expansion,
                norm_moment=norm_moment,
            ),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzDilatedEfficientBottleneck.expansion:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    conv_type,
                    self.manifold,
                    in_channels,
                    out_channels * LorentzDilatedEfficientBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias,
                ),
                get_BatchNorm2d(
                    batch_type,
                    self.manifold,
                    out_channels * LorentzDilatedEfficientBottleneck.expansion,
                    norm_moment=norm_moment,
                ),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        # Residual = Add space components
        out = out.narrow(-1, 1, res.shape[-1] - 1) + res.narrow(
            -1, 1, res.shape[-1] - 1
        )
        out = self.manifold.add_time(out)

        out = self.activation(out)

        return out


class LorentzDilatedInverseCoreBottleneck(nn.Module):
    """Residual block for Lorentz ResNet with > 50 layers"""

    expansion = 4

    def __init__(
        self,
        manifold: CustomLorentz,
        in_channels,
        out_channels,
        stride=1,
        bias=False,
        conv_type="conv2d",
        batch_type="batch2d",
        norm_moment=0.1,
        activation=nn.ReLU(inplace=True),
        simple_swap=False,
        dilation=1,
        previous_dilation=1
    ):
        super(LorentzDilatedInverseCoreBottleneck, self).__init__()

        self.manifold = manifold

        self.activation = activation

        self.conv = nn.Sequential(
            LorentzConv1By1(
                self.manifold,
                in_channels + 1,
                out_channels + 1,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            get_BatchNorm2d(
                batch_type, self.manifold, out_channels, norm_moment=norm_moment
            ),
            get_Activation(self.manifold, activation),
            ManifoldSwapper(manifold, to_euclidean=True, space_only=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                bias=bias,
                dilation=dilation
            ),
            get_BatchNorm2d(
                batch_type, manifold, out_channels, euclid=True, norm_moment=norm_moment
            ),
            activation,
            ManifoldSwapper(manifold),
            LorentzConv1By1(
                self.manifold,
                out_channels + 1,
                out_channels * LorentzDilatedInverseCoreBottleneck.expansion + 1,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            get_BatchNorm2d(
                batch_type,
                self.manifold,
                out_channels * LorentzDilatedInverseCoreBottleneck.expansion,
                norm_moment=norm_moment,
            ),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzDilatedInverseCoreBottleneck.expansion:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    conv_type,
                    self.manifold,
                    in_channels,
                    out_channels * LorentzDilatedInverseCoreBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias,
                ),
                get_BatchNorm2d(
                    batch_type,
                    self.manifold,
                    out_channels * LorentzDilatedInverseCoreBottleneck.expansion,
                    norm_moment=norm_moment,
                ),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        # Residual = Add space components
        out = out.narrow(-1, 1, res.shape[-1] - 1) + res.narrow(
            -1, 1, res.shape[-1] - 1
        )
        out = self.activation(out)

        out = self.manifold.add_time(out)

        return out
