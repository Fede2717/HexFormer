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
    LorentzBatchNorm2d_DirectVar,
    LorentzBatchNormCenterOffset2d,
    LorentzBatchNorm2dLVar,
    LorentzConv2d_kernels_old,
    LorentzKernelTransform,
    LorentzKernelAttentionTransform,
    LorentzPureConv_transform,
    LorentzBatchNorm2d_DistVar,
    PureHyperbolicEfficientConv,
    PureHyperbolicEfficientConvNorm
)
from lib.utils.utils import rescale_to_max


class CustomLayerNorm(nn.Module):
    """ Input Block of ResNet model """

    def __init__(self, in_features):
        super(CustomLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # Make channel last (bs x H x W x C)
        x = self.layer_norm(x)

        return x.permute(0, 3, 1, 2)


CONV_TYPES = {"conv2d": LorentzConv2d,
              "pure_lorentz": LorentzPureConv,
              "kernel_lorentz": LorentzConv2d_kernels_old,
              "kernel_transform": LorentzKernelTransform,
              "kernel_attention": LorentzKernelAttentionTransform,
              "pure_lorentz_transform": LorentzPureConv_transform,
              "efficient": PureHyperbolicEfficientConv,
              "efficient_norm": PureHyperbolicEfficientConvNorm}
BATCH_TYPES = {"batch2d": (LorentzBatchNorm2d, nn.BatchNorm2d),
               "batch2d_direct": (LorentzBatchNorm2d_DirectVar, nn.BatchNorm2d),
               "batch2dvar": (LorentzBatchNorm2d_allvar, nn.BatchNorm2d),
               "batch2dLVar": (LorentzBatchNorm2dLVar, nn.BatchNorm2d),
               "batch_center_offset": (LorentzBatchNormCenterOffset2d, nn.BatchNorm2d),
               "layer2d": (LorentzLayerNorm, CustomLayerNorm),
               "syncBN": (LorentzBatchNorm2d, nn.SyncBatchNorm),
               "dist_lorentz": (LorentzBatchNorm2d_DistVar, nn.BatchNorm2d)}


class ManifoldSwapper(nn.Module):

    def __init__(self, manifold=None, manifold_2=None, to_euclidean=False, from_euclidean=False, space_only=False, skip=False):
        super(ManifoldSwapper, self).__init__()

        self.manifold = manifold
        self.manifold_2 = manifold_2
        self.to_euclidean = to_euclidean
        self.from_euclidean = from_euclidean
        self.space_only = space_only
        self.skip = skip

    def forward(self, x, res=0):

        if self.skip:
            return x + 0

        if self.to_euclidean:
            if self.space_only:
                return x[..., 1:].permute(0, 3, 1, 2)
            out = self.manifold.logmap0(x)[..., 1:]
            return (out + res).permute(0, 3, 1, 2)

        if self.from_euclidean:
            x = x + res
            x = x.permute(0, 2, 3, 1)
            x = self.manifold_2.rescale_to_max(x)
            return self.manifold_2.projx(nn.functional.pad(x, pad=(1, 0)))

        if self.space_only:
            return self.manifold_2.projx(x)

        x = self.manifold.logmap0(x)
        x = self.manifold_2.rescale_to_max(x)

        return self.manifold_2.expmap0(x)


class ManifoldSwapper1D(nn.Module):

    def __init__(self, manifold=None, manifold_2=None, to_euclidean=False, from_euclidean=False, space_only=False):
        super(ManifoldSwapper1D, self).__init__()

        self.manifold = manifold
        self.manifold_2 = manifold_2
        self.to_euclidean = to_euclidean
        self.from_euclidean = from_euclidean
        self.space_only = space_only

    def forward(self, x):

        if self.to_euclidean:
            if self.space_only:
                return x[..., 1:]
            return self.manifold.logmap0(x)[..., 1:]

        if self.from_euclidean:
            x = self.manifold_2.rescale_to_max(x)
            return self.manifold_2.add_time(x)

        if self.space_only:
            return self.manifold_2.projx(x)

        x = self.manifold.logmap0(x)
        x = self.manifold_2.rescale_to_max(x)

        return self.manifold_2.expmap0(x)


def get_Conv2d(conv_type, manifold, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, LFC_normalize=False):

    return CONV_TYPES[conv_type](
        manifold=manifold, 
        in_channels=in_channels+1,
        out_channels=out_channels+1,
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        bias=bias, 
        LFC_normalize=LFC_normalize
    )


def get_BatchNorm2d(batch_type, manifold, num_channels, euclid=False, norm_moment=0.1):

    if not euclid:
        return BATCH_TYPES[batch_type][0](manifold=manifold, num_channels=num_channels+1, norm_moment=norm_moment)

    return BATCH_TYPES[batch_type][1](num_channels)


def get_Activation(manifold, activation=nn.ReLU(inplace=True)):
    return LorentzAct(activation, manifold)


class LorentzInputBlock(nn.Module):
    """ Input Block of ResNet model """

    def __init__(self, manifold: CustomLorentz, img_dim, out_channels, bias=True, padding=1, stride=1, conv_type="conv2d", batch_type="batch2d", norm_moment=0.1, activation=nn.ReLU(inplace=True), input_kernels=3):
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
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels, norm_moment=norm_moment),
            get_Activation(self.manifold, activation),
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # Make channel last (bs x H x W x C)
        x = self.manifold.projx(F.pad(x, pad=(1, 0)))
        return self.conv(x)


class LorentzBasicBlock_backup(nn.Module):
    """ Basic Block for Lorentz ResNet-10, -18 and -34 """

    expansion = 1

    def __init__(self, manifold: CustomLorentz, in_channels, out_channels, stride=1, bias=True, conv_type="conv2d", batch_type="batch2d", norm_moment=0.1, activation=nn.ReLU(inplace=True)):
        super(LorentzBasicBlock_backup, self).__init__()

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
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels, norm_moment=norm_moment),
            get_Activation(self.manifold, activation),
            get_Conv2d(
                conv_type,
                self.manifold,
                out_channels,
                out_channels * LorentzBasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels * LorentzBasicBlock.expansion, norm_moment=norm_moment),
        )

        self.shortcut = nn.Sequential()

        self.addition_weights = torch.nn.Parameter(torch.tensor([1.,1.]))

        if stride != 1 or in_channels != out_channels * LorentzBasicBlock.expansion:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    conv_type,
                    self.manifold,
                    in_channels,
                    out_channels * LorentzBasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias
                ),
                get_BatchNorm2d(
                    batch_type, self.manifold, out_channels * LorentzBasicBlock.expansion, norm_moment=norm_moment
                ),
            )

    def forward(self, x, res=None):
        out = self.conv(x)

        # Residual = Add space components

        # Residual = Add space components
        if res is None:
            res = self.shortcut(x)
            out = out.narrow(-1, 1, res.shape[-1] - 1) + res.narrow(-1, 1, res.shape[-1] - 1)
        else:
            out = out.narrow(-1, 1, res.shape[-1] - 1) + res

        out = self.manifold.add_time(out)

        out = self.activation(out)

        return out


class LorentzBasicBlock(nn.Module):
    """ Basic Block for Lorentz ResNet-10, -18 and -34 """

    expansion = 1

    def __init__(self, manifold: CustomLorentz, in_channels, out_channels, stride=1, bias=True, conv_type="conv2d", batch_type="batch2d", norm_moment=0.1, activation=nn.ReLU(inplace=True), simple_swap=False):
        super(LorentzBasicBlock, self).__init__()

        self.manifold = manifold

        self.activation = get_Activation(self.manifold)

        self.conv = nn.Sequential(
            get_Conv2d(
                conv_type,
                self.manifold,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels),
            get_Activation(self.manifold),
            get_Conv2d(
                conv_type,
                self.manifold,
                out_channels,
                out_channels * LorentzBasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels * LorentzBasicBlock.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzBasicBlock.expansion:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    conv_type,
                    self.manifold,
                    in_channels,
                    out_channels * LorentzBasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias
                ),
                get_BatchNorm2d(
                    batch_type, self.manifold, out_channels * LorentzBasicBlock.expansion
                ),
            )

    def forward(self, x, res=None):
        res = self.shortcut(x)
        out = self.conv(x)

        # Residual = Add space components

        #res = self.manifold.add_time(res)
        #res = self.manifold.logmap0(res)
        #res = self.manifold.transp0(out, res)
        #out = self.manifold.expmap(out, res)
        out = out.narrow(-1, 1, res.shape[-1]-1) + res.narrow(-1, 1, res.shape[-1]-1)
        out = self.manifold.add_time(out)
        out = self.manifold.rescale_to_max(out)
        out = self.manifold.projx(out)
        # center = self.manifold.centroid(torch.cat([res.unsqueeze(-2), out.unsqueeze(-2)], dim=-2))
        # out = self.manifold.scale(center, factor=2)

        out = self.activation(out)
        return out


class LorentzBottleneck(nn.Module):
    expansion = 4

    def __init__(self, manifold: CustomLorentz, in_channels, out_channels, stride=1, bias=False, conv_type="conv2d",
                 batch_type="batch2d", norm_moment=0.1, activation=nn.ReLU(inplace=True), pre_calc_res=False, simple_swap=False):
        super(LorentzBottleneck, self).__init__()

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
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels, norm_moment=norm_moment),
            get_Activation(self.manifold, activation),
            get_Conv2d(
                conv_type,
                self.manifold,
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels, norm_moment=norm_moment),
            get_Activation(self.manifold, activation),
            get_Conv2d(
                conv_type,
                self.manifold,
                out_channels,
                out_channels * LorentzBottleneck.expansion,
                kernel_size=1,
                padding=0,
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels * LorentzBottleneck.expansion, norm_moment=norm_moment),
        )

        self.shortcut = nn.Sequential()

        if (stride != 1 or in_channels != out_channels * LorentzBottleneck.expansion) and not pre_calc_res:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    conv_type,
                    self.manifold,
                    in_channels,
                    out_channels * LorentzBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias
                ),
                get_BatchNorm2d(
                    batch_type, self.manifold, out_channels * LorentzBottleneck.expansion, norm_moment=norm_moment
                ),
            )

    def forward(self, x, res=True):

        out = self.conv(x)

        # Residual = Add space components
        if res is True:
            res = self.shortcut(x)
            out = out.narrow(-1, 1, res.shape[-1] - 1) + res.narrow(-1, 1, res.shape[-1] - 1)
            out = self.manifold.add_time(out)

        out = self.activation(out)

        return out


class LorentzCoreBottleneck(nn.Module):
    """ Residual block for Lorentz ResNet with > 50 layers """

    expansion = 4

    def __init__(self, manifold: CustomLorentz, in_channels, out_channels, stride=1, bias=False, conv_type="conv2d", batch_type="batch2d", norm_moment=0.1, activation=nn.ReLU(inplace=True), simple_swap=False):
        super(LorentzCoreBottleneck, self).__init__()

        self.manifold = manifold

        self.activation = activation

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            get_BatchNorm2d(batch_type, manifold, out_channels, euclid=True),
            activation,
            ManifoldSwapper(manifold, manifold, to_euclidean=False, from_euclidean=True),
            get_Conv2d(
                conv_type,
                self.manifold,
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels, norm_moment=norm_moment),
            get_Activation(self.manifold, activation),
            ManifoldSwapper(manifold, manifold, to_euclidean=True, from_euclidean=False, space_only=simple_swap),
            nn.Conv2d(out_channels, out_channels * LorentzCoreBottleneck.expansion, kernel_size=1, bias=bias),
            get_BatchNorm2d(batch_type, manifold, out_channels * LorentzCoreBottleneck.expansion, euclid=True),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzBottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * LorentzCoreBottleneck.expansion, kernel_size=1, stride=stride, bias=bias),
                get_BatchNorm2d(batch_type, manifold, out_channels * LorentzCoreBottleneck.expansion, euclid=True),
            )

    def forward(self, x, res=True):
        out = self.conv(x)

        # Residual = Add space components
        if res:
            res = self.shortcut(x)
            out = out + res

        out = self.activation(out)

        return out


class LorentzPureCoreBottleneck(nn.Module):
    """ Residual block for Lorentz ResNet with > 50 layers """

    expansion = 4

    def __init__(self, manifold: CustomLorentz, in_channels, out_channels, stride=1, bias=False, conv_type="conv2d", batch_type="batch2d", norm_moment=0.1, activation=nn.ReLU(inplace=True), simple_swap=False):
        super(LorentzPureCoreBottleneck, self).__init__()

        self.manifold = manifold

        self.activation = activation

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            get_BatchNorm2d(batch_type, manifold, out_channels, euclid=True),
            activation,
            ManifoldSwapper(manifold, manifold, to_euclidean=False, from_euclidean=True),
            LorentzPureConv(
                self.manifold,
                out_channels+1,
                out_channels+1,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels, norm_moment=norm_moment),
            get_Activation(self.manifold, activation),
            ManifoldSwapper(manifold, manifold, to_euclidean=True, from_euclidean=False, space_only=simple_swap),
            nn.Conv2d(out_channels, out_channels * LorentzCoreBottleneck.expansion, kernel_size=1, bias=bias),
            get_BatchNorm2d(batch_type, manifold, out_channels * LorentzCoreBottleneck.expansion, euclid=True)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzBottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * LorentzCoreBottleneck.expansion, kernel_size=1, stride=stride, bias=bias),
                get_BatchNorm2d(batch_type, manifold, out_channels * LorentzCoreBottleneck.expansion, euclid=True),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        # Residual = Add space components
        out = out + res

        out = self.activation(out)

        return out


class LorentzEfficientBottleneck(nn.Module):
    expansion = 4

    def __init__(self, manifold: CustomLorentz, in_channels, out_channels, stride=1, bias=False, conv_type="conv2d",
                 batch_type="batch2d", norm_moment=0.1, activation=nn.ReLU(inplace=True), pure_core=False):
        super(LorentzEfficientBottleneck, self).__init__()

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
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels, norm_moment=norm_moment),
            get_Activation(self.manifold, activation),
            get_Conv2d(
                core_type,
                self.manifold,
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels, norm_moment=norm_moment),
            get_Activation(self.manifold, activation),
            LorentzConv1By1(
                self.manifold,
                out_channels,
                out_channels * LorentzBottleneck.expansion,
                kernel_size=1,
                padding=0,
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels * LorentzBottleneck.expansion, norm_moment=norm_moment),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzBottleneck.expansion:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    conv_type,
                    self.manifold,
                    in_channels,
                    out_channels * LorentzBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias
                ),
                get_BatchNorm2d(
                    batch_type, self.manifold, out_channels * LorentzBottleneck.expansion, norm_moment=norm_moment
                ),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        # Residual = Add space components
        out = out.narrow(-1, 1, res.shape[-1] - 1) + res.narrow(-1, 1, res.shape[-1] - 1)
        out = self.manifold.add_time(out)

        out = self.activation(out)

        return out


class LorentzInverseCoreBottleneck(nn.Module):
    """ Residual block for Lorentz ResNet with > 50 layers """

    expansion = 4

    def __init__(self, manifold: CustomLorentz, in_channels, out_channels, stride=1, bias=False, conv_type="conv2d", batch_type="batch2d", norm_moment=0.1, activation=nn.ReLU(inplace=True), simple_swap=False):
        super(LorentzInverseCoreBottleneck, self).__init__()

        self.manifold = manifold

        self.activation = activation


        self.conv = nn.Sequential(
            LorentzConv1By1(
                self.manifold,
                in_channels+1,
                out_channels+1,
                kernel_size=1,
                padding=0,
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels, norm_moment=norm_moment),
            get_Activation(self.manifold, activation),
            ManifoldSwapper(manifold, manifold, to_euclidean=True, from_euclidean=False, space_only=simple_swap),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=bias),
            get_BatchNorm2d(batch_type, manifold, out_channels, euclid=True, norm_moment=norm_moment),
            activation,
            ManifoldSwapper(manifold, manifold, to_euclidean=False, from_euclidean=True),

            LorentzConv1By1(
                self.manifold,
                out_channels+1,
                out_channels * LorentzBottleneck.expansion+1,
                kernel_size=1,
                padding=0,
                bias=bias
            ),
            get_BatchNorm2d(batch_type, self.manifold, out_channels * LorentzBottleneck.expansion, norm_moment=norm_moment),)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LorentzBottleneck.expansion:
            self.shortcut = nn.Sequential(
                get_Conv2d(
                    conv_type,
                    self.manifold,
                    in_channels,
                    out_channels * LorentzBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias
                ),
                get_BatchNorm2d(
                    batch_type, self.manifold, out_channels * LorentzBottleneck.expansion, norm_moment=norm_moment
                ),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        # Residual = Add space components
        out = out.narrow(-1, 1, res.shape[-1] - 1) + res.narrow(-1, 1, res.shape[-1] - 1)
        out = self.activation(out)

        out = self.manifold.add_time(out)

        return out


