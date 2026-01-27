import torch.nn as nn


class CustomLayerNorm(nn.Module):
    """ Input Block of ResNet model """

    def __init__(self, in_features):
        super(CustomLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        b, c,h,w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1,c)  # Make channel last (bs x H x W x C)
        x = self.layer_norm(x).reshape(b, h, w, c)
        return x.permute(0, 3, 1, 2)


BATCH_TYPES = {"batch2d": nn.BatchNorm2d,
               "layer2d": CustomLayerNorm,
               "group2d": nn.GroupNorm,
               "syncBN": nn.SyncBatchNorm}


def get_BatchNorm2d(batch_type, num_channels):
    if batch_type == "group2d":
        return nn.GroupNorm(num_channels//16, num_channels)

    return BATCH_TYPES[batch_type](num_channels)


class BasicBlock(nn.Module):
    """ Basic Block for ResNet-10, -18 and -34 """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, bias=False, batch_type="batch2d"):
        super(BasicBlock, self).__init__()

        self.activation = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias),
            get_BatchNorm2d(batch_type, out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=bias),
            get_BatchNorm2d(batch_type, out_channels * BasicBlock.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=bias),
                get_BatchNorm2d(batch_type, out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        out = out + res

        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    """ Residual block for ResNet with > 50 layers """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, bias=False, batch_type="batch2d", dilation=1):
        super(Bottleneck, self).__init__()

        self.activation = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            get_BatchNorm2d(batch_type, out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, bias=bias, dilation=dilation),
            get_BatchNorm2d(batch_type, out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=bias),
            get_BatchNorm2d(batch_type, out_channels * Bottleneck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=bias),
                get_BatchNorm2d(batch_type, out_channels * Bottleneck.expansion),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)

        out = out + res

        out = self.activation(out)

        return out
