import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.Euclidean.blocks.resnet_blocks import BasicBlock, Bottleneck

from lib.lorentz.blocks.dilated_resnet_blocks import (
    LorentzDilatedBasicBlock,
    LorentzDilatedBottleneck,
    LorentzInputBlock,
    LorentzDilatedCoreBottleneck,
    LorentzDilatedEfficientBottleneck,
    ManifoldSwapper,
    ManifoldSwapper1D,
    LorentzDilatedPureCoreBottleneck,
    LorentzDilatedInverseCoreBottleneck
)

from lib.lorentz.layers import LorentzMLR, LorentzGlobalAvgPool2d, LorentzTransform
from lib.lorentz.manifold import CustomLorentz
from lib.models.decoders import LorentzLinearDecoder, LorentzMLRBlockDecoder


class EuclidDecoder(nn.Module):

    def __init__(self, manifold, in_dim, n_classes):
        super(EuclidDecoder, self).__init__()

        self.layer = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return F.normalize(self.layer(x), dim=1)


INPUT_BLOCKS = {"lorentz": LorentzInputBlock,
                "euclid": "euclid"}

BLOCKS = {"euclid_basic": BasicBlock,
          "euclid_bottle": Bottleneck,
          "lorentz_input": LorentzInputBlock,
          "lorentz_basic": LorentzDilatedBasicBlock,
          "LorentzBottleneck": LorentzDilatedBottleneck,
          "LorentzCoreBottleneck": LorentzDilatedCoreBottleneck,
          "lorentz_efficient_bottle": LorentzDilatedEfficientBottleneck,
          "LorentzPureCoreBottleneck": LorentzDilatedPureCoreBottleneck,
          "LorentzInverseCoreBottleneck": LorentzDilatedInverseCoreBottleneck}

DECODERS = {"LorentzMLR": LorentzMLR,
            "euclid_decoder": EuclidDecoder,
            "LorentzLinear": LorentzLinearDecoder,
            "LorentzMLRBlock": LorentzMLRBlockDecoder}

LOSSES = {"bce": nn.CrossEntropyLoss}

LORENTZ_BLOCKS = [LorentzInputBlock,
                  LorentzDilatedBasicBlock,
                  LorentzDilatedBottleneck,
                  LorentzDilatedEfficientBottleneck,
                  LorentzMLR,
                  LorentzDilatedInverseCoreBottleneck,
                  LorentzLinearDecoder,
                  LorentzMLRBlockDecoder]

EUCLID_BLOCKS = [BasicBlock,
                 Bottleneck,
                 EuclidDecoder,]

EUCLID_IN_BLOCKS = [LorentzDilatedCoreBottleneck,
                    LorentzDilatedPureCoreBottleneck]


def check_swap_req(block_0, block_1):
    if (block_0 in LORENTZ_BLOCKS and block_1 in LORENTZ_BLOCKS) \
        or (block_0 in EUCLID_BLOCKS + EUCLID_IN_BLOCKS and block_1 in EUCLID_BLOCKS + EUCLID_IN_BLOCKS):
        return False
    else:
        return True


class DilatedResNet(nn.Module):
    """ Implementation of ResNet models on manifolds. """

    def __init__(
            self,
            block,
            num_blocks,
            manifold=None,
            img_dim=[3, 32, 32],
            embed_dim=512,
            num_classes=100,
            decoder_type="lorentz",
            bias=False,
            input_kernel=3,
            space_only_swap=False,
            conv_type="conv2d",
            in_conv_type="default",
            batch_type="batch2d",
            norm_moment=0.1,
            loss=nn.CrossEntropyLoss,
            dilated=True,
            output_size=8
    ):
        super(DilatedResNet, self).__init__()

        self.img_dim = img_dim[0]
        self.res = img_dim[1]
        self.in_channels = embed_dim // 8
        self.conv3_dim = embed_dim // 4
        self.conv4_dim = embed_dim // 2
        self.embed_dim = embed_dim

        self.norm_moment = norm_moment

        self.conv_type = conv_type
        self.batch_type = batch_type
        self.input_kernel = input_kernel

        self.space_only_swap = space_only_swap
        self.bias = bias

        if not manifold:
            manifold = CustomLorentz()

        if type(loss) == str:
            self.loss = LOSSES[loss]()
        else:
            self.loss = loss()

        ##################################### set swaps and blocks ###############################################
        if type(block) == list:
            block_0 = BLOCKS[block[0]]
            block_1 = BLOCKS[block[1]]
            block_2 = BLOCKS[block[2]]
            block_3 = BLOCKS[block[3]]
            self.block_4 = BLOCKS[block[4]]

            if block_1 == block_2 or not check_swap_req(block_1, block_2):
                self.swap_1 = nn.Sequential()
            else:
                to_euclid = block_1 in LORENTZ_BLOCKS and block_2 in EUCLID_BLOCKS+EUCLID_IN_BLOCKS
                self.swap_1 = ManifoldSwapper(manifold, to_euclid, space_only=space_only_swap)

            if block_2 == block_3 or not check_swap_req(block_2, block_3):
                self.swap_2 = nn.Sequential()
            else:
                to_euclid = block_2 in LORENTZ_BLOCKS and block_3 in EUCLID_BLOCKS+EUCLID_IN_BLOCKS
                self.swap_2 = ManifoldSwapper(manifold, to_euclid, space_only=space_only_swap)

            if block_3 == self.block_4 or not check_swap_req(block_3, self.block_4):
                self.swap_3 = nn.Sequential()
            else:
                to_euclid = block_3 in LORENTZ_BLOCKS and self.block_4 in EUCLID_BLOCKS+EUCLID_IN_BLOCKS
                self.swap_3 = ManifoldSwapper(manifold, to_euclid, space_only=space_only_swap)
        else:
            block_0 = block_1 = block_2 = block_3 = self.block_4 = BLOCKS[block]
            self.swap_0 = self.swap_1 = self.swap_2 = self.swap_3 = nn.Sequential()

        if in_conv_type!="default":
            block_0 = INPUT_BLOCKS[in_conv_type]

        if block_0 != block_1:
            if check_swap_req(block_0, block_1):
                self.swap_0 = ManifoldSwapper(manifold, block_1 in EUCLID_BLOCKS + EUCLID_IN_BLOCKS,
                                              space_only=space_only_swap)
            else:
                self.swap_0 = nn.Sequential()

        ###########################################################################################################
        self.manifold = manifold

        self.conv0 = self._get_inConv(block_0)
        self.conv1_x = self._make_layer(block_1, out_channels=self.in_channels, num_blocks=num_blocks[0], stride=1)
        self.conv2_x = self._make_layer(block_2, out_channels=self.conv3_dim, num_blocks=num_blocks[1], stride=2)

        dilation_rate = 2

        if dilated and output_size <= 8:
            self.conv3_x = self._make_layer(block_3, out_channels=self.conv4_dim, num_blocks=num_blocks[2], stride=1, dilation=dilation_rate)
            dilation_rate *= 2
        else:
            self.conv3_x = self._make_layer(block_3, out_channels=self.conv4_dim, num_blocks=num_blocks[2], stride=2)

        if dilated and output_size <= 8:
            self.conv4_x = self._make_layer(self.block_4, out_channels=self.embed_dim, num_blocks=num_blocks[3], stride=1, dilation=dilation_rate)
        else:
            self.conv4_x = self._make_layer(self.block_4, out_channels=self.embed_dim, num_blocks=num_blocks[3], stride=2)


        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) if self.block_4 in EUCLID_BLOCKS + EUCLID_IN_BLOCKS else LorentzGlobalAvgPool2d(self.manifold, keep_dim=True)

        decoder = DECODERS[decoder_type]
        self.decoder = decoder(manifold, embed_dim*self.block_4.expansion, num_classes)

        if check_swap_req(self.block_4, decoder):
            self.decoder_swap = nn.Sequential()
        else:
            self.decoder_swap = ManifoldSwapper1D(manifold, self.decoder in EUCLID_BLOCKS+EUCLID_IN_BLOCKS, space_only=space_only_swap)

    def forward(self, x, return_embeddings=False):
        embeddings = self.encode(x, return_embeddings=return_embeddings)

        if return_embeddings:
            return embeddings
        output = self.decode(embeddings)
        return output

    def _make_layer(self, block, out_channels, num_blocks, stride, dilation=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            if block in EUCLID_BLOCKS:
                layers.append(block(self.in_channels, out_channels, stride, self.bias, batch_type=self.batch_type, dilation=dilation))
            else:
                layers.append(
                    block(
                        self.manifold,
                        self.in_channels,
                        out_channels,
                        stride,
                        self.bias,
                        conv_type=self.conv_type,
                        batch_type=self.batch_type,
                        norm_moment=self.norm_moment,
                        dilation=dilation
                    )
                )

            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def encode(self, x, return_embeddings=False):
        out = self.conv0(x)
        out = self.swap_0(out)

        out_1 = self.conv1_x(out)
        out_1 = self.swap_1(out_1)

        out_2 = self.conv2_x(out_1)
        out_2 = self.swap_2(out_2)

        out_3 = self.conv3_x(out_2)
        out_3 = self.swap_3(out_3)

        out_4 = self.conv4_x(out_3)

        out = self.avg_pool(out_4)
        out = out.view(out.size(0), -1)

        if return_embeddings:
            return out, out_1, out_2, out_3, out_4

        return out

    def decode(self, x):
        x = self.decoder_swap(x)
        return self.decoder(x)

    def swap_decoders(self, decoder_type, num_classes):
        decoder = DECODERS[decoder_type]
        old_device = next(self.decoder.parameters()).device
        self.decoder = decoder(self.manifold, self.embed_dim*self.block_4.expansion, num_classes).to(device=old_device)

        if not check_swap_req(self.block_4, decoder):
            self.decoder_swap = nn.Sequential()
        else:
            self.decoder_swap = ManifoldSwapper1D(self.manifold, self.decoder in EUCLID_BLOCKS+EUCLID_IN_BLOCKS, space_only=self.space_only_swap)


    def _get_inConv(self, block):

        if self.res <= 64:
            if block in EUCLID_BLOCKS+EUCLID_IN_BLOCKS:
                return nn.Sequential(
                    nn.Conv2d(
                        self.img_dim,
                        self.in_channels,
                        kernel_size=self.input_kernel,
                        padding=1,
                        bias=self.bias
                    ),
                    nn.BatchNorm2d(self.in_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                return LorentzInputBlock(
                    self.manifold,
                    self.img_dim,
                    self.in_channels,
                    self.bias,
                    conv_type=self.conv_type,
                    batch_type=self.batch_type,
                    input_kernels=self.input_kernel,
                    norm_moment=self.norm_moment
                )
        else:
            if block in EUCLID_BLOCKS+EUCLID_IN_BLOCKS:
                return nn.Sequential(
                    nn.Conv2d(
                        self.img_dim,
                        self.in_channels,
                        kernel_size=7,
                        padding=3,
                        stride=2,
                        bias=self.bias
                    ),
                    nn.BatchNorm2d(self.in_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
            else:
                return LorentzInputBlock(
                    self.manifold,
                    self.img_dim,
                    self.in_channels,
                    self.bias,
                    input_kernels=7,
                    padding=3,
                    stride=2,
                    conv_type=self.conv_type,
                    batch_type=self.batch_type,
                    norm_moment=self.norm_moment)

    def get_loss(self, logits, labels):
        return self.loss(logits, labels)


def resnet10(**kwargs):
    """Constructs a ResNet-10 model."""
    model = DilatedResNet(num_blocks=[1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = DilatedResNet(num_blocks=[2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = DilatedResNet(num_blocks=[3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = DilatedResNet(num_blocks=[3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = DilatedResNet(num_blocks=[3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model."""
    model = DilatedResNet(num_blocks=[3, 8, 36, 3], **kwargs)
    return model


def resnet_selector(num_classes, loss, args):

    sizes = {"10": resnet10,
             "18": resnet18,
             "34": resnet34,
             "50": resnet50,
             "101": resnet101
             }

    return sizes[str(args.resnet_size)](block=args.block[0] if len(args.block)==1 else args.block,
                                   embed_dim=args.embed_dim,
                                   num_classes=num_classes,
                                   decoder_type=args.decoder,
                                   conv_type=args.conv_type,
                                   in_conv_type=args.in_conv_type,
                                   batch_type=args.batch_type,
                                   norm_moment=args.norm_moment,
                                   loss=loss,)
