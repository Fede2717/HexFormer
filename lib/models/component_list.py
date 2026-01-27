import torch.nn as nn

from lib.Euclidean.blocks.resnet_blocks import BasicBlock, Bottleneck

from lib.lorentz.blocks.resnet_blocks import (
    LorentzBasicBlock,
    LorentzBottleneck,
    LorentzInputBlock,
    LorentzCoreBottleneck,
    LorentzEfficientBottleneck,
    LorentzPureCoreBottleneck,
    LorentzInverseCoreBottleneck
)

from lib.lorentz.layers import LorentzMLR, LorentzProjection
from lib.models.decoders import LorentzLinearDecoder, LorentzMLRBlockDecoder, LorentzPureLinearDecoder


def check_swap_req(block_0, block_1, same_man):
    if (block_0 in LORENTZ_BLOCKS and block_1 in LORENTZ_BLOCKS and same_man) \
            or (block_0 in EUCLID_BLOCKS + EUCLID_IN_BLOCKS and block_1 in EUCLID_BLOCKS + EUCLID_IN_BLOCKS):

        return False
    else:
        return True


class EuclidDecoder(nn.Module):

    def __init__(self, manifold, in_dim, n_classes):
        super(EuclidDecoder, self).__init__()

        self.layer = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.layer(x)


INPUT_BLOCKS = {"lorentz": LorentzInputBlock,
                "euclid": "euclid"}

BLOCKS = {"euclid_basic": BasicBlock,
          "euclid_bottle": Bottleneck,
          "lorentz_input": LorentzInputBlock,
          "LorentzBasic": LorentzBasicBlock,
          "LorentzBottleneck": LorentzBottleneck,
          "LorentzCoreBottleneck": LorentzCoreBottleneck,
          "lorentz_efficient_bottle": LorentzEfficientBottleneck,
          "LorentzPureCoreBottleneck": LorentzPureCoreBottleneck,
          "LorentzInverseCoreBottleneck": LorentzInverseCoreBottleneck}

DECODERS = {"LorentzMLR": LorentzMLR,
            "LorentzPure": LorentzPureLinearDecoder,
            "euclid_decoder": EuclidDecoder,
            "LorentzLinear": LorentzLinearDecoder,
            "LorentzMLRBlock": LorentzMLRBlockDecoder}

LOSSES = {"bce": nn.CrossEntropyLoss}

LORENTZ_BLOCKS = [LorentzInputBlock,
                  LorentzBasicBlock,
                  LorentzBottleneck,
                  LorentzEfficientBottleneck,
                  LorentzMLR,
                  LorentzInverseCoreBottleneck,
                  LorentzLinearDecoder,
                  LorentzMLRBlockDecoder,
                  LorentzPureLinearDecoder]

EUCLID_BLOCKS = [BasicBlock,
                 Bottleneck,
                 EuclidDecoder, ]

EUCLID_IN_BLOCKS = [LorentzCoreBottleneck,
                    LorentzPureCoreBottleneck]
