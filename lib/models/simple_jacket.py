import torch.nn as nn
from lib.lorentz.manifold import CustomLorentz

from lib.lorentz.layers import LorentzGlobalAvgPool2d

from lib.models.component_list import BLOCKS, DECODERS


class JacketSwapper(nn.Module):
    def __init__(self, manifold, manifold_2, space_only=False):
        super(JacketSwapper, self).__init__()

        self.manifold_1 = manifold
        self.manifold_2 = manifold_2

        self.space_only = space_only

    def forward(self, x_jacket, x_euclid):

        x_temp = self.manifold_1.logmap0(x_jacket)
        x_temp = x_temp + nn.functional.pad(x_euclid.permute(0, 2, 3, 1), pad=(1, 0))
        x_temp = self.manifold_2.rescale_to_max(x_temp)
        return self.manifold_2.expmap0(x_temp)


class JacketSwapper2D(nn.Module):
    def __init__(self, manifold, manifold_2, space_only=False):
        super(JacketSwapper2D, self).__init__()

        self.manifold_1 = manifold
        self.manifold_2 = manifold_2

        self.space_only = space_only

    def forward(self, x_jacket, x_euclid):

        x_temp = self.manifold_1.logmap0(x_jacket)
        x_temp = x_temp + nn.functional.pad(x_euclid, pad=(1, 0))
        x_temp = self.manifold_2.rescale_to_max(x_temp)
        return self.manifold_2.expmap0(x_temp)


class SimpleJacket(nn.Module):
    """ Implementation of ResNet models on manifolds. """

    def __init__(self, learnable=True):
        super(SimpleJacket, self).__init__()

        print("Initializing Manifolds...")
        self.encoder_manifold = CustomLorentz(k=1, learnable=learnable)
        self.decoder_manifold = CustomLorentz(k=1, learnable=learnable)

        self.input_conv = BLOCKS["lorentz_input"](manifold=self.encoder_manifold,
                                                  img_dim=3,
                                                  out_channels=64,
                                                  conv_type="efficient",
                                                  batch_type="batch2dvar",
                                                  bias=False)

        self.conv_1 = BLOCKS["LorentzCoreBottleneck"](manifold=self.encoder_manifold,
                                                  in_channels=64,
                                                  out_channels=64,
                                                  conv_type="efficient",
                                                  batch_type="batch2dvar",
                                                  bias=False)

        self.conv_2 = BLOCKS["LorentzCoreBottleneck"](manifold=self.encoder_manifold,
                                                  in_channels=64 * self.conv_1.expansion,
                                                  out_channels=128,
                                                  stride=2,
                                                  conv_type="efficient",
                                                  batch_type="batch2dvar",
                                                  bias=False)

        self.conv_3 = BLOCKS["LorentzCoreBottleneck"](manifold=self.encoder_manifold,
                                                  in_channels=128 * self.conv_2.expansion,
                                                  out_channels=256,
                                                  stride=2,
                                                  conv_type="efficient",
                                                  batch_type="batch2dvar",
                                                  bias=False)

        self.conv_4 = BLOCKS["LorentzCoreBottleneck"](manifold=self.encoder_manifold,
                                                  in_channels=256 * self.conv_3.expansion,
                                                  out_channels=512,
                                                  stride=2,
                                                  conv_type="efficient",
                                                  batch_type="batch2dvar",
                                                  bias=False)

        #self.avg_pool = LorentzGlobalAvgPool2d(self.encoder_manifold, w=None, keep_dim=True, last_dim=None)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.decoder = DECODERS["LorentzMLR"](self.decoder_manifold, 2048, 100)

        self.jacket_swap_encoder = JacketSwapper(self.encoder_manifold, self.encoder_manifold)
        self.jacket_swap_decoder = JacketSwapper2D(self.encoder_manifold, self.decoder_manifold)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, embeddings):
        out_in = self.input_conv(x)
        #out_in = self.jacket_swap_encoder(out_in, embeddings[0])
        out_in = self.encoder_manifold.logmap0(out_in)[...,1:].permute(0,3,1,2) + embeddings[0]

        out_1 = self.conv_1(out_in, res=False)
        out_1 = out_1+embeddings[1]
        #out_1 = self.jacket_swap_encoder(out_1, embeddings[1])

        out_2 = self.conv_2(out_1, res=False)
        out_2 = out_2+embeddings[2]
        #out_2 = self.jacket_swap_encoder(out_2, embeddings[2])

        out_3 = self.conv_3(out_2, res=False)
        out_3 = out_3+embeddings[3]
        #out_3 = self.jacket_swap_encoder(out_3, embeddings[3])

        out_4 = self.conv_4(out_3, res=False)
        #out_4 = out_4+embeddings[3]
        out_4 = self.avg_pool(out_4).view(out_4.size(0), -1)

        #out_4 = self.jacket_swap_decoder(out_4, embeddings[4])
        out_4 = self.decoder_manifold.rescale_to_max(out_4 + embeddings[4])
        out_4 = self.decoder_manifold.add_time(out_4)

        return self.decoder(out_4)

    def get_loss(self, logits, labels):
        return self.loss(logits, labels)

