import random

import torch
import torch.nn as nn

from lib.Euclidean.blocks.resnet_blocks import BasicBlock, Bottleneck

from lib.lorentz.blocks.resnet_blocks import (
    LorentzInputBlock,
    ManifoldSwapper,
    ManifoldSwapper1D,
)

from lib.lorentz.layers import LorentzGlobalAvgPool2d
from lib.lorentz.manifold import CustomLorentz
from lib.models.component_list import (EUCLID_BLOCKS,
                                       EUCLID_IN_BLOCKS,
                                       LORENTZ_BLOCKS,
                                       BLOCKS,
                                       INPUT_BLOCKS,
                                       DECODERS,
                                       LOSSES,
                                       check_swap_req)
#from hclassification.gen_tree_dist import get_cifar100_distances


class ResNet(nn.Module):
    """ Implementation of ResNet models on manifolds. """

    def __init__(
            self,
            block,
            num_blocks,
            #manifold=None,
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
            learnk=False,
            sep_mans=False,
            sep_decoder_man=False,
            random_init_k=False
    ):
        super(ResNet, self).__init__()

        self.img_dim = img_dim[0]
        self.res = img_dim[1]
        self.in_channels = 64
        self.conv3_dim = 128
        self.conv4_dim = 256
        self.embed_dim = embed_dim

        self.norm_moment = norm_moment

        self.conv_type = conv_type
        self.batch_type = batch_type
        self.input_kernel = input_kernel
        self.space_only_swap = space_only_swap

        self.bias = bias

        print("making manifold")
        if not sep_mans:
            init = CustomLorentz(k=round(random.uniform(0.41, 7), 2), learnable=learnk)
            self.b0_man = self.b1_man = self.b2_man = self.b3_man = self.b4_man = init
            blocks_same = True
        else:
            self.b0_man = CustomLorentz(k=round(random.uniform(0.41, 7), 2),learnable=learnk)
            self.b1_man = CustomLorentz(k=round(random.uniform(0.41, 7), 2),learnable=learnk)
            self.b2_man = CustomLorentz(k=round(random.uniform(0.41, 7), 2),learnable=learnk)
            self.b3_man = CustomLorentz(k=round(random.uniform(0.41, 7), 2),learnable=learnk)
            self.b4_man = CustomLorentz(k=round(random.uniform(0.41, 7), 2),learnable=learnk)
            blocks_same = False
        if not sep_decoder_man:
            self.decoder_man = self.b4_man
            decoder_same = True
        else:
            self.decoder_man = CustomLorentz(k=round(random.uniform(0.41, 7), 2),learnable=learnk)
            decoder_same = False

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
            block_4 = BLOCKS[block[4]]
        else:
            block_0 = block_1 = block_2 = block_3 = block_4 = BLOCKS[block]

        if (block_1 == block_2 and blocks_same) or not check_swap_req(block_1, block_2, blocks_same):
            self.swap_1 = nn.Sequential()
        else:
            to_euclid = block_1 in LORENTZ_BLOCKS and block_2 in EUCLID_BLOCKS+EUCLID_IN_BLOCKS
            from_euclid = block_1 in EUCLID_BLOCKS + EUCLID_IN_BLOCKS and block_2 in LORENTZ_BLOCKS
            self.swap_1 = ManifoldSwapper(self.b0_man, self.b1_man, to_euclid, from_euclid, space_only=space_only_swap)

        if (block_2 == block_3 and blocks_same) or not check_swap_req(block_2, block_3, blocks_same):
            self.swap_2 = nn.Sequential()
        else:
            to_euclid = block_2 in LORENTZ_BLOCKS and block_3 in EUCLID_BLOCKS+EUCLID_IN_BLOCKS
            from_euclid = block_2 in EUCLID_BLOCKS + EUCLID_IN_BLOCKS and block_3 in LORENTZ_BLOCKS
            self.swap_2 = ManifoldSwapper(self.b1_man, self.b2_man, to_euclid, from_euclid, space_only=space_only_swap)

        if (block_3 == block_4 and blocks_same) or not check_swap_req(block_3, block_4, blocks_same):
            self.swap_3 = nn.Sequential()
        else:
            to_euclid = block_3 in LORENTZ_BLOCKS and block_4 in EUCLID_BLOCKS+EUCLID_IN_BLOCKS
            from_euclid = block_3 in EUCLID_BLOCKS + EUCLID_IN_BLOCKS and block_4 in LORENTZ_BLOCKS
            self.swap_3 = ManifoldSwapper(self.b2_man, self.b3_man, to_euclid, from_euclid, space_only=space_only_swap)

        if in_conv_type!="default":
            block_0 = INPUT_BLOCKS[in_conv_type]

        self.swap_0 = nn.Sequential()
        if block_0 != block_1 or not blocks_same:
            if check_swap_req(block_0, block_1, decoder_same):
                to_euclid = block_0 in LORENTZ_BLOCKS and block_1 in EUCLID_BLOCKS + EUCLID_IN_BLOCKS
                from_euclid = block_0 in EUCLID_BLOCKS + EUCLID_IN_BLOCKS and block_1 in LORENTZ_BLOCKS
                self.swap_0 = ManifoldSwapper(self.b0_man, self.b1_man, to_euclid, from_euclid,
                                              space_only=space_only_swap)
            else:
                self.swap_0 = nn.Sequential()

        ###########################################################################################################

        in_channel = self.in_channels

        self.conv0 = self._get_inConv(block_0, manifold=self.b0_man)
        self.conv1_x = self._make_layer(block_1, out_channels=self.in_channels, num_blocks=num_blocks[0], stride=1, manifold=self.b1_man)
        self.conv2_x = self._make_layer(block_2, out_channels=self.conv3_dim, num_blocks=num_blocks[1], stride=2, manifold=self.b2_man)
        self.conv3_x = self._make_layer(block_3, out_channels=self.conv4_dim, num_blocks=num_blocks[2], stride=2, manifold=self.b3_man)
        self.conv4_x = self._make_layer(block_4, out_channels=self.embed_dim, num_blocks=num_blocks[3], stride=2, manifold=self.b4_man)

        self.in_channels = in_channel

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) if block_4 in EUCLID_BLOCKS + EUCLID_IN_BLOCKS else LorentzGlobalAvgPool2d(self.b4_man, w=None, keep_dim=True, last_dim=None)

        decoder = DECODERS[decoder_type]

        if (block_4 in LORENTZ_BLOCKS and decoder in LORENTZ_BLOCKS and decoder_same) \
                or (block_4 in EUCLID_BLOCKS and decoder in EUCLID_BLOCKS):
            self.decoder_swap = nn.Sequential()
        else:
            to_euclid = block_4 in LORENTZ_BLOCKS and decoder in EUCLID_BLOCKS
            from_euclid = block_4 in EUCLID_BLOCKS + EUCLID_IN_BLOCKS and decoder in LORENTZ_BLOCKS
            self.decoder_swap = ManifoldSwapper1D(self.b4_man, self.decoder_man, to_euclid, from_euclid, space_only=space_only_swap)

        if self.decoder_swap:
            self.decoder = decoder(self.decoder_man, embed_dim * block_4.expansion, num_classes)
        else:

            self.decoder = decoder(self.decoder_man, embed_dim * block_4.expansion, num_classes)

        self.block_4 = block_4

        self.is_euclidean = {block_0, block_1, block_2, block_3, block_4}.issubset(EUCLID_BLOCKS) and decoder_type == "euclid_decoder"

        self.manifolds = [self.b0_man, self.b1_man, self.b2_man, self.b3_man, self.b4_man, self.decoder_man]

    @torch.no_grad()
    def update_mans(self):
        for man in self.manifolds:
            if man is not None:
                man.update_limits()

    def forward(self, x, return_embeddings=False):

        #with torch.no_grad():
        self.update_mans()

        embeddings = self.encode(x, return_embeddings=return_embeddings)

        if return_embeddings:
            return embeddings

        output = self.decode(embeddings)
        return output

    def _make_layer(self, block, out_channels, num_blocks, stride, manifold):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            if block in EUCLID_BLOCKS:
                layers.append(block(self.in_channels, out_channels, stride, self.bias, batch_type=self.batch_type))
            else:
                layers.append(
                    block(
                        manifold,
                        self.in_channels,
                        out_channels,
                        stride,
                        self.bias,
                        conv_type=self.conv_type,
                        batch_type=self.batch_type,
                        norm_moment=self.norm_moment,
                        simple_swap=self.space_only_swap
                    )
                )

            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def prep_x(self, x):
        x = x.permute(0, 2, 3, 1)  # Make channel last (bs x H x W x C)
        return self.manifold.projx(nn.functional.pad(x, pad=(1, 0)))

    def stabalize_ks(self):
        manifolds = [self.b0_man, self.b1_man, self.b2_man, self.b3_man, self.b4_man, self.decoder_man]
        for mdl in manifolds:
            for p in mdl.parameters():
                p.data.clamp_(0.1, 10)

    def encode(self, x, return_embeddings=False):

        out_0 = self.conv0(x)
        out_0 = self.swap_0(out_0)

        out_1 = self.conv1_x(out_0)
        out_1 = self.swap_1(out_1)

        out_2 = self.conv2_x(out_1)
        out_2 = self.swap_2(out_2)

        out_3 = self.conv3_x(out_2)
        out_3 = self.swap_3(out_3)

        out_4 = self.conv4_x(out_3)

        out = self.avg_pool(out_4)
        out = out.view(out.size(0), -1)

        if return_embeddings:
            return out_0, out_1, out_2, out_3, out

        return out

    def decode(self, x):
        x = self.decoder_swap(x)
        return self.decoder(x)

    def decode_hier(self, x):
        x = self.decoder_swap(x)
        return self.decoder(x, return_distance=True)

    def swap_decoders(self, decoder_type, num_classes):
        decoder = DECODERS[decoder_type]
        old_device = next(self.decoder.parameters()).device
        self.decoder = decoder(self.manifold, self.embed_dim*self.block_4.expansion, num_classes).to(device=old_device)

        if not check_swap_req(self.block_4, decoder):
            self.decoder_swap = nn.Sequential()
        else:
            self.decoder_swap = ManifoldSwapper1D(self.manifold, self.decoder in EUCLID_BLOCKS+EUCLID_IN_BLOCKS, space_only=self.space_only_swap)

    def _get_inConv(self, block, manifold):

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
                    manifold,
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
                    manifold,
                    self.img_dim,
                    self.in_channels,
                    self.bias,
                    input_kernels=7,
                    padding=3,
                    stride=2,
                    conv_type=self.conv_type,
                    batch_type=self.batch_type,
                    norm_moment=self.norm_moment)

    def get_loss(self, logits, labels, hierarchy=False):

        if hierarchy:
            CIFAR_DISTANCES = None#torch.tensor(get_cifar100_distances(), device=logits.device)

            predicted = torch.nn.functional.softmax(logits).max(-1)[1]
            predicted = torch.cat((predicted.unsqueeze(-1), labels.unsqueeze(-1)), dim=-1)

            distances = CIFAR_DISTANCES[predicted[:, 0], predicted[:, 1]]/3
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
            loss = (loss*distances).mean()
            return loss

        return self.loss(logits, labels)

    def custom_h_loss(self, x, labels):

        logits, original_distances = self.decode_hier(x)
        CIFAR_DISTANCES = None #torch.tensor(get_cifar100_distances(), device=logits.device)

        predicted = torch.nn.functional.softmax(logits).max(-1)[1]
        predicted = torch.cat((predicted.unsqueeze(-1), labels.unsqueeze(-1)), dim=-1)

        # get tree distance from predicted to all other classes
        tree_distances = CIFAR_DISTANCES[predicted[:, 0]].to(torch.float32)
        distance_modulator = tree_distances

        #distance_modulator = torch.ones_like(original_distances, dtype=tree_distances.dtype)
        #distance_modulator[range(tree_distances.shape[0]), predicted[:, 0]] = tree_distances

        # modulated_distance = logits*(torch.pow(distance_modulator.clamp(min=1), torch.sign(logits)))

        loss = torch.nn.functional.cross_entropy(logits, labels) - 0.01*torch.mean(distance_modulator*original_distances)

        return loss, logits


    def custom_h_loss_embeds(self, x, labels):

        logits, original_distances = self.decode_hier(x)

        loss = torch.nn.functional.cross_entropy(logits, labels)

        CIFAR_DISTANCES = None# torch.tensor(get_cifar100_distances(), device=logits.device)

        distances = 0
        neg_distances = 0
        i = 0

        for c in labels.unique():
            i = i+1
            mask = labels == c

            x_temp = self.manifold.projx(x)

            group = x_temp[mask]
            if group.shape[0] <= 1:
                continue
            dist = self.manifold.sqdist(group, group.unsqueeze(1))
            distances = distances + dist.mean()/2

            #neg_dist = self.manifold.sqdist(group, x_temp[mask == False].unsqueeze(1))
            #neg_dist = 1 / neg_dist
            #neg_distances = neg_distances + neg_dist.mean()

            # tree_dists = CIFAR_DISTANCES[c, labels][CIFAR_DISTANCES[c, labels]>0].unsqueeze(-1)
            # neg_dist = self.manifold.sqdist(group, x_temp[mask==False].unsqueeze(1))
            # neg_dist = tree_dists/neg_dist
            # neg_distances = neg_distances + neg_dist.mean()

        loss = loss + 1e-5 / (distances / i) #+ 500 * neg_distances / i

        return loss, logits


def resnet10(**kwargs):
    """Constructs a ResNet-10 model."""
    model = ResNet(num_blocks=[1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(num_blocks=[2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(num_blocks=[3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(num_blocks=[3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(num_blocks=[3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model."""
    model = ResNet(num_blocks=[3, 8, 36, 3], **kwargs)
    return model


def resnet_selector(num_classes, loss, args):

    sizes = {"10": resnet10,
             "18": resnet18,
             "34": resnet34,
             "50": resnet50,
             "101": resnet101,
             "152": resnet152
             }

    return sizes[str(args.resnet_size)](block=args.block[0] if len(args.block) == 1 else args.block,
                                        embed_dim=args.embed_dim,
                                        num_classes=num_classes,
                                        decoder_type=args.decoder,
                                        conv_type=args.conv_type,
                                        in_conv_type=args.in_conv_type,
                                        batch_type=args.batch_type,
                                        norm_moment=args.norm_moment,
                                        loss=loss,
                                        space_only_swap=args.space_only_swap,
                                        learnk=args.learnk,
                                        sep_mans=args.sep_mans,
                                        sep_decoder_man=args.sep_decoder,
                                        random_init_k=args.rand_init)
