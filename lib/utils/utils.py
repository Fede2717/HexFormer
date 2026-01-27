from enum import Enum

import torch


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


def rescale_to_max(euclid_vector, manifold):

    # scale all point based on the max
    # x_norm = torch.norm(euclid_vector, dim=-1, keepdim=True)
    # max_distance = torch.sqrt(manifold.k) * torch.arccosh(1e4 / manifold.k) - 0.1
    # max_norm = x_norm.view(x_norm.shape[0], -1).max(dim=-1, keepdim=True)[0]
    # max_norm[max_norm < max_distance] = max_distance
    #
    # max_norm = max_norm[(...,) + (None,) * (euclid_vector.ndim - max_norm.ndim)]
    # return euclid_vector / max_norm * max_distance

    # scale only the max back down
    x_norm = torch.norm(euclid_vector, dim=-1, keepdim=True)
    max_distance = torch.sqrt(manifold.k) * torch.arccosh(1e4 / manifold.k) - 0.1
    x_out = x_norm.clone()
    x_out[x_norm < max_distance] = max_distance
    return euclid_vector / x_out * max_distance


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res