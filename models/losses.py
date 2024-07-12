import torch
import torch.nn.functional as F
import torch.nn as nn


def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


def BCEDiceLoss(inputs, targets):
    # print(inputs.shape, targets.shape)
    # if targets.dim() == 4:
    #     targets = torch.squeeze(targets, dim=1)
    targets = targets.float()
    bce = F.binary_cross_entropy(inputs, targets)

    # inter = (inputs * targets).sum()
    # eps = 1e-5
    # dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    # return bce + 1 - dice
    return bce
