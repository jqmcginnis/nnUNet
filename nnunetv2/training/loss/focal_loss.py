from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.tensor_utilities import sum_tensor

class FocalLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, alpha: float = 0.25, gamma: float = 2., ddp: bool = True):
        """
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002
        """
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.apply_nonlin = apply_nonlin
        self.ddp = ddp

    def forward(self, x, y):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        bce_loss = F.binary_cross_entropy_with_logits(x, y, reduction='none')
        pt = torch.exp(-bce_loss)

        # compute the loss
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.ddp:
            loss = AllGatherGrad.apply(loss).mean()

        return loss
