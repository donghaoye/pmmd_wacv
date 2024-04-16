# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmdet.models.builder import HEADS, build_loss, LOSSES


@LOSSES.register_module()
class DepthmapL1Loss(nn.Module):
    """SigLoss.

        We adopt the implementation in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None
                 ):
        super(DepthmapL1Loss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.L1 = nn.L1Loss()

    def L1_mask_loss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input_valid = input[valid_mask]
            target_valid = target[valid_mask]
        
        body_depth_loss = self.L1(input_valid, target_valid)
        global_depth_loss = self.L1(input, target)

        depth_loss = body_depth_loss * 4.0 + global_depth_loss
        return depth_loss

    def forward(self, depth_pred, depth_gt):
        """Forward function."""
        
        loss_depth = self.loss_weight * self.L1_mask_loss(depth_pred, depth_gt)
        return loss_depth
