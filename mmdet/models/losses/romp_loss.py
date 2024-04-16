from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import sys, os
# root_dir = os.path.join(os.path.dirname(__file__),'..')
# if root_dir not in sys.path:
#     sys.path.insert(0, root_dir)

import time
import pickle
import numpy as np

from mmdet.core.utils.eval_utils import batch_compute_similarity_transform_torch


def batch_kp_2d_l2_loss(real, pred, weights=None):
    vis = (real>-1.).sum(-1)==real.shape[-1]
    pred[~vis] = real[~vis]
    error = torch.norm(real-pred, p=2, dim=-1)
    if weights is not None:
        error = error * weights.to(error.device)
    loss = error.sum(-1) / (1e-6+vis.sum(-1))
    return loss

def align_by_parts(joints, align_inds=None):
    if align_inds is None:
        return joints
    pelvis = joints[:, align_inds].mean(1)
    return joints - torch.unsqueeze(pelvis, dim=1)

def calc_mpjpe(real, pred, align_inds=None, sample_wise=True, trans=None, return_org=False):
    vis_mask = real[:,:,[-1]] > 0.1

    if vis_mask.float().sum() < 0.1:
        return torch.tensor(0).float().cuda()

    #print('input', pred.shape, real.shape, vis_mask.shape)
    if align_inds is not None:
        pred_aligned = align_by_parts(pred,align_inds=align_inds)
        if trans is not None:
            pred_aligned += trans
        real_aligned = align_by_parts(real,align_inds=align_inds)
    else:
        pred_aligned, real_aligned = pred, real
    #print('aligned', pred_aligned.shape, real_aligned.shape)

    mpjpe = torch.norm(pred_aligned - real_aligned[:,:,:3], p=2, dim=-1, keepdim=True)
    mpjpe_each = (mpjpe * vis_mask.float()).sum() / vis_mask.float().sum()

    # mpjpe_each = compute_mpjpe(pred_aligned, real_aligned[:,:,:3], vis_mask, sample_wise=sample_wise)

    if return_org:
        return mpjpe_each.mean(), (real_aligned, pred_aligned, vis_mask)
    return mpjpe_each#.mean()

def calc_pampjpe(real, pred, sample_wise=True, return_transform_mat=False):
    real, pred = real.float(), pred.float()
    # extracting the keypoints that all samples have the annotations
    vis_mask = (real[:,:,[-1]] > 0.1).sum(0)==len(real)
    vis_mask = vis_mask.view(-1)

    if vis_mask.float().sum() < 0.1:
        return torch.tensor(0).float().cuda()

    pred_tranformed, PA_transform = batch_compute_similarity_transform_torch(pred[:,vis_mask], real[:,vis_mask,:3], return_pa=True)
    
    # pa_mpjpe = torch.norm(pred_tranformed - real[:,vis_mask,:3], p=2, dim=-1, keepdim=True)
    # pa_mpjpe_each = pa_mpjpe.mean(-1)

    pa_mpjpe_each = compute_mpjpe(pred_tranformed, real[:,vis_mask,:3], sample_wise=sample_wise)

    if return_transform_mat:
        return pa_mpjpe_each.mean(), PA_transform
    else:
        return pa_mpjpe_each.mean()

def compute_mpjpe(predicted, target, valid_mask=None, pck_joints=None, sample_wise=True):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape, print(predicted.shape, target.shape)
    mpjpe = torch.norm(predicted - target, p=2, dim=-1, keepdim=True)
    #print(mpjpe.shape, predicted.shape, target.shape, valid_mask.shape)
    if pck_joints is None:
        if sample_wise:
            mpjpe_batch = (mpjpe*valid_mask.float()).sum(-1)/valid_mask.float().sum(-1) if valid_mask is not None else mpjpe.mean(-1)
            # mpjpe_batch = (mpjpe*valid_mask.float()).sum()/valid_mask.float().sum() if valid_mask is not None else mpjpe.mean()
        else:
            mpjpe_batch = mpjpe[valid_mask] if valid_mask is not None else mpjpe
        return mpjpe_batch
    else:
        mpjpe_pck_batch = mpjpe[:,pck_joints]
        return mpjpe_pck_batch