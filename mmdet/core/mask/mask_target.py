# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch.nn.modules.utils import _pair

import mmcv


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, has_masks_list,
                cfg):
    """Compute mask target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images.
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals.
        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of
            each image.
        cfg (dict): Config dict that specifies the mask size.

    Returns:
        list[Tensor]: Mask target of each image.

    Example:
        >>> import mmcv
        >>> import mmdet
        >>> from mmdet.core.mask import BitmapMasks
        >>> from mmdet.core.mask.mask_target import *
        >>> H, W = 17, 18
        >>> cfg = mmcv.Config({'mask_size': (13, 14)})
        >>> rng = np.random.RandomState(0)
        >>> # Positive proposals (tl_x, tl_y, br_x, br_y) for each image
        >>> pos_proposals_list = [
        >>>     torch.Tensor([
        >>>         [ 7.2425,  5.5929, 13.9414, 14.9541],
        >>>         [ 7.3241,  3.6170, 16.3850, 15.3102],
        >>>     ]),
        >>>     torch.Tensor([
        >>>         [ 4.8448, 6.4010, 7.0314, 9.7681],
        >>>         [ 5.9790, 2.6989, 7.4416, 4.8580],
        >>>         [ 0.0000, 0.0000, 0.1398, 9.8232],
        >>>     ]),
        >>> ]
        >>> # Corresponding class index for each proposal for each image
        >>> pos_assigned_gt_inds_list = [
        >>>     torch.LongTensor([7, 0]),
        >>>     torch.LongTensor([5, 4, 1]),
        >>> ]
        >>> # Ground truth mask for each true object for each image
        >>> gt_masks_list = [
        >>>     BitmapMasks(rng.rand(8, H, W), height=H, width=W),
        >>>     BitmapMasks(rng.rand(6, H, W), height=H, width=W),
        >>> ]
        >>> mask_targets = mask_target(
        >>>     pos_proposals_list, pos_assigned_gt_inds_list,
        >>>     gt_masks_list, cfg)
        >>> assert mask_targets.shape == (5,) + cfg['mask_size']
    """
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]

    # mask_targets = map(mask_target_single, pos_proposals_list,
    #                    pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    # mask_targets = list(mask_targets)
    # if len(mask_targets) > 0:
    #     mask_targets = torch.cat(mask_targets)
    # return mask_targets

    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, has_masks_list, cfg_list)
    
    return tuple(map(torch.cat, zip(*mask_targets)))


# def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
#     """Compute mask target for each positive proposal in the image.

#     Args:
#         pos_proposals (Tensor): Positive proposals.
#         pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
#         gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
#             or Polygon.
#         cfg (dict): Config dict that indicate the mask size.

#     Returns:
#         Tensor: Mask target of each positive proposals in the image.

#     Example:
#         >>> import mmcv
#         >>> import mmdet
#         >>> from mmdet.core.mask import BitmapMasks
#         >>> from mmdet.core.mask.mask_target import *  # NOQA
#         >>> H, W = 32, 32
#         >>> cfg = mmcv.Config({'mask_size': (7, 11)})
#         >>> rng = np.random.RandomState(0)
#         >>> # Masks for each ground truth box (relative to the image)
#         >>> gt_masks_data = rng.rand(3, H, W)
#         >>> gt_masks = BitmapMasks(gt_masks_data, height=H, width=W)
#         >>> # Predicted positive boxes in one image
#         >>> pos_proposals = torch.FloatTensor([
#         >>>     [ 16.2,   5.5, 19.9, 20.9],
#         >>>     [ 17.3,  13.6, 19.3, 19.3],
#         >>>     [ 14.8,  16.4, 17.0, 23.7],
#         >>>     [  0.0,   0.0, 16.0, 16.0],
#         >>>     [  4.0,   0.0, 20.0, 16.0],
#         >>> ])
#         >>> # For each predicted proposal, its assignment to a gt mask
#         >>> pos_assigned_gt_inds = torch.LongTensor([0, 1, 2, 1, 1])
#         >>> mask_targets = mask_target_single(
#         >>>     pos_proposals, pos_assigned_gt_inds, gt_masks, cfg)
#         >>> assert mask_targets.shape == (5,) + cfg['mask_size']
#     """
#     device = pos_proposals.device
#     mask_size = _pair(cfg.mask_size)
#     binarize = not cfg.get('soft_mask_target', False)
#     num_pos = pos_proposals.size(0)
#     if num_pos > 0:
#         proposals_np = pos_proposals.cpu().numpy()
#         maxh, maxw = gt_masks.height, gt_masks.width
#         proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
#         proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
#         pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

#         mask_targets = gt_masks.crop_and_resize(
#             proposals_np,
#             mask_size,
#             device=device,
#             inds=pos_assigned_gt_inds,
#             binarize=binarize).to_ndarray()

#         mask_targets = torch.from_numpy(mask_targets).float().to(device)
#     else:
#         mask_targets = pos_proposals.new_zeros((0, ) + mask_size)

#     return mask_targets


# def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
#     mask_size = cfg.mask_size
#     num_pos = pos_proposals.size(0)
#     mask_targets = []
#     # has_masks_targets = []
#     if num_pos > 0:
#         proposals_np = pos_proposals.cpu().numpy()
#         pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
#         for i in range(num_pos):
#             gt_mask = gt_masks[pos_assigned_gt_inds[i]]
#             bbox = proposals_np[i, :].astype(np.int32)
#             x1, y1, x2, y2 = bbox
#             w = np.maximum(x2 - x1 + 1, 2)
#             h = np.maximum(y2 - y1 + 1, 2)
#             # crop_mask = gt_mask[y1:y1 + h, x1:x1 + w]
            
#             # debug for out of the image
#             # # print(">>> gt_mask.shape: ", gt_mask.shape) # (512, 832)
#             h_img, w_img = gt_mask.shape
#             start_x1, start_y1 = 0, 0
#             if x1 < 0:
#                 start_x1 = -1 * x1
#             if y1 < 0:
#                 start_y1 = -1 * y1
            
#             if (x1 < 0 or y1 < 0) and (start_y1 + h) < h_img and (start_x1 + w) < w_img \
#                 and start_y1 < h and  start_x1 < w:
#                 crop_mask = np.zeros((h, w))
#                 crop_mask[start_y1:, start_x1:] = gt_mask[:h-start_y1, :w-start_x1] # w = np.maximum(x2 - x1 + 1, 2) already add outofimage
#             else:
#                 crop_mask = gt_mask[y1:y1 + h, x1:x1 + w]
            
#             #if (crop_mask.shape[0] < 2 or crop_mask.shape[1] < 2) or crop_mask.shape[0] != h or crop_mask.shape[1] != w:
#             if (crop_mask.shape[0] < 2 or crop_mask.shape[1] < 2):
#                 mask_targets.append(torch.zeros((1, mask_size, mask_size)).to(torch.uint8).to(pos_proposals.device))
#                 # mask_targets.append(torch.zeros((1, mask_size, mask_size)).to(torch.bool).to(pos_proposals.device)) # torch 1.6
#                 # has_masks_targets.append(has_masks[0] * 0)
#                 continue
#             # mask is uint8 both before and after resizing

#             if hasattr(cfg, 'roi_resize') and cfg.roi_resize:
#                 # roi resize
#                 padded_mask = np.zeros((max(h, w), max(h, w)))
#                 padded_mask[:crop_mask.shape[0], :crop_mask.shape[1]] = crop_mask
#                 crop_mask = padded_mask

#             if hasattr(cfg, 'roi_pad_center') and cfg.roi_pad_center:
#                 # roi_pad_center
#                 max_hw = max(h, w)
#                 padded_mask = np.zeros((max_hw, max_hw)).astype(np.uint8)
                
#                 pad_h, pad_w = 0, 0
#                 if max_hw > h:
#                     pad_h = (max_hw - h) * 0.5
#                     pad_h = int(pad_h)
#                 if max_hw > w:
#                     pad_w = (max_hw - w) * 0.5
#                     pad_w = int(pad_w)
                
#                 padded_mask[pad_h:crop_mask.shape[0]+pad_h, pad_w:crop_mask.shape[1]+pad_w] = crop_mask # align center
#                 # padded_mask[:crop_mask.shape[0], :crop_mask.shape[1]] = crop_mask # align left_up point
#                 crop_mask = padded_mask

#             target = mmcv.imresize(crop_mask, (mask_size, mask_size))
#             target = target[None]
#             print("target.shape: ", target.shape)
#             mask_targets.append(torch.from_numpy(target).to(pos_proposals.device))
#             # has_masks_targets.append(has_masks[pos_assigned_gt_inds[i]])
#     else:
#         pass

#     if len(mask_targets) == 0:
#         mask_targets = [torch.zeros((mask_size, mask_size)).to(torch.bool).to(pos_proposals.device)]
#         # has_masks_targets = [has_masks[0] * 0]

#     # return tuple(map(
#     #     lambda x: torch.stack(x).float().to(pos_proposals.device),
#     #     [mask_targets]))

#     # mask_targets = torch.stack(mask_targets).float().to(pos_proposals.device)
#     mask_targets = torch.cat(mask_targets).float().to(pos_proposals.device)
#     # mask_targets = mask_targets.float().to(pos_proposals.device)
#     return mask_targets




def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, has_masks, cfg):
    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    mask_targets = []
    has_masks_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 2)
            h = np.maximum(y2 - y1 + 1, 2)
            # crop_mask = gt_mask[y1:y1 + h, x1:x1 + w]
            
            # debug for out of the image
            # # print(">>> gt_mask.shape: ", gt_mask.shape) # (512, 832)
            h_img, w_img = gt_mask.shape
            start_x1, start_y1 = 0, 0
            if x1 < 0:
                start_x1 = -1 * x1
            if y1 < 0:
                start_y1 = -1 * y1
            
            if (x1 < 0 or y1 < 0) and (start_y1 + h) < h_img and (start_x1 + w) < w_img \
                and start_y1 < h and  start_x1 < w:
                crop_mask = np.zeros((h, w))
                crop_mask[start_y1:, start_x1:] = gt_mask[:h-start_y1, :w-start_x1] # w = np.maximum(x2 - x1 + 1, 2) already add outofimage
            else:
                crop_mask = gt_mask[y1:y1 + h, x1:x1 + w]
            
            #if (crop_mask.shape[0] < 2 or crop_mask.shape[1] < 2) or crop_mask.shape[0] != h or crop_mask.shape[1] != w:
            if (crop_mask.shape[0] < 2 or crop_mask.shape[1] < 2):
                #mask_targets.append(torch.zeros((mask_size, mask_size)).to(torch.uint8).to(pos_proposals.device))
                mask_targets.append(torch.zeros((mask_size, mask_size)).to(torch.bool).to(pos_proposals.device)) # torch 1.6
                has_masks_targets.append(has_masks[0] * 0)
                continue
            # mask is uint8 both before and after resizing

            if hasattr(cfg, 'roi_resize') and cfg.roi_resize:
                # roi resize
                padded_mask = np.zeros((max(h, w), max(h, w)))
                padded_mask[:crop_mask.shape[0], :crop_mask.shape[1]] = crop_mask
                crop_mask = padded_mask

            if hasattr(cfg, 'roi_pad_center') and cfg.roi_pad_center:
                # roi_pad_center
                max_hw = max(h, w)
                padded_mask = np.zeros((max_hw, max_hw)).astype(np.uint8)
                
                pad_h, pad_w = 0, 0
                if max_hw > h:
                    pad_h = (max_hw - h) * 0.5
                    pad_h = int(pad_h)
                if max_hw > w:
                    pad_w = (max_hw - w) * 0.5
                    pad_w = int(pad_w)
                
                padded_mask[pad_h:crop_mask.shape[0]+pad_h, pad_w:crop_mask.shape[1]+pad_w] = crop_mask # align center
                # padded_mask[:crop_mask.shape[0], :crop_mask.shape[1]] = crop_mask # align left_up point
                crop_mask = padded_mask

            target = mmcv.imresize(crop_mask, (mask_size, mask_size))
            mask_targets.append(torch.from_numpy(target).to(pos_proposals.device))
            has_masks_targets.append(has_masks[pos_assigned_gt_inds[i]])
    else:
        pass

    if len(mask_targets) == 0:
        mask_targets = [torch.zeros((mask_size, mask_size)).to(torch.bool).to(pos_proposals.device)]
        has_masks_targets = [has_masks[0] * 0]

    return tuple(map(
        lambda x: torch.stack(x).float().to(pos_proposals.device),
        [mask_targets, has_masks_targets]))
