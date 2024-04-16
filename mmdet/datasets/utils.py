# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

from mmcv.cnn import VGG
from mmcv.runner.hooks import HOOKS, Hook

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import (LoadAnnotations, LoadImageFromFile,
                                      LoadPanopticAnnotations)
# from mmdet.models.dense_heads import GARPNHead, RPNHead
from mmdet.models.roi_heads.mask_heads import FusedSemanticHead


def replace_ImageToTensor(pipelines):
    """Replace the ImageToTensor transform in a data pipeline to
    DefaultFormatBundle, which is normally useful in batch inference.

    Args:
        pipelines (list[dict]): Data pipeline configs.

    Returns:
        list: The new pipeline list with all ImageToTensor replaced by
            DefaultFormatBundle.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='ImageToTensor', keys=['img']),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='DefaultFormatBundle'),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> assert expected_pipelines == replace_ImageToTensor(pipelines)
    """
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_ImageToTensor(
                pipeline['transforms'])
        elif pipeline['type'] == 'ImageToTensor':
            warnings.warn(
                '"ImageToTensor" pipeline is replaced by '
                '"DefaultFormatBundle" for batch inference. It is '
                'recommended to manually replace it in the test '
                'data pipeline in your config file.', UserWarning)
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines


def get_loading_pipeline(pipeline):
    """Only keep loading image and annotations related configuration.

    Args:
        pipeline (list[dict]): Data pipeline configs.

    Returns:
        list[dict]: The new pipeline list with only keep
            loading image and annotations related configuration.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True),
        ...    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        ...    dict(type='RandomFlip', flip_ratio=0.5),
        ...    dict(type='Normalize', **img_norm_cfg),
        ...    dict(type='Pad', size_divisor=32),
        ...    dict(type='DefaultFormatBundle'),
        ...    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True)
        ...    ]
        >>> assert expected_pipelines ==\
        ...        get_loading_pipeline(pipelines)
    """
    loading_pipeline_cfg = []
    for cfg in pipeline:
        obj_cls = PIPELINES.get(cfg['type'])
        # TODO：use more elegant way to distinguish loading modules
        if obj_cls is not None and obj_cls in (LoadImageFromFile,
                                               LoadAnnotations,
                                               LoadPanopticAnnotations):
            loading_pipeline_cfg.append(cfg)
    assert len(loading_pipeline_cfg) == 2, \
        'The data pipeline in your config file must include ' \
        'loading image and annotations related pipeline.'
    return loading_pipeline_cfg


@HOOKS.register_module()
class NumClassCheckHook(Hook):

    def _check_head(self, runner):
        """Check whether the `num_classes` in head matches the length of
        `CLASSES` in `dataset`.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        model = runner.model
        dataset = runner.data_loader.dataset
        if dataset.CLASSES is None:
            runner.logger.warning(
                f'Please set `CLASSES` '
                f'in the {dataset.__class__.__name__} and'
                f'check if it is consistent with the `num_classes` '
                f'of head')
        else:
            assert type(dataset.CLASSES) is not str, \
                (f'`CLASSES` in {dataset.__class__.__name__}'
                 f'should be a tuple of str.'
                 f'Add comma if number of classes is 1 as '
                 f'CLASSES = ({dataset.CLASSES},)')
            for name, module in model.named_modules():
                if hasattr(module, 'num_classes') and not isinstance(
                        module, (RPNHead, VGG, FusedSemanticHead, GARPNHead)):
                    assert module.num_classes == len(dataset.CLASSES), \
                        (f'The `num_classes` ({module.num_classes}) in '
                         f'{module.__class__.__name__} of '
                         f'{model.__class__.__name__} does not matches '
                         f'the length of `CLASSES` '
                         f'{len(dataset.CLASSES)}) in '
                         f'{dataset.__class__.__name__}')

    def before_train_epoch(self, runner):
        """Check whether the training dataset is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner)

    def before_val_epoch(self, runner):
        """Check whether the dataset in val epoch is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner)



import copy
from collections import Sequence

import mmcv
from mmcv.runner import obj_from_dict
import torch

import matplotlib.pyplot as plt
import numpy as np
import cv2
from .concat_dataset import ConcatDataset
from .repeat_dataset import RepeatDataset
from .. import datasets


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.

    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale


def show_ann(coco, img, ann_info):
    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    coco.showAnns(ann_info)
    plt.show()


def get_dataset(data_cfg):
    if data_cfg['type'] == 'RepeatDataset':
        return RepeatDataset(
            get_dataset(data_cfg['dataset']), data_cfg['times'])

    if isinstance(data_cfg['ann_file'], (list, tuple)):
        ann_files = data_cfg['ann_file']
        num_dset = len(ann_files)
    else:
        ann_files = [data_cfg['ann_file']]
        num_dset = 1

    if 'proposal_file' in data_cfg.keys():
        if isinstance(data_cfg['proposal_file'], (list, tuple)):
            proposal_files = data_cfg['proposal_file']
        else:
            proposal_files = [data_cfg['proposal_file']]
    else:
        proposal_files = [None] * num_dset
    assert len(proposal_files) == num_dset

    if isinstance(data_cfg['img_prefix'], (list, tuple)):
        img_prefixes = data_cfg['img_prefix']
    else:
        img_prefixes = [data_cfg['img_prefix']] * num_dset
    assert len(img_prefixes) == num_dset

    dsets = []
    for i in range(num_dset):
        data_info = copy.deepcopy(data_cfg)
        data_info['ann_file'] = ann_files[i]
        data_info['proposal_file'] = proposal_files[i]
        data_info['img_prefix'] = img_prefixes[i]
        dset = obj_from_dict(data_info, datasets)
        dsets.append(dset)
    if len(dsets) > 1:
        dset = ConcatDataset(dsets)
    else:
        dset = dsets[0]
    return dset


def flip_kp(kp, img_width):
    """
    Flip augmentation for keypoints
    :param kp:
    :param img_width: (int)
    :return:
    """
    flipped_parts = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
    kp = kp[flipped_parts]
    kp[:, 0] = img_width - kp[:, 0]
    return kp


def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flippedParts = [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13,
                    14, 18, 19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33,
                    34, 35, 30, 31, 32, 36, 37, 38, 42, 43, 44, 39, 40, 41,
                    45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58, 59, 54, 55,
                    56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68]
    pose = pose[flippedParts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose


def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa

def batch_rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa

def crop_image(img, keypoints_2d, bbox, center_x, center_y, bb_width, bb_height, patch_width, patch_height):
    trans, _ = gen_trans_from_patch_cv(center_x, center_y, bb_width, bb_height, patch_width, patch_height, 1.2, 0.0, inv=False)

    img = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)
    for n_jt in range(len(keypoints_2d)):
        keypoints_2d[n_jt, 0:2] = trans_point2d(keypoints_2d[n_jt, 0:2], trans)
    for i in range(2):
        bbox[i, :] = trans_point2d(bbox[i, :], trans)

    return img, keypoints_2d, bbox

def crop_depthmap(depthmap, center_x, center_y, bb_width, bb_height, patch_width, patch_height):
    trans, _ = gen_trans_from_patch_cv(center_x, center_y, bb_width, bb_height,
                                        patch_width, patch_height, 1.2, 0.0, inv=False)
    # depthmap = cv2.warpAffine(depthmap, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_CUBIC)
    depthmap = cv2.warpAffine(depthmap, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_AREA)
    # depthmap = cv2.warpAffine(depthmap, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

    return depthmap

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans, trans_inv



def get_colors():
    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
    }
    return colors


def get_smpl_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 0, 2 ],
            [ 0, 3 ],
            [ 1, 4 ],
            [ 2, 5 ],
            [ 3, 6 ],
            [ 4, 7 ],
            [ 5, 8 ],
            [ 6, 9 ],
            [ 7, 10],
            [ 8, 11],
            [ 9, 12],
            [ 9, 13],
            [ 9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
    )

def draw_skeleton(image, kp_2d, dataset='smpl', thickness=1):
    # kp_2d[:,2] = kp_2d[:,2] > 0.3
    kp_2d = np.array(kp_2d, dtype=int)

    rcolor = get_colors()['red'].tolist()
    pcolor = get_colors()['green'].tolist()
    lcolor = get_colors()['blue'].tolist()

    skeleton = get_smpl_skeleton()
    common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]
    for idx,pt in enumerate(kp_2d):
        # if pt[2] > 0: # if visible
        cv2.circle(image, (pt[0], pt[1]), 4, pcolor, -1)
        cv2.putText(image, f'{idx}', (pt[0]+1, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))

    for i, (j1,j2) in enumerate(skeleton):
        # if kp_2d[j1, 2] > 0 and kp_2d[j2, 2] > 0: # if visible
        if dataset == 'common':
            color = rcolor if common_lr[i] == 0 else lcolor
        else:
            color = lcolor if i % 2 == 0 else rcolor
        pt1, pt2 = (kp_2d[j1, 0], kp_2d[j1, 1]), (kp_2d[j2, 0], kp_2d[j2, 1])
        # cv2.line(image, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

    return image


def draw_point(image, kp_2d, color=(0, 255, 0), thickness=8):
    # kp_2d[:,2] = kp_2d[:,2] > 0.3
    kp_2d = np.array(kp_2d, dtype=int)
   
    h, w, _ = image.shape
    for idx, pt in enumerate(kp_2d):
        # if h > pt[0] or w > pt[1]:
        #     continue
        cv2.circle(image, (pt[0], pt[1]), thickness, color, -1)
        cv2.putText(image, f'{idx}', (pt[0]+1, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
    
    return image



def get_vertexs_img(pred_verts, update_global_translation_F1000, kpts2d_gt, image, render_smplx):

    min_pt = np.min(kpts2d_gt, axis=0, keepdims=False)
    max_pt = np.max(kpts2d_gt, axis=0, keepdims=False)
    length = (max_pt - min_pt)  
    center = (max_pt + min_pt) / 2.
    bbox = np.array([int(center[0] - length[0]/2.), int(center[1] - length[1]/2.), int(center[0] + length[0]/2.), int(center[1] + length[1]/2.)])
    bbox = np.expand_dims(bbox, axis=0)
    
    H, W, _ = image.shape
    img = image[:, :, ::-1].copy()
    img = img.transpose([2, 0, 1])
    img = to_tensor(img)
    img = img / 255.0

    front_view_color = np.array([255, 244, 229]) / 255.0
    side_view_color = np.array([212, 235, 255]) / 255.0
    top_view_color = np.array([255, 245, 182]) / 255.0

    fv_bg_color = torch.ones_like(img)
    fv_bg_color[0] = fv_bg_color[0] * front_view_color[0]
    fv_bg_color[1] = fv_bg_color[1] * front_view_color[1]
    fv_bg_color[2] = fv_bg_color[2] * front_view_color[2]
    sv_bg_color = torch.ones_like(img)
    sv_bg_color[0] = sv_bg_color[0] * side_view_color[0]
    sv_bg_color[1] = sv_bg_color[1] * side_view_color[1]
    sv_bg_color[2] = sv_bg_color[2] * side_view_color[2]
    bv_bg_color = torch.ones_like(img)
    bv_bg_color[0] = bv_bg_color[0] * top_view_color[0]
    bv_bg_color[1] = bv_bg_color[1] * top_view_color[1]
    bv_bg_color[2] = bv_bg_color[2] * top_view_color[2]

    pred_verts = torch.from_numpy(pred_verts)
    update_global_translation_F1000 = torch.from_numpy(update_global_translation_F1000)
    # try:
    fv_rendered = render_smplx([img], [pred_verts], translation=[update_global_translation_F1000])[0]
    # fv_rendered = render_smplx([fv_bg_color], [pred_verts], translation=[update_global_translation_F1000])[0]
    # sv_rendered = renderer_sv([sv_bg_color], pred_verts, update_global_translation_F1000, bbox, 1000, render_smplx)
    # bv_rendered = renderer_bv([bv_bg_color], pred_verts, update_global_translation_F1000, bbox, 1000, render_smplx)
    # except Exception as e:
    #     print("Exception: ", e)
    #     # return None

    total_img = np.zeros((H, 2*W, 3))
    # total_img = np.zeros((H, 4*W, 3))
    total_img[:H, :W] += img.cpu().numpy().transpose([1, 2, 0])
    total_img[:H, W:2*W] += fv_rendered.transpose([1, 2, 0])
    # total_img[:H, 2*W:3*W] += bv_rendered.transpose([1, 2, 0])
    # total_img[:H, 3*W:4*W] += sv_rendered.transpose([1, 2, 0])
    total_img = (total_img * 255).astype(np.uint8)

    return total_img
        
def project_point_np(joint, RT, KKK):
    P = np.dot(KKK, RT)
    joints_2d = np.dot(P, joint)
    joints_2d = joints_2d[0:2] / joints_2d[2]

    return joints_2d