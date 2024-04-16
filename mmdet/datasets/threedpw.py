# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
# import sys

import os
import torch
import random
import logging
import numpy as np
import os.path as osp
# import joblib
import copy

from torch.utils.data import Dataset, DataLoader


import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC

# add dot for below!!!
# from .threedpw_utils import convert_kps
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor, DepthMapTransform)
from .utils import to_tensor, random_scale, flip_kp, flip_pose, rot_aa
from .extra_aug import ExtraAugmentation

from .occlude import load_occluders, occlude_with_objects


# import os.path as osp
# import tensorboardX
import math
import json
import pickle
import matplotlib.pyplot as plt
from mmdet.models.utils.smpl_utils import batch_rodrigues
from mmdet.models.utils.smpl.smpl import SMPLR
import random
import cv2
from copy import deepcopy
import scipy
import scipy.misc
# import seaborn as sns
import torch
# import png

import skimage.transform

from .utils import project_point_np, draw_point
from mmdet.models.utils.smpl.renderer import Renderer

from mmdet.models.utils import read_pfm, write_pfm, resize_depth, write_depth, resize_depth_np

denormalize = lambda x: x * np.array([0.229, 0.224, 0.225])[None, None, :] + np.array([0.485, 0.456, 0.406])[None, None, :]

def scale2depth(scale, f, c, x, o_depth):
    assert scale > 0
    scale = 1/scale
    offset = f * x / o_depth + c
    #print(scale, x, f, c, offset)
    z =1 /((offset - scale * c) / (scale * x * f))
    return z

def scale2depth2(scale, f, c, x, z):
    offset = f * x / z
    offset *= scale # should be less
    return x*f / offset


from .builder import DATASETS

@DATASETS.register_module()
class ThreeDPW(Dataset):

    # CLASSES = ('Human',)
    CLASSES = ('Background', 'Human',)

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 enable_mask=False,
                 #set='train', # old arg
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 with_shape=True,
                 with_kpts3d=True,
                 with_kpts2d=True,
                 with_pose=True,
                 with_trans=True,
                 max_samples=-1,  # Commonly used in validating
                 noise_factor=0.4,
                 square_bbox=True,
                 rot_factor=0,
                 sample_weight=1,
                 with_dp=False,
                 mosh_path=None,
                 ignore_3d=False,
                 ignore_smpl=False,
                 with_nr=False,
                 sample_by_persons=False,
                 use_poly=False,
                 global_augment=None,
                 visible_only=False,
                 occlude=None,
                 occluders=None,
                 use_gender=False,
                 regress_trans=False,
                 use_clip_bbox=True,
                 color_jitter=None,
                 use_padShape_as_imgShape=True,
                 depthmap_scale_by_augFactor=False,
                 dataset_name=None,
                 **kwargs,
                 ):
        if test_mode:
            #set = 'val'
            set = 'test'
        else:
            set = 'train'

        self.use_gender = use_gender
        self.regress_trans = regress_trans
        if self.regress_trans:
            self.smplr = SMPLR('data/smpl', use_gender=use_gender)

        self.set = set
        # prefix of images path
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        # self.size_divisor = None #size_divisor
        self.size_divisor = size_divisor
        self.noise_factor = noise_factor
        self.with_nr = with_nr
        self.enable_mask = enable_mask
        self.seg_prefix = seg_prefix
        self.rot_factor = rot_factor
        self.square_bbox = square_bbox
        self.with_dp = False

        self.use_padShape_as_imgShape = use_padShape_as_imgShape
        self.depthmap_scale_by_augFactor = depthmap_scale_by_augFactor
        print(">>>>threedpw depthmap_scale_by_augFactor:", depthmap_scale_by_augFactor)

        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1

        # load annotations (and proposals)
        #self.img_infos = self.load_annotations(ann_file)
        # self.max_samples = max_samples

        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None

        self.color_jitter = color_jitter
        self.global_augment = global_augment
        if global_augment:
            print('enable global augment!!')

        self.visible_only = visible_only
        if visible_only:
            self.vis_db = self.load_vis_db()
        # Select a subset for quick validation
        # if self.max_samples > 0:
        #     self.img_infos = random.sample(self.img_infos, max_samples)
        #     # self.img_infos = self.img_infos[:max_samples]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale, list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # transforms
        self.img_transform = ImageTransform(size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.depthmap_transform = DepthMapTransform(size_divisor=self.size_divisor)
        self.bbox_transform = BboxTransform(use_clip_bbox=use_clip_bbox)
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

        self.sample_weight = sample_weight

        # occlude
        self.occlude = occlude
        self.occluders = occluders
        if occlude is not None and occluders is None:
            # /data/Mono3DPerson/VOCdevkit/VOC2012
            self.occluders = load_occluders(occlude['path'])
            print('Loaded {} suitable occluders from VOC2012'.format(len(self.occluders)))
        elif occlude is not None and occluders is not None:
            print('3DPW Received {} occluders!'.format(len(self.occluders)))
        else:
            self.occluders = None
        # if occlude is not None:
        #     # /data/Mono3DPerson/VOCdevkit/VOC2012
        #     self.occluders = load_occluders(occlude['path'])
        #     print('Loaded {} suitable occluders from VOC2012'.format(len(self.occluders)))

        # load data
        self.dataset_name = dataset_name

        # self.img_infos = self.load_db()
        # flatten sequence
        # frame_datas = []
        # for frame_data in self.img_infos:
        #     print("len(frame_data): ", len(frame_data))
        #     frame_datas += frame_data
        # self.img_infos = frame_datas

        self.img_infos = self.load_annotations(ann_file)
        print("len(db): ", len(self.img_infos))
        print(self.img_infos[0].keys())
        # dict_keys(['filename', 'width', 'height', 'bboxes', 'kpts2d', 'kpts2d_smpl_proj', 
        # 'kpts3d', 'betas', 'pose', 'has_smpl', 'camMat_original', 'camMat', 
        # 'global_translations', 'vertices'])

        if sample_by_persons:
            persons_cnt = np.zeros(len(self.img_infos))
            for i in range(len(self.img_infos)):
                # persons_cnt[i] = self.img_infos[i]['joints2D'].shape[0]
                # persons_cnt[i] = self.img_infos[i]['kpts2d'].shape[0]
                persons_cnt[i] = self.img_infos[i]['kpts2d_smpl_proj'].shape[0]
            self.density = sample_weight * persons_cnt / persons_cnt.sum()
        else:
            self.density = sample_weight * np.ones(len(self.img_infos)) / len(self.img_infos)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def __len__(self):
        return len(self.img_infos)

    def _rand_another(self, idx):
        return np.random.randint(len(self.img_infos))

    def __getitem__(self, idx):
        while True:
            data = self.get_single_item(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data


    # def load_db(self):
    #     db_file = osp.join(VIBE_DB_DIR, f'{self.dataset_name}_dense_ss_{self.set}_db.pt')

    #     if osp.isfile(db_file):
    #         db = joblib.load(db_file)
    #     else:
    #         raise ValueError(f'{db_file} do not exists')

    #     print(f'Loaded {self.dataset_name} dataset from {db_file}')
    #     return db

    
    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as f:
            raw_infos = pickle.load(f)
            print("raw_infos:", len(raw_infos), type(raw_infos))

        # print(raw_infos[5000])
        print(f'Loaded {self.dataset_name} dataset from {ann_file}')
        return raw_infos


    # def load_vis_db(self):
    #     db = {}
    #     path = os.path.join(VIBE_DB_DIR, 'vis_db', self.set)
    #     files = os.listdir(path)
    #     for f in files:
    #         if not f.endswith('.npy'):
    #             continue
    #         db[f[:-4]] = np.load(os.path.join(path, f)) # p, num frame, 6890. type bool
    #     print(f'Loaded {self.set} visibility data')
    #     return db


    def load_depthmap_per_person(self, img_path, data):
        body_depthmap_list = []
        person_ids = data['person_ids']
        for pid in person_ids:
            body_depthmap_path = img_path.replace('/imageFiles/', '/bodymesh_depthmap_rescale_832x512_per_person/')
            body_depthmap_path = body_depthmap_path[:-4] + f'_pid{pid}.pfm'
            body_depthmap, _ = read_pfm(body_depthmap_path)
            body_depthmap_list.append(body_depthmap[None, :, :])

        body_depthmap_arr = np.concatenate(body_depthmap_list, axis=0)
        return body_depthmap_arr


    def get_single_item(self, index):
        is_train = self.set == 'train'

        data = copy.deepcopy(self.img_infos[index]) # dataset -> dict
        num_people = data['kpts2d_smpl_proj_originalFocal'].shape[0]
        filename = data['filename']
        img_path = os.path.join(self.img_prefix, filename)
        img = mmcv.imread(img_path)
        ori_shape = (img.shape[0], img.shape[1], 3)

        body_depthmap =  self.load_depthmap_per_person(img_path, data)
        # print(body_depthmap.shape)

        # bboxes = data['bbox'].astype(np.float32) # cx cy w h
        gt_bboxes = data['bboxes'].astype(np.float32) # cx cy cx cy
        # following lines for cx cy w h
        # cx, cy, w, h = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
        # w = w * 1.1
        # h = h * 1.1
        # gt_bboxes = np.zeros_like(bboxes)
        # gt_bboxes[:,0] = cx - w/2
        # gt_bboxes[:,1] = cy - h/2
        # gt_bboxes[:,2] = cx + w/2
        # gt_bboxes[:,3] = cy + h/2 

        # resacel bbox to 1.2x
        center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.
        length = np.abs((gt_bboxes[:, :2] - gt_bboxes[:, 2:]))
        # length = length * 1.2
        gt_bboxes = np.zeros_like(gt_bboxes)
        gt_bboxes[:,0] = center[:,0] - length[:,0] / 2.
        gt_bboxes[:,1] = center[:,1] - length[:,1] / 2.
        gt_bboxes[:,2] = center[:,0] + length[:,0] / 2.
        gt_bboxes[:,3] = center[:,1] + length[:,1] / 2.
        data['bboxes'] = gt_bboxes

        occluder = None
        if is_train and self.occluders is not None and np.random.rand() < self.occlude['ratio']:
            self.occluders = np.array(self.occluders, dtype=object)
            occluder = np.random.choice(self.occluders, num_people)
        
        if is_train:
            pn = np.random.uniform(1 - self.noise_factor, 1 + self.noise_factor, 3)

            # Color jittering:
            if self.color_jitter is not None:
                color_ratio = self.color_jitter['ratio']
                if np.random.rand() < color_ratio:
                    img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, img[:, :, 0] * pn[0]))
                    img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, img[:, :, 1] * pn[1]))
                    img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, img[:, :, 2] * pn[2]))

            if occluder is not None:
                for o_idx in range(num_people):
                    occluder[o_idx][:, :, 0] = np.minimum(255.0, np.maximum(0.0, occluder[o_idx][:, :, 0] * pn[2]))
                    occluder[o_idx][:, :, 1] = np.minimum(255.0, np.maximum(0.0, occluder[o_idx][:, :, 1] * pn[1]))
                    occluder[o_idx][:, :, 2] = np.minimum(255.0, np.maximum(0.0, occluder[o_idx][:, :, 2] * pn[0]))
                    occluder[o_idx][:, :, :3] -= self.img_norm_cfg['mean'][::-1]
                    occluder[o_idx][:, :, :3] /= self.img_norm_cfg['std'][::-1]
                    #occluder[o_idx][:, :, :3] /= 255.

        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes, gt_labels)
        
        # apply transforms
        flip = True if (np.random.rand() < self.flip_ratio and is_train) else False
        # flip = True # for debug...
        
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)

        # global augment image scale
        augment_scale_factor = 1
        enable_global_augment = False
        if is_train and self.global_augment is not None:
            if np.random.rand() < self.global_augment['ratio']:
                enable_global_augment = True
                augment_scale_factor = np.random.uniform(self.global_augment['scale'][0], self.global_augment['scale'][1])

        scaled_img_scale = (img_scale[0] * augment_scale_factor, img_scale[1] * augment_scale_factor)
        img, img_shape, pad_shape, scale_factor = self.img_transform(img, scaled_img_scale, flip, keep_ratio=self.resize_keep_ratio)
        raw_img_shape = copy.deepcopy(img_shape)

        body_depthmap = body_depthmap.transpose((1, 2, 0))
        body_depthmap, _, _, depthmap_scale_factor = self.depthmap_transform(body_depthmap, scaled_img_scale, flip, keep_ratio=self.resize_keep_ratio)
        
        # scale depthmap value
        if self.depthmap_scale_by_augFactor == True:
            # body_depthmap = body_depthmap / depthmap_scale_factor
            body_depthmap = body_depthmap / scale_factor

        if len(body_depthmap.shape) > 2:
            body_depthmap = body_depthmap.transpose((2, 0, 1))
        else:
            body_depthmap = body_depthmap[None, :, :]

        # print(">>> 3dpw: ", body_depthmap.shape, img.shape)

        # overwrite img_shape
        if enable_global_augment:   #augment_scale_factor != 1:
            h, w = ori_shape[0], ori_shape[1]
            scale = min(img_scale[1] / h, img_scale[0] / w)
            img_shape = (int(h * float(scale) + 0.5), int(w * float(scale) + 0.5), img_shape[-1])
            pad_shape = img_shape

        # center crop large image
        diff = (0, 0)
        if enable_global_augment and augment_scale_factor > 1:
            diff = (img.shape[1] - img_shape[0], img.shape[2] - img_shape[1]) 
            img = img[:, diff[0]//2:diff[0]//2 + img_shape[0], diff[1]//2:diff[1]//2 + img_shape[1]]
            body_depthmap = body_depthmap[:, diff[0]//2:diff[0]//2 + img_shape[0], diff[1]//2:diff[1]//2 + img_shape[1]]

        # Force padding for the issue of multi-GPU training
        padded_img = np.zeros((img.shape[0], img_scale[1], img_scale[0]), dtype=img.dtype)
        padded_depthmap = np.zeros((body_depthmap.shape[0], img_scale[1], img_scale[0]), dtype=body_depthmap.dtype)

        # new ver
        # global augment xy
        if enable_global_augment:#is_train and self.global_augment is not None and augment_scale_factor != 1:
            start_pos_x = np.random.randint(0, img_scale[1] - img.shape[-2]) if img_scale[1] > img.shape[-2] else 0
            start_pos_y = np.random.randint(0, img_scale[0] - img.shape[-1]) if img_scale[0] > img.shape[-1] else 0
        else:
            # center started
            start_pos_x = img_scale[1] // 2 - img.shape[-2] // 2
            start_pos_y = img_scale[0] // 2 - img.shape[-1] // 2

            # if is_train == True:
            #     # center started
            #     start_pos_x = img_scale[1] // 2 - img.shape[-2] // 2
            #     start_pos_y = img_scale[0] // 2 - img.shape[-1] // 2
            # else:
            #     # corner started for eval
            #     start_pos_x = 0
            #     start_pos_y = 0

        padded_img[:, start_pos_x:start_pos_x+img.shape[-2], start_pos_y:start_pos_y+img.shape[-1]] = img
        padded_depthmap[:, start_pos_x:start_pos_x+body_depthmap.shape[-2], start_pos_y:start_pos_y+body_depthmap.shape[-1]] = body_depthmap
        img = padded_img
        body_depthmap = padded_depthmap

        gt_camera_trans = data['global_translations'] # p, 3
        gt_shapes = data['betas']
        # gt_vertices = data['vertices']
        gt_poses = data['pose'].reshape(num_people, -1, 3) #  # pose flip cane affect the flip of kpts3d and vertices
        if flip:
            for i, ps in enumerate(gt_poses):
                gt_poses[i] = flip_pose(ps.reshape(-1)).reshape(-1, 3)

        gt_vertices = None
        if not self.regress_trans:
            gt_kpts3d = data['kpts3d'] # n, 24, 4 
            # verts_local = data['vertices'] # flip?
            # flip vertices?
            gt_vertices = np.zeros((gt_poses.shape[0], 6890, 4)).astype(np.float32)
            if flip:
                for i, kp in enumerate(gt_kpts3d):
                    gt_kpts3d[i] = flip_kp(kp, 0)  # Not the image width as the pose is centered by hip.
        else:
            genders = data['genders'] 
            kpts3d_list = []
            vertices_list = []
            for i in range(genders.shape[0]):
                gender = genders[i]
                pose_hand_theta = np.zeros([1, 2, 3]).astype(np.float32)
                gt_poses[i:i+1, -2:, :] = pose_hand_theta
                outputs = self.smplr(betas=gt_shapes[i:i+1], body_pose=gt_poses[i:i+1, 1:, :], global_orient=gt_poses[i:i+1, :1, :], pose2rot=True, gender=gender)
                vertices = outputs.vertices.detach().numpy().astype(np.float32)
                joints = outputs.joints.detach().numpy().astype(np.float32) # n, 24, 3
                j3d_score = np.ones((joints.shape[0], joints.shape[1], 4)).astype(np.float32)
                j3d_score[:, :, :3] = joints
                kpts3d_list.append(j3d_score[0])
                vertices_list.append(vertices[0])

            gt_kpts3d = np.asarray(kpts3d_list, dtype=np.float32)
            gt_vertices = np.asarray(vertices_list, dtype=np.float32)

        # gt_kpts3d和gt_camera_trans都是focal_len 1000下的，所以也要做scale
        gt_camera_trans = gt_camera_trans + gt_kpts3d[:, 14, :3]
        gt_camera_trans[:, -1] /= scale_factor
        gt_camera_trans = gt_camera_trans - gt_kpts3d[:, 14, :3]

        # body_depthmap_tmp = torch.from_numpy(body_depthmap)
        # body_depthmap_tmp_mean = body_depthmap_tmp[torch.where(body_depthmap_tmp > 1.0)].mean() 
        # print("gt_camera_trans[:, -1]: ", gt_camera_trans[:, -1])
        # print("body_depthmap_tmp_mean: ", body_depthmap_tmp_mean)
        # print("-" * 50)
        # gt_camera_trans[:, -1] = body_depthmap_tmp_mean
        # print("threedpw gt_camera_trans[:, -1]: ", gt_camera_trans[:, -1])
        # print("----------------")
        
        if flip:
            gt_camera_trans[:, 0] *= -1. 

        pixel_offset_y = img_scale[0] // 2
        if augment_scale_factor < 1:
            pixel_offset_y -= (start_pos_y + raw_img_shape[1] // 2)
        else:
            pixel_offset_y -= (start_pos_y + img_shape[1] // 2)  # 画图示可知
            # pixel_offset_y -= (start_pos_y - diff[1]//2 + raw_img_shape[1] // 2) 
        if pixel_offset_y != 0:
            camera_offset_y = (gt_camera_trans[:,-1] * pixel_offset_y) / 1000.
            gt_camera_trans[:,0] -= camera_offset_y

        pixel_offset_x = img_scale[1] // 2
        if augment_scale_factor < 1:
            pixel_offset_x -= (start_pos_x + raw_img_shape[0] // 2)
        else:
            pixel_offset_x -=  (start_pos_x + img_shape[0] // 2)
            # pixel_offset_x -=  (start_pos_x - diff[0]//2 + raw_img_shape[0] // 2)
        if pixel_offset_x != 0:
            camera_offset_x = (gt_camera_trans[:,-1] * pixel_offset_x) / 1000.
            gt_camera_trans[:,1] -= camera_offset_x

        # # depth out of range
        # if np.min(gt_camera_trans[:,-1]) < 1.1 or np.max(gt_camera_trans[:,-1]) > 19.9:
        if np.min(gt_camera_trans[:, -1]) < 1.0:
            return None

        # p, 4: x1 y1 x2 y2
        gt_bboxes = self.bbox_transform(gt_bboxes, raw_img_shape, scale_factor,  flip, 
            translation = (start_pos_y - diff[1]//2, start_pos_x - diff[0]//2))

        # 2d joints
        # gt_kpts2d = data['kpts2d'] # p, 24, 3
        # gt_kpts2d = data['kpts2d_smpl_proj'] # p, 24, 3
        gt_kpts2d = data['kpts2d_smpl_proj_originalFocal'] # p, 24, 3 # project from original 3d_smpl and original camera_mat
        s_kpts2d = np.zeros_like(gt_kpts2d)
        s_kpts2d[..., -1] = gt_kpts2d[..., -1]
        s_kpts2d[..., :-1] = gt_kpts2d[..., :-1] * scale_factor 
        gt_kpts2d = s_kpts2d
        if flip:
            for i, kp in enumerate(gt_kpts2d):
                gt_kpts2d[i] = flip_kp(kp, raw_img_shape[1])
                #gt_kpts2d[i] = flip_kp(kp, img_shape[1])  # img is (C, H, W)
                # NOTE: I use the img_shape to avoid the influence of padding.
        
        # global augment 2d joints
        gt_kpts2d[..., 0] = gt_kpts2d[..., 0] + start_pos_y - diff[1]//2
        gt_kpts2d[..., 1] = gt_kpts2d[..., 1] + start_pos_x - diff[0]//2
        
        if self.use_padShape_as_imgShape:
            img_shape=padded_img.transpose([1, 2, 0]).shape # img_shape should be consistent with the input img of model
            pad_shape=padded_img.transpose([1, 2, 0]).shape # pad_shape should be consistent with the input img of model
            
        img_metas = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip,
            idx=index,
            file_name=img_path
        )

        # # visibility
        # if not self.visible_only or num_people != self.vis_db[seq_name].shape[0]:
        #     visibility = np.ones((num_people,6890)).astype(np.bool)
        # else:
        #     visibility = self.vis_db[seq_name][:,int(frame_name[-5:])]
        #     # print(num_people, seq_name, frame_name)

        # occlude
        if occluder is not None:
            areas = np.abs((gt_bboxes[:,2] - gt_bboxes[:,0]) * (gt_bboxes[:,3] - gt_bboxes[:,1]))
            img = occlude_with_objects(img, occluder, areas, gt_kpts2d, cross_prob=self.occlude['cross_prob'])
        
        has_trans = np.ones((num_people,))
        target = dict(
            gt_labels=DC(to_tensor(np.ones((num_people,)).astype(np.int_))), # 0 means background, 1 means Human gt_labels (list[Tensor]): class indices corresponding to each box
            # gt_labels=DC(to_tensor(np.zeros((num_people,)).astype(np.int_))), 
            has_smpl=DC(to_tensor(np.ones((num_people,)).astype(np.int_))), # all human
            gt_poses=DC(to_tensor(gt_poses)),
            gt_kpts2d=DC(to_tensor(gt_kpts2d)),
            gt_kpts3d=DC(to_tensor(gt_kpts3d)),
            gt_shapes=DC(to_tensor(gt_shapes)),
            img=DC(to_tensor(img), stack=True),
            # img_meta=DC(img_meta, cpu_only=True),
            img_metas=DC(img_metas, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes.astype(np.float32))),
            # gt_trans=DC(to_tensor(data['trans'].astype(np.float32))),
            gt_trans=DC(to_tensor(np.zeros((num_people, 3)).astype(np.float32))),
            gt_camera_trans=DC(to_tensor(gt_camera_trans.astype(np.float32))),
            has_trans=DC(to_tensor(has_trans.astype(np.int_))),
            gt_scale = DC(to_tensor(np.ones((num_people, 3)).astype(np.float32))),
            gt_vertices = DC(to_tensor(gt_vertices.astype(np.float32))),
            #visibility = DC(to_tensor(visibility))
        )
        
        
        body_depthmap = to_tensor(body_depthmap)
        body_depthmap[torch.isnan(body_depthmap)] = 0.0

        body_depthmap[body_depthmap <= 0.0] = float('inf')
        body_depthmap_onechnl = torch.min(body_depthmap, dim=0, keepdim=True)[0]
        body_depthmap_onechnl[torch.isinf(body_depthmap_onechnl)] = 0.0
        body_depthmap[torch.isinf(body_depthmap)] = 0.0
        target['gt_depthmap'] = DC(body_depthmap_onechnl) # !!! body_depthmap_onechnl
        target['has_depthmap'] = DC(to_tensor(np.ones(1).astype(np.int64)))

        # # debug visual gt_depth_mask
        # print(body_depthmap_onechnl.shape)
        # gt_mask_np = body_depthmap_onechnl[0].cpu().numpy()
        # depth_min = gt_mask_np.min()
        # depth_max = gt_mask_np.max()
        # if depth_max - depth_min > np.finfo("float").eps:
        #     depth_vis = 255 * (gt_mask_np - depth_min) / (depth_max - depth_min)
        # else:
        #     depth_vis = np.zeros(gt_mask_np.shape, dtype=gt_mask_np.dtype)
        # output_path = "./debug/gt_depthmap.png"
        # img = img.transpose([1, 2, 0])
        # img = cv2.cvtColor((denormalize(img) * 255).astype(np.uint8).copy(), cv2.COLOR_BGR2RGB)
        # depth_vis = depth_vis.astype("uint8")[:, :, None]
        # depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)
        # img = cv2.addWeighted(img, 0.5, depth_vis, 0.5, 0)
        # cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # cv2.imwrite(output_path[:-4] + "_depth.png", depth_vis, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # print("gt_depthmap_debug.shape: ", gt_mask_np.shape)
        # print(output_path)
        # print(output_path[:-4] + "_depth.png")
        # print("....")

        # body_depthmap_onechnl_vis = body_depthmap_onechnl.cpu().numpy()
        # body_depthmap_onechnl_vis = body_depthmap_onechnl_vis.transpose((1, 2, 0))
        # body_depthmap_onechnl_vis = (body_depthmap_onechnl_vis > 0) * 255
        # print(body_depthmap_onechnl_vis.shape)
        # output_path = f"./debug/body_depthmap_onechnl_vis_{i}.jpg"
        # cv2.imwrite(output_path, body_depthmap_onechnl_vis)
        # print(output_path)
        # print(img_path, flip)
        # print("...")

        #### mask from depthmap
        gt_masks = np.zeros_like(body_depthmap).astype(np.uint8)
        gt_masks[body_depthmap > 0] = 1
        target['gt_masks'] = DC(gt_masks.astype(np.uint8), cpu_only=True)
        target['has_masks'] = DC(to_tensor(np.ones((num_people,)).astype(np.int64)))

        # for i in range(gt_masks.shape[0]):
        #     gt_masks_vis = gt_masks[i][:, :, None] * 255
        #     output_path = f"./debug/gt_masks_{i}.jpg"
        #     cv2.imwrite(output_path, gt_masks_vis)
        #     print(output_path)
        #     print(img_path, flip)
        #     print("...")
        
        # target['gt_depth'] = DC(to_tensor(np.zeros((1, 1, img.shape[-2], img.shape[-1])).astype(np.float32)))     
        # target['has_depth'] = DC(to_tensor(np.zeros(num_people, ).astype(np.int64)))
        # target['has_depth'] = DC(to_tensor(np.zeros(1).astype(np.int64)))
        # target['has_bbox'] = DC(to_tensor(np.ones(1).astype(np.int64)))
        # target['gt_bboxes_ignore'] = DC(to_tensor(np.zeros(1).astype(np.int64)))
        # target['gt_bboxes_ignore']=DC(to_tensor(np.zeros((num_people,)).astype(np.int_))), # all human
        # target['gt_bboxes_ignore']=DC(to_tensor(np.ones(num_people).astype(np.int_))), # all human


        # if self.visible_only:
        #     target['visibility'] = DC(to_tensor(self.vis_db[seq_name][:,int(frame_name[-5:])])) # p 6890
        if self.with_nr:
            target['scene'] = DC(to_tensor(np.zeros((img.shape[-2], img.shape[-1]), dtype=np.uint8)))

        # #############################
        # ### 可视化 dataloader data ###
        # #############################
        # debug_filename = os.path.basename(data['filename'])
        # print(debug_filename, debug_filename[0])
        # img_viz = self.prepare_dump(target)
        
        # _, H, W = img.shape
        # camMat = np.array([[1000, 0, W/2],
        #         [0, 1000, H/2],
        #         [0, 0, 1]]).astype(np.float32)
        # RT = np.concatenate((np.diag([1., 1., 1.]), np.zeros((3, 1))), axis=1)
        
        # gt_kpts3d_global = gt_kpts3d[:, :, :3] + gt_camera_trans[:, None, :]
        # kpts3d_F1000_proj = np.zeros((gt_kpts3d.shape[0], gt_kpts3d.shape[1], 2))
        # for i in range(gt_kpts3d.shape[0]):
        #     for j in range(gt_kpts3d.shape[1]):
        #         kpts3d_F1000_proj[i, j, :] = project_point_np(np.concatenate([gt_kpts3d_global[i, j, :], np.array([1])]), RT, camMat)
        
        # for i in range(gt_kpts2d.shape[0]):
        #     img_viz = draw_point(img_viz, kpts3d_F1000_proj[i], color=(0, 255, 0))
        #     img_viz = draw_point(img_viz, gt_kpts2d[i], color=(0, 0, 255))

        # # vis bbox, 把bbox可视化, 确认bboxes需要flip, 其他数据集也是
        # for i in range(gt_bboxes.shape[0]):
        #     bbox = gt_bboxes[i]
        #     img_viz = cv2.rectangle(img_viz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

        # vis_path = f"./debug/{debug_filename[:-4]}.smpl_render_pixel_offset_x:{pixel_offset_x}_pixel_offset_y:{pixel_offset_y}_aug_scale:{augment_scale_factor}_scale:{round(scale_factor, 2)}.jpg"
        # cv2.imwrite(vis_path, img_viz[:, :, :])
        # print(img_path)
        # print(vis_path)
        # print("smpl img_viz", img_viz.shape)

        # print("3dpw: ", sorted(target.keys()), len(sorted(target.keys())))
        # print("+" * 50)

        return target

    def prepare_dump(self, data):
        from mmdet.models.utils.smpl.smpl import SMPL
        FOCAL_LENGTH = 1000
        render = Renderer(focal_length=FOCAL_LENGTH)

        # baseline
        # pred_trans = pred_results['pred_translation'].cpu()
        # pred_camera = pred_results['pred_camera'].cpu()
        # pred_betas = pred_results['pred_betas'].cpu()
        # pred_rotmat = pred_results['pred_rotmat'].cpu()
        # pred_verts = pred_results['pred_vertices'].cpu()

        gt_bboxes = data['gt_bboxes'].data
        gt_trans = data['gt_trans'].data                     # 000
        gt_camera_trans = data['gt_camera_trans'].data       # 3.88, 
        gt_poses = data['gt_poses'].data
        gt_shapes = data['gt_shapes'].data
        img = data['img'].data

        print(gt_camera_trans.shape, type(gt_camera_trans))
        print(gt_poses.shape, type(gt_poses))
        print(gt_shapes.shape, type(gt_shapes))
        print(img.shape, type(img))

        # img = cv2.cvtColor((denormalize(img.transpose([1, 2, 0])) * 255).astype(np.uint8).copy(),
        #                           cv2.COLOR_BGR2RGB)
     
        smpl = SMPL('data/smpl')
        # 这里需要改为smplr加入性别
        # output = smplr(betas=gt_shapes, body_pose=gt_poses[:, 3:], global_orient=gt_poses[:, :3], pose2rot=True)
        # output = smpl(betas=gt_shapes, body_pose=gt_poses[:, 3:], global_orient=gt_poses[:, :3], pose2rot=True)
        # verts = output.vertices
        verts = data['gt_vertices'].data
        
        print(verts.shape, gt_camera_trans.shape)

        # fv_rendered = render([img], [verts], translation=[gt_camera_trans])[0]
        # print(fv_rendered.shape, fv_rendered.shape)
        # total_img = fv_rendered.transpose([1, 2, 0])
        # total_img = (total_img * 255).astype(np.uint8)

        # bboxes = data['bboxes']
        # img_bbox = img.copy()
        # for bbox in bboxes:
        #     print('bbox:', bbox)
        #     img_bbox = cv2.rectangle(img_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (1., 0, 0), 2)
        # img_th = torch.tensor(img_bbox.transpose([2, 0, 1]))
        
        _, H, W = img.shape
        print(H, W)
        try:
            fv_rendered = render([img.clone()], [verts], translation=[gt_camera_trans])[0]
            bv_rendered = self.renderer_bv(img, verts, gt_camera_trans, gt_bboxes[0], FOCAL_LENGTH, render)
        except Exception as e:
            print(e)
            return None

        img = np.float32(img.cpu().numpy())
        print(img.shape, fv_rendered.shape, bv_rendered.shape)
        print(type(img), type(fv_rendered), type(bv_rendered))
        total_img = np.zeros((3 * H, W, 3))
        total_img[:H] += img.transpose([1, 2, 0])
        total_img[H:2 * H] += fv_rendered.transpose([1, 2, 0])
        total_img[2 * H:] += bv_rendered.transpose([1, 2, 0])
        # total_img = (total_img * 255).astype(np.uint8)

        total_img = cv2.cvtColor((denormalize(total_img) * 255).astype(np.uint8).copy(),
                            cv2.COLOR_BGR2RGB)

        return total_img


    def renderer_bv(self, img_t, verts_t, trans_t, bboxes_t, focal_length, render):
        # rotation
        R_bv = torch.zeros(3, 3)
        R_bv[0, 0] = R_bv[2, 1] = 1
        R_bv[1, 2] = -1

        # # filter our small detections
        # bbox_area = (bboxes_t[:, 2] - bboxes_t[:, 0]) * (bboxes_t[:, 3] - bboxes_t[:, 1])
        # area_mask = torch.tensor(bbox_area > bbox_area.max() * 0.05)
        # verts_t, trans_t = verts_t[area_mask], trans_t[area_mask]

        # camera space
        verts_t = verts_t + trans_t.unsqueeze(1)

        # rotate to up view
        verts_tr = torch.einsum('bij,kj->bik', verts_t, R_bv)
        verts_tfar = verts_tr  # verts_tr + trans_t.unsqueeze(1)
        
        # min point, max point position
        p_min, p_max = verts_tfar.view(-1, 3).min(0)[0], verts_tfar.view(-1, 3).max(0)[0]
        
        # center of min/max
        p_center = 0.5 * (p_min + p_max)
        # trans_tr = torch.einsum('bj,kj->bk', trans_t, R_bv)

        # normalize points to center
        verts_center = (verts_tfar.view(-1, 3) - p_center).view(verts_t.shape[0], -1, 3)

        # normalized min/max
        dis_min, dis_max = (verts_tfar.view(-1, 3) - p_center).min(0)[0], (
                verts_tfar.view(-1, 3) - p_center).max(0)[0]
        
        h, w = img_t.shape[-2:]
        # h, w = min(h, w), min(h, w)
        ratio_max = abs(0.9 - 0.5)
        z_x = dis_max[0] * focal_length / (ratio_max * w) + torch.abs(dis_min[2])
        z_y = dis_max[1] * focal_length / (ratio_max * h) + torch.abs(dis_min[2])
        z_x_0 = (-dis_min[0]) * focal_length / (ratio_max * w) + torch.abs(
            dis_min[2])
        z_y_0 = (-dis_min[1]) * focal_length / (ratio_max * h) + torch.abs(
            dis_min[2])
        z = max(z_x, z_y, z_x_0, z_y_0)
        
        # z controls the view
        verts_right = verts_tfar - p_center + torch.tensor([0, 0, z])
        img_right = render([torch.ones_like(img_t)], [verts_right],
                        translation=[torch.zeros_like(trans_t)])
        return img_right[0]

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

if __name__ == '__main__':
    img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    dataset = ThreeDPW(None, None, 256, img_norm_cfg)
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    loader = iter(loader)
    data = next(loader)
    print(type(data))
