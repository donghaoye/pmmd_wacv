#coding=utf-8

from logging.config import valid_ident
import os.path as osp
from re import I

import copy

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor, DepthMapTransform)
from .utils import to_tensor, random_scale, flip_kp, flip_pose, rot_aa, crop_image
from .extra_aug import ExtraAugmentation
from .custom import CustomDataset
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
from mmdet.datasets.utils import draw_skeleton
from .utils import project_point_np, draw_point
from mmdet.models.utils import read_pfm, write_pfm, resize_depth, write_depth, resize_depth_np


denormalize = lambda x: x * np.array([0.229, 0.224, 0.225])[None, None, :] + np.array([0.485, 0.456, 0.406])[None, None,
                                                                             :]
import pycocotools.mask as mask_util


from .occlude import load_occluders, occlude_with_objects


# SELECTED_JOINT_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18]
# UNSELECTED_JOINT_IDS = [13, 15, 17, 19, 20, 21, 22, 23]       # eval on vibe protocol H36M to J14
# UNSELECTED_JOINT_IDS = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23] # eval on LSP 14joints

def rot2DPts(x, y, rotMat):
    new_x = rotMat[0, 0] * x + rotMat[0, 1] * y + rotMat[0, 2]
    new_y = rotMat[1, 0] * x + rotMat[1, 1] * y + rotMat[1, 2]
    return new_x, new_y


from .builder import DATASETS

@DATASETS.register_module()
class CLIFF_COCO_MPII_Dataset_StageII(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    # CLASSES = ('Human',)
    CLASSES = ('Background', 'Human',)
    
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
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
                 with_camera_trans=True,
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

                 # additional
                 occlude=None,
                 occluders=None,
                 crop=False,
                 global_augment=None,
                 extra_annot_path=None,
                 dataset_name=None,
                 center_def_kp=True,
                 use_gender=False,
                 regress_trans=False,
                 JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy',
                 use_clip_bbox=True,
                 color_jitter=None,
                 use_padShape_as_imgShape=True,
                 depthmap_scale_by_augFactor=False,
                 **kwargs,
                 ):

        self.extra_annot_path = extra_annot_path
        self.dataset_name = dataset_name
        self.center_def_kp = center_def_kp
        self.use_gender = use_gender
        self.regress_trans = regress_trans
        self.crop = crop
        self.with_camera_trans = with_camera_trans

        self.use_padShape_as_imgShape = use_padShape_as_imgShape
        self.depthmap_scale_by_augFactor = depthmap_scale_by_augFactor

        print(">>>>>>>>cliff stageII coco mpii with_camera_trans:", self.with_camera_trans)
        print(">>>>>>>>cliff stageII coco mpii depthmap_scale_by_augFactor:", depthmap_scale_by_augFactor)

        self.color_jitter = color_jitter
        self.global_augment = global_augment
        if global_augment:
            print('enable global augment!!')

        if self.regress_trans:
            self.smplr = SMPLR('data/smpl', use_gender=use_gender)
            
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations by each person
        self.img_infos = self.load_annotations(ann_file)
        # self.img_infos = self.load_each_person_annotations(ann_file)
        print("load cliff coco:", len(self.img_infos), ann_file)
        print("=" * 50)

        # self.max_samples = max_samples

        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None

        # filter images with no annotation during training
        if not test_mode and False:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # # Select a subset for quick validation
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

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # prefix of semantic segmentation map path
        self.seg_prefix = seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not
        self.test_mode = test_mode
        # For dataset with SMPL shape parameters
        self.with_shape = with_shape
        # For dataset with joints rotation matrix in SMPL model
        self.with_kpts3d = with_kpts3d
        # For dataset with 2D pose
        self.with_kpts2d = with_kpts2d
        # For dataset with camera parameters in SMPL model
        self.with_trans = with_trans
        # For pose in axis angle of the joints
        self.with_pose = with_pose

        # Densepose annotations
        self.with_dp = with_dp

        # noise factor for color jittering
        self.noise_factor = noise_factor

        # Whether to adjust bbox to square manually..
        self.square_bbox = square_bbox

        # Rotation facotr
        self.rot_factor = rot_factor

        # Mosh dataset for generator
        self.mosh_path = None  # mosh_path

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
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

        if sample_by_persons:
            persons_cnt = np.zeros(len(self.img_infos))
            for i in range(len(self.img_infos)):
                persons_cnt[i] = self.get_ann_info(i)['kpts2d'].shape[0]
            self.density = sample_weight * persons_cnt / persons_cnt.sum()
        else:
            self.density = sample_weight * np.ones(len(self.img_infos)) / len(self.img_infos)

        self.ignore_3d = ignore_3d
        self.ignore_smpl = ignore_smpl
        self.with_nr = with_nr
        self.use_poly = use_poly

        if self.mosh_path:
            mosh = np.load(mosh_path)
            self.mosh_shape = mosh['shape'].copy()
            self.mosh_pose = mosh['pose'].copy()
            self.mosh_sample_list = range(self.mosh_shape.shape[0])

        # occlude
        self.occlude = occlude
        self.occluders = occluders
        # if occlude is not None and occluders is None:
        #     # /data/Mono3DPerson/VOCdevkit/VOC2012
        #     self.occluders = load_occluders(occlude['path'])
        #     print('Loaded {} suitable occluders from VOC2012'.format(len(self.occluders)))
        if occlude is not None and occluders is not None:
            print('Received {} occluders!'.format(len(self.occluders)))
        else:
            self.occluders = None

        # # ROMP
        # if self.extra_annot_path is not None:
        #     self.extra_smpl_gt = self.load_extra_annotations(self.extra_annot_path)
        # else:
        #     self.extra_smpl_gt = None

    def __len__(self):
        return len(self.img_infos)


    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as f:
            raw_infos = pickle.load(f)

        raw_infos_new = []
        for i in range(len(raw_infos)):
            filename = raw_infos[i]['filename']
            width = raw_infos[i]['width']
            height = raw_infos[i]['height']
            camMat_original = raw_infos[i]['camMat_original']
            camMat = raw_infos[i]['camMat']
            if 'id' not in raw_infos[i].keys():
                continue
            id = raw_infos[i]['id']
            
            person_num = raw_infos[i]['bboxes'].shape[0]
            # for j in range(person_num):
            bbox = raw_infos[i]['bboxes']
            kpts2d = raw_infos[i]['kpts2d'].astype(np.float32)
            kpts3d = raw_infos[i]['kpts3d'].astype(np.float32)
            beta = raw_infos[i]['betas']
            pose = raw_infos[i]['pose'].reshape(person_num, -1, 3)
            has_smpl = raw_infos[i]['has_smpl']
            global_translation = raw_infos[i]['global_translations']
            kpts2d_smpl_proj = raw_infos[i]['kpts2d_smpl_proj']
            # vertices = raw_infos[i]['vertices']
            
            # # set unused joint confidence as 0
            # kpts2d[:, UNSELECTED_JOINT_IDS, 2] = 0
            # kpts3d[:, UNSELECTED_JOINT_IDS, 3] = 0

            datum_each_person = dict(
                id=id,
                filename=filename,
                camMat_original=camMat_original,
                camMat=camMat,
                width=width,
                height=height,
                bbox=bbox,
                kpts2d=kpts2d,
                kpts3d=kpts3d,
                pose=pose,
                beta=beta,
                has_smpl=has_smpl,
                global_translation=global_translation,
                kpts2d_smpl_proj=kpts2d_smpl_proj,
                # vertices=vertices,
            )
            raw_infos_new.append(datum_each_person)
        raw_infos_new = np.array(raw_infos_new)

        return raw_infos_new

    def load_each_person_annotations(self, ann_file):
        with open(ann_file, 'rb') as f:
            raw_infos = pickle.load(f)

        raw_infos_by_person = []
        for i in range(len(raw_infos)):
            filename = raw_infos[i]['filename']
            width = raw_infos[i]['width']
            height = raw_infos[i]['height']
            camMat_original = raw_infos[i]['camMat_original']
            camMat = raw_infos[i]['camMat']
            
            person_num = raw_infos[i]['bboxes'].shape[0]
            for j in range(person_num):
                bbox = raw_infos[i]['bboxes'][j]
                kpts2d = raw_infos[i]['kpts2d'][j].astype(np.float32)
                kpts3d = raw_infos[i]['kpts3d'][j].astype(np.float32)
                beta = raw_infos[i]['betas'][j]
                pose = raw_infos[i]['pose'][j].reshape(-1, 3)
                has_smpl = raw_infos[i]['has_smpl'][j]
                global_translation = raw_infos[i]['global_translations'][j]
                
                # # set unused joint confidence as 0
                # kpts2d[UNSELECTED_JOINT_IDS, 2] = 0
                # kpts3d[UNSELECTED_JOINT_IDS, 3] = 0

                bbox = bbox[None,...]
                kpts2d = kpts2d[None,...]
                kpts3d = kpts3d[None,...]
                pose = pose[None,...]
                beta = beta[None,...]
                has_smpl = has_smpl[None,...]
                global_translation = global_translation[None,...]

                datum_each_person = dict(
                    filename=filename,
                    camMat_original=camMat_original,
                    camMat=camMat,
                    width=width,
                    height=height,
                    bbox=bbox,
                    kpts2d=kpts2d,
                    kpts3d=kpts3d,
                    pose=pose,
                    beta=beta,
                    has_smpl=has_smpl,
                    global_translation=global_translation,
                )
                raw_infos_by_person.append(datum_each_person)

        raw_infos_by_person = np.array(raw_infos_by_person)
        return raw_infos_by_person

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        """
        :param idx:
        :return:A dict of the following iterms:
            bboxes: [x1, y1, x2, y2]
            labels: number
            kpts3d: (24, 4)
            kpts2d: (24, 3)
            pose: (72,)
            shape: (10,)
            cam: (3,) (The trans in SMPL model)
        """
        raw_info = deepcopy(self.img_infos[idx])
        num_persons = raw_info['bbox'].shape[0]
        bbox = raw_info['bbox']

        if self.square_bbox:
            bbox = bbox[0]
            center = np.array([int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)])
            bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            half_size = int(math.floor(bbox_size / 2))
            square_bbox = np.array(
                [center[0] - half_size, center[1] - half_size, center[0] + half_size, center[1] + half_size]).reshape(1, -1)
        else:
            square_bbox = bbox


        ret_dict = {
                'id': raw_info['id'],
                'filename': raw_info['filename'],  
                'width': raw_info['width'],  
                'height': raw_info['height'],  
                # 'genders': raw_info['genders'], 
                'bboxes': square_bbox.astype(np.float32),  
                'labels':  np.array([1] * num_persons),
                # 'labels':  np.array([0] * num_persons),
                'kpts3d': raw_info['kpts3d'].astype(np.float32),  # (n, 24,4) extra chanel for visibility
                'kpts2d': raw_info['kpts2d_smpl_proj'].astype(np.float32),  # (n, 24,3) extra chanel for visibility
                # 'kpts2d': raw_info['kpts2d'].astype(np.float32),  # (n, 24,3) extra chanel for visibility
                # 'kpts2d_smpl_proj': raw_info['kpts2d_smpl_proj'].astype(np.float32), 
                'pose': raw_info['pose'].astype(np.float32),  # (n, 24, 3)
                'shape': raw_info['beta'].astype(np.float32),  # (n, 10)
                'trans': np.zeros((num_persons, 3)).astype(np.float32),  # (n, 3)
                'camera_trans': raw_info['global_translation'].astype(np.float32), 
                'has_smpl': raw_info['has_smpl'].astype(np.float32), 
                # 'vertices': raw_info['vertices'].astype(np.float32), 
                'masks': None,
                }  
        return ret_dict

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

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

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        # if self.test_mode:
        #     return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    @staticmethod
    def val(runner, dataloader, **kwargs):
        from IPython import embed
        embed()
        pass

    @staticmethod
    def annToRLE(ann):
        h, w = ann['height'], ann['width']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_util.frPyObjects(segm, h, w)
            rle = mask_util.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = mask_util.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def prepare_train_img(self, idx):
        is_train = not self.test_mode

        ann = self.get_ann_info(idx)
        if ann == None:
            return None

        img_path = osp.join(self.img_prefix, ann['filename'])
        img = mmcv.imread(img_path)
        ori_shape = (img.shape[0], img.shape[1], 3)

        # body_depthmap
        if img_path.find('/train2014/') > -1:
            body_depthmap_path = img_path.replace('/train2014/', '/train2014_bodymesh_depthmap_rescale_832x512/')
        else:
            body_depthmap_path = img_path.replace('/images/', '/bodymesh_depthmap_rescale_832x512/')
        # body_depthmap_path = body_depthmap_path[:-4] + '.pfm'
        depthmap_id = ann['id']
        body_depthmap_path = body_depthmap_path[:-4] + f'_id{depthmap_id}.pfm'
        body_depthmap, _ = read_pfm(body_depthmap_path)

        gt_bboxes = ann['bboxes']
        num_people = gt_bboxes.shape[0]

        # resacel bbox to 1.2x
        center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.
        length = np.abs((gt_bboxes[:, :2] - gt_bboxes[:, 2:]))
        # length = length * 1.2
        gt_bboxes = np.zeros_like(gt_bboxes)
        gt_bboxes[:,0] = center[:,0] - length[:,0] / 2.
        gt_bboxes[:,1] = center[:,1] - length[:,1] / 2.
        gt_bboxes[:,2] = center[:,0] + length[:,0] / 2.
        gt_bboxes[:,3] = center[:,1] + length[:,1] / 2.
        ann['bboxes'] = gt_bboxes

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
                    #print('range',np.max(occluder[o_idx]))
                    occluder[o_idx][:, :, 0] = np.minimum(255.0, np.maximum(0.0, occluder[o_idx][:, :, 0] * pn[2]))
                    occluder[o_idx][:, :, 1] = np.minimum(255.0, np.maximum(0.0, occluder[o_idx][:, :, 1] * pn[1]))
                    occluder[o_idx][:, :, 2] = np.minimum(255.0, np.maximum(0.0, occluder[o_idx][:, :, 2] * pn[0]))
                    occluder[o_idx][:, :, :3] -= self.img_norm_cfg['mean'][::-1]
                    occluder[o_idx][:, :, :3] /= self.img_norm_cfg['std'][::-1]

        if not (np.sum(ann['kpts2d'][:,:,-1], axis=1, keepdims=False) > 0).all():
            return None

        if len(gt_bboxes) == 0:
            return None

        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes, gt_labels)

        # apply transforms
        flip = True if (np.random.rand() < self.flip_ratio and is_train) else False
        
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)

        ######### global augmentation start ##########
        # global augment image scale
        augment_scale_factor = 1
        enable_global_augment = False
        if is_train and self.global_augment is not None:
            if np.random.rand() < self.global_augment['ratio']:
                enable_global_augment = True
                augment_scale_factor = np.random.uniform(self.global_augment['scale'][0], self.global_augment['scale'][1])
        scaled_img_scale = (img_scale[0] * augment_scale_factor, img_scale[1] * augment_scale_factor)

        img, img_shape, pad_shape, scale_factor = self.img_transform(img, scaled_img_scale, flip, keep_ratio=self.resize_keep_ratio)
        body_depthmap, _, _, depthmap_scale_factor = self.depthmap_transform(body_depthmap, scaled_img_scale, flip, keep_ratio=self.resize_keep_ratio)
        raw_img_shape = copy.deepcopy(img_shape)

        # scale depthmap value
        if self.depthmap_scale_by_augFactor == True:
            # body_depthmap = body_depthmap / depthmap_scale_factor
            body_depthmap = body_depthmap / scale_factor
        
        if len(body_depthmap.shape) > 2:
            body_depthmap = body_depthmap.transpose((2, 0, 1))
        else:
            body_depthmap = body_depthmap[None, :, :]

        if enable_global_augment:
            h, w = ori_shape[0], ori_shape[1]
            scale = min(img_scale[1] / h, img_scale[0] / w)
            img_shape = (int(h * float(scale) + 0.5), int(w * float(scale) + 0.5), img_shape[-1])
            pad_shape = img_shape

        # 当augment_scale_factor>1
        # center crop large image
        diff = (0, 0)
        if enable_global_augment and augment_scale_factor > 1:
            diff = (img.shape[1] - img_shape[0], img.shape[2] - img_shape[1]) 
            img = img[:,diff[0]//2:diff[0]//2 + img_shape[0],diff[1]//2:diff[1]//2 + img_shape[1]]
            body_depthmap = body_depthmap[:, diff[0]//2:diff[0]//2 + img_shape[0], diff[1]//2:diff[1]//2 + img_shape[1]]

        # Force padding for the issue of multi-GPU training
        padded_img = np.zeros((img.shape[0], img_scale[1], img_scale[0]), dtype=img.dtype)
        padded_depthmap = np.zeros((1, img_scale[1], img_scale[0]), dtype=body_depthmap.dtype)

        # global augment xy
        if enable_global_augment:#is_train and self.global_augment is not None and augment_scale_factor != 1:
            start_pos_x = np.random.randint(0, img_scale[1] - img.shape[-2]) if img_scale[1] > img.shape[-2] else 0
            start_pos_y = np.random.randint(0, img_scale[0] - img.shape[-1]) if img_scale[0] > img.shape[-1] else 0
        else:
            # corner started
            # start_pos_x = 0
            # start_pos_y = 0
            # center started
            start_pos_x = img_scale[1] // 2 - img.shape[-2] // 2
            start_pos_y = img_scale[0] // 2 - img.shape[-1] // 2

        padded_img[:, start_pos_x:start_pos_x+img.shape[-2], start_pos_y:start_pos_y+img.shape[-1]] = img
        padded_depthmap[:, start_pos_x:start_pos_x+body_depthmap.shape[-2], start_pos_y:start_pos_y+body_depthmap.shape[-1]] = body_depthmap
        img = padded_img
        body_depthmap = padded_depthmap
        ######### global augmentation end ##########
        
        gt_bboxes = self.bbox_transform(gt_bboxes, raw_img_shape, scale_factor,
                                        flip, translation = (start_pos_y - diff[1]//2, start_pos_x - diff[0]//2))
        
        #print('box shape', gt_bboxes.shape)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape, scale_factor, flip)
        if self.with_shape:
            gt_shapes = ann['shape']
        if self.with_kpts2d:
            gt_kpts2d = ann['kpts2d']
            # Rescale the 2D keypoints now.
            s_kpts2d = np.zeros_like(gt_kpts2d)
            s_kpts2d[..., -1] = gt_kpts2d[..., -1]
            s_kpts2d[..., :-1] = gt_kpts2d[..., :-1] * scale_factor
            gt_kpts2d = s_kpts2d

            if flip:
                for i, kp in enumerate(gt_kpts2d):
                    gt_kpts2d[i] = flip_kp(kp, raw_img_shape[1])  # img is (C, H, W)
                    # NOTE: I use the img_shape to avoid the influence of padding.
            
            # global augment 2d joints
            gt_kpts2d[..., 0] = gt_kpts2d[..., 0] + start_pos_y - diff[1]//2
            gt_kpts2d[..., 1] = gt_kpts2d[..., 1] + start_pos_x - diff[0]//2

        if self.with_trans:
            gt_trans = ann['trans']
        if self.with_pose:
            gt_poses = ann['pose'].reshape(num_people, -1, 3)
            if flip:
                for i, ps in enumerate(gt_poses):
                    gt_poses[i] = flip_pose(ps.reshape(-1)).reshape(-1, 3)
        
        if self.with_kpts3d:
            if not self.regress_trans:
                gt_kpts3d = ann['kpts3d']
                # verts_local = data['vertices'] # flip?
                # flip vertices?
                if flip:
                    for i, kp in enumerate(gt_kpts3d):
                        gt_kpts3d[i] = flip_kp(kp, 0)  # Not the image width as the pose is centered by hip.
            else:
                pose_hand_theta = np.zeros([1, 2, 3]).astype(np.float32)
                gt_poses[:, -2:, :] = pose_hand_theta
                outputs = self.smplr(betas=gt_shapes, body_pose=gt_poses[:, 1:, :], global_orient=gt_poses[:, :1, :], pose2rot=True)
                gt_vertices = outputs.vertices.detach().numpy().astype(np.float32)
                joints = outputs.joints.detach().numpy().astype(np.float32) # n, 24, 3
                gt_kpts3d = np.ones((joints.shape[0], joints.shape[1], 4)).astype(np.float32)
                gt_kpts3d[:, :, :3] = joints

        # occlude
        if occluder is not None:
            areas = np.abs((gt_bboxes[:,2] - gt_bboxes[:,0]) * (gt_bboxes[:,3] - gt_bboxes[:,1]))
            img = occlude_with_objects(img, occluder, areas, gt_kpts2d)
        
        if self.use_padShape_as_imgShape:
            img_shape=padded_img.transpose([1, 2, 0]).shape # img_shape should be consistent with the input img of model
            pad_shape=padded_img.transpose([1, 2, 0]).shape # pad_shape should be consistent with the input img of model

        img_metas = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip,
            idx=idx,
            file_name=ann['filename']
        )
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_metas=DC(img_metas, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes.astype(np.float32))))
        # if self.proposals is not None:
        #     data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_kpts2d:
            data['gt_kpts2d'] = DC(to_tensor(gt_kpts2d))
        if self.with_kpts3d:
            data['gt_kpts3d'] = DC(to_tensor(gt_kpts3d))
        if self.with_pose:
            data['gt_poses'] = DC(to_tensor(gt_poses))
        if self.with_shape:
            data['gt_shapes'] = DC(to_tensor(gt_shapes))
        if self.with_trans:
            data['gt_trans'] = DC(to_tensor(gt_trans))

        data['has_smpl'] = DC(to_tensor(np.ones((gt_bboxes.shape[0],)).astype(np.int_)))

        if self.mosh_path:
            sampled_idxs = np.array(random.sample(self.mosh_sample_list, 36))
            mosh_pose = torch.tensor(deepcopy(self.mosh_pose[sampled_idxs].astype(np.float32)))
            mosh_shape = torch.tensor(deepcopy(self.mosh_shape[sampled_idxs].astype(np.float32)))
            mosh_bs = mosh_shape.shape[0]
            mosh_pose_shape = torch.cat([batch_rodrigues(mosh_pose.view(-1, 3)).view(mosh_bs, -1), mosh_shape], dim=1)
            data['mosh'] = DC(mosh_pose_shape)

        # make 3dpw happy
        data['gt_scale'] = DC(to_tensor(np.zeros((gt_trans.shape[0], 3)).astype(np.float32))) # p 3

        gt_camera_trans = ann['camera_trans']
        if flip:
            gt_camera_trans[:,0] *= -1. # is it x?
        # gt_kpts3d和gt_camera_trans都是focal_len 1000下的，所以也要做scale
        gt_camera_trans = gt_camera_trans + gt_kpts3d[:, 14, :3]
        gt_camera_trans[:, -1] /= scale_factor
        gt_camera_trans = gt_camera_trans - gt_kpts3d[:, 14, :3]

        # print("Cliff gt_camera_trans[:, -1]: ", gt_camera_trans[:, -1])
        # print("----------------")

        pixel_offset_y = img_scale[0] // 2
        # pixel_offset_x -= (start_pos_y + raw_img_shape[1] // 2) if augment_scale_factor < 1 else (start_pos_y + img_shape[1] // 2)
        if augment_scale_factor < 1:
            pixel_offset_y -= (start_pos_y + raw_img_shape[1] // 2)
        else:
            pixel_offset_y -= (start_pos_y + img_shape[1] // 2)
            # pixel_offset_y -= (start_pos_y - diff[1]//2 + raw_img_shape[1] // 2)
        
        if pixel_offset_y != 0:
            camera_offset_y = (gt_camera_trans[:,-1] * pixel_offset_y) / 1000.
            gt_camera_trans[:,0] -= camera_offset_y

        pixel_offset_x = img_scale[1] // 2
        # pixel_offset_x -= (start_pos_x + raw_img_shape[0] // 2) if augment_scale_factor < 1 else (start_pos_x + img_shape[0] // 2)
        if augment_scale_factor < 1:
            pixel_offset_x -= (start_pos_x + raw_img_shape[0] // 2)
        else:
            pixel_offset_x -= (start_pos_x + img_shape[0] // 2)
            # pixel_offset_x -=  (start_pos_x - diff[0]//2 + raw_img_shape[0] // 2)

        if pixel_offset_x != 0:
            camera_offset_x = (gt_camera_trans[:,-1] * pixel_offset_x) / 1000.
            gt_camera_trans[:,1] -= camera_offset_x

        # depth out of range
        # if np.min(gt_camera_trans[:, -1]) < 1.0 or np.max(gt_camera_trans[:, -1]) > 26.0:
        if np.min(gt_camera_trans[:, -1]) < 1.0:
            return None

        # align with the fake focal length
        data['gt_camera_trans'] = DC(to_tensor(gt_camera_trans)) # p 3
        data['has_trans'] = DC(to_tensor(np.ones((gt_bboxes.shape[0],)).astype(np.int_)))
        data['gt_vertices'] = DC(to_tensor(gt_vertices))
        
        # is_nan = np.any(np.isnan(body_depthmap))
        # if is_nan == True:
        #     print(body_depthmap)
        #     print(np.isnan(body_depthmap))
        #     print(">>>>>>>>>>>is_nan: ", is_nan)

        #     # debug visual gt_depth_mask
        #     print(body_depthmap.shape)
        #     gt_mask_np = body_depthmap[0]
        #     bits = 1
        #     depth_min = gt_mask_np.min()
        #     depth_max = gt_mask_np.max()
        #     max_val = (2 ** (8 * bits)) - 1
        #     if depth_max - depth_min > np.finfo("float").eps:
        #         out = max_val * (gt_mask_np - depth_min) / (depth_max - depth_min)
        #     else:
        #         out = np.zeros(gt_mask_np.shape, dtype=gt_mask_np.dtype)
        #     import cv2
        #     output_path = "./debug/gt_depthmap_debug_isnan.png"

        #     img = img.transpose([1, 2, 0])
        #     img = cv2.cvtColor((denormalize(img) * 255).astype(np.uint8).copy(), cv2.COLOR_BGR2RGB)
        #     depthmap = out.astype("uint8")[:, :, None]
        #     depthmap = cv2.cvtColor(depthmap, cv2.COLOR_GRAY2RGB)
        #     print(depthmap.shape)
        #     img = cv2.addWeighted(img, 0.5, depthmap, 0.5, 0)
        #     cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #     print("gt_depthmap_debug.shape: ", gt_mask_np.shape)
        #     print(output_path)

        body_depthmap = to_tensor(body_depthmap)
        body_depthmap[torch.isnan(body_depthmap)] = 0.0
        data['gt_depthmap'] = DC(body_depthmap)
        data['has_depthmap'] = DC(to_tensor(np.ones(1).astype(np.int64)))

        # data['gt_masks'] = DC(np.zeros((gt_bboxes.shape[0], img.shape[-2], img.shape[-1]), dtype=np.uint8), cpu_only=True)
        # data['has_masks'] = DC(to_tensor(np.zeros(gt_bboxes.shape[0]).astype(np.int64)))
        #### mask from depthmap
        gt_masks = np.zeros_like(body_depthmap).astype(np.uint8)
        gt_masks[body_depthmap > 0] = 1
        gt_masks = np.repeat(gt_masks, num_people, axis=0)
        data['gt_masks'] = DC(gt_masks.astype(np.uint8), cpu_only=True)
        data['has_masks'] = DC(to_tensor(np.ones((num_people,)).astype(np.int64)))
        
        # gt_masks_vis = gt_masks[0][:, :, None] * 255
        # output_path = "./debug/gt_masks.jpg"
        # cv2.imwrite(output_path, gt_masks_vis)
        # print(output_path)
        # print(img_path, flip)
        
        # #############################
        # ### 可视化 dataloader data ###
        # #############################
        # import os
        # debug_filename = os.path.basename(ann['filename'])
        # img_viz = self.prepare_dump(data, debug_filename)
        
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

        # for i in range(gt_bboxes.shape[0]):
        #     bbox = gt_bboxes[i]
        #     img_viz = cv2.rectangle(img_viz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        
        # output_path = f"./debug/{debug_filename}_scale_factor:{augment_scale_factor}.jpg"
        # cv2.imwrite(output_path, img_viz[:, :, :])
        # print(img_path)
        # print(output_path)
    
        return data


    def prepare_dump(self, data, debug_filename):
        from mmdet.models.utils.smpl.renderer import Renderer
        from mmdet.models.utils.smpl.smpl import SMPL
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
        gt_vertices = data['gt_vertices'].data
        img = data['img'].data

        _, H, W = img.shape
        FOCAL_LENGTH = 1000
        render = Renderer(focal_length=FOCAL_LENGTH, height=H, width=W)

        # img = cv2.cvtColor((denormalize(img.transpose([1, 2, 0])) * 255).astype(np.uint8).copy(),
        #                           cv2.COLOR_BGR2RGB)
        # from scipy.spatial.transform import Rotation as R
        # smpl = SMPL('data/smpl')
        # output = smpl(betas=gt_shapes, body_pose=gt_poses[:, 1:], global_orient=gt_poses[:, :1], transl=gt_trans, pose2rot=True)
        # output = smpl(betas=gt_shapes, body_pose=gt_poses[:, 3:], global_orient=gt_poses[:, :3], transl=gt_trans, pose2rot=True)
        # verts = output.vertices

        verts = gt_vertices

        # np.save(f'./{debug_filename}_gt_vertices.npy', verts.cpu().numpy())
        # np.save(f'./{debug_filename}_gt_trans.npy', gt_camera_trans.cpu().numpy())
        
        try:
            fv_rendered = render([img.clone()], [verts], translation=[gt_camera_trans])[0]
            bv_rendered = self.renderer_bv(img, verts, gt_camera_trans, gt_bboxes[0], FOCAL_LENGTH, render)
        except Exception as e:
            print(e)
            return None

        img = np.float32(img.cpu().numpy())
        total_img = np.zeros((3 * H, W, 3))
        total_img[:H] += img.transpose([1, 2, 0])
        total_img[H:2 * H] += fv_rendered.transpose([1, 2, 0])
        total_img[2 * H:] += bv_rendered.transpose([1, 2, 0])
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



    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_metas=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
