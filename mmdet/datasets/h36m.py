import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor, DepthMapTransform)
from .utils import to_tensor, random_scale, flip_kp, flip_pose, rot_aa, crop_image
from .extra_aug import ExtraAugmentation
from .custom import CustomDataset
import os.path as osp
# import tensorboardX
import math
import json
import pickle
import matplotlib.pyplot as plt
from mmdet.models.utils.smpl_utils import batch_rodrigues
from mmdet.models.utils.smpl.smpl import SMPLR
import random
from copy import deepcopy
import scipy
import scipy.misc
# import seaborn as sns
import torch
# import png
import skimage.transform
import copy
from .utils import project_point_np, draw_point
from mmdet.models.utils import read_pfm, write_pfm, resize_depth, write_depth, resize_depth_np


denormalize = lambda x: x * np.array([0.229, 0.224, 0.225])[None, None, :] + np.array([0.485, 0.456, 0.406])[None, None,
                                                                             :]
import pycocotools.mask as mask_util


from .occlude import load_occluders, occlude_with_objects

from mmdet.datasets.utils import draw_skeleton
import cv2


def rot2DPts(x, y, rotMat):
    new_x = rotMat[0, 0] * x + rotMat[0, 1] * y + rotMat[0, 2]
    new_y = rotMat[1, 0] * x + rotMat[1, 1] * y + rotMat[1, 2]
    return new_x, new_y

from .builder import DATASETS

@DATASETS.register_module()
class H36MDataset(Dataset):
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

    # CLASSES = None
    # CLASSES = ('Human',)
    CLASSES = ('Background', 'Human',)
    subject_gender = {'S1':1, 'S5':1, 'S6':0, 'S7':1, 'S8':0, 'S9':0, 'S11':0}

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

                 # ROMP
                 extra_annot_path=None,
                 dataset_name=None,
                 center_def_kp=True,
                 use_gender=False,
                 regress_trans=False,
                 JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy',
                 use_clip_bbox=True,
                 color_jitter=None,
                 is_stageII=False,
                 use_padShape_as_imgShape=True,
                 depthmap_scale_by_augFactor=False,
                 data_root="/data/datasets/",
                 **kwargs,
                 ):

        self.extra_annot_path = extra_annot_path
        self.dataset_name = dataset_name
        self.center_def_kp = center_def_kp
        self.use_gender = use_gender
        self.regress_trans = regress_trans
        self.crop = crop
        self.with_camera_trans = with_camera_trans
        self.is_stageII = is_stageII
        self.use_padShape_as_imgShape = use_padShape_as_imgShape
        self.depthmap_scale_by_augFactor = depthmap_scale_by_augFactor
        self.data_root = data_root

        print("!>>>>>>>h36m self.use_padShape_as_imgShape: ", self.use_padShape_as_imgShape)
        print(">>>>>>>>h36m depthmap_scale_by_augFactor:", depthmap_scale_by_augFactor)
        print(">>>>>>>>h36m with_camera_trans:", self.with_camera_trans)

        self.color_jitter = color_jitter
        self.global_augment = global_augment
        if global_augment:
            print('enable global augment!!')

        if self.regress_trans:
            self.smplr = SMPLR('data/smpl', use_gender=use_gender)

        if 'mpi_inf_3dhp' in ann_file:
            self.dataset_name = 'mpi_inf_3dhp'
        elif 'lsp' in ann_file:
            self.dataset_name = 'lsp'
        elif 'mpii' in ann_file:
            self.dataset_name = 'mpii'
        elif 'posetrack' in ann_file:
            self.dataset_name = 'posetrack'
        elif 'mask_train' in ann_file and 'coco' in ann_file:
            self.dataset_name = 'coco_smpl'

        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        print("ann_file: ", ann_file)
        print("=" * 30)
        self.img_infos = self.load_annotations(ann_file)
        # self.max_samples = max_samples

        # # 模拟越界bbox
        # for i in range(len(self.img_infos)):
        #     bboxes = self.img_infos[i]['bboxes'] 
        #     # print(">>>>bboxes.shape[0]:", bboxes.shape[0])
        #     for j in range(bboxes.shape[0]):
        #         self.img_infos[i]['bboxes'][j] = np.array([1300, 140, 3000, 3000])

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
        """
        filename:
        height: 1000
        width: commonly 1002 in h36m
        :param ann_file:
        :return:
        """
        with open(ann_file, 'rb') as f:
            raw_infos = pickle.load(f)
        return raw_infos

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
        # Visualization needed
        raw_info = deepcopy(self.img_infos[idx])
        num_persons = raw_info['kpts2d'].shape[0]
        if num_persons > 1:
            print(">>>>>!!!! h36m.py num_persons > 1!!!!")
            exit()

        bbox = raw_info['bboxes']
        if self.square_bbox:
            bbox = bbox[0]
            center = np.array([int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)])
            bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            half_size = int(math.floor(bbox_size / 2))
            square_bbox = np.array(
                [center[0] - half_size, center[1] - half_size, center[0] + half_size, center[1] + half_size]).reshape(1, -1)
        else:
            square_bbox = bbox

        has_smpl = raw_info['has_smpl'].astype(np.int64)
        if self.dataset_name == 'h36m':
            has_smpl = has_smpl[0] # fixed preprocess bugs: has_smpl_s = np.ones((1, 1)).astype(np.float32)

        ret_dict =  {
                'filename': raw_info['filename'],  
                'width': raw_info['width'],  
                'height': raw_info['height'],  
                'genders': raw_info['genders'], 
                'bboxes': square_bbox.astype(np.float32),  # (1,4)
                'labels':  np.array([1] * num_persons),
                # 'labels':  np.array([0] * num_persons),  # only one class, 0 means Human gt_labels (list[Tensor]): class indices corresponding to each box
                'kpts3d': raw_info['kpts3d'].astype(np.float32),  # (1, 24,4) extra chanel for visibility
                # 'kpts2d': raw_info['kpts2d'].astype(np.float32),  # (1, 24,3) extra chanel for visibility
                'kpts2d': raw_info['kpts2d_smpl_proj'].astype(np.float32),  # (1, 24,3) extra chanel for visibility
                'pose': raw_info['pose'].reshape(1, -1, 3).astype(np.float32),  # (1, 24, 3)
                'shape': raw_info['betas'].astype(np.float32),  # (1,10)
                'trans': np.zeros((num_persons, 3)).astype(np.float32),  # (n, 3)
                'camera_trans': raw_info['global_translations'].astype(np.float32),
                'has_smpl': has_smpl,
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
        
        if self.dataset_name == 'coco_smpl':
            filename = ann['filename'][:-4]
            if filename.find('pretrain') > -1:
                mask_root = f"{self.data_root}/coco_smpl/pretrain/mask_files"
            else: 
                mask_root = f"{self.data_root}/coco_smpl/mask_files"

            mask_path = osp.join(mask_root, osp.basename(filename) + '.pkl')
            mask_ann = pickle.load(open(mask_path, 'rb'))

            if not filename.find('pretrain') > -1: # stageI not include 'segmentation'
                ann['segmentation'] = mask_ann['segmentation']
            ann['masks'] = mask_ann['masks']

        # filter non-2djoint
        joint_num = ann['kpts2d'].shape[0]
        valid_joint_indexs = np.where(ann['kpts2d'][:, :, -1] > 0.5)
        if len(valid_joint_indexs[1]) < 8 * joint_num:
            # img_path = osp.join(self.img_prefix, ann['filename'])
            # print(img_path)
            return None

        img_path = osp.join(self.img_prefix, ann['filename'])
        # print(img_path)
        img = mmcv.imread(img_path)
        ori_shape = (img.shape[0], img.shape[1], 3)
        raw_shape = copy.deepcopy(img.shape)

        
        ## stageI的depthmap应该也需要body_depthmap
        # if is_train and self.is_stageII == True and self.dataset_name == 'h36m': 
        if is_train and self.dataset_name == 'h36m': 
            # body_depthmap
            body_depthmap_path = img_path.replace('/images/', '/bodymesh_depthmap_rescale_832x512/')
            body_depthmap_path = body_depthmap_path[:-4] + '.pfm'
            body_depthmap, _ = read_pfm(body_depthmap_path)

        gt_bboxes = ann['bboxes']
        num_people = gt_bboxes.shape[0]

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

        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        if self.ignore_3d:
            ann['kpts3d'] = np.zeros_like(ann['kpts3d'])

        if self.ignore_smpl:
            ann['has_smpl'] = np.zeros_like(ann['has_smpl'])

        # gt bbox is in xyxy
        if self.crop:
            # random_p_idx = np.random.randint(0, gt_bboxes.shape[0]) # random 1 person
            random_p_idx = 0 # only one person
            kpts2d = ann['kpts2d'][random_p_idx]
            if not self.dataset_name == "h36m":
                valid_indexs = np.where(kpts2d[:, -1] > 0)
                min_pt = np.min(kpts2d[valid_indexs], axis=0, keepdims=False)
                max_pt = np.max(kpts2d[valid_indexs], axis=0, keepdims=False)
                length = (max_pt - min_pt)  
                center = (max_pt + min_pt) / 2.
                bbox = np.array([int(center[0] - length[0]/2.), int(center[1] - length[1]/2.), int(center[0] + length[0]/2.), int(center[1] + length[1]/2.)])
                gt_bboxes = bbox[None, ...]
     
            bb_size = gt_bboxes[random_p_idx, 2:] - gt_bboxes[random_p_idx, :2]
            bb_width, bb_height = bb_size[0], bb_size[1]
            center = (gt_bboxes[random_p_idx, 2:] + gt_bboxes[random_p_idx, :2]) / 2.
            box_size = max(bb_width, bb_height)  
            bbox = gt_bboxes[random_p_idx].reshape(2,2)
            
            img, cropped_kpts2d, bbox = crop_image(img, kpts2d, bbox, center[0], center[1], box_size, box_size, 256, 256)
            #print(ann['kpts2d'].shape, cropped_kpts2d.shape)
            ann['kpts2d'] = cropped_kpts2d[None,...]
            gt_bboxes = bbox.reshape(1,4)

            ann['kpts3d'] = ann['kpts3d'][[random_p_idx]]
            ann['pose'] = ann['pose'][[random_p_idx]]
            ann['shape'] = ann['shape'][[random_p_idx]]
            ann['labels'] = ann['labels'][[random_p_idx]]
            ann['trans'] = ann['trans'][[random_p_idx]]
            ann['camera_trans'] = ann['camera_trans'][[random_p_idx]]
            ann['has_smpl'] = ann['has_smpl'][[random_p_idx]]

        if not (np.sum(ann['kpts2d'][:,:,-1], axis=1, keepdims=False) > 0).all():
            return None

        # get bbox by 2djoint
        if self.is_stageII == True and self.dataset_name in ['posetrack', 'mpii']:
            joint_num = ann['kpts2d'].shape[0]
            gt_bboxes = np.zeros((joint_num, 4), dtype=np.float32)
            for i in range(joint_num):
                kpts2d = ann['kpts2d'][i]
                valid_indexs = np.where(kpts2d[:, -1] > 0)
                min_pt = np.min(kpts2d[valid_indexs], axis=0, keepdims=False)
                max_pt = np.max(kpts2d[valid_indexs], axis=0, keepdims=False)
                length = (max_pt - min_pt) * 1.1
                center = (max_pt + min_pt) / 2.
                bbox = np.array([int(center[0] - length[0]/2.), int(center[1] - length[1]/2.), int(center[0] + length[0]/2.), int(center[1] + length[1]/2.)])
                gt_bboxes[i:i+1] = bbox[None, ...]


        bbox_scale = 1.0
        # rescale bbox
        if self.is_stageII == True and self.dataset_name in ['coco', 'mpi_inf_3dhp']: # stageII: posetrack mpii alreader to 1.2x scale (stageII coco mpi_inf_3dhp 1.0x)
            center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.
            length = np.abs((gt_bboxes[:, :2] - gt_bboxes[:, 2:]))
            length = length * bbox_scale
            gt_bboxes = np.zeros_like(gt_bboxes)
            gt_bboxes[:,0] = center[:,0] - length[:,0] / 2.
            gt_bboxes[:,1] = center[:,1] - length[:,1] / 2.
            gt_bboxes[:,2] = center[:,0] + length[:,0] / 2.
            gt_bboxes[:,3] = center[:,1] + length[:,1] / 2.
            ann['bboxes'] = gt_bboxes

        # stage I
        if self.is_stageII == False:
            center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.
            length = np.abs((gt_bboxes[:, :2] - gt_bboxes[:, 2:]))
            length = length * bbox_scale
            gt_bboxes = np.zeros_like(gt_bboxes)
            gt_bboxes[:,0] = center[:,0] - length[:,0] / 2.
            gt_bboxes[:,1] = center[:,1] - length[:,1] / 2.
            gt_bboxes[:,2] = center[:,0] + length[:,0] / 2.
            gt_bboxes[:,3] = center[:,1] + length[:,1] / 2.
            ann['bboxes'] = gt_bboxes

        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes, gt_labels)

        # apply transforms
        flip = True if (np.random.rand() < self.flip_ratio and is_train) else False
        #flip = False
        
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)

        ######### global augmentation ##########
        # global augment image scale
        augment_scale_factor = 1
        enable_global_augment = False
        if is_train and self.dataset_name == 'h36m' and self.global_augment is not None:
            if np.random.rand() < self.global_augment['ratio']:
                enable_global_augment = True
                augment_scale_factor = np.random.uniform(self.global_augment['scale'][0], self.global_augment['scale'][1])
        scaled_img_scale = (img_scale[0] * augment_scale_factor, img_scale[1] * augment_scale_factor)

        img, img_shape, pad_shape, scale_factor = self.img_transform(img, scaled_img_scale, flip, keep_ratio=self.resize_keep_ratio)
        raw_img_shape = copy.deepcopy(img_shape)

        # if is_train and self.is_stageII == True and self.dataset_name == 'h36m': 
        if is_train and self.dataset_name == 'h36m': 
            body_depthmap, _, _, depthmap_scale_factor = self.depthmap_transform(body_depthmap, scaled_img_scale, flip, keep_ratio=self.resize_keep_ratio)
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

        # center crop large image
        diff = (0, 0)
        if enable_global_augment and augment_scale_factor > 1:
            diff = (img.shape[1] - img_shape[0], img.shape[2] - img_shape[1]) 
            img = img[:,diff[0]//2:diff[0]//2 + img_shape[0],diff[1]//2:diff[1]//2 + img_shape[1]]
            # if is_train and self.is_stageII == True and self.dataset_name == 'h36m': 
            if is_train and self.dataset_name == 'h36m': 
                body_depthmap = body_depthmap[:, diff[0]//2:diff[0]//2 + img_shape[0], diff[1]//2:diff[1]//2 + img_shape[1]]

        # Force padding for the issue of multi-GPU training
        padded_img = np.zeros((img.shape[0], img_scale[1], img_scale[0]), dtype=img.dtype)
        padded_depthmap = np.zeros((1, img_scale[1], img_scale[0]), dtype=np.float32)

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


        #print(start_pos_x, start_pos_y, img.shape, padded_img.shape)
        padded_img[:, start_pos_x:start_pos_x+img.shape[-2], start_pos_y:start_pos_y+img.shape[-1]] = img
        img = padded_img

        # if is_train and self.is_stageII == True and self.dataset_name == 'h36m': 
        if is_train and self.dataset_name == 'h36m': 
            padded_depthmap[:, start_pos_x:start_pos_x+body_depthmap.shape[-2], start_pos_y:start_pos_y+body_depthmap.shape[-1]] = body_depthmap
        body_depthmap = padded_depthmap
        ######### global augmentation ##########

        if self.with_nr:
            padded_scene = np.zeros((img.shape[-2], img.shape[-1]), dtype=np.uint8)

        if self.with_nr:
            has_mask = True
            if 'dp_mask_path' in ann:
                #print(osp.join(self.img_prefix, ann['dp_mask_path']))
                raw_mask = cv2.imread(osp.join(self.img_prefix, ann['dp_mask_path']), cv2.IMREAD_UNCHANGED)
            elif 'segmentation' in ann and self.use_poly:
                assert 'COCO' in ann['filename'], "Only support coco segmentation now"
                raw_mask = np.zeros((ann['height'], ann['width']), dtype=np.uint8)
                for i, seg in enumerate(ann['segmentation']):
                    ori_mask = mask_util.decode(
                        self.annToRLE({'width': ann['width'], 'height': ann['height'], 'segmentation': seg}))
                    raw_mask[ori_mask > 0] = i + 1
            else:
                has_mask = False
            padded_scene = np.zeros((img.shape[-2], img.shape[-1]), dtype=np.uint8)
            if has_mask:
                if raw_mask is None:
                    return None
                target_shape = int(np.round(raw_shape[1] * scale_factor)), int(np.round(raw_shape[0] * scale_factor))
                #print(raw_mask.shape, target_shape)
                resized_mask = cv2.resize(raw_mask, target_shape, interpolation=cv2.INTER_NEAREST)
                if flip:
                    resized_mask = np.flip(resized_mask, axis=1)
                padded_scene[:resized_mask.shape[-2], :resized_mask.shape[-1]] = resized_mask

        if self.with_seg:
            gt_seg = mmcv.imread(osp.join(self.seg_prefix, ann['file_name'].replace('jpg', 'png')), flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        
        gt_bboxes = self.bbox_transform(gt_bboxes, raw_img_shape, scale_factor,
                                        flip, translation = (start_pos_y - diff[1]//2, start_pos_x - diff[0]//2))

        #print('box shape', gt_bboxes.shape)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            if ann['masks'] is None:
                gt_masks = np.zeros((num_people, img_scale[1], img_scale[0]), dtype=np.uint8)
                has_masks = np.zeros((num_people, ), dtype=np.int64)
                #print('has no mask!')
            else:
                #print(ann['masks'].shape, pad_shape)
                gt_masks = self.mask_transform(ann['masks'], pad_shape, scale_factor, flip)
                padded_mask = np.zeros((gt_masks.shape[0], img_scale[1], img_scale[0]), dtype=gt_masks.dtype)
                padded_mask[:, start_pos_x:start_pos_x+gt_masks.shape[-2], start_pos_y:start_pos_y+gt_masks.shape[-1]] = gt_masks
                #padded_mask[:, :gt_masks.shape[-2], :gt_masks.shape[-1]] = gt_masks
                gt_masks = padded_mask
                has_masks = np.ones((gt_masks.shape[0], ), dtype=np.int64)

                # output_path = "./debug/coco_mask.jpg"
                # # cv2.imwrite(output_path, gt_masks[0].astype("uint8") * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                # cv2.imwrite(output_path, ann['masks'][0].astype("uint8") * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                # print("gt_masks.shape: ", gt_masks.shape)
                # print("ann['masks'].shape: ", ann['masks'].shape)
                # print(output_path)
                # print('has mask!')

        if self.with_shape:
            gt_shapes = ann['shape']
        if self.with_kpts2d:
            gt_kpts2d = ann['kpts2d']
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
                if self.dataset_name == 'h36m':
                    gender = ann['genders'][0]
                    pose_hand_theta = np.zeros([1, 2, 3]).astype(np.float32)
                    gt_poses[:, -2:, :] = pose_hand_theta
                    outputs = self.smplr(betas=gt_shapes, body_pose=gt_poses[:, 1:, :], global_orient=gt_poses[:, :1, :], pose2rot=True, gender=gender)
                    gt_vertices = outputs.vertices.detach().numpy().astype(np.float32)
                    joints = outputs.joints.detach().numpy().astype(np.float32) # n, 24, 3
                    gt_kpts3d = np.ones((joints.shape[0], joints.shape[1], 4)).astype(np.float32)
                    gt_kpts3d[:, :, :3] = joints
                else:
                    # gender = 'n'
                    # print(f">>>>self.dataset_name: ", self.dataset_name)
                    gt_kpts3d = np.zeros((gt_poses.shape[0], 24, 4)).astype(np.float32)
                    gt_vertices = np.zeros((gt_poses.shape[0], 6890, 4)).astype(np.float32)
        
        if self.with_dp:
            dp_num_pts = ann['dp_num_pts']
            dp_x = ann['dp_x']
            dp_y = ann['dp_y']
            dp_U = ann['dp_U']
            dp_V = ann['dp_V']
            dp_I = ann['dp_I']
            dp_x = img_shape[1] - dp_x

        # occlude
        if occluder is not None:
            areas = np.abs((gt_bboxes[:,2] - gt_bboxes[:,0]) * (gt_bboxes[:,3] - gt_bboxes[:,1]))
            img = occlude_with_objects(img, occluder, areas, gt_kpts2d)

        # filter unmatch point
        if  self.dataset_name == "posetrack":
            gt_kpts2d[:, 20, :] *= 0 
            gt_kpts2d[:, 21, :] *= 0 

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
            # img_meta=DC(img_meta, cpu_only=True),
            img_metas=DC(img_metas, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes.astype(np.float32))))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
            data['has_masks'] = DC(to_tensor(has_masks))
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
        if self.with_kpts2d:
            # Filter out un-used kpts.
            kpts_filter = gt_kpts2d[..., -1].sum(-1) < 8
            if gt_kpts2d.shape[0] == 1:
                gt_kpts2d[kpts_filter] = 0
            data['gt_kpts2d'] = DC(to_tensor(gt_kpts2d))
        if self.with_kpts3d:
            data['gt_kpts3d'] = DC(to_tensor(gt_kpts3d))
        if self.with_pose:
            data['gt_poses'] = DC(to_tensor(gt_poses))
        if self.with_shape:
            data['gt_shapes'] = DC(to_tensor(gt_shapes))
        if self.with_trans:
            data['gt_trans'] = DC(to_tensor(gt_trans))
        if self.with_dp:
            data['dp_x'] = DC(to_tensor(dp_x))
            data['dp_y'] = DC(to_tensor(dp_y))
            data['dp_U'] = DC(to_tensor(dp_U))
            data['dp_V'] = DC(to_tensor(dp_V))
            data['dp_I'] = DC(to_tensor(dp_I))
            data['dp_num_pts'] = DC(to_tensor(dp_num_pts))
        if self.with_nr:
            data['scene'] = DC(to_tensor(padded_scene))
        data['has_smpl'] = DC(to_tensor(ann['has_smpl']))
        if self.mosh_path:
            sampled_idxs = np.array(random.sample(self.mosh_sample_list, 36))
            mosh_pose = torch.tensor(deepcopy(self.mosh_pose[sampled_idxs].astype(np.float32)))
            mosh_shape = torch.tensor(deepcopy(self.mosh_shape[sampled_idxs].astype(np.float32)))
            mosh_bs = mosh_shape.shape[0]
            mosh_pose_shape = torch.cat([batch_rodrigues(mosh_pose.view(-1, 3)).view(mosh_bs, -1), mosh_shape], dim=1)
            data['mosh'] = DC(mosh_pose_shape)

        # make 3dpw happy
        data['gt_scale'] = DC(to_tensor(np.zeros((gt_trans.shape[0], 3)).astype(np.float32))) # p 3

        if self.with_camera_trans and self.dataset_name == 'h36m':
            gt_camera_trans = ann['camera_trans']
            if flip:
                gt_camera_trans[:,0] *= -1. # is it x?
            # gt_kpts3d和gt_camera_trans都是focal_len 1000下的，所以也要做scale
            gt_camera_trans = gt_camera_trans + gt_kpts3d[:, 14, :3]
            gt_camera_trans[:, -1] /= scale_factor
            gt_camera_trans = gt_camera_trans - gt_kpts3d[:, 14, :3]

            # print("H36M gt_camera_trans[:, -1]: ", gt_camera_trans[:, -1])
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

            # filter min trans
            if np.min(gt_camera_trans[:, -1]) < 1.0:
                return None

            data['gt_camera_trans'] = DC(to_tensor(gt_camera_trans)) # p 3
            data['has_trans'] = DC(to_tensor(np.ones((gt_bboxes.shape[0],)).astype(np.int_)))
            data['gt_vertices'] = DC(to_tensor(gt_vertices))
        else:
            # don't use global coords
            data['gt_camera_trans'] = DC(to_tensor(np.zeros_like(gt_trans).astype(np.float32))) # p 3
            data['has_trans'] = DC(to_tensor(np.zeros((gt_bboxes.shape[0],)).astype(np.int_)))
            if self.dataset_name == 'h36m':
                data['gt_vertices'] = DC(to_tensor(gt_vertices))
            else:
                data['gt_vertices'] = DC(to_tensor(np.zeros((gt_bboxes.shape[0], 6890, 3)).astype(np.int_)))
        
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


        # if self.is_stageII == True and self.dataset_name == 'h36m': 
        if self.dataset_name == 'h36m': 
            body_depthmap = to_tensor(body_depthmap)
            body_depthmap[torch.isnan(body_depthmap)] = 0.0
            data['gt_depthmap'] = DC(body_depthmap)
            
            if self.is_stageII == True:
                data['has_depthmap'] = DC(to_tensor(np.ones(1).astype(np.int64)))
            else:
                data['has_depthmap'] = DC(to_tensor(np.zeros(1).astype(np.int64))) # stageI not predict depthmap
        else:
            body_depthmap = np.zeros((1, img_scale[1], img_scale[0]), dtype=np.float32)
            body_depthmap = to_tensor(body_depthmap)
            data['gt_depthmap'] = DC(body_depthmap)
            data['has_depthmap'] = DC(to_tensor(np.zeros(1).astype(np.int64)))

        if self.dataset_name == 'h36m': 
            #### mask from depthmap
            gt_masks = np.zeros_like(body_depthmap).astype(np.uint8)
            gt_masks[body_depthmap > 0] = 1
            gt_masks = np.repeat(gt_masks, num_people, axis=0)
            data['gt_masks'] = DC(gt_masks.astype(np.uint8), cpu_only=True)
            data['has_masks'] = DC(to_tensor(np.ones((num_people,)).astype(np.int64)))
            # data['gt_masks'] = DC(gt_masks.astype(np.uint8), cpu_only=True)
            # data['has_masks'] = DC(to_tensor(np.ones(gt_poses.shape[0]).astype(np.int64)))

            # gt_masks_vis = gt_masks[0][:, :, None] * 255
            # output_path = "./debug/gt_masks.jpg"
            # cv2.imwrite(output_path, gt_masks_vis)
            # print(output_path)
            # print(img_path, flip)
        elif not self.with_mask:
            data['gt_masks'] = DC(np.zeros((num_people, img_scale[1], img_scale[0]), dtype=np.uint8), cpu_only=True)
            data['has_masks'] = DC(to_tensor(np.zeros((num_people, )).astype(np.int64)))
        
        # #############################
        # ### 可视化 dataloader data ###
        # #############################
        # # debug visual gt_depth_mask
        # print(body_depthmap.shape)
        # gt_mask_np = body_depthmap[0].cpu().numpy()
        # bits = 1
        # depth_min = gt_mask_np.min()
        # depth_max = gt_mask_np.max()
        # max_val = (2 ** (8 * bits)) - 1
        # if depth_max - depth_min > np.finfo("float").eps:
        #     out = max_val * (gt_mask_np - depth_min) / (depth_max - depth_min)
        # else:
        #     out = np.zeros(gt_mask_np.shape, dtype=gt_mask_np.dtype)
        # output_path = "./debug/gt_depthmap_debug_h36m.png"

        # img_vis = cv2.cvtColor((denormalize(img.transpose([1, 2, 0])) * 255).astype(np.uint8).copy(), cv2.COLOR_BGR2RGB)
        # depthmap = out.astype("uint8")[:, :, None]
        # depthmap = cv2.cvtColor(depthmap, cv2.COLOR_GRAY2RGB)
        # print(depthmap.shape)
        # img_vis = cv2.addWeighted(img_vis, 0.5, depthmap, 0.5, 0)
        # cv2.imwrite(output_path, img_vis, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # print("gt_depthmap_debug_h36m.shape: ", gt_mask_np.shape)
        # print(output_path)

        # debug_filename = osp.basename(ann['filename'])
        # if self.dataset_name == 'h36m':
        #     if self.with_camera_trans:
        #         img_viz = self.prepare_dump(data)
        #     else:
        #         img_viz = self.prepare_dump_stageI(data)
        #     # _, H, W = img.shape
        #     # camMat = np.array([[1000, 0, W/2],
        #     #         [0, 1000, H/2],
        #     #         [0, 0, 1]]).astype(np.float32)
        #     # RT = np.concatenate((np.diag([1., 1., 1.]), np.zeros((3, 1))), axis=1)
        #     # gt_kpts3d_global = gt_kpts3d[:, :, :3] + gt_camera_trans[:, None, :]
        #     # kpts3d_F1000_proj = np.zeros((gt_kpts3d.shape[0], gt_kpts3d.shape[1], 2))
        #     # for i in range(gt_kpts3d.shape[0]):
        #     #     for j in range(gt_kpts3d.shape[1]):
        #     #         kpts3d_F1000_proj[i, j, :] = project_point_np(np.concatenate([gt_kpts3d_global[i, j, :], np.array([1])]), RT, camMat)
            
        #     for i in range(gt_kpts2d.shape[0]):
        #         # img_viz = draw_point(img_viz, kpts3d_F1000_proj[i], color=(0, 255, 0))
        #         img_viz = draw_point(img_viz, gt_kpts2d[i], color=(0, 0, 255))

        #     for i in range(gt_bboxes.shape[0]):
        #         bbox = gt_bboxes[i]
        #         img_viz = cv2.rectangle(img_viz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            
        #     output_path = f"./debug/debug_h36m_vis/{self.dataset_name}_{debug_filename}_scale_factor:{augment_scale_factor}.jpg".replace(" ", "")
        #     cv2.imwrite(output_path, img_viz[:, :, :])
        #     print(img_path)
        #     print(output_path)
        #     print(img_viz.shape)

        # else:
        #     # img = np.float32(img.cpu().numpy())
        #     img = img.transpose([1, 2, 0])
        #     img = cv2.cvtColor((denormalize(img) * 255).astype(np.uint8).copy(), cv2.COLOR_BGR2RGB)

        #     for i in range(gt_kpts2d.shape[0]):
        #         img = draw_point(img, gt_kpts2d[i], color=(0, 0, 255))

        #     for i in range(gt_bboxes.shape[0]):
        #         bbox = gt_bboxes[i]
        #         img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            
        #     output_path = f"./debug/debug_h36m_vis/{self.dataset_name}_{debug_filename}_scale_factor:{augment_scale_factor}.jpg".replace(" ", "")
        #     cv2.imwrite(output_path, img[:, :, :])
        #     print(img_path)
        #     print(output_path)
        #     print(img.shape)

        # print("h36m: ", sorted(data.keys()), len(sorted(data.keys())))
        # print("-" * 50)

        return data


    def prepare_dump(self, data):
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
        total_img = cv2.cvtColor((denormalize(total_img) * 255).astype(np.uint8).copy(), cv2.COLOR_BGR2RGB)

        return total_img



    def prepare_dump_stageI(self, data):
        from mmdet.models.utils.smpl.renderer import Renderer
        from mmdet.models.utils.smpl.smpl import SMPL

        gt_bboxes = data['gt_bboxes'].data
        gt_trans = data['gt_trans'].data                    
        gt_camera_trans = data['gt_camera_trans'].data     
        gt_poses = data['gt_poses'].data
        gt_shapes = data['gt_shapes'].data
        img = data['img'].data

        gt_trans = torch.zeros_like(gt_trans)
        gt_camera_trans = torch.zeros_like(gt_camera_trans)
        gt_camera_trans[:, 0] = 1
        
        _, H, W = img.shape
        FOCAL_LENGTH = 1000
        img_size = torch.zeros(1, 2).to(gt_camera_trans.device)
        img_size += torch.tensor(img.shape[:-3:-1], dtype=gt_camera_trans.dtype).to(gt_camera_trans.device)

        center_pts = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2
        bboxes_size = torch.max(torch.abs(gt_bboxes[..., 0] - gt_bboxes[..., 2]), torch.abs(gt_bboxes[..., 1] - gt_bboxes[..., 3]))
        
        # FOCAL_LENGTH = FOCAL_LENGTH * (bboxes_size / 256) 
        depth = 2 * FOCAL_LENGTH / (1e-6 + gt_camera_trans[..., 0] * bboxes_size)
        render_translation = torch.zeros((1, 3), dtype=gt_camera_trans.dtype).to(gt_camera_trans.device)
        render_translation[:, :-1] = depth * (center_pts + gt_camera_trans[:, 1:] * bboxes_size.unsqueeze(-1) - img_size / 2) / FOCAL_LENGTH
        render_translation[:, -1] = depth
       
        render = Renderer(focal_length=FOCAL_LENGTH, height=H, width=W)

        # from scipy.spatial.transform import Rotation as R
        # smpl = SMPL('data/smpl')
        # output = smpl(betas=gt_shapes, body_pose=gt_poses[:, 1:], global_orient=gt_poses[:, :1], transl=gt_trans, pose2rot=True)
        # verts = output.vertices
        verts = data['gt_vertices'].data
        
        try:
            fv_rendered = render([img.clone()], [verts], translation=[render_translation])[0]
            bv_rendered = self.renderer_bv(img, verts, render_translation, gt_bboxes[0], FOCAL_LENGTH, render)
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
