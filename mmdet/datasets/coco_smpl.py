import os.path as osp
from copy import deepcopy
import mmcv
import lap
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale, flip_kp, flip_pose
from pycocotools.coco import COCO
from .extra_aug import ExtraAugmentation
from .custom import CustomDataset
import os.path as osp
# import tensorboardX
import math
import json
import pickle
import matplotlib.pyplot as plt
from mmdet.models.utils.smpl.viz import draw_skeleton, J24_TO_J14
import random
import cv2
import torch
from .transforms import coco17_to_superset
# from .h36m import H36MDataset
# from .h36m import denormalize

import mmdet.datasets.constants as constants
from .common import CommonDataset


FLOAT_DTYPE = np.float32
INT_DTYPE = np.int64

from .builder import DATASETS

@DATASETS.register_module()
class COCO_SMPL(CommonDataset):
    def __init__(self,
                 **kwargs,
                 ):
        super(COCO_SMPL, self).__init__(**kwargs)
        # if self.extra_annot_path is not None:
        #     print('begin loading')
        #     self.extra_smpl_gt = self.load_extra_annotations(self.extra_annot_path)
        #     print('loaded!!!!')
        # else:
        #     self.extra_smpl_gt = None

        self.torso_ids = [constants.OUR_SMPL_24[part] for part in ['Neck', 'R_Shoulder', 'L_Shoulder','Pelvis', 'R_Hip', 'L_Hip']]


    def get_extra_annotations(self, img_name, kp2ds):
        if 'pretrain' == img_name[:8]:
            img_name = img_name.split('_')[-1]
            img_name = 'COCO_train2014_' + img_name
            
        if self.extra_smpl_gt is not None and img_name in self.extra_smpl_gt:
            eft_annot = self.extra_smpl_gt[img_name].copy()
            bbox_center_list, pose_list, betas_list = [], [], []
            for bbox_center, pose, betas in eft_annot:
                bbox_center_list.append(bbox_center)
                #pose_list.append(pose[:66])
                pose_list.append(pose[:72])
                betas_list.append(betas)
            bbox_center_list = np.array(bbox_center_list)

            cdist = []
            for kp2d in kp2ds:
                c = self._calc_center_(kp2d)
                if c is None:
                    continue
                cdist.append(np.linalg.norm(bbox_center_list-c[:2][None], axis=-1))
                #cdist = np.array([np.linalg.norm(bbox_center_list-self._calc_center_(kp2d)[:2][None], axis=-1) for kp2d in kp2ds])
            
            if len(cdist) == 0:
                return None
            
            cdist = np.array(cdist)

            matches = []
            cost, x, y = lap.lapjv(cdist, extend_cost=True)
            for ix, mx in enumerate(x):
                if mx >= 0:
                    matches.append([ix, mx])
            matches = np.asarray(matches)
            
            picked_pose, picked_beta, has_smpl = np.zeros((len(kp2ds), 72)), np.zeros((len(kp2ds), 10)), np.zeros(len(kp2ds),)
            for kid, pid in matches:
                picked_pose[kid] = pose_list[pid]
                picked_beta[kid] = betas_list[pid]
                has_smpl[kid] = 1.
                #print('matched!')

            return dict(pose=np.array(picked_pose).reshape(len(kp2ds), -1, 3).astype(FLOAT_DTYPE), shape=np.array(picked_beta).astype(FLOAT_DTYPE), has_smpl=has_smpl.astype(np.int64)) # should be: np, c
        else:
            return None
