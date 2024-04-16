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
from .h36m import H36MDataset

from .h36m import denormalize

import mmdet.datasets.constants as constants
from .common import CommonDataset


FLOAT_DTYPE = np.float32
INT_DTYPE = np.int64


from .builder import DATASETS

@DATASETS.register_module()
class MPII(CommonDataset):
    def __init__(self,
                 **kwargs,
                 ):
        super(MPII, self).__init__(**kwargs)
        # if self.extra_annot_path is not None:
        #     self.extra_smpl_gt = self.load_extra_annotations(self.extra_annot_path)
        # else:
        #     self.extra_smpl_gt = None

        self.torso_ids = [constants.OUR_SMPL_24[part] for part in ['Neck', 'R_Shoulder', 'L_Shoulder','Pelvis', 'R_Hip', 'L_Hip']]


    def get_extra_annotations(self, img_name, kp2ds):
        if self.extra_smpl_gt is not None and img_name in self.extra_smpl_gt:
            eft_annot = self.extra_smpl_gt[img_name].copy()
            bbox_center_list, pose_list, betas_list = [], [], []
            for bbox_center, pose, betas in eft_annot:
                bbox_center_list.append(bbox_center)
                pose_list.append(pose[:72])
                #pose_list.append(pose[:66])
                betas_list.append(betas)
            bbox_center_list = np.array(bbox_center_list)

            picked_pose, picked_beta, has_smpl = [], [], np.ones(len(kp2ds),)
            matched = 0
            for inds, kp2d in enumerate(kp2ds):
                center_i = self._calc_center_(kp2d)
                if center_i is None:
                    has_smpl[inds] *= 0.
                    picked_pose.append(np.zeros((24,3)))
                    picked_beta.append(np.zeros((10,)))
                    continue
                matched += 1
                center_dist = np.linalg.norm(bbox_center_list-center_i[:2][None], axis=-1)
                closet_idx = np.argmin(center_dist)
                picked_pose.append(pose_list[closet_idx].reshape(-1, 3))
                picked_beta.append(betas_list[closet_idx])
                #matched_param = np.concatenate([pose_list[closet_idx], betas_list[closet_idx]])
                #params.append(matched_param)
            if matched == 0:
                return None
            #print('yeah!!')
            return dict(pose=np.array(picked_pose).astype(FLOAT_DTYPE), shape=np.array(picked_beta).astype(FLOAT_DTYPE), has_smpl=has_smpl.astype(np.int64)) # should be: np, c
        else:
            return None
