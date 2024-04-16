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
class LSPET(CommonDataset):
    def __init__(self,
                 **kwargs,
                 ):
        super(LSPET, self).__init__(**kwargs)
        # if self.extra_annot_path is not None:
        #     self.extra_smpl_gt = self.load_extra_annotations(self.extra_annot_path)
        # else:
        #     self.extra_smpl_gt = None

    def get_extra_annotations(self, img_name, kp2ds):
        img_name = img_name.split('_')[-1]
        if self.extra_smpl_gt is not None and img_name in self.extra_smpl_gt:
            eft_annot = self.extra_smpl_gt[img_name].copy()
            bbox_center, pose, betas = eft_annot[0]
            
            return dict(pose=np.array(pose[:72]).reshape(-1,3)[None,...].astype(FLOAT_DTYPE), shape=np.array(betas[:10])[None,...].astype(FLOAT_DTYPE), has_smpl=np.ones((1,)).astype(np.int64)) # should be: np, c
        else:
            return None
