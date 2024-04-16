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

import torch
import joblib
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import pickle

class AMASS(object):
    def __init__(self):
        self.seqlen = 1

        self.stride = 1

        self.db = self.load_db()
        self.total_num = self.db['theta'].shape[0]
        self.total_range = range(self.total_num)
        print('total AMASSS data:', self.total_num)
        #self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)
        del self.db['vid_name']
        #print(f'AMASS dataset number of videos: {len(self.vid_indices)}')

    def __len__(self):
        return len(self.vid_indices)

    # def __getitem__(self, index):
    #     return self.get_single_item(index)

    def batch_random_get(self, batch_size):
        indexes = np.random.choice(self.total_range, batch_size, replace=False)
        thetas = self.db['theta'][indexes]
        return thetas

    def load_db(self):
        db_file = osp.join('/data/Mono3DPerson/VIBE/data/vibe_db', 'amass_db.pt')
        db = joblib.load(db_file)
        return db

    # def get_single_item(self, index):
    #     start_index, end_index = self.vid_indices[index]
    #     thetas = self.db['theta'][start_index:end_index+1]

    #     cam = np.array([1., 0., 0.])[None, ...]
    #     cam = np.repeat(cam, thetas.shape[0], axis=0)
    #     theta = np.concatenate([cam, thetas], axis=-1)

    #     target = {
    #         'theta': torch.from_numpy(theta).float(),  # cam, pose and shape
    #     }
    #     return target


class H36M(object):
    def __init__(self, ann_file):
        self.img_infos = self.load_annotations(ann_file)
        # use only 0.1 for complexity issue
        random_idx = list(np.random.choice(len(self.img_infos), int(len(self.img_infos)/10), replace=False))
        #self.img_infos = self.img_infos[random_idx]
        self.total_range = range(len(random_idx))
        self.shape = []
        self.pose = []
        for idx in random_idx:
            self.shape.append(self.img_infos[idx]['betas'][0].astype(np.float32))
            self.pose.append(self.img_infos[idx]['pose'][0].astype(np.float32))
        self.shape = np.array(self.shape)
        self.pose = np.array(self.pose)
        del self.img_infos

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

    def batch_random_get(self, batch_size):
        indexes = np.random.choice(self.total_range, batch_size, replace=False)
        thetas = self.shape[indexes]
        return thetas



