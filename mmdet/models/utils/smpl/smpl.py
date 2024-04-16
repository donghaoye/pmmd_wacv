import torch
import torch.nn as nn
import numpy as np
import smplx
from smplx import SMPL as _SMPL
from smplx.body_models import SMPLXOutput
from smplx.lbs import vertices2joints

joint_names = [
    # 24 Ground Truth joints (superset of joints from different datasets)
    'Right Ankle',
    'Right Knee',
    'Right Hip',
    'Left Hip',
    'Left Knee',
    'Left Ankle',
    'Right Wrist',
    'Right Elbow',
    'Right Shoulder',
    'Left Shoulder',
    'Left Elbow',
    'Left Wrist',
    'Neck (LSP)',
    'Top of Head (LSP)',
    'Pelvis (MPII)',
    'Thorax (MPII)',
    'Spine (H36M)',
    'Jaw (H36M)',
    'Head (H36M)',
    'Nose',
    'Left Eye',
    'Right Eye',
    'Left Ear',
    'Right Ear'
]
joint_map = {
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, joint_regressor_extra='data/J_regressor_extra.npy', **kwargs):
        super(SMPL, self).__init__(*args, create_global_orient=False, create_body_pose=True, create_betas=True, create_transl=False, **kwargs)
        # super(SMPL, self).__init__(*args, create_global_orient=True, create_body_pose=True, create_betas=True, create_transl=True, **kwargs)
        joints = [joint_map[i] for i in joint_names]
        J_regressor_extra = np.load(joint_regressor_extra)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True

        #print('before', kwargs['body_pose'].shape)

        # if 'pose2rot' in kwargs and kwargs['pose2rot']:
        #     if len(kwargs['body_pose'].shape) == 2:
        #         kwargs['body_pose'] = torch.cat([kwargs['body_pose'], torch.zeros(kwargs['body_pose'].shape[0],6).to(kwargs['body_pose'].device)], 1) 
        #     else:
        #         kwargs['body_pose'] = torch.cat([kwargs['body_pose'], torch.zeros(kwargs['body_pose'].shape[0],2,3).to(kwargs['body_pose'].device)], 1) 

        '''
        ******************************
        !!!!!after torch.Size([1, 23, 3]) torch.Size([1, 10]) torch.Size([1, 1, 3])
        !!!!!after torch.Size([26, 23, 3, 3]) torch.Size([26, 10]) torch.Size([26, 1, 3, 3])
        !!!!!after torch.Size([33, 23, 3, 3]) torch.Size([33, 10]) torch.Size([33, 1, 3, 3])
        !!!!!after torch.Size([29, 23, 3, 3]) torch.Size([29, 10]) torch.Size([29, 1, 3, 3])
        !!!!!after torch.Size([1, 23, 3]) torch.Size([8, 10]) torch.Size([1, 1, 3])
        '''
        # print('!!!!!after', kwargs['body_pose'].shape, kwargs['betas'].shape, kwargs['global_orient'].shape, kwargs['pose2rot'])
        # print('!!!!!after', kwargs['body_pose'].shape, kwargs['betas'].shape, kwargs['global_orient'].shape)
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = SMPLXOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output

class SMPLR(nn.Module):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, joint_regressor_h36m='data/J_regressor_h36m.npy', use_gender=False, **kwargs):
        super(SMPLR, self).__init__()
 
        #H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
        #J_regressor_h36m17 = np.load(J_reg_h36m17_path)[H36M_TO_J17]
        #J_regressor_h36m17 = to_tensor(to_np(J_regressor_h36m17), dtype=dtype)
        #self.register_buffer('J_regressor_h36m17', J_regressor_h36m17) 
        if use_gender:
            self.smpl_female = SMPL(*args, gender='female', **kwargs)
            self.smpl_male = SMPL(*args, gender='male', **kwargs)
            self.smpls = {'f':self.smpl_female, 'm':self.smpl_male}
        else:
            self.smpl_neutral = SMPL(*args, gender='neutral', **kwargs)
            self.smpls = {'n':self.smpl_neutral}

    def forward(self, *args, gender='n', **kwargs):
        betas = torch.tensor(kwargs.get('betas'))
        body_pose = torch.tensor(kwargs.get('body_pose'))
        global_orient = torch.tensor(kwargs.get('global_orient'))
        #print(betas.shape, body_pose.shape, global_orient.shape)
        kwargs.update({'betas': betas, 'body_pose': body_pose, 'global_orient': global_orient})

        outputs = self.smpls[gender](*args, **kwargs)
        return outputs

        # j = outputs.joints # n, 24, 3
        # j3d = torch.ones(j.shape[0], j.shape[1], 1, dtype=j.dtype)
        # j3d = torch.cat((j, j3d), dim=-1)
        # return j3d
