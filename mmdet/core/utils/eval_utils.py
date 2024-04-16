import os.path as osp
import torch
import torchvision
from mmdet.models.utils.smpl_utils import batch_rodrigues, J24_TO_J14, H36M_TO_J14, J24_TO_H36M, ADS24_TO_LSP14
from mmdet.models.utils.pose_utils import reconstruction_error, vectorize_distance
from mmdet.core.utils import AverageMeter
#from mmdet.models.utils.camera import PerspectiveCamera
from mmdet.models.utils.smpl.renderer import Renderer
from mmdet.models.utils.smpl.smpl import SMPL#, JointMapper
#from smplx.body_models import SMPL#, JointMapper
from mmdet.models.utils.smpl.viz import draw_skeleton, get_bv_verts, plot_pose_H36M
import cv2
import matplotlib.pyplot as plt
import numpy as np
import abc
import math
import os
# import h5py
import scipy.io as scio
# import joblib
#from sdf import CollisionVolume
# from mmdet.models.losses.romp_loss import calc_mpjpe, calc_pampjpe
import mmdet.datasets.constants as constants

from mmdet.models.utils.smpl_utils import rotation_matrix_to_angle_axis, rot6d_to_rotmat, batch_rodrigues
# from mmdet.datasets.utils import project_point_np, draw_point

import time


denormalize = lambda x: x * np.array([0.229, 0.224, 0.225])[None, None, :] + np.array([0.485, 0.456, 0.406])[None, None, :]





def get_center_point(kpts2d):
    xmin = np.min(kpts2d[:,:,0], axis=1)
    ymin = np.min(kpts2d[:,:,1], axis=1)
    xmax = np.max(kpts2d[:,:,0], axis=1)
    ymax = np.max(kpts2d[:,:,1], axis=1)
    bbox_center = []
    for i in range(kpts2d.shape[0]):
        bbox_center.append([(xmin[i] + xmax[i]) / 2, (ymin[i] + ymax[i]) / 2])
    center_points = np.asarray(bbox_center).astype(np.float32)

    return center_points

def prepare_dump(data, pred_vertices, gt_verts, pred_trans_tmp, gt_trans_tmp, img, FOCAL_LENGTH=1000):
    # gt_bboxes = data['gt_bboxes'].data
    # gt_trans = data['gt_trans'].data                     # 000
    # # gt_camera_trans = data['gt_camera_trans'].data[0][0]       # 3.88, 
    # gt_poses = data['gt_poses'].data
    # gt_shapes = data['gt_shapes'].data
    # img = data['img'].data[0][0]
    img = img[:, :, ::-1]
    img = img.transpose([2, 0, 1]) / 255.0
    img = torch.from_numpy(img)
    pred_trans = torch.from_numpy(pred_trans_tmp)
    gt_trans = torch.from_numpy(gt_trans_tmp)

    _, H, W = img.shape
    # FOCAL_LENGTH = 1000
    render = Renderer(focal_length=FOCAL_LENGTH, height=H, width=W)
    try:
        pred_rendered = render([img.clone()], [pred_vertices], translation=[pred_trans])[0]
        # gt_rendered = render([img.clone()], [gt_verts], translation=[gt_camera_trans])[0]
        gt_rendered = render([img.clone()], [gt_verts], translation=[gt_trans], colors=[(.7, .7, .6, 1.)])[0]
    except Exception as e:
        print(e)
        return None

    img = np.float32(img.cpu().numpy())
    total_img = np.zeros((3 * H, W, 3))
    total_img[:H] += img.transpose([1, 2, 0])
    total_img[H:2 * H] += pred_rendered.transpose([1, 2, 0])
    total_img[2 * H:] += gt_rendered.transpose([1, 2, 0])
    # total_img = cv2.cvtColor((denormalize(total_img) * 255).astype(np.uint8).copy(), cv2.COLOR_BGR2RGB)
    total_img = cv2.cvtColor((total_img * 255).astype(np.uint8).copy(), cv2.COLOR_BGR2RGB)

    return total_img

def calc_gpve(gt_verts, pred_verts, mode='min'):
    assert len(pred_verts.shape) == 3 and len(gt_verts.shape) == 3

    if mode == 'min':
        gpve = np.mean(np.sqrt(np.sum((gt_verts - pred_verts) ** 2, axis=2)), axis=1).min() * 1000.
    elif mode == 'mean':
        gpve = np.mean(np.sqrt(np.sum((gt_verts - pred_verts) ** 2, axis=2)), axis=1).mean() * 1000.
    else:
        gpve = np.mean(np.sqrt(np.sum((gt_verts - pred_verts) ** 2, axis=2)), axis=1) * 1000.
    return gpve

def calc_avg_gpve(gt_verts, pred_verts, mode='min'):
    assert len(pred_verts.shape) == 3 and len(gt_verts.shape) == 3

    if mode == 'min':
        avg_gpve = np.sqrt(np.sum((np.mean(gt_verts, axis=1) - np.mean(pred_verts, axis=1)) ** 2, axis=-1)).min() * 1000.
    elif mode == 'mean':
        avg_gpve = np.sqrt(np.sum((np.mean(gt_verts, axis=1) - np.mean(pred_verts, axis=1)) ** 2, axis=-1)).mean() * 1000.
    else:
        avg_gpve = np.sqrt(np.sum((np.mean(gt_verts, axis=1) - np.mean(pred_verts, axis=1)) ** 2, axis=-1)) * 1000.
    return avg_gpve


def calc_gce(gt_kp, pred_kp, pelvis_index, mode='min'):
    

    if pelvis_index is not None:
        assert len(pred_kp.shape) == 3 and len(gt_kp.shape) == 3
        gt_kp, pred_kp = gt_kp[:,pelvis_index,:], pred_kp[:,pelvis_index,:]

    if mode == 'min':
        gce = np.sqrt(np.sum((gt_kp - pred_kp) ** 2, axis=-1)).min() * 1000.
    elif mode == 'mean':
        gce = np.sqrt(np.sum((gt_kp - pred_kp) ** 2, axis=-1)).mean() * 1000.
    else:
        gce = np.sqrt(np.sum((gt_kp - pred_kp) ** 2, axis=-1)) * 1000.
    
    return gce



class Converter():
    def __init__(self):
        pass

    def batch_orth_proj(self, X, camera, mode='2d',keep_dim=False):
        camera = camera.view(-1, 1, 3)
        X_camed = X[:,:,:2] * camera[:, :, 0].unsqueeze(-1)
        X_camed += camera[:, :, 1:]
        if keep_dim:
            X_camed = torch.cat([X_camed, X[:,:,2].unsqueeze(-1)],-1)
        return X_camed

    def get_romp_camera_verts(self, cam, verts):
        cam = torch.tensor(cam)

        verts = torch.tensor(verts)

        verts = self.batch_orth_proj(verts, cam, mode='3d',keep_dim=True)

        verts[:,:,-1] += 4.
        return verts

    # for 3dpw only
    def align(self, img_size, camera_verts):
        #img_size = cv2.imread(img_path) / 255.
        if img_size[0] == 1920:
            # align x only
            scale = 1920 / 512
            width = 1080 / scale
            # print(camera_verts.shape,'*****')
            pixel_offset = 832//2 - width//2
            if pixel_offset != 0:
                camera_offset = (camera_verts[:,:,-1] * pixel_offset) / 1000.
                camera_verts[:,:,0] -= camera_offset
        else:
            # more complex

            # align depth
            camera_verts[:,:,-1] /= (832 / 512)     

            # no need to align x

            # align y
            scale = 1920 / 832
            height = 1080 / scale
            pixel_offset = 512//2 - height//2
            if pixel_offset != 0:
                camera_offset = (camera_verts[:,:,-1] * pixel_offset) / 1000.
                camera_verts[:,:,1] -= camera_offset  

        return camera_verts

    def get_ours_camera_verts(self, img_size, cam, verts):
        verts = self.get_romp_camera_verts(cam, verts=verts)
        verts = self.align(img_size, verts)
        return verts


def compute_scale_transform(S1, S2):
    '''
    Computes a scale transform (s, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    t 3x1 translation, s scale.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    # 2. Compute variance of X1,X2 used for scale.
    var1 = np.sum(X1 ** 2)
    var2 = np.sum(X2 ** 2)

    # 3. Recover scale.
    scale = np.sqrt(var2 / var1)

    # 4. Recover translation.
    t = mu2 - scale * (mu1)

    # 5. Error:
    S1_hat = scale * S1 + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_scale_transform_batch(S1, S2, visibility):
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i, visibility[i]] = compute_scale_transform(S1[i, visibility[i]], S2[i, visibility[i]])
    return S1_hat


def draw_text(input_image, content):
    """
    content is a dict. draws key: val on image
    Assumes key is str, val is float
    """
    image = input_image.copy()
    input_is_float = False
    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        image = (image * 255).astype(np.uint8)

    black = np.array([0, 0, 255])
    margin = 45
    start_x = 15
    start_y = margin
    for key in sorted(content.keys()):
        text = f"{key}: {content[key]}"
        image = cv2.putText(image, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        start_y += margin

    if input_is_float:
        image = image.astype(np.float32) / 255.
    return image


class EvalHandler(metaclass=abc.ABCMeta):
    def __init__(self, writer=print, log_every=1, viz_dir='', FOCAL_LENGTH=1000, work_dir=''):
        self.call_cnt = 0
        self.log_every = log_every
        self.writer = writer
        self.viz_dir = viz_dir
        self.work_dir = work_dir
        #self.camera = PerspectiveCamera(FOCAL_LENGTH=FOCAL_LENGTH)
        self.FOCAL_LENGTH = FOCAL_LENGTH
        if self.viz_dir:
            self.renderer = Renderer(focal_length=FOCAL_LENGTH)
        else:
            self.renderer = None

    def __call__(self, *args, **kwargs):
        self.call_cnt += 1
        res = self.handle(*args, **kwargs)
        if self.log_every > 0 and (self.call_cnt % self.log_every == 0):
            self.log()
        return res

    @abc.abstractmethod
    def handle(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def log(self):
        pass

    def finalize(self):
        pass


class H36MEvalHandler(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', pattern='.60457274_', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.pattern = pattern
        self.p1_meter = AverageMeter('P1', ':.2f')
        # self.p2_meter = AverageMeter('P2', ':.2f')
        self.gpve_meter = AverageMeter('gpve', ':.2f')
        self.avg_gpve_meter = AverageMeter('avg_gpve', ':.2f')
        self.gce_meter = AverageMeter('gce', ':.2f')
        self.papve_meter = AverageMeter('papve', ':.2f')
        self.smpl = SMPL(
                'data/smpl'
            ).to('cpu')
        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/h36m_train_ads_zy.txt', 'w+')
        self.logger = open('output/h36m_train_ads_zy.txt', 'w+')


    def handle(self, data_batch, pred_results, use_gt=False):
        pred_vertices = pred_results['pred_vertices'].cpu()
        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)

        # gt_shapes = data_batch['gt_shapes'].data[0][0].clone()
        # gt_shapes = gt_shapes.repeat([pred_num_people, 1])
        # gt_poses = data_batch['gt_poses'].data[0][0].clone()
        # gt_poses = gt_poses.view(gt_poses.shape[0], -1)
        # gt_poses = gt_poses.repeat([pred_num_people, 1])
        # gt_keypoints_3d, gt_verts = self.get_gt_joints(gt_shapes,
        #                                                gt_poses,
        #                                                J_regressor_batch)

        # # 
        # pred_keypoints_3d, pred_verts = self.get_gt_joints(gt_shapes,
        #                                                gt_poses,
        #                                                J_regressor_batch)

        # # this is the offcial kp3ds
        # # gt_keypoints_3d = data_batch['gt_kpts3d'].data[0][0].clone().repeat([pred_vertices.shape[0], 1, 1])
        # # gt_pelvis_smpl = gt_keypoints_3d[:, [14], :-1].clone()
        # # gt_keypoints_3d = gt_keypoints_3d[:, J24_TO_J14, :-1].clone()
        # # gt_keypoints_3d = gt_keypoints_3d - gt_pelvis_smpl

        # # this is got from smpl
        # gt_pelvis = (gt_keypoints_3d[:,[2],:] + gt_keypoints_3d[:,[3],:]) / 2.0
        # gt_keypoints_3d -= gt_pelvis

        # J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
        #     pred_vertices.device)
        pred_num_people = pred_vertices.shape[0]
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        # provided way
        # Get 14 predicted joints from the SMPL mesh
        # pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
        # pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        # pred_keypoints_3d_smpl = pred_keypoints_3d_smpl - pred_pelvis_smpl

        # vibe way
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_pelvis = (pred_keypoints_3d_smpl[:,[2],:] + pred_keypoints_3d_smpl[:,[3],:]) / 2.0
        pred_keypoints_3d_smpl -= pred_pelvis

        gt_verts = data_batch['gt_vertices'].data[0][0] # already inference with genders
        gt_J_regressor_batch = self.J_regressor[None, :].to(gt_verts.device)
        gt_keypoints_3d_smpl = torch.matmul(gt_J_regressor_batch, gt_verts)
        gt_keypoints_3d_smpl = gt_keypoints_3d_smpl[:, H36M_TO_J14, :]
        gt_pelvis = (gt_keypoints_3d_smpl[:,[2],:] + gt_keypoints_3d_smpl[:,[3],:]) / 2.0
        gt_keypoints_3d_smpl -= gt_pelvis
        gt_keypoints_3d_smpl = gt_keypoints_3d_smpl.repeat([pred_num_people, 1, 1])


        # # Compute error metrics
        # # Absolute error (MPJPE)
        # error_smpl = torch.sqrt(((pred_keypoints_3d_smpl - gt_keypoints_3d_smpl) ** 2).sum(dim=-1)).mean(dim=-1)
        # mpjpe = float(error_smpl.min() * 1000)
        # self.p1_meter.update(mpjpe)

        # pa mpjpe
        #print(gt_keypoints_3d.shape)
        S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d_smpl)
        # errors_pa = torch.sqrt(((S1_hat.double() - gt_keypoints_3d.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        errors_pa_tmp = torch.sqrt(((S1_hat.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1))
        # print(errors_pa_tmp.shape)
        errors_pa = errors_pa_tmp.mean(dim=-1)
        # print(errors_pa.shape)
        min_idx = torch.argmin(errors_pa, dim=0)
        pampjpe  = float(errors_pa.min() * 1000)
        self.p1_meter.update(pampjpe)

        # pa-pve
        S1_hat = batch_compute_similarity_transform_torch(pred_vertices[min_idx].unsqueeze(0), gt_verts)
        # S1_hat = batch_compute_similarity_transform_torch(pred_vertices, gt_verts.repeat([pred_num_people, 1, 1]))
        errors_pa = np.mean(np.sqrt(np.sum((S1_hat.detach().cpu().numpy() - gt_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        self.papve_meter.update(errors_pa)

        # if self.pattern in file_name:
        #     # Reconstruction error
        #     r_error_smpl = reconstruction_error(pred_keypoints_3d_smpl.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
        #                                         reduction=None)
        #     r_error = float(r_error_smpl.min() * 1000)
        #     self.p2_meter.update(r_error)
        # else:
        #     r_error = -1

        pred_trans = pred_results['pred_translation'].squeeze(1).detach().cpu().numpy()  # n 3
        
        gt_trans = data_batch['gt_camera_trans'].data[0][0].clone()
        #print(gt_trans.shape)
        #print(gt_trans)

        gt_trans = gt_trans.repeat([pred_num_people, 1, 1]).detach().cpu().numpy()  # n ,3
        gt_verts = gt_verts.repeat([pred_num_people, 1, 1]).detach().cpu().numpy()  # n ,3
        
        # 可视化 pre_verts和gt_verts
        # print("pred_vertices.shape, gt_verts.shape: ", pred_vertices.shape, gt_verts.shape)
        # img_viz = prepare_dump(data_batch, pred_vertices, gt_verts, pred_trans)
        # cv2.imwrite(f"./debug_eval_imgs/h36m_eval_visual_{file_name}.jpg", img_viz[:, :, :])
        # print("smpl img_viz", img_viz.shape, file_name)

        # # GPVE
        # gt_verts = gt_verts.detach().cpu().numpy()
        # gt_verts += gt_trans#[:,None]
        # pred_vertices = pred_vertices.detach().cpu().numpy()

        # pred_vertices += pred_trans[:, None]
        # gpve = np.mean(np.sqrt(np.sum((gt_verts - pred_vertices) ** 2, axis=2)), axis=1).min() * 1000.
        # self.gpve_meter.update(gpve)

        # # gce
        # gce = np.sqrt(np.sum((np.mean(gt_verts, axis=1) - np.mean(pred_vertices, axis=1)) ** 2, axis=-1)).min() * 1000.
        # self.gce_meter.update(gce)

       
        pred_vertices = pred_vertices.detach().cpu().numpy()
        pred_vertices += pred_trans[:, None]
        gt_verts += gt_trans

        gpve = calc_gpve(gt_verts, pred_vertices)
        self.gpve_meter.update(gpve)

        avg_gpve = calc_avg_gpve(gt_verts, pred_vertices)
        self.avg_gpve_meter.update(avg_gpve)

        gt_pelvis += gt_trans
        gt_pelvis = gt_pelvis.detach().cpu().numpy()

        pred_pelvis += pred_trans[:, None]
        pred_pelvis = pred_pelvis.detach().cpu().numpy()

        gce = calc_gce(gt_pelvis[:,0], pred_pelvis[:,0], pelvis_index=None)
        self.gce_meter.update(gce)

        for kk in range(gt_trans.shape[0]):
            self.logger.write('%.2f_%.2f\n' % (gt_trans[kk,0,1], gt_trans[kk,0,2]))
            # print('%.2f_%.2f\n' % (gt_trans[kk,0,1], gt_trans[kk,0,2]))

        file_name = data_batch['img_metas'].data[0][0]['file_name']
        file_name = os.path.basename(file_name)[:-4]
        save_pack = {'file_name': file_name,
                     'pampjpe': pampjpe,
                     'gce':  gce,
                     'avg_gpve': avg_gpve,
                     'gpve': gpve,
                     #'r_error': r_error,
                    #  'pred_rotmat': pred_results['pred_rotmat'],
                    #  'pred_betas': pred_results['pred_betas'],
                     }
        return save_pack

    def log(self):
        self.writer(f'pampjpe: {self.p1_meter.avg:.2f}mm, papve: {self.papve_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm avg_gpve: {self.avg_gpve_meter.avg:.2f} h36m')

    def get_gt_joints(self, gt_shape, gt_pose, J_regressor_batch):
        # print("H36MEvalHandler get_gt_joints gt_pose.shape", gt_pose.shape)
        # batch_size = gt_pose.shape[0]
        # gt_pose_hand = torch.zeros(batch_size, 12).to(gt_pose.device)
        # gt_pose_hand_mat = rot6d_to_rotmat(gt_pose_hand).view(batch_size, 2, 3, 3)
        # gt_pose_hand_theta = rotation_matrix_to_angle_axis(gt_pose_hand_mat).view(batch_size, -1)
        # gt_pose_hand_theta = print(gt_pose[:, -6:].shape, gt_pose_hand_theta.shape)
        # # gt_pose_hand_theta = torch.zeros(batch_size, 6).to(gt_pose.device)
        # gt_pose[:, -6:] = gt_pose_hand_theta

        output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3], pose2rot=True)
        #output = smpl(betas=gt_shape, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3], transl=gt_trans)
        verts = output.vertices

        j3d = torch.matmul(J_regressor_batch, verts)
        j3d = j3d[:, H36M_TO_J14, :]
        #print(j3d.shape, verts.shape)
        return j3d, verts



class H36MEvalHandlerROMPBEV(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', pattern='.60457274_', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.pattern = pattern
        self.p1_meter = AverageMeter('P1', ':.2f')
        self.p2_meter = AverageMeter('P2', ':.2f')
        self.gpve_meter = AverageMeter('gpve', ':.2f')
        self.avg_gpve_meter = AverageMeter('avg_gpve', ':.2f')
        self.gce_meter = AverageMeter('gce', ':.2f')
        self.papve_meter = AverageMeter('papve', ':.2f')
        self.smpl = SMPL(
                'data/smpl'
            ).to('cpu')
        self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/h36m_train_ads_zy.txt', 'w+')


    def handle(self, data_batch, pred_results, use_gt=False):
        pred_center_preds = pred_results['pred_center_preds'].float().detach().cpu().numpy()
        gt_center_preds = pred_results['gt_center_preds'].float().detach().cpu().numpy()
        pred_vertices = pred_results['pred_vertices'].cpu()
        pred_trans = pred_results['pred_translation'].squeeze(1).detach().cpu().numpy()  # n 3
        gt_trans = data_batch['gt_camera_trans'].data[0][0].clone().detach().cpu().numpy()

        # for ROMP and BEV 应该选择center map最近的pair
        # pred_bbox 与 gt_bbox距离
        dist_array = np.linalg.norm(pred_center_preds - gt_center_preds, axis=1) 
        min_idx = np.argmin(dist_array)
        # print(dist_array)
        # print(min_idx)
        pred_vertices = pred_vertices[min_idx:min_idx+1]
        pred_trans = pred_trans[min_idx:min_idx+1]

        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)

        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_pelvis = (pred_keypoints_3d_smpl[:,[2],:] + pred_keypoints_3d_smpl[:,[3],:]) / 2.0
        pred_keypoints_3d_smpl -= pred_pelvis

        gt_verts = data_batch['gt_vertices'].data[0][0] # already inference with genders
        gt_J_regressor_batch = self.J_regressor[None, :].to(gt_verts.device)
        gt_keypoints_3d_smpl = torch.matmul(gt_J_regressor_batch, gt_verts)
        gt_keypoints_3d_smpl = gt_keypoints_3d_smpl[:, H36M_TO_J14, :]
        gt_pelvis = (gt_keypoints_3d_smpl[:,[2],:] + gt_keypoints_3d_smpl[:,[3],:]) / 2.0
        gt_keypoints_3d_smpl -= gt_pelvis

        # Compute error metrics
        # Absolute error (MPJPE)
        error_smpl = torch.sqrt(((pred_keypoints_3d_smpl - gt_keypoints_3d_smpl) ** 2).sum(dim=-1)).mean(dim=-1)
        mpjpe = float(error_smpl.min() * 1000)
        self.p1_meter.update(mpjpe)

        # pa mpjpe
        S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d_smpl)
        pampjpe = torch.sqrt(((S1_hat.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        pampjpe = float(pampjpe * 1000)
        self.p2_meter.update(pampjpe)

        # pa-pve
        S1_hat = batch_compute_similarity_transform_torch(pred_vertices, gt_verts)
        errors_pa = torch.sqrt(((S1_hat.double() - gt_verts.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        # errors_pa = np.mean(np.sqrt(np.sum((S1_hat.detach().cpu().numpy() - gt_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        errors_pa = float(errors_pa * 1000)
        self.papve_meter.update(errors_pa)

        gt_verts = gt_verts.detach().cpu().numpy()
        gt_verts += gt_trans[:, None]

        pred_vertices = pred_vertices.detach().cpu().numpy()
        pred_vertices += pred_trans[:, None]
        
        # gpve
        gpve = calc_gpve(gt_verts, pred_vertices)
        self.gpve_meter.update(gpve)

        # avg_gpve
        avg_gpve = calc_avg_gpve(gt_verts, pred_vertices)
        self.avg_gpve_meter.update(avg_gpve)

        # gce
        gt_pelvis += gt_trans[:, None]
        gt_pelvis = gt_pelvis.detach().cpu().numpy()
        pred_pelvis += pred_trans[:, None]
        pred_pelvis = pred_pelvis.detach().cpu().numpy()
        gce = calc_gce(gt_pelvis[:,0], pred_pelvis[:,0], pelvis_index=None)
        self.gce_meter.update(gce)

        # for kk in range(gt_trans.shape[0]):
        #     self.logger.write('%.2f_%.2f\n' % (gt_trans[kk,0,1], gt_trans[kk,0,2]))
        #     # print('%.2f_%.2f\n' % (gt_trans[kk,0,1], gt_trans[kk,0,2]))

        file_name = data_batch['img_metas'].data[0][0]['file_name']
        file_name = os.path.basename(file_name)[:-4]
        save_pack = {'file_name': file_name,
                     'pampjpe': pampjpe,
                     'gce':  gce,
                     'avg_gpve': avg_gpve,
                     'gpve': gpve,
                     #'r_error': r_error,
                    #  'pred_rotmat': pred_results['pred_rotmat'],
                    #  'pred_betas': pred_results['pred_betas'],
                     }
        return save_pack

    def log(self):
        self.writer(f'mpjpe: {self.p1_meter.avg:.2f}mm, pampjpe: {self.p2_meter.avg:.2f}mm, papve: {self.papve_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm avg_gpve: {self.avg_gpve_meter.avg:.2f} h36m')

    def get_gt_joints(self, gt_shape, gt_pose, J_regressor_batch):
        # print("H36MEvalHandler get_gt_joints gt_pose.shape", gt_pose.shape)
        # batch_size = gt_pose.shape[0]
        # gt_pose_hand = torch.zeros(batch_size, 12).to(gt_pose.device)
        # gt_pose_hand_mat = rot6d_to_rotmat(gt_pose_hand).view(batch_size, 2, 3, 3)
        # gt_pose_hand_theta = rotation_matrix_to_angle_axis(gt_pose_hand_mat).view(batch_size, -1)
        # gt_pose_hand_theta = print(gt_pose[:, -6:].shape, gt_pose_hand_theta.shape)
        # # gt_pose_hand_theta = torch.zeros(batch_size, 6).to(gt_pose.device)
        # gt_pose[:, -6:] = gt_pose_hand_theta

        output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3], pose2rot=True)
        #output = smpl(betas=gt_shape, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3], transl=gt_trans)
        verts = output.vertices

        j3d = torch.matmul(J_regressor_batch, verts)
        j3d = j3d[:, H36M_TO_J14, :]
        #print(j3d.shape, verts.shape)
        return j3d, verts


class OriginH36MEvalHandler(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.p1_meter = AverageMeter('P1', ':.2f')
        # self.p2_meter = AverageMeter('P2', ':.2f')
        self.gpve_meter = AverageMeter('gpve', ':.2f')
        self.gce_meter = AverageMeter('gce', ':.2f')
        self.papve_meter = AverageMeter('papve', ':.2f')
        self.smpl = SMPL(
                'data/smpl'
            ).to('cpu')

    def handle(self, data_batch, pred_results, use_gt=False):
        pred_vertices = pred_results['pred_vertices'].cpu()
        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
            pred_vertices.device)

        # pred_num_people = pred_vertices.shape[0]
        # if gt_keypoints_3d.shape[0] > 1:
        pred_num_people = 1

        gt_shapes = data_batch['gt_shapes'].data[0][0].clone()
        gt_shapes = gt_shapes.repeat([pred_num_people, 1])

        gt_poses = data_batch['gt_poses'].data[0][0].clone()
        gt_poses = gt_poses.view(gt_poses.shape[0], -1)
        gt_poses = gt_poses.repeat([pred_num_people, 1])
        #print(gt_shapes.shape, gt_poses.shape)
        gt_keypoints_3d, gt_verts = self.get_gt_joints(gt_shapes,
                                                       gt_poses,
                                                       J_regressor_batch)


        # this is the offcial kp3ds
        gt_keypoints_3d = data_batch['gt_kpts3d'].data[0][0].clone().repeat([pred_vertices.shape[0], 1, 1])
        gt_pelvis_smpl = gt_keypoints_3d[:, [14], :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[:, J24_TO_J14, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis_smpl

        # this is got from smpl
        # gt_pelvis = (gt_keypoints_3d[:,[2],:] + gt_keypoints_3d[:,[3],:]) / 2.0
        # gt_keypoints_3d -= gt_pelvis



        # J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
        #     pred_vertices.device)

        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        # provided way
        # Get 14 predicted joints from the SMPL mesh
        pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl - pred_pelvis_smpl

        # vibe way
        # pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        # pred_pelvis = (pred_keypoints_3d_smpl[:,[2],:] + pred_keypoints_3d_smpl[:,[3],:]) / 2.0
        # pred_keypoints_3d_smpl -= pred_pelvis

        file_name = data_batch['img_metas'].data[0][0]['file_name']

        # Compute error metrics
        # Absolute error (MPJPE)
        # error_smpl = torch.sqrt(((pred_keypoints_3d_smpl - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(
        #     dim=-1)
        # mpjpe = float(error_smpl.min() * 1000)
        # self.p1_meter.update(mpjpe)

        # pa mpjpe

        #print(gt_keypoints_3d.shape)
        S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d.repeat([pred_num_people, 1, 1]))
        errors_pa = torch.sqrt(((S1_hat.double() - gt_keypoints_3d.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        min_idx = torch.argmin(errors_pa, dim=0)
        pampjpe  = float(errors_pa.min() * 1000)
        self.p1_meter.update(pampjpe)

        # pa-pve
        S1_hat = batch_compute_similarity_transform_torch(pred_vertices[min_idx].unsqueeze(0), gt_verts)
        errors_pa = np.mean(np.sqrt(np.sum((S1_hat.detach().cpu().numpy() - gt_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        self.papve_meter.update(errors_pa)

        # if self.pattern in file_name:
        #     # Reconstruction error
        #     r_error_smpl = reconstruction_error(pred_keypoints_3d_smpl.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
        #                                         reduction=None)
        #     r_error = float(r_error_smpl.min() * 1000)
        #     self.p2_meter.update(r_error)
        # else:
        #     r_error = -1

        pred_trans = pred_results['pred_translation'].squeeze(1).detach().cpu().numpy()  # n 3
        
        gt_trans = data_batch['gt_camera_trans'].data[0][0].clone()
        #print(gt_trans.shape)
        #print(gt_trans)
        gt_trans = gt_trans.repeat([pred_num_people, 1, 1]).detach().cpu().numpy()  # n ,3
        

        # GPVE
        gt_verts = gt_verts.detach().cpu().numpy()
        gt_verts += gt_trans#[:,None]
        pred_vertices = pred_vertices.detach().cpu().numpy()

        pred_vertices += pred_trans[:, None]
        gpve = np.mean(np.sqrt(np.sum((gt_verts - pred_vertices) ** 2, axis=2)), axis=1).min() * 1000.
        self.gpve_meter.update(gpve)

        # gce
        gce = np.sqrt(np.sum((np.mean(gt_verts, axis=1) - np.mean(pred_vertices, axis=1)) ** 2, axis=-1)).min() * 1000.
        self.gce_meter.update(gce)

        save_pack = {'file_name': file_name,
                     'pampjpe': pampjpe,
                     'gce':  gce,
                     'gpve': gpve,
                     #'r_error': r_error,
                     'pred_rotmat': pred_results['pred_rotmat'],
                     'pred_betas': pred_results['pred_betas'],
                     }
        return save_pack

    def log(self):
        self.writer(f'pampjpe: {self.p1_meter.avg:.2f}mm, papve: {self.papve_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm h36m')

    def get_gt_joints(self, gt_shape, gt_pose, J_regressor_batch):
        # print("OriginH36MEvalHandler get_gt_joints gt_pose.shape", gt_pose.shape)
        output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3], pose2rot=True)
        #output = smpl(betas=gt_shape, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3], transl=gt_trans)
        verts = output.vertices

        j3d = torch.matmul(J_regressor_batch, verts)
        j3d = j3d[:, H36M_TO_J14, :]
        #print(j3d.shape, verts.shape)
        return j3d, verts


class ThreeDPWHandler(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', pattern='.60457274_', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.pattern = pattern
        self.p1_meter = AverageMeter('P1', ':.3f')
        self.p2_meter = AverageMeter('P2', ':.3f')
        self.p3_meter = AverageMeter('P3', ':.3f')
        self.papve_meter = AverageMeter('papve', ':.3f')

        self.avg_gpve_meter = AverageMeter('avg_gpve', ':.3f')
        self.gce_meter = AverageMeter('P3', ':.3f')
        self.gpve_meter = AverageMeter('P3', ':.3f')
        self.saved = False
        self.smpl = SMPL(
                'data/smpl'
            ).to('cpu')

        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/threedpw_train_ads_zy.txt', 'w+')
        self.logger = open('./output/threedpw_train_ads_zy.txt', 'w+')

    def get_gt_joints(self, gt_shape, gt_pose, J_regressor_batch):
        # print("ThreeDPWHandler get_gt_joints gt_pose.shape", gt_pose.shape)
        # ThreeDPWHandler get_gt_joints gt_pose.shape torch.Size([1, 24, 3])

        # 应该分gender smplr, todo...
        output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 1:], global_orient=gt_pose[:, :1], pose2rot=True)
        # output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3], pose2rot=True)
        #output = smpl(betas=gt_shape, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3], transl=gt_trans)
        verts = output.vertices

        # 这里可视化 verts 看看
        

        j3d = torch.matmul(J_regressor_batch, verts)
        j3d = j3d[:, H36M_TO_J14, :]
        #print(j3d.shape, verts.shape)
        return j3d, verts

    def handle(self, data_batch, pred_results, use_gt=False):
        gt_num_people = data_batch['gt_kpts3d'].data[0][0].shape[0]
        file_name = data_batch['img_metas'].data[0][0]['file_name']
        metric_values = {
                     'MPJPE': [],
                     'pve': [],
                     'PAMPJPE': [],
                     # global metrics
                     'gce': [],
                     'gpve': []}

        for pidx in range(gt_num_people):
            packs = self.handle_single(data_batch, pred_results, pidx, use_gt=use_gt)
            for k in packs:
                metric_values[k].append(packs[k])

        for k in metric_values:
            metric_values[k] = np.mean(metric_values[k])

        metric_values.update({
            'file_name': file_name,
            # 'pred_rotmat': pred_results['pred_rotmat'],
            # 'pred_betas': pred_results['pred_betas'],
        })

        return metric_values

    def handle_single(self, data_batch, pred_results, pidx, use_gt=False):
        gt_num_people = data_batch['gt_kpts3d'].data[0][0].shape[0]
        #print('gt num people', data_batch['gt_kpts3d'].data[0][0].shape, gt_num_people)
        o_device = pred_results['pred_vertices'].device
        pred_vertices = pred_results['pred_vertices'].cpu()
        pred_trans = pred_results['pred_translation'].squeeze(1).detach().cpu().numpy()  # n 3
        
        pred_num_people = pred_vertices.shape[0]
        # print('>>>>>!!!', pred_trans.shape)
        #pred_vertices = pred_vertices.repeat([gt_num_people, 1, 1])
        #print('???', pred_vertices.shape)

        #J_regressor_batch = self.J_regressor[None, :].expand(gt_num_people, -1, -1).to(
        #    pred_vertices.device)
        J_regressor_batch = self.J_regressor[None, :].to(pred_vertices.device)

        # if not os.path.isfile('test_pred.npy'):
        #     save_pred = {}
        #     save_pred['pred_vertices'] = pred_results['pred_vertices'].cpu().numpy()
        #     save_pred['pred_translation'] = pred_trans
        #     np.save('test_pred.npy', save_pred)
        #     save_gt = {}
        #     save_gt['gt_shapes'] = data_batch['gt_shapes'].data[0][0].detach().clone().cpu().numpy()
        #     save_gt['gt_poses'] = data_batch['gt_poses'].data[0][0].detach().clone().cpu().numpy()
        #     save_gt['gt_camera_trans'] = data_batch['gt_camera_trans'].data[0][0].clone().detach().cpu().numpy() 
        #     np.save('test_gt.npy', save_gt)


        # gt_keypoints_3d, gt_verts = self.get_gt_joints(data_batch['gt_shapes'].data[0][0].clone()[pidx].unsqueeze(0), 
        #                                                data_batch['gt_poses'].data[0][0].clone()[pidx].unsqueeze(0),
        #                                                J_regressor_batch)
        # #print(gt_keypoints_3d.shape, gt_verts.shape)
        # #gt_verts = gt_verts.repeat([pred_num_people, 1, 1])
        # gt_pelvis = (gt_keypoints_3d[:,[2],:] + gt_keypoints_3d[:,[3],:]) / 2.0
        # gt_keypoints_3d -= gt_pelvis
        #gt_keypoints_3d = gt_keypoints_3d.repeat([pred_num_people, 1, 1])

        # if pred_num_people == 1 and gt_num_people == 1 and not self.saved:
        #     print('gt_verts:', gt_verts.shape)
        #     np.save('/home/tiange/work/playground/gt_verts.npy', gt_verts.detach().cpu().numpy())
        #     print('pred_verts:', pred_vertices.shape)
        #     np.save('/home/tiange/work/playground/pred_verts.npy', pred_vertices.numpy())
        #     self.saved = True
        
        # Get 14 predicted joints from the SMPL mesh
        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
        
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_pelvis = (pred_keypoints_3d_smpl[:,[2],:] + pred_keypoints_3d_smpl[:,[3],:]) / 2.0
        pred_keypoints_3d_smpl -= pred_pelvis

        gt_verts = data_batch['gt_vertices'].data[0][0][pidx:pidx+1] # already inference with genders
        gt_J_regressor_batch = self.J_regressor[None, :].to(gt_verts.device)
        gt_keypoints_3d_smpl = torch.matmul(gt_J_regressor_batch, gt_verts)
        gt_keypoints_3d_smpl = gt_keypoints_3d_smpl[:, H36M_TO_J14, :]
        gt_pelvis = (gt_keypoints_3d_smpl[:,[2],:] + gt_keypoints_3d_smpl[:,[3],:]) / 2.0
        gt_keypoints_3d_smpl -= gt_pelvis

        gt_keypoints_3d_smpl = gt_keypoints_3d_smpl.repeat([pred_num_people, 1, 1])

        # # 可视化 pre_verts和gt_verts
        # print("pred_vertices.shape, gt_verts.shape: ", pred_vertices.shape, gt_verts.shape)
        # img_viz = prepare_dump(data_batch, pred_vertices, gt_verts, pred_trans)
        # cv2.imwrite(f"./output/threedpw_eval_trainset.jpg", img_viz[:, :, :])
        # print("smpl img_viz", img_viz.shape)

        # Compute error metrics
        # Absolute error (MPJPE)
        error_smpl = torch.sqrt(((pred_keypoints_3d_smpl.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        mpjpe = float(error_smpl.min() * 1000)
        self.p1_meter.update(mpjpe)

        # pampjpe
        # print(pred_keypoints_3d_smpl.shape, gt_keypoints_3d.shape)
        # S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d_smpl.repeat([pred_num_people, 1, 1]))
        S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d_smpl)
        errors_pa = torch.sqrt(((S1_hat.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        # min_idx = np.argmin(errors_pa, axis=0)
        pampjpe  = float(errors_pa.min() * 1000)
        self.p2_meter.update(pampjpe)

        # # pve
        # pve = np.mean(np.sqrt(np.sum((gt_verts.detach().cpu().numpy() - pred_vertices.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        # #pve = float(np.mean(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        # #pve = float(self.compute_error_verts(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        # self.p3_meter.update(pve)

        # pve
        pve = np.mean(np.sqrt(np.sum((gt_verts.detach().cpu().numpy() - pred_vertices.detach().cpu().numpy()) ** 2, axis=2, keepdims=False)), axis=1, keepdims=False)
        min_idx = np.argmin(pve, axis=0)
        pve = pve.min() * 1000
        #pve = float(np.mean(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        #pve = float(self.compute_error_verts(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        self.p3_meter.update(pve)

        # pa-pve
        # min_idx from pa-mpjpe
        S1_hat = batch_compute_similarity_transform_torch(pred_vertices[min_idx:min_idx+1], gt_verts)
        # S1_hat = batch_compute_similarity_transform_torch(pred_vertices, gt_verts.repeat([pred_num_people, 1, 1]))
        errors_pa = np.mean(np.sqrt(np.sum((S1_hat.detach().cpu().numpy() - gt_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        self.papve_meter.update(errors_pa)


        #########global metrics########
        gt_trans = data_batch['gt_camera_trans'].data[0][0].clone()[pidx].unsqueeze(0).detach().cpu().numpy()  # n ,3


        # test depth only
        # gt_depth = torch.zeros_like(gt_trans)
        # gt_depth[:,-1] = gt_trans[:,-1]
        # gt_trans = gt_depth
        # pred_real_depth = torch.zeros_like(pred_trans)
        # pred_real_depth[:,-1] = pred_depth[:,-1]
        # pred_trans = pred_real_depth


        # if pred_num_people != gt_num_people:
        #     gt_trans = gt_trans.repeat([pred_num_people, 1]).detach().cpu().numpy() 
        #     pred_trans = pred_trans.repeat([gt_num_people, 1]).detach().cpu().numpy() 
        # else:
        #     gt_trans = gt_trans.detach().cpu().numpy() 
        #     pred_trans = pred_trans.detach().cpu().numpy() 

        # # GPVE
        # gt_verts = gt_verts.detach().cpu().numpy()
        # gt_verts += gt_trans[:,None]
        # pred_vertices = pred_vertices.detach().cpu().numpy()

        # # if scale available
        # if False: #'pred_scale' in pred_results:
        #     pred_scale = pred_results['pred_scale'].repeat([gt_num_people, 1]).detach().cpu().numpy()
        #     norm_vert = pred_vertices - np.mean(pred_vertices, axis=1, keepdims=True)
        #     vert_max = np.max(norm_vert, axis=1, keepdims=False) # p 3
        #     vert_min = np.min(norm_vert, axis=1, keepdims=False) # p 3
        #     norm_scale = vert_max - vert_min
        #     multiplier = pred_scale / norm_scale
        #     #print('**', pred_scale.shape, norm_scale.shape)
        #     #scale_multiplier = pred_scale / norm_scale
        #     pred_vertices *= multiplier[:, None]

        # pred_vertices += pred_trans[:, None]
        # gpve = np.mean(np.sqrt(np.sum((gt_verts - pred_vertices) ** 2, axis=2)), axis=1).min() * 1000
        # self.gpve_meter.update(gpve)

        # gce
        #gce = np.sqrt(np.sum((np.mean(gt_verts, axis=1) - np.mean(pred_vertices, axis=1)) ** 2, axis=-1)).min() * 1000.
        #self.gce_meter.update(gce)

        gt_verts = gt_verts.detach().cpu().numpy()
        gt_verts += gt_trans[:,None]

        pred_vertices = pred_vertices.detach().cpu().numpy()
        pred_vertices += pred_trans[:, None]

        gpve = calc_gpve(gt_verts, pred_vertices)
        self.gpve_meter.update(gpve)

        avg_gpve = calc_avg_gpve(gt_verts, pred_vertices)
        self.avg_gpve_meter.update(avg_gpve)

        gt_pelvis += gt_trans[:,None]
        gt_pelvis = gt_pelvis.detach().cpu().numpy()

        pred_pelvis += pred_trans[:, None]
        pred_pelvis = pred_pelvis.detach().cpu().numpy()

        gce = calc_gce(gt_pelvis[:,0], pred_pelvis[:,0], pelvis_index=None)
        self.gce_meter.update(gce)

        for kk in range(gt_trans.shape[0]):
            self.logger.write('%.2f_%.2f\n' % (gt_trans[kk,1], gt_trans[kk,2]))
            # print('%.2f_%.2f\n' % (gt_trans[kk,1], gt_trans[kk,2]))

        save_pack = {#'file_name': file_name,
                     'MPJPE': mpjpe,
                     'pve': pve,
                     'PAMPJPE': pampjpe,
                     #'pred_rotmat': pred_results['pred_rotmat'],
                     #'pred_betas': pred_results['pred_betas'],
                     # global metrics
                     
                     'gce': gce,
                     'gpve': gpve,
                     #'avg_gpve': avg_gpve,
                     }
        return save_pack

    def compute_error_verts(self, pred_verts, target_shape=None, target_pose=None, target_verts=None):
        """
        Computes MPJPE over 6890 surface vertices.
        Args:
            verts_gt (Nx6890x3).
            verts_pred (Nx6890x3).
        Returns:
            error_verts (N).
        """

        if target_verts is None:

            betas = target_shape
            pose = target_pose

            target_verts = []
            b_ = torch.split(betas, 5000)
            p_ = torch.split(pose, 5000)

            for b,p in zip(b_,p_):
                #print(b.shape, p.shape)
                print("ThreeDPWHandler compute_error_verts p.shape", p.shape)
                output = self.smpl(betas=b, body_pose=p[:, 1:], global_orient=p[:, :1], pose2rot=True)
                # output = self.smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], pose2rot=True)
                target_verts.append(output.vertices.detach().cpu().numpy())

            target_verts = np.concatenate(target_verts, axis=0)

        assert len(pred_verts) == len(target_verts)
        error_per_vert = np.sqrt(np.sum((target_verts - pred_verts.detach().cpu().numpy()) ** 2, axis=2))
        return np.mean(error_per_vert, axis=1)


    def log(self):
        # self.writer(f'mpjpe: {self.p1_meter.avg:.2f}mm, pampjpe: {self.p2_meter.avg:.2f}mm, pve: {self.p3_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm, avg_gpve: {self.avg_gpve_meter.avg:.2f}mm 3dpw')
        self.writer(f'mpjpe: {self.p1_meter.avg:.2f}mm, pampjpe: {self.p2_meter.avg:.2f}mm, pve: {self.p3_meter.avg:.2f}mm, papve: {self.papve_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm, avg_gpve: {self.avg_gpve_meter.avg:.2f}mm 3dpw')



class ThreeDPWHandlerCenterPoint(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', pattern='.60457274_', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.pattern = pattern
        self.p1_meter = AverageMeter('P1', ':.3f')
        self.p2_meter = AverageMeter('P2', ':.3f')
        self.p3_meter = AverageMeter('P3', ':.3f')
        self.papve_meter = AverageMeter('papve', ':.3f')

        self.avg_gpve_meter = AverageMeter('avg_gpve', ':.3f')
        self.gce_meter = AverageMeter('P3', ':.3f')
        self.gpve_meter = AverageMeter('P3', ':.3f')
        self.saved = False
        self.smpl = SMPL(
                'data/smpl'
            ).to('cpu')

        self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/threedpw_train_ads_zy.txt', 'w+')

    def get_gt_joints(self, gt_shape, gt_pose, J_regressor_batch):
        # print("ThreeDPWHandler get_gt_joints gt_pose.shape", gt_pose.shape)
        # ThreeDPWHandler get_gt_joints gt_pose.shape torch.Size([1, 24, 3])

        # 应该分gender smplr, todo...
        output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 1:], global_orient=gt_pose[:, :1], pose2rot=True)
        # output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3], pose2rot=True)
        #output = smpl(betas=gt_shape, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3], transl=gt_trans)
        verts = output.vertices

        # 这里可视化 verts 看看
        

        j3d = torch.matmul(J_regressor_batch, verts)
        j3d = j3d[:, H36M_TO_J14, :]
        #print(j3d.shape, verts.shape)
        return j3d, verts

    def handle(self, data_batch, pred_results, use_gt=False):
        gt_num_people = data_batch['gt_kpts3d'].data[0][0].shape[0]
        file_name = data_batch['img_metas'].data[0][0]['file_name']
        metric_values = {
                     'MPJPE': [],
                     'pve': [],
                     'PAMPJPE': [],
                     # global metrics
                     'gce': [],
                     'gpve': []}

        for pidx in range(gt_num_people):
            packs = self.handle_single(data_batch, pred_results, pidx, use_gt=use_gt)
            if packs == None:
                continue
            for k in packs:
                metric_values[k].append(packs[k])

        for k in metric_values:
            metric_values[k] = np.mean(metric_values[k])

        metric_values.update({
            'file_name': file_name,
            # 'pred_rotmat': pred_results['pred_rotmat'],
            # 'pred_betas': pred_results['pred_betas'],
        })

        return metric_values

    def handle_single(self, data_batch, pred_results, pidx, use_gt=False):
        pred_verts = pred_results['pred_vertices'].cpu()
        pred_trans = pred_results['pred_translation'].squeeze(1).detach().cpu().numpy()  # n 3
        img = data_batch['img'].data[0][0]
        img_path = data_batch['img_metas'].data[0][0]['file_name']
        filename = osp.basename(img_path)
        gt_trans = data_batch['gt_camera_trans'].data[0][0].clone()[pidx].unsqueeze(0).detach().cpu().numpy()  # n ,3

        # pred_bbox 与 gt_bbox距离
        pred_bboxes = pred_results['bboxes'][0]
        gt_bboxes = data_batch['gt_bboxes'].data[0][0][pidx:pidx+1].detach().cpu().numpy()

        img = np.float32(img.cpu().numpy())
        _, H, W = img.shape
        img_viz = cv2.cvtColor((denormalize(img.transpose([1, 2, 0])) * 255).astype(np.uint8).copy(), cv2.COLOR_BGR2RGB)
        for i in range(pred_bboxes.shape[0]):
            bbox = pred_bboxes[i]
            img_viz = cv2.rectangle(img_viz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        # select high confidence
        # high_conf_mask = pred_bboxes[:, 4] > 0.8
        high_conf_mask = pred_bboxes[:, 4] > 0.3
        # print(len(high_conf_mask))
        if high_conf_mask.shape[0] > 0:
            pred_bboxes = pred_bboxes[high_conf_mask]
            pred_verts = pred_verts[high_conf_mask]
            pred_trans = pred_trans[high_conf_mask]

        # print(pred_bboxes.shape, pred_verts.shape, pred_trans.shape)
        if pred_bboxes.shape[0] == 0:
            return None

        pred_center_point_x = (pred_bboxes[:, 2] + pred_bboxes[:, 0]) * 0.5
        pred_center_point_y = (pred_bboxes[:, 3] + pred_bboxes[:, 1]) * 0.5
        pred_center_point = np.zeros((pred_bboxes.shape[0], 2)).astype(np.float32)
        pred_center_point[:, 0] = pred_center_point_x
        pred_center_point[:, 1] = pred_center_point_y

        gt_center_point_x = (gt_bboxes[:, 2] + gt_bboxes[:, 0]) * 0.5
        gt_center_point_y = (gt_bboxes[:, 3] + gt_bboxes[:, 1]) * 0.5
        gt_center_point = np.zeros((gt_bboxes.shape[0], 2)).astype(np.float32)
        gt_center_point[:, 0] = gt_center_point_x
        gt_center_point[:, 1] = gt_center_point_y

        dist_array = np.linalg.norm(pred_center_point - gt_center_point, axis=1) 
        min_idx = np.argmin(dist_array)
        # print(min_idx)
        # print(len(dist_array), pred_bboxes.shape, pred_verts.shape, pred_trans.shape)
        pred_verts = pred_verts[min_idx:min_idx+1]
        pred_trans = pred_trans[min_idx:min_idx+1]
        pred_bboxes = pred_bboxes[min_idx:min_idx+1]

        ################
        # visual start #
        ################
        pred_center_tuple = (pred_center_point[min_idx][0], pred_center_point[min_idx][1])
        gt_center_tuple = (gt_center_point[0][0], gt_center_point[0][1])
        img_viz = cv2.circle(img_viz, pred_center_tuple, 5, (0, 255, 0), -1)
        img_viz = cv2.circle(img_viz, gt_center_tuple, 5, (0, 0, 255), -1)
        for i in range(pred_bboxes.shape[0]):
            bbox = pred_bboxes[i]
            img_viz = cv2.rectangle(img_viz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 4)
        for i in range(gt_bboxes.shape[0]):
            bbox = gt_bboxes[i]
            img_viz = cv2.rectangle(img_viz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        
        # output_path = f"./debug/centerpoint_eval.jpg"
        # cv2.imwrite(output_path, img_viz[:, :, :])
        # print(output_path)

        # # 可视化 pre_verts和gt_verts
        gt_verts = data_batch['gt_vertices'].data[0][0][pidx:pidx+1] # already inference with genders
        # # print("pred_verts.shape, gt_verts.shape: ", pred_verts.shape, gt_verts.shape)
        # # print("pred_trans: ", pred_trans)

        # # in stageI, pred_trans is zero
        # FOCAL_LENGTH = 256/832 * 1000
        FOCAL_LENGTH = 1000
        # # print('pred_bboxes.shape: ', pred_bboxes.shape)
        # # bboxes_size = np.max(np.abs(pred_bboxes[:, 0] - pred_bboxes[:, 2]), np.abs(pred_bboxes[:, 1] - pred_bboxes[:, 3]), axis=0)
        # bboxes_size = max(np.abs(pred_bboxes[:, 0] - pred_bboxes[:, 2]), np.abs(pred_bboxes[:, 1] - pred_bboxes[:, 3]))
        # # print(bboxes_size)
        # # print(bboxes_size.shape)
        # bboxes_size = bboxes_size[0]
        # center_pts = (pred_bboxes[:, :2] + pred_bboxes[:, 2:4]) / 2
        # center_pts = torch.from_numpy(center_pts)
        # image_center_np = np.array([H, W]) * 0.5
        # image_center = torch.from_numpy(image_center_np)
        # pred_camera = pred_results['pred_camera'].cpu()[min_idx:min_idx+1]
        # # print('FOCAL_LENGTH: ', FOCAL_LENGTH)
        # depth = 2 * FOCAL_LENGTH / (pred_camera[:, 0] * bboxes_size + 1e-9)
        # depth = depth.unsqueeze(0)
        # render_translation = torch.zeros((1, 3), dtype=pred_camera.dtype).to(pred_camera.device)
        # render_translation[:, :-1] = pred_camera[:, 1:] + (center_pts - image_center) * depth / FOCAL_LENGTH
        # render_translation[:, -1] = depth[:, 0]

        # pred_translation = render_translation.cpu().numpy()
        # gt_translation = render_translation.cpu().numpy()

        pred_translation = pred_trans
        # pred_translation = gt_trans
        gt_translation = gt_trans

        img_viz = prepare_dump(data_batch, pred_verts, gt_verts, pred_translation, gt_translation, img_viz, FOCAL_LENGTH=FOCAL_LENGTH)
        output_path = f"./debug/centerpoint_mesh_{filename[:-4]}.jpg"
        cv2.imwrite(output_path, img_viz[:, :, :])
        print("smpl img_viz", img_viz.shape, output_path)
        ################
        # visual end   #
        ################

        # Get 14 predicted joints from the SMPL mesh
        J_regressor_batch = self.J_regressor[None, :].expand(pred_verts.shape[0], -1, -1).to(pred_verts.device)
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_verts)
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_pelvis = (pred_keypoints_3d_smpl[:,[2],:] + pred_keypoints_3d_smpl[:,[3],:]) / 2.0
        pred_keypoints_3d_smpl -= pred_pelvis

        gt_verts = data_batch['gt_vertices'].data[0][0][pidx:pidx+1] # already inference with genders
        gt_J_regressor_batch = self.J_regressor[None, :].to(gt_verts.device)
        gt_keypoints_3d_smpl = torch.matmul(gt_J_regressor_batch, gt_verts)
        gt_keypoints_3d_smpl = gt_keypoints_3d_smpl[:, H36M_TO_J14, :]
        gt_pelvis = (gt_keypoints_3d_smpl[:,[2],:] + gt_keypoints_3d_smpl[:,[3],:]) / 2.0
        gt_keypoints_3d_smpl -= gt_pelvis

        # Compute error metrics
        # Absolute error (MPJPE)
        error_smpl = torch.sqrt(((pred_keypoints_3d_smpl.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        mpjpe = float(error_smpl * 1000)
        self.p1_meter.update(mpjpe)

        # pampjpe
        S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d_smpl)
        errors_pa = torch.sqrt(((S1_hat.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        pampjpe  = float(errors_pa * 1000)
        self.p2_meter.update(pampjpe)

        # pve
        pve = np.mean(np.sqrt(np.sum((gt_verts.detach().cpu().numpy() - pred_verts.detach().cpu().numpy()) ** 2, axis=2, keepdims=False)), axis=1, keepdims=False)
        pve = pve.min() * 1000
        self.p3_meter.update(pve)

        # pa-pve
        # min_idx from pa-mpjpe
        S1_hat = batch_compute_similarity_transform_torch(pred_verts, gt_verts)
        errors_pa = np.mean(np.sqrt(np.sum((S1_hat.detach().cpu().numpy() - gt_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1) 
        errors_pa = errors_pa.min()* 1000
        self.papve_meter.update(errors_pa)

        #########global metrics########
        gt_verts = gt_verts.detach().cpu().numpy()
        gt_verts += gt_trans[:,None]

        pred_verts = pred_verts.detach().cpu().numpy()
        pred_verts += pred_trans[:, None]

        gpve = calc_gpve(gt_verts, pred_verts)
        self.gpve_meter.update(gpve)

        avg_gpve = calc_avg_gpve(gt_verts, pred_verts)
        self.avg_gpve_meter.update(avg_gpve)

        gt_pelvis += gt_trans[:,None]
        gt_pelvis = gt_pelvis.detach().cpu().numpy()

        pred_pelvis += pred_trans[:, None]
        pred_pelvis = pred_pelvis.detach().cpu().numpy()

        gce = calc_gce(gt_pelvis[:,0], pred_pelvis[:,0], pelvis_index=None)
        self.gce_meter.update(gce)

        for kk in range(gt_trans.shape[0]):
            self.logger.write('%.2f_%.2f\n' % (gt_trans[kk,1], gt_trans[kk,2]))
            # print('%.2f_%.2f\n' % (gt_trans[kk,1], gt_trans[kk,2]))

        save_pack = {#'file_name': file_name,
                     'MPJPE': mpjpe,
                     'pve': pve,
                     'PAMPJPE': pampjpe,
                     #'pred_rotmat': pred_results['pred_rotmat'],
                     #'pred_betas': pred_results['pred_betas'],
                     # global metrics
                     
                     'gce': gce,
                     'gpve': gpve,
                     #'avg_gpve': avg_gpve,
                     }
        return save_pack

    def compute_error_verts(self, pred_verts, target_shape=None, target_pose=None, target_verts=None):
        """
        Computes MPJPE over 6890 surface vertices.
        Args:
            verts_gt (Nx6890x3).
            verts_pred (Nx6890x3).
        Returns:
            error_verts (N).
        """

        if target_verts is None:

            betas = target_shape
            pose = target_pose

            target_verts = []
            b_ = torch.split(betas, 5000)
            p_ = torch.split(pose, 5000)

            for b,p in zip(b_,p_):
                #print(b.shape, p.shape)
                print("ThreeDPWHandler compute_error_verts p.shape", p.shape)
                output = self.smpl(betas=b, body_pose=p[:, 1:], global_orient=p[:, :1], pose2rot=True)
                # output = self.smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], pose2rot=True)
                target_verts.append(output.vertices.detach().cpu().numpy())

            target_verts = np.concatenate(target_verts, axis=0)

        assert len(pred_verts) == len(target_verts)
        error_per_vert = np.sqrt(np.sum((target_verts - pred_verts.detach().cpu().numpy()) ** 2, axis=2))
        return np.mean(error_per_vert, axis=1)


    def log(self):
        # self.writer(f'mpjpe: {self.p1_meter.avg:.2f}mm, pampjpe: {self.p2_meter.avg:.2f}mm, pve: {self.p3_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm, avg_gpve: {self.avg_gpve_meter.avg:.2f}mm 3dpw')
        self.writer(f'mpjpe: {self.p1_meter.avg:.2f}mm, pampjpe: {self.p2_meter.avg:.2f}mm, pve: {self.p3_meter.avg:.2f}mm, papve: {self.papve_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm, avg_gpve: {self.avg_gpve_meter.avg:.2f}mm 3dpw')


class ThreeDPWHandlerROMPBEV(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', pattern='.60457274_', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.pattern = pattern
        self.p1_meter = AverageMeter('P1', ':.3f')
        self.p2_meter = AverageMeter('P2', ':.3f')
        self.p3_meter = AverageMeter('P3', ':.3f')
        self.papve_meter = AverageMeter('papve', ':.3f')

        self.avg_gpve_meter = AverageMeter('avg_gpve', ':.3f')
        self.gce_meter = AverageMeter('P3', ':.3f')
        self.gpve_meter = AverageMeter('P3', ':.3f')
        self.saved = False
        self.smpl = SMPL(
                'data/smpl'
            ).to('cpu')

        self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/threedpw_train_ads_zy.txt', 'w+')

    def get_gt_joints(self, gt_shape, gt_pose, J_regressor_batch):
        # print("ThreeDPWHandler get_gt_joints gt_pose.shape", gt_pose.shape)
        # ThreeDPWHandler get_gt_joints gt_pose.shape torch.Size([1, 24, 3])

        # 应该分gender smplr, todo...
        output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 1:], global_orient=gt_pose[:, :1], pose2rot=True)
        # output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3], pose2rot=True)
        #output = smpl(betas=gt_shape, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3], transl=gt_trans)
        verts = output.vertices

        # 这里可视化 verts 看看
        

        j3d = torch.matmul(J_regressor_batch, verts)
        j3d = j3d[:, H36M_TO_J14, :]
        #print(j3d.shape, verts.shape)
        return j3d, verts

    def handle(self, data_batch, pred_results, use_gt=False):
        gt_num_people = data_batch['gt_kpts3d'].data[0][0].shape[0]
        file_name = data_batch['img_metas'].data[0][0]['file_name']
        metric_values = {
                     'MPJPE': [],
                     'pve': [],
                     'PAMPJPE': [],
                     # global metrics
                     'gce': [],
                     'gpve': []}

        for pidx in range(gt_num_people):
            packs = self.handle_single(data_batch, pred_results, pidx, use_gt=use_gt)
            for k in packs:
                metric_values[k].append(packs[k])

        for k in metric_values:
            metric_values[k] = np.mean(metric_values[k])

        metric_values.update({
            'file_name': file_name,
            # 'pred_rotmat': pred_results['pred_rotmat'],
            # 'pred_betas': pred_results['pred_betas'],
        })

        return metric_values

    def handle_single(self, data_batch, pred_results, pidx, use_gt=False):
        pred_center_preds = pred_results['pred_center_preds'].float().detach().cpu().numpy()
        gt_center_preds = pred_results['gt_center_preds'].float().detach().cpu().numpy()
        pred_vertices = pred_results['pred_vertices'].cpu()
        pred_trans = pred_results['pred_translation'].squeeze(1).detach().cpu().numpy()  # n 3
        # pred_num_people = pred_vertices.shape[0]

        # for ROMP and BEV 应该选择center map最近的pair
        # pred_bbox 与 gt_bbox距离
        dist_array = np.linalg.norm(pred_center_preds - gt_center_preds[pidx], axis=1) 
        min_idx = np.argmin(dist_array)
        # print(dist_array)
        # print(min_idx)
        pred_vertices = pred_vertices[min_idx:min_idx+1]
        pred_trans = pred_trans[min_idx:min_idx+1]
        
        # Get 14 predicted joints from the SMPL mesh
        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
        
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_pelvis = (pred_keypoints_3d_smpl[:,[2],:] + pred_keypoints_3d_smpl[:,[3],:]) / 2.0
        pred_keypoints_3d_smpl -= pred_pelvis

        gt_verts = data_batch['gt_vertices'].data[0][0][pidx:pidx+1] # already inference with genders
        gt_J_regressor_batch = self.J_regressor[None, :].to(gt_verts.device)
        gt_keypoints_3d_smpl = torch.matmul(gt_J_regressor_batch, gt_verts)
        gt_keypoints_3d_smpl = gt_keypoints_3d_smpl[:, H36M_TO_J14, :]
        gt_pelvis = (gt_keypoints_3d_smpl[:,[2],:] + gt_keypoints_3d_smpl[:,[3],:]) / 2.0
        gt_keypoints_3d_smpl -= gt_pelvis

        # Compute error metrics
        # Absolute error (MPJPE)
        error_smpl = torch.sqrt(((pred_keypoints_3d_smpl.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        mpjpe = float(error_smpl.min() * 1000)
        self.p1_meter.update(mpjpe)

        # pampjpe
        S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d_smpl)
        pampjpe = torch.sqrt(((S1_hat.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        # errors_pa = torch.sqrt(((S1_hat.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        # pampjpe  = float(errors_pa.min() * 1000)
        # self.p2_meter.update(pampjpe)
        pampjpe = float(pampjpe * 1000)
        self.p2_meter.update(pampjpe)

        # # pve
        # pve = np.mean(np.sqrt(np.sum((gt_verts.detach().cpu().numpy() - pred_vertices.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        # #pve = float(np.mean(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        # #pve = float(self.compute_error_verts(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        # self.p3_meter.update(pve)

        # pve
        pve = np.mean(np.sqrt(np.sum((gt_verts.detach().cpu().numpy() - pred_vertices.detach().cpu().numpy()) ** 2, axis=2, keepdims=False)), axis=1, keepdims=False)
        # min_idx = np.argmin(pve, axis=0)
        pve = pve.min() * 1000
        #pve = float(np.mean(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        #pve = float(self.compute_error_verts(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        self.p3_meter.update(pve)

        # pa-pve
        S1_hat = batch_compute_similarity_transform_torch(pred_vertices, gt_verts)
        errors_pa = torch.sqrt(((S1_hat.double() - gt_verts.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        # errors_pa = np.mean(np.sqrt(np.sum((S1_hat.detach().cpu().numpy() - gt_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        errors_pa = float(errors_pa * 1000)
        self.papve_meter.update(errors_pa)
        # print("!" * 50)


        #########global metrics########
        gt_trans = data_batch['gt_camera_trans'].data[0][0].clone()[pidx].unsqueeze(0).detach().cpu().numpy()  # n ,3
        gt_verts = gt_verts.detach().cpu().numpy()
        gt_verts += gt_trans[:,None]

        pred_vertices = pred_vertices.detach().cpu().numpy()
        pred_vertices += pred_trans[:, None]

        # gpve
        gpve = calc_gpve(gt_verts, pred_vertices)
        self.gpve_meter.update(gpve)

        # avg_gpve
        avg_gpve = calc_avg_gpve(gt_verts, pred_vertices)
        self.avg_gpve_meter.update(avg_gpve)

        # gce
        gt_pelvis += gt_trans[:,None]
        gt_pelvis = gt_pelvis.detach().cpu().numpy()
        pred_pelvis += pred_trans[:, None]
        pred_pelvis = pred_pelvis.detach().cpu().numpy()
        gce = calc_gce(gt_pelvis[:,0], pred_pelvis[:,0], pelvis_index=None)
        self.gce_meter.update(gce)

        for kk in range(gt_trans.shape[0]):
            self.logger.write('%.2f_%.2f\n' % (gt_trans[kk,1], gt_trans[kk,2]))
            # print('%.2f_%.2f\n' % (gt_trans[kk,1], gt_trans[kk,2]))

        save_pack = {#'file_name': file_name,
                     'MPJPE': mpjpe,
                     'pve': pve,
                     'PAMPJPE': pampjpe,
                     #'pred_rotmat': pred_results['pred_rotmat'],
                     #'pred_betas': pred_results['pred_betas'],
                     # global metrics
                     
                     'gce': gce,
                     'gpve': gpve,
                     #'avg_gpve': avg_gpve,
                     }
        return save_pack

    def compute_error_verts(self, pred_verts, target_shape=None, target_pose=None, target_verts=None):
        """
        Computes MPJPE over 6890 surface vertices.
        Args:
            verts_gt (Nx6890x3).
            verts_pred (Nx6890x3).
        Returns:
            error_verts (N).
        """

        if target_verts is None:

            betas = target_shape
            pose = target_pose

            target_verts = []
            b_ = torch.split(betas, 5000)
            p_ = torch.split(pose, 5000)

            for b,p in zip(b_,p_):
                #print(b.shape, p.shape)
                print("ThreeDPWHandler compute_error_verts p.shape", p.shape)
                output = self.smpl(betas=b, body_pose=p[:, 1:], global_orient=p[:, :1], pose2rot=True)
                # output = self.smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], pose2rot=True)
                target_verts.append(output.vertices.detach().cpu().numpy())

            target_verts = np.concatenate(target_verts, axis=0)

        assert len(pred_verts) == len(target_verts)
        error_per_vert = np.sqrt(np.sum((target_verts - pred_verts.detach().cpu().numpy()) ** 2, axis=2))
        return np.mean(error_per_vert, axis=1)


    def log(self):
        # self.writer(f'mpjpe: {self.p1_meter.avg:.2f}mm, pampjpe: {self.p2_meter.avg:.2f}mm, pve: {self.p3_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm, avg_gpve: {self.avg_gpve_meter.avg:.2f}mm 3dpw')
        self.writer(f'mpjpe: {self.p1_meter.avg:.2f}mm, pampjpe: {self.p2_meter.avg:.2f}mm, pve: {self.p3_meter.avg:.2f}mm, papve: {self.papve_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm, avg_gpve: {self.avg_gpve_meter.avg:.2f}mm 3dpw')



class ThreeDPWHandler_ROMP(EvalHandler):
    # def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', pattern='.60457274_', **kwargs):
    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_extra.npy', pattern='.60457274_', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.pattern = pattern

        self.mpjpe_meter = AverageMeter('P1', ':.3f')
        self.pampjpe_meter = AverageMeter('P2', ':.3f')
        self.pve_meter = AverageMeter('P3', ':.3f')
        self.papve_meter = AverageMeter('papve', ':.3f')
        self.avg_gpve_meter = AverageMeter('avg_gpve', ':.3f')
        self.gce_meter = AverageMeter('P3', ':.3f')
        self.gpve_meter = AverageMeter('P3', ':.3f')

        self.smpl = SMPL('data/smpl').to('cpu') # should to cuda for speeding?

        self.saved = False
        self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/threedpw_train_ads_zy.txt', 'w+')

    def get_gt_joints(self, gt_shape, gt_pose, J_regressor_batch, joint_mapping=ADS24_TO_LSP14):
        output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 1:], global_orient=gt_pose[:, :1], pose2rot=True)
        verts = output.vertices
        j3d = output.joints

        # j3d = torch.matmul(J_regressor_batch, verts)
        j3d = j3d[:, joint_mapping, :]
        #print(j3d.shape, verts.shape)

        return j3d, verts

    def handle(self, data_batch, pred_results, use_gt=False):
        gt_num_people = data_batch['gt_kpts3d'].data[0][0].shape[0]
        file_name = data_batch['img_metas'].data[0][0]['file_name']
        metric_values = {
                     'mpjpe': [],
                     'pampjpe': [],
                     'pve': [],
                     'gce': [],
                     'gpve': [],
                     'avg_gpve': []
                     }

        for pidx in range(gt_num_people):
            packs = self.handle_single(data_batch, pred_results, pidx, use_gt=use_gt)
            for k in packs:
                metric_values[k].append(packs[k])

        for k in metric_values:
            metric_values[k] = np.mean(metric_values[k])

        metric_values.update({
            'file_name': file_name,
            'pred_rotmat': pred_results['pred_rotmat'],
            'pred_betas': pred_results['pred_betas'],
        })

        return metric_values

    def handle_single(self, data_batch, pred_results, pidx, use_gt=False):
        pred_joints = pred_results['pred_joints'].cpu()
        # pred_poses = pred_results['pred_rotmat'].cpu()
        # pred_betas = pred_results['pred_betas'].cpu()
        pred_vertices = pred_results['pred_vertices'].cpu()
        pred_trans = pred_results['pred_translation'].squeeze(1).detach().cpu().numpy()  # n 3
        pred_num_people = pred_vertices.shape[0]
        J_regressor_batch = self.J_regressor[None, :].to(pred_vertices.device)

        # should load 3djoint from file direatly
        gt_keypoints_3d, gt_verts = self.get_gt_joints(data_batch['gt_shapes'].data[0][0].clone()[pidx].unsqueeze(0), 
                                                       data_batch['gt_poses'].data[0][0].clone()[pidx].unsqueeze(0),
                                                       J_regressor_batch, 
                                                       joint_mapping=ADS24_TO_LSP14
                                                       )
        score_place = torch.ones([gt_keypoints_3d.shape[0], gt_keypoints_3d.shape[1], 1])
        gt_keypoints_3d = torch.cat([gt_keypoints_3d, score_place], dim=-1)
        gt_keypoints_3d = gt_keypoints_3d.repeat([pred_num_people, 1, 1])
        align_inds = [constants.ADS_SMPL_24['Right Hip'], constants.ADS_SMPL_24['Left Hip']]
        gt_pelvis = gt_keypoints_3d[:, align_inds, :3].mean(1)
        
        # Get 14 predicted joints from the SMPL mesh
        # J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
        # pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        # print(pred_poses.shape, pred_betas.shape)
        # gt_keypoints_3d, gt_verts = self.get_gt_joints(pred_poses, 
        #                                                pred_betas,
        #                                                J_regressor_batch, 
        #                                                joint_mapping=ADS24_TO_LSP14
        #                                                )
        # print(pred_joints.shape)
        pred_keypoints_3d_smpl = pred_joints[:, ADS24_TO_LSP14, :]
        pred_pelvis = pred_keypoints_3d_smpl[:, align_inds, :3].mean(1)

        
        # mpjpe
        mpjpe_arr = calc_mpjpe(gt_keypoints_3d, pred_keypoints_3d_smpl, align_inds=align_inds)
        mpjpe_arr = mpjpe_arr.mean(-1)
        mpjpe = float(mpjpe_arr.min() * 1000)
        self.mpjpe_meter.update(mpjpe)

        # pampjpe
        pampjpe_arr = calc_pampjpe(gt_keypoints_3d, pred_keypoints_3d_smpl)
        pampjpe_arr = pampjpe_arr.mean(-1)
        pampjpe = float(pampjpe_arr.min() * 1000)
        self.pampjpe_meter.update(pampjpe)

        # # Absolute error (MPJPE)
        # error_smpl = torch.sqrt(((pred_keypoints_3d_smpl.double() - gt_keypoints_3d.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        # mpjpe = float(error_smpl.min() * 1000)
        # self.mpjpe_meter.update(mpjpe)

        # # pampjpe
        # S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d.repeat([pred_num_people, 1, 1]))
        # errors_pa = torch.sqrt(((S1_hat.double() - gt_keypoints_3d.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        # pampjpe  = float(errors_pa.min() * 1000)
        # self.pampjpe_meter.update(pampjpe)

        # # pve
        # pve = np.mean(np.sqrt(np.sum((gt_verts.detach().cpu().numpy() - pred_vertices.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        # #pve = float(np.mean(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        # #pve = float(self.compute_error_verts(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        # self.p3_meter.update(pve)

        # pve
        pve = np.mean(np.sqrt(np.sum((gt_verts.detach().cpu().numpy() - pred_vertices.detach().cpu().numpy()) ** 2, axis=2, keepdims=False)), axis=1, keepdims=False)
        min_idx = np.argmin(pve, axis=0)
        pve = pve.min() * 1000
        self.pve_meter.update(pve)

        # pa-pve
        S1_hat = batch_compute_similarity_transform_torch(pred_vertices[min_idx].unsqueeze(0), gt_verts)
        errors_pa = np.mean(np.sqrt(np.sum((S1_hat.detach().cpu().numpy() - gt_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        self.papve_meter.update(errors_pa)


        #########global metrics########
        gt_trans = data_batch['gt_camera_trans'].data[0][0].clone()[pidx].unsqueeze(0)
        gt_trans = gt_trans.repeat([pred_num_people, 1])
        gt_verts = gt_verts.repeat([pred_num_people, 1, 1])
        gt_trans = gt_trans.detach().cpu().numpy()
        gt_verts = gt_verts.detach().cpu().numpy()


        gt_verts += gt_trans[:,None]

        pred_vertices = pred_vertices.detach().cpu().numpy()
        pred_vertices += pred_trans[:, None]

        gpve = calc_gpve(gt_verts, pred_vertices)
        self.gpve_meter.update(gpve)

        avg_gpve = calc_avg_gpve(gt_verts, pred_vertices)
        self.avg_gpve_meter.update(avg_gpve)

        gt_pelvis += gt_trans[:,None]
        gt_pelvis = gt_pelvis.detach().cpu().numpy()

        pred_pelvis += pred_trans[:, None]
        pred_pelvis = pred_pelvis.detach().cpu().numpy()

        gce = calc_gce(gt_pelvis[:,0], pred_pelvis[:,0], pelvis_index=None)
        self.gce_meter.update(gce)

        for kk in range(gt_trans.shape[0]):
            self.logger.write('%.2f_%.2f\n' % (gt_trans[kk,1], gt_trans[kk,2]))
            # print('%.2f_%.2f\n' % (gt_trans[kk,1], gt_trans[kk,2]))

        save_pack = {
                     'mpjpe': mpjpe,
                     'pampjpe': pampjpe,
                     'pve': pve,
                     'gce': gce,
                     'gpve': gpve,
                     'avg_gpve': avg_gpve,
                     }

        return save_pack

    def compute_error_verts(self, pred_verts, target_shape=None, target_pose=None, target_verts=None):
        """
        Computes MPJPE over 6890 surface vertices.
        Args:
            verts_gt (Nx6890x3).
            verts_pred (Nx6890x3).
        Returns:
            error_verts (N).
        """
        if target_verts is None:
            betas = target_shape
            pose = target_pose
            target_verts = []
            b_ = torch.split(betas, 5000)
            p_ = torch.split(pose, 5000)

            for b, p in zip(b_, p_):
                #print(b.shape, p.shape)
                print("ThreeDPWHandler compute_error_verts p.shape", p.shape)
                output = self.smpl(betas=b, body_pose=p[:, 1:], global_orient=p[:, :1], pose2rot=True)
                # output = self.smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], pose2rot=True)
                target_verts.append(output.vertices.detach().cpu().numpy())
            target_verts = np.concatenate(target_verts, axis=0)

        assert len(pred_verts) == len(target_verts)
        error_per_vert = np.sqrt(np.sum((target_verts - pred_verts.detach().cpu().numpy()) ** 2, axis=2))

        return np.mean(error_per_vert, axis=1)

    def log(self):
        self.writer(f'mpjpe: {self.mpjpe_meter.avg:.2f}mm, pampjpe: {self.pampjpe_meter.avg:.2f}mm, pve: {self.pve_meter.avg:.2f}mm, papve: {self.papve_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm, avg_gpve: {self.avg_gpve_meter.avg:.2f}mm 3dpw')



def single_compute_similarity_transform_torch(S1, S2, return_pa=False):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    S1 = S1.float()
    S2 = S2.float()
    #print(S1.shape, S2.shape)
    transposed = False
    #if S1.shape[0] != 3 and S1.shape[0] != 2:
    S1 = S1.permute(0,2,1)
    S2 = S2.permute(0,2,1)
    transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(dim=-1, keepdim=True)
    mu2 = S2.mean(dim=-1, keepdim=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    #print(K.shape)
    # U, s, Vh = torch.linalg.svd(K, full_matrices=False)
    # V = Vh.transpose(-2, -1).conj()
    #U, s, V = torch.svd(K)
    us = []
    ss = []
    vs = []
    for bidx in range(K.shape[0]):
        k = K[bidx]
        U, s, V = torch.svd(k)
        us.append(U.unsqueeze(0))
        ss.append(s.unsqueeze(0))
        vs.append(V.unsqueeze(0))
    U = torch.cat(us, dim=0)
    s = torch.cat(ss, dim=0)
    V = torch.cat(vs, dim=0)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    value = U.bmm(V.permute(0,2,1))
    for bidx in range(value.shape[0]):
        Z[bidx, -1, -1] *= torch.sign(torch.det(value[bidx]))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    if return_pa:
        return S1_hat, (scale, R, t)
    else:
        return S1_hat

# def compute_mpjpe(predicted, target, valid_mask=None, pck_joints=None, sample_wise=True):
#     """
#     Mean per-joint position error (i.e. mean Euclidean distance),
#     often referred to as "Protocol #1" in many papers.
#     """
#     assert predicted.shape == target.shape, print(predicted.shape, target.shape)
#     mpjpe = torch.norm(predicted - target, p=2, dim=-1, keepdim=True)
#     print(mpjpe.shape, predicted.shape, target.shape, valid_mask.shape)
#     if pck_joints is None:
#         if sample_wise:
#             mpjpe_batch = (mpjpe*valid_mask.float()).sum(-1)/valid_mask.float().sum(-1) if valid_mask is not None else mpjpe.mean(-1)
#         else:
#             mpjpe_batch = mpjpe[valid_mask] if valid_mask is not None else mpjpe
#         return mpjpe_batch
#     else:
#         mpjpe_pck_batch = mpjpe[:,pck_joints]
#         return mpjpe_pck_batch

def batch_compute_similarity_transform_torch(S1, S2, return_pa=False):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    # U, s, V = torch.svd(K)
    try:
        U, s, V = torch.svd(K)
    except:
        U, s, V = torch.svd(K + 0.00001)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    if return_pa:
        return S1_hat, (scale, R, t)
    else:
        return S1_hat
    
class PanopticEvalHandler(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.p1_meter = AverageMeter('P1', ':.2f')
        self.stats = list()
        self.mismatch_cnt = 0
        # Initialize SMPL model
        openpose_joints = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                           7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        extra_joints = [8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47, 48, 49, 50, 51, 52, 53, 24, 26, 25, 28, 27]
        joints = torch.tensor(openpose_joints + extra_joints, dtype=torch.int32)
        joint_mapper = JointMapper(joints)
        smpl_params = dict(model_folder='data/smpl',
                           joint_mapper=joint_mapper,
                           create_glb_pose=True,
                           body_pose_param='identity',
                           create_body_pose=True,
                           create_betas=True,
                           # create_trans=True,
                           dtype=torch.float32,
                           vposer_ckpt=None,
                           gender='neutral')
        self.smpl = SMPL(**smpl_params)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.collision_meter = AverageMeter('P3', ':.2f')
        self.collision_volume = CollisionVolume(self.smpl.faces, grid_size=64).cuda()
        self.coll_cnt = 0
        self.threshold_list = [0.1, 0.15, 0.2]
        self.total_ordinal_cnt = {i: 0 for i in self.threshold_list}
        self.correct_ordinal_cnt = {i: 0 for i in self.threshold_list}

    def handle(self, data_batch, pred_results, use_gt=False):
        # Evaluate collision metric
        pred_vertices = pred_results['pred_vertices']
        pred_translation = pred_results['pred_translation']
        cur_collision_volume = self.collision_volume(pred_vertices, pred_translation)
        if cur_collision_volume.item() > 0:
            # self.writer(f'Collision found with {cur_collision_volume.item() * 1000} L')
            self.coll_cnt += 1
        self.collision_meter.update(cur_collision_volume.item() * 1000.)

        pred_vertices = pred_results['pred_vertices'].cpu()
        pred_camera = pred_results['pred_camera'].cpu()
        pred_translation = pred_results['pred_translation'].cpu()
        bboxes = pred_results['bboxes'][0][:, :4]
        img = data_batch['img'].data[0][0].clone()

        gt_keypoints_3d = data_batch['gt_kpts3d'].data[0][0].clone()
        gt_pelvis_smpl = gt_keypoints_3d[:, [14], :-1].clone()
        visible_kpts = gt_keypoints_3d[:, J24_TO_H36M, -1].clone()
        origin_gt_kpts3d = data_batch['gt_kpts3d'].data[0][0].clone().cpu()
        origin_gt_kpts3d = origin_gt_kpts3d[:, J24_TO_H36M]
        # origin_gt_kpts3d[:, :, :-1] -= gt_pelvis_smpl
        gt_keypoints_3d = gt_keypoints_3d[:, J24_TO_H36M, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis_smpl

        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
            pred_vertices.device)
        # Get 14 predicted joints from the SMPL mesh
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
        # pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl - pred_pelvis_smpl

        file_name = data_batch['img_metas'].data[0][0]['file_name']
        fname = osp.basename(file_name)

        # To select closest points
        glb_vis = (visible_kpts.sum(0) >= (
                visible_kpts.shape[0] - 0.1)).float()[None, :, None]  # To avoid in-accuracy in float point number
        if use_gt:
            paired_idxs = torch.arange(gt_keypoints_3d.shape[0])
        else:
            dist = vectorize_distance((glb_vis * gt_keypoints_3d).numpy(),
                                      (glb_vis * pred_keypoints_3d_smpl).numpy())
            paired_idxs = torch.from_numpy(dist.argmin(1))
        is_mismatch = len(set(paired_idxs.tolist())) < len(paired_idxs)
        if is_mismatch:
            self.mismatch_cnt += 1

        selected_prediction = pred_keypoints_3d_smpl[paired_idxs]

        # Compute error metrics
        # Absolute error (MPJPE)
        error_smpl = (torch.sqrt(((selected_prediction - gt_keypoints_3d) ** 2).sum(dim=-1)) * visible_kpts)

        mpjpe = float(error_smpl.mean() * 1000)
        self.p1_meter.update(mpjpe, n=error_smpl.shape[0])

        save_pack = {'file_name': osp.basename(file_name),
                     'MPJPE': mpjpe,
                     'pred_rotmat': pred_results['pred_rotmat'].cpu(),
                     'pred_betas': pred_results['pred_betas'].cpu(),
                     'gt_kpts': origin_gt_kpts3d,
                     'kpts_paired': selected_prediction,
                     'pred_kpts': pred_keypoints_3d_smpl,
                     }

        if self.viz_dir and (is_mismatch or error_smpl.mean(-1).min() * 1000 > 200):
            img = img.clone() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                [0.485, 0.456, 0.406]).view(3, 1, 1)
            img_cv = img.clone().numpy()
            img_cv = (img_cv * 255).astype(np.uint8).transpose([1, 2, 0]).copy()
            for bbox in bboxes[paired_idxs]:
                img_cv = cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            img_cv = draw_text(img_cv, {'mismatch': is_mismatch, 'error': str(error_smpl.mean(-1) * 1000)});
            img_cv = (img_cv / 255.)

            torch.set_printoptions(precision=1)
            img_render = self.renderer([torch.tensor(img_cv.transpose([2, 0, 1]))], [pred_vertices],
                                       translation=[pred_translation])

            bv_verts = get_bv_verts(bboxes, pred_vertices, pred_translation,
                                    img.shape, self.FOCAL_LENGTH)
            img_bv = self.renderer([torch.ones_like(img)], [bv_verts],
                                   translation=[torch.zeros(bv_verts.shape[0], 3)])
            img_grid = torchvision.utils.make_grid(torch.tensor(([img_render[0], img_bv[0]])),
                                                   nrow=2).numpy().transpose([1, 2, 0])
            img_grid[img_grid > 1] = 1
            img_grid[img_grid < 0] = 0
            plt.imsave(osp.join(self.viz_dir, fname), img_grid)
        return save_pack

    def log(self):
        self.writer(
            f'p1: {self.p1_meter.avg:.2f}mm, coll_cnt: {self.coll_cnt} coll: {self.collision_meter.avg} L')


class MuPoTSEvalHandler(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.p1_meter = AverageMeter('P1', ':.2f')
        self.p2_meter = AverageMeter('P2', ':.2f')
        self.p3_meter = AverageMeter('P3', ':.2f')
        self.stats = list()
        self.mismatch_cnt = 0
        # Initialize SMPL model
        openpose_joints = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                           7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        extra_joints = [8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47, 48, 49, 50, 51, 52, 53, 24, 26, 25, 28, 27]
        joints = torch.tensor(openpose_joints + extra_joints, dtype=torch.int32)
        joint_mapper = JointMapper(joints)
        smpl_params = dict(model_folder='data/smpl',
                           joint_mapper=joint_mapper,
                           create_glb_pose=True,
                           body_pose_param='identity',
                           create_body_pose=True,
                           create_betas=True,
                           # create_trans=True,
                           dtype=torch.float32,
                           vposer_ckpt=None,
                           gender='neutral')
        self.smpl = SMPL(**smpl_params)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.result_list = list()
        self.result_list_2d = list()
        self.h36m_to_MPI = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]

        self.collision_meter = AverageMeter('collision', ':.2f')
        self.collision_volume = CollisionVolume(self.smpl.faces, grid_size=64).cuda()
        self.coll_cnt = 0

    def handle(self, data_batch, pred_results, use_gt=False):
        # Evaluate collision metric
        pred_vertices = pred_results['pred_vertices']
        pred_translation = pred_results['pred_translation']
        cur_collision_volume = self.collision_volume(pred_vertices, pred_translation)
        if cur_collision_volume.item() > 0:
            # self.writer(f'Collision found with {cur_collision_volume.item() * 100 } L')
            self.coll_cnt += 1
        self.collision_meter.update(cur_collision_volume.item() * 1000.)

        pred_vertices = pred_results['pred_vertices'].cpu()
        pred_camera = pred_results['pred_camera'].cpu()
        pred_translation = pred_results['pred_translation'].cpu()
        bboxes = pred_results['bboxes'][0][:, :4]
        img = data_batch['img'].data[0][0].clone()

        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
            pred_vertices.device)
        # Get 14 predicted joints from the SMPL mesh
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
        self.result_list.append(
            (pred_keypoints_3d_smpl[:, self.h36m_to_MPI] + pred_translation[:, None]).numpy())
        batch_size = pred_keypoints_3d_smpl.shape[0]
        img_size = torch.zeros(batch_size, 2).to(pred_keypoints_3d_smpl.device)
        img_size += torch.tensor(img.shape[:-3:-1], dtype=img_size.dtype).to(img_size.device)
        rotation_Is = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_keypoints_3d_smpl.device)
        pred_keypoints_2d_smpl = self.camera(pred_keypoints_3d_smpl, batch_size=batch_size, rotation=rotation_Is,
                                             translation=pred_translation,
                                             center=img_size / 2)
        if self.viz_dir:
            img = img.clone() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                [0.485, 0.456, 0.406]).view(3, 1, 1)
            img_cv = img.clone().numpy()
            img_cv = (img_cv * 255).astype(np.uint8).transpose([1, 2, 0]).copy()
            for kpts, bbox in zip(pred_keypoints_2d_smpl.numpy(), bboxes):
                img_cv = cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                img_cv = draw_skeleton(img_cv, kpts[H36M_TO_J14, :2])
            # img_cv = draw_text(img_cv, {'mismatch': is_mismatch, 'error': str(error_smpl.mean(-1) * 1000)});
            img_cv = (img_cv / 255.)
            fname = osp.basename(data_batch['img_metas'].data[0][0]['file_name'])
            plt.imsave(osp.join(self.viz_dir, fname), img_cv)

        scale_factor = data_batch['img_metas'].data[0][0]['scale_factor']
        raw_kpts2d = pred_keypoints_2d_smpl / scale_factor
        self.result_list_2d.append(raw_kpts2d[:, self.h36m_to_MPI])
        return {'file_name': data_batch['img_metas'].data[0][0]['file_name'], 'pred_kpts3d': pred_keypoints_3d_smpl}

    def log(self):
        self.writer(f'coll_cnt: {self.coll_cnt} coll {self.collision_meter.avg} L')

    def finalize(self):
        max_persons = max([i.shape[0] for i in self.result_list])
        result = np.zeros((len(self.result_list), max_persons, 17, 3))
        result_2d = np.zeros((len(self.result_list), max_persons, 17, 2))
        for i, (r, r_2d) in enumerate(zip(self.result_list, self.result_list_2d)):
            result[i, :r.shape[0]] = r
            result_2d[i, :r.shape[0]] = r_2d
        scio.savemat(osp.join(self.work_dir, 'mupots.mat'), {'result': result, 'result_2d': result_2d})



class ROMPThreeDPWHandler(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', pattern='.60457274_', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.pattern = pattern
        self.p1_meter = AverageMeter('P1', ':.3f')
        self.p2_meter = AverageMeter('P2', ':.3f')
        self.p3_meter = AverageMeter('P3', ':.3f')
        self.papve_meter = AverageMeter('papve', ':.3f')
        self.gce_meter = AverageMeter('P3', ':.3f')
        self.gpve_meter = AverageMeter('P3', ':.3f')
        self.saved = False
        self.smpl = SMPL(
                'data/smpl'
            ).to('cpu')
        self.romp_pred = np.load("/home/tiange/work/result_ft_real.npy", allow_pickle=True)[()]

        self.converter = Converter()

    def get_gt_joints(self, gt_shape, gt_pose, J_regressor_batch):
        print("ROMPThreeDPWHandler get_gt_joints gt_pose.shape", gt_pose.shape)
        output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3], pose2rot=True)
        #output = smpl(betas=gt_shape, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3], transl=gt_trans)
        verts = output.vertices

        j3d = torch.matmul(J_regressor_batch, verts)
        j3d = j3d[:, H36M_TO_J14, :]
        #print(j3d.shape, verts.shape)
        return j3d, verts

    def handle(self, data_batch, pred_results, use_gt=False):
        prefix = '/home/yutian/dataset/3DPW/imageFiles/'
        gt_num_people = data_batch['gt_kpts3d'].data[0][0].shape[0]
        file_name = data_batch['img_metas'].data[0][0]['file_name']

        use_path = file_name.split('/')

        name = prefix + use_path[-2] + '/' + use_path[-1]

        if name not in self.romp_pred:
            print(name, 'cannot be found!', file_name)
            return {}

        values = self.romp_pred[name]
        #print(values.keys())
        vertices = np.array(values['vertices'])
        cam = np.array(values['cam'])

        # cam = cam / np.tan(np.radians(30))[None][None] 
        # trans = np.zeros_like(cam)
        # trans[:,-1] = cam[:,0]
        # trans[:,:-1] = cam[:,1:]
        #trans = np.array(values['trans'])

        # print(vertices.shape, trans.shape)

        # trans[:,-1] *= 1.6 # 1.6
        # trans[:,0] /= (1920/512)/2.3

        # trans[:,1] /= (1920/512)/2.3


        oshape =  data_batch['img_metas'].data[0][0]['ori_shape']

        #print(oshape)


        # if oshape[0] != 1920:
        #     pixel_offset = 832 //2 - 512//2
        # else:
        #     pixel_offset = 0
        # if pixel_offset != 0:
        #     camera_offset = (trans[:,-1] * pixel_offset) / 5000.
        #     trans[:,0] -= camera_offset

        # if oshape[1] != 1920:
        #     pixel_offset = 512 //2 - 297//2
        # else:
        #     pixel_offset = 0
        # if pixel_offset != 0:
        #     camera_offset = (trans[:,-1] * pixel_offset) / 5000.
        #     trans[:,1] -= camera_offset

        camed_verts = self.converter.get_ours_camera_verts(oshape, cam, vertices)
        

        pred_results = {}
        #pred_results['pred_translation'] = torch.tensor(trans).to(data_batch['gt_kpts3d'].data[0][0].device)
        pred_results['pred_vertices'] = torch.tensor(vertices).to(data_batch['gt_kpts3d'].data[0][0].device)
        pred_results['pred_camed_vertices'] = camed_verts.cpu().numpy()
        
        metric_values = {
                     'MPJPE': [],
                     'pve': [],
                     'PAMPJPE': [],
                     # global metrics
                     'gce': [],
                     'gpve': []}

        for pidx in range(gt_num_people):
            packs = self.handle_single(data_batch, pred_results, pidx, use_gt=use_gt)
            for k in packs:
                metric_values[k].append(packs[k])

        for k in metric_values:
            metric_values[k] = np.mean(metric_values[k])

        metric_values.update({
            'file_name': file_name,
            #'pred_rotmat': pred_results['pred_rotmat'],
            #'pred_betas': pred_results['pred_betas'],
        })

        return metric_values

    def handle_single(self, data_batch, pred_results, pidx, use_gt=False):
        gt_num_people = data_batch['gt_kpts3d'].data[0][0].shape[0]
        #print('gt num people', data_batch['gt_kpts3d'].data[0][0].shape, gt_num_people)
        o_device = pred_results['pred_vertices'].device
        pred_vertices = pred_results['pred_vertices'].cpu()
        #pred_trans = pred_results['pred_translation'].squeeze(1).detach().cpu().numpy()  # n 3

        
        pred_num_people = pred_vertices.shape[0]

        J_regressor_batch = self.J_regressor[None, :].to(
            pred_vertices.device)

        gt_keypoints_3d, gt_verts = self.get_gt_joints(data_batch['gt_shapes'].data[0][0].clone()[pidx].unsqueeze(0), 
                                                       data_batch['gt_poses'].data[0][0].clone()[pidx].unsqueeze(0),
                                                       J_regressor_batch)
        #print(gt_keypoints_3d.shape, gt_verts.shape)
        #gt_verts = gt_verts.repeat([pred_num_people, 1, 1])
        # gt_pelvis = (gt_keypoints_3d[:,[2],:] + gt_keypoints_3d[:,[3],:]) / 2.0
        # gt_keypoints_3d -= gt_pelvis
        
        # # Get 14 predicted joints from the SMPL mesh
        # J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
        #     pred_vertices.device)
        # pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)

        # pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        # pred_pelvis = (pred_keypoints_3d_smpl[:,[2],:] + pred_keypoints_3d_smpl[:,[3],:]) / 2.0
        # pred_keypoints_3d_smpl -= pred_pelvis

        #file_name = data_batch['img_metas'].data[0][0]['file_name']

        # Compute error metrics
        # Absolute error (MPJPE)
        # error_smpl = torch.sqrt(((pred_keypoints_3d_smpl.double() - gt_keypoints_3d.double()) ** 2).sum(dim=-1)).mean(
        #     dim=-1)

        # mpjpe = float(error_smpl.min() * 1000)
        # self.p1_meter.update(mpjpe)

        # # pampjpe
        # # print(pred_keypoints_3d_smpl.shape, gt_keypoints_3d.shape)
        # S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d.repeat([pred_num_people, 1, 1]))
        # errors_pa = torch.sqrt(((S1_hat.double() - gt_keypoints_3d.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        # pampjpe  = float(errors_pa.min() * 1000)
        # self.p2_meter.update(pampjpe)

        # pve
        pve = np.mean(np.sqrt(np.sum((gt_verts.detach().cpu().numpy() - pred_vertices.detach().cpu().numpy()) ** 2, axis=2, keepdims=False)), axis=1, keepdims=False)
        min_idx = np.argmin(pve, axis=0)
        pve = pve.min() * 1000
        #pve = float(np.mean(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        #pve = float(self.compute_error_verts(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        self.p3_meter.update(pve)

        # pa-pve
        S1_hat = batch_compute_similarity_transform_torch(pred_vertices[min_idx].unsqueeze(0), gt_verts)
        errors_pa = np.mean(np.sqrt(np.sum((S1_hat.detach().cpu().numpy() - gt_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        self.papve_meter.update(errors_pa)

        #########global metrics########
        gt_trans = data_batch['gt_camera_trans'].data[0][0].clone()[pidx].unsqueeze(0).detach().cpu().numpy()  # n ,3

        # GPVE
        gt_verts = gt_verts.detach().cpu().numpy()
        gt_verts += gt_trans[:,None]
        #pred_vertices = pred_vertices.detach().cpu().numpy()

        #pred_vertices += pred_trans[:, None]
        pred_vertices = pred_results['pred_camed_vertices']
        gpve = np.mean(np.sqrt(np.sum((gt_verts - pred_vertices) ** 2, axis=2)), axis=1).min() * 1000
        self.gpve_meter.update(gpve)

        # gce
        gce = np.sqrt(np.sum((np.mean(gt_verts, axis=1) - np.mean(pred_vertices, axis=1)) ** 2, axis=-1)).min() * 1000.
        self.gce_meter.update(gce)

        save_pack = {#'file_name': file_name,
                     #'MPJPE': mpjpe,
                     'pve': pve,
                     #'PAMPJPE': pampjpe,
                     #'pred_rotmat': pred_results['pred_rotmat'],
                     #'pred_betas': pred_results['pred_betas'],
                     # global metrics
                     'gce': gce,
                     'gpve': gpve,
                     }
        return save_pack

    def compute_error_verts(self, pred_verts, target_shape=None, target_pose=None, target_verts=None):
        """
        Computes MPJPE over 6890 surface vertices.
        Args:
            verts_gt (Nx6890x3).
            verts_pred (Nx6890x3).
        Returns:
            error_verts (N).
        """

        if target_verts is None:

            betas = target_shape
            pose = target_pose

            target_verts = []
            b_ = torch.split(betas, 5000)
            p_ = torch.split(pose, 5000)

            for b,p in zip(b_,p_):
                #print(b.shape, p.shape)
                print("ROMPThreeDPWHandler compute_error_verts p.shape", p.shape)
                output = self.smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], pose2rot=True)
                target_verts.append(output.vertices.detach().cpu().numpy())

            target_verts = np.concatenate(target_verts, axis=0)

        assert len(pred_verts) == len(target_verts)
        error_per_vert = np.sqrt(np.sum((target_verts - pred_verts.detach().cpu().numpy()) ** 2, axis=2))
        return np.mean(error_per_vert, axis=1)


    def log(self):
        self.writer(f'papve: {self.papve_meter.avg:.2f}mm, pve: {self.p3_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm')



class AGORAHandler(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', pattern='.60457274_', **kwargs):
        super().__init__(**kwargs)

        self.enable_LSP14_joint_order = False
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        
        self.pattern = pattern
        self.p1_meter = AverageMeter('P1', ':.3f')
        self.p2_meter = AverageMeter('P2', ':.3f')
        self.p3_meter = AverageMeter('P3', ':.3f')

        self.papve_meter = AverageMeter('papve', ':.3f')
        self.avg_gpve_meter = AverageMeter('avg_gpve', ':.3f')
        
        self.gce_meter = AverageMeter('P3', ':.3f')
        self.gpve_meter = AverageMeter('P3', ':.3f')
        self.saved = False
        self.smpl = SMPL('data/smpl').to('cpu')

        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/ads_y.txt', 'w+')
        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/ads_y_z_V1_2022_11_09.txt', 'w')
        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/val_ads_zy_2022_11_09_V1.txt', 'w')
        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/val_ads_zy_2022_11_13_V1.txt', 'w')
        # self.logger = open('./data/table2/ADRT_table2_agora_val_ads_zy_2023_03_04.txt', 'w')
        # self.logger = open('./data/table2/ROMP_table2_agora_val_ads_zy_2023_03_04.txt', 'w')
        # self.logger = open('./data/table2/ADRT_table2_agora_val_ads_zy_2023_05_04_EP180_w_weightConf_28D_continueFrom160.txt', 'w')

        # self.logger = open('./data/table2/ADRT_table2_agora_val_ads_zy_2023_05_04_ep160.txt', 'w')
        # self.logger = open('./data/table2/ADRT_table2_agora_val_ads_zy_2023_05_04_ep160_tmp.txt', 'w')

        # self.logger = open('./data/table2/PMMD_table2_agora_val_ads_zy_2023_08_29_ep19.txt', 'w')
        self.logger = open('./data/table2/PMMD_table2_agora_val_ads_zy_2023_08_29_ep19_tmp.txt', 'w')


    def get_gt_joints(self, gt_shape, gt_pose, J_regressor_batch):
        output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 1:], global_orient=gt_pose[:, :1], pose2rot=True)
        verts = output.vertices
        if self.enable_LSP14_joint_order:
            j3d = output.joints
            j3d = j3d[:, ADS24_TO_LSP14, :]
        else:
            j3d = torch.matmul(J_regressor_batch, verts)
            j3d = j3d[:, H36M_TO_J14, :]

        return j3d, verts

    def handle(self, data_batch, pred_results, use_gt=False):

        ##### for bev #####
        # file_name = data_batch['img_metas'].data[0][0]['file_name'] # images/S9_Directions_1.60457274_000201.jpg
        # #print(file_name,'???????')
        # names = file_name.split('/')
        # folder_name = names[-2]
        # img_name = names[-1]
        # pkl_name = names[-1][:-4]+'.pkl'

        # data = joblib.load('/data/ADS/bev_results/agora_results/'+pkl_name)
        # pred_results['pred_vertices'] = torch.tensor(data['verts'])
        # pred_results['pred_translation'] = torch.tensor(data['trans']).unsqueeze(1) # n, 1, 3
        ##### for bev #####

        gt_num_people = data_batch['gt_kpts3d'].data[0][0].shape[0]
        file_name = data_batch['img_metas'].data[0][0]['file_name']
        metric_values = {
                     'mpjpe': [],
                     'pve': [],
                     'pampjpe': [],
                     'gce': [],
                     'gpve': [],
                     'avg_gpve': []
                     }

        for pidx in range(gt_num_people):
            packs = self.handle_single(data_batch, pred_results, pidx, use_gt=use_gt)
            for k in packs:
                metric_values[k].append(packs[k])

        for k in metric_values:
            metric_values[k] = np.mean(metric_values[k])

        metric_values.update({
            'file_name': file_name,
            # 'pred_rotmat': pred_results['pred_rotmat'],
            # 'pred_betas': pred_results['pred_betas'],
        })

        return metric_values
    


    def handle_single(self, data_batch, pred_results, pidx, use_gt=False):
        pred_verts = pred_results['pred_vertices'].float().cpu()
        # pred_trans = pred_results['pred_translation'].float().squeeze(1).detach().cpu().numpy()  # n 3
        pred_trans = pred_results['pred_translation'].float().squeeze(1).detach().cpu()
        pred_num_people = pred_verts.shape[0]

        # J_regressor_batch = self.J_regressor[None, :].to(pred_verts.device)
        J_regressor_batch = self.J_regressor[None, :].expand(pred_verts.shape[0], -1, -1).to(pred_verts.device)

        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_verts)
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_pelvis = (pred_keypoints_3d_smpl[:,[2],:] + pred_keypoints_3d_smpl[:,[3],:]) / 2.0
        pred_keypoints_3d_smpl -= pred_pelvis

        gt_verts = data_batch['gt_vertices'].data[0][0][pidx:pidx+1] # already inference with genders
        gt_J_regressor_batch = self.J_regressor[None, :].to(gt_verts.device)
        gt_keypoints_3d_smpl = torch.matmul(gt_J_regressor_batch, gt_verts)
        gt_keypoints_3d_smpl = gt_keypoints_3d_smpl[:, H36M_TO_J14, :]
        gt_pelvis = (gt_keypoints_3d_smpl[:, [2], :] + gt_keypoints_3d_smpl[:, [3], :]) / 2.0
        gt_keypoints_3d_smpl -= gt_pelvis

        gt_keypoints_3d_smpl_copy = gt_keypoints_3d_smpl.clone()
        gt_keypoints_3d_smpl = gt_keypoints_3d_smpl.repeat([pred_num_people, 1, 1])

        # for ROMP and BEV 应该选择center map最近的pair来算

        # mpjpe
        error_smpl = torch.sqrt(((pred_keypoints_3d_smpl.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        mpjpe = float(error_smpl.min() * 1000)
        self.p1_meter.update(mpjpe)

        # pampjpe
        S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d_smpl)
        # errors_pa = torch.sqrt(((S1_hat.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        errors_pa = torch.sqrt(((S1_hat.double() - gt_keypoints_3d_smpl_copy.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        min_idx = torch.argmin(errors_pa, dim=0)
        pampjpe  = float(errors_pa.min() * 1000)
        self.p2_meter.update(pampjpe)

        S1_hat = batch_compute_similarity_transform_torch(pred_verts[min_idx].unsqueeze(0), gt_verts)
        papve = np.mean(np.sqrt(np.sum((S1_hat.detach().cpu().numpy() - gt_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        self.papve_meter.update(papve)

        # pve # should match each 
        pve = np.mean(np.sqrt(np.sum((gt_verts.detach().cpu().numpy() - pred_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        self.p3_meter.update(pve)

        #########global metrics########
        gt_trans = data_batch['gt_camera_trans'].data[0][0].clone()[pidx].unsqueeze(0) # n ,3
        gt_verts = gt_verts.detach()   
        gt_verts += gt_trans[:,None]
        # pred_trans = pred_results['pred_translation'].squeeze(1).detach().cpu()  # n 3
        pred_verts = pred_verts.detach()
        pred_verts += pred_trans[:, None]

        gt_verts = gt_verts.detach().cpu().numpy()
        pred_verts = pred_verts.detach().cpu().numpy()

        # gpve
        gpve = calc_gpve(gt_verts, pred_verts)
        self.gpve_meter.update(gpve)

        # avg_gpve
        avg_gpve = calc_avg_gpve(gt_verts, pred_verts)
        self.avg_gpve_meter.update(avg_gpve)

        # gce
        gt_pelvis += gt_trans[:,None]
        gt_pelvis = gt_pelvis.detach()
        pred_pelvis += pred_trans[:, None]

        gt_pelvis = gt_pelvis.detach().cpu().numpy()
        pred_pelvis = pred_pelvis.detach().cpu().numpy()
        gce = calc_gce(gt_pelvis[:,0], pred_pelvis[:,0], pelvis_index=None)
        self.gce_meter.update(gce)

        # logger for table2
        self.logger.write('%.2f_%.2f_%.2f_%.2f\n' % (gt_trans[0,1], gt_trans[0,2], pampjpe, gpve))
        # print('%.2ff_%.2f_%.2f_%.2f\n' % (gt_trans[0,1], gt_trans[0,2], pampjpe, gpve))

        save_pack = {#'file_name': file_name,
                     'mpjpe': mpjpe,
                     'pve': pve,
                     'pampjpe': pampjpe,
                     'gce': gce,
                     'gpve': gpve,
                     'avg_gpve': avg_gpve,
                     }
        return save_pack

    def compute_error_verts(self, pred_verts, target_shape=None, target_pose=None, target_verts=None):
        """
        Computes MPJPE over 6890 surface vertices.
        Args:
            verts_gt (Nx6890x3).
            verts_pred (Nx6890x3).
        Returns:
            error_verts (N).
        """

        if target_verts is None:

            betas = target_shape
            pose = target_pose

            target_verts = []
            b_ = torch.split(betas, 5000)
            p_ = torch.split(pose, 5000)

            for b,p in zip(b_,p_):
                #print(b.shape, p.shape)
                print("agora compute_error_verts p shape: ", p.shape)
                output = self.smpl(betas=b, body_pose=p[:, 1:], global_orient=p[:, :1], pose2rot=True)
                target_verts.append(output.vertices.detach().cpu().numpy())

            target_verts = np.concatenate(target_verts, axis=0)

        assert len(pred_verts) == len(target_verts)
        error_per_vert = np.sqrt(np.sum((target_verts - pred_verts.detach().cpu().numpy()) ** 2, axis=2))
        return np.mean(error_per_vert, axis=1)


    def log(self):
        self.writer(f'mpjpe: {self.p1_meter.avg:.2f}mm, pampjpe: {self.p2_meter.avg:.2f}mm, pve: {self.p3_meter.avg:.2f}mm, papve: {self.papve_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm, avg_gpve: {self.avg_gpve_meter.avg:.2f}mm agora')



class AGORAHandlerCenterPoint(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', pattern='.60457274_', **kwargs):
        super().__init__(**kwargs)

        self.enable_LSP14_joint_order = False
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        
        self.pattern = pattern
        self.p1_meter = AverageMeter('P1', ':.3f')
        self.p2_meter = AverageMeter('P2', ':.3f')
        self.p3_meter = AverageMeter('P3', ':.3f')

        self.papve_meter = AverageMeter('papve', ':.3f')
        self.avg_gpve_meter = AverageMeter('avg_gpve', ':.3f')
        
        self.gce_meter = AverageMeter('P3', ':.3f')
        self.gpve_meter = AverageMeter('P3', ':.3f')
        self.saved = False
        self.smpl = SMPL('data/smpl').to('cpu')

        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/ads_y.txt', 'w+')
        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/ads_y_z_V1_2022_11_09.txt', 'w')
        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/val_ads_zy_2022_11_09_V1.txt', 'w')
        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/val_ads_zy_2022_11_13_V1.txt', 'w')
        self.logger = open('./data/table2/ADRT_table2_agora_val_ads_zy_2023_03_04.txt', 'w')
        # self.logger = open('./data/table2/ROMP_table2_agora_val_ads_zy_2023_03_04.txt', 'w')

    def get_gt_joints(self, gt_shape, gt_pose, J_regressor_batch):
        output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 1:], global_orient=gt_pose[:, :1], pose2rot=True)
        verts = output.vertices
        if self.enable_LSP14_joint_order:
            j3d = output.joints
            j3d = j3d[:, ADS24_TO_LSP14, :]
        else:
            j3d = torch.matmul(J_regressor_batch, verts)
            j3d = j3d[:, H36M_TO_J14, :]

        return j3d, verts

    def handle(self, data_batch, pred_results, use_gt=False):

        ##### for bev #####
        # file_name = data_batch['img_metas'].data[0][0]['file_name'] # images/S9_Directions_1.60457274_000201.jpg
        # #print(file_name,'???????')
        # names = file_name.split('/')
        # folder_name = names[-2]
        # img_name = names[-1]
        # pkl_name = names[-1][:-4]+'.pkl'

        # data = joblib.load('/data/ADS/bev_results/agora_results/'+pkl_name)
        # pred_results['pred_vertices'] = torch.tensor(data['verts'])
        # pred_results['pred_translation'] = torch.tensor(data['trans']).unsqueeze(1) # n, 1, 3
        ##### for bev #####

        gt_num_people = data_batch['gt_kpts3d'].data[0][0].shape[0]
        file_name = data_batch['img_metas'].data[0][0]['file_name']
        metric_values = {
                     'mpjpe': [],
                     'pve': [],
                     'pampjpe': [],
                     'gce': [],
                     'gpve': [],
                     'avg_gpve': []
                     }

        for pidx in range(gt_num_people):
            packs = self.handle_single(data_batch, pred_results, pidx, use_gt=use_gt)
            if packs == None:
                continue
            for k in packs:
                metric_values[k].append(packs[k])

        for k in metric_values:
            metric_values[k] = np.mean(metric_values[k])

        metric_values.update({
            'file_name': file_name,
            # 'pred_rotmat': pred_results['pred_rotmat'],
            # 'pred_betas': pred_results['pred_betas'],
        })

        return metric_values

    def handle_single(self, data_batch, pred_results, pidx, use_gt=False):
        pred_verts = pred_results['pred_vertices'].float().cpu()
        pred_trans = pred_results['pred_translation'].float().squeeze(1).detach().cpu().numpy()  # n 3

        # gt_kpts2d = data_batch['gt_kpts2d'].data[0][0][pidx:pidx+1].detach().cpu().numpy()
        # gt_center_point = get_center_point(gt_kpts2d) # (1, 2)
        gt_trans = data_batch['gt_camera_trans'].data[0][0].clone()[pidx].unsqueeze(0).detach().cpu().numpy() # n ,3
        img_path = data_batch['img_metas'].data[0][0]['file_name']
        filename = osp.basename(img_path)

        # dict_keys(['pred_rotmat', 'pred_betas', 'pred_camera', 'pred_vertices', 'pred_joints', 'pred_trans', 'pred_translation', 'bboxes'])
        img = data_batch['img'].data[0][0]
        # img = data_batch['img'].data[0][0].detach().cpu().numpy()
        _, H, W = img.shape
        # FOCAL_LENGTH = 256/832 * 1000
        FOCAL_LENGTH = 1000
        # camMat = np.array([[FOCAL_LENGTH, 0, W/2],
        #         [0, FOCAL_LENGTH, H/2],
        #         [0, 0, 1]]).astype(np.float32)
        # RT = np.concatenate((np.diag([1., 1., 1.]), np.zeros((3, 1))), axis=1)
        # pred_joints = pred_results['pred_joints'].float().detach().cpu().numpy()

        # J_regressor_batch = self.J_regressor[None, :].expand(pred_verts.shape[0], -1, -1).to(pred_verts.device)
        # pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_verts)
        # pred_joints = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]

        # gt_verts = data_batch['gt_vertices'].data[0][0][pidx:pidx+1] # already inference with genders
        # gt_J_regressor_batch = self.J_regressor[None, :].to(gt_verts.device)
        # gt_keypoints_3d_smpl = torch.matmul(gt_J_regressor_batch, gt_verts)
        # gt_keypoints_3d_smpl = gt_keypoints_3d_smpl[:, H36M_TO_J14, :]
        # gt_trans = data_batch['gt_camera_trans'].data[0][0].clone().unsqueeze(0) # n ,3
        # # gt_keypoints_3d_smpl = gt_keypoints_3d_smpl + gt_trans
        # gt_keypoints_3d_smpl = gt_keypoints_3d_smpl + pred_trans[:, None, :]

        # print("pred_joints.shape: ", pred_joints.shape)
        # pred_joints_global = pred_joints[:, :, :3] + pred_trans[:, None, :]
        # pred_joints_F1000_proj = np.zeros((pred_joints.shape[0], pred_joints.shape[1], 2))
        # # gt_keypoints_3d_smpl_proj = np.zeros((pred_joints.shape[0], pred_joints.shape[1], 2))
        # for i in range(pred_joints.shape[0]):
        #     for j in range(pred_joints.shape[1]):
        #         pred_joints_F1000_proj[i, j, :] = project_point_np(np.concatenate([pred_joints_global[i, j, :], np.array([1])]), RT, camMat)
        #         # gt_keypoints_3d_smpl_proj[i, j, :] = project_point_np(np.concatenate([gt_keypoints_3d_smpl[i, j, :], np.array([1])]), RT, camMat)
        # pred_center_point = get_center_point(pred_joints_F1000_proj) # (100, 2)
        # # gt_kpts2d = gt_keypoints_3d_smpl_proj

        # pred_bbox 与 gt_bbox距离
        pred_bboxes = pred_results['bboxes'][0]
        gt_bboxes = data_batch['gt_bboxes'].data[0][0][pidx:pidx+1].detach().cpu().numpy()

        img = np.float32(img.cpu().numpy())
        img_viz = cv2.cvtColor((denormalize(img.transpose([1, 2, 0])) * 255).astype(np.uint8).copy(), cv2.COLOR_BGR2RGB)
        for i in range(pred_bboxes.shape[0]):
            bbox = pred_bboxes[i]
            img_viz = cv2.rectangle(img_viz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        # select high confidence
        # high_conf_mask = pred_bboxes[:, 4] > 0.8
        high_conf_mask = pred_bboxes[:, 4] > 0.3
        if high_conf_mask.shape[0] > 0:
            pred_bboxes = pred_bboxes[high_conf_mask]
            pred_verts = pred_verts[high_conf_mask]
            pred_trans = pred_trans[high_conf_mask]

        if pred_bboxes.shape[0] == 0:
            print(">>>> pred_bboxes.shape[0] == 0")
            return None
        # print(pred_bboxes.shape, pred_verts.shape, pred_trans.shape)

        pred_center_point_x = (pred_bboxes[:, 2] + pred_bboxes[:, 0]) * 0.5
        pred_center_point_y = (pred_bboxes[:, 3] + pred_bboxes[:, 1]) * 0.5
        pred_center_point = np.zeros((pred_bboxes.shape[0], 2)).astype(np.float32)
        pred_center_point[:, 0] = pred_center_point_x
        pred_center_point[:, 1] = pred_center_point_y

        gt_center_point_x = (gt_bboxes[:, 2] + gt_bboxes[:, 0]) * 0.5
        gt_center_point_y = (gt_bboxes[:, 3] + gt_bboxes[:, 1]) * 0.5
        gt_center_point = np.zeros((gt_bboxes.shape[0], 2)).astype(np.float32)
        gt_center_point[:, 0] = gt_center_point_x
        gt_center_point[:, 1] = gt_center_point_y

        dist_array = np.linalg.norm(pred_center_point - gt_center_point, axis=1) 
        min_idx = np.argmin(dist_array)
        # print(min_idx)
        # print(len(dist_array), pred_bboxes.shape, pred_verts.shape, pred_trans.shape)
        pred_verts = pred_verts[min_idx:min_idx+1]
        pred_trans = pred_trans[min_idx:min_idx+1]
        pred_bboxes = pred_bboxes[min_idx:min_idx+1]
        # pred_joints_F1000_proj = pred_joints_F1000_proj[min_idx:min_idx+1]

        ################
        # visual start #
        ################
        pred_center_tuple = (pred_center_point[min_idx][0], pred_center_point[min_idx][1])
        gt_center_tuple = (gt_center_point[0][0], gt_center_point[0][1])
        img_viz = cv2.circle(img_viz, pred_center_tuple, 5, (0, 255, 0), -1)
        img_viz = cv2.circle(img_viz, gt_center_tuple, 5, (0, 0, 255), -1)
        for i in range(pred_bboxes.shape[0]):
            bbox = pred_bboxes[i]
            img_viz = cv2.rectangle(img_viz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 4)
        for i in range(gt_bboxes.shape[0]):
            bbox = gt_bboxes[i]
            img_viz = cv2.rectangle(img_viz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        
        # output_path = f"./debug/centerpoint_eval.jpg"
        # cv2.imwrite(output_path, img_viz[:, :, :])
        # print(output_path)

        # # 可视化 pre_verts和gt_verts
        gt_verts = data_batch['gt_vertices'].data[0][0][pidx:pidx+1] # already inference with genders
        # # print("pred_verts.shape, gt_verts.shape: ", pred_verts.shape, gt_verts.shape)
        # # print("pred_trans: ", pred_trans)

        # # in stageI, pred_trans is zero
        # FOCAL_LENGTH = 256/832 * 1000
        # # FOCAL_LENGTH = 1000
        # # print('pred_bboxes.shape: ', pred_bboxes.shape)
        # # bboxes_size = np.max(np.abs(pred_bboxes[:, 0] - pred_bboxes[:, 2]), np.abs(pred_bboxes[:, 1] - pred_bboxes[:, 3]), axis=0)
        # bboxes_size = max(np.abs(pred_bboxes[:, 0] - pred_bboxes[:, 2]), np.abs(pred_bboxes[:, 1] - pred_bboxes[:, 3]))
        # bboxes_size = bboxes_size[0]
        # center_pts = (pred_bboxes[:, :2] + pred_bboxes[:, 2:4]) / 2
        # center_pts = torch.from_numpy(center_pts)
        # image_center_np = np.array([H, W]) * 0.5
        # image_center = torch.from_numpy(image_center_np)
        # pred_camera = pred_results['pred_camera'].cpu()[min_idx:min_idx+1]
        # # print('FOCAL_LENGTH: ', FOCAL_LENGTH)
        # depth = 2 * FOCAL_LENGTH / (pred_camera[:, 0] * bboxes_size + 1e-9)
        # depth = depth.unsqueeze(0)
        # render_translation = torch.zeros((1, 3), dtype=pred_camera.dtype).to(pred_camera.device)
        # render_translation[:, :-1] = pred_camera[:, 1:] + (center_pts - image_center) * depth / FOCAL_LENGTH
        # render_translation[:, -1] = depth[:, 0]

        # pred_translation = render_translation.cpu().numpy()
        # gt_translation = render_translation.cpu().numpy()

        pred_translation = pred_trans
        # pred_translation = gt_trans
        gt_translation = gt_trans

        img_viz = prepare_dump(data_batch, pred_verts, gt_verts, pred_translation, gt_translation, img_viz, FOCAL_LENGTH=FOCAL_LENGTH)
        output_path = f"./debug/centerpoint_mesh_{filename[:-4]}.jpg"
        cv2.imwrite(output_path, img_viz[:, :, :])
        print("smpl img_viz", img_viz.shape, output_path)

        # ################
        # # visual end   #
        # ################

        J_regressor_batch = self.J_regressor[None, :].expand(pred_verts.shape[0], -1, -1).to(pred_verts.device)
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_verts)
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_pelvis = (pred_keypoints_3d_smpl[:,[2],:] + pred_keypoints_3d_smpl[:,[3],:]) / 2.0
        pred_keypoints_3d_smpl -= pred_pelvis

        gt_verts = data_batch['gt_vertices'].data[0][0][pidx:pidx+1] # already inference with genders
        gt_J_regressor_batch = self.J_regressor[None, :].to(gt_verts.device)
        gt_keypoints_3d_smpl = torch.matmul(gt_J_regressor_batch, gt_verts)
        gt_keypoints_3d_smpl = gt_keypoints_3d_smpl[:, H36M_TO_J14, :]
        gt_pelvis = (gt_keypoints_3d_smpl[:, [2], :] + gt_keypoints_3d_smpl[:, [3], :]) / 2.0
        gt_keypoints_3d_smpl -= gt_pelvis

        # gt_keypoints_3d_smpl_copy = gt_keypoints_3d_smpl.clone()
        # gt_keypoints_3d_smpl = gt_keypoints_3d_smpl.repeat([pred_num_people, 1, 1])

        # mpjpe
        error_smpl = torch.sqrt(((pred_keypoints_3d_smpl.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        mpjpe = float(error_smpl * 1000)
        self.p1_meter.update(mpjpe)

        # pampjpe
        S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d_smpl)
        pampjpe = torch.sqrt(((S1_hat.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        pampjpe = float(pampjpe * 1000)
        self.p2_meter.update(pampjpe)

        S1_hat = batch_compute_similarity_transform_torch(pred_verts, gt_verts)
        papve = np.mean(np.sqrt(np.sum((S1_hat.detach().cpu().numpy() - gt_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        self.papve_meter.update(papve)

        # pve # should match each 
        pve = np.mean(np.sqrt(np.sum((gt_verts.detach().cpu().numpy() - pred_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        self.p3_meter.update(pve)

        #########global metrics########
        gt_verts = gt_verts.detach().cpu().numpy()   
        gt_verts += gt_trans[:,None]

        # pred_trans = pred_results['pred_translation'].squeeze(1).detach().cpu()  # n 3
        pred_verts = pred_verts.detach().cpu().numpy()
        pred_verts += pred_trans[:, None]

        # gpve
        gpve = calc_gpve(gt_verts, pred_verts)
        self.gpve_meter.update(gpve)

        # avg_gpve
        avg_gpve = calc_avg_gpve(gt_verts, pred_verts)
        self.avg_gpve_meter.update(avg_gpve)

        # gce
        gt_pelvis += gt_trans[:,None]
        gt_pelvis = gt_pelvis.detach()
        pred_pelvis += pred_trans[:, None]
        pred_pelvis = pred_pelvis.detach()
        gt_pelvis = gt_pelvis.detach().cpu().numpy()
        pred_pelvis = pred_pelvis.detach().cpu().numpy()
        gce = calc_gce(gt_pelvis[:,0], pred_pelvis[:,0], pelvis_index=None)
        self.gce_meter.update(gce)

        # logger for table2
        self.logger.write('%.2f_%.2f_%.2f_%.2f\n' % (gt_trans[0,1], gt_trans[0,2], pampjpe, gpve))
        # print('%.2ff_%.2f_%.2f_%.2f\n' % (gt_trans[0,1], gt_trans[0,2], pampjpe, gpve))

        save_pack = {#'file_name': file_name,
                     'mpjpe': mpjpe,
                     'pve': pve,
                     'pampjpe': pampjpe,
                     'gce': gce,
                     'gpve': gpve,
                     'avg_gpve': avg_gpve,
                     }
        return save_pack

    def compute_error_verts(self, pred_verts, target_shape=None, target_pose=None, target_verts=None):
        """
        Computes MPJPE over 6890 surface vertices.
        Args:
            verts_gt (Nx6890x3).
            verts_pred (Nx6890x3).
        Returns:
            error_verts (N).
        """

        if target_verts is None:

            betas = target_shape
            pose = target_pose

            target_verts = []
            b_ = torch.split(betas, 5000)
            p_ = torch.split(pose, 5000)

            for b,p in zip(b_,p_):
                #print(b.shape, p.shape)
                print("agora compute_error_verts p shape: ", p.shape)
                output = self.smpl(betas=b, body_pose=p[:, 1:], global_orient=p[:, :1], pose2rot=True)
                target_verts.append(output.vertices.detach().cpu().numpy())

            target_verts = np.concatenate(target_verts, axis=0)

        assert len(pred_verts) == len(target_verts)
        error_per_vert = np.sqrt(np.sum((target_verts - pred_verts.detach().cpu().numpy()) ** 2, axis=2))
        return np.mean(error_per_vert, axis=1)


    def log(self):
        self.writer(f'mpjpe: {self.p1_meter.avg:.2f}mm, pampjpe: {self.p2_meter.avg:.2f}mm, pve: {self.p3_meter.avg:.2f}mm, papve: {self.papve_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm, avg_gpve: {self.avg_gpve_meter.avg:.2f}mm agora')





class AGORAHandlerROMPBEV(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', pattern='.60457274_', **kwargs):
        super().__init__(**kwargs)

        self.enable_LSP14_joint_order = False
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        
        self.pattern = pattern
        self.p1_meter = AverageMeter('P1', ':.3f')
        self.p2_meter = AverageMeter('P2', ':.3f')
        self.p3_meter = AverageMeter('P3', ':.3f')

        self.papve_meter = AverageMeter('papve', ':.3f')
        self.avg_gpve_meter = AverageMeter('avg_gpve', ':.3f')
        
        self.gce_meter = AverageMeter('P3', ':.3f')
        self.gpve_meter = AverageMeter('P3', ':.3f')
        self.saved = False
        self.smpl = SMPL('data/smpl').to('cpu')

        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/ads_y.txt', 'w+')
        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/ads_y_z_V1_2022_11_09.txt', 'w')
        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/val_ads_zy_2022_11_09_V1.txt', 'w')
        # self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/val_ads_zy_2022_11_13_V1.txt', 'w')
        # self.logger = open('./data/table2/ADRT_table2_agora_val_ads_zy_2023_03_04.txt', 'w')
        # self.logger = open('./data/table2/ROMP_table2_agora_val_ads_zy_2023_03_04.txt', 'w')
        # self.logger = open('./data/table2/BEV_focal_scale_table2_agora_val_ads_zy_2023_03_04.txt', 'w')
        # self.logger = open('./data/table2/ROMP_focal_scale_table2_agora_val_ads_zy_2023_03_04.txt', 'w')
        self.logger = open('./data/table2/CRMH_focal_scale_table2_agora_val_ads_zy_2023_03_04.txt', 'w')

    def get_gt_joints(self, gt_shape, gt_pose, J_regressor_batch):
        output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 1:], global_orient=gt_pose[:, :1], pose2rot=True)
        verts = output.vertices
        if self.enable_LSP14_joint_order:
            j3d = output.joints
            j3d = j3d[:, ADS24_TO_LSP14, :]
        else:
            j3d = torch.matmul(J_regressor_batch, verts)
            j3d = j3d[:, H36M_TO_J14, :]

        return j3d, verts

    def handle(self, data_batch, pred_results, use_gt=False):

        ##### for bev #####
        # file_name = data_batch['img_metas'].data[0][0]['file_name'] # images/S9_Directions_1.60457274_000201.jpg
        # #print(file_name,'???????')
        # names = file_name.split('/')
        # folder_name = names[-2]
        # img_name = names[-1]
        # pkl_name = names[-1][:-4]+'.pkl'

        # data = joblib.load('/data/ADS/bev_results/agora_results/'+pkl_name)
        # pred_results['pred_vertices'] = torch.tensor(data['verts'])
        # pred_results['pred_translation'] = torch.tensor(data['trans']).unsqueeze(1) # n, 1, 3
        ##### for bev #####

        gt_num_people = data_batch['gt_kpts3d'].data[0][0].shape[0]
        file_name = data_batch['img_metas'].data[0][0]['file_name']
        metric_values = {
                     'mpjpe': [],
                     'pve': [],
                     'pampjpe': [],
                     'gce': [],
                     'gpve': [],
                     'avg_gpve': []
                     }

        for pidx in range(gt_num_people):
            packs = self.handle_single(data_batch, pred_results, pidx, use_gt=use_gt)
            for k in packs:
                metric_values[k].append(packs[k])

        for k in metric_values:
            metric_values[k] = np.mean(metric_values[k])

        metric_values.update({
            'file_name': file_name,
            # 'pred_rotmat': pred_results['pred_rotmat'],
            # 'pred_betas': pred_results['pred_betas'],
        })

        return metric_values

    def handle_single(self, data_batch, pred_results, pidx, use_gt=False):
        pred_center_preds = pred_results['pred_center_preds'].float().detach().cpu().numpy()
        gt_center_preds = pred_results['gt_center_preds'].float().detach().cpu().numpy()
        pred_verts = pred_results['pred_vertices'].float().detach().cpu()
        pred_trans = pred_results['pred_translation'].float().squeeze(1).detach().cpu().numpy()  # n 3
        # pred_num_people = pred_verts.shape[0]

        # for ROMP and BEV 应该选择center map最近的pair来算
        # pred_bbox 与 gt_bbox距离
        dist_array = np.linalg.norm(pred_center_preds - gt_center_preds[pidx], axis=1) 
        min_idx = np.argmin(dist_array)
        # print(dist_array)
        # print(min_idx)
        pred_verts = pred_verts[min_idx:min_idx+1]
        pred_trans = pred_trans[min_idx:min_idx+1]

        # J_regressor_batch = self.J_regressor[None, :].to(pred_verts.device)
        J_regressor_batch = self.J_regressor[None, :].expand(pred_verts.shape[0], -1, -1).to(pred_verts.device)

        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_verts)
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_pelvis = (pred_keypoints_3d_smpl[:,[2],:] + pred_keypoints_3d_smpl[:,[3],:]) / 2.0
        pred_keypoints_3d_smpl -= pred_pelvis

        gt_verts = data_batch['gt_vertices'].data[0][0][pidx:pidx+1] # already inference with genders
        gt_J_regressor_batch = self.J_regressor[None, :].to(gt_verts.device)
        gt_keypoints_3d_smpl = torch.matmul(gt_J_regressor_batch, gt_verts)
        gt_keypoints_3d_smpl = gt_keypoints_3d_smpl[:, H36M_TO_J14, :]
        gt_pelvis = (gt_keypoints_3d_smpl[:, [2], :] + gt_keypoints_3d_smpl[:, [3], :]) / 2.0
        gt_keypoints_3d_smpl -= gt_pelvis

        # gt_keypoints_3d_smpl_copy = gt_keypoints_3d_smpl.clone()
        # gt_keypoints_3d_smpl = gt_keypoints_3d_smpl.repeat([pred_num_people, 1, 1])

        # mpjpe
        error_smpl = torch.sqrt(((pred_keypoints_3d_smpl.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        mpjpe = float(error_smpl.min() * 1000)
        self.p1_meter.update(mpjpe)

        # pampjpe
        S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d_smpl)
        pampjpe = torch.sqrt(((S1_hat.double() - gt_keypoints_3d_smpl.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        pampjpe = float(pampjpe * 1000)
        self.p2_meter.update(pampjpe)

        # S1_hat = batch_compute_similarity_transform_torch(pred_verts[min_idx].unsqueeze(0), gt_verts)
        S1_hat = batch_compute_similarity_transform_torch(pred_verts, gt_verts)
        papve = np.mean(np.sqrt(np.sum((S1_hat.detach().cpu().numpy() - gt_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        self.papve_meter.update(papve)

        # pve # should match each 
        pve = np.mean(np.sqrt(np.sum((gt_verts.detach().cpu().numpy() - pred_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        self.p3_meter.update(pve)

        #########global metrics########
        gt_trans = data_batch['gt_camera_trans'].data[0][0].clone()[pidx].unsqueeze(0) # n ,3
        gt_verts = gt_verts.detach()   
        gt_verts += gt_trans[:,None]

        # pred_trans = pred_results['pred_translation'].squeeze(1).detach().cpu()  # n 3
        pred_verts = pred_verts.detach()
        pred_verts += pred_trans[:, None]

        gt_verts = gt_verts.detach().cpu().numpy()
        pred_verts = pred_verts.detach().cpu().numpy()

        # gpve
        gpve = calc_gpve(gt_verts, pred_verts)
        self.gpve_meter.update(gpve)

        # avg_gpve
        avg_gpve = calc_avg_gpve(gt_verts, pred_verts)
        self.avg_gpve_meter.update(avg_gpve)

        # gce
        gt_pelvis += gt_trans[:,None]
        gt_pelvis = gt_pelvis.detach()
        pred_pelvis += pred_trans[:, None]
        pred_pelvis = pred_pelvis.detach()
        gt_pelvis = gt_pelvis.detach().cpu().numpy()
        pred_pelvis = pred_pelvis.detach().cpu().numpy()
        gce = calc_gce(gt_pelvis[:,0], pred_pelvis[:,0], pelvis_index=None)
        self.gce_meter.update(gce)

        # logger for table2
        self.logger.write('%.2f_%.2f_%.2f_%.2f\n' % (gt_trans[0,1], gt_trans[0,2], pampjpe, gpve))
        # print('%.2ff_%.2f_%.2f_%.2f\n' % (gt_trans[0,1], gt_trans[0,2], pampjpe, gpve))

        save_pack = {#'file_name': file_name,
                     'mpjpe': mpjpe,
                     'pve': pve,
                     'pampjpe': pampjpe,
                     'gce': gce,
                     'gpve': gpve,
                     'avg_gpve': avg_gpve,
                     }
        return save_pack

    def compute_error_verts(self, pred_verts, target_shape=None, target_pose=None, target_verts=None):
        """
        Computes MPJPE over 6890 surface vertices.
        Args:
            verts_gt (Nx6890x3).
            verts_pred (Nx6890x3).
        Returns:
            error_verts (N).
        """

        if target_verts is None:

            betas = target_shape
            pose = target_pose

            target_verts = []
            b_ = torch.split(betas, 5000)
            p_ = torch.split(pose, 5000)

            for b,p in zip(b_,p_):
                #print(b.shape, p.shape)
                print("agora compute_error_verts p shape: ", p.shape)
                output = self.smpl(betas=b, body_pose=p[:, 1:], global_orient=p[:, :1], pose2rot=True)
                target_verts.append(output.vertices.detach().cpu().numpy())

            target_verts = np.concatenate(target_verts, axis=0)

        assert len(pred_verts) == len(target_verts)
        error_per_vert = np.sqrt(np.sum((target_verts - pred_verts.detach().cpu().numpy()) ** 2, axis=2))
        return np.mean(error_per_vert, axis=1)


    def log(self):
        self.writer(f'mpjpe: {self.p1_meter.avg:.2f}mm, pampjpe: {self.p2_meter.avg:.2f}mm, pve: {self.p3_meter.avg:.2f}mm, papve: {self.papve_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm, avg_gpve: {self.avg_gpve_meter.avg:.2f}mm agora')


class AGORAHandlerV2(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', pattern='.60457274_', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.pattern = pattern
        self.p1_meter = AverageMeter('P1', ':.3f')
        self.p2_meter = AverageMeter('P2', ':.3f')
        self.p3_meter = AverageMeter('P3', ':.3f')

        self.avg_gpve_meter = AverageMeter('avg_gpve', ':.3f')
        self.gce_meter = AverageMeter('P3', ':.3f')
        self.gpve_meter = AverageMeter('P3', ':.3f')
        self.papve_meter = AverageMeter('papve', ':.3f')

        self.saved = False
        self.smpl = SMPL(
                'data/smpl'
            ).to('cpu')
        self.logger = open('/home/haoye/codes/ADS-internal/mmdetection/fig3/val_ads_zy.txt', 'w+')

    def get_gt_joints(self, gt_shape, gt_pose, J_regressor_batch):
        # print("agora v2 gt_pose shape: ", gt_pose.shape)
        output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 1:], global_orient=gt_pose[:, :1], pose2rot=True) 
        # gt_pose shape:  torch.Size([10, 24, 3])
        # 要debug检查下训练代码是不是也是这样[:, 3:]
        # output = self.smpl(betas=gt_shape, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3], pose2rot=True)
        #output = smpl(betas=gt_shape, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3], transl=gt_trans)
        verts = output.vertices

        j3d = torch.matmul(J_regressor_batch, verts)
        j3d = j3d[:, H36M_TO_J14, :]
        #print(j3d.shape, verts.shape)
        return j3d, verts

    def handle(self, data_batch, pred_results, use_gt=False):
        ##### for bev #####
        # file_name = data_batch['img_metas'].data[0][0]['file_name'] # images/S9_Directions_1.60457274_000201.jpg
        # #print(file_name,'???????')
        # names = file_name.split('/')
        # folder_name = names[-2]
        # img_name = names[-1]
        # pkl_name = names[-1][:-4]+'.pkl'

        # data = joblib.load('/data/ADS/bev_results/agora_results/'+pkl_name)
        # pred_results['pred_vertices'] = torch.tensor(data['verts'])
        # pred_results['pred_translation'] = torch.tensor(data['trans']).unsqueeze(1) # n, 1, 3
        ##### for bev #####

        gt_num_people = data_batch['gt_kpts3d'].data[0][0].shape[0]
        file_name = data_batch['img_metas'].data[0][0]['file_name']
        metric_values = {
                     'MPJPE': [],
                     'pve': [],
                     'papve': [],
                     'PAMPJPE': [],
                     # global metrics
                     'gce': [],
                     'gpve': []}

        #for pidx in range(gt_num_people):
        packs = self.handle_single(data_batch, pred_results, 0, use_gt=use_gt)

        for k in packs:
            metric_values[k].append(packs[k])

        for k in metric_values:
            metric_values[k] = np.mean(metric_values[k])

        metric_values.update({
            'file_name': file_name,
            'pred_rotmat': pred_results['pred_rotmat'],
            'pred_betas': pred_results['pred_betas'],
        })

        return metric_values

    def handle_single(self, data_batch, pred_results, pidx, use_gt=False):
        gt_num_people = data_batch['gt_kpts3d'].data[0][0].shape[0]
        o_device = pred_results['pred_vertices'].device
        pred_vertices = pred_results['pred_vertices'].cpu()
        pred_trans = pred_results['pred_translation'].squeeze(1).detach().cpu().numpy()  # n 3

        pred_num_people = pred_vertices.shape[0]

        J_regressor_batch = self.J_regressor[None, :].to(
            pred_vertices.device)

        gt_trans = data_batch['gt_camera_trans'].data[0][0].clone().detach().cpu().numpy()  # n ,3
        gt_keypoints_3d, gt_verts = self.get_gt_joints(data_batch['gt_shapes'].data[0][0].clone(), 
                                                       data_batch['gt_poses'].data[0][0].clone(),
                                                       J_regressor_batch)
 
        gt_pelvis = (gt_keypoints_3d[:,[2],:] + gt_keypoints_3d[:,[3],:]) / 2.0
        gt_keypoints_3d -= gt_pelvis

        visible_kpts = gt_keypoints_3d[:, :, -1].clone()


        # # 可视化 pre_verts和gt_verts
        # print("pred_vertices.shape, gt_verts.shape: ", pred_vertices.shape, gt_verts.shape)
        # img_viz = prepare_dump(data_batch, pred_vertices, gt_verts, pred_trans)
        # cv2.imwrite(f"./debug_eval_imgs/agora_eval.jpg", img_viz[:, :, :])
        # print("smpl img_viz", img_viz.shape)


        # Get 14 predicted joints from the SMPL mesh
        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
            pred_vertices.device)
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)

        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_pelvis = (pred_keypoints_3d_smpl[:,[2],:] + pred_keypoints_3d_smpl[:,[3],:]) / 2.0
        pred_keypoints_3d_smpl -= pred_pelvis

        # match
        # To select closest points
        glb_vis = (visible_kpts.sum(0) >= (
                visible_kpts.shape[0] - 0.1)).float()[None, :, None]  # To avoid in-accuracy in float point number
        if use_gt:
            paired_idxs = torch.arange(gt_keypoints_3d.shape[0])
        else:
            # dist = vectorize_distance((glb_vis * (gt_keypoints_3d + gt_trans[:,None])).numpy(),
            #                           (glb_vis * (pred_keypoints_3d_smpl + pred_trans[:,None])).numpy())
            # dist = vectorize_distance((gt_keypoints_3d + gt_trans[:,None]).numpy(),
            #                           (pred_keypoints_3d_smpl + pred_trans[:,None]).numpy())
            
            # print(gt_trans.shape)
            dist = vectorize_distance(gt_trans,
                                      pred_trans)
            # paired_idxs = torch.from_numpy(dist.argmin(1))
            # for every prediction find closest GT
            paired_idxs = dist.argmin(1)
            #print('???')
            
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[paired_idxs]
        pred_vertices = pred_vertices[paired_idxs]
        pred_pelvis = pred_pelvis[paired_idxs]
        pred_trans = pred_trans[paired_idxs]
        if len(pred_trans.shape) == 1:
            pred_trans = pred_trans[None,...]
        ################
        
        #file_name = data_batch['img_metas'].data[0][0]['file_name']
        # error_smpl = (torch.sqrt(((pred_keypoints_3d_smpl - gt_keypoints_3d) ** 2).sum(dim=-1)) * visible_kpts)
        error_smpl = torch.sqrt(((pred_keypoints_3d_smpl.double() - gt_keypoints_3d.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        mpjpe = float(error_smpl.mean() * 1000)
        self.p1_meter.update(mpjpe)

        # pampjpe
        # print(pred_keypoints_3d_smpl.shape, gt_keypoints_3d.shape)
        S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d)
        errors_pa = torch.sqrt(((S1_hat.double() - gt_keypoints_3d.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        min_idx = torch.argmin(errors_pa, dim=0)
        flatten_pampjpe = errors_pa * 1000.
        pampjpe  = float(flatten_pampjpe.mean())
        self.p2_meter.update(pampjpe)

        # !!!! 这里 gt_keypoints_3d.repeat的原因是？
        # S1_hat = batch_compute_similarity_transform_torch(pred_keypoints_3d_smpl, gt_keypoints_3d.repeat([pred_num_people, 1, 1]))
        # errors_pa = torch.sqrt(((S1_hat.double() - gt_keypoints_3d.double()) ** 2).sum(dim=-1)).mean(dim=-1)
        # min_idx = torch.argmin(errors_pa, dim=0)
        # pampjpe  = float(errors_pa.min() * 1000)
        # self.p1_meter.update(pampjpe)

        # pve
        flatten_pve = np.mean(np.sqrt(np.sum((gt_verts.detach().cpu().numpy() - pred_vertices.detach().cpu().numpy()) ** 2, axis=2)), axis=1)
        
        pve = flatten_pve.mean() * 1000
        #pve = float(np.mean(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        #pve = float(self.compute_error_verts(pred_verts=pred_vertices.to(o_device), target_shape=gt_shapes, target_pose=gt_poses).min() * 1000)
        self.p3_meter.update(pve)

        # pa-pve
        # z
        # S1_hat = batch_compute_similarity_transform_torch(pred_vertices[min_idx].unsqueeze(0), gt_verts)
        S1_hat = batch_compute_similarity_transform_torch(pred_vertices, gt_verts)

        papve = np.mean(np.sqrt(np.sum((S1_hat.detach().cpu().numpy() - gt_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).mean() * 1000
        # papve = np.mean(np.sqrt(np.sum((S1_hat.detach().cpu().numpy() - gt_verts.detach().cpu().numpy()) ** 2, axis=2)), axis=1).min() * 1000
        self.papve_meter.update(papve)

        #########global metrics########
        

        #print(gt_verts.shape, gt_trans.shape, pred_vertices.shape, pred_trans.shape)
        gt_verts = gt_verts.detach().cpu().numpy()
        gt_verts += gt_trans[:,None]

        pred_vertices = pred_vertices.detach().cpu().numpy()
        pred_vertices += pred_trans[:, None]

        gpve = calc_gpve(gt_verts, pred_vertices, mode='none')
        flatten_gpve = gpve
        self.gpve_meter.update(gpve.mean())

        avg_gpve = calc_avg_gpve(gt_verts, pred_vertices, mode='none')
        self.avg_gpve_meter.update(avg_gpve.mean())

        gt_pelvis += gt_trans[:,None]
        gt_pelvis = gt_pelvis.detach().cpu().numpy()

        pred_pelvis += pred_trans[:, None]
        pred_pelvis = pred_pelvis.detach().cpu().numpy()

        gce = calc_gce(gt_pelvis[:,0], pred_pelvis[:,0], pelvis_index=None, mode='none')
        self.gce_meter.update(gce.mean())

        #### logger #####
        for kk in range(gt_trans.shape[0]):
            self.logger.write('%.2f_%.2f_%.2f_%.2f\n' % (gt_trans[kk,1], gt_trans[kk,2], flatten_pampjpe[kk], flatten_gpve[kk]))
            print('%.2f_%.2f_%.2f_%.2f\n' % (gt_trans[kk,1], gt_trans[kk,2], flatten_pampjpe[kk], flatten_gpve[kk]))
        #### logger #####


        save_pack = {#'file_name': file_name,
                     'MPJPE': mpjpe,
                     'PAMPJPE': pampjpe,
                     'pve': pve,
                     'papve': papve,
                     #'pred_rotmat': pred_results['pred_rotmat'],
                     #'pred_betas': pred_results['pred_betas'],
                     # global metrics
                     
                     'gce': gce,
                     'gpve': gpve,
                     #'avg_gpve': avg_gpve,
                     }
        return save_pack

    def compute_error_verts(self, pred_verts, target_shape=None, target_pose=None, target_verts=None):
        """
        Computes MPJPE over 6890 surface vertices.
        Args:
            verts_gt (Nx6890x3).
            verts_pred (Nx6890x3).
        Returns:
            error_verts (N).
        """

        if target_verts is None:

            betas = target_shape
            pose = target_pose

            target_verts = []
            b_ = torch.split(betas, 5000)
            p_ = torch.split(pose, 5000)

            for b,p in zip(b_,p_):
                #print(b.shape, p.shape)
                print("agora v2 p shape: ", p.shape)
                output = self.smpl(betas=b, body_pose=p[:, 1:], global_orient=p[:, :1], pose2rot=True)
                # output = self.smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], pose2rot=True)
                target_verts.append(output.vertices.detach().cpu().numpy())

            target_verts = np.concatenate(target_verts, axis=0)

        assert len(pred_verts) == len(target_verts)
        error_per_vert = np.sqrt(np.sum((target_verts - pred_verts.detach().cpu().numpy()) ** 2, axis=2))
        return np.mean(error_per_vert, axis=1)


    def log(self):
        self.writer(f'mpjpe: {self.p1_meter.avg:.2f}mm, pampjpe: {self.p2_meter.avg:.2f}mm, pve: {self.p3_meter.avg:.2f}mm, papve: {self.papve_meter.avg:.2f}mm, gce: {self.gce_meter.avg:.2f}mm, gpve: {self.gpve_meter.avg:.2f}mm, avg_gpve: {self.avg_gpve_meter.avg:.2f}mm agora')

    def vectorize_distance(self, a, b):
        """
        Calculate euclid distance on each row of a and b
        :param a: Nx... np.array
        :param b: Mx... np.array
        :return: MxN np.array representing correspond distance
        """
        N = a.shape[0]
        a = a.view( N, -1 )
        M = b.shape[0]
        b = b.view( M, -1 )
        a2 = torch.sum( a ** 2, dim=1 ).view( -1, 1 ).expand(-1, M)
        b2 = torch.sum( b ** 2, dim=1 ).expand(N, -1)
        dist = a2 + b2 - 2 * (a @ b.T)
        return torch.sqrt(dist)