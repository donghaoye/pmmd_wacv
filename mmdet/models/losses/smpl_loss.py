import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weighted_loss
# from ..registry import LOSSES
from ..utils.smpl_utils import batch_rodrigues, perspective_projection, rotation_matrix_to_angle_axis
# from ..utils.pose_utils import reconstruction_error
from ..utils.smpl.smpl import SMPL

from mmdet.models.builder import HEADS, build_loss, LOSSES

import random
# from sdf import SDFLoss
# import neural_renderer as nr
import numpy as np
# import quaternion

# ROMP
from .prior_loss import angle_prior, MaxMixturePrior
from .romp_loss import calc_mpjpe, calc_pampjpe
import mmdet.datasets.constants as constants


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb


def batch_encoder_disc_l2_loss(disc_value):
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


@weighted_loss
def smpl_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@torch.no_grad()
def select_index(im_id, pids, metric, invalid_mask=None):
    im_id = im_id.clone().int()[:, 0]
    num_imgs = im_id.max().item()
    selected_idxs = list()
    full_idxs = torch.arange(im_id.shape[0], device=im_id.device)
    for bid in set(im_id.tolist()):
        batch_mask = bid == im_id
        cur_pids = pids[batch_mask]
        cur_select = list()
        for pid in set(cur_pids.tolist()):
            person_mask = (pid == cur_pids)
            idx_to_select = full_idxs[batch_mask][person_mask][metric[batch_mask][person_mask].argmax()]
            if invalid_mask and invalid_mask[idx_to_select]:
                continue
            cur_select.append(idx_to_select)
        selected_idxs.append(cur_select)
    return selected_idxs


def adversarial_loss(discriminator, pred_pose_shape, real_pose_shape):
    g_loss_disc = batch_encoder_disc_l2_loss(discriminator(pred_pose_shape))
    
    fake_pose_shape = pred_pose_shape.detach()
    # fake_pose_shape = pred_pose_shape
    fake_disc_value, real_disc_value = discriminator(fake_pose_shape), discriminator(real_pose_shape)
    d_disc_real, d_disc_fake, d_disc_loss = batch_adv_disc_l2_loss(real_disc_value, fake_disc_value)
    return g_loss_disc, d_disc_fake, d_disc_real


@LOSSES.register_module
class SMPLLoss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', re_weight=None,
                 normalize_kpts=False, pad_size=False, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy',
                 debugging=False, adversarial_cfg=None, use_sdf=False, FOCAL_LENGTH=1000,
                 kpts_loss_type='L1Loss', kpts_3d_loss_type=None, img_size=None,
                 use_trans=False,
                 not_use_256p_focal=False,
                 not_use_resTrans=False,
                 is_woLocalTrans=False,
                 is_woGlobalTransPart=False,
                 use_pejpe_loss=False,
                 depth_range=(0., 50.),
                 wo_validDepth=False,
                 wo_lossVertices=False,
                 wo_lossVertices_global=False,
                 gmm_path='/home/tiange/work/multiperson/mmdetection/data/model_data/parameters/', # ROMP
                 nr_batch_rank=False, 
                 inner_robust_sdf=None,
                 use_global_trans_feats=False,
                 **kwargs):
        super(SMPLLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.criterion_shape = nn.L1Loss(reduction='none')
        self.criterion_keypoints = getattr(nn, kpts_loss_type)(reduction='none')  # nn.L1Loss(reduction='none')
        if kpts_3d_loss_type is not None:
            self.criterion_3d_keypoints = getattr(nn, kpts_3d_loss_type)(reduction='none')
        # self.criterion_regr = nn.MSELoss(reduction='none')
        self.criterion_regr = nn.MSELoss()
        self.re_weight = re_weight
        self.normalize_kpts = normalize_kpts
        self.pad_size = pad_size
        self.FOCAL_LENGTH = FOCAL_LENGTH
        self.debugging = debugging

        self.use_trans = use_trans
        self.not_use_256p_focal = not_use_256p_focal
        self.not_use_resTrans = not_use_resTrans
        self.use_pejpe_loss = use_pejpe_loss
        self.depth_range = torch.tensor(depth_range, dtype=torch.float)
        self.is_woLocalTrans = is_woLocalTrans              # for wo local trans ablation 
        self.is_woGlobalTransPart = is_woGlobalTransPart      # for wo global trans ablation 
        self.wo_validDepth = wo_validDepth
        self.wo_lossVertices = wo_lossVertices
        self.wo_lossVertices_global = wo_lossVertices_global
        self.use_global_trans_feats = use_global_trans_feats

        # Initialize SMPL model
        self.smpl = SMPL('data/smpl')
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()

        self.adversarial_cfg = adversarial_cfg

        # ROMP
        # self.gmm_prior = MaxMixturePrior(prior_folder=gmm_path,num_gaussians=8,dtype=torch.float32).cuda()
        # self.shape_pca_weight = F.softmax(torch.Tensor([1, 0.32, 0.16, 0.16, 0.08, 0.08, 0.08, 0.04, 0.02, 0.01]).unsqueeze(0).float(), dim=1) * 10.
        self.align_idx = np.array([constants.ADS_SMPL_24['Right Hip'], constants.ADS_SMPL_24['Left Hip']])

        self.pose_prior = MaxMixturePrior(prior_folder=gmm_path, num_gaussians=8, dtype=torch.float32)

        self.use_sdf = use_sdf
        # if debugging:
        #     self.sdf_loss = SDFLoss(self.smpl.faces, debugging=self.debugging, robustifier=inner_robust_sdf)
        # else:
        #     self.sdf_loss = SDFLoss(self.smpl.faces, robustifier=inner_robust_sdf)
        self.nr_batch_rank = nr_batch_rank
        if self.nr_batch_rank:
            # setup renderer
            self.image_size = max(img_size)
            self.w_diff, self.h_diff = (self.image_size - img_size[0]) // 2, (self.image_size - img_size[1]) // 2

            self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.image_size,
                                               image_size=self.image_size,
                                               light_intensity_ambient=1,
                                               light_intensity_directional=0,
                                               anti_aliasing=False)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                bboxes_confidence=None,
                discriminator=None,
                nested=False,
                residual_depth=False,
                global_pelvis_depth=None,
                global_trans_feats=None,
                **kwargs):
        """

        :param pred: SMPL parameters with 24*6+10+3
        :param target: same as pred
        :param weight:
        :param avg_factor:
        :param reduction_override:
        :param kwargs:
        :param bboxes_confidence:
        :return: loss: dict. All the value whose keys contain 'loss' will be summed up.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        pred_rotmat = pred['pred_rotmat'] # rot6d to rotmat, B 24 3 3
        pred_camera = pred['pred_camera'] # B, 3 weak perspective camera
        pred_joints = pred['pred_joints'] # smpl joints
        pred_vertices = pred['pred_vertices'] # smpl vertices
        pred_betas = pred['pred_betas'] # B 10 
        #

        gt_rotmat = target['gt_rotmat']  # It's not rotmat actually. This is a (B, 24, 3) tensor.
        gt_shape = target['gt_shape']
        gt_camera = target['gt_camera']
        gt_keypoints_2d = target['gt_keypoints_2d']
        gt_keypoints_3d = target['gt_keypoints_3d']
        has_smpl = target['has_smpl']
        gt_vertices = target['gt_vertices']
        pred_bboxes = target['pred_bboxes']
        #
        raw_images = target['raw_images'] # 256, 256
        img_meta = target['img_meta']
        ori_shape = [i['ori_shape'] for i in img_meta]
        idxs_in_batch = target['idxs_in_batch']
        pose_idx = target['pose_idx']
        # scene = target['scene']

        #gt_visibility = target['gt_visibility']
        gt_camera_trans = target['gt_camera_trans'] # B, 3
        has_trans = target['has_trans']

        batch_size = pred_joints.shape[0]
        if self.pad_size:
            img_pad_shape = torch.tensor([i['pad_shape'][:2] for i in img_meta], dtype=torch.float32).to(
                pred_joints.device)
            img_size = img_pad_shape[idxs_in_batch[:, 0].long()]
        else:
            img_size = torch.zeros(batch_size, 2).to(pred_joints.device)
            img_size += torch.tensor(raw_images.shape[:-3:-1], dtype=img_size.dtype).to(img_size.device)
        
        
        valid_boxes = (torch.abs(pred_bboxes[..., 0] - pred_bboxes[..., 2]) > 5) & (torch.abs(pred_bboxes[..., 1] - pred_bboxes[..., 3]) > 5)
        center_pts = (pred_bboxes[..., :2] + pred_bboxes[..., 2:]) / 2
        bboxes_size = torch.max(torch.abs(pred_bboxes[..., 0] - pred_bboxes[..., 2]), torch.abs(pred_bboxes[..., 1] - pred_bboxes[..., 3]))
        translation = torch.zeros((batch_size, 3), dtype=pred_camera.dtype).to(pred_joints.device)

        focal_length = self.FOCAL_LENGTH 
        pred_trans = pred['pred_trans'] # B, 1, 3

        if self.is_woLocalTrans == True:
            if not self.wo_validDepth:
                self.depth_range = self.depth_range.to(global_pelvis_depth.device)
                valid_depth = (global_pelvis_depth > self.depth_range[0]) & (global_pelvis_depth < self.depth_range[1])
                valid_boxes = valid_boxes & valid_depth

            global_pelvis_depth = global_pelvis_depth.unsqueeze(-1)
            # depth = global_pelvis_depth + pred_trans[:, :, -1]
            depth = global_pelvis_depth

            translation[:, :-1] = pred_camera[:, 1:] + (center_pts - img_size / 2) * depth / self.FOCAL_LENGTH
            translation[:, -1] = depth[:, 0]
            translation[:, :2] = translation[:, :2] + pred_trans[:, 0, :2]
            # print(">>>>> self.is_woLocalTrans", self.is_woLocalTrans)

        elif self.is_woGlobalTransPart == True:
            if not self.wo_validDepth:
                self.depth_range = self.depth_range.to(global_pelvis_depth.device)
                valid_depth = (global_pelvis_depth > self.depth_range[0]) & (global_pelvis_depth < self.depth_range[1])
                valid_boxes = valid_boxes & valid_depth

            # global_pelvis_depth = global_pelvis_depth.unsqueeze(-1)
            # depth = global_pelvis_depth + pred_trans[:, :, -1]
            depth = pred_trans[:, :, -1]

            translation[:, :-1] = pred_camera[:, 1:] + (center_pts - img_size / 2) * depth / self.FOCAL_LENGTH
            translation[:, -1] = depth[:, 0]
            translation[:, :2] = translation[:, :2] + pred_trans[:, 0, :2]
            # print(">>>>> self.is_woGlobalTransPart", self.is_woGlobalTransPart)

        elif self.use_global_trans_feats == True:
            if not self.wo_validDepth:
                self.depth_range = self.depth_range.to(global_pelvis_depth.device)
                valid_depth = (global_pelvis_depth > self.depth_range[0]) & (global_pelvis_depth < self.depth_range[1])
                valid_boxes = valid_boxes & valid_depth

            depth = pred_trans[:, :, -1]

            translation[:, :-1] = pred_camera[:, 1:] + (center_pts - img_size / 2) * depth / self.FOCAL_LENGTH
            translation[:, -1] = depth[:, 0]
            translation[:, :2] = translation[:, :2] + pred_trans[:, 0, :2]
            # print(">>>>> self.use_global_trans_feats", self.use_global_trans_feats)

        elif self.use_trans and 'pred_trans' in pred and pred['pred_trans'] is not None and not self.not_use_resTrans:
            if not self.wo_validDepth:
                self.depth_range = self.depth_range.to(global_pelvis_depth.device)
                valid_depth = (global_pelvis_depth > self.depth_range[0]) & (global_pelvis_depth < self.depth_range[1])
                valid_boxes = valid_boxes & valid_depth

            global_pelvis_depth = global_pelvis_depth.unsqueeze(-1)
            depth = global_pelvis_depth + pred_trans[:, :, -1]

            translation[:, :-1] = pred_camera[:, 1:] + (center_pts - img_size / 2) * depth / self.FOCAL_LENGTH
            translation[:, -1] = depth[:, 0]
            translation[:, :2] = translation[:, :2] + pred_trans[:, 0, :2]
            
        else:
            # old ADS crop_img version
            # kp_2d_name = 'fake_kp_loss'
            # depth = 2 * self.FOCAL_LENGTH / (1e-6 + pred_camera[..., 0] * bboxes_size)
            # translation[:, :-1] = depth[:, None] * (center_pts + pred_camera[:, 1:] * bboxes_size.unsqueeze(-1) - img_size / 2) / self.FOCAL_LENGTH
            # translation[:, -1] = depth
            
            # newly corped_img version 02/10/2023
            # refer to https://github.com/nkolot/ProHMR/blob/6c9d956eedab9aa2d4214ddc3051c43b2ee20f24/prohmr/optimization/keypoint_fitting.py#L40
            

            # adjust focal_len, because size of stageI image is 256x256
            if not self.not_use_256p_focal:
                # self.FOCAL_LENGTH  = 256 / 832 * self.FOCAL_LENGTH 
                focal_length = 256 / 832 * self.FOCAL_LENGTH 
            depth = 2 * focal_length / (pred_camera[..., 0] * bboxes_size + 1e-9)
            depth = depth[:, None]
            translation[:, :-1] = pred_camera[:, 1:] + (center_pts - img_size / 2) * depth / focal_length
            translation[:, -1] = depth[:, 0]
        

        # pred_keypoints_2d_smpl = perspective_projection(pred_joints,
        #                                      torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_joints.device),
        #                                      translation,
        #                                      focal_length,
        #                                      img_size / 2)
        # gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        # pred_keypoints_2d_smpl_orig = pred_keypoints_2d_smpl.clone()
        # if self.normalize_kpts:
        #     scale_w = torch.clamp(pred_bboxes[..., 2] - pred_bboxes[..., 0], 1, img_size[..., 0].max())
        #     scale_h = torch.clamp(pred_bboxes[..., 3] - pred_bboxes[..., 1], 1, img_size[..., 1].max())
        #     bboxes_scale = torch.stack([scale_w, scale_h], dim=1)
        #     gt_keypoints_2d[..., :2] = (gt_keypoints_2d[..., :2] - center_pts.unsqueeze(1)) / bboxes_scale.unsqueeze(1)
        #     pred_keypoints_2d_smpl = (pred_keypoints_2d_smpl - center_pts.unsqueeze(1)) / bboxes_scale.unsqueeze(1)
        # else:
        #     pred_keypoints_2d_smpl = pred_keypoints_2d_smpl / img_size.unsqueeze(1)
        #     gt_keypoints_2d[:, :, :-1] = gt_keypoints_2d[:, :, :-1] / img_size.unsqueeze(1)
        # loss_keypoints_2d_smpl, error_ranks = self.keypoint_loss(pred_keypoints_2d_smpl[valid_boxes],
        #                                                       gt_keypoints_2d[valid_boxes])

        '''
        应该
        len valid_boxes:  16
        valid_boxes:  tensor([False, False, False, False, False, False, False, False, False, False,
                False, False, False, False, False, False], device='cuda:0')
        pred_joints[valid_boxes]:  torch.Size([0, 24, 3])
        gt_keypoints_2d[valid_boxes]:  torch.Size([0, 24, 3])
        rotation.shape:  torch.Size([0, 3, 3])
        points.shape:  torch.Size([0, 24, 3])
        '''
        # print("len valid_boxes: ", len(valid_boxes))
        # print("valid_boxes: ", valid_boxes)
        # print("pred_joints[valid_boxes]: ", pred_joints[valid_boxes].shape)
        # print("gt_keypoints_2d[valid_boxes]: ", gt_keypoints_2d[valid_boxes].shape)



        # loss_keypoints_2d_smpl, pred_keypoints_2d_smpl_orig, gt_keypoints_2d_orig = 0, torch.zeros(1, 24, 2), torch.zeros(1, 24, 2)
        # loss_keypoints_3d_smpl = 0
        # if pred_joints[valid_boxes][valid_has_depth].shape[0] > 0:
        loss_keypoints_2d_smpl, pred_keypoints_2d_smpl_orig, gt_keypoints_2d_orig = self.keypoint_loss(pred_joints[valid_boxes], 
                                    gt_keypoints_2d[valid_boxes], pred_bboxes[valid_boxes], center_pts[valid_boxes], translation[valid_boxes], 
                                    focal_length, img_size[valid_boxes])

        #  这里应该做has_2djoint做判断
        loss_keypoints_3d_smpl = self.keypoint_3d_loss(pred_joints[valid_boxes], gt_keypoints_3d[valid_boxes])

        # ### ---------------- visual start ---------------------- ###
        # gt_keypoints_3d_tmp = gt_keypoints_3d.clone()[:, :, :3].to(gt_camera_trans.device)
        # print("gt_keypoints_3d.shape, pred_joints.shape: ", gt_keypoints_3d_tmp.shape, pred_joints.shape)
        # # print("gt_camera_trans: ", gt_camera_trans)
        # print("focal_length: ", focal_length)
        # # print("img_size: ", img_size)
        # gt_keypoints_2d_projection = perspective_projection(gt_keypoints_3d_tmp,
        #                         torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(gt_keypoints_3d_tmp.device),
        #                         translation,
        #                         # gt_camera_trans,
        #                         focal_length,
        #                         img_size / 2)

        # # visual pred_2d, gt_2d, to verify the joint order
        # s_img = target['raw_images'][0].cpu().numpy().astype(np.float32).transpose([1, 2, 0])
        # gt_keypoints_2d_orig_np = gt_keypoints_2d_orig[0:1].cpu().numpy().astype(np.float32)
        # gt_keypoints_2d_projection_np = gt_keypoints_2d_projection[0:1].cpu().detach().numpy().astype(np.float32)
        # pred_keypoints_2d_smpl_orig_np = pred_keypoints_2d_smpl_orig[0:1].cpu().detach().numpy().astype(np.float32)

        # print(s_img.shape, gt_keypoints_2d_orig.shape, gt_keypoints_2d_projection_np.shape)
        # import cv2
        # from mmdet.datasets.utils import draw_skeleton
        # denormalize = lambda x: x * np.array([0.229, 0.224, 0.225])[None, None, :] \
        #             + np.array([0.485, 0.456, 0.406])[None, None, :]
        # skeleton_img_gt = cv2.cvtColor((denormalize(s_img) * 255).astype(np.uint8).copy(),
        #                            cv2.COLOR_BGR2RGB)
        # skeleton_img_pred = skeleton_img_gt.copy()
        # skeleton_img_gt_projection = skeleton_img_gt.copy()

        # for i in range(gt_keypoints_2d_orig_np.shape[0]):
        #     draw_skeleton(skeleton_img_gt, gt_keypoints_2d_orig_np[i])
        #     # for i in range(pred_keypoints_2d_smpl_orig_np.shape[0]):
        #     draw_skeleton(skeleton_img_pred, pred_keypoints_2d_smpl_orig_np[i])
        #     # for i in range(gt_keypoints_2d_projection_np.shape[0]):
        #     draw_skeleton(skeleton_img_gt_projection, gt_keypoints_2d_projection_np[i])

        #     gt_kp_path = f"./debug/keypoint_skeleton_gt_{i}.jpg"
        #     gt_kp_proj_path = f"./debug/keypoint_skeleton_gt_projection_{i}.jpg"
        #     pred_kp_path = f"./debug/keypoint_skeleton_pred_{i}.jpg"
        #     cv2.imwrite(gt_kp_path, skeleton_img_gt)
        #     cv2.imwrite(gt_kp_proj_path, skeleton_img_gt_projection)
        #     cv2.imwrite(pred_kp_path, skeleton_img_pred)
        #     print(gt_kp_path)
        #     print(gt_kp_proj_path)
        #     print(pred_kp_path)
        #     print("-" * 30)
        # ### ---------------- visual end ---------------------- ###
        
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat[valid_boxes], pred_betas[valid_boxes],
                                                           gt_rotmat[valid_boxes], gt_shape[valid_boxes],
                                                           has_smpl[valid_boxes])

        loss_dict = {
                     'loss_keypoints_2d_smpl': loss_keypoints_2d_smpl, 
                    
                     'loss_keypoints_3d_smpl': loss_keypoints_3d_smpl,
                    # #  'loss_vertices': loss_vertices, 
                     'loss_regr_pose': loss_regr_pose,
                     'loss_regr_betas': loss_regr_betas,

                    #  'img$raw_images': raw_images.detach(), 
                    
                    #  'img$idxs_in_batch': idxs_in_batch.detach(),
                    #  'img$pose_idx': pose_idx.detach(),
                    #  'img$pred_vertices': pred_vertices.detach(),
                     
                    #  'img$translation': translation.detach(), 
                    #  'img$error_rank': -bboxes_confidence.detach(),
                    #  'img$pred_bboxes': pred_bboxes.detach(),
                    #  'img$pred_keypoints_2d_smpl': (pred_keypoints_2d_smpl_orig[:, -24:, :]).detach().clone(),
                    #  'img$gt_keypoints_2d': gt_keypoints_2d_orig.detach().clone(),
                     }
        
        if not self.wo_lossVertices:
            loss_vertices = self.shape_loss(pred_vertices[valid_boxes], gt_vertices[valid_boxes], 
                                            has_smpl[valid_boxes], visibility=None)
            loss_dict.update({'loss_vertices': loss_vertices})
        # else:
        #     print("wo_lossVertices!!!!!")

        # integrate shape loss and dense loss
        # if self.dense_loss_weight > 0.001:
        if self.use_trans and not self.wo_lossVertices_global:
            loss_dense = self.shape_loss(pred_vertices[valid_boxes] + translation[valid_boxes].unsqueeze(1), 
                                              gt_vertices[valid_boxes] + gt_camera_trans[valid_boxes].unsqueeze(1), # N 1 3
                                              has_trans[valid_boxes], visibility=None)
            loss_dict.update({'loss_dense': loss_dense})

        # if self.global_loss_weight > 0.001:
        if self.use_trans:
            # borrow shape loss
            loss_global = self.shape_loss(translation[valid_boxes], gt_camera_trans[valid_boxes], 
                                              has_trans[valid_boxes], visibility=None)
            loss_dict.update({'loss_global': loss_global})

        # # prior loss
        # #print(pred_rotmat[valid_boxes].shape)
        # loss_prior = self.prior_loss(pred_rotmat[valid_boxes], pred_betas[valid_boxes], has_smpl[valid_boxes])
        # loss_dict.update({'loss_prior': loss_prior})

        # mpjpe loss
        # if False: #self.training:
        if self.use_pejpe_loss: 
            # print(type(pred_joints), pred_joints.shape, pred_joints.device)
            # print(type(gt_keypoints_3d), gt_keypoints_3d.shape, gt_keypoints_3d.device)
            # print('-' * 50)
            mpjpe_each, pampjpe_each = self.mpjpe_loss(pred_joints[valid_boxes], gt_keypoints_3d[valid_boxes])
            loss_dict.update({'loss_mpjpe': mpjpe_each, 'loss_pampjpe': pampjpe_each})

            # mpjpe_each = self.mpjpe_loss(pred_joints[valid_boxes], gt_keypoints_3d[valid_boxes])
            # loss_dict.update({'loss_mpjpe': mpjpe_each})

        if self.adversarial_cfg:
            valid_batch_size = pred_rotmat[valid_boxes].shape[0]
            pred_pose_shape = torch.cat([pred_rotmat[valid_boxes].view(valid_batch_size, -1), pred_betas[valid_boxes]], dim=1)
            loss_dict.update(
                {'pred_pose_shape': pred_pose_shape}
            )

        # best_idxs = select_index(idxs_in_batch[valid_boxes], pose_idx[valid_boxes].int()[:, 0],
        #                          bboxes_confidence[valid_boxes])

        
        # sdf loss
        if False: 
            sdf_loss = torch.zeros(len(best_idxs)).to(pred_vertices.device)
            for bid, ids in enumerate(best_idxs):
                if len(ids) <= 1:
                    continue
                ids = torch.tensor(ids)
                sdf_loss[bid] = self.sdf_loss(pred_vertices[valid_boxes][ids], translation[valid_boxes][ids])
            loss_dict.update({
                'loss_sdf': sdf_loss.sum() if self.use_sdf else sdf_loss.sum().detach() * 1e-4
            })

        # rank loss
        if False:#self.nr_batch_rank:
            rank_loss = self.rank_loss(pred_vertices, valid_boxes, scene, pose_idx, translation)
            loss_dict.update(rank_loss)

        if self.re_weight is not None:
            for k, v in self.re_weight.items():
                if k in loss_dict.keys():
                    loss_dict[k] *= v

        return loss_dict

    #def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d):
    def keypoint_loss(self, pred_joints, gt_keypoints_2d, pred_bboxes, center_pts, 
                      translation, focal_length, img_size):
        """
        Compute 2D reprojection loss on the keypoints.
        The confidence is binary and indicates whether the keypoints exist or not.
        The available keypoints are different for each dataset.
        """
        batch_size = pred_joints.shape[0]
        # if batch_size == 0:
        #     loss = 0
        #     pred_keypoints_2d_smpl_orig = torch.zeros()
        #     gt_keypoints_2d_orig
        #     return 0, 
        
        pred_keypoints_2d_smpl = perspective_projection(pred_joints,
                                             torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_joints.device),
                                             translation,
                                             focal_length,
                                             img_size / 2)
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        pred_keypoints_2d_smpl_orig = pred_keypoints_2d_smpl.clone()
        if self.normalize_kpts:
            scale_w = torch.clamp(pred_bboxes[..., 2] - pred_bboxes[..., 0], 1, img_size[..., 0].max())
            scale_h = torch.clamp(pred_bboxes[..., 3] - pred_bboxes[..., 1], 1, img_size[..., 1].max())
            bboxes_scale = torch.stack([scale_w, scale_h], dim=1)
            gt_keypoints_2d[..., :2] = (gt_keypoints_2d[..., :2] - center_pts.unsqueeze(1)) / bboxes_scale.unsqueeze(1)
            pred_keypoints_2d_smpl = (pred_keypoints_2d_smpl - center_pts.unsqueeze(1)) / bboxes_scale.unsqueeze(1)
        else:
            pred_keypoints_2d_smpl = pred_keypoints_2d_smpl / img_size.unsqueeze(1)
            gt_keypoints_2d[:, :, :-1] = gt_keypoints_2d[:, :, :-1] / img_size.unsqueeze(1)

        pred_keypoints_2d = pred_keypoints_2d_smpl#[valid_boxes]
        gt_keypoints_2d = gt_keypoints_2d#[valid_boxes]

        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d[:, -24:], gt_keypoints_2d[:, :, :-1]))
        return loss.mean(), pred_keypoints_2d_smpl_orig, gt_keypoints_2d_orig

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence
        """
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        pred_keypoints_3d = pred_keypoints_3d[..., -24:, :]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            if hasattr(self, 'criterion_3d_keypoints'):
                return (conf * self.criterion_3d_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
            else:
                return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.tensor(0).float().cuda()

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl, visibility=None):
        """
        Compute per-vertex loss on the shape for the examples that SMPL annotations are available.
        """
        # disable shape loss!!!
        
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            if visibility is not None:
                visibility = visibility[has_smpl == 1].float().unsqueeze(-1)
                loss = self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
                #print(loss.shape, visibility.shape)
                return (loss * visibility).sum() / (visibility.sum() * 3.)
            else:
                return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape).mean()
        else:
            return torch.tensor(0).float().cuda()
        #return torch.tensor(0).float().cuda()

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        """
        Compute SMPL parameter loss for the examples that SMPL annotations are available.
        """
        batch_size = pred_rotmat.shape[0]

        # pred_rotmat_valid = pred_rotmat[has_smpl == 1][:,:22].contiguous()
        pred_rotmat_valid = pred_rotmat[has_smpl == 1][:,:22]
        # pred_rotmat_valid = pred_rotmat_valid.view(-1, 3, 3)
        pred_rotmat_valid = pred_rotmat_valid.reshape(-1, 3, 3)
        #print(gt_pose.shape)
        
        gt_rotmat_valid = batch_rodrigues(gt_pose[has_smpl == 1][:,:22].reshape(-1, 3))
        # gt_rotmat_valid = batch_rodrigues(gt_pose[has_smpl == 1][:,:22].contiguous().view(-1, 3))
        #gt_rotmat_valid = batch_rodrigues(gt_pose[has_smpl == 1].contiguous().view(-1, 3))
        
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            # loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid).mean()
            # loss_regr_betas = (self.criterion_regr(pred_betas_valid, gt_betas_valid) * self.shape_pca_weight.to(pred_betas_valid.device)).mean()
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.tensor(0).float().cuda()
            loss_regr_betas = torch.tensor(0).float().cuda()
        return loss_regr_pose, loss_regr_betas


    """
    https://github.com/mkocabas/VIBE/blob/851f779407445b75cd1926402f61c931568c6947/lib/smplify/temporal_smplify.py#L49
    https://github.com/mkocabas/VIBE/blob/851f779407445b75cd1926402f61c931568c6947/lib/smplify/losses.py#L50
    """
    def prior_loss(self, pred_rotmat, pred_betas, has_smpl):
        pred_rotmat = pred_rotmat[has_smpl == 1]
        pred_betas = pred_betas[has_smpl == 1]

        batch_size = pred_rotmat.shape[0]
        if batch_size == 0:
            return torch.tensor(0).float().cuda()
        pred_pose = rotation_matrix_to_angle_axis(pred_rotmat).view(batch_size, -1) # N, 72

        # gmm_prior_loss = self.gmm_prior(pred_pose[:, 3:], pred_betas).mean() / 10000.
        # angle_prior_loss = angle_prior(pred_pose[:, 3:]).mean() / 5.
        # return gmm_prior_loss + angle_prior_loss

        body_pose = pred_pose[:, 3:]

        # pose_prior_weight=4.78
        # angle_prior_weight=15.2
        # shape_prior_weight=5

        pose_prior_weight = 0.01
        angle_prior_weight = 0.4
        shape_prior_weight = 0.6

        # Pose prior loss
        pose_prior_loss = (pose_prior_weight ** 2) * self.pose_prior(body_pose, pred_betas).mean()

        # Angle prior for knees and elbows
        angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1).mean()

        # Regularizer to prevent betas from taking large values
        shape_prior_loss = (shape_prior_weight ** 2) * (pred_betas ** 2).sum(dim=-1).mean()

        total_loss = pose_prior_loss + angle_prior_loss + shape_prior_loss

        # print("pose_prior_loss: ", pose_prior_loss)
        # print("angle_prior_loss: ", angle_prior_loss)
        # print("shape_prior_loss: ", shape_prior_loss)
        # print("total_loss: ", total_loss)
        # print("-" * 50)

        return total_loss
        

    def mpjpe_loss(self, preds_kp3d, kp3d_gt):
        kp3d_gt, preds_kp3d = kp3d_gt.contiguous(), preds_kp3d.contiguous()
        
        mpjpe_each = calc_mpjpe(kp3d_gt, preds_kp3d, align_inds=self.align_idx)
        pampjpe_each = calc_pampjpe(kp3d_gt, preds_kp3d)
        
        return mpjpe_each, pampjpe_each



    def collision_loss(self, pred_vertices, best_idxs, loss_dict):
        """
        Calculate collision losses
        :param pred_vertices: Predicted vertices
        :param best_idxs: 2D list, each row means the selected meshes for this image
        :return:
        """
        device = pred_vertices.device
        max_persons = self.collision_param.get('max_persons', 16)
        loss = torch.zeros(len(best_idxs)).to(device)
        for bid, ids in enumerate(best_idxs):
            if len(ids) == 0:
                continue
            # Only select a limited number of persons to avoid OOM
            if len(ids) > max_persons:
                ids = random.sample(ids, max_persons)
            ids = torch.tensor(ids)

            verts = pred_vertices[ids]  # num_personx6890x3
            bs, nv = verts.shape[:2]
            face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long,
                                       device=device).unsqueeze_(0).repeat([bs, 1, 1])
            bs, nf = face_tensor.shape[:2]
            faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(device) * nv)[:, None, None]
            # We cannot make a batch because the number of persons is not determined
            triangles = verts.view([-1, 3])[faces_idx.view([1, -1, 3])]

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)
                collision_idxs_ = torch.div(collision_idxs.clone(), nf)
                msk_self = (collision_idxs_[..., 0] == collision_idxs_[..., 1])
                collision_idxs[msk_self] = -1

            if collision_idxs.max() >= bs * nf:
                print(f'An overflow detected with bs: {bs}')
                continue

            cur_loss = self.pen_distance(triangles, collision_idxs).mean()

            if torch.isnan(cur_loss):
                print(f'A NaN is detected inside intersection loss with bs: {bs}')
                continue
            loss[bid] += cur_loss

        return loss.mean()

        def rank_loss(self, pred_vertices, valid_boxes, scene, pose_idx, translation):
            device = pred_vertices.device
            batch_rank_loss = torch.zeros(len(best_idxs)).to(pred_vertices.device)
            num_intruded_pixels = torch.zeros(len(best_idxs)).to(pred_vertices.device)
            erode_mask_loss = torch.zeros(len(best_idxs)).to(pred_vertices.device)

            K = torch.eye(3, device=device)
            K[0, 0] = K[1, 1] = self.FOCAL_LENGTH
            K[2, 2] = 1
            K[1, 2] = K[0, 2] = self.image_size / 2  # Because the neural renderer only support squared images
            K = K.unsqueeze(0)  # Our batch size is 1
            R = torch.eye(3, device=device).unsqueeze(0)
            t = torch.zeros(3, device=device).unsqueeze(0)

            for bid, ids in enumerate(best_idxs):
                if len(ids) <= 1 or scene[bid].max() < 1:
                    continue
                ids = torch.tensor(ids)
                verts = pred_vertices[valid_boxes][ids] + translation[valid_boxes][ids].unsqueeze(
                    1)  # num_personx6890x3
                cur_pose_idxs = pose_idx[valid_boxes][ids, 0]
                with torch.no_grad():
                    pose_idxs_int = cur_pose_idxs.int()
                    has_mask_gt = torch.zeros_like(pose_idxs_int)
                    for has_mask_idx, cur_pid in enumerate(pose_idxs_int):
                        has_mask_gt[has_mask_idx] = 1 if torch.sum(scene[bid] == (cur_pid + 1).item()) > 0 else 0

                if has_mask_gt.sum() < 1:
                    continue

                verts = verts[has_mask_gt > 0]
                cur_pose_idxs = cur_pose_idxs[has_mask_gt > 0]

                bs, nv = verts.shape[:2]
                face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long,
                                           device=device).unsqueeze_(0).repeat([bs, 1, 1])
                bs, nf = face_tensor.shape[:2]
                textures = torch.ones_like(face_tensor).float() + cur_pose_idxs.to(device)[:, None, None]
                textures = textures[:, :, None, None, None, :]
                rgb, depth, mask = self.neural_renderer(verts, face_tensor.int(), textures=textures, K=K, R=R, t=t,
                                                        dist_coeffs=torch.tensor([[0., 0., 0., 0., 0.]], device=device))
                predicted_depth = depth[:, self.h_diff:rgb.shape[-2] - self.h_diff,
                                  self.w_diff:rgb.shape[-1] - self.w_diff]
                predicted_mask = mask[:, self.h_diff:rgb.shape[-2] - self.h_diff,
                                 self.w_diff:rgb.shape[-1] - self.w_diff]

                with torch.no_grad():
                    gt_foreground = scene[bid] > 0
                    foreground_select = (cur_pose_idxs.round().int() + 1)[:, None, None] == scene[bid].int()
                    intruded_parts_mask = torch.prod(predicted_mask, dim=0)
                    supervising_mask = intruded_parts_mask.unsqueeze(0).float() * gt_foreground.unsqueeze(
                        0).float() * (~foreground_select).float()
                    if supervising_mask.norm() == 0:  # No ordinal relationship errors is detected.
                        continue

                gt_closest_depth_multi = torch.zeros_like(predicted_depth)
                gt_closest_depth_multi[foreground_select] += predicted_depth[foreground_select]
                gt_closest_depth = gt_closest_depth_multi.sum(0)

                gt_closest_depth = gt_closest_depth.repeat([bs, 1, 1])
                ordinal_distance = (gt_closest_depth - predicted_depth) * supervising_mask
                penalize_ranks = torch.log(1. + torch.exp(ordinal_distance)) * supervising_mask
                # To avoid instable gradient:
                if torch.sum(ordinal_distance > 10) > 0:
                    penalize_ranks[ordinal_distance.detach() > 10] = ordinal_distance[ordinal_distance.detach() > 10]
                    print(f'{torch.sum(ordinal_distance > 10)} pixels found to be greater than 10 in batch rank loss')
                batch_rank_loss[bid] = penalize_ranks.mean()
                num_intruded_pixels[bid] = supervising_mask.sum()

            return {'loss_batch_rank': batch_rank_loss, 'num_intruded_pixels': num_intruded_pixels}

