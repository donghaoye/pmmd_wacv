# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin, SMPLTestMixin

import torch.nn.functional as F

import numpy as np
import cv2


@HEADS.register_module()
class SMPLRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin, SMPLTestMixin):
    """SMPL base roi head including one bbox head, one mask head, one SMPL head, and one Translation head."""
    
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_smpl_head(self, smpl_roi_extractor, smpl_head):
        """Initialize ``smpl_head``"""
        if smpl_roi_extractor is not None:
            self.smpl_roi_extractor = build_roi_extractor(smpl_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.smpl_roi_extractor = self.bbox_roi_extractor
        self.smpl_head = build_head(smpl_head)

    def init_trans_head(self, trans_roi_extractor, trans_head):
        """Initialize ``trans_head``"""
        if trans_roi_extractor is not None:
            self.trans_roi_extractor = build_roi_extractor(trans_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.trans_roi_extractor = self.bbox_roi_extractor
        self.trans_head = build_head(trans_head)

    def init_adabin_head(self, adabin_roi_extractor, adabin_head):
        """Initialize ``trans_head``"""
        # if trans_roi_extractor is not None:
        #     self.trans_roi_extractor = build_roi_extractor(trans_roi_extractor)
        #     self.share_roi_extractor = False
        # else:
        #     self.share_roi_extractor = True
        #     self.trans_roi_extractor = self.bbox_roi_extractor
        self.adabin_head = build_head(adabin_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )

        # smpl head and trans head todo...

        # todo...
        # self._smpl_forward
        # self._trans_forward

        return outs

    def forward_train(self,
                      img,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_depthmap=None,
                      has_depthmap=None,
                      gt_kpts3d=None,
                      gt_kpts2d=None,
                      gt_shapes=None,
                      gt_poses=None,
                      gt_vertices=None,
                      gt_trans=None,
                    #   gt_scale=None,
                      gt_camera_trans=None,
                    #   gt_depth=None,
                      has_masks=None,
                      has_smpl=None,
                      has_trans=None,
                      has_depth=None,
                      has_bbox=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        mask_results = None
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, has_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        # trans head forward and loss
        if self.with_trans and not self.not_use_depthmap:
            if mask_results is not None:
                pred_masks = mask_results['mask_pred']
            else:
                pred_masks = None
            trans_results = self._trans_forward_train(x, sampling_results,
                                                    gt_depthmap, has_depthmap, pred_masks)
            # losses.update(trans_results['loss_body_depthmap'])
            losses.update({'loss_body_depthmap': trans_results['loss_body_depthmap']})

        if self.with_adabin and not self.not_use_depthmap:
            pred_masks = mask_results['mask_pred']
            trans_results = self._adabin_forward_train(x, sampling_results,
                                                    gt_depthmap, has_depthmap, pred_masks)
            
            losses.update({'loss_adabin_L1': trans_results['loss_adabin_L1']})
            losses.update({'loss_adabin_sig': trans_results['loss_adabin_sig']})
            losses.update({'loss_adabin_chamfer': trans_results['loss_adabin_chamfer']})

        # smpl head forward and loss
        if self.with_smpl:
            cls_scores = bbox_results['cls_score']

            if self.mask_attention == False:
                pred_masks = None
            else:
                pred_masks = mask_results['mask_pred']

            if self.not_use_depthmap:
                pred_z = None
                global_trans_feats = None
            else:
                pred_z = trans_results['pred_z_tensor']
                global_trans_feats = trans_results['global_trans_feats']

            smpl_results = self._smpl_forward_train(img, x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    cls_scores=cls_scores,
                                                    pred_masks=pred_masks,
                                                    pred_z=pred_z,
                                                    img_meta=img_metas,
                                                    gt_kpts3d=gt_kpts3d,
                                                    gt_kpts2d=gt_kpts2d,
                                                    gt_shapes=gt_shapes,
                                                    gt_poses=gt_poses,
                                                    gt_vertices=gt_vertices,
                                                    gt_trans=gt_trans,
                                                    # gt_scale=gt_scale,
                                                    gt_camera_trans=gt_camera_trans,
                                                    has_trans=has_trans,
                                                    has_smpl=has_smpl,
                                                    global_trans_feats=global_trans_feats,
                                                    not_use_localFeatsForTrans=self.not_use_localFeatsForTrans,
                                                    not_use_globalFeatsForTrans=self.not_use_globalFeatsForTrans
                                                    )
            losses.update(smpl_results['loss_smpl'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks, has_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets, has_masks_targets = self.mask_head.get_targets(sampling_results, gt_masks, has_masks, self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        has_masks_targets = has_masks_targets.long()
        loss_mask = self.mask_head.loss(mask_results['mask_pred'][has_masks_targets==1], mask_targets[has_masks_targets==1], pos_labels[has_masks_targets==1])
        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)

        # ##################
        # ## visual start ##
        # ##################
        # pred_masks_softmax = F.softmax(mask_pred, dim=1)
        # # Get the channel with the highest probability for each pixel
        # predicted_masks = torch.argmax(pred_masks_softmax, dim=1)
        # # Convert the predicted masks to a binary mask
        # pred_mask_arr = (predicted_masks == 1)  # index 1 means foreground
        # # pred_mask_arr = (predicted_masks == 0)  # index 0 means background
        # pred_mask_arr = pred_mask_arr.long()
        # pred_mask_arr = pred_mask_arr.detach().cpu().numpy().astype(np.uint8)

        # for i in range(pred_mask_arr.shape[0]):
        #     pred_mask = pred_mask_arr[i]
        #     pred_masks_vis = pred_mask[:, :, None] * 255
        #     # print("np.max(pred_mask): ", np.max(pred_mask))         #  15.844506
        #     # print("np.min(pred_mask): ", np.min(pred_mask))         # -22.093634
        #     # print("np.mean(pred_mask): ", np.mean(pred_mask))       # -2.082635
        #     # print("np.median(pred_mask): ", np.median(pred_mask))   # -1.7037759
        #     print("pred_mask.shape: ", pred_mask.shape)             # (2, 32, 32)
        #     print("pred_masks_vis.shape: ", pred_masks_vis.shape)   # (32, 32, 1)
        #     output_path = f"./output/pred_masks_{i}_foreground.jpg"
        #     cv2.imwrite(output_path, pred_masks_vis)
        #     print(output_path)
        #     print("...")
        # ##################
        # ## visual end   ##
        # ##################

        return mask_results

    def _trans_forward_train(self, x, sampling_results, gt_depthmap, 
                            has_depthmap, pred_masks):
        """Run forward function and calculate loss for trans head in training."""

        trans_results = self.trans_head(x, sampling_results, pred_masks)
        pred_depthmap = trans_results['pred_depthmap']
        # pred_z_tensor = trans_results['pred_z_tensor']
        gt_depthmap = torch.cat(gt_depthmap, dim=0)
        has_depthmap_targets = torch.cat(has_depthmap, dim=0)
        gt_depthmap = gt_depthmap.unsqueeze(1)
        
        depthmap_loss = torch.tensor(0).float().to(gt_depthmap.device)
        sum_depthmap_tensor = sum(has_depthmap)
        sum_depthmap = torch.sum(sum_depthmap_tensor)
        if sum_depthmap > 0:
            depthmap_loss = self.trans_head.loss(pred_depthmap[has_depthmap_targets == 1], gt_depthmap[has_depthmap_targets == 1])
            depthmap_loss = depthmap_loss * 2 # for wo_sigmoid
            # depthmap_loss = depthmap_loss * 10
            # depthmap_loss = depthmap_loss * 0.1 # for w_sigmoid
        depthmap_loss_dict = {'loss_body_depthmap': depthmap_loss.float()} 
        trans_results.update(depthmap_loss_dict)

        if self.trans_head.use_trans_roi == True or self.trans_head.use_trans_roi_depthmap == True:
            global_trans_feats = trans_results['global_trans_feats']
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            global_trans_feats = self.trans_roi_extractor([global_trans_feats], pos_rois)

            if self.trans_head.use_trans_roi_depthmap == True:
                global_trans_feats = torch.cat([global_trans_feats] * 4, dim=1)
            trans_results['global_trans_feats'] = global_trans_feats

        return trans_results

    def _adabin_forward_train(self, x, sampling_results, gt_depthmap, 
                            has_depthmap, pred_masks):
        """Run forward function and calculate loss for trans head in training."""

        trans_results = self.adabin_head(x, sampling_results, pred_masks)
        pred_depthmap = trans_results['pred_depthmap']
        bin_edges = trans_results['bin_edges']

        gt_depthmap = torch.cat(gt_depthmap, dim=0)
        has_depthmap_targets = torch.cat(has_depthmap, dim=0)
        gt_depthmap = gt_depthmap.unsqueeze(1)
        
        sum_depthmap_tensor = sum(has_depthmap)
        sum_depthmap = torch.sum(sum_depthmap_tensor)

        # depthmap_loss = torch.tensor(0).float().to(gt_depthmap.device)      
        # if sum_depthmap > 0:
        #     depthmap_loss = self.adabin_head.loss(pred_depthmap[has_depthmap_targets == 1], gt_depthmap[has_depthmap_targets == 1])
        #     depthmap_loss = depthmap_loss * 2 # for wo_sigmoid
        # depthmap_loss_dict = {'loss_body_depthmap': depthmap_loss.float()} 
        # trans_results.update(depthmap_loss_dict)

        loss_adabin_L1 = torch.tensor(0).float().to(gt_depthmap.device)
        loss_adabin_sig = torch.tensor(0).float().to(gt_depthmap.device)
        loss_adabin_chamfer = torch.tensor(0).float().to(gt_depthmap.device)
        if sum_depthmap > 0:
            loss_adabin_L1 = self.adabin_head.loss_L1(pred_depthmap[has_depthmap_targets == 1], gt_depthmap[has_depthmap_targets == 1])
            loss_adabin_sig = self.adabin_head.loss_decode(pred_depthmap[has_depthmap_targets == 1], gt_depthmap[has_depthmap_targets == 1])
            loss_adabin_chamfer = self.adabin_head.loss_chamfer(bin_edges[has_depthmap_targets == 1], gt_depthmap[has_depthmap_targets == 1])

        trans_results['loss_adabin_L1'] = loss_adabin_L1.float()
        trans_results['loss_adabin_sig'] = loss_adabin_sig.float()
        trans_results['loss_adabin_chamfer'] = loss_adabin_chamfer.float()


        return trans_results

    def _smpl_forward_train(self, img, x, sampling_results, bbox_feats, cls_scores,
                            pred_masks, pred_z, img_meta=None,
                            gt_kpts3d=None,
                            gt_kpts2d=None,
                            gt_shapes=None,
                            gt_poses=None,
                            gt_vertices=None,
                            gt_trans=None,
                            # gt_scale=None,
                            gt_camera_trans=None,
                            has_trans=None,
                            has_smpl=None,
                            global_trans_feats=None,
                            not_use_localFeatsForTrans=False,
                            not_use_globalFeatsForTrans=False
                            ):
        """Run forward function and calculate loss for mask head in
        training."""

        # apply mask attention
        if self.with_mask:
            pred_masks = F.softmax(pred_masks, dim=1)[:,-1,:,:].unsqueeze(1)
        else:
            pred_masks = None

        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            smpl_results = self._smpl_forward(x, pos_rois, pred_masks=pred_masks, global_trans_feats=global_trans_feats, not_use_localFeatsForTrans=not_use_localFeatsForTrans, not_use_globalFeatsForTrans=not_use_globalFeatsForTrans)
        # else:
        #     pos_inds = []
        #     device = bbox_feats.device
        #     for res in sampling_results:
        #         pos_inds.append(
        #             torch.ones(
        #                 res.pos_bboxes.shape[0],
        #                 device=device,
        #                 dtype=torch.uint8))
        #         pos_inds.append(
        #             torch.zeros(
        #                 res.neg_bboxes.shape[0],
        #                 device=device,
        #                 dtype=torch.uint8))
        #     pos_inds = torch.cat(pos_inds)

        #     smpl_results = self._smpl_forward(x, pos_inds=pos_inds, bbox_feats=bbox_feats, pred_masks=pred_masks)

        # To get the confidence from detection head.
        pos_inds = []
        device = bbox_feats.device
        for res in sampling_results:
            pos_inds.append(
                torch.ones(
                    res.pos_bboxes.shape[0],
                    device=device,
                    #dtype=torch.uint8))
                    dtype=torch.bool)) # torch 1.6
            pos_inds.append(
                torch.zeros(
                    res.neg_bboxes.shape[0],
                    device=device,
                    #dtype=torch.uint8))
                    dtype=torch.bool)) # torch 1.6
        pos_inds = torch.cat(pos_inds)
        bboxes_confidence = cls_scores[pos_inds, 1]

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        pred_bboxes, kpts2d_target, kpts3d_target, poses_target, shapes_target, trans_target, has_smpl_target, camera_trans_target, has_trans_target, gt_vertices, idxs_in_batch, pose_idx = self.smpl_head.get_target(
                sampling_results, gt_kpts2d, gt_kpts3d, gt_poses, gt_shapes, gt_vertices, gt_trans, has_smpl,
                gt_camera_trans, has_trans,
                self.train_cfg)
        
        smpl_targets = {
            'gt_keypoints_2d': kpts2d_target,
            'gt_keypoints_3d': kpts3d_target,
            'gt_camera_trans': camera_trans_target,
            'has_trans': has_trans_target,
            'gt_rotmat': poses_target,
            'gt_shape': shapes_target,
            'gt_camera': trans_target,
            'has_smpl': has_smpl_target,
            'gt_vertices': gt_vertices,
            'pred_bboxes': pred_bboxes,
            'raw_images': img.clone(),
            'img_meta': img_meta,
            'idxs_in_batch': idxs_in_batch,
            'pose_idx': pose_idx,
            # 'mosh': kwargs.get('mosh', None),
            # 'scene': scene,
            # 'log_depth': log_depth,
        }

        # smpl_targets = self.smpl_head.get_targets(sampling_results, gt_smpls, self.train_cfg)
        # pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        # loss_smpl = self.smpl_head.loss(smpl_results['smpl_pred'], smpl_targets, pos_labels)
        # smpl_results.update(loss_smpl=loss_smpl, smpl_targets=smpl_targets)



        loss_smpl = self.smpl_head.loss(smpl_results, 
                                        smpl_targets,
                                        pos_labels, 
                                        bboxes_confidence=bboxes_confidence,
                                        # discriminator=kwargs.get('discriminator', None), 
                                        discriminator=None, 
                                        # nested=self.nested,
                                        nested=True,
                                        residual_depth=True, # new version (02/10/2023): global_pelvis_trans + local_pelvis_trans 
                                        # global_pelvis_depth=pred_z_tensor[gt_has_trans == 1]
                                        # global_pelvis_depth=pred_z_tensor
                                        global_pelvis_depth=pred_z,
                                        global_trans_feats=global_trans_feats
                                        )
        
        # print("loss_smpl:", loss_smpl)
        smpl_results.update(loss_smpl=loss_smpl, smpl_targets=smpl_targets)

        return smpl_results

    def _smpl_forward(self, x, rois=None, pos_inds=None, bbox_feats=None, pred_masks=None, global_trans_feats=None, not_use_localFeatsForTrans=False, not_use_globalFeatsForTrans=False):
        """smpl head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            smpl_feats = self.smpl_roi_extractor(
                x[:self.smpl_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                smpl_feats = self.shared_head(smpl_feats)
        else:
            assert bbox_feats is not None
            smpl_feats = bbox_feats[pos_inds]

        # smpl_pred = self.smpl_head(smpl_feats)
        # smpl_results = dict(smpl_pred=smpl_pred, smpl_feats=smpl_feats)

        smpl_results = self.smpl_head(smpl_feats, mask_pred=pred_masks, global_trans_feats=global_trans_feats, not_use_localFeatsForTrans=not_use_localFeatsForTrans, not_use_globalFeatsForTrans=not_use_globalFeatsForTrans)

        return smpl_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    img,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    gt_bboxes=None,
                    gt_labels=None,
                    gt_bboxes_ignore=None,
                    gt_masks=None,
                    gt_depthmap=None,
                    has_depthmap=None,
                    gt_kpts3d=None,
                    gt_kpts2d=None,
                    gt_shapes=None,
                    gt_poses=None,
                    gt_vertices=None,
                    gt_trans=None,
                #   gt_scale=None,
                    gt_camera_trans=None,
                #   gt_depth=None,
                    has_masks=None,
                    has_smpl=None,
                    has_trans=None,
                    has_depth=None,
                    has_bbox=None,
                    is_woLocalTrans=False,
                    is_woGlobalTransPart=False,
                    **kwargs):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
       
       
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        
        # return  det_bboxes, det_labels
        
        
        
        # ## num_class = 2 
        # det_bboxes_select = []
        # det_labels_select = []
        # for i in range(len(det_bboxes)):
        #     select_inds = det_labels[i] == 1 # select human class
        #     bbox_tmp = det_bboxes[i][select_inds]
        #     label_tmp = det_labels[i][select_inds]
        #     det_bboxes_select.append(bbox_tmp)
        #     det_labels_select.append(label_tmp)
            
        # bbox_results = []
        # for i in range(len(det_bboxes_select)):
        #     pred_bbox = bbox2result(det_bboxes_select[i], det_labels_select[i], 
        #                             self.bbox_head.num_classes)
        #     # bbox_results.append(pred_bbox[0])     # index 0 is background
        #     bbox_results.append(pred_bbox[1])       # select the human (index=1)
    
        # bbox_results = []
        # for i in range(len(det_bboxes_select)):
        #     bbox_results.append(det_bboxes_select[i])
        
        bbox_results = det_bboxes
            
        
        # # visual bbox_results
        # print("visual bbox....")
        # print("-" * 50)
        # denormalize = lambda x: x.transpose([1, 2, 0]) * np.array([0.229, 0.224, 0.225])[None, None, :] + \
        #                 np.array([0.485, 0.456, 0.406])[None, None,]
        # img_bbox = denormalize(img.cpu().numpy()[0]) * 255
        
        # for bbox in bbox_results[0]:
        #     bbox = bbox.cpu().numpy()
        #     img_bbox = cv2.rectangle(img_bbox, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (1., 0, 0), 2)
        # output_path = f"./output/bbox.jpg"
        # cv2.imwrite(output_path, img_bbox[:, :, ::-1])
        # print(output_path)
        # print("-" * 50)

        pred_masks = None
        pred_masks_rate = None
        segm_results = None
        if self.with_mask:
            pred_masks, segm_results = self.simple_test_mask(x, img_metas, det_bboxes, det_labels, rescale=rescale)
            # pred_masks, segm_results = self.mask_onnx_export(x, img_metas, det_bboxes, det_labels, rescale=rescale)


            if len(pred_masks) == 0 or len(pred_masks[0]) == 0 or len(pred_masks[0][0]) == 0:
                return None, None, None
            
            if pred_masks != None and len(pred_masks) > 0:
                if isinstance(pred_masks, list):
                    pred_masks = torch.cat(pred_masks, dim=0)
                elif isinstance(pred_masks, torch.Tensor):
                    pass
                pred_masks_rate = F.softmax(pred_masks, dim=1)[:, -1, :, :].unsqueeze(1)
            else:
                # pred_masks, segm_results, pred_masks_rate = None, None, None
                print("pred_masks is None!!")

            # ##################
            # ## visual start ##
            # ##################
            # pred_masks_softmax = F.softmax(pred_masks, dim=1)
            # # Get the channel with the highest probability for each pixel
            # predicted_masks = torch.argmax(pred_masks_softmax, dim=1)
            # # Convert the predicted masks to a binary mask
            # pred_mask_arr = (predicted_masks == 1)  # index 1 means foreground
            # # pred_mask_arr = (predicted_masks == 0)  # index 0 means background
            # pred_mask_arr = pred_mask_arr.long()
            # pred_mask_arr = pred_mask_arr.detach().cpu().numpy().astype(np.uint8)

            # for i in range(pred_mask_arr.shape[0]):
            #     pred_mask = pred_mask_arr[i]
            #     pred_masks_vis = pred_mask[:, :, None] * 255
            #     # print("np.max(pred_mask): ", np.max(pred_mask))         #  15.844506
            #     # print("np.min(pred_mask): ", np.min(pred_mask))         # -22.093634
            #     # print("np.mean(pred_mask): ", np.mean(pred_mask))       # -2.082635
            #     # print("np.median(pred_mask): ", np.median(pred_mask))   # -1.7037759
            #     print("pred_mask.shape: ", pred_mask.shape)             # (2, 32, 32)
            #     print("pred_masks_vis.shape: ", pred_masks_vis.shape)   # (32, 32, 1)
            #     output_path = f"./output/pred_masks_{i}_foreground.jpg"
            #     cv2.imwrite(output_path, pred_masks_vis)
            #     print(output_path)
            #     print("...")
            # ##################
            # ## visual end   ##
            # ##################
    
        pred_z_tensor = None
        global_trans_feats = None
        if self.with_trans and not self.not_use_depthmap:
            bbox_results_perImg = bbox_results # image batch_size = 1 for testing and eval
            pred_depthmap = torch.zeros((1, 1, img.shape[2], img.shape[3])).to(img.device)
            pred_z_tensor = torch.tensor(0).float().cuda()
            
            # if not pred_masks is None:
            trans_dict_preds = self.trans_head.forward_test(x, bbox_results_perImg, pred_masks)
            # trans_dict_preds = self.trans_head.forward(x, bbox_results_perImg, pred_masks)
            pred_depthmap = trans_dict_preds['pred_depthmap']
            pred_z_tensor = trans_dict_preds['pred_z_tensor']
            global_trans_feats = trans_dict_preds['global_trans_feats']
                    
            if self.trans_head.use_trans_roi == True or self.trans_head.use_trans_roi_depthmap == True:
                rescale_trans = False
                if rescale_trans:
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]

                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale_trans else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                pos_rois = bbox2roi(_bboxes)

                global_trans_feats = self.trans_roi_extractor([global_trans_feats], pos_rois)
                
                if self.trans_head.use_trans_roi_depthmap == True:
                    global_trans_feats = torch.cat([global_trans_feats] * 4, dim=1)
            
            # if self.pred_depth_is_detach:
            # pred_z_tensor = pred_z_tensor.detach()
            global_trans_feats = global_trans_feats.detach()

        if self.with_adabin and not self.not_use_depthmap:
            bbox_results_perImg = bbox_results # image batch_size = 1 for testing and eval
            pred_depthmap = torch.zeros((1, 1, img.shape[2], img.shape[3])).to(img.device)
            pred_z_tensor = torch.tensor(0).float().cuda()
            if not pred_masks is None:
                trans_dict_preds = self.adabin_head.forward_test(x, bbox_results_perImg, pred_masks)
                pred_depthmap = trans_dict_preds['pred_depthmap']
                pred_z_tensor = trans_dict_preds['pred_z_tensor']

            
            # ##################################
            # ### start pred_depthmap visual ###
            # ##################################
            
            # output_path = "./debug/pred_depthmap.png"
            # pred_depthmap_np = pred_depthmap.detach().cpu().numpy()[0][0] # 1x1xhxw --> hxw

            # # output_path = "./debug/gt_depthmap.png"
            # # pred_depthmap_np = gt_depthmap[0].cpu().numpy()[0] # 1x1xhxw --> hxw

            # depth_min = pred_depthmap_np.min()
            # depth_max = pred_depthmap_np.max()
            # print("depth_max: ", depth_max)
            # print("depth_min: ", depth_min)
            # if depth_max - depth_min > np.finfo("float").eps:
            #     out = 255 * (pred_depthmap_np - depth_min) / (depth_max - depth_min)
            # else:
            #     out = np.zeros(pred_depthmap_np.shape, dtype=pred_depthmap_np.dtype)
            # cv2.imwrite(output_path, out.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # print("pred_depthmap.shape: ", pred_depthmap.shape)
            # print(output_path)

            # print("pred_depthmap_np.shape: ", pred_depthmap_np.shape)
            # depth_map_normalized = (pred_depthmap_np - depth_min) / (depth_max - depth_min)

            # # plt.subplot(1, 2, 1)
            # # plt.imshow(cv2.cvtColor(pred_depthmap_np, cv2.COLOR_BGR2RGB))
            # # plt.subplot(1, 2, 2)
            # # plt.imshow(depth_map_normalized, cmap='jet')
            # # plt.colorbar(label='Depth Value')
            # # plt.title('Depthmap Colormap')
            # # color_map_path = output_path[:-4] + "_depthmap_color.jpg"
            # # plt.savefig(color_map_path)


            # depth_map_colored = cv2.applyColorMap((depth_map_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # print(depth_map_colored.shape) # (512, 832, 3)
            # # Create a color bar using the 'jet' colormap
            # color_bar = np.zeros((256, 50, 3), dtype=np.uint8)
            # for i in range(256):
            #     color_bar[i, :, :] = cv2.applyColorMap(np.array([[i]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0, :]

            # color_bar = cv2.resize(color_bar, (50, 512))

            # mid_bar = np.ones((512, 10, 3), dtype=np.uint8) * 255
            # end_bar = np.ones((512, 50, 3), dtype=np.uint8) * 255
            # depth_map_colored = cv2.hconcat([depth_map_colored, mid_bar, color_bar, end_bar])

            # # Generate scale values corresponding to the color bar
            # scale_values = np.linspace(0, depth_max, num=10)

            # # Calculate the pixel positions for the scale values
            # scale_pixel_positions = np.linspace(0, 512-40, num=10, dtype=int)

            # # Add scale values to the right side of the concatenated image
            # for i, value in enumerate(scale_values):
            #     cv2.putText(depth_map_colored, f'{value:.1f}', (832 + 10 + 50 + 12, scale_pixel_positions[i] + 20),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # denormalize = lambda x: x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

            # img_vis = img.data[0].cpu().numpy()
            # img_vis = img_vis.transpose([1, 2, 0])
            # img_vis = denormalize(img_vis)
            # img_vis = (img_vis * 255).astype(np.uint8)
            # img_depth_map_colored = cv2.hconcat([img_vis[:, :, ::-1], depth_map_colored])

            # color_map_path = output_path[:-4] + "_depthmap_color.jpg"
            # # color_map_path = output_path[:-4] + "_depthmap_color_gt.jpg"
            # cv2.imwrite(color_map_path, img_depth_map_colored.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # print(color_map_path)

            # # gt_depthmap
            # output_path = "./debug/gt_depthmap_diff.png"
            # pred_depthmap_np_diff = np.abs(gt_depthmap - pred_depthmap_np) # 1x1xhxw --> hxw
            # depth_min = pred_depthmap_np_diff.min()
            # depth_max = pred_depthmap_np_diff.max()
            # print("gt_depthmap depth_max: ", depth_max)
            # print("gt_depthmap depth_min: ", depth_min)
            # if depth_max - depth_min > np.finfo("float").eps:
            #     out = 255 * (pred_depthmap_np_diff - depth_min) / (depth_max - depth_min)
            # else:
            #     out = np.zeros(pred_depthmap_np_diff.shape, dtype=pred_depthmap_np_diff.dtype)
            # cv2.imwrite(output_path, out.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # print("pred_depthmap_np_diff.shape: ", pred_depthmap_np_diff.shape)
            # print(output_path)


            # output_path2 = "./debug/gt_depthmap_mask.png"
            # gt_depthmap_vis_onehot = np.copy(gt_depthmap)
            # gt_depthmap_vis_onehot[gt_depthmap > 0] = 1.0
            # gt_depthmap_vis = gt_depthmap_vis_onehot * 255.0
            # cv2.imwrite(output_path2, gt_depthmap_vis.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # print(output_path2)


            # # diff_map + gt_mask
            # output_path = "./debug/gt_depthmap_diff_gt_mask.png"
            # pred_depthmap_np_diff_gt_mask = pred_depthmap_np_diff * gt_depthmap_vis_onehot
            # depth_min = pred_depthmap_np_diff_gt_mask.min()
            # depth_max = pred_depthmap_np_diff_gt_mask.max()
            # print("gt_depthmap depth_max: ", depth_max)
            # print("gt_depthmap depth_min: ", depth_min)
            # if depth_max - depth_min > np.finfo("float").eps:
            #     out = 255 * (pred_depthmap_np_diff_gt_mask - depth_min) / (depth_max - depth_min)
            # else:
            #     out = np.zeros(pred_depthmap_np_diff_gt_mask.shape, dtype=pred_depthmap_np_diff_gt_mask.dtype)
            # cv2.imwrite(output_path, out.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # print("pred_depthmap_np_diff_gt_mask.shape: ", pred_depthmap_np_diff_gt_mask.shape)
            # print(output_path)
            #################################
            ## end pred_depthmap visual ###
            #################################

        # ## debug for onnx2trt
        # b = det_bboxes[0].shape[0]
        # g_feat = det_bboxes[0].view(b, -1)
        # return g_feat, g_feat, g_feat, g_feat

        if self.with_smpl:
            smpl_results, roi_idx = self.simple_test_smpl(x, img_metas, det_bboxes, 
                                                        img.shape, rescale=rescale, nested=True, # nested is a MUST
                                                        mask_pred=pred_masks_rate, 
                                                        # convs=self.convs, 
                                                        img=img,
                                                        residual_depth=True,
                                                        # residual_depth='v4' in self.smpl_head.trans_head.style
                                                        # use_trans=False, # for stageI, 应该外面传参数
                                                        use_trans=True, # for stageII
                                                        global_pelvis_depth=pred_z_tensor,
                                                        is_woLocalTrans=is_woLocalTrans,
                                                        is_woGlobalTransPart=is_woGlobalTransPart,
                                                        depth_range=(0.0, 50.0),
                                                        wo_validDepth=True,
                                                        # use_global_trans_feats=self.not_use_depthmap,
                                                        use_global_trans_feats=True,
                                                        global_trans_feats=global_trans_feats,
                                                        not_use_localFeatsForTrans=self.not_use_localFeatsForTrans,
                                                        not_use_globalFeatsForTrans=self.not_use_globalFeatsForTrans
                                                        ) 

        if smpl_results is not None and not self.not_use_depthmap:
            smpl_results['pred_depthmap'] = pred_depthmap

        smpl_results['pred_bboxes'] = det_bboxes[0]
        smpl_results['pred_labels'] = det_labels[0]

        return smpl_results
        # return bbox_results, segm_results, smpl_results

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        return det_bboxes, det_labels
    
        # if not self.with_mask:
        #     return det_bboxes, det_labels
        # else:
        #     mask_pred, segm_results = self.mask_onnx_export(
        #         x, img_metas, det_bboxes, det_labels, rescale=rescale)
        #     return det_bboxes, det_labels, mask_pred, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        
        max_shape = img_metas[0]['img_shape_for_onnx']
        
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        # return mask_pred, segm_results
        return mask_pred, segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        
        # img_shapes = img_metas[0]['img_shape_for_onnx']
        img_shapes = torch.tensor(np.asarray(img_metas[0]['img_shape'])).to(x[0].device)

        rois = proposals
        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)

        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels
        