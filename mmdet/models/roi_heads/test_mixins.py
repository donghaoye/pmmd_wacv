# Copyright (c) OpenMMLab. All rights reserved.
import sys
import warnings

import numpy as np
import torch

from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


class BBoxTestMixin:

    if sys.version_info >= (3, 7):

        async def async_test_bboxes(self,
                                    x,
                                    img_metas,
                                    proposals,
                                    rcnn_test_cfg,
                                    rescale=False,
                                    **kwargs):
            """Asynchronized test for box head without augmentation."""
            rois = bbox2roi(proposals)
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            sleep_interval = rcnn_test_cfg.get('async_sleep_interval', 0.017)

            async with completed(
                    __name__, 'bbox_head_forward',
                    sleep_interval=sleep_interval):
                cls_score, bbox_pred = self.bbox_head(roi_feats)

            img_shape = img_metas[0]['img_shape']
            scale_factor = img_metas[0]['scale_factor']
            det_bboxes, det_labels = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
            return det_bboxes, det_labels

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            rois = bbox2roi([proposals])
            bbox_results = self._bbox_forward(x, rois)
            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                bbox_results['cls_score'],
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        if merged_bboxes.shape[0] == 0:
            # There is no proposal in the single image
            det_bboxes = merged_bboxes.new_zeros(0, 5)
            det_labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)
        else:
            det_bboxes, det_labels = multiclass_nms(merged_bboxes,
                                                    merged_scores,
                                                    rcnn_test_cfg.score_thr,
                                                    rcnn_test_cfg.nms,
                                                    rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin:

    if sys.version_info >= (3, 7):

        async def async_test_mask(self,
                                  x,
                                  img_metas,
                                  det_bboxes,
                                  det_labels,
                                  rescale=False,
                                  mask_test_cfg=None):
            """Asynchronized test for mask head without augmentation."""
            # image shape of the first image in the batch (only one)
            ori_shape = img_metas[0]['ori_shape']
            scale_factor = img_metas[0]['scale_factor']
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head.num_classes)]
            else:
                if rescale and not isinstance(scale_factor,
                                              (float, torch.Tensor)):
                    scale_factor = det_bboxes.new_tensor(scale_factor)
                _bboxes = (
                    det_bboxes[:, :4] *
                    scale_factor if rescale else det_bboxes)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)

                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                if mask_test_cfg and mask_test_cfg.get('async_sleep_interval'):
                    sleep_interval = mask_test_cfg['async_sleep_interval']
                else:
                    sleep_interval = 0.035
                async with completed(
                        __name__,
                        'mask_head_forward',
                        sleep_interval=sleep_interval):
                    mask_pred = self.mask_head(mask_feats)
                segm_result = self.mask_head.get_seg_masks(
                    mask_pred, _bboxes, det_labels, self.test_cfg, ori_shape,
                    scale_factor, rescale)
            return segm_result

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shapes of images in the batch
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            warnings.warn(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            mask_preds = [[[] for _ in range(self.mask_head.num_classes)]
                            for _ in range(num_imgs)]
            segm_results = [[[] for _ in range(self.mask_head.num_classes)]
                            for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]

            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            mask_pred = mask_results['mask_pred']
            # split batch mask prediction back to each image
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    # print(scale_factors)
                    # print(scale_factors.shape)
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i], _bboxes[i], det_labels[i],
                        self.test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    segm_results.append(segm_result)
        return mask_preds, segm_results

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        """Test for mask head with test time augmentation."""
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                flip_direction = img_meta[0]['flip_direction']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip, flip_direction)
                mask_rois = bbox2roi([_bboxes])
                mask_results = self._mask_forward(x, mask_rois)
                # convert to numpy array to save memory
                aug_masks.append(
                    mask_results['mask_pred'].sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas, self.test_cfg)

            ori_shape = img_metas[0][0]['ori_shape']
            scale_factor = det_bboxes.new_ones(4)
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg,
                ori_shape,
                scale_factor=scale_factor,
                rescale=False)
        return segm_result


class SMPLTestMixin(object):

    def simple_test_smpl(self,
                         x,
                         img_meta,
                         det_bboxes,
                         img_shape,
                         FOCAL_LENGTH=1000,
                         rescale=False,
                         nested=False,
                         mask_pred=None,
                         convs=torch.nn.ModuleList(),
                         img=None,
                         residual_depth=False,
                         use_trans=True,
                         global_pelvis_depth=None,
                         is_woLocalTrans=False,
                         is_woGlobalTransPart=False,
                         depth_range=(0.0, 50.0),
                         wo_validDepth=False,
                         use_global_trans_feats=False,
                         global_trans_feats=None,
                         not_use_localFeatsForTrans=False,
                         not_use_globalFeatsForTrans=False
                         ):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        depth_range = torch.tensor(depth_range, dtype=torch.float)

        if isinstance(det_bboxes, list):
            det_bboxes = torch.cat(det_bboxes, dim=0)
            
        if len(det_bboxes.shape) == 3:
            bboxex_tmp = []
            for i in range(det_bboxes.shape[0]):
                bboxex_tmp.append(det_bboxes[0])
            det_bboxes = torch.cat(bboxex_tmp, dim=0)

        
        if det_bboxes.shape[0] == 0:
            return None, None
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            smpl_rois = bbox2roi([_bboxes])
            smpl_feats = self.smpl_roi_extractor(
                x[:len(self.smpl_roi_extractor.featmap_strides)], smpl_rois)
            if self.with_shared_head:
                smpl_feats = self.shared_head(smpl_feats, mask_pred=mask_pred)

            for conv in convs:
                print('yeah')
                smpl_feats = conv(smpl_feats)

            # 
            pred_bboxes = det_bboxes[:, :4].clone()

            if hasattr(self.smpl_head, 'bbox_feat') and self.smpl_head.bbox_feat and img is not None:
                im_w, im_h = img.shape[-2:]
                bboxes_w = torch.clamp(pred_bboxes[..., 2] - pred_bboxes[..., 0], 1, im_w)
                bboxes_h = torch.clamp(pred_bboxes[..., 3] - pred_bboxes[..., 1], 1, im_h)
                bboxes_aspect_ratio = bboxes_w / bboxes_h
                bbox_info_feat = torch.stack([bboxes_w / im_w, bboxes_h / im_h, bboxes_aspect_ratio], dim=1) # B, 3
                #print(bbox_info_feat)
                bbox_info_feat[torch.isnan(bbox_info_feat)] = 0.
                bbox_info_feat[bbox_info_feat == float("Inf")] = 0.
                print('use bbox feat!!!!!!')
                print("!!>" * 50)
            else:
                bbox_info_feat = None
            smpl_pred = self.smpl_head(smpl_feats, mask_pred=mask_pred, 
                                       bbox_feat=bbox_info_feat, 
                                       global_trans_feats=global_trans_feats,
                                       not_use_localFeatsForTrans=not_use_localFeatsForTrans,
                                       not_use_globalFeatsForTrans=not_use_globalFeatsForTrans
                                       )

            # # debug...
            smpl_pred['pred_trans'] = smpl_pred['pred_trans'][:, 0, :] # batch, 1, 3
            return smpl_pred, smpl_rois

        # pred_rotmat = smpl_pred['pred_rotmat']
        # pred_joints = smpl_pred['pred_joints']
        # pred_vertices = smpl_pred['pred_vertices']
        
        pred_camera = smpl_pred['pred_camera']
        pred_6d = smpl_pred['pred_6d']
        pred_betas = smpl_pred['pred_betas']
        pred_trans = smpl_pred['pred_trans'] if 'pred_trans' in smpl_pred else None
        
        batch_size = pred_bboxes.shape[0]
        img_size = torch.zeros(batch_size, 2).to(pred_camera.device)

        img_size += torch.tensor(img_shape[:-3:-1], dtype=img_size.dtype).to(img_size.device)
        center_pts = (pred_bboxes[..., :2] + pred_bboxes[..., 2:]) / 2
        bboxes_size = torch.max(torch.abs(pred_bboxes[..., 0] - pred_bboxes[..., 2]),
                                torch.abs(pred_bboxes[..., 1] - pred_bboxes[..., 3]))
        valid_boxes = (torch.abs(pred_bboxes[..., 0] - pred_bboxes[..., 2]) > 5) & (torch.abs(
            pred_bboxes[..., 1] - pred_bboxes[..., 3]) > 5)
        
        translation = torch.zeros((batch_size, 3), dtype=pred_camera.dtype).to(pred_camera.device)

        if is_woLocalTrans == True:
            # pred_local_trans = pred['pred_trans'] # B, 1, 3
            # pred_local_trans = pred_trans

            # # if comment the follow 2-lines, it will fail to converge
            # global_pelvis_depth_rate = torch.sigmoid(global_pelvis_depth)
            # global_pelvis_depth = (depth_range[0] + global_pelvis_depth_rate * (depth_range[1] - depth_range[0]))

            # if not self.wo_validDepth:
            # filter by self.depth_range[0] < global_trans < self.depth_range[1]
            depth_range = depth_range.to(global_pelvis_depth.device)
            valid_depth = (global_pelvis_depth > depth_range[0]) & (global_pelvis_depth < depth_range[1])
            valid_boxes = valid_boxes & valid_depth

            global_pelvis_depth = global_pelvis_depth.unsqueeze(-1)
            # depth = pred_local_trans[:, :, -1]                            # debug... # mpjpe: 84.54mm, pampjpe: 57.27mm, pve: 101.09mm, papve: 77.14mm, gce: 592.30mm, gpve: 614.78mm, avg_gpve: 589.55mm agora                                                    
            # depth = global_pelvis_depth + pred_local_trans[:, :, -1]      # debug... # mpjpe: 84.54mm, pampjpe: 57.27mm, pve: 101.09mm, papve: 77.14mm, gce: 2864.02mm, gpve: 2884.99mm, avg_gpve: 2872.42mm agora
            depth = global_pelvis_depth

            translation[:, :-1] = pred_camera[:, 1:] + (center_pts - img_size / 2) * depth / FOCAL_LENGTH
            translation[:, -1] = depth[:, 0]
            # translation[:, :2] = translation[:, :2] + pred_local_trans[:, 0, :2] # debug...
            translation[:, :2] = translation[:, :2]

            print(">>>>> is_woLocalTrans: ", is_woLocalTrans)

        elif is_woGlobalTransPart == True:
            # pred_local_trans = pred['pred_trans'] # B, 1, 3
            pred_local_trans = pred_trans # B, 1, 3

            # # filter by self.depth_range[0] < global_trans < self.depth_range[1]
            # self.depth_range = self.depth_range.to(global_pelvis_depth.device)
            # valid_depth = (global_pelvis_depth > self.depth_range[0]) & (global_pelvis_depth < self.depth_range[1])
            # valid_boxes = valid_boxes & valid_depth

            # global_pelvis_depth = global_pelvis_depth.unsqueeze(-1)
            # depth = global_pelvis_depth + pred_local_trans[:, :, -1]
            depth = pred_local_trans[:, :, -1]

            translation[:, :-1] = pred_camera[:, 1:] + (center_pts - img_size / 2) * depth / FOCAL_LENGTH
            translation[:, -1] = depth[:, 0]
            translation[:, :2] = translation[:, :2] + pred_local_trans[:, 0, :2]

            print(">>>>> is_woGlobalTransPart", is_woGlobalTransPart)


        elif use_global_trans_feats == True:
            if not wo_validDepth:
                self.depth_range = self.depth_range.to(global_pelvis_depth.device)
                valid_depth = (global_pelvis_depth > self.depth_range[0]) & (global_pelvis_depth < self.depth_range[1])
                valid_boxes = valid_boxes & valid_depth

            depth = pred_trans[:, :, -1]

            translation[:, :-1] = pred_camera[:, 1:] + (center_pts - img_size / 2) * depth / FOCAL_LENGTH
            translation[:, -1] = depth[:, 0]
            translation[:, :2] = translation[:, :2] + pred_trans[:, 0, :2]
            # print(">>>>> use_global_trans_feats: ", use_global_trans_feats)
            # print(">>>>> global_trans_feats is None: ", global_trans_feats is None)

        # elif use_trans and pred_trans is not None and not self.not_use_resTrans:
        elif use_trans and pred_trans is not None:
            if not residual_depth:
                depth = pred_trans[:,0,-1] # P, 1, 3
                depth = depth[:, None]
            else:
                # newly residual_depth version 02/10/2023
                # depth = 2 * self.FOCAL_LENGTH / (1e-6 + pred_camera[..., 0] * bboxes_size)
                # depth = depth[:, None]

                if not wo_validDepth:
                    depth_range = depth_range.to(global_pelvis_depth.device)
                    valid_depth = (global_pelvis_depth > depth_range[0]) & (global_pelvis_depth < depth_range[1])
                    valid_boxes = valid_boxes & valid_depth

                global_pelvis_depth = global_pelvis_depth.unsqueeze(-1)
                depth = global_pelvis_depth + pred_trans[:, :, -1]

                # print("pred_trans[:, :, -1]: ", pred_trans[:, :, -1])
                # print("global_pelvis_depth: ", global_pelvis_depth)
                # print("depth: ", depth)

                # # filter by self.depth_range[0] < global_trans + local_trans < self.depth_range[1]
                # valid_depth = (depth[:, 0] > self.depth_range[0]) & (depth[:, 0] < self.depth_range[1])
                # valid_boxes = valid_boxes & valid_depth

                translation[:, :-1] = pred_camera[:, 1:] + (center_pts - img_size / 2) * depth / FOCAL_LENGTH
                translation[:, -1] = depth[:, 0]
                translation[:, :2] = translation[:, :2] + pred_trans[:, 0, :2]
                print("ours")
        else:
            # depth = 2 * FOCAL_LENGTH / (1e-6 + pred_camera[..., 0] * bboxes_size)
            # depth = depth[:, None]
            depth = 2 * FOCAL_LENGTH / (1e-9 + pred_camera[..., 0] * bboxes_size)
            depth = depth[:, None]
            # translation[:, :-1] = depth[:, None] * (center_pts + pred_camera[:, 1:] * bboxes_size.unsqueeze(-1) - img_size / 2) / self.FOCAL_LENGTH
            translation[:, :-1] = pred_camera[:, 1:] + (center_pts - img_size / 2) * depth / FOCAL_LENGTH
            translation[:, -1] = depth[:, 0]
            print("else")

        # translation = torch.zeros((batch_size, 3), dtype=pred_camera.dtype).to(pred_joints.device)
        # translation[:, :-1] = depth * (center_pts + pred_camera[:, 1:] * bboxes_size.unsqueeze(-1) - img_size / 2) / FOCAL_LENGTH
        # translation[:, -1] = depth[:, 0]
        # if use_trans and pred_trans is not None:
        #     translation[:, :2] += pred_trans[:,0,:2]

        # print("smpl_pred.keys: ", smpl_pred.keys())

        smpl_pred['pred_trans'] = translation
        smpl_pred['valid_bboxes'] = valid_boxes

        return smpl_pred, smpl_rois

    def aug_test_smpl(self, feats, img_metas, det_bboxes, det_labels):
        raise NotImplementedError("SMPL test with augmentation has not been implemented yet.")
