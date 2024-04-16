# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule

from ..builder import build_shared_head


class BaseRoIHead(BaseModule, metaclass=ABCMeta):
    """Base class for RoIHeads."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 smpl_roi_extractor=None,
                 smpl_head=None,
                 trans_roi_extractor=None,
                 trans_head=None,
                 adabin_roi_extractor=None,
                 adabin_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 smpl_weight=1,
                 nested=False,
                 mask_attention=False,
                 wo_SMPL=False,
                 not_use_depthmap=False,
                 not_use_localFeatsForTrans=False,
                 not_use_globalFeatsForTrans=False,
                 init_cfg=None):
        super(BaseRoIHead, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.not_use_depthmap = not_use_depthmap
        self.not_use_localFeatsForTrans = not_use_localFeatsForTrans
        self.not_use_globalFeatsForTrans = not_use_globalFeatsForTrans
        self.mask_attention = mask_attention

        if shared_head is not None:
            shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        if smpl_head is not None:
            self.init_smpl_head(smpl_roi_extractor, smpl_head)

        if trans_head is not None:
            self.init_trans_head(trans_roi_extractor, trans_head)

        if adabin_head is not None:
            self.init_adabin_head(adabin_roi_extractor, adabin_head)

        self.init_assigner_sampler()

    @property
    def with_bbox(self):
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_smpl(self):
        """bool: whether the RoI head contains a `smpl_head`"""
        return hasattr(self, 'smpl_head') and self.smpl_head is not None
    
    @property
    def with_trans(self):
        """bool: whether the RoI head contains a `trans_head`"""
        return hasattr(self, 'trans_head') and self.trans_head is not None
    
    @property
    def with_adabin(self):
        """bool: whether the RoI head contains a `adabin_head`"""
        return hasattr(self, 'adabin_head') and self.adabin_head is not None
    
    @property
    def with_shared_head(self):
        """bool: whether the RoI head contains a `shared_head`"""
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @abstractmethod
    def init_bbox_head(self):
        """Initialize ``bbox_head``"""
        pass

    @abstractmethod
    def init_mask_head(self):
        """Initialize ``mask_head``"""
        pass

    @abstractmethod
    def init_smpl_head(self):
        """Initialize ``smpl_head``"""
        pass

    @abstractmethod
    def init_trans_head(self):
        """Initialize ``trans_head``"""
        pass

    @abstractmethod
    def init_adabin_head(self):
        """Initialize ``trans_head``"""
        pass

    @abstractmethod
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    @abstractmethod
    def forward_train(self,
                      x,
                      img_meta,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """Forward function during training."""

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False,
                                **kwargs):
        """Asynchronized test function."""
        raise NotImplementedError

    def simple_test(self,
                    x,
                    proposal_list,
                    img_meta,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """