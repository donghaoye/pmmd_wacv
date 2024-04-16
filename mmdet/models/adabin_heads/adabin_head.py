
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.adabin_heads.miniViT import mViT

from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer

import warnings

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


# @HEADS.register_module()
class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)
        return x
    
class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=64, n_classes=1, bottleneck_features=64):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=64 * 2, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 64, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 64, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 64, output_features=features // 16)

        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(features // 16, n_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        
        # for i in range(len(features)):
        #     print(i, features[i].shape)
        # print('-' * 50)
    
        '''
        0 torch.Size([2, 256, 128, 208])
        1 torch.Size([2, 256, 64, 104])
        2 torch.Size([2, 256, 32, 52])
        3 torch.Size([2, 256, 16, 26])
        4 torch.Size([2, 256, 8, 13])
        '''
        # x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[0], features[1], features[1], features[3], features[4]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        #         x_d5 = self.up5(x_d4, features[0])
        out = self.conv3(x_d4)
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        #     return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
        return out


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


# @HEADS.register_module()
class UnetAdaptiveBins(BaseModule):
    def __init__(self, n_bins=100, min_val=0.1, max_val=50, norm='linear'):
        super(UnetAdaptiveBins, self).__init__()
        self.n_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        # self.encoder = Encoder(backend)
        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=128, norm=norm)

        self.decoder = DecoderBN(n_classes=128)
        self.conv_out = nn.Sequential(
                                    # nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                    nn.Conv2d(103, n_bins, kernel_size=1, stride=1, padding=0),
                                    nn.Softmax(dim=1))

    def forward(self, x, **kwargs):
        # unet_out = self.decoder(self.encoder(x), **kwargs)
        unet_out = self.decoder(x, **kwargs)

        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)

        # Post process
        # n, c, h, w = out.shape
        # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        return bin_edges, pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, **kwargs):
        basemodel_name = 'tf_efficientnet_b5_ap'

        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, n_bins=n_bins, **kwargs)
        print('Done.')
        return m


@HEADS.register_module
class AdaBinHead(nn.Module):
    # def __init__(self, in_channels=256, out_channels=1, loss_cfg=dict(type='DepthLoss')):
    def __init__(self, 
                in_channels=256, 
                out_channels=1,
                # depth_type='mean',
                depth_type='mask_confidence',
                use_sigmoid=False,
                depth_range=(1., 26.),
                only_mask_confidence=False,
                loss_L1=dict(type='DepthmapL1Loss', valid_mask=True, loss_weight=0.1),
                loss_decode=dict(type='SigLoss', valid_mask=True, loss_weight=0.1),
                loss_chamfer=dict(type='BinsChamferLoss', loss_weight=0.001),
                n_bins=100, 
                min_val=0.1, 
                max_val=50, 
                norm='linear'
                ):
        super(AdaBinHead, self).__init__()
        # self.trans = Translation(in_channels)
        dim = in_channels
        self.depth_type = depth_type
        self.use_sigmoid = use_sigmoid
        self.depth_range = depth_range
        self.only_mask_confidence = only_mask_confidence

        self.conv64 = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        print("initial adabin Head...")
        self.adabin_model = UnetAdaptiveBins(n_bins=n_bins, min_val=min_val, max_val=max_val, norm=norm)

        # self.conv_last = nn.Sequential(
        #     # nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
        #     Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        #     nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
        #     Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        #     nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # )


        self.loss_L1 = build_loss(loss_L1)
        self.loss_decode = build_loss(loss_decode)
        self.loss_chamfer = build_loss(loss_chamfer)


    '''
    5
    trans head x_s shape:  torch.Size([8, 256, 128, 208])
    trans head x_s shape:  torch.Size([8, 256, 64, 104])
    trans head x_s shape:  torch.Size([8, 256, 32, 52])
    trans head x_s shape:  torch.Size([8, 256, 16, 26])
    trans head x_s shape:  torch.Size([8, 256, 8, 13])
    target_lvls tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 2], device='cuda:0')
    targert_img_ids [0, 0, 1, 1, 2, 3, 4, 5, 6, 7]
    '''
    def forward(self, x, sampling_results, pred_masks):
        # feature_ly1 = self.conv1(x[0])
        # feature_ly2 = self.conv2(x[1])
        # feature_ly3 = self.conv3(x[2])
        # feature_ly4 = self.conv4(x[3])
        # feature_ly1234 = torch.cat((feature_ly1, feature_ly2, feature_ly3, feature_ly4), dim=1)
        # pred_depthmap = self.conv_last(feature_ly1234)

        x_dim64 = []
        for x_sub in x:
            x_dim64.append(self.conv64(x_sub))

        # bins, pred = self.adabin_model(x)
        bin_edges, depth_pred = self.adabin_model(x_dim64)
        # print("bins.shape, depth_pred.shape: ", bin_edges.shape, depth_pred.shape)
        # pred_depthmap = self.conv_last(depth_pred)
        # print("pred_depthmap.shape: ", pred_depthmap.shape)

        pred_depthmap = resize(
            input=depth_pred,
            size=(512, 832),
            mode='bilinear',
            align_corners=True,
            warning=False)
        
        # print("pred_depthmap2.shape: ", pred_depthmap2.shape)

        # print("TRANS_HEAD before conv_mask pred_masks.shape: ", pred_masks.shape)
        # pred_masks = self.conv_mask(pred_masks)
        # print("TRANS_HEAD after conv_mask pred_masks.shape: ", pred_masks.shape)

        if self.use_sigmoid:
            pred_depthmap_rate = torch.sigmoid(pred_depthmap)
            pred_depthmap = (self.depth_range[0] + pred_depthmap_rate * (self.depth_range[1] - self.depth_range[0]))

        pred_z_tensor_list = list()
        if self.depth_type == 'mask_confidence_original':
            pred_depthmap_mask = torch.zeros_like(pred_depthmap).to(pred_depthmap.device)
            pred_depthmap_mask[torch.where(pred_depthmap > 0.0)] = 1
            pred_depthmap_copy = pred_depthmap.clone() * pred_depthmap_mask

            pred_masks_softmax = F.softmax(pred_masks, dim=1)
            predicted_masks = torch.argmax(pred_masks_softmax, dim=1)
            pred_masks_onehot = (predicted_masks == 1).float()
            pred_masks_confidence = pred_masks_softmax[:, -1, :, :]

            bbox_index_global = 0
            for img_id in range(len(sampling_results)):
                curr_pos_bboxes = sampling_results[img_id].pos_bboxes
                
                # 取pred_mask, 后与depthmap做交集, 后再取值
                for c_bbox in curr_pos_bboxes:
                    curr_pred_mask = pred_masks_onehot[bbox_index_global:bbox_index_global+1]
                    curr_pred_mask_confidence = pred_masks_confidence[bbox_index_global:bbox_index_global+1]
                    bbox_index_global = bbox_index_global + 1

                    x1, y1, x2, y2 = c_bbox
                    x1, y1, x2, y2 = x1.long().cpu().numpy(), y1.long().cpu().numpy(), x2.long().cpu().numpy(), y2.long().cpu().numpy()
                    
                    # 超出图像外是负数
                    w = np.maximum(x2 - x1 + 1, 2)
                    h = np.maximum(y2 - y1 + 1, 2)
                    b, c, h_img, w_img = pred_depthmap_copy.shape
                    start_x1, start_y1 = 0, 0
                    if x1 < 0:
                        start_x1 = -1 * x1
                    if y1 < 0:
                        start_y1 = -1 * y1
                    # left or up, out of image
                    if (x1 < 0 or y1 < 0) and (start_y1 + h) < h_img and (start_x1 + w) < w_img \
                        and start_y1 < h and  start_x1 < w:
                        cropped_pred_depthmap = torch.zeros((b, c, h, w)).to(pred_depthmap_copy.device)
                        cropped_pred_depthmap[:, :, start_y1:, start_x1:] = pred_depthmap_copy[:, :, 0:h-start_y1, 0:w-start_x1] # w = np.maximum(x2 - x1 + 1, 2), if x1<0, means w has been add delta_x
                    else:
                        cropped_pred_depthmap = pred_depthmap_copy[:, :, y1:y1 + h, x1:x1 + w]

                    b, c, h, w = cropped_pred_depthmap.shape
                    max_hw = max(h, w)
                    square_cropped_pred_depthmap = torch.zeros((b, c, max_hw, max_hw)).to(cropped_pred_depthmap.device)
                    pad_h, pad_w = 0, 0
                    if max_hw > h:
                        pad_h = (max_hw - h) * 0.5
                        pad_h = int(pad_h)
                    if max_hw > w:
                        pad_w = (max_hw - w) * 0.5
                        pad_w = int(pad_w)
                    square_cropped_pred_depthmap[:, :, pad_h:cropped_pred_depthmap.shape[-2]+pad_h, pad_w:cropped_pred_depthmap.shape[-1]+pad_w] = cropped_pred_depthmap # align center
                    cropped_pred_depthmap = square_cropped_pred_depthmap

                    if max_hw > 0:
                        curr_pred_mask = curr_pred_mask.unsqueeze(0)
                        curr_pred_mask = torch.nn.functional.interpolate(curr_pred_mask, size=(max_hw, max_hw), mode='bilinear', align_corners=False)
                        curr_pred_mask[torch.where(curr_pred_mask > 0.0)] = 1.0
                        cropped_pred_depthmap = curr_pred_mask * cropped_pred_depthmap

                        curr_pred_mask_confidence = curr_pred_mask_confidence.unsqueeze(0)
                        curr_pred_mask_confidence = torch.nn.functional.interpolate(curr_pred_mask_confidence, size=(max_hw, max_hw), mode='bilinear', align_corners=False)
                        curr_pred_mask_confidence = curr_pred_mask * curr_pred_mask_confidence

                    if cropped_pred_depthmap.numel() == 0 or curr_pred_mask_confidence.numel() == 0: # check empty
                        pred_z = torch.tensor(0).float().to(cropped_pred_depthmap.device)
                    else:
                        pred_z = (cropped_pred_depthmap * curr_pred_mask_confidence).sum() / curr_pred_mask_confidence.sum()
                    if torch.isnan(pred_z):
                        pred_z = torch.tensor(0).float().to(cropped_pred_depthmap.device)

                    pred_z = pred_z.unsqueeze(0)
                    pred_z_tensor_list.append(pred_z)

        else:
            ## 框内人物深度的均值(或中值)作为depth
            pred_depthmap_mask = torch.zeros_like(pred_depthmap).to(pred_depthmap.device)
            if self.depth_type == 'mask_confidence_initial_w_mean':
                pred_depthmap_mask[torch.where(pred_depthmap > 0.0)] = 1
            else:
                pred_depthmap_mask[torch.where(pred_depthmap > 1.0)] = 1
            pred_depthmap_copy = pred_depthmap.clone() * pred_depthmap_mask
            # pred_depthmap_copy = pred_depthmap # depthmap range already > 0.0, but could == 0.0

            # print(f"torch.max(pred_depthmap_copy): {torch.max(pred_depthmap_copy)}")
            # print(f"torch.min(pred_depthmap_copy): {torch.min(pred_depthmap_copy)}")
            # print(f"torch.max(pred_depthmap): {torch.max(pred_depthmap)}")
            # print(f"torch.min(pred_depthmap): {torch.min(pred_depthmap)}")
            # print("-" * 30)

            if self.depth_type == 'mask_confidence' or self.depth_type == 'mask_confidence_initial_w_mean'\
                  or self.depth_type == 'mask_confidence_initial_w_mean_full_depthmap'\
                  or self.depth_type == 'mask_confidence_initial_w_median':
                if pred_masks is not None:
                    pred_masks_softmax = F.softmax(pred_masks, dim=1)
                    predicted_masks = torch.argmax(pred_masks_softmax, dim=1)
                    pred_masks_onehot = (predicted_masks == 1).float()
                    pred_masks_confidence = pred_masks_softmax[:, -1, :, :]
                    # # 检查每个概率分布向量中的元素和是否为 1, 是第二通道上面的概率累加和，并不是每个像素上面累加和为1
                    # for i in range(pred_masks_softmax.shape[0]):
                    #     for j in range(pred_masks_softmax.shape[2]):
                    #         for k in range(pred_masks_softmax.shape[3]):
                    #             sum_jk = pred_masks_softmax[i, :, j, k].sum()
                    #             print("sum_jk: ", sum_jk) # sum_jk:  tensor(1.0000, device='cuda:0', grad_fn=<SumBackward0>)
                    #             print("-" * 50)

            # pred_z_tensor_list = list()
            bbox_index_global = 0
            for img_id in range(len(sampling_results)):
                curr_pos_bboxes = sampling_results[img_id].pos_bboxes
                for c_bbox in curr_pos_bboxes:
                    if self.depth_type == 'mask_confidence' or self.depth_type == 'mask_confidence_initial_w_mean'\
                          or self.depth_type == 'mask_confidence_initial_w_mean_full_depthmap'\
                          or self.depth_type == 'mask_confidence_initial_w_median':
                        if pred_masks is not None:
                            curr_pred_mask = pred_masks_onehot[bbox_index_global:bbox_index_global+1]
                            curr_pred_mask_confidence = pred_masks_confidence[bbox_index_global:bbox_index_global+1]
                        else:
                            curr_pred_mask = torch.ones([1, 14, 14]).to(pred_depthmap.device)
                            curr_pred_mask_confidence = torch.ones([1, 14, 14]).to(pred_depthmap.device)
                        # curr_pred_mask = pred_masks_onehot[bbox_index_global:bbox_index_global+1]
                        # curr_pred_mask_confidence = pred_masks_confidence[bbox_index_global:bbox_index_global+1]
                        bbox_index_global = bbox_index_global + 1

                    x1, y1, x2, y2 = c_bbox
                    x1, y1, x2, y2 = x1.long().cpu().numpy(), y1.long().cpu().numpy(), x2.long().cpu().numpy(), y2.long().cpu().numpy()
                    # cropped_pred_depthmap = pred_depthmap_copy[:, :, y1:y2, x1:x2]
                    
                    # 超出图像外是负数
                    w = np.maximum(x2 - x1 + 1, 2)
                    h = np.maximum(y2 - y1 + 1, 2)
                    b, c, h_img, w_img = pred_depthmap_copy.shape
                    start_x1, start_y1 = 0, 0
                    if x1 < 0:
                        start_x1 = -1 * x1
                    if y1 < 0:
                        start_y1 = -1 * y1
                    # left or up, out of image
                    if (x1 < 0 or y1 < 0) and (start_y1 + h) < h_img and (start_x1 + w) < w_img \
                        and start_y1 < h and  start_x1 < w:
                        cropped_pred_depthmap = torch.zeros((b, c, h, w)).to(pred_depthmap_copy.device)
                        cropped_pred_depthmap[:, :, start_y1:, start_x1:] = pred_depthmap_copy[:, :, 0:h-start_y1, 0:w-start_x1] # w = np.maximum(x2 - x1 + 1, 2), if x1<0, means w has been add delta_x
                    else:
                        cropped_pred_depthmap = pred_depthmap_copy[:, :, y1:y1 + h, x1:x1 + w]

                    b, c, h, w = cropped_pred_depthmap.shape
                    max_hw = max(h, w)
                    square_cropped_pred_depthmap = torch.zeros((b, c, max_hw, max_hw)).to(cropped_pred_depthmap.device)
                    pad_h, pad_w = 0, 0
                    if max_hw > h:
                        pad_h = (max_hw - h) * 0.5
                        pad_h = int(pad_h)
                    if max_hw > w:
                        pad_w = (max_hw - w) * 0.5
                        pad_w = int(pad_w)
                    square_cropped_pred_depthmap[:, :, pad_h:cropped_pred_depthmap.shape[-2]+pad_h, pad_w:cropped_pred_depthmap.shape[-1]+pad_w] = cropped_pred_depthmap # align center
                    cropped_pred_depthmap = square_cropped_pred_depthmap

                    pred_z_initial = torch.tensor(0).float().to(cropped_pred_depthmap.device)
                    if max_hw > 0 and (self.depth_type == 'mask_confidence' or self.depth_type == 'mask_confidence_initial_w_mean'\
                                        or self.depth_type == 'mask_confidence_initial_w_mean_full_depthmap' \
                                        or self.depth_type == 'mask_confidence_initial_w_median'):
                        if self.depth_type == 'mask_confidence_initial_w_mean':
                            # pred_z_initial = cropped_pred_depthmap.mean()
                            pred_z_initial = cropped_pred_depthmap[torch.where(cropped_pred_depthmap > 1.0)].mean() # filter 0
                        elif self.depth_type == 'mask_confidence_initial_w_mean_full_depthmap':
                            pred_z_initial = cropped_pred_depthmap.mean()
                        elif self.depth_type == 'mask_confidence_initial_w_median':
                            if torch.sum(cropped_pred_depthmap) > 0:
                                pred_z_initial = cropped_pred_depthmap[torch.where(cropped_pred_depthmap > 1.0)].median() # filter 0
                       
                        # print(f"torch.max(pred_z_initial): {torch.max(pred_z_initial)}")
                        # print(f"torch.max(pred_depthmap): {torch.max(pred_depthmap)}")
                        # print(f"torch.min(pred_depthmap): {torch.min(pred_depthmap)}")
                        # print("-" * 30)

                        if torch.isnan(pred_z_initial):
                            pred_z_initial = torch.tensor(0).float().to(cropped_pred_depthmap.device)

                        curr_pred_mask = curr_pred_mask.unsqueeze(0)
                        curr_pred_mask = torch.nn.functional.interpolate(curr_pred_mask, size=(max_hw, max_hw), mode='bilinear', align_corners=False)
                        curr_pred_mask[torch.where(curr_pred_mask > 0.0)] = 1.0
                        if not self.only_mask_confidence:
                            cropped_pred_depthmap = curr_pred_mask * cropped_pred_depthmap

                        curr_pred_mask_confidence = curr_pred_mask_confidence.unsqueeze(0)
                        curr_pred_mask_confidence = torch.nn.functional.interpolate(curr_pred_mask_confidence, size=(max_hw, max_hw), mode='bilinear', align_corners=False)
                        if not self.only_mask_confidence:
                            curr_pred_mask_confidence = curr_pred_mask * curr_pred_mask_confidence

                    if self.depth_type == 'mask_confidence' or self.depth_type == 'mask_confidence_initial_w_mean'\
                          or self.depth_type == 'mask_confidence_initial_w_mean_full_depthmap'\
                          or self.depth_type == 'mask_confidence_initial_w_median':
                        if cropped_pred_depthmap.numel() == 0 or curr_pred_mask_confidence.numel() == 0: # check empty
                            pred_z = pred_z_initial
                        else:
                            if not self.depth_type == 'mask_confidence_initial_w_mean_full_depthmap':
                                cropped_pred_depthmap_mask = torch.zeros_like(cropped_pred_depthmap).to(cropped_pred_depthmap.device)
                                if self.depth_type == 'mask_confidence_initial_w_mean':
                                    cropped_pred_depthmap_mask[torch.where(cropped_pred_depthmap > 0.0)] = 1
                                else:
                                    cropped_pred_depthmap_mask[torch.where(cropped_pred_depthmap > 1.0)] = 1
                                curr_pred_mask_confidence = cropped_pred_depthmap_mask * curr_pred_mask_confidence
                            
                            curr_pred_mask_confidence_sum = curr_pred_mask_confidence.sum()
                            if curr_pred_mask_confidence_sum > 0.0:
                                pred_z = (cropped_pred_depthmap * curr_pred_mask_confidence).sum() / curr_pred_mask_confidence_sum
                            else:
                                pred_z = pred_z_initial
                        if torch.isnan(pred_z) or (pred_z - 0.0) < sys.float_info.epsilon:
                            # print("pred_z: ", pred_z)
                            # print("sys.float_info.epsilon: ", sys.float_info.epsilon)
                            pred_z = pred_z_initial
                                
                    elif self.depth_type == 'median':
                        cropped_pred_depthmap = cropped_pred_depthmap[torch.where(cropped_pred_depthmap > 1.0)] # filter 0
                        if cropped_pred_depthmap.numel() == 0: # check empty
                            pred_z = torch.tensor(0).float().to(cropped_pred_depthmap.device)
                        else:
                            pred_z = cropped_pred_depthmap.median()
                            
                    elif self.depth_type == 'mean':
                        cropped_pred_depthmap = cropped_pred_depthmap[torch.where(cropped_pred_depthmap > 1.0)] # filter 0
                        if cropped_pred_depthmap.numel() == 0: # check empty
                            pred_z = torch.tensor(0).float().to(cropped_pred_depthmap.device)
                        else:
                            pred_z = cropped_pred_depthmap.mean()
                        if torch.isnan(pred_z):
                            pred_z = torch.tensor(0).float().to(cropped_pred_depthmap.device)

                    pred_z = pred_z.unsqueeze(0)
                    pred_z_tensor_list.append(pred_z)

        pred_z_tensor = torch.cat(pred_z_tensor_list, dim=0)
        outputs = {
                'pred_depthmap': pred_depthmap.float(),
                'pred_z_tensor': pred_z_tensor.float(),
                'bin_edges': bin_edges
            }

        return outputs
    

    # def forward_test(self, x, sampling_results):
    def forward_test(self, x, pred_bboxes, pred_masks):
        x_dim64 = []
        for x_sub in x:
            x_dim64.append(self.conv64(x_sub))

        bin_edges, depth_pred = self.adabin_model(x_dim64)

        pred_depthmap = resize(
            input=depth_pred,
            size=(512, 832),
            mode='bilinear',
            align_corners=True,
            warning=False)
        
        if self.use_sigmoid:
            # depth_range=(1., 26.) # hard code for depth range
            # depth_range=(-2., 30.) # hard code for depth range # for more robust
            pred_depthmap_rate = torch.sigmoid(pred_depthmap)
            pred_depthmap = (self.depth_range[0] + pred_depthmap_rate * (self.depth_range[1] - self.depth_range[0]))


        '''
        torch.Size([8, 256, 128, 208]) torch.Size([8, 256, 64, 104]) torch.Size([8, 256, 32, 52]) torch.Size([8, 256, 16, 26])
        torch.Size([8, 1, 128, 208]) torch.Size([8, 1, 128, 208]) torch.Size([8, 1, 128, 208]) torch.Size([8, 1, 128, 208])
        torch.Size([8, 4, 128, 208])
        torch.Size([8, 1, 512, 832])
        '''
        # 选出以人为单位的框
        # 遍历框, 去选人对应的图
        # 然后在图, 用坐标选择depth

        # pred_z_tensor = pred_bboxes.new_zeros(center_xy.size()[0])
        pred_z_tensor_list = list()
        if self.depth_type == 'mask_confidence_original':
            pred_depthmap_mask = torch.zeros_like(pred_depthmap).to(pred_depthmap.device)
            pred_depthmap_mask[torch.where(pred_depthmap > 0.0)] = 1
            pred_depthmap_copy = pred_depthmap.clone() * pred_depthmap_mask

            pred_masks_softmax = F.softmax(pred_masks, dim=1)
            predicted_masks = torch.argmax(pred_masks_softmax, dim=1)
            pred_masks_onehot = (predicted_masks == 1).float()
            pred_masks_confidence = pred_masks_softmax[:, -1, :, :]

            bbox_index_global = 0
            for img_id in range(x[0].shape[0]): # image batch_size = 1 for testing and eval
                curr_pos_bboxes = pred_bboxes[img_id]
                # 取pred_mask, 后与depthmap做交集, 后再取值
                for c_bbox in curr_pos_bboxes:
                    curr_pred_mask = pred_masks_onehot[bbox_index_global:bbox_index_global+1]
                    curr_pred_mask_confidence = pred_masks_confidence[bbox_index_global:bbox_index_global+1]
                    bbox_index_global = bbox_index_global + 1

                    x1, y1, x2, y2 = c_bbox[:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 超出图像外是负数
                    w = np.maximum(x2 - x1 + 1, 2)
                    h = np.maximum(y2 - y1 + 1, 2)
                    b, c, h_img, w_img = pred_depthmap_copy.shape
                    start_x1, start_y1 = 0, 0
                    if x1 < 0:
                        start_x1 = -1 * x1
                    if y1 < 0:
                        start_y1 = -1 * y1
                    # left or up, out of image
                    if (x1 < 0 or y1 < 0) and (start_y1 + h) < h_img and (start_x1 + w) < w_img \
                        and start_y1 < h and  start_x1 < w:
                        cropped_pred_depthmap = torch.zeros((b, c, h, w)).to(pred_depthmap_copy.device)
                        cropped_pred_depthmap[:, :, start_y1:, start_x1:] = pred_depthmap_copy[:, :, 0:h-start_y1, 0:w-start_x1] # w = np.maximum(x2 - x1 + 1, 2), if x1<0, means w has been add delta_x
                    else:
                        cropped_pred_depthmap = pred_depthmap_copy[:, :, y1:y1 + h, x1:x1 + w]

                    b, c, h, w = cropped_pred_depthmap.shape
                    max_hw = max(h, w)
                    square_cropped_pred_depthmap = torch.zeros((b, c, max_hw, max_hw)).to(cropped_pred_depthmap.device)
                    pad_h, pad_w = 0, 0
                    if max_hw > h:
                        pad_h = (max_hw - h) * 0.5
                        pad_h = int(pad_h)
                    if max_hw > w:
                        pad_w = (max_hw - w) * 0.5
                        pad_w = int(pad_w)
                    square_cropped_pred_depthmap[:, :, pad_h:cropped_pred_depthmap.shape[-2]+pad_h, pad_w:cropped_pred_depthmap.shape[-1]+pad_w] = cropped_pred_depthmap # align center
                    cropped_pred_depthmap = square_cropped_pred_depthmap

                    if max_hw > 0:
                        curr_pred_mask = curr_pred_mask.unsqueeze(0)  # torch.Size([2, 56, 56]), 最后一层是foreground, need to visual
                        curr_pred_mask = torch.nn.functional.interpolate(curr_pred_mask, size=(max_hw, max_hw), mode='bilinear', align_corners=False)
                        curr_pred_mask[torch.where(curr_pred_mask > 0.0)] = 1.0
                        cropped_pred_depthmap = curr_pred_mask * cropped_pred_depthmap

                        curr_pred_mask_confidence = curr_pred_mask_confidence.unsqueeze(0)
                        curr_pred_mask_confidence = torch.nn.functional.interpolate(curr_pred_mask_confidence, size=(max_hw, max_hw), mode='bilinear', align_corners=False)
                        curr_pred_mask_confidence = curr_pred_mask * curr_pred_mask_confidence
                    
                    if cropped_pred_depthmap.numel() == 0 or curr_pred_mask_confidence.numel() == 0: # check empty
                        pred_z = torch.tensor(0).float().to(cropped_pred_depthmap.device)
                    else:
                        pred_z = (cropped_pred_depthmap * curr_pred_mask_confidence).sum() / curr_pred_mask_confidence.sum()
                    if torch.isnan(pred_z):
                        pred_z = torch.tensor(0).float().to(cropped_pred_depthmap.device)
                    
                    pred_z = pred_z.unsqueeze(0)
                    pred_z_tensor_list.append(pred_z)

        else:
            pred_depthmap_mask = torch.zeros_like(pred_depthmap).to(pred_depthmap.device)
            if self.depth_type == 'mask_confidence_initial_w_mean':
                pred_depthmap_mask[torch.where(pred_depthmap > 0.0)] = 1
            else:
                pred_depthmap_mask[torch.where(pred_depthmap > 1.0)] = 1
            pred_depthmap_copy = pred_depthmap.clone() * pred_depthmap_mask

            # print(f"torch.max(pred_depthmap): {torch.max(pred_depthmap)}")
            # print(f"torch.min(pred_depthmap): {torch.min(pred_depthmap)}")
            # print("-" * 30)
            # # pred_depthmap_copy = pred_depthmap # depthmap range already > 0.0, but will == 0.0

            if self.depth_type == 'mask_confidence' or self.depth_type == 'mask_confidence_initial_w_mean'\
                  or self.depth_type == 'mask_confidence_initial_w_mean_full_depthmap'\
                  or self.depth_type == 'mask_confidence_initial_w_median':
                if pred_masks is not None:
                    pred_masks_softmax = F.softmax(pred_masks, dim=1)
                    predicted_masks = torch.argmax(pred_masks_softmax, dim=1)
                    pred_masks_onehot = (predicted_masks == 1).float()
                    pred_masks_confidence = pred_masks_softmax[:, -1, :, :]

            # pred_z_tensor_list = list()
            bbox_index_global = 0
            for img_id in range(x[0].shape[0]): # image batch_size = 1 for testing and eval
                curr_pos_bboxes = pred_bboxes[img_id]
                
                # 取pred_mask, 后与depthmap做交集, 后再取值
                count = 0
                for c_bbox in curr_pos_bboxes:
                    if self.depth_type == 'mask_confidence' or self.depth_type == 'mask_confidence_initial_w_mean'\
                          or self.depth_type == 'mask_confidence_initial_w_mean_full_depthmap'\
                          or self.depth_type == 'mask_confidence_initial_w_median':
                        if pred_masks is not None:
                            curr_pred_mask = pred_masks_onehot[bbox_index_global:bbox_index_global+1]
                            curr_pred_mask_confidence = pred_masks_confidence[bbox_index_global:bbox_index_global+1]
                        else:
                            curr_pred_mask = torch.ones([1, 14, 14]).to(pred_depthmap.device)
                            curr_pred_mask_confidence = torch.ones([1, 14, 14]).to(pred_depthmap.device)
                    
                        # curr_pred_mask = pred_masks_onehot[bbox_index_global:bbox_index_global+1]
                        # curr_pred_mask_confidence = pred_masks_confidence[bbox_index_global:bbox_index_global+1]
                        bbox_index_global = bbox_index_global + 1

                    x1, y1, x2, y2 = c_bbox[:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # x1, y1, x2, y2 = c_bbox
                    # x1, y1, x2, y2 = x1.long().cpu().numpy(), y1.long().cpu().numpy(), x2.long().cpu().numpy(), y2.long().cpu().numpy()
                    # cropped_pred_depthmap = pred_depthmap_copy[:, :, y1:y2, x1:x2]
                    
                    # 超出图像外是负数
                    w = np.maximum(x2 - x1 + 1, 2)
                    h = np.maximum(y2 - y1 + 1, 2)
                    b, c, h_img, w_img = pred_depthmap_copy.shape
                    start_x1, start_y1 = 0, 0
                    if x1 < 0:
                        start_x1 = -1 * x1
                    if y1 < 0:
                        start_y1 = -1 * y1
                    # left or up, out of image
                    if (x1 < 0 or y1 < 0) and (start_y1 + h) < h_img and (start_x1 + w) < w_img \
                        and start_y1 < h and  start_x1 < w:
                        cropped_pred_depthmap = torch.zeros((b, c, h, w)).to(pred_depthmap_copy.device)
                        cropped_pred_depthmap[:, :, start_y1:, start_x1:] = pred_depthmap_copy[:, :, 0:h-start_y1, 0:w-start_x1] # w = np.maximum(x2 - x1 + 1, 2), if x1<0, means w has been add delta_x
                    else:
                        cropped_pred_depthmap = pred_depthmap_copy[:, :, y1:y1 + h, x1:x1 + w]

                    b, c, h, w = cropped_pred_depthmap.shape
                    max_hw = max(h, w)
                    square_cropped_pred_depthmap = torch.zeros((b, c, max_hw, max_hw)).to(cropped_pred_depthmap.device)
                    pad_h, pad_w = 0, 0
                    if max_hw > h:
                        pad_h = (max_hw - h) * 0.5
                        pad_h = int(pad_h)
                    if max_hw > w:
                        pad_w = (max_hw - w) * 0.5
                        pad_w = int(pad_w)
                    square_cropped_pred_depthmap[:, :, pad_h:cropped_pred_depthmap.shape[-2]+pad_h, pad_w:cropped_pred_depthmap.shape[-1]+pad_w] = cropped_pred_depthmap # align center
                    cropped_pred_depthmap = square_cropped_pred_depthmap
                   
                    pred_z_initial = torch.tensor(0).float().to(cropped_pred_depthmap.device)
                    if max_hw > 0 and (self.depth_type == 'mask_confidence' or self.depth_type == 'mask_confidence_initial_w_mean'\
                                        or self.depth_type == 'mask_confidence_initial_w_mean_full_depthmap' \
                                        or self.depth_type == 'mask_confidence_initial_w_median'):
                        if self.depth_type == 'mask_confidence_initial_w_mean':
                            # pred_z_initial = cropped_pred_depthmap.mean()
                            pred_z_initial = cropped_pred_depthmap[torch.where(cropped_pred_depthmap > 1.0)].mean() # filter 0
                        elif self.depth_type == 'mask_confidence_initial_w_mean_full_depthmap':
                            pred_z_initial = cropped_pred_depthmap.mean()
                        elif self.depth_type == 'mask_confidence_initial_w_median':
                            if torch.sum(cropped_pred_depthmap) > 0:
                                pred_z_initial = cropped_pred_depthmap[torch.where(cropped_pred_depthmap > 1.0)].median() # filter 0

                        if torch.isnan(pred_z_initial):
                            pred_z_initial = torch.tensor(0).float().to(cropped_pred_depthmap.device)

                        curr_pred_mask = curr_pred_mask.unsqueeze(0)  # torch.Size([2, 56, 56]), 最后一层是foreground, need to visual
                        curr_pred_mask = torch.nn.functional.interpolate(curr_pred_mask, size=(max_hw, max_hw), mode='bilinear', align_corners=False)
                        curr_pred_mask[torch.where(curr_pred_mask > 0.0)] = 1.0

                        # ########################################
                        # ### visual pred_mask n depthmap_mask ###
                        # count = count + 1
                        # print("curr_pred_mask.shape: ", curr_pred_mask.shape)
                        # vis = curr_pred_mask[0, 0].cpu().detach().numpy()   # B, 1, w, h
                        # output_path = f'./debug/vis_curr_pred_mask_{count}.png'
                        # plt.imsave(output_path, vis, cmap='gray')
                        # print(output_path)
                        # print("curr_pred_mask ...")

                        # vis_cropped_pred_depthmap = cropped_pred_depthmap[0, 0].cpu().detach().numpy()   # B, 1, w, h
                        # output_path = f'./debug/vis_cropped_pred_depthmap_{count}.png'
                        # plt.imsave(output_path, vis_cropped_pred_depthmap, cmap='gray')
                        # print(output_path)
                        # print("cropped_pred_depthmap ...")
                        # ###      end visual attention mask   ###
                        # ########################################
                        
                        # print("torch.sum(cropped_pred_depthmap): ", torch.sum(cropped_pred_depthmap)) # torch.sum(cropped_pred_depthmap):  tensor(21129.8203, device='cuda:0')
                        if not self.only_mask_confidence:
                            cropped_pred_depthmap = curr_pred_mask * cropped_pred_depthmap
                        # print("torch.sum(cropped_pred_depthmap): ", torch.sum(cropped_pred_depthmap)) # torch.sum(cropped_pred_depthmap):  tensor(0., device='cuda:0')

                        curr_pred_mask_confidence = curr_pred_mask_confidence.unsqueeze(0)
                        curr_pred_mask_confidence = torch.nn.functional.interpolate(curr_pred_mask_confidence, size=(max_hw, max_hw), mode='bilinear', align_corners=False)
                        if not self.only_mask_confidence:
                            curr_pred_mask_confidence = curr_pred_mask * curr_pred_mask_confidence
                    
                    if self.depth_type == 'mask_confidence' or self.depth_type == 'mask_confidence_initial_w_mean'\
                          or self.depth_type == 'mask_confidence_initial_w_mean_full_depthmap'\
                          or self.depth_type == 'mask_confidence_initial_w_median':
                        
                        # ### visual pred_mask n depthmap_mask
                        # print("curr_pred_mask_confidence.shape: ", curr_pred_mask_confidence.shape)
                        # vis = curr_pred_mask_confidence[0, 0].cpu().detach().numpy()   # B, 1, w, h
                        # output_path = './debug/vis_curr_pred_mask_confidence.png'
                        # plt.imsave(output_path, vis, cmap='gray')
                        # print(output_path)
                        # print("...")

                        # vis_cropped_pred_depthmap = cropped_pred_depthmap[0, 0].cpu().detach().numpy()   # B, 1, w, h
                        # output_path = './debug/vis_cropped_pred_depthmap.png'
                        # plt.imsave(output_path, vis_cropped_pred_depthmap, cmap='gray')
                        # print(output_path)
                        # print("...")

                        # vis_cropped_pred_depthmap_and_pred_mask_conf = cropped_pred_depthmap * curr_pred_mask_confidence
                        # vis_cropped_pred_depthmap_and_pred_mask_conf = vis_cropped_pred_depthmap_and_pred_mask_conf[0, 0].cpu().detach().numpy()   # B, 1, w, h
                        # output_path = './debug/vis_cropped_pred_depthmap_and_pred_mask_conf.png'
                        # plt.imsave(output_path, vis_cropped_pred_depthmap_and_pred_mask_conf, cmap='gray')
                        # print(output_path)
                        # print("...")
                        # ### end visual attention mask ###

                        if cropped_pred_depthmap.numel() == 0 or curr_pred_mask_confidence.numel() == 0: # check empty
                            pred_z = pred_z_initial
                        else:
                            # print("torch.sum(cropped_pred_depthmap): ", torch.sum(cropped_pred_depthmap))
                            # print("torch.sum(curr_pred_mask_confidence): ", torch.sum(curr_pred_mask_confidence))
                            # pred_z = (cropped_pred_depthmap * curr_pred_mask_confidence).sum() / curr_pred_mask_confidence.sum()
                            # print("before valid depthmap_confidence pred_z: ", pred_z)
                            if not self.depth_type == 'mask_confidence_initial_w_mean_full_depthmap':
                                cropped_pred_depthmap_mask = torch.zeros_like(cropped_pred_depthmap).to(cropped_pred_depthmap.device)
                                if self.depth_type == 'mask_confidence_initial_w_mean':
                                    cropped_pred_depthmap_mask[torch.where(cropped_pred_depthmap > 0.0)] = 1
                                else:
                                    cropped_pred_depthmap_mask[torch.where(cropped_pred_depthmap > 1.0)] = 1
                                curr_pred_mask_confidence = cropped_pred_depthmap_mask * curr_pred_mask_confidence

                            # pred_z = (cropped_pred_depthmap * curr_pred_mask_confidence).sum() / curr_pred_mask_confidence.sum()
                            curr_pred_mask_confidence_sum = curr_pred_mask_confidence.sum()
                            if curr_pred_mask_confidence_sum > 0.0:
                                pred_z = (cropped_pred_depthmap * curr_pred_mask_confidence).sum() / curr_pred_mask_confidence_sum
                            else:
                                pred_z = pred_z_initial
                        if torch.isnan(pred_z) or (pred_z - 0.0) < sys.float_info.epsilon:
                            pred_z = pred_z_initial

                    elif self.depth_type == 'median':
                        cropped_pred_depthmap = cropped_pred_depthmap[torch.where(cropped_pred_depthmap > 1.0)] # filter 0
                        if cropped_pred_depthmap.numel() == 0: # check empty
                            pred_z = torch.tensor(0).float().to(cropped_pred_depthmap.device)
                        else:
                            pred_z = cropped_pred_depthmap.median()
                            
                    elif self.depth_type == 'mean':
                        cropped_pred_depthmap = cropped_pred_depthmap[torch.where(cropped_pred_depthmap > 1.0)] # filter 0
                        if cropped_pred_depthmap.numel() == 0: # check empty
                            pred_z = torch.tensor(0).float().to(cropped_pred_depthmap.device)
                        else:
                            pred_z = cropped_pred_depthmap.mean()

                    pred_z = pred_z.unsqueeze(0)
                    pred_z_tensor_list.append(pred_z)

            # ## 框内人物深度的均值(或中值)作为depth
            # pred_depthmap_mask = torch.zeros_like(pred_depthmap).to(pred_depthmap.device)
            # pred_depthmap_mask[torch.where(pred_depthmap > 1.0)] = 1
            # pred_depthmap_copy = pred_depthmap.clone() * pred_depthmap_mask
            # # pred_z_tensor_list = list()
            # for img_id in range(x[0].shape[0]): # image batch_size = 1 for testing and eval
            #     curr_pos_bboxes = pred_bboxes[img_id]
            #     for c_bbox in curr_pos_bboxes:
            #         x1, y1, x2, y2 = c_bbox[:4]
            #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #         cropped_pred_depthmap = pred_depthmap_copy[:, :, y1:y2, x1:x2]
            #         cropped_pred_depthmap = cropped_pred_depthmap[torch.where(cropped_pred_depthmap > 1.0)] # filter 0
            #         if cropped_pred_depthmap.numel() == 0: # check empty
            #             pred_z = torch.tensor(0).float().to(cropped_pred_depthmap.device)
            #             # 不存在depthmap的时候默认应该用弱透视方法？
            #             # pred_z = 2 * self.FOCAL_LENGTH / (bboxes_size * pred_camera[..., 0] + 1e-9)
            #         else:
            #             if self.depth_type == 'median':
            #                 pred_z = cropped_pred_depthmap.median()
            #             elif self.depth_type == 'mean':
            #                 pred_z = cropped_pred_depthmap.mean()

            #         pred_z = pred_z.unsqueeze(0)
            #         pred_z_tensor_list.append(pred_z)

        
        if len(pred_z_tensor_list) > 0:
            pred_z_tensor = torch.cat(pred_z_tensor_list, dim=0)
        else:
            pred_z_tensor = x[0].new_zeros(len(pred_bboxes))

        outputs = {
                'pred_depthmap': pred_depthmap.float(),
                'pred_z_tensor': pred_z_tensor.float(),
                'bin_edges': bin_edges
            }

        return outputs

    def map_roi_levels(self, rois, num_levels, finest_scale=56):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale: level 0
        - finest_scale <= scale < finest_scale * 2: level 1
        - finest_scale * 2 <= scale < finest_scale * 4: level 2
        - scale >= finest_scale * 4: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls
    
    def get_pred_bboxes(self, sampling_results):
            pos_proposals = [res.pos_bboxes for res in sampling_results]
            pred_bbox_list = list()
            for i in range(len(pos_proposals)):
                num_pos = pos_proposals[i].size(0)
                for j in range(num_pos):
                    pred_bbox_list.append(pos_proposals[i][j:j+1])

            pred_bboxes = torch.cat(pred_bbox_list, dim=0)
            return pred_bboxes


    def get_target(self, sampling_results, gt_translation, has_trans):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        
        pred_bbox_list, gt_translation_list, has_trans_list = list(), list(), list()
        for i in range(len(pos_proposals)):
            pos_assigned_gt_inds_per_img = pos_assigned_gt_inds[i].cpu().numpy()
            num_pos = pos_proposals[i].size(0)
            for j in range(num_pos):
                pred_bbox_list.append(pos_proposals[i][j:j+1])
                idx = pos_assigned_gt_inds_per_img[j]
                gt_translation_list.append(gt_translation[i][idx:idx+1])
                has_trans_list.append(has_trans[i][idx:idx+1])
        
        pred_bboxes = torch.cat(pred_bbox_list, dim=0)
        translation_target = torch.cat(gt_translation_list, dim=0)
        has_trans_target = torch.cat(has_trans_list, dim=0)

        return pred_bboxes, translation_target, has_trans_target


    def init_weights(self):
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.zeros_(self.fc1.bias)
        # nn.init.xavier_uniform_(self.dec.weight, gain=0.1)
        # nn.init.zeros_(self.dec.bias)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.zeros_(self.fc2.bias)
        # for conv in self.convs:
        #     for sub_layer in conv:
        #         if isinstance(sub_layer, nn.Conv2d):
        #             kaiming_init(sub_layer)
        #         elif isinstance(sub_layer, nn.BatchNorm2d):
        #             nn.init.constant_(sub_layer.weight, 1)
        #             nn.init.constant_(sub_layer.bias, 0)
        #         else:
        #             raise RuntimeError("Unknown network component in SMPL Head")

        print("trans_pyramid submodel init weights todo...")


if __name__ == '__main__':
    model = UnetAdaptiveBins.build(100)
    x = torch.rand(2, 3, 480, 640)
    bins, pred = model(x)
    print(bins.shape, pred.shape)
