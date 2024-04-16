
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import nn

import argparse
import os
import os.path as osp
import sys
import cv2
import numpy as np

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_PATH)

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn


from mmcv.runner import Runner

from mmcv.parallel import DataContainer as DC
from mmcv.parallel import MMDataParallel
from mmdet.apis.train import build_optimizer
from mmdet.models.utils.smpl.renderer import Renderer
from mmdet import __version__
from mmdet.models import build_detector
from mmdet.datasets.transforms import ImageTransform
from mmdet.datasets.utils import to_tensor
from time import time

from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)
from tqdm import tqdm
import warnings
import mmcv

import copy
import time

denormalize = lambda x: x.transpose([1, 2, 0]) * np.array([0.229, 0.224, 0.225])[None, None, :] + \
                        np.array([0.485, 0.456, 0.406])[None, None,]

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

def scale2depth(scale, f, c, x, o_depth):
    assert scale > 0
    scale = 1/scale
    offset = f * x / o_depth + c
    #print(scale, x, f, c, offset)
    z =1 /((offset - scale * c) / (scale * x * f))
    return z

# 1 0 0
# 0 1 0
# 0 
# bbox_thres = 0.05
bbox_thres = 0.0005
def renderer_bv(img_t, verts_t, trans_t, bboxes_t, focal_length, render, colors=None):
    # rotation
    R_bv = torch.zeros(3, 3)
    R_bv[0, 0] = R_bv[2, 1] = 1
    R_bv[1, 2] = -1

    # filter our small detections
    bbox_area = (bboxes_t[:, 2] - bboxes_t[:, 0]) * (bboxes_t[:, 3] - bboxes_t[:, 1])
    area_mask = torch.tensor(bbox_area > bbox_area.max() * bbox_thres)
    verts_t, trans_t = verts_t[area_mask], trans_t[area_mask]

    # camera space
    verts_t = verts_t + trans_t.unsqueeze(1)

    # rotate to up view
    verts_tr = torch.einsum('bij,kj->bik', verts_t, R_bv)
    verts_tfar = verts_tr  # verts_tr + trans_t.unsqueeze(1)
    
    # min point, max point position
    p_min, p_max = verts_tfar.view(-1, 3).min(0)[0], verts_tfar.view(-1, 3).max(0)[0]
    
    # center of min/max
    p_center = 0.5 * (p_min + p_max)
    # trans_tr = torch.einsum('bj,kj->bk', trans_t, R_bv)

    # normalize points to center
    verts_center = (verts_tfar.view(-1, 3) - p_center).view(verts_t.shape[0], -1, 3)

    # normalized min/max
    dis_min, dis_max = (verts_tfar.view(-1, 3) - p_center).min(0)[0], (
            verts_tfar.view(-1, 3) - p_center).max(0)[0]
    
    h, w = img_t[0].shape[-2:]
    # h, w = min(h, w), min(h, w)
    ratio_max = abs(0.9 - 0.5)
    z_x = dis_max[0] * focal_length / (ratio_max * w) + torch.abs(dis_min[2])
    z_y = dis_max[1] * focal_length / (ratio_max * h) + torch.abs(dis_min[2])
    z_x_0 = (-dis_min[0]) * focal_length / (ratio_max * w) + torch.abs(
        dis_min[2])
    z_y_0 = (-dis_min[1]) * focal_length / (ratio_max * h) + torch.abs(
        dis_min[2])
    z = max(z_x, z_y, z_x_0, z_y_0)
    
    # z controls the view
    verts_right = verts_tfar - p_center + torch.tensor([0, 0, z])
    img_right = render(img_t, [verts_right], translation=[torch.zeros_like(trans_t)], colors=colors)
    return img_right[0]


def renderer_sv(img_t, verts_t, trans_t, bboxes_t, focal_length, render, fixed_range=None, colors=None):
    # rotation
    R_bv = torch.zeros(3, 3)
    # R_bv[0, 0] = R_bv[2, 1] = 1
    # R_bv[1, 2] = -1
    
    R_bv[0, 2] = -1
    R_bv[1, 1] = 1
    R_bv[2, 0] = 1

    #print(R_bv.T)

    # filter our small detections
    bbox_area = (bboxes_t[:, 2] - bboxes_t[:, 0]) * (bboxes_t[:, 3] - bboxes_t[:, 1])
    area_mask = torch.tensor(bbox_area > bbox_area.max() * bbox_thres)
    verts_t, trans_t = verts_t[area_mask], trans_t[area_mask]

    # camera space
    verts_t = verts_t + trans_t.unsqueeze(1)

    # rotate to up view
    verts_tr = torch.einsum('bij,kj->bik', verts_t, R_bv)
    verts_tfar = verts_tr  # verts_tr + trans_t.unsqueeze(1)
    
    # min point, max point position 
    if fixed_range is None:
        p_min, p_max = verts_tfar.view(-1, 3).min(0)[0], verts_tfar.view(-1, 3).max(0)[0]
    else:
        p_min, p_max = verts_tfar.view(-1, 3).min(0)[0], verts_tfar.view(-1, 3).max(0)[0]
        p_min[-1] = -1.
        p_max[-1] = fixed_range
    # center of min/max
    p_center = 0.5 * (p_min + p_max)
    # trans_tr = torch.einsum('bj,kj->bk', trans_t, R_bv)

    # normalize points to center # this
    verts_center = (verts_tfar.view(-1, 3) - p_center).view(verts_t.shape[0], -1, 3)

    # normalized min/max
    dis_min, dis_max = (verts_tfar.view(-1, 3) - p_center).min(0)[0], (
            verts_tfar.view(-1, 3) - p_center).max(0)[0]
    
    h, w = img_t[0].shape[-2:]
    # h, w = min(h, w), min(h, w)
    ratio_max = abs(0.9 - 0.5)
    z_x = dis_max[0] * focal_length / (ratio_max * w) + torch.abs(dis_min[2])
    z_y = dis_max[1] * focal_length / (ratio_max * h) + torch.abs(dis_min[2])
    z_x_0 = (-dis_min[0]) * focal_length / (ratio_max * w) + torch.abs(
        dis_min[2])
    z_y_0 = (-dis_min[1]) * focal_length / (ratio_max * h) + torch.abs(
        dis_min[2])
    z = max(z_x, z_y, z_x_0, z_y_0)
    
    # z controls the view
    verts_right = verts_tfar - p_center + torch.tensor([0, 0, z])
    
    img_right = render(img_t, [verts_right], translation=[torch.zeros_like(trans_t)], colors=colors)
    return img_right[0]


def prepare_dump(pred_results, img, render, bbox_results, FOCAL_LENGTH, mask_results=None):
    # 'pred_rotmat', 'pred_betas', 'pred_camera', 'pred_vertices', 'pred_joints', 'pred_translation', 'bboxes'
    
    pred_trans = pred_results['pred_translation'].cpu()
    # pred_camera = pred_results['pred_camera'].cpu()
    # pred_betas = pred_results['pred_betas'].cpu()
    # pred_rotmat = pred_results['pred_rotmat'].cpu()
    pred_verts = pred_results['pred_vertices'].cpu()

    bboxes = pred_results['bboxes']
    img_bbox = img.copy()
    # for bbox in bboxes:
    #     img_bbox = cv2.rectangle(img_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (1., 0, 0), 2)
    img_th = torch.tensor(img_bbox.transpose([2, 0, 1]))
    _, H, W = img_th.shape

    front_view_color = np.array([255, 244, 229]) / 255.0
    side_view_color = np.array([255, 245, 182]) / 255.0
    top_view_color = np.array([212, 235, 255]) / 255.0

    fv_bg_color = torch.ones_like(img_th)
    fv_bg_color[0] = fv_bg_color[0] * front_view_color[0]
    fv_bg_color[1] = fv_bg_color[1] * front_view_color[1]
    fv_bg_color[2] = fv_bg_color[2] * front_view_color[2]
    sv_bg_color = torch.ones_like(img_th)
    sv_bg_color[0] = sv_bg_color[0] * side_view_color[0]
    sv_bg_color[1] = sv_bg_color[1] * side_view_color[1]
    sv_bg_color[2] = sv_bg_color[2] * side_view_color[2]
    bv_bg_color = torch.ones_like(img_th)
    bv_bg_color[0] = bv_bg_color[0] * top_view_color[0]
    bv_bg_color[1] = bv_bg_color[1] * top_view_color[1]
    bv_bg_color[2] = bv_bg_color[2] * top_view_color[2]

    try:

        fv_rendered = render([img_th], [pred_verts], translation=[pred_trans])[0]
        # fv_rendered = render([fv_bg_color], [pred_verts], translation=[pred_trans])[0]
        sv_rendered = renderer_sv([sv_bg_color], pred_verts, pred_trans, bbox_results[0], FOCAL_LENGTH, render)
        bv_rendered = renderer_bv([bv_bg_color], pred_verts, pred_trans, bbox_results[0], FOCAL_LENGTH, render)
    except Exception as e:
        print(e)
        return None

    total_img = np.zeros((H, 4*W, 3))
    total_img[:H, :W] += img
    total_img[:H, W:2*W] += fv_rendered.transpose([1, 2, 0])
    total_img[:H, 2*W:3*W] += bv_rendered.transpose([1, 2, 0])
    total_img[:H, 3*W:4*W] += sv_rendered.transpose([1, 2, 0])
    total_img = (total_img * 255).astype(np.uint8)

    return total_img


def is_image_filename(filename):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 添加其他图片格式的扩展名
    lowercase_filename = filename.lower()
    for ext in image_extensions:
        if lowercase_filename.endswith(ext):
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description='')

    # parser.add_argument('--config', default="configs/smpl/ADRT_GlobalTransRoI_localFeats320_dim128_multiLayersTrans_28E.py", help='test config file path')
    parser.add_argument('--config', default="configs/smpl/ADRT_GlobalTransRoI_localFeats320_dim128_multiLayersTrans_28E_demo.py", help='test config file path')
    parser.add_argument('--checkpoint', default="work_dirs/ADRT_GlobalTransRoI_localFeats320_dim128_multiLayersTrans_28E/epoch_15.pth", help='checkpoint file')
   
    # parser.add_argument('--config', default="configs/smpl/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_GlobalTransRoI_localFeats512_dim128_bs8x8_28D.py", help='test config file path')
    # parser.add_argument('--checkpoint', default="work_dirs/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_GlobalTransRoI_localFeats512_dim128_bs8x8_28D/epoch_16.pth", help='checkpoint file')
   
    # parser.add_argument('--config', default="configs/smpl/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_bs8x8_28E.py", help='test config file path')
    # parser.add_argument('--checkpoint', default="work_dirs/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_bs8x8_28E/epoch_19.pth", help='checkpoint file')

    # parser.add_argument('--config', default="configs/smpl/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_bs8x8_28E.py", help='test config file path')
    # parser.add_argument('--checkpoint', default="work_dirs/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_bs8x8_28E/epoch_19.pth", help='checkpoint file')
    
    # parser.add_argument('--image_fold', default="/data/datasets/3DPW_official/imageFiles/downtown_sitOnStairs_00/", help='Path to folder with images')
    # parser.add_argument('--output_fold', default="./output/fig1_front_top_side", help='Path to save results')

    # parser.add_argument('--image_fold', default="/home/haoye/codes/ads/input/test", help='Path to folder with images')
    # parser.add_argument('--output_fold', default="./output/test", help='Path to save results')


    # # front view ##
    # parser.add_argument('--image_fold', default="./figs/fig5_frontview/src_imgs", help='Path to folder with images')
    # parser.add_argument('--output_fold', default="./figs/fig5_frontview/ours", help='Path to save results')

    ## top view ##
    # parser.add_argument('--image_fold', default="./figs/fig6_topview/src_imgs", help='Path to folder with images')
    # parser.add_argument('--output_fold', default="./figs/fig6_topview/ours", help='Path to save results')

    # ## failure cases
    # parser.add_argument('--image_fold', default="./figs/supp_failure/src_imgs", help='Path to folder with images')
    # parser.add_argument('--output_fold', default="./figs/supp_failure/failure_cases_result", help='Path to save results')

    # ## supp fig1 fig2
    parser.add_argument('--image_fold', default="./figs/supp_fig1_fig2_in_the_wild/src_imgs", help='Path to folder with images')
    parser.add_argument('--output_fold', default="./figs/supp_fig1_fig2_in_the_wild/in_the_wild_result", help='Path to save results')

    # ## wifi human
    # parser.add_argument('--image_fold', default="./figs/wifi_human/src_imgs", help='Path to folder with images')
    # parser.add_argument('--output_fold', default="./figs/wifi_human/wifi_human_result", help='Path to save results')


    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    
    # parser.add_argument(
    #     '--work-dir',
    #     help='the directory to save the file containing evaluation metrics')
    # parser.add_argument('--out', default="test_result_debug.pkl", help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # cfg.model.test_cfg.rcnn.score_thr = 0.1
    # cfg.model.test_cfg.rcnn.score_thr = 0.2
    # cfg.model.test_cfg.rcnn.score_thr = 0.3
    # # cfg.model.test_cfg.rcnn.score_thr = 0.4
    cfg.model.test_cfg.rcnn.score_thr = 0.5
    # cfg.model.test_cfg.rcnn.score_thr = 0.6

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.generator.backbone:
        cfg.model.generator.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.generator.neck, list):
            for neck_cfg in cfg.model.generator.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.generator.neck.get('rfp_backbone'):
            if cfg.model.generator.neck.rfp_backbone.get('pretrained'):
                cfg.model.generator.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    # elif isinstance(cfg.data.test, list):
    #     for ds_cfg in cfg.data.test:
    #         ds_cfg.test_mode = True
    #     if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
    #         for ds_cfg in cfg.data.test:
    #             ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }


    # ##---
    # rank, _ = get_dist_info()
    # # allows not to create
    # if args.work_dir is not None and rank == 0:
    #     mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    #     timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    #     json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # # build the dataloader
    # print(">>>>args.eval_data: ", args.eval_data)
    # if args.eval_data == "agora":
    #     dataset = build_dataset(cfg.data.test_agora)
    # else:
    #     dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(dataset, **test_loader_cfg)



    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # init rfnext if 'RFSearchHook' is defined in cfg
    rfnext_init_model(model, cfg=cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        # model.CLASSES = dataset.CLASSES
        model.CLASSES = ('Background', 'Human')

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
       

    FOCAL_LENGTH = cfg.get('FOCAL_LENGTH', 1000)
    render = Renderer(focal_length=FOCAL_LENGTH)
    # img_transform = ImageTransform(size_divisor=32, **img_norm_cfg)
    img_transform = ImageTransform(size_divisor=None, **img_norm_cfg)
    img_scale = cfg.common_val_cfg.img_scale

    folder_name = args.image_fold
    output_fold = args.output_fold
    os.makedirs(output_fold, exist_ok=True)

    images = os.listdir(folder_name)
    images = sorted(images)
    # images = images[700:]

    # for image in images[:30]:
    for image in images:
        
        file_name = osp.join(folder_name, image)

        if not is_image_filename(file_name):
            continue
        # print('processing:', file_name)
        # if 'wifi' not in file_name:
        #     continue
        
        file_name = "/home/delight/codes/adrt/figs/wifi_human/src_imgs/021520.png"
        img = cv2.imread(file_name)
        if img is None:
            continue

        ori_shape = img.shape

        start = time.time()
        img, img_shape, pad_shape, scale_factor = img_transform(img, img_scale)

        _, h, w = img.shape
        # h, w, _ = pad_shape
        pad_w, pad_h = (img_scale[0] - w) * 0.5, (img_scale[1] - h) * 0.5

        # Force padding for the issue of multi-GPU training
        padded_img = np.zeros((img.shape[0], img_scale[1], img_scale[0]), dtype=img.dtype)
        # padded_img[:, :img.shape[-2], :img.shape[-1]] = img
        padded_img[:, int(pad_h):img.shape[-2]+int(pad_h), int(pad_w):img.shape[-1]+int(pad_w)] = img
        img = padded_img

        # print("img_shape: ", img_shape)
        # print("img.shape: ", img.shape)

        data_batch = dict(
            img=DC([to_tensor(img[None, ...])], stack=True),
            # img_metas=DC([{'img_shape':img_shape, 'scale_factor':scale_factor, 'flip':False, 'ori_shape':ori_shape}], cpu_only=True),
            img_metas=[DC([{'img_shape':img.transpose([1, 2, 0]).shape, 'scale_factor':scale_factor, 'flip':False, 'ori_shape':ori_shape}], cpu_only=True)],
            )

        model.eval()
        
        with torch.no_grad():
            start_time = time.time()
            smpl_loss_cfg = cfg.model.generator.roi_head.smpl_head.loss_cfg
            # bbox_results, mask_results, smpl_results = model(return_loss=False,
            #                                                     rescale=False, 
            #                                                     **data_batch,
            #                                                     is_woLocalTrans=smpl_loss_cfg['is_woLocalTrans'],
            #                                                     is_woGlobalTransPart=smpl_loss_cfg['is_woGlobalTransPart']
            #                                                     )
            
            det_bboxes, det_labels = model(return_loss=False,
                                                            rescale=False, 
                                                            **data_batch,
                                                            is_woLocalTrans=smpl_loss_cfg['is_woLocalTrans'],
                                                            is_woGlobalTransPart=smpl_loss_cfg['is_woGlobalTransPart']
                                                            )
            
            # print(det_bboxes.shape, det_labels.shape)
            img = denormalize(img)
            img_bbox = img.copy()
            for bbox in det_bboxes[0]:
                img_bbox = cv2.rectangle(img_bbox, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (1., 0, 0), 2)
            img_th = img_bbox * 255
            _, H, W = img_th.shape
            # out_path = './output/test_bbox_woRoIPadCenter.jpg'
            out_path = './output/test_bbox_wRoIPadCenter.jpg'
            cv2.imwrite(out_path, img_th[:, :, ::-1])
            print(out_path)
            print("=" * 50)
            print("cost time: ", time.time() - start_time)






            # if smpl_results is not None:
            #     smpl_results['bboxes'] = bbox_results[0]
            #     img = denormalize(img)
            #     img_render = prepare_dump(smpl_results, img, render, bbox_results, FOCAL_LENGTH, mask_results=mask_results)
            #     out_path = f'{file_name.replace(folder_name, output_fold)}.output.png'
            #     cv2.imwrite(out_path, img_render[:, :, ::-1])
            #     print(out_path)
            #     print("=" * 50)


            #     ##################################
            #     ### start pred_depthmap visual ###
            #     ##################################
            #     pred_depthmap = smpl_results['pred_depthmap']
            #     # output_path = "./debug/pred_depthmap.png"
            #     pred_depthmap_np = pred_depthmap.detach().cpu().numpy()[0][0] # 1x1xhxw --> hxw
            #     depth_min = pred_depthmap_np.min()
            #     depth_max = pred_depthmap_np.max()
            #     print("depth_max: ", depth_max)
            #     print("depth_min: ", depth_min)
            #     # if depth_max - depth_min > np.finfo("float").eps:
            #     #     out = 255 * (pred_depthmap_np - depth_min) / (depth_max - depth_min)
            #     # else:
            #     #     out = np.zeros(pred_depthmap_np.shape, dtype=pred_depthmap_np.dtype)
            #     # cv2.imwrite(output_path, out.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
            #     # print("pred_depthmap.shape: ", pred_depthmap.shape)
            #     # print(output_path)

            #     print("pred_depthmap_np.shape: ", pred_depthmap_np.shape)
            #     depth_map_normalized = (pred_depthmap_np - depth_min) / (depth_max - depth_min)
            #     depth_map_colored = cv2.applyColorMap((depth_map_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
            #     depth_map_colored = depth_map_colored[int(pad_h):512-int(pad_h), int(pad_w):832-int(pad_w), :]
            #     c_h, c_w, _ = depth_map_colored.shape

            #     color_bar = np.zeros((256, 50, 3), dtype=np.uint8)
            #     for i in range(256):
            #         color_bar[i, :, :] = cv2.applyColorMap(np.array([[i]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0, :]
            #     color_bar = cv2.resize(color_bar, (50, c_h))
        
            #     mid_bar = np.ones((c_h, 10, 3), dtype=np.uint8) * 255
            #     end_bar = np.ones((c_h, 63, 3), dtype=np.uint8) * 255
            #     depth_map_colored = cv2.hconcat([depth_map_colored, mid_bar, color_bar, end_bar])

            #     # Generate scale values corresponding to the color bar
            #     scale_values = np.linspace(0, depth_max, num=10)

            #     # Calculate the pixel positions for the scale values
            #     scale_pixel_positions = np.linspace(0, c_h-40, num=10, dtype=int)

            #     # Add scale values to the right side of the concatenated image
            #     for i, value in enumerate(scale_values):
            #         cv2.putText(depth_map_colored, f'{value:.1f}', (c_w + 10 + 50 + 8, scale_pixel_positions[i] + 20),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            #     img_vis = copy.copy(img)
            #     # img_vis = img_vis.transpose([1, 2, 0])
            #     # img_vis = denormalize(img_vis)
            #     img_vis = (img_vis * 255).astype(np.uint8)

            #     # img_vis = img_vis[int(pad_h):512-int(pad_h), int(pad_w):832-int(pad_w), :]
            #     # img_depth_map_colored = cv2.hconcat([img_vis[:, :, ::-1], depth_map_colored])

            #     # img_viz = img_viz[int(pad_h):512-int(pad_h), :, :]
            #     # img_depth_map_colored = cv2.hconcat([img_render[:, :, ::-1], depth_map_colored])

            #     img_render = img_render[:, :, ::-1]
            #     img_src = img_render[int(pad_h):512-int(pad_h), int(pad_w):832-int(pad_w), :]
            #     img_front = img_render[int(pad_h):512-int(pad_h), 832 + int(pad_w) : 832*2 - int(pad_w), :]
            #     img_top = img_render[int(pad_h):512-int(pad_h), 832*2 + int(pad_w) : 832*3 - int(pad_w), :]
            #     img_side = img_render[int(pad_h):512-int(pad_h), 832*3 + int(pad_w) : 832*4 - int(pad_w), :]

            #     img_mesh_depthmap_output = cv2.hconcat([img_src, img_front, img_top, img_side, depth_map_colored])

            #     out_depthmap_path = f'{file_name.replace(folder_name, output_fold)}_depthmap_color.png'
            #     cv2.imwrite(out_depthmap_path, img_mesh_depthmap_output.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
            #     print(out_depthmap_path)
            #     print(out_depthmap_path)

            #     ##################################
            #     ### end pred_depthmap visual ###
            #     ##################################



if __name__ == '__main__':
    main()


