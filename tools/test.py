# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
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

from mmdet.core.utils.eval_utils import H36MEvalHandler, EvalHandler, PanopticEvalHandler, \
    MuPoTSEvalHandler, ThreeDPWHandler, OriginH36MEvalHandler, ROMPThreeDPWHandler,\
          AGORAHandler, AGORAHandlerV2, ThreeDPWHandler_ROMP, ThreeDPWHandlerCenterPoint, AGORAHandlerCenterPoint

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    
    # parser.add_argument('--config', default="configs/smpl/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_28D.py", help='test config file path')
    # parser.add_argument('--checkpoint', default="work_dirs/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_28D/epoch_2.pth", help='checkpoint file')
    
    # parser.add_argument('--config', default="configs/smpl/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_setGlobalFeatsZero_28E.py", help='test config file path')
    # parser.add_argument('--checkpoint', default="work_dirs/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_setGlobalFeatsZero_28E/epoch_6.pth", help='checkpoint file')
    

    # parser.add_argument('--config', default="configs/smpl/pretrain_ADRT_ROIPadCenter_pyramidTrans_woScale_wGAN_256p_lr04_transFeats_clsNum2_28E.py", help='test config file path')
    # parser.add_argument('--checkpoint', default="work_dirs/pretrain_ADRT_ROIPadCenter_pyramidTrans_woScale_wGAN_256p_lr04_transFeats_clsNum2_28E/epoch_10.pth", help='checkpoint file')
    

    #### abaltion
    # parser.add_argument('--config', default="configs/smpl/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_woMaskAtt_bs8x8_28D.py", help='test config file path')
    # parser.add_argument('--checkpoint', default="work_dirs/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_woMaskAtt_bs8x8_28D/epoch_5.pth", help='checkpoint file')
    
    # parser.add_argument('--config', default="configs/smpl/pretrain_ADRT_ROIPadCenter_pyramidTrans_woScale_wGAN_256p_lr04_transFeats_clsNum2_28E.py", help='test config file path')
    # parser.add_argument('--checkpoint', default="work_dirs/pretrain_ADRT_ROIPadCenter_pyramidTrans_woScale_wGAN_256p_lr04_transFeats_clsNum2_28E/epoch_10.pth", help='checkpoint file')
    
    # parser.add_argument('--config', default="configs/smpl/pretrain_ADRT_ROIPadCenter_pyramidTrans_woScale_wGAN_256p_lr04_transFeats_clsNum2_maskNearest_28E.py", help='test config file path')
    # parser.add_argument('--checkpoint', default="work_dirs/pretrain_ADRT_ROIPadCenter_pyramidTrans_woScale_wGAN_256p_lr04_transFeats_clsNum2_maskNearest_28E/epoch_2.pth", help='checkpoint file')
    
    # parser.add_argument('--config', default="configs/smpl/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_bs8x8_28E.py", help='test config file path')
    # parser.add_argument('--checkpoint', default="work_dirs/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_bs8x8_28E/epoch_19.pth", help='checkpoint file')
    
    # parser.add_argument('--config', default="configs/smpl/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_GlobalTransRoI_bs8x8_28E.py", help='test config file path')
    # parser.add_argument('--checkpoint', default="work_dirs/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_GlobalTransRoI_bs8x8_28E/epoch_2.pth", help='checkpoint file')

    parser.add_argument('--config', default="configs/smpl/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_GlobalTransRoIDepthmap_bs8x8_28D.py", help='test config file path')
    parser.add_argument('--checkpoint', default="work_dirs/ADRT_PyramidTrans_woSigmoid_wGlobal_wScale_wGAN_stageIlr04_globalLoss1_woValidDepth_transFeats_GlobalTransRoIDepthmap_bs8x8_28D/epoch_2.pth", help='checkpoint file')
       
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', default="test_result_debug.pkl", help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
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
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
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
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument(
        '--eval-data',
        default='agora',
        choices=['threedpw', 'agora'],
        help='test set')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args




eval_handler_mapper = dict(
    threedpw=ThreeDPWHandler,
    # threedpw=ThreeDPWHandlerCenterPoint,
    
    # threedpw=ThreeDPWHandlerV2, # match
    # threedpw=ROMPThreeDPWHandler, # this is for romp only
    # threedpw=BEVThreeDPWHandler,

    # cropped_h36m=H36MEvalHandler,
    full_h36m=H36MEvalHandler, # use this
    # full_h36m=BEVH36MEvalHandler,#SPECH36MEvalHandler,
    # full_h36m=OriginH36MEvalHandler,
    panoptic=PanopticEvalHandler,
    ultimatum=PanopticEvalHandler,
    haggling=PanopticEvalHandler,
    pizza=PanopticEvalHandler,
    mafia=PanopticEvalHandler,
    mupots=MuPoTSEvalHandler,

    agora=AGORAHandler,
    # agora=AGORAHandlerCenterPoint, # for box-center-close eval
)

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # cfg.model.test_cfg.rcnn.score_thr = 0.1
    # cfg.model.test_cfg.rcnn.score_thr = 0.3
    # cfg.model.test_cfg.rcnn.score_thr = 0.5

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # cfg.model.test_cfg.rcnn.score_thr = 0.1
    # cfg.model.test_cfg.rcnn.score_thr = 0.3
    # cfg.model.test_cfg.rcnn.score_thr = 0.5

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

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    print(">>>>args.eval_data: ", args.eval_data)
    if args.eval_data == "agora":
        dataset = build_dataset(cfg.data.test_agora)
    else:
        dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

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
        model.CLASSES = dataset.CLASSES

    print("dataset.dataset_name:", dataset.dataset_name)
    eval_handler = eval_handler_mapper[dataset.dataset_name](writer=tqdm.write, 
                                                             viz_dir=None,
                                                             FOCAL_LENGTH=1000,
                                                             work_dir=None) 
    if not distributed:
        model = build_dp(model.generator, cfg.device, device_ids=cfg.gpu_ids)
        # outputs = single_gpu_test(model, data_loader, args.show,
        outputs = single_gpu_test(model, data_loader, args.show,
                                   args.show_dir, args.show_score_thr,
                                   eval_handler, cfg.model.generator.roi_head.smpl_head.loss_cfg)
    


if __name__ == '__main__':
    main()
