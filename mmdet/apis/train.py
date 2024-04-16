# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_runner, IterBasedRunner,
                         get_dist_info)

from mmdet.core import DistEvalHook, EvalHook, build_optimizer, build_optimizers
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                         find_latest_checkpoint, get_root_logger)

from mmdet.models.smpl_heads.smpl_common import Discriminator
from mmdet.apis.adv_runner import AdvRunner
from copy import deepcopy
from mmdet.models.utils.smpl_utils import batch_rodrigues
from mmdet.models.losses.smpl_loss import batch_adv_disc_l2_loss, batch_encoder_disc_l2_loss, \
    adversarial_loss
from collections import OrderedDict
import logging
import os.path as osp
from mmcv.utils import build_from_cfg


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.train_dataloader.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s) with {samples_per_gpu} '
                f'samples per GPU. The total batch size is {batch_size}.')

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    print(">>>>cfg.data.samples_per_gpu:", cfg.data.train_dataloader['samples_per_gpu'])
    print(">>>>cfg.data.workers_per_gpu:", cfg.data.train_dataloader['workers_per_gpu'])
    print("-" * 50)

    train_dataloader_default_args = dict(
        samples_per_gpu=int(cfg.data.train_dataloader['samples_per_gpu']),
        workers_per_gpu=int(cfg.data.train_dataloader['workers_per_gpu']),
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        print("!!!!!!!!!!>>>>> find_unused_parameters:", find_unused_parameters)
        # find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
   
    optimizer = build_optimizers(model, cfg.optimizer)
    print(optimizer.keys())

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    
    if cfg.get('optimizer_cfg', None) is None:
        optimizer_config = None
    elif fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    if distributed:
        ddp_reducer = model.reducer
    else:
        ddp_reducer = None
    runner.run(data_loaders, cfg.workflow, ddp_reducer=ddp_reducer)


def parse_adv_losses(losses, tag_tail=''):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if not ('loss' in loss_name):
            if isinstance(losses[loss_name], torch.Tensor):
                losses[loss_name] = loss_value.cpu()

    if 'img$idxs_in_batch' in losses:
        last_idx = -1
        split_idx = -1
        for i, idx in enumerate(losses['img$idxs_in_batch'].squeeze()):
            if last_idx > idx:
                split_idx = i
                break
            else:
                last_idx = idx
        split_idx = int(split_idx)
        if last_idx > 0:
            losses['img$raw_images'] = losses['img$raw_images'][:int(last_idx) + 1]
        if split_idx > 0:
            for loss_name, loss_value in losses.items():
                if loss_name.startswith('img$') and loss_name != 'img$raw_images':
                    losses[loss_name] = losses[loss_name][:split_idx]

    for loss_name, loss_value in losses.items():
        # To avoid stats pollution for validation inside training epoch.
        loss_name = f'{loss_name}/{tag_tail}'
        if loss_name.startswith('img$'):
            log_vars[loss_name] = loss_value
            continue
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key and not _key.startswith('adv'))
    adv_loss = sum(_value for _key, _value in log_vars.items() if _key.startswith('adv_loss'))
    # TODO: Make the code more elegant here.
    log_vars[f'loss/{tag_tail}'] = loss
    log_vars[f'adv_loss/{tag_tail}'] = adv_loss
    for name in log_vars:
        if not name.startswith('img$'):
            log_vars[name] = log_vars[name].item()

    return loss, adv_loss, log_vars


def adv_batch_processor(model, data, mode, **kwargs):
    # NOTE: The mode is str instead of boolean now.
    discriminator = kwargs.get('discriminator')
    re_weight = kwargs.get('re_weight', dict())
    losses, _ = model(**data)
    pred_pose_shape = losses.pop('pred_pose_shape')
    #print('pred pose shape', pred_pose_shape.shape)
    batch_size = pred_pose_shape.shape[0]

    mosh = kwargs.get('mosh')
    #mosh_pose_shape = mosh.batch_random_get(batch_size)
    #print('amass batch shape', mosh_pose_shape.shape)
    sampled_idxs = np.round(np.random.sample(batch_size) * (len(mosh['pose']) - 2)).astype(np.int)

    mosh_pose = torch.tensor(deepcopy(mosh['pose'][sampled_idxs].astype(np.float32)))
    mosh_shape = torch.tensor(deepcopy(mosh['shape'][sampled_idxs].astype(np.float32)))
    #print(mosh_pose.shape, mosh_shape.shape)
    mosh_pose_shape = torch.cat([batch_rodrigues(mosh_pose.view(-1, 3)).view(batch_size, -1), mosh_shape], dim=1)

    loss_disc, adv_loss_fake, adv_loss_real = adversarial_loss(discriminator, pred_pose_shape, mosh_pose_shape)
    losses.update({
        'g_loss_disc': loss_disc, # g_loss through D network
        'adv_loss_fake': adv_loss_fake,
        'adv_loss_real': adv_loss_real
    })
    for k, v in re_weight.items():
        if k.startswith('adv_loss') or k.startswith('g_loss_disc'):
            losses[k] *= v
        else:
            # generator loss has been set in smpl_loss.py
            continue

    tag_tail = mode
    loss, adv_loss, log_vars = parse_adv_losses(losses, tag_tail)

    # if loss.item() > 1:
    #     print('=' * 30, 'start of meta', '=' * 30)
    #     print([(i['idx'].item(), i['flip']) for i in data['img_meta'].data[0]])
    #     print('=' * 30, 'end of meta', '=' * 30)

    #print('yeah1')

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data),
        adv_loss=adv_loss
    )

    if kwargs.get('log_grad', False):
        with torch.no_grad():
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm()
                    total_norm += param_norm.item()
            if total_norm:
                outputs['log_vars'][f'total_grad/{tag_tail}'] = total_norm

    return outputs

def _add_file_handler(
                    logger,
                    filename=None,
                    mode='w',
                    level=logging.INFO):
    # TODO: move this method out of runner
    file_handler = logging.FileHandler(filename, mode)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger
    
def init_logger(log_dir=None, level=logging.INFO, timestamp=None):
    """Init the logger.

    Args:
        log_dir(str, optional): Log file directory. If not specified, no
            log file will be used.
        level (int or str): See the built-in python logging module.

    Returns:
        :obj:`~logging.Logger`: Python logger.
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    logger = logging.getLogger(__name__)
    if log_dir and int(os.environ['LOCAL_RANK']) == 0:
        filename = '{}.log'.format(timestamp)
        log_file = osp.join(log_dir, filename)
        _add_file_handler(logger, log_file, level=level)
    return logger

def train_detector_adv(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    model_adv = Discriminator(include_last_two_pose=False).cuda()

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        
        model_adv = build_ddp(
            model_adv,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        model_adv = build_dp(model_adv, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)
    optimizer_adv = build_optimizer(model, cfg.adv_optimizer)

    # runner = build_runner(
    #     cfg.runner,
    #     default_args=dict(
    #         model=model,
    #         optimizer=optimizer,
    #         work_dir=cfg.work_dir,
    #         logger=logger,
    #         meta=meta))

    # global runner
    # runner = AdvRunner(model_adv, optimizer_adv, model, adv_batch_processor, 
    #                    optimizer, cfg.work_dir, cfg.log_level)
    
    logger = init_logger(log_dir=cfg.work_dir, level=logging.INFO, timestamp=timestamp)
    runner = AdvRunner(model_adv, optimizer_adv, model, batch_processor=None, 
                       optimizer=optimizer, work_dir=cfg.work_dir, logger=logger)


    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    
    if cfg.get('optimizer_cfg', None) is None:
        optimizer_config = None
    elif fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    # runner.register_training_hooks(
    #     cfg.lr_config,
    #     optimizer_config,
    #     cfg.checkpoint_config,
    #     cfg.log_config,
    #     cfg.get('momentum_config', None),
    #     custom_hooks_config=cfg.get('custom_hooks', None))

    runner.register_training_hooks(cfg.adv_optimizer_config, cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    # if validate:
    #     val_dataloader_default_args = dict(
    #         samples_per_gpu=1,
    #         workers_per_gpu=2,
    #         dist=distributed,
    #         shuffle=False,
    #         persistent_workers=False)

    #     val_dataloader_args = {
    #         **val_dataloader_default_args,
    #         **cfg.data.get('val_dataloader', {})
    #     }
    #     # Support batch_size > 1 in validation

    #     if val_dataloader_args['samples_per_gpu'] > 1:
    #         # Replace 'ImageToTensor' to 'DefaultFormatBundle'
    #         cfg.data.val.pipeline = replace_ImageToTensor(
    #             cfg.data.val.pipeline)
    #     val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

    #     val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
    #     eval_cfg = cfg.get('evaluation', {})
    #     eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
    #     eval_hook = DistEvalHook if distributed else EvalHook
    #     # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
    #     # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
    #     runner.register_hook(
    #         eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    # runner.run(data_loaders, cfg.workflow)
    print("cfg.runner.max_epochs: ", cfg.runner.max_epochs)
    runner.run(data_loaders, cfg.workflow, cfg.runner.max_epochs)