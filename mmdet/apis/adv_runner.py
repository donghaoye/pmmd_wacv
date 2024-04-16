import mmcv
from mmcv.runner import Runner
import torch
from ..core.utils.adv_optimizer import AdvOptimizerHook
import os.path as osp
from mmcv.runner.checkpoint import save_checkpoint, load_checkpoint
from ..models.smpl_heads.smpl_common import Discriminator
import logging

# from . import hooks
# from .hooks import Hook
from mmcv.runner.hooks.optimizer import OptimizerHook as Hook


class AdvRunner(Runner):

    def __init__(self, adv_model, adv_optimizer, *args, **kwargs):
        self.adv_optimizer = adv_optimizer
        self.adv_model = adv_model
        super(AdvRunner, self).__init__(*args, **kwargs)

    def register_training_hooks(self, adv_optimizer_config, *args, **kwargs):
        super(AdvRunner, self).register_training_hooks(*args, **kwargs)
        self.register_hook(self.build_hook(adv_optimizer_config, AdvOptimizerHook), priority='HIGH')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):

        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        print("!! start save_checkpoint ", f"self.epoch: {self.epoch + 1}", f"self.iter: {self.iter}")
        
        # ## debug for common iteration
        # if self.iter >= 8000:
        #     print("self.iter >= 8000!!!!!: ", self.iter)
        #     meta.update(epoch=22, iter=self.iter)
        #     print("!! start save_checkpoint ", "self.epoch: 22", f"self.iter: {self.iter}")

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, 'latest.pth')
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # use relative symlink
        mmcv.symlink(filename, linkpath)

        filename_tmpl = 'adv_' + filename_tmpl
        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, 'adv_latest.pth')
        optimizer = self.adv_optimizer if save_optimizer else None
        save_checkpoint(self.adv_model, filepath, optimizer=optimizer, meta=meta)
        # use relative symlink
        mmcv.symlink(filename, linkpath)

        print("!! end save_checkpoint ", f"self.epoch: {self.epoch + 1}", f"self.iter: {self.iter}")

    def load_adv_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.adv_model, filename, map_location, strict,
                               self.logger)

    def resume(self, checkpoint, resume_optimizer=True,
               map_location='default'):
        # adv_checkpoint = checkpoint.replace('latest.pth', 'adv_latest.pth')
        filename = osp.basename(checkpoint)
        adv_checkpoint = checkpoint.replace(filename, 'adv_' + filename)
        print("resume from: ", adv_checkpoint)
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
            adv_checkpoint = self.load_adv_checkpoint(
                adv_checkpoint, map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)
            adv_checkpoint = self.load_adv_checkpoint(
                adv_checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'optimizer' in adv_checkpoint and resume_optimizer:
            self.adv_optimizer.load_state_dict(adv_checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        # super(AdvRunner, self).run(data_loaders, workflow, max_epochs, discriminator=self.adv_model, **kwargs)
        super(AdvRunner, self).run(data_loaders, workflow, max_epochs, **kwargs)

    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError('"args" must be either a Hook object'
                            ' or dict, not {}'.format(type(args)))

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)