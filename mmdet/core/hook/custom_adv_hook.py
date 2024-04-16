# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner.hooks import HOOKS, Hook, OptimizerHook
# from mmdet.models.common import set_requires_grad

@HOOKS.register_module()
class CustomAdvHook(OptimizerHook):
    def __init__(self, grad_clip=None):
        super(CustomAdvHook, self).__init__(
            grad_clip=grad_clip)
        print("custom hook")


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad for all the networks.

        Args:
            nets (nn.Module | list[nn.Module]): A list of networks or a single
                network.
            requires_grad (bool): Whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def after_train_iter(self, runner):
        pass

        # runner.optimizer['discriminator'].zero_grad()
        # runner.outputs['adv_loss'].backward()
        # # for name, param in runner.outputs['discriminator'].named_parameters():
        # #     if param.grad is None:
        # #         print(name)

        # if self.grad_clip is not None:
        #     # grad_norm_d = self.clip_grads(runner.model.discriminator.parameters())
        #     grad_norm_d = self.clip_grads(runner.outputs['discriminator'].parameters())
        #     if grad_norm_d is not None:
        #         runner.log_buffer.update({'grad_norm_d': float(grad_norm_d)},
        #                                  runner.outputs['num_samples'])
        # runner.optimizer['discriminator'].step()

        # self.set_requires_grad(runner.outputs['discriminator'], False)
        # runner.optimizer['generator'].zero_grad()
        # runner.outputs['loss'].backward()
        # if self.grad_clip is not None:
        #     # grad_norm_g = self.clip_grads(runner.model.generator.parameters())
        #     grad_norm_g = self.clip_grads(runner.outputs['generator'].parameters())
        #     if grad_norm_g is not None:
        #         # Add grad norm to the logger
        #         runner.log_buffer.update({'grad_norm_g': float(grad_norm_g)},
        #                                  runner.outputs['num_samples'])
        # runner.optimizer['generator'].step()
