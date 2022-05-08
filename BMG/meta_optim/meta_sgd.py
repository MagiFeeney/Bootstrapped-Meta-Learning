"""
modified from torch.optim.meta_sgd.py
"""
import torch
from . import _functional as F
from .meta_optimizer import MetaOptimizer


class MetaSGD(MetaOptimizer):
    def __init__(self, params, lr=1e-3, momentum=0., dampening=0.,
                 weight_decay=0., nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def _step(self):
        for i, group in enumerate(self.param_groups):
            net = group['params']
            params_with_grad = self._parameter_list[i]
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in params_with_grad:
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

            F.meta_sgd(net,
                       momentum_buffer_list,
                       weight_decay=weight_decay,
                       momentum=momentum,
                       lr=lr,
                       dampening=dampening,
                       nesterov=nesterov)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
