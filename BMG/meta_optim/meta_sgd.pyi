"""
modified from torch.optim.meta_sgd.pyi
"""
import torch
from typing import Union
from .meta_optimizer import _params_t, MetaOptimizer

class MetaSGD(MetaOptimizer):
    def __init__(self, params: _params_t, lr: Union[float, torch.Tensor], momentum: Union[float, torch.Tensor]=..., dampening: Union[float, torch.Tensor]=..., weight_decay: Union[float, torch.Tensor]=..., nesterov:bool=...) -> None: ...
