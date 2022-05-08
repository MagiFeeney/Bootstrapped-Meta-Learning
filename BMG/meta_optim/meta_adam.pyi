import torch
from typing import Tuple, Union
from .meta_optimizer import _params_t, MetaOptimizer

class MetaAdam(MetaOptimizer):
    def __init__(self, params: _params_t, lr: Union[float, torch.Tensor]=..., betas: Tuple[Union[float, torch.Tensor]]=..., eps: Union[float, torch.Tensor]=..., weight_decay: Union[float, torch.Tensor]=..., amsgrad: bool = ...) -> None: ...
