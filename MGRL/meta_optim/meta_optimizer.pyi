"""
modified from torch.optim.meta_optimizer.pyi
"""
from typing import Callable, Optional, List, Union

import torch
from torch import Tensor, nn


class SlidingWindow(object):
    PICKUP_STEP = int
    REGISTERED_OPTIMIZERS: dict
    OPTIMIZER_COUNTER: int
    ALREADY_SLIDING: bool
    OFFLINE: bool
    REPLACE_NAN: bool
    RECORD_GRAD_TAPE: bool
    def __init__(self,
                 offline: Optional[bool]=...,
                 replace_nan: Optional[bool]=...,
                 record_grad_tape: Optional[bool]=...,
                 pickup_step: Optional[int]=...) -> None:...
    def __enter__(self) -> None:...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:...


_params_t = nn.Module


class GradNormalizer:
    max_norm: Union[float, torch.Tensor]
    norm_type: float
    error_if_nonfinite: bool
    def __init__(self,
                 max_norm: Union[float, torch.Tensor],
                 norm_type: float,
                 error_if_nonfinite: bool) -> torch.Tensor: ...


class MetaOptimizer(object):
    defaults: dict
    state: dict
    param_groups: List[dict]
    parameters_backup: List[nn.Parameter]
    first_offline_step: bool
    _parameter_list: List[List[str]]
    _plain_parameter_list: List[Tensor]
    _grad_normalizer: Union[GradNormalizer, None]
    _last_state_dict: dict

    def __init__(self, nets: List[_params_t], default: dict) -> None: ...
    def __setstate__(self, state: dict) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> None: ...
    def zero_grad(self, set_to_none: Optional[bool]=...) -> None: ...
    def step(self,
             loss: Tensor,
             closure: Optional[Callable[[], float]]=...,
             allow_unused: Optional[bool]=...) -> Optional[float]: ...
    def _step(self) -> None: ...
    def named_parameters(self) -> dict: ...
    def add_param_group(self, param_group: dict) -> None: ...
    def reset_grad_normalizer(self,
                              max_norm: Union[float, torch.Tensor],
                              norm_type: float= ...,
                              error_if_nonfinite: bool= ...) -> None: ...
