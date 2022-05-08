from typing import Tuple
from .meta_optimizer import _params_t, MetaOptimizer

class MetaRMSprop(MetaOptimizer):
    def __init__(self, params: _params_t, lr: float=..., alpha: float=..., eps: float=..., weight_decay: float=..., momentum: float=...,  centered: bool=...) -> None: ...
