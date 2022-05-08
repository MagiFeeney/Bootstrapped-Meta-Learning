"""
modified from torch.optim.__init__.pyi
"""

from torch.optim import *

from .meta_optimizer import SlidingWindow as SlidingWindow
from .meta_optimizer import MetaOptimizer as MetaOptimizer
from .meta_sgd import MetaSGD as MetaSGD
from .meta_adam import MetaAdam as MetaAdam
from .meta_rmsprop import MetaRMSprop as MetaRMSprop

from ._functional import broadcast_net as broadcast_net
from ._functional import cosine_similarity as cosine_similarity
