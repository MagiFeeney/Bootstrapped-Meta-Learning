"""
modified from torch.optim.__init__.py
"""

from torch.optim import *

from .meta_optimizer import SlidingWindow
from .meta_optimizer import MetaOptimizer
from .meta_sgd import MetaSGD
from .meta_adam import MetaAdam
from .meta_rmsprop import MetaRMSprop

from ._functional import broadcast_net
from ._functional import cosine_similarity