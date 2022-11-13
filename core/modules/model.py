# Model class: main class for building a complete Neural Network
from __future__ import annotations
from ..utils import *
import core.diffs as dfs
import core.functions as cf
from .layers import *
from .parameters import *
from .losses import *
from .optimizers import *


class Model:
    """
    Base class for a Neural Network
    """
    def __init__(self, layers: Layer | Sequence[Layer] | Iterable[Layer], ):
        ...