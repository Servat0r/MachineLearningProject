"""
Basic "all-imports" file.
"""
from __future__ import annotations
from abc import abstractmethod, ABC
from typing import *
import numpy as np
import numpy.random as npr
import cupy as cp
import numba as nb


# TypeVars
TReal = TypeVar('TReal', bound=Union[int, float])
TShape = TypeVar('TShape', bound=Union[int, Iterable[int], tuple[int]])
TBoolStr = TypeVar('TBoolStr', bound=tuple[bool, Optional[str]])
TBoolAny = TypeVar('TBoolAny', bound=tuple[bool, Optional[Any]])
TDesc = TypeVar('TDesc', bound=dict[str, Any])
