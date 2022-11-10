"""
Base handlers for differentiation, input shape checking and normalization,
and vector-jacobian products (VJPs) for backpropagation.
This module is intended as a 'singleton', i.e. it is better to do
import (...).diffs (as ...), rather than from (...).diffs import *.
"""
from __future__ import annotations
from functools import wraps
from .utils import *

__diffs__: dict[Callable, dict[str, Callable]] = {}
__shape_checks__: bool = __debug__
SCALAR = 'scalar'  # f : R -> R
GRADIENT = 'gradient'  # f : R^n -> R
EGRADIENT = 'egradient'  # f : R^n -> R^n, element-wise gradient (diagonal jacobian)
JACOBIAN = 'jacobian'  # f : R^m -> R^n (generic)
VJP = 'vjp'  # vector-jacobian product (base for backprop)


def is_shape_checking():
    return __shape_checks__


def enable_shape_checks():
    global __shape_checks__
    __shape_checks__ = True


def disable_shape_checks():
    global __shape_checks__
    __shape_checks__ = False


def row_vector_input_shape_checker(x: np.ndarray) -> TBoolStr:
    """
    Checker for row-vector inputs batches, i.e. accepted shapes are n, (n,), (l, n), (l, 1, n).
    :param x:
    :return:
    """
    if not is_shape_checking():
        return True, None
    x_shape = x.shape
    if isinstance(x_shape, int):
        result = x_shape > 0
        return result, (None if result else f"Invalid integer input shape: expected n > 0, got {x_shape}")
    elif not isinstance(x_shape, tuple):
        return False, "Shape of input is neither a tuple nor an integer!"
    else:
        result = all([1 <= len(x_shape) <= 3, x_shape[0] > 0, x_shape[-1] > 0, len(x_shape) <= 2 or (x_shape[1] == 1)])
        return result, \
            (None if result else f"Invalid input shape: expected one of n, (n,), (l,n), (l,1,n), got {x_shape}")


def row_vector_input_shape_normalizer(x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    Default reshaping to a canonical (l, 1, n) row vector form.
    :param x:
    :param inplace:
    :return:
    """
    shape = x.shape
    l, n = 1, 1
    if isinstance(shape, int):
        n = shape
    elif len(shape) == 1:
        n = shape[0]
    else:
        l, n = shape[0], shape[-1]
    if inplace:
        # Create a view for operating in-place
        xb = x.view()
        xb = np.reshape(xb, (l, 1, n))  # xb points to the same memory location of x
        return xb
    else:
        return np.reshape(x, (l, 1, n))  # this is another array


def row_vector_target_shape_normalizer(target_truth_values: np.ndarray) -> np.ndarray:
    """
    Normalization of target shapes:
        1) l, (l,) -> (l,)
        2) (l,n), (l,1,n) -> (l,1,n)
    :param target_truth_values: Values to normalize.
    :return:
    """
    result, msg = row_vector_input_shape_checker(target_truth_values)
    if not result:
        raise ValueError(msg)
    shape = target_truth_values.shape
    # Shapes l, (l,), (l,1) are for categorical labels
    if isinstance(shape, int):
        shape = (shape,)
    # Shapes (l, n), (l, 1, n) are for one-hot encoded labels
    elif len(shape) >= 2:
        shape = (shape[0], 1, shape[-1])
    # Reshaping
    return np.reshape(target_truth_values, shape)


def set_input_shape_checker(checker: Callable = row_vector_input_shape_checker, inarg=0):
    if checker is None:
        raise ValueError(f"Input shape checker cannot be None!")

    def wrapper(func: Callable):
        @wraps(func)
        def new_func(*args, **kwargs):
            result, msg = checker(args[inarg])
            if not result:
                return ValueError(msg)
            return func(*args, **kwargs)
        return new_func
    return wrapper


def set_input_shape_normalizer(normalizer: Callable = row_vector_input_shape_normalizer, inarg=0):
    if normalizer is None:
        raise ValueError(f"Normalizer for input shape cannot be None!")

    def wrapper(func: Callable):
        @wraps(func)
        def new_func(*args, **kwargs):
            x = normalizer(args[inarg])
            args = list(args)
            args[inarg] = x
            args = tuple(args)
            return func(*args, **kwargs)
        return new_func
    return wrapper


def set_primitive(input_checker: Optional[Callable] = None,
                  input_normalizer: Optional[Callable] = None,
                  input_arg: int = 0):
    """
    Simple decorator to register a primitive (yet not necessarily
    differentiable) function.
    """
    def wrapper(func: Callable):
        # Apply input shape decorators to func
        if input_checker is not None:
            func = set_input_shape_checker(input_checker, input_arg)(func)
        if input_normalizer is not None:
            func = set_input_shape_normalizer(input_normalizer, input_arg)(func)
        __diffs__[func] = {}
        return func
    return wrapper


def set_diff(func: Callable, diff_type=GRADIENT, input_checker: Optional[Callable] = None,
             input_normalizer: Optional[Callable] = None, input_arg: int = 0):
    """
    Decorator to register a DiffObject class as differential of
    another already registered DiffObjectclass 'func'.
    :param func: Function to which differential is added.
    :param diff_type: Type of differential. Each function has actually
    one differential, this is a shorthand for example for registering
    vector-jacobian products with backprop arguments rather than simply
    the entire jacobian, or for elementwise activation functions.
    It is GRADIENT for actual gradients, EGRADIENT for elementwise ones,
    SCALAR for derivatives and JACOBIAN for jacobians. Defaults to GRADIENT.
    """

    def wrapper(dfunc: Callable):
        # Checks if 'func' is not already registered with a differential.
        diffs: dict = __diffs__.get(func) or {}
        old_dfunc = diffs.get(diff_type, None)
        if old_dfunc is not None:
            raise KeyError(f"Function {func} has already a '{diff_type}' " +
                           f"differential registered: {old_dfunc}")
        # Apply input shape decorators to dfunc
        if input_checker is not None:
            dfunc = set_input_shape_checker(input_checker, input_arg)(dfunc)
        if input_normalizer is not None:
            dfunc = set_input_shape_normalizer(input_normalizer, input_arg)(dfunc)
        diffs[diff_type] = dfunc
        __diffs__[func] = diffs
        return dfunc

    return wrapper


def set_grad(func: Callable, input_checker: Optional[Callable] = None,
             input_normalizer: Optional[Callable] = None, input_arg: int = 0):
    return set_diff(func, GRADIENT, input_checker, input_normalizer, input_arg)


def set_egrad(func: Callable, input_checker: Optional[Callable] = None,
              input_normalizer: Optional[Callable] = None, input_arg: int = 0):
    return set_diff(func, EGRADIENT, input_checker, input_normalizer, input_arg)


def set_jacobian(func: Callable, input_checker: Optional[Callable] = None,
                 input_normalizer: Optional[Callable] = None, input_arg: int = 0):
    return set_diff(func, JACOBIAN, input_checker, input_normalizer, input_arg)


def set_derivative(func: Callable, input_checker: Optional[Callable] = None,
                   input_normalizer: Optional[Callable] = None, input_arg: int = 0):
    return set_diff(func, SCALAR, input_checker, input_normalizer, input_arg)


def set_vjp(func: Callable, input_checker: Optional[Callable] = None,
            input_normalizer: Optional[Callable] = None, input_arg: int = 0):
    return set_diff(func, VJP, input_checker, input_normalizer, input_arg)


def get_diff(func: Callable, diff_type=GRADIENT):
    diffs = __diffs__.get(func)
    diff = diffs.get(diff_type, None)
    if diff is None:
        raise KeyError(f"Function {func} has no registered '{diff_type}' differential")
    return diff


def grad(func, *args, **kwargs):
    return get_diff(func, GRADIENT)(*args, **kwargs)


def egrad(func, *args, **kwargs):
    return get_diff(func, EGRADIENT)(*args, **kwargs)


def jacobian(func, *args, **kwargs):
    return get_diff(func, JACOBIAN)(*args, **kwargs)


def derivative(func, *args, **kwargs):
    return get_diff(func, SCALAR)(*args, **kwargs)


def vjp(func, *args, **kwargs):
    return get_diff(func, VJP)(*args, **kwargs)


__all__ = [
    'is_shape_checking',
    'enable_shape_checks',
    'disable_shape_checks',
    'row_vector_input_shape_checker',
    'row_vector_input_shape_normalizer',
    'row_vector_target_shape_normalizer',
    'set_input_shape_checker',
    'set_input_shape_normalizer',
    'SCALAR',
    'GRADIENT',
    'EGRADIENT',
    'JACOBIAN',
    'VJP',
    'set_primitive',
    'set_diff',
    'set_derivative',
    'set_grad',
    'set_egrad',
    'set_jacobian',
    'set_vjp',
    'derivative',
    'grad',
    'egrad',
    'jacobian',
    'vjp',
]
