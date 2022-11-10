from __future__ import annotations
from .utils import *


class BaseFunction(Callable):
    """
    Base class for activation and loss functions. It basically acts as either
    a normal class with its function definition or as a wrapper for an external
    function.
    """
    _diffs__ = {}

    SCALAR = 'scalar'  # scalar (f: R -> R)
    GRADIENT = 'gradient'  # gradient (f: R^n -> R)
    EGRADIENT = 'egradient'  # elementwise gradient (f: R^n -> R^n with each f_i dependent only on x_i)
    JACOBIAN = 'jacobian'  # jacobian (f: R^m -> R^n)

    @staticmethod
    def set_primitive(func: Type[BaseFunction]):
        """
        Simple decorator to register a primitive (yet not necessarily
        differentiable) function.
        """
        BaseFunction._diffs__[func] = {}
        return func

    @staticmethod
    def set_diff(func: Type[BaseFunction], diff_type=GRADIENT):
        """
        Decorator to register a BaseFunction class as differential of
        another already registered BaseFunctionclass 'func'.
        :param func: Function to which differential is added.
        :param diff_type: Type of differential. Each function has actually
        one differential, this is a shorthand for example for registering
        vector-jacobian products with backprop arguments rather than simply
        the entire jacobian, or for elementwise activation functions.
        It is GRADIENT for actual gradients, EGRADIENT for elementwise ones,
        SCALAR for derivatives and JACOBIAN for jacobians. Defaults to GRADIENT.
        """
        def wrapper(dfunc: Type[BaseFunction]):
            # Checks if 'func' is not already registered with a differential.
            diffs: dict = BaseFunction._diffs__.get(func) or {}
            old_dfunc = diffs.get(diff_type, None)
            if old_dfunc is not None:
                raise KeyError(f"Function {func} has already a '{diff_type}' " +
                                 f"differential registered: {old_dfunc}")
            diffs[diff_type] = dfunc
            BaseFunction._diffs__[func] = diffs
            return func
        return wrapper

    @staticmethod
    def set_grad(func: Type[BaseFunction]):
        return BaseFunction.set_diff(func, diff_type=BaseFunction.GRADIENT)

    @staticmethod
    def set_egrad(func: Type[BaseFunction]):
        return BaseFunction.set_diff(func, diff_type=BaseFunction.EGRADIENT)

    @staticmethod
    def set_jacobian(func: Type[BaseFunction]):
        return BaseFunction.set_diff(func, diff_type=BaseFunction.JACOBIAN)

    @staticmethod
    def set_derivative(func: Type[BaseFunction]):
        return BaseFunction.set_diff(func, diff_type=BaseFunction.SCALAR)

    def get_diff(self, diff_type=GRADIENT):
        diffs = BaseFunction._diffs__.get(type(self))
        diff = diffs.get(diff_type, None)
        if diff is None:
            raise KeyError(f"Function {self} has no registered '{diff_type}' differential")
        return diff

    def grad(self, *args, **kwargs) -> BaseFunction | None:
        return self.get_diff(self.GRADIENT)(*args, **kwargs)

    def egrad(self, *args, **kwargs) -> BaseFunction | None:
        return self.get_diff(self.EGRADIENT)(*args, **kwargs)

    def jacobian(self, *args, **kwargs) -> BaseFunction | None:
        return self.get_diff(self.JACOBIAN)(*args, **kwargs)

    def derivative(self, *args, **kwargs) -> BaseFunction | None:
        return self.get_diff(self.SCALAR)(*args, **kwargs)

    @abstractmethod
    def __call__(self, x: np.ndarray):
        pass

    def check_input_shape(self, x: np.ndarray) -> bool:
        """
        By default for these functions, accepted shapes are n, (n,), (l, n), (l, 1, n).
        :param x:
        :return:
        """
        x_shape = x.shape
        if isinstance(x_shape, int):
            return x_shape > 0
        elif not isinstance(x_shape, tuple):
            return False
        else:
            return all([1 <= len(x_shape) <= 3, x_shape[0] > 0, x_shape[-1] > 0,
                        len(x_shape) <= 2 or (x_shape[1] == 1)])

    def normalize_input_shape(self, x: np.ndarray, inplace: bool = False) -> np.ndarray:
        """
        Default reshaping to a canonical (l, 1, n) row vector form.
        :param x:
        :param inplace:
        :return:
        """
        shape = x.shape
        if not self.check_input_shape(x):
            raise ValueError(f"Invalid input shape: expected one of n, (n,), (l,n), (l,1,n), got {shape}")
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
            xb = np.reshape(xb, (l, 1, n))   # xb points to the same memory location of x
            return xb
        else:
            return np.reshape(x, (l, 1, n))  # this is another array

    # todo if convenient, overload __add__, __sub__, __mul__ and __matmul__


class ActivationFunction(BaseFunction):
    """
    Base class for activation functions.
    """
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass


@BaseFunction.set_primitive
class Sigmoid(ActivationFunction):

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = self.normalize_input_shape(x)
        return 1.0/(1.0 + np.exp(-x))


@BaseFunction.set_egrad(Sigmoid)
class _SigmoidEGrad(BaseFunction):

    func = Sigmoid()  # As class attribute, this is instantiated only one time

    def __call__(self, x: np.ndarray) -> np.ndarray:
        s = self.func(x)
        return s * (1.0 - s)


@BaseFunction.set_primitive
class Sign(ActivationFunction):

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = self.normalize_input_shape(x)
        return np.sign(x)


@BaseFunction.set_egrad(Sign)
class _SignEGrad(BaseFunction):

    func = Sign()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.abs(self.func(x))


@BaseFunction.set_primitive
class Tanh(ActivationFunction):

    def __call__(self, x: np.ndarray):
        x = self.normalize_input_shape(x)
        return np.tanh(x)


@BaseFunction.set_egrad(Tanh)
class _TanhEGrad(BaseFunction):

    func = Tanh()

    def __call__(self, x: np.ndarray):
        return 1.0 - np.square(self.func(x))


@BaseFunction.set_primitive
class ReLU(ActivationFunction):

    def __call__(self, x: np.ndarray):
        x = self.normalize_input_shape(x)
        return np.maximum(x, np.zeros_like(x))  # todo can we write it better?


@BaseFunction.set_egrad(ReLU)
class _ReLUEGrad(BaseFunction):
    # todo this gradient is actually incorrect since ReLU is not differentiable in 0
    # todo see if it is necessary to change it as it is said in ML or CM
    func = ReLU()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y: np.ndarray = self.func(x)
        np.place(y, y > 0, 1)
        return y


@BaseFunction.set_primitive
class Softmax(BaseFunction):

    def __init__(self, const_shift=0, max_shift=False):
        super(Softmax, self).__init__()
        self.const_shift = const_shift  # constant shift for arguments
        self.max_shift = max_shift      # subtracting the maximum input value from all inputs

    def __shift_x(self, x: np.ndarray) -> np.ndarray:
        """
        Applies constant and/or maximum shift to input data BEFORE applying Softmax.
        This is a common technique for avoiding numerical instability if the input
        array contains large positive values. Since the function x -> x - c for c
        constant has the identity matrix as differential, the operation is made
        in-place since the backprop value is 1.  # todo check if it is correct!
        """
        if self.const_shift != 0:
            x += self.const_shift
        if self.max_shift:
            x -= np.max(x, axis=2, keepdims=True)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = self.normalize_input_shape(x)
        self.__shift_x(x)   # todo check if actually modifies inplace!
        xc = np.exp(x)
        s = np.sum(xc, axis=2, keepdims=True)
        xc /= s
        return xc


@BaseFunction.set_jacobian(Softmax)
class _SoftmaxJac(BaseFunction):

    def __init__(self, smax: Softmax):
        self.func = smax

    def __call__(self, x: np.ndarray):
        """
        Calculates the jacobian of Softmax in x.
        """
        s = self.func(x)  # (l, 1, n)
        # Calculate the common part
        st = np.transpose(s, axes=[0, 2, 1])  # (l, n, 1)
        y = (-1.0) * (st @ s)  # (l, n, n)
        # Reshape for summing elements on the ("last two") diagonals
        l, n = y.shape[0], y.shape[2]
        ind = np.array(range(n))
        y[:, ind, ind] = s[:, 0, ind]
        # todo actually for small matrices (10^1 or 10^2) a double for-loop is faster
        # todo for 10^3 or higher, the above y[:,ind,ind] ... is much faster
        return y


@BaseFunction.set_grad(Softmax)
class _SoftmaxVJP(BaseFunction):

    def __init__(self, smax: Softmax):
        self.func = smax

    def __call__(self, x: np.ndarray, dvals: np.ndarray):
        """
        Calculates the vector-jacobian product of Softmax in x as a backward pass
        assuming that Softmax is composed with a scalar-valued function
        (e.g. a loss) and that 'dvals' are the values calculated
        in the previous backward pass.
        :param x: array of shape (l, 1, n)
        :param dvals: array of shape (l, n, 1)
        :return: Gradient of Softmax in x, with shape (l, n, 1)
        """
        s = self.func(x)
        sd = s * dvals
        sd_sum = np.sum(sd, axis=1, keepdims=True)
        return sd - sd_sum * s


@BaseFunction.set_primitive
class CategoricalCrossEntropy(BaseFunction):

    def __init__(self, target_truth_values: np.ndarray, clip_value: TReal = 1e-7):
        super(CategoricalCrossEntropy, self).__init__()
        target_truth_values = self.normalize_target_shape(target_truth_values)
        self.target_truth_values = target_truth_values
        self.clip_value = clip_value

    def set_truth_values(self, truth_values: np.ndarray):
        self.target_truth_values = truth_values

    def check_input_shape(self, x: np.ndarray) -> bool:
        """
        Input values should have the shape (l, 1, n).
        :param x:
        :return:
        """
        x_shape = x.shape
        t_shape = self.target_truth_values.shape
        return all([
            len(x_shape) == 3, x_shape[0] == t_shape[0], x_shape[1] == 1,
            x_shape[2] > 0 if len(t_shape) == 1 else x_shape[2] == t_shape[2],  # todo check what's going on here!
        ])

    @staticmethod
    def check_targets_shape(target_truth_values: np.ndarray) -> TBoolStr:
        """
        Target values should have shape of either l, (l,), (l, n), (l, 1, n).
        Normalized values will be (l,) and (l, 1, n) for use with losses.
        :param target_truth_values: Target distribution.
        :return:
        """
        shape = target_truth_values.shape
        err_msg = f"Invalid target shape: expected one of l, (l,), (l,n), (l,1,n), got {shape}"
        if isinstance(shape, int):
            result = shape > 0
        elif not 1 <= len(shape) <= 3:
            result = False
        elif len(shape) == 3:
            result = all([shape[0] > 0, shape[1] == 1, shape[2] > 0])
        elif len(shape) == 2:
            result = all([shape[0] > 0, shape[1] > 0])
        else:
            result = shape[0] > 0
        return result, (None if result else err_msg)

    def normalize_target_shape(self, target_truth_values: np.ndarray) -> np.ndarray:
        """
        Normalization of target shapes:
            1) l, (l,) -> (l,)
            2) (l,n), (l,1,n) -> (l,1,n)
        :param target_truth_values: Values to normalize.
        :return:
        """
        result, msg = self.check_targets_shape(target_truth_values)
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

    def __call__(self, x: np.ndarray):
        x = self.normalize_input_shape(x)
        samples = len(x)
        x_clipped = np.clip(x, self.clip_value, 1 - self.clip_value)
        correct_confidences = []

        trshape = self.target_truth_values.shape
        if len(trshape) == 1:  # (l,)
            correct_confidences = x_clipped[range(samples), 0, self.target_truth_values]
        elif len(trshape) == 3:  # (l, 1, n)
            filtered_x = x_clipped * self.target_truth_values
            correct_confidences = np.sum(filtered_x, axis=2)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


@BaseFunction.set_grad(CategoricalCrossEntropy)
class _CatCEGrad(BaseFunction):

    def __init__(self, func: CategoricalCrossEntropy):
        self.func = func
        self.clip_value = func.clip_value
        self.target_truth_values = func.target_truth_values

    def __call__(self, x: np.ndarray):
        x = self.func.normalize_input_shape(x)
        x_clip = - 1.0 / np.clip(x, self.clip_value, 1 - self.clip_value)
        trshape = self.target_truth_values.shape
        if len(trshape) == 1:
            # If labels are sparse, turn them into one-hot encoded ones
            self.target_truth_values = np.eye(x.shape[2])[self.target_truth_values]
            self.target_truth_values = np.reshape(self.target_truth_values, (trshape[0], 1, x.shape[2]))
        result = x_clip * self.target_truth_values
        return result


@BaseFunction.set_primitive
class SquareError(BaseFunction):
    """
    Calculates square error (coefficient * ||x||^2) for the patterns SEPARATELY.
    """
    def __init__(self, coefficient=0.5):
        self.coefficient = coefficient

    def __call__(self, x: np.ndarray):
        x = self.normalize_input_shape(x)
        y = self.coefficient * np.square(np.linalg.norm(x, axis=2))
        return y


@BaseFunction.set_grad(SquareError)
class _SquareErrorGrad(BaseFunction):

    def __init__(self, const=0.5):
        self.const = const

    def __call__(self, x: np.ndarray):
        x = self.normalize_input_shape(x)
        return self.const * 2 * x


@BaseFunction.set_primitive
class AbsError(BaseFunction):

    def __call__(self, x: np.ndarray):
        x = self.normalize_input_shape(x)
        y = np.sum(np.abs(x), axis=2)
        return y


__all__ = [
    'BaseFunction',
    'ActivationFunction',
    'Sigmoid',
    'Sign',
    'Tanh',
    'ReLU',
    'Softmax',
    'CategoricalCrossEntropy',
    'SquareError',
    'AbsError',
]
