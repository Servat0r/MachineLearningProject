from __future__ import annotations
from .utils import np, abstractmethod, Callable, TBoolStr, TReal


class DifferentiableFunction(Callable):
    """
    Base class for activation and loss functions.
    """
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
        elif not 1 <= len(x_shape) <= 3:
            return False
        else:
            return all([x_shape[0] > 0, x_shape[-1] > 0, len(x_shape) <= 2 or (x_shape[1] == 1)])

    def normalize_input_shape(self, x: np.ndarray) -> np.ndarray:
        """
        Default reshaping to a canonical (l, 1, n) row vector form.
        :param x:
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
        return np.reshape(x, (l, 1, n))

    @abstractmethod
    def function(self, x: np.ndarray):
        pass

    @abstractmethod
    def grad(self, *args, **kwargs) -> Gradient | None:
        """
        Returns a Gradient object that can be called for getting the actual gradient value on a point.
        This allows for better control over a gradient value (e.g., if it is a diagonal matrix etc.).
        If the function is not actually differentiable, this method must return None.
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def __init__(self, function: Callable = None, grad: Callable = None):
        if function is not None:
            self.function = function
        if grad is not None:
            self.grad = grad

    def __call__(self, x: np.ndarray):
        return self.function(x)

    # todo if convenient, overload __add__, __sub__, __mul__ and __matmul__


class Gradient(Callable):
    """
    Base class for gradients of differentiable functions.
    """
    def __init__(self, function: Callable = None, is_sparse: bool = False,
                 is_diagonal: bool = False, is_matricial: bool = False, **kwargs):
        if function is not None:
            self.function = function
        self.is_sparse = is_sparse
        self.is_diagonal = is_diagonal
        self.is_matricial = is_matricial
        if len(kwargs) > 0:
            for arg_name, arg_value in kwargs.items():
                setattr(self, arg_name, arg_value)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.function(x)

    # todo if convenient, overload __add__, __sub__, __mul__ and __matmul__


class ActivationFunction(DifferentiableFunction):
    """
    Base class for activation functions.
    """
    @abstractmethod
    def function(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _gradient(self, x: np.ndarray) -> np.ndarray:
        pass

    # Default implementation
    def grad(self, *args, **kwargs) -> Gradient | None:
        return Gradient(self._gradient, is_diagonal=True)


class Sigmoid(ActivationFunction):

    def function(self, x: np.ndarray) -> np.ndarray:
        x = self.normalize_input_shape(x)
        return 1.0/(1.0 + np.exp(-x))

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        return self.function(x) * (1.0 - self.function(x))


class Sign(ActivationFunction):

    def function(self, x: np.ndarray) -> np.ndarray:
        x = self.normalize_input_shape(x)
        return np.sign(x)

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        x = super(Sign, self).function(x)
        return np.abs(np.sign(x))


class Tanh(ActivationFunction):

    def function(self, x: np.ndarray):
        x = self.normalize_input_shape(x)
        return np.tanh(x)

    def _gradient(self, x: np.ndarray):
        return 1.0 - np.square(self.function(x))


class CustomActivationFunction(ActivationFunction):

    def __init__(self, func: Callable, gradient: Callable):
        super(CustomActivationFunction, self).__init__()
        self.orig_func = func
        self.orig_gradient = gradient

    def function(self, x: np.ndarray):
        x = self.normalize_input_shape(x)
        return np.vectorize(self.orig_func)(x)

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        x = self.normalize_input_shape(x)
        return np.vectorize(self.orig_gradient)(x)


class ReLU(ActivationFunction):

    def function(self, x: np.ndarray):
        x = self.normalize_input_shape(x)
        return np.maximum(x, np.zeros_like(x))

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        x = self.normalize_input_shape(x)
        return np.vectorize(lambda y: 1 if y > 0 else 0)(x)


class Softmax(DifferentiableFunction):

    def __init__(self, const_shift=0, max_shift=False):
        super(Softmax, self).__init__()
        self.const_shift = const_shift
        self.max_shift = max_shift

    def __shift_x(self, x: np.ndarray) -> np.ndarray:
        xc = x.copy()
        if self.const_shift != 0:
            xc += self.const_shift
        if self.max_shift:
            xmax = np.max(xc, axis=len(xc.shape)-1)  # todo check!
            for i in range(xc.shape[0]):
                xc[i] -= xmax[i][0]
        return xc

    def function(self, x: np.ndarray) -> np.ndarray:
        x = self.normalize_input_shape(x)
        xc = self.__shift_x(x)
        xc = np.exp(xc)
        s = np.sum(xc, axis=len(xc.shape)-1)
        for i in range(xc.shape[0]):
            xc[i] /= s[i][0]
        return xc

    def _gradient(self, x: np.ndarray):
        s = self(x)  # (l, 1, n)
        # Calculate the common part
        st = np.transpose(s, axes=[0, 2, 1])  # (l, n, 1)
        y = (-1.0) * (st @ s)  # (l, n, n)
        # Reshape for summing elements on the ("last two") diagonals
        target_shape = y.shape
        u, n = np.prod(target_shape[:-2]), target_shape[-1]
        y = np.reshape(y, (u, n, n))
        s = np.reshape(s, (u, n))
        for i in range(u):
            for j in range(n):
                y[i][j][j] = s[i][j]
        y = np.reshape(y, target_shape)
        return y

    def grad(self, *args, **kwargs) -> Gradient | None:
        return Gradient(self._gradient, is_matricial=True)


class CategoricalCrossEntropy(DifferentiableFunction):

    def __init__(self, target_truth_values: np.ndarray, clip_value: TReal = 1e-7):
        super(CategoricalCrossEntropy, self).__init__()
        target_truth_values = self.normalize_target_shape(target_truth_values)
        self.target_truth_values = target_truth_values
        self.clip_value = clip_value

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
            x_shape[2] > 0 if len(t_shape) == 1 else x_shape[2] == t_shape[2],
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

    def function(self, x: np.ndarray):
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

    def _gradient(self, x: np.ndarray):
        x = self.normalize_input_shape(x)
        x_clip = - 1.0 / np.clip(x, self.clip_value, 1 - self.clip_value)
        trshape = self.target_truth_values.shape
        if len(trshape) == 1:
            # If labels are sparse, turn them into one-hot encoded ones
            self.target_truth_values = np.eye(x.shape[2])[self.target_truth_values]
            self.target_truth_values = np.reshape(self.target_truth_values, (trshape[0], 1, x.shape[2]))
        result = x_clip * self.target_truth_values
        return result

    def grad(self, *args, **kwargs) -> Gradient | None:
        return Gradient(self._gradient)


class SquareError(DifferentiableFunction):
    """
    Calculates square error (||x||^2) for the patterns SEPARATELY.
    """
    def function(self, x: np.ndarray):
        x = self.normalize_input_shape(x)
        y = np.square(np.linalg.norm(x, axis=2))
        return y

    def _gradient(self, x: np.ndarray):
        x = self.normalize_input_shape(x)
        return 2 * x

    def grad(self, *args, **kwargs) -> Gradient | None:
        return Gradient(self._gradient)


class AbsError(DifferentiableFunction):

    def function(self, x: np.ndarray):
        x = self.normalize_input_shape(x)
        y = np.sum(np.abs(x), axis=2)
        return y

    def _gradient(self, x: np.ndarray):
        pass

    def grad(self, *args, **kwargs) -> Gradient | None:
        return Gradient(self._gradient)


__all__ = [
    'DifferentiableFunction',
    'Gradient',
    'ActivationFunction',
    'Sigmoid',
    'Sign',
    'Tanh',
    'ReLU',
    'CustomActivationFunction',
    'Softmax',
    'CategoricalCrossEntropy',
    'SquareError',
    'AbsError',
]