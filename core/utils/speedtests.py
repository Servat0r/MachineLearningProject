# A couple of utilities for testing speed of functions/methods on CPU and GPU
from .types import *
from time import perf_counter, perf_counter_ns
from cupyx.profiler import benchmark as gpu_benchmark


def timeit(func):
    def new_f(*args, **kwargs):
        crt = perf_counter()
        res = func(*args, **kwargs)
        crt = perf_counter() - crt
        print(f"Elapsed time for {func.__name__}: {crt} seconds")
        return res
    return new_f


def timeit_ns(func):
    def new_f(*args, **kwargs):
        crt = perf_counter_ns()
        res = func(*args, **kwargs)
        crt = perf_counter_ns() - crt
        print(f"Elapsed time for {func.__name__}: {crt} seconds")   # todo nanoseconds?
        return res
    return new_f


# noinspection PyDefaultArgument
def cpu_benchmark(func: Callable, args: tuple = (), kwargs: dict = {}, n_repeat: int = 1000):
    times = np.zeros(n_repeat)
    for i in range(n_repeat):
        times[i] = perf_counter()
        func(*args, **kwargs)
        times[i] = perf_counter() - times[i]
    mean, std, mint, maxt = np.mean(times), np.std(times), np.min(times), np.max(times)
    print(f'CPU Time: {mean} +- {std} (min: {mint}, max: {maxt})')


__all__ = [
    'timeit',
    'timeit_ns',
    'cpu_benchmark',
    'gpu_benchmark',
]
