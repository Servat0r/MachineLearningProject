# A couple of utilities for testing speed of functions/methods on CPU and GPU
from time import perf_counter, perf_counter_ns


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


__all__ = [
    'timeit',
    'timeit_ns',
]
