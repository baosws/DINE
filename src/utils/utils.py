import contextlib, time
from socket import gethostname
from typing import NamedTuple
from itertools import chain, combinations
import numpy as np
import yaml

def read_config(path):
    with open(path, 'r') as f:
        params = yaml.safe_load(f) or {}
    return params

@contextlib.contextmanager
def Timer(name=None, verbose=False):
    start = time.time()
    timer = NamedTuple('timer', elapsed=str)
    yield timer
    timer.elapsed = time.time() - start
    if verbose:
        print(f'{name:<6}: {timer.elapsed:.3f}s')

def CIT_wrapper(cit, **kwargs):
    def func(dm, i, j, k):
        k = list(k)
        X = dm[:, i]
        Y = dm[:, j]
        Z = dm[:, k]
        return cit(X, Y, Z, **kwargs)
    setattr(func, '__name__', cit.__name__)
    return func

def power_set(d):
    return chain.from_iterable(combinations(range(d), k) for k in range(1, d + 1))

def is_on_server():
    servers = [
        'login',
        'cpu-01',
        'cpu-02',
        'cpu-03',
        'gamling-4',
        'gamling-5',
        'gamling-6',
        'gamling-7',
    ]
    return gethostname() in servers

def strip_outliers(x):
    l, r = np.quantile(x, q=[0.025, 0.975], axis=0)
    x = np.clip(x, l, r)
    return x