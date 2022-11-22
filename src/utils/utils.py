import contextlib, time
from typing import NamedTuple
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


def strip_outliers(x):
    l, r = np.quantile(x, q=[0.025, 0.975], axis=0)
    x = np.clip(x, l, r)
    return x