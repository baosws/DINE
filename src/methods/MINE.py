from .Baselines import method_wrapper

def MINE(X, Y, Z, **kwargs):
    return method_wrapper('MINE', X, Y, Z, **kwargs)