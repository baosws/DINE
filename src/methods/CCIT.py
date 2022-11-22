from CCIT.CCIT import CCIT as CCIT_
import numpy as np

def CCIT(X, Y, Z, n_bootstraps=20, normalize=True, nthread=8, **kwargs):
    N, dz = Z.shape
    X = X.reshape(N, -1)
    Y = Y.reshape(N, -1)
    if normalize:
        X, Y, Z = map(lambda x: (x - np.mean(x)) / np.std(x), (X, Y, Z))
    p_value = CCIT_(X, Y, Z, num_iter=n_bootstraps, bootstrap=False, nthread=nthread)
    return p_value