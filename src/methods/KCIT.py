from causallearn.utils.cit import CIT
import pandas as pd, numpy as np

def KCIT(X, Y, Z, **kargs):
    X, Y, Z = map(lambda x: (x - np.mean(x)) / np.std(x), (X, Y, Z))
    data = np.column_stack((X, Y, Z))
    dz = Z.shape[1]
    kci = CIT(method='kci', data=data)
    p_value = kci(0, 1, list(range(2, 2 + dz)))
    return p_value