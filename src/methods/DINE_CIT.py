from .DINE import DINE
import numpy as np

def DINE_CIT(X, Y, Z, n_bootstraps=20, **params):
    assert X.shape[1] == Y.shape[1] == 1
    rng = np.random.RandomState(seed=params.get('random_state', 0))
    e_x, e_y, mi = DINE(X=X, Y=Y, Z=Z, return_latents=True, **params)

    cov_x = np.cov(e_x.T).reshape(e_x.shape[1], e_x.shape[1])
    count = 0
    for i in range(n_bootstraps):
        idx = rng.permutation(e_y.shape[0])
        cov_y = np.cov(e_y[idx].T).reshape(e_y.shape[1], e_y.shape[1])
        cov_all = np.cov(np.column_stack((e_x, e_y[idx])).T)
        perm_mi = 0.5 * (np.log(np.linalg.det(cov_x)) + np.log(np.linalg.det(cov_y)) - np.log(np.linalg.det(cov_all)))
        count += mi <= perm_mi
    
    p_value = count / n_bootstraps
    return p_value