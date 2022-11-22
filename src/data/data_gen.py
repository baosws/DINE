from itertools import product
import numpy as np
from numpy.linalg import det

def linear(h, rng):
    h = h / np.std(h)
    a = rng.uniform(0.2, 2)
    return a * h

def cube(X, rng=None):
    X = X / np.std(X)
    return X ** 3

def inverse(X, rng=None):
    X = X / np.std(X)
    return 1 / (X - np.min(X) + 1)

def nexp(X, rng=None):
    X = X / np.std(X)
    return np.exp(-X)

def log(X, rng=None):
    X = X / np.std(X)
    return np.log1p(X - np.min(X))
    
def sigmoid(X, rng=None):
    X = X / np.std(X)
    return 1 / (1 + np.exp(-X))

EPS = 1e-10
def simulate_mi(N, d, rho=None, random_state=None, **kwargs):
    rng = np.random.RandomState(random_state)
    if rho is None:
        rho = rng.uniform(-1 + EPS, 1 - EPS)
    cov = np.eye(2 * d)
    cov[:d, d:] = cov[d:, :d] = np.diag([rho] * d)
    D = rng.multivariate_normal(mean=np.zeros(2 * d), cov=cov, size=N)
    X, Y = D[:, :d], D[:, d:]
    mi = -0.5 * np.log(det(cov))
    funcs = [linear, cube, nexp, log, inverse, sigmoid]
    funcs = list(product(funcs, repeat=2))
    f1, f2 = funcs[rng.choice(range(len(funcs)))]
    X = f1(X, rng)
    Y = f2(Y, rng)
    Z = np.empty((N, 0))
    return X, Y, Z, mi

def simulate_cmi(N, d, dz, rho=None, noise_coeff=0.01, random_state=None, **kwargs):
    rng = np.random.RandomState(random_state)
    if rho is None:
        rho = rng.uniform(-1 + EPS, 1 - EPS)
    def gaussian_noise(*size):
        return rng.randn(*size) * noise_coeff
    def uniform_noise(*size):
        return rng.uniform(-1, 1, size=size) * noise_coeff
    def laplace_noise(*size):
        return rng.laplace(size=size) * noise_coeff

    cov = np.eye(2 * d)
    cov[:d, d:] = cov[d:, :d] = np.diag([rho] * d)
    D = rng.multivariate_normal(mean=np.zeros(2 * d), cov=cov, size=N)
    X, Y = D[:, :d], D[:, d:]
    cmi = -0.5 * np.log(det(cov))

    noises = [uniform_noise, gaussian_noise, laplace_noise]
    funcs = [linear, cube, nexp, log, sigmoid]
    funcs = list(product(funcs, repeat=2))
    f1, f2 = funcs[rng.choice(range(len(funcs)))]
    a, b = rng.randn(2, dz, d)

    Z = noises[rng.randint(len(noises))](N, dz)
    X = f1(Z @ a + X, rng)
    Y = f2(Z @ b + Y, rng)

    return X, Y, Z, cmi

def simulate_cit(N, d, dz, indep=None, random_state=None, **kwargs):
    rng = np.random.RandomState(random_state)
    if indep is None:
        indep = rng.randint(2)
    rho = 0 if indep else rng.uniform(low=0.1, high=0.99)
    if rng.randint(2):
        rho = -rho
    X, Y, Z, cmi = simulate_cmi(N=N, d=d, dz=dz, rho=rho, random_state=random_state)
    return X, Y, Z, indep