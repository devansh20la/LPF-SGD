import numpy as np


def fro_norm(func, dim, mcmc_itr):
    out = 0.0
    for i in range(mcmc_itr):
        v = np.random.normal(loc=0.0, scale=1.0, size=(dim, 1))
        out += np.linalg.norm(func.hvp(v))**2

    return out/mcmc_itr
