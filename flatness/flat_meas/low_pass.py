import torch
import numpy as np


def low_pass(func, theta_star, mcmc_itr, sigma):
    out = 0.0
    d = theta_star.shape[0]
    for i in range(mcmc_itr):
        theta = theta_star + torch.from_numpy(np.random.multivariate_normal(np.zeros((d)), (sigma**2)*np.eye(d))).type(torch.float64).reshape(-1,1)
        out += func(theta)

    return out/mcmc_itr
