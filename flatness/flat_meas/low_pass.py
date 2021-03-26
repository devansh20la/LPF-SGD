import torch
import numpy as np
import copy
from tqdm import tqdm


def _low_pass(func, theta_star, mcmc_itr, sigma):
    out = 0.0
    d = theta_star.shape[0]
    for i in range(mcmc_itr):
        theta = theta_star + torch.from_numpy(np.random.multivariate_normal(np.zeros((d)), (sigma**2)*np.eye(d))).type(torch.float32).reshape(-1,1)
        out += func(theta)

    return out/mcmc_itr


def load_weights(model, params):
    for mp, p in zip(model.parameters(), params):
        mp.data = copy.deepcopy(p.data)

def low_pass(model_func, sigma, mcmc_itr):
    out = 0.0
    with torch.no_grad():
        theta_star = [p.data.clone() for p in model_func.model.parameters()]
    for i in tqdm(range(mcmc_itr)):

        for mp, p in zip(model_func.model.parameters(), theta_star):
            mp.data.copy_(p + torch.zeros(p.shape, device=mp.data.device).normal_(0, sigma))

        out += model_func.compute_loss()[0]

    load_weights(model_func.model, theta_star)
    return out/mcmc_itr

def low_pass2(model_func, sigma, mcmc_itr):
    out = 0.0
    with torch.no_grad():
        theta_star = [p.data.clone() for p in model_func.model.parameters()]
    for i in tqdm(range(mcmc_itr)):

        for mp, p in zip(model_func.model.parameters(), theta_star):
            mp.data.copy_(p + torch.zeros(p.shape, device=mp.data.device).normal_(0, sigma))

        out += np.abs(model_func.compute_loss()[0] - model_func.train_loss) / model_func.train_loss

    load_weights(model_func.model, theta_star)
    return out/mcmc_itr