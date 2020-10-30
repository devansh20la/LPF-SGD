import torch
import numpy as np
import copy
from tqdm import tqdm


def _entropy(func, theta_star, gamma, mcmc_itr):

    out = []
    for k in range(mcmc_itr):
        theta = theta_star + torch.randn((theta_star.shape)).normal_(0, 1/gamma)
        out += [-func(theta)]

    return -(torch.logsumexp(torch.cat(out), 0, False) + np.log(1/mcmc_itr) + theta_star.shape[0]/2 * np.log(2*np.pi) - np.log(gamma**0.5))


def load_weights(model, params):
    for mp, p in zip(model.parameters(), params):
        mp.data = copy.deepcopy(p.data)


def entropy(model_func, gamma, mcmc_itr):
    with torch.no_grad():
        theta_star = [p.data.clone() for p in model_func.model.parameters()]

    out = []
    for k in tqdm(range(mcmc_itr)):
        for mp, p in zip(model_func.model.parameters(), theta_star):
            mp.data.copy_(p + torch.zeros(p.shape, device=mp.data.device).normal_(0, 1/gamma))
        out += [torch.Tensor([model_func.compute_loss()[0]])]

    load_weights(model_func.model, theta_star)
    return -(torch.logsumexp(torch.cat(out), 0, False) + np.log(1/mcmc_itr) + model_func.dim/2 * np.log(2*np.pi) - np.log(gamma**0.5)).item()
