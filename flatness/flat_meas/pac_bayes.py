import torch
import numpy as np
import copy
from utils.train_utils import AverageMeter
from utils import vector_to_parameter_tuple
import time


def load_weights(model, params):
    for mp, p in zip(model.parameters(), params):
        mp.data = copy.deepcopy(p.data)


def pac_bayes(model_func, mcmc_itr, eps, theta_init=False, tol=1e-6, verbose=False):
    with torch.no_grad():
        theta_star = [p.data.clone() for p in model_func.model.parameters()]

    if theta_init is not True:
        theta_init = [torch.zeros(p.shape).to(p.device) for p in model_func.model.parameters()]

    base_loss = model_func.train_loss
    sigma = 0.001
    sigma_range = [np.nextafter(0, 1), 10**2]

    print(f"{'Itr':^10} {'sigma':^10} {'sigma_min':^10} {'sigma_max':^10} {'d':^10} {'base_loss':^10} {'curr_loss':^10}")
    for itr in range(10**7):
        curr_loss = 0.0

        for k in range(mcmc_itr):
            t = time.time()
            for mp, p in zip(model_func.model.parameters(), theta_star):
                mp.data.copy_(p + torch.zeros(p.shape, device=mp.data.device).normal_(0, sigma))
            curr_loss += model_func.compute_loss()[0]

        curr_loss /= mcmc_itr

        d = curr_loss - base_loss

        if itr % 1 == 0 and verbose:
            print(f"{itr:2.3E}, {sigma:2.3E}, {sigma_range[0]:2.3E}, {sigma_range[1]:2.3E}, {d:2.3E}, {base_loss:2.3E}, {curr_loss:2.3E}")

        if (eps - tol <= d <= eps + tol) or ((sigma_range[1] - sigma_range[0]) < 0) or (np.abs(sigma_range[1] - sigma_range[0]) < tol):
            break
        elif d < eps - tol:
            sigma_range[0] = sigma
        else:
            sigma_range[1] = sigma
        sigma = np.mean(sigma_range)
    if verbose:
        print(f"Sigma found:{sigma}")

    param_norm_sq = 0.0
    for init, star in zip(theta_init, theta_star):
        param_norm_sq += (init.view(-1) - star.view(-1)).norm().item()**2

    load_weights(model_func.model, theta_star)
    return param_norm_sq / (4 * (sigma**2)) + np.log(2*len(model_func.dataloader['train'].dataset) / 1e-6)


def _pac_bayes(func, theta_init, theta_star, mcmc_itr, data_size, eps, tol=1e-6, verbose=False):

    if verbose:
        print("Computing PAC bayes bound")

    base_loss = func(theta_star)
    sigma = 0.001
    sigma_range = [np.nextafter(0, 1), 10**20]

    for itr in range(10**7):
        curr_loss = 0.0
        for k in range(mcmc_itr):
            theta = theta_star + torch.randn((theta_star.shape)).normal_(0, sigma)
            curr_loss += func(theta)
        curr_loss /= mcmc_itr

        d = curr_loss - base_loss

        if itr % 100 == 0 and verbose:
            print(f"itr:{itr:2.3E}, sigma:{sigma:2.3E},d:{d.item():2.3E}, base_loss:{base_loss.item():2.3E}, curr_loss:{func(theta).item():2.3E}")

        if eps - tol <= d <= eps + tol or np.abs(sigma_range[0] - sigma_range[1]) < tol:
            break
        elif d < eps - tol:
            sigma_range[0] = sigma
        else:
            sigma_range[1] = sigma
        sigma = np.mean(sigma_range)
    if verbose:
        print(f"Sigma found:{sigma}")

    return torch.norm(theta_star - theta_init, 2)**2 / (4 * (sigma**2)) + np.log(2*data_size / 1e-6)
