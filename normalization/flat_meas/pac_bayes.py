import torch
import numpy as np


def pac_bayes(func, theta_init, theta_star, mcmc_itr, data_size, eps, tol=1e-6, verbose=False):

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
