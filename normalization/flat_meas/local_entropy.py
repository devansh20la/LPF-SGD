import torch
import numpy as np


def entropy(func, theta_star, gamma, mcmc_itr):

    out = []
    for k in range(mcmc_itr):
        theta = theta_star + torch.randn((theta_star.shape)).normal_(0, 1/gamma)
        out += [-func(theta)]

    return -(torch.logsumexp(torch.cat(out), 0, False) + np.log(1/mcmc_itr) + theta_star.shape[0]/2 * np.log(2*np.pi) - np.log(gamma**0.5))


def entropy2(func, theta_star, gamma, mcmc_itr, eps, tol=1e-6, verbose=False):

    base_loss = func(theta_star)
    sigma = 0.001
    sigma_range = [-10**10, 10**10]

    # print("Estimating sigma")
    for itr in range(10**7):
        curr_loss = 0.0
        for k in range(mcmc_itr):
            theta = theta_star + torch.randn((theta_star.shape)).uniform_(-sigma, sigma)
            curr_loss += func(theta)
        curr_loss /= mcmc_itr

        d = curr_loss - base_loss

        if itr % 100 == 0 and verbose:
            print(f"itr:{itr:2.3E}, sigma:{sigma:2.3E},d:{d.item():2.3E}, base_loss:{base_loss.item():2.3E}, curr_loss:{func(theta).item():2.3E}")

        if eps - tol <= d <= eps + tol or np.abs(sigma_range[0] - sigma_range[1]) <tol:
            break
        elif d < eps - tol:
            sigma_range[0] = sigma
        else:
            sigma_range[1] = sigma
        sigma = np.mean(sigma_range)

    # print(sigma, "Estimating entropy")
    out = []
    for k in range(mcmc_itr):
        theta = theta_star + torch.randn((theta_star.shape)).uniform_(-sigma, sigma)
        out += [-func(theta) - gamma/2 * torch.norm(theta - theta_star, 2)**2]

    return -(torch.logsumexp(torch.cat(out), 0, False) + np.log(1/mcmc_itr) + np.log(2*sigma)*theta_star.shape[0])
