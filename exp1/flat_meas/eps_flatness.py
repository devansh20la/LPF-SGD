import torch
import numpy as np
import torch.optim as optim
from utils.train_utils import AverageMeter
import copy
import sys
import logging


def load_weights(model, params, grads):
    for mp, p, g in zip(model.parameters(), params, grads):
        mp.data = copy.deepcopy(p.data)
        mp.grad.data = copy.deepcopy(g)


def eps_flatness(model_func, epsilon, tol=1e-6, use_cuda=False, verbose=False):
    logger = logging.getLogger('my_log')
    r'''
        Function to compute eps flatness similar to description in keskar
        Inputs:
            model_func: model class that contains relevant functions
            epsilon   : loss deviation
            tolerance : tolerance on the loss deviation
            use_cuda  : use cuda
            verbose   : logger.info details
        Output:
            return sharpness measure 1/alpha
    '''
    logger.info("......... Computing base loss and gradients .........")
    theta_star_params, theta_star_grads, \
        base_loss, _ = model_func.compute_loss(ascent_stats=True)

    eta_range = [1e-16, 1]

    logger.info("........ Computing eta_max .........")
    for itr in range(10**7):
        eta = eta_range[1]
        optimizer = optim.SGD(model_func.model.parameters(), eta, 0.0, 0.0)
        optimizer.step()

        curr_loss, _ = model_func.compute_loss()

        d = curr_loss - base_loss
        if d < epsilon:
            eta_range[1] = eta*5
        else:
            load_weights(model_func.model, theta_star_params, theta_star_grads)
            break

        if itr % 10 == 0 and verbose:
            logger.info(f"{itr:2.3E}, {eta:2.3E}, {eta_range[0]:2.3E}, {eta_range[1]:2.3E}, {d:2.3E}, {base_loss:2.3E}, {curr_loss:2.3E}")
        load_weights(model_func.model, theta_star_params, theta_star_grads)

    # we need norm of the theta_star_grads as per the simplification
    flatness = 0.0
    for p in theta_star_grads:
        param_norm = p.data.view(-1).norm(2)
        flatness += param_norm.item() ** 2
    flatness = flatness ** (0.5)

    logger.info(f"eta_max found: {eta_range[1]:.6E}")
    logger.info("Computing eta")
    logger.info(f"{'Itr':^10} {'eta':^10} {'eta_min':^10} {'eta_max':^10} {'d':^10} {'base_loss':^10} {'curr_loss':^10}")
    for itr in range(10**7):
        eta = np.mean(eta_range)
        optimizer = optim.SGD(model_func.model.parameters(), eta, 0.0, 0.0)
        optimizer.step()

        curr_loss, _ = model_func.compute_loss()

        d = curr_loss - base_loss
        if (epsilon - tol <= d <= epsilon + tol) or ((eta_range[1] - eta_range[0]) < 0) or (np.abs(eta_range[1] - eta_range[0]) < tol):
            logger.info(f"{itr:2.3E}, {eta:2.3E}, {eta_range[0]:2.6E}, {eta_range[1]:2.6E}, {d:2.3E}, {base_loss:2.3E}, {curr_loss:2.3E}")
            load_weights(model_func.model, theta_star_params, theta_star_grads)
            model_func.model.zero_grad()
            return 1/(np.abs(eta)*flatness)
        elif d < epsilon - tol:
            eta_range[0] = eta
        else:
            eta_range[1] = eta

        if itr % 1 == 0 and verbose:
            logger.info(f"{itr:2.3E}, {eta:2.3E}, {eta_range[0]:2.6E}, {eta_range[1]:2.6E}, {d:2.3E}, {base_loss:2.3E}, {curr_loss:2.3E}")
        load_weights(model_func.model, theta_star_params, theta_star_grads)


def _eps_flatness(func, theta_star, grad, eps, tol=1e-6, verbose=False):
    logger = logging.getLogger('my_log')
    if verbose:
        logger.info("Computing epsilon flatness")

    base_loss = func(theta_star)
    eta = 0.001
    eta_range = [1e-20, 10**128]

    for itr in range(10**7):
        theta = theta_star + eta * grad

        d = func(theta) - base_loss

        if eps - tol <= d <= eps + tol:
            if verbose:
                logger.info(f"eta found:{eta}")
            return 1/torch.norm(theta - theta_star, 2)

        elif d < eps - tol:
            eta_range[0] = eta
        else:
            eta_range[1] = eta

        eta = np.mean(eta_range)
        if itr % 100 == 0 and verbose:
            logger.info(f"itr:{itr:2.3E}, eta:{eta:2.3E},d:{d.item():2.3E}, base_loss:{base_loss.item():2.3E}, curr_loss:{func(theta).item():2.3E}")

    logger.info("Max iteration reached")
    return 1/torch.norm(theta - theta_star, 2)
