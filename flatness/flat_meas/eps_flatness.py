import torch
import numpy as np
import torch.optim as optim
from utils.train_utils import AverageMeter
import copy
import sys


def load_weights(model, params, grads):
    for mp, p, g in zip(model.parameters(), params, grads):
        mp.data = copy.deepcopy(p.data)
        mp.grad.data = copy.deepcopy(g)


def compute_gradient(model, data_loader, criterion, use_cuda):
    model.eval()
    loss_mtr = AverageMeter()

    for inp_data in data_loader:
        inputs, targets = inp_data

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)
            loss_mtr.update(batch_loss.item(), inputs.size(0))
            batch_loss = -1*batch_loss * inputs.size(0) / len(data_loader.dataset)
            batch_loss.backward()

    # copy.deepcopy does not copy gradients so we have to do it separately
    theta_star_params = []
    theta_star_grads = []
    for n, p in model.named_parameters():
        theta_star_params.append(copy.deepcopy(p))
        theta_star_grads.append(copy.deepcopy(p.grad))

    return theta_star_params, theta_star_grads, loss_mtr.avg


def eps_flatness_model(model, criterion, data_loader, epsilon,
                       tol=1e-6, use_cuda=False, verbose=False):
    print("Computing base loss and gradients")
    theta_star_params, theta_star_grads, \
        base_loss = compute_gradient(model, data_loader, criterion, use_cuda)
    eta_range = [1e-16, 100]

    flatness = 0.0
    for p in theta_star_grads:
        param_norm = p.data.view(-1).norm(2)
        flatness += param_norm.item() ** 2
    flatness = flatness ** (0.5)

    print("Computing eta_max")
    # first compute eta max
    for itr in range(10**7):

        load_weights(model, theta_star_params, theta_star_grads)

        eta = eta_range[1]

        optimizer = optim.SGD(model.parameters(), eta, 0.0, 0.0)
        print(theta_star_params[0])
        optimizer.step()
        print(theta_star_params[0])
        quit()
        # for p1, p2 in zip(model.parameters(), theta_star_grads):
        #     print((p1.grad.view(-1) - p2.view(-1)).norm())
        curr_loss = 0.0
        for inp_data in data_loader:
            inputs, targets = inp_data
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets) * inputs.size(0) / len(data_loader.dataset)
                curr_loss += batch_loss.item()

        d = curr_loss - base_loss
        if d < epsilon:
            eta_range[1] = eta*1000
        else:
            break

        if itr % 1 == 0 and verbose:
            print(f"itr:{itr:2.3E}, eta:{eta:2.6E}, eta_min:{eta_range[0]:2.6E}, eta_max:{eta_range[1]:2.6E}, d:{d:2.6E}")

    print(f"eta_max found: {eta_range[1]:.6E}")
    print("Computing eta")
    for itr in range(10**7):
        load_weights(model, theta_star_params, theta_star_grads)

        eta = np.mean(eta_range)
        optimizer = optim.SGD(model.parameters(), eta, 0.0, 0.0)
        optimizer.step()

        curr_loss = 0.0
        for inp_data in data_loader:
            inputs, targets = inp_data

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets) * inputs.size(0) / len(data_loader.dataset)
                curr_loss += batch_loss.item()

        d = curr_loss - base_loss

        if epsilon - tol <= d <= epsilon + tol:
            return eta*flatness
        elif d < epsilon - tol:
            eta_range[0] = eta
        else:
            eta_range[1] = eta

        if itr % 10 == 0 and verbose:
            print(f"itr:{itr:2.3E}, eta:{eta:2.6E}, eta_min:{eta_range[0]:2.6E}, eta_max:{eta_range[1]:2.6E}, d:{d:2.6E}, base_loss:{base_loss:2.6E}, curr_loss:{curr_loss:2.6E}")


def eps_flatness(func, theta_star, grad, eps, tol=1e-6, verbose=False):
    if verbose:
        print("Computing epsilon flatness")

    base_loss = func(theta_star)
    eta = 0.001
    eta_range = [1e-20, 10**128]

    for itr in range(10**7):
        theta = theta_star + eta * grad

        d = func(theta) - base_loss

        if eps - tol <= d <= eps + tol:
            if verbose:
                print(f"eta found:{eta}")
            return 1/torch.norm(theta - theta_star, 2)

        elif d < eps - tol:
            eta_range[0] = eta
        else:
            eta_range[1] = eta

        eta = np.mean(eta_range)
        if itr % 100 == 0 and verbose:
            print(f"itr:{itr:2.3E}, eta:{eta:2.3E},d:{d.item():2.3E}, base_loss:{base_loss.item():2.3E}, curr_loss:{func(theta).item():2.3E}")

    print("Max iteration reached")
    return 1/torch.norm(theta - theta_star, 2)
