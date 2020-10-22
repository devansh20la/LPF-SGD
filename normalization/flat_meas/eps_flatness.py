import torch
import numpy as np
import torch.optim as optim
from utils.train_utils import AverageMeter
import copy
import sys


def load_params(model, parameters, grads):
    for m, p, g in zip(model.parameters(), parameters,  grads):
        m.data = p
        m.grad.data = g


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

    return [param.grads for param in model.parameters()], loss_mtr.avg


def eps_flatness_model(model, criterion, data_loader, epsilon,
                       tol=1e-6, use_cuda=False):
    theta_star = copy.deepcopy(model.parameters())
    grads, base_loss = compute_gradient(model, data_loader, criterion, use_cuda)
    flatness = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        flatness += param_norm.item() ** 2
    flatness = flatness ** (0.5)

    eta = 0.001
    eta_range = [np.nextafter(0, 1), sys.maxsize]

    while(True):
        load_params(model, theta_star, grads)

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
