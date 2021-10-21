import torch
import numpy as np
import copy
from tqdm import tqdm
import sys; sys.path.append('..')
from utils.train_utils import AverageMeter, accuracy


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


def low_pass2(model_func, sigma, mcmc_itr):
    loss = AverageMeter()

    for inp_data in model_func.dataloader['train']:
        inputs, targets = inp_data

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                noise = []
                for mp in model_func.model.parameters():
                    temp = torch.empty_like(mp, device=mp.data.device)
                    temp.normal_(0, sigma)
                    noise.append(temp)
                    mp.data.add_(noise[-1])

                batch_loss = model_func.criterion(model_func.model(inputs), targets)
                loss.update(batch_loss.item(), inputs.size(0))

                for mp, n in zip(model_func.model.parameters(), noise):
                    mp.data.sub_(n)
    return loss.avg


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
