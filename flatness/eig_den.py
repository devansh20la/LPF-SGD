import torch
import torch.nn as nn
from models import resnet18_narrow as resnet18
from functools import reduce
import numpy as np

from utils import HesssianVectorProduct as hvp
from utils import vector_to_parameter_list as v2p
from torch.nn.utils import parameters_to_vector as p2v
from scipy import linalg

from tqdm import tqdm
import h5py
from utils import get_loader

import argparse
from distutils import util


def lanczos(batch_loss, model, d, max_it):
    '''
        Lanczos iteration following the wikipedia article here
            https://en.wikipedia.org/wiki/Lanczos_algorithm
        Inputs:
            batch_loss: f(x)
            model: input model
            d: hessian dimension
            max_it: max iteration
        Outputs:
            eigven values
            weights
    '''
    params_list = list(model.parameters())

    # Initializing empty arrays for storing
    diag = np.zeros(max_it)
    off_diag = np.zeros(max_it - 1)
    V = torch.zeros((d, max_it))

    v = torch.normal(mean=0.0, std=1/d, size=(d, 1))
    v /= torch.norm(v)
    V[:, 0:1] = v

    for k in range(max_it-1):
        v = V[:, k:k+1]

        if torch.cuda.is_available():
            v = v.cuda()

        w = hvp(batch_loss, params_list, v2p(v, model.parameters()))
        w = p2v(w)
        w = w.cpu().reshape(-1, 1)

        v = v.cpu()

        diag[k] = w.T @ v
        w -= diag[k]*v + (off_diag[k-1]*V[:, k-1:k] if k > 0 else 0)

        # Additional step of orthogonalization
        for j in range(k):
            tau = V[:, j:j+1]
            coeff = w.T @ tau
            w = w - coeff * tau

        off_diag[k] = np.linalg.norm(w)

        if off_diag[k] < 100*d*np.finfo(float).eps:
            raise ZeroDivisionError
            quit()

        V[:, k+1:k+2] = w / off_diag[k]

    # complete final iteration
    k = max_it - 1
    v = V[:, k:k+1]

    if torch.cuda.is_available():
        v = v.cuda()

    w = hvp(batch_loss, params_list, v2p(v, model.parameters()))
    w = p2v(w)
    w = w.cpu().reshape(-1, 1)

    v = v.cpu()

    diag[j] = w.T @ v

    l, u = linalg.eigh_tridiagonal(diag, off_diag)
    w = u[0, :]**2

    return (l, w)


def main(args):

    ################# DATA INITIALIZATION ######################
    dset_loaders = get_loader(args, training=True, lp=0.1)

    ############## MODEL INITIALIZATION ###########################
    model = resnet18(args)
    model.load_state_dict(torch.load(f"{args.cp_dir}/trained_model.pth.tar"))
    # model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    # error = {'train': AverageMeter(), 'val': AverageMeter()}
    # for phase in ['train', 'val']:
    #     # computing loss
    #     for inp_data in dset_loaders[phase]:
    #         inputs, targets = inp_data

    #         if args.use_cuda:
    #             inputs, targets = inputs.cuda(), targets.cuda()

    #         with torch.set_grad_enabled(False):
    #             outputs = model(inputs)
    #             err = accuracy(outputs, targets, topk=(1,))
    #             error[phase].update(err[0], inputs.size(0))

    # gen_gap = error['val'] - error['train']

    batch_loss = None

    for batch_idx, inp_data in enumerate(tqdm(dset_loaders['train']), 1):
        inputs, targets = inp_data["img"], inp_data["target"]

        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if batch_loss is not None:
                batch_loss += loss
            else:
                batch_loss = loss

    batch_loss = batch_loss/batch_idx

    d = 0
    for param in model.parameters():
        d += reduce(lambda x, y: x * y, param.shape)

    hf = h5py.File(f"{args.cp_dir}/density.h5", 'w')
    num_draws = 10
    max_it = 100
    eigv = np.zeros((num_draws, max_it))
    weights = np.zeros((num_draws, max_it))

    for i in tqdm(range(num_draws), position=0):
        l, w = lanczos(batch_loss, model, d, max_it)
        eigv[i] = l
        weights[i] = w

    hf.create_dataset("eigv", data=eigv)
    hf.create_dataset("weights", data=weights)
    hf.close()


def none_or_str(value):
    if value == 'None':
        return None
    return value


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-dir', type=str)
    parser.add_argument('-print_freq', type=int)
    parser.add_argument('-dtype', type=str)
    parser.add_argument('-optim', type=str)
    parser.add_argument('-ms', type=int)
    parser.add_argument('-ep', type=int)
    parser.add_argument('-lr', type=float)
    parser.add_argument('-wd', type=float)
    parser.add_argument('-bs', type=int)
    parser.add_argument('-m', type=float)
    parser.add_argument('-regular', type=none_or_str, nargs='?',)
    parser.add_argument('-skip', type=lambda x: bool(util.strtobool(x)))
    parser.add_argument('-lr_decay', type=lambda x: bool(util.strtobool(x)))

    parser.add_argument('-num_classes', type=str)
    parser.add_argument('-data_dir', type=str)
    parser.add_argument('-use_cuda', type=str)
    parser.add_argument('-cp_dir', type=str)

    myargs = []
    with open("checkpoints/mnist/holygrail.txt", 'r') as f:
        header = f.readline().split(',')
        for line in f:
            for value, key in zip(line.split(',')[:-1], header):
                myargs += [f"-{key}", f"{value}"]

            args = parser.parse_args(myargs)
            main(args)
