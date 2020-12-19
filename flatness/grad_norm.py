import torch
import torch.nn as nn
from models import ResNet18
import numpy as np
import random
from utils import get_loader
import copy
from tqdm import tqdm
from scipy.integrate import quad
import argparse


@torch.no_grad()
def load_weights(model, params):
    for mp, p in zip(model.parameters(), params):
        mp.data = copy.deepcopy(p.data)


def main(args):
    dset_loaders = get_loader(args, training=True)

    # initialize model
    model = ResNet18()
    model.load_state_dict(torch.load(f"{args.cp_dir}/trained_model.pth.tar"))
    if args.use_cuda:
        model = model.cuda()
    model.eval()
    model.norm()

    # intialize loss
    criterion = nn.CrossEntropyLoss()

    # get theta_star
    with torch.no_grad():
        theta_star = [p.data.clone() for p in model.parameters()]

    # compute loss
    loss = 0.0
    with torch.no_grad():
        for inputs, targets in dset_loaders['train']:
            if args.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)
            loss += criterion(outputs, targets) * inputs.shape[0] / len(dset_loaders['train'].dataset)
            # loss.backward()
    print(loss)

    load_weights(model, theta_star)
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for inputs, targets in dset_loaders['train']:
            if args.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)
            loss += criterion(outputs, targets) * inputs.shape[0] / len(dset_loaders['train'].dataset)
            # loss.backward()
    print(loss)

    # with torch.no_grad():
    #     g_norm = 0.0
    #     grads = []
    #     for p in model.parameters():
    #         grads += [copy.deepcopy(p.grad)]
    #         g_norm += torch.norm(grads[-1].reshape(-1))**2
    #     g_norm *= 0.5
    # model.zero_grad()

    # print(g_norm)

    # @torch.no_grad()
    # def func(t):
    #     gamma = 0.1
    #     load_weights(model, theta_star)
    #     model.eval()
    #     model.norm()
        
    #     for mp, g in zip(model.parameters(), grads):
    #         mp.add_(g, alpha=t)
        
    #     entropy_loss = 0.0
    #     for inputs, targets in tqdm(dset_loaders['train']):
    #         if args.use_cuda:
    #             inputs = inputs.cuda()
    #             targets = targets.cuda()
    #         outputs = model(inputs)
    #         entropy_loss += criterion(outputs, targets) * inputs.shape[0] / len(dset_loaders['train'].dataset)
        
    #     tqdm.write(f"{entropy_loss.item()}, {t}")
    #     tqdm.write(f"{torch.exp(-entropy_loss - t*gamma/2 * g_norm).item() * g_norm}")
    #     return torch.exp(-entropy_loss - t*gamma/2 * g_norm) * g_norm

    # I, e = quad(func, -0.1, 0.1, epsabs=1e-2, epsrel=1e-2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--dtype', type=str, default="mnist", help='Data type')

    parser.add_argument('--ms', type=int, default=0, help='manual seed')
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--ln', type=int, help='label noise')

    args = parser.parse_args()

    if args.dtype == 'cifar10':
        args.num_classes = 10
    elif args.dtype == 'mnist':
        args.num_classes = 10
    else:
        print(f"BAD COMMAND dtype: {args.dtype}")

    # Random seed
    random.seed(args.ms)
    torch.manual_seed(args.ms)
    np.random.seed(args.ms)
    args.use_cuda = torch.cuda.is_available()

    if args.use_cuda:
        torch.cuda.manual_seed_all(args.ms)
        torch.backends.cudnn.benchmark = True

    # This is done to run job on cluster with support for array jobs
    if args.ln > 10:
        print("Label noise should be in the range 0 - 10")
        quit()
    else:
        args.ln = float(args.ln) / 10

    args.data_dir = f"{args.dir}/data/{args.dtype}"
    args.use_cuda = torch.cuda.is_available()

    args.n = f"{args.dtype}_label_noise/resnet_label_noise_{args.ln}"

    # Intialize directory and create path
    args.bs = 256
    args.cp_dir = f"{args.dir}/checkpoints/{args.n}/run_ms_{args.ms}"
    main(args)
