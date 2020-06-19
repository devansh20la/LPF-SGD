import argparse
import os
import torch
from distutils import util

# from distutils.utils import strtobool


def none_or_str(value):
    if value == 'None':
        return None
    return value


def get_args(*args):

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--print_freq', type=int, default=500)
    # parser.add_argument('--ms', type=int, default=5)

    parser.add_argument('--dtype', type=str, default="cifar10", help='Data type')

    parser.add_argument('--optim', type=str, default="sgd")
    parser.add_argument('--ep', type=int, default=250, help='Epochs')
    parser.add_argument('--exp_num', type=int, required=True)

    # parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    # parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    # parser.add_argument('--bs', type=int, default=128, help='batch size')
    # parser.add_argument('--m', type=float, default=0.9, help='momentum')

    # regularization parameters
    # parser.add_argument('--regular', type=none_or_str, nargs='?', default="batch_norm")
    # parser.add_argument('--skip', type=lambda x: bool(util.strtobool(x)), default=True)
    # parser.add_argument('--lr_decay', type=lambda x: bool(util.strtobool(x)), default=True)

    args = parser.parse_args(*args)

    from sklearn.model_selection import ParameterGrid
    param_grid = {'mo': [0.9],  # momentum
                  'ms': [0],  # seed
                  'width': [4, 6, 8],  # network width
                  'wd': [1e-3, 1e-2, 0.0],  # weight decay
                  'lr': [5e-2, 1e-2, 2e-3],  # learning rate
                  'bs': [32, 64, 128],  # batch size
                  'lr_decay': ["True", "False"],  # learning rate decay
                  'skip': ["True", "False"],  # skip connection
                  'regular': ["batch_norm", "dropout", "none"]  #regularization
                  }

    grid = list(ParameterGrid(param_grid))

    params = grid[args.exp_num]

    args.width = params['width']
    args.mo = params['mo']
    args.wd = params['wd']
    args.ms = params['ms']
    args.lr_decay = params['lr_decay']
    args.lr = params['lr']
    args.bs = params['bs']
    args.skip = params['skip']
    args.regular = params['regular']

    if args.dtype == 'cifar10':
        args.num_classes = 10

    elif args.dtype == 'cifar100':
        args.num_classes = 100

    elif args.dtype == 'imagenet':
        args.num_classes = 1000

    elif args.dtype == 'mnist':
        args.num_classes = 10
    else:
        print(f"BAD COMMAND dtype: {args.dtype}")

    if args.regular not in ["batch_norm", "dropout", "none"]:
        print(f"BAD COMMAND regular: {args.regular}")

    args.data_dir = f"{args.dir}/data/{args.dtype}"
    args.use_cuda = torch.cuda.is_available()

    args.n = f"large_resnet_{args.dtype}/" \
             f"{args.width}_{args.regular}_{args.skip}_" \
             f"{args.optim}_{args.ep}_{args.lr}_{args.wd}_" \
             f"{args.bs}_{args.mo}_{args.lr_decay}"

    return args


if __name__ == '__main__':
    args = get_args()
