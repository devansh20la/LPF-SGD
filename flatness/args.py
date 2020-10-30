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
    parser.add_argument('--ep', type=int, default=500, help='Epochs')
    parser.add_argument('--exp_num', type=int, required=True)

    # parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    # parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    # parser.add_argument('--bs', type=int, default=128, help='batch size')
    # parser.add_argument('--m', type=float, default=0.9, help='momentum')

    # regularization parameters
    # parser.add_argument('--regular', type=none_or_str, nargs='?', default="batchnorm")
    # parser.add_argument('--skip', type=lambda x: bool(util.strtobool(x)), default=True)
    # parser.add_argument('--lr_decay', type=lambda x: bool(util.strtobool(x)), default=True)

    args = parser.parse_args(*args)

    from sklearn.model_selection import ParameterGrid
    param_grid = {'ms': [0],  # seed
                  'mo': [0.0, 0.5, 0.9],  # momentum
                  # 'width': [4, 6, 8],  # network width
                  'wd': [0.0, 1e-4, 5e-4],  # weight decay
                  'lr': [7e-3, 0.0085, 1e-2],  # learning rate
                  'bs': [32, 128, 512],  # batch size
                  'lr_decay': [False],  # learning rate decay
                  # 'skip': [True, False],  # skip connection
                  # 'batchnorm': [True, False]  # batchnorm
                  }

    grid = list(ParameterGrid(param_grid))

    params = grid[args.exp_num]

    args.ms = params['ms']
    args.mo = params['mo']
    # args.width = params['width']
    args.wd = params['wd']
    args.lr = params['lr']
    args.bs = params['bs']
    args.lr_decay = params['lr_decay']
    # args.skip = params['skip']
    # args.batchnorm = params['batchnorm']

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

    args.data_dir = f"{args.dir}/data/{args.dtype}"
    args.use_cuda = torch.cuda.is_available()

    # args.n = f"{args.dtype}/" \
    #          f"{args.exp_num}_{args.width}_{args.batchnorm}_{args.skip}_" \
    #          f"{args.optim}_{args.ep}_{args.lr}_{args.wd}_" \
    #          f"{args.bs}_{args.mo}_{args.lr_decay}"

    args.n = f"{args.dtype}/lenet/" \
             f"{args.exp_num}_{args.ms}_{args.mo}_{args.wd}" \
             f"_{args.lr}_{args.bs}_{args.lr_decay}"

    return args


if __name__ == '__main__':
    args = get_args(["--exp_num", "123"])
