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
    parser.add_argument('--print_freq', type=int, default=50)

    parser.add_argument('--dtype', type=str, default="cifar10", help='Data type')

    parser.add_argument('--optim', type=str, default="sgd")
    parser.add_argument('--ms', type=int, default=5)
    parser.add_argument('--ep', type=int, default=250, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')

    # regularization parameters
    parser.add_argument('--regular', type=none_or_str, nargs='?',default="batch_norm")
    parser.add_argument('--skip', type=lambda x: bool(util.strtobool(x)), default=True)
    parser.add_argument('--lr_decay', type=lambda x: bool(util.strtobool(x)), default=True)

    parser.add_argument('-hg', type=str)

    args = parser.parse_args(*args)

    if args.dtype == 'cifar10':
        args.num_classes = 10

    elif args.dtype == 'cifar100':
        args.num_classes = 100

    elif args.dtype == 'imagenet':
        args.num_classes = 1000

    elif args.dtype == 'mnist':
        args.num_classes = 10

    args.data_dir = f"{args.dir}/data/{args.dtype}"
    args.use_cuda = torch.cuda.is_available()

    args.n = f"{args.dtype}"

    return args


if __name__ == '__main__':
    args = get_args(fromfile_prefix_chars='@checkpoints/mnist/holygrail.txt')
