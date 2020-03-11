import argparse
import os
import torch


def get_args(*args):

    parser = argparse.ArgumentParser()

    parser.add_argument('--n', type=str, default=None)
    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--print_freq', type=int, default=50)

    parser.add_argument('--dtype', type=str, required=True, help='Data type')

    parser.add_argument('--ms', type=int, default=5)
    parser.add_argument('--ep', type=int, default=250, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--m', type=float, default=0.9, help='momentun')

    parser.add_argument('--saved_model', type=str, required=False)

    args = parser.parse_args(*args)

    if args.dtype == 'cifar10':
        args.num_classes = 10
        args.milestones = [150, 250]
        args.data_dir = os.path.join(args.dir, 'data', 'cifar')

    elif args.dtype == 'cifar100':
        args.num_classes = 100
        args.milestones = [150, 250]
        args.data_dir = os.path.join(args.dir, 'data', 'cifar')

    elif args.dtype == 'imagenet':
        args.num_classes = 1000
        args.milestones = [30, 60, 90]
        args.data_dir = os.path.join(args.dir, 'data', 'imagenet')

    elif args.dtype == 'mnist':
        args.num_classes = 10
        args.data_dir = os.path.join(args.dir, 'data', 'mnist')

    args.use_cuda = torch.cuda.is_available()
    if args.n is None:
        args.n = args.dtype
    return args


if __name__ == '__main__':
    args = get_args()
