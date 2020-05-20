import argparse
import torch


def none_or_str(value):
    if value == 'None':
        return None
    return value


def get_args(*args):

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--dtype', type=str, help='Data type')

    parser.add_argument('--ms', type=int, default=5)
    parser.add_argument('--ep', type=int, default=250, help='Epochs')
    parser.add_argument('--bs', type=int, default=128, help='batch size')

    parser.add_argument('-opt', type=str, required=True, help='Optimizer')
    parser.add_argument('-lr', type=float, required=True, help='learning rate')
    parser.add_argument('-wd', type=float, required=True, help='weight decay')
    parser.add_argument('-m', type=float, required=True, help='momentum')

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

    args.n = f"{args.dtype}/{args.opt}_{args.lr}_{args.m}_{args.bs}_{args.wd}"

    return args


if __name__ == '__main__':
    args = get_args()
