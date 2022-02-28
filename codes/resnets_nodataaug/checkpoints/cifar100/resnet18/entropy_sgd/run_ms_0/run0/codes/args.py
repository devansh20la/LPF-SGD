import argparse
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args(*args):

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--dtype', type=str, default="cifar10", help='Data type')
    parser.add_argument('--ep', type=int, default=500, help='Epochs')
    parser.add_argument('--loadckp', default=False, action='store_true')

    # params
    parser.add_argument('--ms', type=int, default=0, help='ms')
    parser.add_argument('--mo', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--bs', type=int, default=128, help='batch size')

    args = parser.parse_args(*args)

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
    args.n = f"{args.dtype}/resnet/sgd"

    return args


if __name__ == '__main__':
    args = get_args()
