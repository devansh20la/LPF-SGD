import torch
import torch.nn as nn
from models import ResNet18, LeNet
import logging
from utils import get_loader
import argparse
from utils.train_utils import AverageMeter, accuracy


def main(args):
    logger = logging.getLogger('my_log')

    dset_loaders = get_loader(args, training=False)
    model = ResNet18()
    model.load_state_dict(torch.load("checkpoints/cifar10/resnet/run_ms_1/trained_model.pth.tar", map_location='cpu'))

    model.norm()
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    model.eval()

    loss = AverageMeter()
    err1 = AverageMeter()
    err5 = AverageMeter()

    for batch_idx, inp_data in enumerate(dset_loaders['test'], 1):

        inputs, targets = inp_data

        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)

        loss.update(batch_loss.item(), inputs.size(0))
        batch_err = accuracy(outputs, targets, topk=(1, 5))
        err1.update(float(100.0 - batch_err[0]), inputs.size(0))
        err5.update(float(100.0 - batch_err[1]), inputs.size(0))

    logger.info('Val_Loss = {0}, Val_Err = {1:6f}'.format(loss.avg, err1.avg, err5.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--dtype', type=str, default="cifar10", help='Data type')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--cp_dir', type=str, default=64)
    parser.add_argument('--print_freq', type=int, default=50)

    args = parser.parse_args()

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

    # Logging tools
    logger = logging.getLogger('my_log')
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(args)
    main(args)
