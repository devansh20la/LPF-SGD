import torch
import torch.nn as nn
import torch.optim as optim
from models import resnet18_narrow
import time
import logging
import numpy as np
import glob
import os
import random
from torch.utils.tensorboard import SummaryWriter
from utils import get_loader, class_model_run
import argparse


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
    parser.add_argument('--ep', type=int, default=500, help='Epochs')
    parser.add_argument('--exp_num', type=int, required=True)

    args = parser.parse_args(*args)

    from sklearn.model_selection import ParameterGrid
    param_grid = {'ms': [0, 1, 2],  # seed
                  'mo': [0.0, 0.5, 0.9],  # momentum
                  'width': [4, 6, 8],  # network width
                  'wd': [0.0, 1e-4, 5e-4],  # weight decay
                  'lr': [0.01, 0.0075, 0.005],  # learning rate
                  'bs': [32, 128, 512],  # batch size
                  'skip': [True, False], # skip
                  'batchnorm': [True, False]  # batchnorm
                  }

    grid = list(ParameterGrid(param_grid))
    if args.exp_num > len(grid):
        print("Experiment number is invalid")
        quit()
    params = grid[args.exp_num]

    args.ms = params['ms']
    args.mo = params['mo']
    args.width = params['width']
    args.wd = params['wd']
    args.lr = params['lr']
    args.bs = params['bs']
    args.skip = params['skip']
    args.batchnorm = params['batchnorm']

    args.dtype = 'cifar10'
    args.num_classes = 10

    args.data_dir = f"{args.dir}/data/{args.dtype}"
    args.use_cuda = torch.cuda.is_available()

    args.n = f"{args.dtype}/resnet/" \
             f"{args.exp_num}_{args.ms}_{args.mo}_{args.width}_{args.wd}_" \
             f"{args.lr}_{args.bs}_{args.skip}_{args.batchnorm}"

    return args


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        print("File exists")
        quit()


def main(args):
    logger = logging.getLogger('my_log')

    dset_loaders = get_loader(args, training=True)
    model = resnet18_narrow(args)
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.mo, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.1)

    writer = SummaryWriter(log_dir=args.cp_dir)
    torch.save(model.state_dict(), f"{args.cp_dir}/model_init.pth.tar")

    for epoch in range(args.ep):

        logger.info('Epoch: [%d | %d]' % (epoch, args.ep))

        trainloss, trainerr1 = class_model_run('train', dset_loaders['train'],
                                               model, criterion, optimizer, args)

        # check train loss in validation model to check for convergence
        trainloss, trainerr1 = class_model_run('val', dset_loaders['train'],
                                               model, criterion, optimizer, args)

        logger.info('Train_Loss = {0}, Train_Err = {1}'.format(trainloss, trainerr1))
        writer.add_scalar('Train/Train_Loss', trainloss, epoch)
        writer.add_scalar('Train/Train_Err1', trainerr1, epoch)

        torch.save(model.state_dict(), f"{args.cp_dir}/trained_model.pth.tar")

        if trainloss <= 0.01:
            logger.info("Stopping criterion achieved")
            break

        scheduler.step()


if __name__ == '__main__':
    args = get_args()

    # Random seed
    random.seed(args.ms)
    torch.manual_seed(args.ms)
    np.random.seed(args.ms)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.ms)

    # Intialize directory and create path
    args.cp_dir = f"{args.dir}/checkpoints/{args.n}/run_ms_{args.ms}"
    create_path(args.cp_dir)

    # Logging tools
    logger = logging.getLogger('my_log')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(
        args.cp_dir, time.strftime("%Y%m%d-%H%M%S") + '.log'))
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(args)

    main(args)
