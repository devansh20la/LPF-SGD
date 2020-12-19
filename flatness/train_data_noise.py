import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import ResNet18
import time
import logging
import numpy as np
import glob
import os
import random
from torch.utils.tensorboard import SummaryWriter
from utils import get_loader, class_model_run


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        file = glob.glob(f"{path}/*.log")[0]
        with open(file, 'r') as f:
            file = f.read()
            if 'Epoch: [499 | 500]' in file or 'Stopping criterion achieved' in file or 'nan' in file:
                print("exists")
                quit()
            else:
                for file in glob.glob(path+'/*'):
                    print(f"deleting {file}")
                    os.remove(file)


def main(args):
    logger = logging.getLogger('my_log')

    dset_loaders = get_loader(args, training=True)
    model = ResNet18()
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

        logger.info('Train_Loss = {0}, Train_Err = {1}'.format(trainloss, trainerr1))
        writer.add_scalar('Train/Train_Loss', trainloss, epoch)
        writer.add_scalar('Train/Train_Err1', trainerr1, epoch)

        valloss, valerr1 = class_model_run('val', dset_loaders['val'], model,
                                           criterion, optimizer, args)
        writer.add_scalar('Val/Val_Loss', valloss, epoch)
        writer.add_scalar('Val/Val_Err1', valerr1, epoch)
        logger.info('Val_Loss = {0}, Val_Err = {1}'.format(valloss, valerr1))

        torch.save(model.state_dict(), f"{args.cp_dir}/trained_model.pth.tar")

        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--dtype', type=str, default="cifar10_noisy", help='Data type')
    parser.add_argument('--print_freq', type=int, default=500)

    parser.add_argument('--ep', type=int, default=500, help='epochs')
    parser.add_argument('--ms', type=int, default=0, help='manula seed')
    parser.add_argument('--mo', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--dn', type=float, default=0, help='Data noise')

    args = parser.parse_args()

    if args.dtype == 'cifar10' or args.dtype == 'cifar10_noisy':
        args.num_classes = 10
    elif args.dtype == 'mnist':
        args.num_classes = 10
    else:
        print(f"BAD COMMAND dtype: {args.dtype}")

    # This is done to run job on cluster with support for array jobs
    args.dn = float(args.dn) / 10
    if args.dn > 30:
        print("Label noise should be in the range 0 - 20")
        quit()

    args.data_dir = f"{args.dir}/data/{args.dtype}"
    args.use_cuda = torch.cuda.is_available()

    args.n = f"{args.dtype}/resnet_data_noise_{args.dn}"

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
