import torch
import torch.nn as nn
import torch.optim as optim
from models import resnet_dd
import time
import logging
import numpy as np
import glob
import os
import random
from torch.utils.tensorboard import SummaryWriter
from utils import get_loader, class_model_run
import argparse
import shutil


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        file = glob.glob(f"{path}/*.log")[0]
        with open(file, 'r') as f:
            file = f.read()
            if 'Epoch: [3999 | 4000]' in file or 'Stopping criterion achieved' in file:
                print("exists")
                quit()
            else:
                for file in glob.glob(path+'**/*'):
                    print(f"deleting {file}")
                    os.remove(file)


def main(args):
    logger = logging.getLogger('my_log')

    dset_loaders = get_loader(args, training=True, label_noise=0.20)
    model = resnet_dd(args.width)
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    torch.save(model.state_dict(), f"{args.cp_dir}/model_init.pth.tar")

    writer = SummaryWriter(log_dir=args.cp_dir)

    for epoch in range(args.ep):

        logger.info('Epoch: [%d | %d]' % (epoch, args.ep))

        trainloss, trainerr1 = class_model_run('train', dset_loaders['train'],
                                               model, criterion, optimizer, args)
        writer.add_scalar('Train/Train_Loss', trainloss, epoch)
        writer.add_scalar('Train/Train_Err1', trainerr1, epoch)
        logger.info('Train_Loss = {0}, Train_Err = {1}'.format(trainloss, trainerr1))

        valloss, valerr1 = class_model_run('val', dset_loaders['val'], model,
                                           criterion, optimizer, args)
        writer.add_scalar('Val/Val_Loss', valloss, epoch)
        writer.add_scalar('Val/Val_Err1', valerr1, epoch)
        logger.info('Val_Loss = {0}, Val_Err = {1}'.format(valloss, valerr1))

        torch.save(model.state_dict(), f"{args.cp_dir}/trained_model.pth.tar")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--ep', type=int, default=500, help='epochs')
    parser.add_argument('--ms', type=int, default=0, help='manula seed')
    parser.add_argument('--mo', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--width', type=int, default=64, help='width')

    args = parser.parse_args()

    args.dtype = 'cifar10':
    args.num_classes = 10


    args.data_dir = f"{args.dir}/data/{args.dtype}"
    args.use_cuda = torch.cuda.is_available()

    args.n = f"{args.dtype}_dd/resnet_dd_{args.width}"

    # Random seed
    random.seed(args.ms)
    torch.manual_seed(args.ms)
    np.random.seed(args.ms)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.ms)

    # Intialize directory and create path
    args.cp_dir = f"{args.dir}/checkpoints/{args.n}/run_ms_{args.ms}"
    create_path(args.cp_dir)
    for file in glob.glob("**/*.py", recursive=True):
        if "checkpoints" in file or "data" in file or "results" in file:
            continue
        os.makedirs(os.path.dirname(f"{args.cp_dir}/codes/{file}"), exist_ok=True)
        shutil.copy(file, f"{args.cp_dir}/codes/{file}")

    # Logging tools
    logger = logging.getLogger('my_log')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.cp_dir, time.strftime("%Y%m%d-%H%M%S") + '.log'))
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info(args)

    main(args)
