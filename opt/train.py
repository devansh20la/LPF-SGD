from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
from models import LeNet
import time
import logging
import numpy as np
import glob
import os
import random
from torch.utils.tensorboard import SummaryWriter
from utils import get_loader, class_model_run, AMSGD, SGD


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


def main(args):
    logger = logging.getLogger('my_log')

    dset_loaders = get_loader(args, training=True)

    model = LeNet()
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    if args.opt == "amsgd":
        optimizer = AMSGD(model.parameters(), lr=args.lr,
                          momentum=args.m, weight_decay=args.wd)
    elif args.opt == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr,
                        momentum=args.m, weight_decay=args.wd)

    writer = SummaryWriter(log_dir=args.cp_dir)

    saved_dict = {'cos_dist': [],
                  "lr": [],
                  "mu": []}

    for epoch in range(args.ep):

        logger.info('Epoch: [%d | %d]' % (epoch, args.ep))

        stats = \
            class_model_run('train', dset_loaders,
                            model, criterion, optimizer, args)

        logger.info('Train_Loss = {0}, Train_Err = {1}'.format(stats["loss"].avg, stats["err1"].avg))

        writer.add_scalar('Train/Train_Loss', stats["loss"].avg, epoch)
        writer.add_scalar('Train/Train_Err1', stats["err1"].avg, epoch)
        writer.add_scalar('Train/Train_Err5', stats["err5"].avg, epoch)
        saved_dict["cos_dist"] += stats["cos_dist"]
        saved_dict["lr"] += stats["lr"]
        saved_dict["mu"] += stats["mu"]

        grads = stats["grad_update"]

        stats = \
            class_model_run('val', dset_loaders, model,
                            criterion, optimizer, args)
        logger.info('Val_Loss = {0}, Val_Err = {1}'.format(stats["loss"].avg, stats["err1"].avg))
        writer.add_scalar('Val/Val_Loss', stats["loss"].avg, epoch)
        writer.add_scalar('Val/Val_Err1', stats["err1"].avg, epoch)
        writer.add_scalar('Val/Val_Err5', stats["err5"].avg, epoch)

        if epoch % 1 == 0:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                "grads": grads
            }
            torch.save(state, f"{args.cp_dir}/trained_model_ep{epoch}.pth.tar")
            torch.save(saved_dict, f"{args.cp_dir}/stats.pth.tar")


if __name__ == '__main__':
    # args = get_args(["-opt","sgd","-lr","0.01","-m","0.9","-wd","0.0","--dtype","mnist","--print_freq","50"])
    args = get_args()

    # Random seed
    random.seed(args.ms)
    torch.manual_seed(args.ms)
    np.random.seed(args.ms)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.ms)

    # Intialize directory and create path
    args.cp_dir = f"{args.dir}/checkpoints/{args.n}"
    list_of_files = sorted(glob.glob1(args.cp_dir, '*run*'))
    if len(list_of_files) == 0:
        list_of_files = 0
    else:
        list_of_files = len(list_of_files)

    args.cp_dir = f"{args.cp_dir}/run{list_of_files}"
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
