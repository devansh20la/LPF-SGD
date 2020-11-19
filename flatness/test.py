from args import get_args
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
from utils.train_utils import AverageMeter, accuracy


def main(args):

    dset_loaders = get_loader(args, training=True)
    model = resnet18_narrow(args)
    model.load_state_dict(torch.load(f"{args.cp_dir}/trained_model.pth.tar", map_location="cpu"))
    print(args.cp_dir)
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    model.eval()
    model.norm()

    for phase in ["train", "val"]:
        loss = AverageMeter()
        err1 = AverageMeter()

        for batch_idx, inp_data in enumerate(dset_loaders[phase], 1):

            inputs, targets = inp_data

            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)

            loss.update(batch_loss.item(), inputs.size(0))
            batch_err = accuracy(outputs, targets, topk=(1, 5))
            err1.update(float(100.0 - batch_err[0]), inputs.size(0))

        print(loss.avg, err1.avg)


if __name__ == '__main__':
    args = get_args(["--exp_num", "0"])

    # Intialize directory and create path
    args.cp_dir = f"{args.dir}/checkpoints/{args.n}/run_ms_{args.ms}"
    main(args)
