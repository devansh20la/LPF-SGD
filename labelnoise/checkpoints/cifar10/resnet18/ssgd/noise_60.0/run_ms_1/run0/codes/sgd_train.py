import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import cifar_resnet18
import time
import logging
import numpy as np
import os
import random
from torch.utils.tensorboard import SummaryWriter
from utils import get_loader
from utils.train_utils import AverageMeter, accuracy
import shutil
import glob


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        print("exists")
        quit()


def main(args):
    dset_loaders = get_loader(args, training=True)
    model = cifar_resnet18(num_classes=args.num_classes)
    torch.save(model.state_dict(), f"{args.cp_dir}/model_init.pth.tar")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mo, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(log_dir=args.cp_dir)
    start_epoch = 0
    best_err = float('inf')

    for epoch in range(start_epoch, args.ep):
        logging.info('Epoch: [%d | %d]' % (epoch, args.ep))
        
        ########################### Train #########################################
        model.train()
        loss, err1 = AverageMeter(), AverageMeter()
        t = time.time()
        for batch_idx, inp_data in enumerate(dset_loaders['train'], 1):
            inputs, targets = inp_data

            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            loss.update(batch_loss.item(), inputs.size(0))
            err1.update(float(100.0 - accuracy(outputs, targets, topk=(1,))[0]),
                        inputs.size(0))

            if batch_idx % args.print_freq == 0:
                info = f"Phase:Train -- Batch_idx:{batch_idx}/{len(dset_loaders['train'])}" \
                       f"-- {err1.count / (time.time() - t):.2f} samples/sec" \
                       f"-- Loss:{loss.avg:.2f} -- Error1:{err1.avg:.2f}"
                logging.info(info)

        logging.info('Train_Loss = {0}, Train_Err = {1}'.format(loss.avg, err1.avg))
        writer.add_scalar('Train/Train_Loss', loss.avg, epoch)
        writer.add_scalar('Train/Train_Err1', err1.avg, epoch)
        writer.add_scalar('Metrics/lr', scheduler._last_lr[0], epoch)
        scheduler.step()

        ########################### EVALUATE #########################################
        model.eval()
        loss, err1 = AverageMeter(), AverageMeter()
        for batch_idx, inp_data in enumerate(dset_loaders['val'], 1):
            inputs, targets = inp_data

            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)

            loss.update(batch_loss.item(), inputs.size(0))
            err1.update(float(100.0 - accuracy(outputs, targets, topk=(1,))[0]),
                        inputs.size(0))

        writer.add_scalar('Val/Val_Loss', loss.avg, epoch)
        writer.add_scalar('Val/Val_Err1', err1.avg, epoch)
        logging.info('Val_Loss = {0}, Val_Err = {1}'.format(loss.avg, err1.avg))

        if err1.avg < best_err:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_err': best_err
            }
            torch.save(state, f"{args.cp_dir}/best_model.pth.tar")
            best_err = err1.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--dtype', type=str, default="cifar10", help='Data type')
    parser.add_argument('--ep', type=int, default=500, help='Epochs')
    parser.add_argument('--loadckp', default=False, action='store_true')
    parser.add_argument('--mtype', default='resnet18')
    parser.add_argument('--noise', type=float, default='20')

    # params
    parser.add_argument('--ms', type=int, default=0, help='ms')
    parser.add_argument('--mo', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--bs', type=int, default=128, help='batch size')

    args = parser.parse_args()

    if args.dtype == 'cifar10':
        args.num_classes = 10
        args.milestones = [150, 200]
        args.data_dir = f"{args.dir}/data/{args.dtype}"
    elif args.dtype == 'cifar100':
        args.num_classes = 100
        args.milestones = [150, 200]
        args.data_dir = f"{args.dir}/data/{args.dtype}"
    else:
        print(f"BAD COMMAND dtype: {args.dtype}")

    args.use_cuda = torch.cuda.is_available()
    args.n = f"{args.dtype}/{args.mtype}/sgd/noise_{args.noise}/run_ms_{args.ms}"

    # Random seed
    random.seed(args.ms)
    torch.manual_seed(args.ms)
    np.random.seed(args.ms)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.ms)

    # Intialize directory and create path
    args.cp_dir = f"{args.dir}/checkpoints/{args.n}/run0/"
    create_path(args.cp_dir)

    # Logging tools
    train_log = os.path.join(args.cp_dir, time.strftime("%Y%m%d-%H%M%S") + '.log')
    logging.basicConfig(
        format="%(name)s: %(message)s",
        level="INFO",
        handlers=[
            logging.FileHandler(train_log),
            logging.StreamHandler()
        ]
    )
    logging.info(args)
    main(args)
