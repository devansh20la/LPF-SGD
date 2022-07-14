import argparse
import torch
import os
from model.wide_res_net import WideResNet
from cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.cos_inc import CosineInc
import sys; sys.path.append("..")
import shutil
import time
from torch.utils.tensorboard import SummaryWriter
from utility.train_utils import AverageMeter, accuracy
import glob
import logging
import numpy as np


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        print("exists")
        quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=4, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--std", required=True, type=float)
    parser.add_argument("--M", default=1, type=int)
    parser.add_argument("--dtype", default='cifar10', type=str, help="dtype")
    parser.add_argument("--seed", required=True, type=int, help="seed")
    parser.add_argument("--inc", default=7, type=int, help="increase")
    args = parser.parse_args()

    args.batch_size = int(args.batch_size / args.M)
    initialize(args, seed=args.seed)
    
    args.cp_dir = f"checkpoints/{args.dtype}/ssgd/run_ms_{args.seed}/{args.depth}_{args.width_factor}_{args.M}_{args.std}"
    files = len(glob.glob(f"{args.cp_dir}/run*"))
    args.cp_dir = f"{args.cp_dir}/run{files}"
    create_path(args.cp_dir)
    for file in glob.glob("**/*.py", recursive=True):
        if "checkpoints" in file or "data" in file or "results" in file:
            continue
        os.makedirs(os.path.dirname(f"{args.cp_dir}/codes/{file}"), exist_ok=True)
        shutil.copy(file, f"{args.cp_dir}/codes/{file}")
    
    logger = logging.getLogger('my_log')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(
        args.cp_dir, time.strftime("%Y%m%d-%H%M%S") + '.log'))
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(args)

    writer = SummaryWriter(log_dir=args.cp_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dtype == 'cifar100':
        dataset = Cifar(args.batch_size, args.threads, want_cifar100=True)
        model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=100).to(device)
    else:
        dataset = Cifar(args.batch_size, args.threads)
        model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    torch.save(model.state_dict(), f"{args.cp_dir}/model_init.pth.tar")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    std_scheduler = CosineInc(args.std, args.epochs, len(dataset.train) / args.M, args.inc - 1)
    std = std_scheduler(0)
    criterion = torch.nn.CrossEntropyLoss()

    best_err = float('inf')
    current_step = 0.0
    for epoch in range(args.epochs):
        logger.info('Epoch: [%d | %d]' % (epoch, args.epochs))
        loss, err1, avgitr = AverageMeter(), AverageMeter(), AverageMeter()

        for batch_idx, batch in enumerate(dataset.train, 1):
            inputs, targets = (b.to(device) for b in batch)

            model.eval()
            with torch.no_grad():
                base_loss = criterion(model(inputs), targets) / args.M
                
                for itr in range(100):
                    noise = []
                    for mp in model.parameters():
                        temp = torch.empty_like(mp, device=mp.data.device)
                        temp.normal_(0, std*(mp.view(-1).norm().item() + 1e-16))
                        noise.append(temp)
                        mp.data.add_(noise[-1])

                    curr_loss = criterion(model(inputs), targets) / args.M
                    if curr_loss > base_loss:
                        break
                    else:
                        for mp, n in zip(model.parameters(), noise):
                            mp.data.sub_(n)
                avgitr.update(itr, 1)

            model.train()
            with torch.set_grad_enabled(True):
                predictions = model(inputs)
                batch_loss = criterion(predictions, targets) / args.M
                batch_loss.backward()

            with torch.no_grad():
                for mp, n in zip(model.parameters(), noise):
                    mp.data.sub_(n)

            with torch.no_grad():
                loss.update(batch_loss, inputs.shape[0])
                err1.update(float(100 - accuracy(predictions, targets, topk=(1,))[0]), inputs.shape[0])

            if batch_idx % args.M == 0:
                current_step += 1
                std = std_scheduler(current_step)
                optimizer.step()
                optimizer.zero_grad()

        scheduler(epoch)
        writer.add_scalar('std', std, epoch)
        writer.add_scalar('train_loss', loss.avg, epoch)
        writer.add_scalar('train_err1', err1.avg, epoch)
        logger.info(f'Train_Loss = {loss.avg}, Train_Err = {err1.avg}, Avgitr: {avgitr.avg}')

        model.eval()
        loss, err1 = AverageMeter(), AverageMeter()

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                batch_loss = criterion(predictions, targets)
                loss.update(batch_loss, inputs.shape[0])
                err1.update(float(100 - accuracy(predictions, targets, topk=(1,))[0]), inputs.shape[0])

        writer.add_scalar('val_loss', loss.avg, epoch)
        writer.add_scalar('val_err1', err1.avg, epoch)
        logger.info('Val_Loss = {0}, Val_Err = {1}'.format(loss.avg, err1.avg))

        if err1.avg < best_err:
            torch.save(model.state_dict(), f"{args.cp_dir}/best_model.pth.tar")
            best_err = err1.avg
