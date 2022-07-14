import argparse
import torch
import os
from model.wide_res_net import Wide_ResNet
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
import time


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
    
    args.cp_dir = f"checkpoints/basic_none/{args.dtype}/ssgd/run_ms_{args.seed}/{args.depth}_{args.width_factor}_{args.M}_{args.std}"
    while True:
        files = len(glob.glob(f"{args.cp_dir}/run*"))
        cp_dir = f"{args.cp_dir}/run{files}"
        try:
            os.makedirs(cp_dir)
            args.cp_dir = cp_dir
            break
        except:
            continue

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
        model = Wide_ResNet(args.depth, args.width_factor, num_classes=100).to(device)
    else:
        dataset = Cifar(args.batch_size, args.threads)
        model = Wide_ResNet(args.depth, args.width_factor, num_classes=10).to(device)

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
        model.train()
        
        loss, err1 = AverageMeter(), AverageMeter()
        for batch_idx, batch in enumerate(dataset.train, 1):
            inputs, targets = (b.to(device) for b in batch)

            # add noise to theta
            # Classical technique
            # with torch.no_grad():
            #     noise = []
            #     for mp in model.parameters():
            #         temp = torch.empty_like(mp, device=mp.data.device)
            #         temp.normal_(0, std*(mp.view(-1).norm().item() + 1e-16))
            #         noise.append(temp)
            #         mp.data.add_(noise[-1])

            # new technique
            with torch.no_grad():
                noise = []
                for mp in model.parameters():
                    if len(mp.shape) > 1:
                        sh = mp.shape
                        sh_mul = np.prod(sh[1:])
                        temp = mp.view(sh[0], -1).norm(dim=1, keepdim=True).repeat(1, sh_mul).view(mp.shape)
                        temp = torch.normal(0, std*temp).to(mp.data.device)
                    else:
                        temp = torch.empty_like(mp, device=mp.data.device)
                        temp.normal_(0, std*(mp.view(-1).norm().item() + 1e-16))
                    noise.append(temp)
                    mp.data.add_(noise[-1])

            # single sample convolution approximation
            with torch.set_grad_enabled(True):
                predictions = model(inputs)
                batch_loss = criterion(predictions, targets) / args.M
                batch_loss.backward()

            # going back to without theta
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
        logger.info('Train_Loss = {0}, Train_Err = {1}'.format(loss.avg, err1.avg))

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
