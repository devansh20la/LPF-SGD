import argparse
import torch
import os
from model.wide_res_net import Wide_ResNet
from model.shake_shake import ShakeShake
from cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR, get_cosine_annealing_scheduler
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

    # model configs
    parser.add_argument("--mtype", default="wrn", type=str, help="Model Type")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")

    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")

    # seed
    parser.add_argument("--seed", required=True, type=int, help="seed")

    # Data preprocessing and loading
    parser.add_argument("--dtype", default='cifar10', type=str, help="dtype")
    parser.add_argument("--img_aug", default='basic_none', type=str)
    parser.add_argument("--threads", default=4, type=int, help="Number of CPU threads for dataloaders.")

    # ssgd parameters
    parser.add_argument("--std", required=True, type=float)
    parser.add_argument("--M", default=1, type=int)
    parser.add_argument("--inc", default=7, type=int, help="increase")

    args = parser.parse_args()

    # initialze num_classes
    if args.dtype == 'cifar10': 
        args.num_classes = 10
    else:
        args.num_classes = 100

    args.batch_size = int(args.batch_size / args.M)

    # initialize seed
    initialize(args, seed=args.seed)

    # initialize directory
    args.cp_dir = f"checkpoints/{args.img_aug}/{args.dtype}/ssgd/run_ms_{args.seed}/{args.depth}_{args.width_factor}_{args.M}_{args.std}"
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

    # initialize logging
    train_log = os.path.join(args.cp_dir, time.strftime("%Y%m%d-%H%M%S") + '.log')
    logging.basicConfig(
        format="%(name)s: %(message)s",
        level="INFO",
        handlers=[
            logging.FileHandler(train_log),
            logging.StreamHandler()
        ]
    )

    writer = SummaryWriter(log_dir=args.cp_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize data
    if args.dtype == 'cifar100':
        if args.img_aug == "basic_none":
            dataset = Cifar(args.batch_size, 4, want_cifar100=True)
        elif args.img_aug == "basic_cutout":
            dataset = Cifar(args.batch_size, 4, want_cifar100=True, cutout=True)
        elif args.img_aug == "autoaugment_cutout":
            dataset = Cifar(args.batch_size, 4, want_cifar100=True, autoaugment=True)
        else:
            logging.fatal("Select correct data type")
            sys.exit()
    else:
        if args.img_aug == "basic_none":
            dataset = Cifar(args.batch_size, 4)
        elif args.img_aug == "basic_cutout":
            dataset = Cifar(args.batch_size, 4, cutout=True)
        elif args.img_aug == "autoaugment_cutout":
            dataset = Cifar(args.batch_size, 4, autoaugment=True)
        else:
            logging.fatal("Select correct data type")
            sys.exit()

    # initialize model
    if args.mtype == "wrn":
        model = Wide_ResNet(
            args.depth, 
            args.width_factor, 
            num_classes=args.num_classe).to(device)
    elif args.mtype == "shakeshake":
        model = ShakeShake(
            args.depth, 
            args.width_factor, 
            input_shape=(1, 3, 32, 32), 
            num_classes=args.num_classes).to(device)
    else:
        logging.fatal("Select correct model type")
        sys.exit()

    # initialize base optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.learning_rate, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay,
        nesterov=True
    )

    # initialize scheduler
    if args.mtype == "wrn":
        scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    elif args.mtype == "shakeshake":
        scheduler = get_cosine_annealing_scheduler(
            optimizer, args.epochs, len(dataset.train) / args.M, args.learning_rate)
    else:
        logging.fatal("Select correct model type")
        sys.exit()

    torch.save(model.state_dict(), f"{args.cp_dir}/model_init.pth.tar")

    # intialize std scheduler
    std_scheduler = CosineInc(args.std, args.epochs, len(dataset.train) / args.M, args.inc - 1)
    std = std_scheduler(0)
    
    criterion = torch.nn.CrossEntropyLoss()
    best_err = float('inf')

    current_step = 0.0
    for epoch in range(args.epochs):
        logging.info('Epoch: [%d | %d]' % (epoch, args.epochs))

        # Training step
        model.train()
        loss, err1 = AverageMeter(), AverageMeter()

        for batch_idx, batch in enumerate(dataset.train, 1):
            inputs, targets = (b.to(device) for b in batch)

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
                optimizer.step()
                optimizer.zero_grad()
                
                std = std_scheduler(current_step)
                # THIS IS ONLY FOR COSINE SCHDULE BE CAREFUL
                scheduler.step()

        
        writer.add_scalar('std', std, epoch)
        writer.add_scalar('train_loss', loss.avg, epoch)
        writer.add_scalar('train_err1', err1.avg, epoch)
        logging.info('Train_Loss = {0}, Train_Err = {1}'.format(loss.avg, err1.avg))

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
        logging.info('Val_Loss = {0}, Val_Err = {1}'.format(loss.avg, err1.avg))

        if err1.avg < best_err:
            torch.save(model.state_dict(), f"{args.cp_dir}/best_model.pth.tar")
            best_err = err1.avg
