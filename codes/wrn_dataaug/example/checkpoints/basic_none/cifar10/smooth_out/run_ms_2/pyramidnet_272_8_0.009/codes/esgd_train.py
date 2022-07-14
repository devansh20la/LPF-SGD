import argparse
import torch
import os
from model.wide_res_net import Wide_ResNet
from model.shake_shake import ShakeShake
from model.pyramidnet import PyramidNet
from cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR, get_cosine_annealing_scheduler, pyramidnet_scheduler
from utility.cos_inc import CosineInc
from utility.entropy_sgd import EntropySGD
import sys; sys.path.append("..")
import shutil
import time
from torch.utils.tensorboard import SummaryWriter
from utility.train_utils import AverageMeter, accuracy
import glob
import logging
import numpy as np
import time
from tqdm import tqdm 


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        if len(glob.glob(f"{path}/*.log", recursive=True)) > 0:
            file = glob.glob(f"{path}/*.log", recursive=True)[0]
            with open(file, 'r') as f:
                text = f.read()
            if "Epoch: [199 | 200]" in text:
                print("File exists")
                quit()
            else:
                shutil.rmtree(path)
                os.makedirs(path)
                print("Removing old files")
        else:
            shutil.rmtree(path)
            os.makedirs(path)
            print("Removing old files")
                
    return 

def main():
    parser = argparse.ArgumentParser()
    # model configs
    parser.add_argument("--mtype", default="wrn", type=str, help="Model Type")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--resume", default = None, type=str, help="resume")

    # optim config
    parser.add_argument("--epochs", default=50, type=int, help="Total number of epochs.")
    parser.add_argument("--learning_rate", default=0.5, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--g0", default=0.1, type=float)
    parser.add_argument("--g1", default=0.0001, type=float)
    parser.add_argument("--sgld_lr", default=0.05, type=float)

    # seed
    parser.add_argument("--seed", required=False, default=0, type=int, help="seed")

    # Data preprocessing and loading
    parser.add_argument("--dtype", default='cifar10', type=str, help="dtype")
    parser.add_argument("--img_aug", default='basic_none', type=str)
    parser.add_argument("--threads", default=8, type=int, help="Number of CPU threads for dataloaders.")

    args = parser.parse_args()

    # initialze num_classes
    if args.dtype == 'cifar10': 
        args.num_classes = 10
    else:
        args.num_classes = 100

    # initialize seed
    initialize(seed=args.seed)

    # initialize directory and save codes
    args.cp_dir = f"checkpoints/{args.img_aug}/{args.dtype}/esgd/run_ms_{args.seed}/" + \
                  f"{args.mtype}_{args.depth}_{args.width_factor}/" + \
                  f"run_{args.g0}_{args.g1}_{args.sgld_lr}_{args.learning_rate}_5"
    create_path(args.cp_dir)

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
    
    logging.info(args)
    writer = SummaryWriter(log_dir=args.cp_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize data
    if args.dtype == 'cifar100':
        if args.img_aug == "basic_none":
            dataset = Cifar(args.batch_size, args.threads, want_cifar100=True)
        elif args.img_aug == "basic_cutout":
            dataset = Cifar(args.batch_size, args.threads, want_cifar100=True, cutout=True)
        elif args.img_aug == "autoaugment_cutout":
            dataset = Cifar(args.batch_size, args.threads, want_cifar100=True, autoaugment=True)
        else:
            logging.fatal("Select correct data type")
            sys.exit()
    else:
        if args.img_aug == "basic_none":
            dataset = Cifar(args.batch_size, args.threads)
        elif args.img_aug == "basic_cutout":
            dataset = Cifar(args.batch_size, args.threads, cutout=True)
        elif args.img_aug == "autoaugment_cutout":
            dataset = Cifar(args.batch_size, args.threads, autoaugment=True)
        else:
            logging.fatal("Select correct data type")
            sys.exit()

    # initialize model
    if args.mtype == "wrn":
        model = Wide_ResNet(
            args.depth, 
            args.width_factor, 
            num_classes=args.num_classes).to(device)
    elif args.mtype == "shakeshake":
        model = ShakeShake(
            args.depth, 
            args.width_factor, 
            input_shape=(1, 3, 32, 32), 
            num_classes=args.num_classes).to(device)
    elif args.mtype == "pyramidnet":
        model = PyramidNet(args.depth, 200, args.num_classes, bottleneck=True).to(device)
    else:
        logging.fatal("Select correct model type")
        sys.exit()

    # initialize base optimizer
    optimizer = EntropySGD(
        model.parameters(),
        dict(lr=args.learning_rate, 
             num_batches=len(dataset.train), 
             gtime=float('inf'), 
             momentum=args.momentum,
             momentum_sgld=0.9, nesterov=True, 
             weight_decay=args.weight_decay, L=5,
             eps=1e-4, g0=args.g0, g1=args.g1, gmax=float('inf'), 
             epochs=args.epochs, sgld_lr=args.sgld_lr, alpha_arg=0.75, gscale=True))

    # initialize scheduler
    if args.mtype == "wrn":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [12,24,32], gamma=0.2)
    elif args.mtype == "shakeshake":
        scheduler = get_cosine_annealing_scheduler(
            optimizer, args.epochs, len(dataset.train), args.learning_rate)
    elif args.mtype == "pyramidnet":
        scheduler = pyramidnet_scheduler(optimizer, args.learning_rate, args.epochs)
    else:
        logging.fatal("Select correct model type")
        sys.exit()

    torch.save(model.state_dict(), f"{args.cp_dir}/model_init.pth.tar")
    
    criterion = torch.nn.CrossEntropyLoss()
    best_err = float('inf')

    if args.resume:
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(f"{args.resume}/best_model.pth.tar")
        start_epoch = checkpoint['epoch']
        best_err = checkpoint['best_err']
        current_step = checkpoint['current_step']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        logging.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
    else:
        start_epoch = 0
        current_step = 0.0
        state = {
            'epoch': 1,
            'state_dict': model.state_dict(),
            'best_err': best_err,
            'optimizer' : optimizer.state_dict(),
            'current_step': current_step
        }
        torch.save(state, f'{args.cp_dir}/model_init.pth.tar')

    for epoch in range(start_epoch, args.epochs):
        logging.info('Epoch: [%d | %d]' % (epoch, args.epochs))

        # Training step
        model.train()
        loss, err1 = AverageMeter(), AverageMeter()
        dataiter = iter(dataset.train)
        
        for step in tqdm(range(len(dataiter))):
            def closure():
                nonlocal dataiter
                try:
                    inp_data = next(dataiter)
                except StopIteration:
                    dataiter = iter(dataset.train)
                    inp_data = next(dataiter)
                
                inputs, targets = inp_data
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                batch_loss.backward()

                loss.update(batch_loss.item(), inputs.size(0))
                err1.update(float(100.0 - accuracy(outputs, targets, topk=(1,))[0]),
                            inputs.size(0))
                return batch_loss

            g = optimizer.step(closure=closure)
            scheduler.step()

        writer.add_scalar('train_loss', loss.avg, epoch)
        writer.add_scalar('train_err1', err1.avg, epoch)
        writer.add_scalar('Params/lr', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('Params/g', g, epoch)
        logging.info('Train_Loss = {0}, Train_Err = {1}'.format(loss.avg, err1.avg))
        
        model.eval()
        loss, err1 = AverageMeter(), AverageMeter()
        for batch_idx, inp_data in enumerate(dataset.test, 1):
            inputs, targets = inp_data

            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)

            loss.update(batch_loss.item(), inputs.size(0))
            err1.update(float(100.0 - accuracy(outputs, targets, topk=(1,))[0]),
                        inputs.size(0))

        writer.add_scalar('val_loss', loss.avg, epoch)
        writer.add_scalar('val_err1', err1.avg, epoch)
        logging.info('Val_Loss = {0}, Val_Err = {1}'.format(loss.avg, err1.avg))

        if err1.avg < best_err:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_err': best_err,
                'optimizer' : optimizer.state_dict(),
                'current_step': current_step
            }
            torch.save(state, f"{args.cp_dir}/best_model.pth.tar")
            best_err = err1.avg

if __name__ == "__main__":
    main()
