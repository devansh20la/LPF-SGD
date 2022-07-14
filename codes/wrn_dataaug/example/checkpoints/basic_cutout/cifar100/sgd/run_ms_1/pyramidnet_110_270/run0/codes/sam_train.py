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
import sys; sys.path.append("..")
from sam import SAM
import shutil
import time
from torch.utils.tensorboard import SummaryWriter
from utility.train_utils import AverageMeter, accuracy
import glob
import logging


def create_path(args, path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        if len(glob.glob(f"{path}/*.log", recursive=True)) > 0:
            file = glob.glob(f"{path}/*.log", recursive=True)[0]
            with open(file, 'r') as f:
                text = f.read()
            if f"Epoch: [{args.epochs - 1} | {args.epochs}]" in text:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model configs
    parser.add_argument("--mtype", default="wrn", type=str, help="Model Type")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--resume", default = None, type=str, help="resume")

    # optim config
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")

    # seed
    parser.add_argument("--seed", required=True, type=int, help="seed")

    # Data preprocessing and loading
    parser.add_argument("--dtype", default='cifar10', type=str, help="dtype")
    parser.add_argument("--img_aug", default='basic_none', type=str)
    parser.add_argument("--threads", default=4, type=int, help="Number of CPU threads for dataloaders.")
    
    args = parser.parse_args()

    # initialze num_classes
    if args.dtype == 'cifar10': 
        args.num_classes = 10
    else:
        args.num_classes = 100

    # initialize seed
    initialize(seed=args.seed)

    # initialize directory
    args.cp_dir = f"checkpoints/{args.img_aug}/{args.dtype}/" + \
                  f"sam/run_ms_{args.seed}/{args.mtype}_{args.depth}_{args.width_factor}"    

    files = len(glob.glob(f"{args.cp_dir}/run*"))
    args.cp_dir = f"{args.cp_dir}/run{files}"
    create_path(args, args.cp_dir)
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
        nesterov = False
    elif args.mtype == "shakeshake":
        model = ShakeShake(
            args.depth, 
            args.width_factor, 
            input_shape=(1, 3, 32, 32), 
            num_classes=args.num_classes).to(device)
    elif args.mtype == "pyramidnet":
        if args.depth == 110:
            model = PyramidNet(args.depth, args.width_factor, args.num_classes, bottleneck=True).to(device)
        else:
            model = PyramidNet(args.depth, args.width_factor, args.num_classes, bottleneck=False).to(device)
        nesterov = True
    else:
        logging.fatal("Select correct model type")
        sys.exit()

    base_optimizer = torch.optim.SGD
    optimizer = SAM(
        model.parameters(), 
        base_optimizer, 
        rho=args.rho, 
        lr=args.learning_rate, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay,
        nesterov=nesterov
    )
    
    # initialize scheduler
    if args.mtype == "wrn":
        scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    elif args.mtype == "shakeshake":
        scheduler = get_cosine_annealing_scheduler(
            optimizer, args.epochs, len(dataset.train), args.learning_rate)
    elif args.mtype == "pyramidnet":
        scheduler = pyramidnet_scheduler(optimizer, args.learning_rate, args.epochs)
    else:
        logging.fatal("Select correct model type")
        sys.exit()

    criterion = torch.nn.CrossEntropyLoss()
    best_err = float('inf')

    start_epoch = 0
    state = {
        'epoch': 1,
        'state_dict': model.state_dict(),
        'best_err': best_err,
        'optimizer' : optimizer.state_dict(),
    }
    torch.save(state, f'{args.cp_dir}/model_init.pth.tar')

    for epoch in range(start_epoch, args.epochs):
        logging.info('Epoch: [%d | %d]' % (epoch, args.epochs))
        model.train()
        loss, err1 = AverageMeter(), AverageMeter()

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            predictions = model(inputs)
            batch_loss = criterion(predictions, targets)
            batch_loss.backward()
            loss.update(batch_loss, inputs.shape[0])

            with torch.no_grad():
                err1.update(float(100 - accuracy(predictions, targets, topk=(1,))[0]), inputs.shape[0])

            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            criterion(model(inputs), targets).backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                if args.mtype == 'shakeshake':
                    scheduler.step()
                else:
                    scheduler.step(epoch)

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
            best_err = err1.avg
            state = {
                'epoch': 1,
                'state_dict': model.state_dict(),
                'best_err': best_err,
                'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, f'{args.cp_dir}/best_model.pth.tar')

