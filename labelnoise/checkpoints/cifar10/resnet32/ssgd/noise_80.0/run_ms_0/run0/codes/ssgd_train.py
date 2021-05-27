import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import cifar_resnet32
import time
import logging
import numpy as np
import os
import random
from torch.utils.tensorboard import SummaryWriter
from utils import get_loader, AverageMeter, accuracy, CosineInc
import shutil
import glob


if __name__ == "__main__":
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
    
    # ssgd parameters
    parser.add_argument("--std", default=0.001, type=float)
    parser.add_argument("--M", default=8, type=int)
    parser.add_argument("--inc", default=1, type=int, help="increase")

    args = parser.parse_args()

    # initialze num_classes
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
    args.n = f"{args.dtype}/{args.mtype}/ssgd/noise_{args.noise}"

    args.bs = int(args.bs / args.M)

    # Random seed
    random.seed(args.ms)
    torch.manual_seed(args.ms)
    np.random.seed(args.ms)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.ms)

    # Intialize directory and create path
    args.cp_dir = f"{args.dir}/checkpoints/{args.n}/run_ms_{args.ms}/"
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
    writer = SummaryWriter(log_dir=args.cp_dir)

    #####################################################################################################
    dset_loaders = get_loader(args, training=True)

    if args.dtype == 'cifar10' or args.dtype == 'cifar100':
        if args.mtype == 'resnet32':
            model = cifar_resnet32(num_classes=args.num_classes)
        else:
            print("define model")
            quit()
    else:
        print("define dataset type")
        quit()

    criterion = torch.nn.CrossEntropyLoss()
    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    # initialize base optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.mo, 
        weight_decay=args.wd,
        nesterov=True
    )

    # initialize scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    torch.save(model.state_dict(), f"{args.cp_dir}/model_init.pth.tar")

    # intialize std scheduler
    std_scheduler = CosineInc(args.std, args.ep, len(dset_loaders['train'].dataset) / args.M, args.inc - 1)
    std = std_scheduler(0)

    best_err = float('inf')

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

    for epoch in range(start_epoch, args.ep):
        logging.info('Epoch: [%d | %d]' % (epoch, args.ep))

        # Training step
        model.train()
        loss, err1 = AverageMeter(), AverageMeter()

        for batch_idx, inp_data in enumerate(dset_loaders['train'], 1):
            inputs, targets = inp_data

            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            
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

        writer.add_scalar('std', std, epoch)
        writer.add_scalar('Train/Train_Loss', loss.avg, epoch)
        writer.add_scalar('Train/Train_Err1', err1.avg, epoch)
        logging.info('Train_Loss = {0}, Train_Err = {1}'.format(loss.avg, err1.avg))
        scheduler.step()

        model.eval()
        loss, err1 = AverageMeter(), AverageMeter()

        with torch.no_grad():
            for inp_data in dset_loaders['val']:
                inputs, targets = inp_data
                
                if args.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                predictions = model(inputs)
                batch_loss = criterion(predictions, targets)
                loss.update(batch_loss, inputs.shape[0])
                err1.update(float(100 - accuracy(predictions, targets, topk=(1,))[0]), inputs.shape[0])

        writer.add_scalar('Val/Val_Loss', loss.avg, epoch)
        writer.add_scalar('Val/Val_Err1', err1.avg, epoch)
        logging.info('Val_Loss = {0}, Val_Err = {1}'.format(loss.avg, err1.avg))

        if err1.avg < best_err:
            torch.save(model.state_dict(), f"{args.cp_dir}/best_model.pth.tar")
            best_err = err1.avg

        
