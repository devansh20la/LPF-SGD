import torch
import torch.nn as nn
import torch.optim as optim
from models import cifar_resnet50, cifar_resnet18, cifar_resnet101, LeNet
from torchvision.models import resnet18 as imagenet_resnet18
from torchvision.models import resnet50 as imagenet_resnet50
from torchvision.models import resnet101 as imagenet_resnet101
import time
import logging
import numpy as np
import os
import random
from torch.utils.tensorboard import SummaryWriter
from utils import get_loader
from utils.train_utils import AverageMeter, accuracy
from utils import EntropySGD
import argparse
import shutil
import glob


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        file = glob.glob(f"{path}/*.log", recursive=True)[0]
        with open(file, 'r') as f:
            text = f.read()

        if "Epoch: [19 | 20]" in text:
            print("File exists")
            quit()
        else:
            shutil.rmtree(path)
            print("Removing old files")
                
    return 


def entropy_fit(phase, loader, model, criterion, optimizer, args):
    """
        Function to forward pass through classification problem
    """
    logger = logging.getLogger('my_log')

    loss, err1 = AverageMeter(), AverageMeter()
    t = time.time()
    if phase == 'train':
        model.train()
        dataiter = iter(loader)
        def closure():
            nonlocal dataiter
            try:
                inp_data = next(dataiter)
            except StopIteration:
                dataiter = iter(loader)
                inp_data = next(dataiter)
            
            inputs, targets = inp_data
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)
            batch_loss.backward()

            loss.update(batch_loss.item(), inputs.size(0))
            err1.update(float(100.0 - accuracy(outputs, targets, topk=(1,))[0]),
                        inputs.size(0))
            return batch_loss

        for step in range(len(dataiter)):
            g = optimizer.step(closure=closure)

    elif phase == 'val':
        model.eval()
        for batch_idx, inp_data in enumerate(loader, 1):
            inputs, targets = inp_data
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)

            loss.update(batch_loss.item(), inputs.size(0))
            err1.update(float(100.0 - accuracy(outputs, targets, topk=(1,))[0]),
                        inputs.size(0))
    else:
        logger.info('Define correct phase')
        quit()

    info = f"Phase:{phase} -- " \
           f"-- {err1.count / (time.time() - t):.2f} samples/sec" \
           f"-- Loss:{loss.avg:.2f} -- Error1:{err1.avg:.2f}"
    logger.info(info)

    return loss.avg, err1.avg


def main(args):
    logger = logging.getLogger('my_log')

    dset_loaders = get_loader(args, training=True)
    if args.dtype == 'cifar10' or args.dtype == 'cifar100':
        if args.mtype == 'resnet50':
            model = cifar_resnet50(num_classes=args.num_classes)
        elif args.mtype == 'resnet18':
            model = cifar_resnet18(num_classes=args.num_classes)
        elif args.mtype == 'resnet101':
            model = cifar_resnet101(num_classes=args.num_classes)
        else:
            print("define model")
            quit()
    elif args.dtype == 'mnist':
        if args.mtype == 'lenet':
            model = LeNet(num_classes=args.num_classes)
        else:
            print("define model")
            quit()
    elif 'imagenet' in args.dtype:
        if args.mtype == 'resnet50':
            model = imagenet_resnet50(num_classes=args.num_classes)
        elif args.mtype == 'resnet18':
            model = imagenet_resnet18(num_classes=args.num_classes)
        elif args.mtype == 'resnet101':
            model = imagenet_resnet101(num_classes=args.num_classes)
        else:
            print("define model")
            quit()
    else:
        print("define dataset type")
        quit()
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    optimizer = EntropySGD(
        model.parameters(),
        dict(lr=args.lr, 
             num_batches=len(dset_loaders['train']), 
             gtime=float('inf'), 
             momentum=args.mo,
             momentum_sgld=0.9, nesterov=False, 
             weight_decay=args.wd, L=5, 
             eps=1e-4, g0=args.g0, g1=args.g1, gmax=2.38, 
             epochs=args.ep, sgld_lr=args.sgld_lr, alpha_arg=0.75, gscale=True))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    writer = SummaryWriter(log_dir=args.cp_dir)
    torch.save(model.state_dict(), f"{args.cp_dir}/model_init.pth.tar")

    if args.loadckp:
        state = torch.load(args.cp_dir + "trained_model.pth.tar")
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        best_err = state['best_err']
        start_epoch = state['epoch'] + 1
    else:
        start_epoch = 0
        best_err = float('inf')

    for epoch in range(start_epoch, args.ep):

        logger.info('Epoch: [%d | %d]' % (epoch, args.ep))

        trainloss, trainerr1 = entropy_fit('train', dset_loaders['train'],
                                           model, criterion, optimizer, args)
        logger.info('Train_Loss = {0}, Train_Err = {1}'.format(trainloss, trainerr1))
        writer.add_scalar('Train/Train_Loss', trainloss, epoch)
        writer.add_scalar('Train/Train_Err1', trainerr1, epoch)

        valloss, valerr1 = entropy_fit('val', dset_loaders['val'], model,
                                       criterion, optimizer, args)
        writer.add_scalar('Val/Val_Loss', valloss, epoch)
        writer.add_scalar('Val/Val_Err1', valerr1, epoch)
        logger.info('Val_Loss = {0}, Val_Err = {1}'.format(valloss, valerr1))

        scheduler.step()

        if valerr1 < best_err:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_err': best_err
            }
            torch.save(state, f"{args.cp_dir}/best_model.pth.tar")
            best_err = valerr1


def get_args(*args):

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--dtype', type=str, default="mnist", help='Data type')
    parser.add_argument('--ep', type=int, default=100, help='Epochs')
    parser.add_argument('--mtype', default='lenet')

    # params
    parser.add_argument('--ms', type=int, default=0, help='ms')
    parser.add_argument('--mo', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--bs', type=int, default=256, help='batch size')
    parser.add_argument('--loadckp', default=False, action='store_true')

    parser.add_argument("--g0", default=0.5, type=float)
    parser.add_argument("--g1", default=0.00001, type=float)
    parser.add_argument("--sgld_lr", default=0.05, type=float)

    args = parser.parse_args(*args)
    args.ep = int(args.ep / 5)

    if args.dtype == 'cifar10':
        args.num_classes = 10
        args.milestones = [int(100/5), int(120/5)]
        args.data_dir = f"{args.dir}/data/{args.dtype}"
    elif args.dtype == 'cifar100':
        args.num_classes = 100
        args.milestones = [int(100/5), int(120/5)]
        args.data_dir = f"{args.dir}/data/{args.dtype}"
    elif args.dtype == 'imagenet':
        args.num_classes = 1000
        args.milestones = [30/5, 60/5, 90/5]
        args.data_dir = f"{args.dir}/data/imagenet_2/"
    elif args.dtype == 'tinyimagenet':
        args.num_classes = 200
        args.milestones = [30/5, 60/5, 90/5]
        args.data_dir = f"{args.dir}/data/{args.dtype}/tiny-imagenet-200/"
    elif args.dtype == 'mnist':
        args.num_classes = 10
        args.milestones = [int(50/5), int(100/5)]
        args.data_dir = f"{args.dir}/data/{args.dtype}"
    else:
        print(f"BAD COMMAND dtype: {args.dtype}")
        quit()

    args.use_cuda = torch.cuda.is_available()
    args.n = f"{args.dtype}/{args.mtype}/entropy_sgd_{args.g0}_{args.g1}_{args.sgld_lr}_5_{args.lr}"

    return args


if __name__ == '__main__':
    args = get_args()

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
    fh = logging.FileHandler(os.path.join(
        args.cp_dir, time.strftime("%Y%m%d-%H%M%S") + '.log'))
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(args)

    main(args)
