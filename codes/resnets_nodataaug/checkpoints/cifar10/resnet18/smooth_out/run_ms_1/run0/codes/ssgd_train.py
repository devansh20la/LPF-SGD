import torch
import torch.nn as nn
import torch.optim as optim
from models import cifar_resnet50, cifar_resnet18, cifar_resnet101
import time
import logging
import numpy as np
import os
import random
from torch.utils.tensorboard import SummaryWriter
from utils import get_loader
from utils.train_utils import AverageMeter, accuracy
import copy
import argparse
import shutil
import glob


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        print("Deleting")
        shutil.rmtree(path)


def ssgd_fit(phase, loader, model, noise, criterion, optimizer, args):
    """
        Function to forward pass through classification problem
    """
    logger = logging.getLogger('my_log')

    if phase == 'train':
        model.train()
    else:
        model.eval()

    loss, err1 = AverageMeter(), AverageMeter()
    t = time.time()

    for batch_idx, inp_data in enumerate(loader, 1):

        inputs, targets = inp_data

        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if phase == 'train':
            optimizer.zero_grad()
            # normalize noise
            with torch.no_grad():
                total_norm = 0.0
                for i in noise:
                    total_norm += i.reshape(-1).norm()**2
                total_norm = total_norm**0.5
                scale = args.rho/total_norm
                for i in noise:
                    i *= scale

            # add noise to theta
            with torch.no_grad():
                for mp, n in zip(model.parameters(), noise):
                    mp.data.add_(n)

            # compute grads
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                loss.update(batch_loss.item(), inputs.size(0))
                batch_loss.backward()

            # delete noise and add gradient of noise
            with torch.no_grad():
                for mp, n in zip(model.parameters(), noise):
                    mp.data.sub_(n)
                    n.grad = copy.deepcopy(-mp.grad)

            optimizer.step()

        elif phase == 'val':
            with torch.no_grad():
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                loss.update(batch_loss.item(), inputs.size(0))
        else:
            logger.info('Define correct phase')
            quit()

        err1.update(float(100.0 - accuracy(outputs, targets, topk=(1,))[0]),
                    inputs.size(0))
        if batch_idx % args.print_freq == 0:
            info = f"Phase:{phase} -- Batch_idx:{batch_idx}/{len(loader)}" \
                   f"-- {err1.count / (time.time() - t):.2f} samples/sec" \
                   f"-- Loss:{loss.avg:.2f} -- Error1:{err1.avg:.2f}"
            logger.info(info)

    return loss.avg, err1.avg


def main(args):
    logger = logging.getLogger('my_log')

    dset_loaders = get_loader(args, training=True)
    if args.mtype == 'resnet50':
        model = cifar_resnet50(num_classes=args.num_classes)
    elif args.mtype == 'resnet18':
        model = cifar_resnet18(num_classes=args.num_classes)
    elif args.mtype == 'resnet101':
        model = cifar_resnet101(num_classes=args.num_classes)
    else:
        print("define model")
        quit()
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    noise = []
    for mp in model.parameters():
        temp = torch.empty_like(mp, device=mp.device)
        temp.normal_(0, mp.view(-1).norm().item()+1e-16)
        temp.requires_grad = True
        noise.append(temp)

    optimizer = optim.SGD(
        [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.wd}] +
        [{'params': x, 'lr': args.lr} for x in noise], momentum=args.mo
    )
    writer = SummaryWriter(log_dir=args.cp_dir)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

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
        trainloss, trainerr1 = ssgd_fit('train', dset_loaders['train'], model, noise,
                                        criterion, optimizer, args)
        logger.info('Train_Loss = {0}, Train_Err = {1}'.format(trainloss, trainerr1))
        writer.add_scalar('Train/Train_Loss', trainloss, epoch)
        writer.add_scalar('Train/Train_Err1', trainerr1, epoch)

        valloss, valerr1 = ssgd_fit('val', dset_loaders['val'], model, noise,
                                    criterion, optimizer, args)
        writer.add_scalar('Val/Val_Loss', valloss, epoch)
        writer.add_scalar('Val/Val_Err1', valerr1, epoch)
        logger.info('Val_Loss = {0}, Val_Err = {1}'.format(valloss, valerr1))
        scheduler.step()

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_err': best_err
        }
        if valerr1 < best_err:
            torch.save(state, f"{args.cp_dir}/best_model.pth.tar")
            best_err = valerr1


def get_args(*args):

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--dtype', type=str, default="cifar10", help='Data type')
    parser.add_argument('--ep', type=int, default=500, help='Epochs')
    parser.add_argument('--mtype', default='resnet18')
    parser.add_argument('--rho', type=float, default=0.05)

    # params
    parser.add_argument('--ms', type=int, default=0, help='ms')
    parser.add_argument('--mo', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--loadckp', default=False, action='store_true')

    args = parser.parse_args(*args)

    if args.dtype == 'cifar10':
        args.num_classes = 10
        args.milestones = [150, 200]
        args.data_dir = f"{args.dir}/data/{args.dtype}"
    elif args.dtype == 'cifar100':
        args.num_classes = 100
        args.milestones = [150, 200]
        args.data_dir = f"{args.dir}/data/{args.dtype}"
    elif args.dtype == 'imagenet':
        args.num_classes = 1000
        args.milestones = [30, 60, 90]
        args.data_dir = "/imagenet/"
    else:
        print(f"BAD COMMAND dtype: {args.dtype}")

    args.use_cuda = torch.cuda.is_available()

    args.n = f"{args.dtype}/{args.mtype}/s_sgd"

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
    files = len(glob.glob(f"{args.cp_dir}/run*"))
    args.cp_dir = f"{args.cp_dir}/run{files}"
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
