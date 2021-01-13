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
import glob
import os
import random
from torch.utils.tensorboard import SummaryWriter
from utils import get_loader
from utils.train_utils import AverageMeter, accuracy
import argparse
import shutil


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        print("deleting")
        shutil.rmtree(path)


def fsdg_fit(phase, loader, model, criterion, optimizer, args):
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
            itr = 1
            for _ in range(itr):
                # add noise to theta
                with torch.no_grad():
                    noise = []
                    for mp in model.parameters():
                        temp = torch.empty_like(mp, device=mp.data.device)
                        temp.normal_(0, args.std*mp.view(-1).norm().item()+1e-16)
                        noise.append(temp)
                        mp.data.add_(noise[-1])

                # single sample convolution approximation
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, targets) * 1/itr
                    batch_loss.backward()
                    loss.update(batch_loss.item(), inputs.size(0))

                # going back to without theta
                with torch.no_grad():
                    for mp, n in zip(model.parameters(), noise):
                        mp.data.sub_(n)

            # with torch.set_grad_enabled(True):
            #     # compute output
            #     optimizer.zero_grad()
            #     outputs = model(inputs)
            #     batch_loss = criterion(outputs, targets)
            #     batch_loss.backward()
            # loss.update(batch_loss.item(), inputs.size(0))

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
    elif args.dtype == 'imagenet':
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

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.mo, weight_decay=args.wd)
    writer = SummaryWriter(log_dir=args.cp_dir)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    if args.loadckp is not None:
        state = torch.load(args.loadckp)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        from collections import Counter
        scheduler.milestones = Counter([150, 185])
        start_epoch = state['epoch'] + 1
        best_err = state['best_err']
    else:
        torch.save(model.state_dict(), f"{args.cp_dir}/model_init.pth.tar")
        start_epoch = 0
        best_err = float('inf')

    for epoch in range(start_epoch, args.ep):

        logger.info('Epoch: [%d | %d]' % (epoch, args.ep))

        trainloss, trainerr1 = fsdg_fit('train', dset_loaders['train'],
                                        model, criterion, optimizer, args)
        logger.info('Train_Loss = {0}, Train_Err = {1}'.format(trainloss, trainerr1))
        writer.add_scalar('Train/Train_Loss', trainloss, epoch)
        writer.add_scalar('Train/Train_Err1', trainerr1, epoch)

        valloss, valerr1 = fsdg_fit('val', dset_loaders['val'], model,
                                    criterion, optimizer, args)
        writer.add_scalar('Val/Val_Loss', valloss, epoch)
        writer.add_scalar('Val/Val_Err1', valerr1, epoch)
        logger.info('Val_Loss = {0}, Val_Err = {1}'.format(valloss, valerr1))
        scheduler.step()

        if valerr1 < best_err:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_err': best_err
            }
            torch.save(state, f"{args.cp_dir}/best_model.pth.tar")
            best_err = valerr1

        if epoch == 100:
            torch.save(state, f"{args.cp_dir}/model_ep_{epoch}.pth.tar")


def get_args(*args):

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--dtype', type=str, default="cifar10", help='Data type')
    parser.add_argument('--ep', type=int, default=500, help='Epochs')
    parser.add_argument('--loadckp', type=str, default=None)
    parser.add_argument('--std', type=float, default=1)
    parser.add_argument('--mtype', default='resnet18')

    # params
    parser.add_argument('--ms', type=int, default=0, help='ms')
    parser.add_argument('--mo', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--bs', type=int, default=128, help='batch size')

    args = parser.parse_args(*args)
    if args.dtype == 'cifar10':
        args.num_classes = 10
        args.milestones = [100, 120]
        args.data_dir = f"{args.dir}/data/{args.dtype}"
    elif args.dtype == 'cifar100':
        args.num_classes = 100
        args.milestones = [100, 120]
        args.data_dir = f"{args.dir}/data/{args.dtype}"
    elif args.dtype == 'imagenet':
        args.num_classes = 1000
        args.milestones = [30, 60, 90]
        args.data_dir = "/imagenet/"
    elif args.dtype == 'mnist':
        args.num_classes = 10
        args.milestones = [50, 100]
        args.data_dir = f"{args.dir}/data/{args.dtype}"
    else:
        print(f"BAD COMMAND dtype: {args.dtype}")

    args.use_cuda = torch.cuda.is_available()
    args.n = f"{args.dtype}_new/{args.mtype}/fsgd"

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
