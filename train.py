from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from models import conv_net
from fit import class_model_run
import time
import logging
import numpy as np
import glob
import os
import random
from torch.utils.tensorboard import SummaryWriter


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


def main(args):
    logger = logging.getLogger('my_log')

    data_tranforms = {
         'train': transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))]),
         'val': transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))])
    }

    dsets = {
        'train': datasets.MNIST('data/', train=True, download=True,
                                transform=data_tranforms['train']),
        'val': datasets.MNIST('data/', train=False, download=True,
                              transform=data_tranforms['val'])
    }

    dset_loaders = {
        x: torch.utils.data.DataLoader(dsets[x], batch_size=128, shuffle=(x == 'train'))
        for x in ['train', 'val']
    }

    model = conv_net()

    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        torch.backends.cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.m, weight_decay=args.wd)

    writer = SummaryWriter(log_dir=args.cp_dir)

    start_epoch = 0
    best_err = float('inf')

    for epoch in range(start_epoch, args.ep):

        logger.info('Epoch: [%d | %d]' % (epoch, args.ep))

        trainloss, trainerr1, trainerr5 = \
            class_model_run('train', dset_loaders,
                            model, criterion, optimizer, args)

        logger.info('Train_Loss = {0}, Train_Err = {1}'.format(trainloss, trainerr1))
        writer.add_scalar('Train/Train_Loss', trainloss, epoch)
        writer.add_scalar('Train/Train_Err1', trainerr1, epoch)
        writer.add_scalar('Train/Train_Err5', trainerr5, epoch)

        valloss, valerr1, valerr5 = \
            class_model_run('val', dset_loaders, model,
                            criterion, optimizer, args)
        logger.info('Val_Loss = {0}, Val_Err = {1}'.format(valloss, valerr1))
        writer.add_scalar('Val/Val_Loss', valloss, epoch)
        writer.add_scalar('Val/Val_Err1', valerr1, epoch)
        writer.add_scalar('Val/Val_Err5', valerr5, epoch)

        is_best = valerr1 < best_err

        if epoch % 50 == 0:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_err': best_err
            }

            torch.save(state,
                       os.path.join(args.cp_dir,
                                    'train_model_ep{}.pth.tar'.format(epoch)))
        if is_best:
            best_err = min(valerr1, best_err)
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_err': best_err
            }

            logger.info("New Best Model Found")
            logger.info("Best Loss:{0:2f}".format(best_err))

            torch.save(state,
                       os.path.join(args.cp_dir, 'best_model.pth.tar'))
            is_best = False


if __name__ == '__main__':
    args = get_args()

    # Random seed
    random.seed(args.ms)
    torch.manual_seed(args.ms)
    np.random.seed(args.ms)

    if args.use_cuda:
        torch.cuda.manual_seed_all(args.ms)

    # Intialize directory and create path
    args.cp_dir = os.path.join(args.dir, "checkpoints", args.n)
    list_of_files = sorted(glob.glob1(args.cp_dir, '*run*'))

    if len(list_of_files) == 0:
        list_of_files = 0
    else:
        list_of_files = list_of_files[-1]
        list_of_files = int(list_of_files[3:]) + 1

    args.cp_dir = os.path.join(args.cp_dir,
                                       'run{0}'.format(list_of_files))
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
