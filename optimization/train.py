from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
from models import ResNet18
import time
import logging
import numpy as np
import glob
import os
import random
from torch.utils.tensorboard import SummaryWriter
from utils import get_loader
from utils.train_utils import AverageMeter, accuracy


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        file = glob.glob(f"{path}/*.log")[0]
        with open(file, 'r') as f:
            file = f.read()
            if 'Epoch: [499 | 500]' in file or 'Stopping criterion achieved' in file or 'nan' in file:
                print("exists")
                quit()
            else:
                for file in glob.glob(path+'/*'):
                    print(f"deleting {file}")
                    os.remove(file)


def class_model_run(phase, loader, model, criterion, optimizer, args):
    """
        Function to forward pass through classification problem
    """
    logger = logging.getLogger('my_log')

    if phase == 'train':
        model.train()
    else:
        model.eval()

    loss = AverageMeter()
    err1 = AverageMeter()
    t = time.time()

    for batch_idx, inp_data in enumerate(loader, 1):

        inputs, targets = inp_data

        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if phase == 'train':
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
        elif phase == 'val':
            with torch.no_grad():
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
        else:
            logger.info('Define correct phase')
            quit()

        if np.isnan(batch_loss.item()):
            logger.fatal("Training loss is nan .. quitting application")
            quit()

        loss.update(batch_loss.item(), inputs.size(0))
        batch_err = accuracy(outputs, targets, topk=(1,))[0]
        err1.update(float(100.0 - batch_err), inputs.size(0))

        if batch_idx % args.print_freq == 0:
            logger.info("Phase:{0} -- Batch_idx:{1}/{2} -- {3:.2f} samples/sec"
                        "-- Loss:{4:.2f} -- Error1:{5:.2f}".format(
                          phase, batch_idx, len(loader),
                          err1.count / (time.time() - t), loss.avg, err1.avg))

    return loss.avg, err1.avg


def main(args):
    logger = logging.getLogger('my_log')

    dset_loaders = get_loader(args, training=True)
    model = ResNet18()
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.mo, weight_decay=args.wd)

    writer = SummaryWriter(log_dir=args.cp_dir)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.1)
    torch.save(model.state_dict(), f"{args.cp_dir}/model_init.pth.tar")

    for epoch in range(args.ep):

        logger.info('Epoch: [%d | %d]' % (epoch, args.ep))

        trainloss, trainerr1 = class_model_run('train', dset_loaders['train'],
                                               model, criterion, optimizer, args)
        logger.info('Train_Loss = {0}, Train_Err = {1}'.format(trainloss, trainerr1))
        writer.add_scalar('Train/Train_Loss', trainloss, epoch)
        writer.add_scalar('Train/Train_Err1', trainerr1, epoch)

        valloss, valerr1 = class_model_run('val', dset_loaders['val'], model,
                                           criterion, optimizer, args)
        writer.add_scalar('Val/Val_Loss', valloss, epoch)
        writer.add_scalar('Val/Val_Err1', valerr1, epoch)
        logger.info('Val_Loss = {0}, Val_Err = {1}'.format(valloss, valerr1))

        torch.save(model.state_dict(), f"{args.cp_dir}/trained_model.pth.tar")
        scheduler.step()


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
    torch.autograd.set_detect_anomaly(True)
    main(args)
