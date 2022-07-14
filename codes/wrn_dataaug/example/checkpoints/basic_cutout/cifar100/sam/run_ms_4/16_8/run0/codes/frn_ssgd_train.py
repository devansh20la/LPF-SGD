import argparse
import torch
import os
from model.wide_res_net import WideResNet
from data.cifar import Cifar
from utility.initialize import initialize
from utility.step_lr import StepLR
import sys; sys.path.append("..")
import shutil
import time
from torch.utils.tensorboard import SummaryWriter
import glob
import copy
from utility.train_utils import AverageMeter, accuracy
import glob
import logging

def create_path(path, remove=False):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        if remove:
            print("deleting everything")
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print("exists")
            quit()


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
    parser.add_argument("--M", default=1, type=int)
    parser.add_argument("--remove", default=False, action='store_true')
    parser.add_argument("--lam", required=True, type=float)
    parser.add_argument("--std", required=True, type=float)
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    args.batch_size = int(args.batch_size / args.M)
    initialize(args, seed=args.seed)
    args.cp_dir = f"checkpoints/cifar10/frn_ssgd/run_ms_{args.seed}/{args.depth}_{args.width_factor}_{args.lam}_{args.std}_{args.M}"
    files = len(glob.glob(f"{args.cp_dir}/run*"))
    args.cp_dir = f"{args.cp_dir}/run{files}"
    create_path(args.cp_dir, args.remove)
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

    dataset = Cifar(args.batch_size, args.threads)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    torch.save(model.state_dict(), f"{args.cp_dir}/model_init.pth.tar")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    current_step = 0.0
    best_err = float('inf')

    for epoch in range(args.epochs):
        logger.info('Epoch: [%d | %d]' % (epoch, args.epochs))
        model.train()
        loss, err1 = AverageMeter(), AverageMeter()

        for batch_idx, batch in enumerate(dataset.train, 1):
            inputs, targets = (b.to(device) for b in batch)
            predictions = model(inputs)

            # l_i
            batch_loss = criterion(predictions, targets)
            loss.update(batch_loss, inputs.shape[0])
            with torch.no_grad():
                err1.update(float(100 - accuracy(predictions, targets, topk=(1,))[0]), inputs.shape[0])

            # g_i
            alpha = 0.0
            grads = torch.autograd.grad(batch_loss, model.parameters())
            for m,g in zip(grads, model.parameters()):
                alpha += torch.dot(m.view(-1), g.view(-1))
            alpha *= args.lam

            for m,g in zip(grads, model.parameters()):
                g.grad = torch.zeros_like(m, device=m.device)
                g.grad.copy_(g).mul_(alpha)

            # add noise to theta
            with torch.no_grad():
                noise = []
                for mp in model.parameters():
                    temp = torch.empty_like(mp, device=mp.data.device)
                    temp.normal_(0, args.std*(mp.view(-1).norm().item() + 1e-16))
                    noise.append(args.std*temp)
                    mp.data.add_(noise[-1])

            # single sample convolution approximation
            with torch.set_grad_enabled(True):
                batch_loss = criterion(model(inputs), targets)
                batch_loss.backward()

            # going back to without theta
            with torch.no_grad():
                for mp, n in zip(model.parameters(), noise):
                    mp.data.sub_(n)

            optimizer.step()
            optimizer.zero_grad()
            current_step+=1
            writer.add_scalar('alpha', alpha, current_step)
            with torch.no_grad():
                scheduler(epoch)

        
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