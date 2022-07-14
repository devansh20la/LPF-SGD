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
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--M", default=1, type=int)
    parser.add_argument("--remove", default=False, action='store_true')
    parser.add_argument("--lam_1", default=0.01, type=float)
    parser.add_argument("--lam_2", default=0.01, type=float)
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    initialize(args, seed=args.seed)
    args.cp_dir = f"checkpoints/cifar10/frn/run_ms_{args.seed}/{args.depth}_{args.width_factor}_{args.lam_1}_{args.lam_2}_{args.M}"
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
        # get required params
        alpha = args.lam_1
        beta = 0.0
        norm = 0.0
        for p in model.parameters():
            norm += p.data.norm(2).item()**2
        alpha *= norm

        for batch_idx, batch in enumerate(dataset.train, 1):
            inputs, targets = (b.to(device) for b in batch)
            predictions = model(inputs)

            # l_i
            batch_loss = criterion(predictions, targets)
            loss.update(batch_loss, inputs.shape[0])
            
            # store various metrics
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                err1.update(float(100 - accuracy(predictions, targets, topk=(1,))[0]), inputs.shape[0])

            # g_i
            batch_loss.backward()
            with torch.no_grad():
                noise = []
                for mp in model.parameters():
                    e_w = mp.grad * args.lam_2
                    mp.add_(e_w)
                    noise.append(e_w)
                    mp.grad.mul_(1 - alpha)

            # l_hat_i
            predictions = model(inputs)
            new_batch_loss = criterion(predictions, targets)
            
            # g_hat_i
            (new_batch_loss*alpha).backward()

            # going back to without noise
            with torch.no_grad():
                beta += 2*args.lam_1*(new_batch_loss.item() - batch_loss.item())
                for mp, n in zip(model.parameters(), noise):
                    mp.data.sub_(n)

            current_step += 1
            for mp, n in zip(model.parameters(), noise):
                mp.grad.add_(mp*(beta/args.M))

            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('beta', beta, current_step)
            writer.add_scalar('alpha', alpha, current_step)

            beta = 0.0
            alpha = args.lam_1
            norm = 0.0
            for p in model.parameters():
                norm += p.data.norm(2).item()**2
            alpha *= norm

        scheduler(epoch)
        writer.add_scalar('train_loss', loss.avg, epoch)
        writer.add_scalar('train_err1', err1.avg, epoch)
        logger.info('Train_Loss = {0}, Train_Err = {1}'.format(loss.avg, err1.avg))

        model.eval()

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                batch_loss = criterion(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets

        writer.add_scalar('val_loss', loss.avg, epoch)
        writer.add_scalar('val_err1', err1.avg, epoch)
        logger.info('Val_Loss = {0}, Val_Err = {1}'.format(loss.avg, err1.avg))

        if err1.avg < best_err:
            torch.save(model.state_dict(), f"{args.cp_dir}/best_model.pth.tar")
            best_err = err1.avg