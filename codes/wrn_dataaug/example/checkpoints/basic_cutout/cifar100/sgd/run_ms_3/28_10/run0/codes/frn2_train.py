import argparse
import torch
import os
from model.wide_res_net import WideResNet
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
import sys; sys.path.append("..")
import shutil
import time
from torch.utils.tensorboard import SummaryWriter
import glob
import copy


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
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--M", default=1, type=int)
    parser.add_argument("--lam", default=1e-3, type=float)
    parser.add_argument("--remove", default=False, action='store_true')

    args = parser.parse_args()

    args.batch_size = int(args.batch_size / args.M)
    initialize(args, seed=42)
    args.cp_dir = f"checkpoints/cifar10/frn2/run_ms_42/{args.depth}_{args.width_factor}_{args.M}"
    files = len(glob.glob(f"{args.cp_dir}/run*"))
    args.cp_dir = f"{args.cp_dir}/run{files}"
    create_path(args.cp_dir, args.remove)
    for file in glob.glob("**/*.py", recursive=True):
        if "checkpoints" in file or "data" in file or "results" in file:
            continue
        os.makedirs(os.path.dirname(f"{args.cp_dir}/codes/{file}"), exist_ok=True)
        shutil.copy(file, f"{args.cp_dir}/codes/{file}")

    writer = SummaryWriter(log_dir=args.cp_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    current_step = 0.0
    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch_idx, batch in enumerate(dataset.train, 1):
            inputs, targets = (b.to(device) for b in batch)
            predictions = model(inputs)
            
            loss = criterion(predictions, targets)
            # store various metrics
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())

            grads = torch.autograd.grad(loss.mean(), model.parameters(), create_graph = True, retain_graph = True)

            alpha = 0.0
            for mp,g in zip(model.parameters(), grads):
                alpha += torch.dot(mp.view(-1), g.view(-1))
            alpha *= args.lam


            # compute theta.T grad
            grad_grad_theta = torch.autograd.grad(
                grad, model.parameters(), grad_outputs=model.parameters(), only_inputs=True
            )

            # going back to without noise and accumulating true update
            with torch.no_grad():
                for mp, gp1, gp2 in zip(model.parameters(), grad, grad_grad_theta):
                    gp1.mul_(1 + args.lam).mul_(1/args.M)
                    gp2.mul_(args.lam).mul_(1/args.M)
                    
                    if mp.grad is None:
                        mp.grad = torch.zeros_like(mp, device=mp.device)

                    mp.grad.add_(gp1).add_(gp2)

            if batch_idx % args.M == 0:
                current_step += 1
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            scheduler(epoch)

        writer.add_scalar('train_loss', log.epoch_state["loss"] / log.epoch_state["steps"], epoch)
        writer.add_scalar('train_err1', 1 - log.epoch_state["accuracy"] / log.epoch_state["steps"], epoch)

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = criterion(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

        writer.add_scalar('val_loss', log.epoch_state["loss"] / log.epoch_state["steps"], epoch)
        writer.add_scalar('val_err1', 1 - log.epoch_state["accuracy"] / log.epoch_state["steps"], epoch)

    log.flush()
