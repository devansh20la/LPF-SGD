from args import get_args
import torch
import torch.nn as nn
from models import resnet18_narrow as resnet18
import numpy as np
import random
from utils import get_loader
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
from utils.train_utils import AverageMeter, accuracy
import copy


def main(args):
    movement_grad = AverageMeter()

    dset_loaders = get_loader(args, training=True)

    print(f'loading model from {args.cp_dir}')
    model = resnet18(args)

    if args.use_cuda:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    model.load_state_dict(torch.load(f"{args.cp_dir}/trained_model.pth.tar"))
    criterion = nn.CrossEntropyLoss()

    # compute generalization gap
    print("Computing generalization gap")
    model.eval()

    base_loss = 0.0
    loss_mtr = {'train': AverageMeter(), 'val':  AverageMeter()}
    err_mtr = {'train': AverageMeter(), 'val': AverageMeter()}

    for phase in ["train", "val"]:
        for inp_data in dset_loaders[phase]:
            inputs, targets = inp_data

            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)

                batch_loss = criterion(outputs, targets)
                batch_err = accuracy(outputs, targets, topk=(1, ))

                loss_mtr[phase].update(batch_loss.item(), inputs.size(0))
                err_mtr[phase].update(100.0 - batch_err[0].item(), inputs.size(0))

                if phase == 'train':
                    batch_loss = -1*batch_loss * inputs.size(0) / len(dset_loaders['train'].dataset)
                    batch_loss.backward()
                    base_loss += -1*batch_loss.item()

    gen_gap = loss_mtr['val'].avg - loss_mtr['train'].avg

    lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0,
                          weight_decay=0.0)
    a = 0.0

    for itr in range(10000):
        optimizer.step()
        a += lr

        curr_loss = 0.0
        for inp_data in dset_loaders['train']:
            inputs, targets = inp_data

            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets) * inputs.size(0) / len(dset_loaders['train'].dataset)
                curr_loss += batch_loss.item()

        if curr_loss/base_loss > 2:
            print(f"Itr: {itr}, base_loss:{base_loss:0.7f}, curr_loss:{curr_loss:0.7f}")
            break
        else:
            if itr > 0:
                movement_grad.update(curr_loss/prev_curr_loss, 1)
            else:
                prev_curr_loss = curr_loss

            if (itr+1) % 10 == 0:
                if movement_grad.avg < 1.1:
                    lr *= 2
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay = 0.0)
                    print(f"increased learning rate to :{lr}, grad_movement:{movement_grad.avg}")
                    movement_grad = AverageMeter()
                else:
                    print(movement_grad.avg)

            total_norm = 0.0
            for p in model.parameters():
                print(p.grad.data)
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

            print(f"Itr: {itr}, base_loss:{base_loss:0.7f}, curr_loss:{curr_loss:0.7f}, norm:{total_norm:0.07f}")
            if total_norm == np.nan:
                print("Gradient norm is nan")
                quit()

    flatness = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        flatness += param_norm.item() ** 2
    flatness = flatness ** (1. / 2)
    flatness = a*flatness

    with open(f"{args.cp_dir}/dist_bw_params.txt", 'w') as f:
        f.write(f"train_loss, train_err, val_loss, val_err, flatness, div\n")
        f.write(f"{loss_mtr['train'].avg}, {err_mtr['train'].avg}, {loss_mtr['val'].avg}, {err_mtr['val'].avg}, {flatness}, {curr_loss/base_loss}\n")


if __name__ == '__main__':
    args = get_args()

    args.bs = 1024

    # Random seed
    random.seed(args.ms)
    torch.manual_seed(args.ms)
    np.random.seed(args.ms)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.ms)

    # Intialize directory and create path
    args.cp_dir = f"{args.dir}/checkpoints/{args.n}/run_ms_{args.ms}"
    args.bs = 1024

    main(args)
