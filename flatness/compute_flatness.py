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
    # print(f'loading model from {args.cp_dir}')
    model = resnet18(args)

    if args.use_cuda:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    model.load_state_dict(torch.load(f"{args.cp_dir}/trained_model.pth.tar"))
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # compute generalization gap
    # print("Computing generalization gap")
    train_loss_mtr = AverageMeter()
    val_loss_mtr = AverageMeter()
    err1 = {'train': AverageMeter(), 'val': AverageMeter()}
    for phase in ["train", "val"]:
        for inp_data in dset_loaders[phase]:
            inputs, targets = inp_data

            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                batch_err = accuracy(outputs, targets, topk=(1, 5)) 
                if phase == "train":
                    train_loss_mtr.update(batch_loss.item(), inputs.size(0))
                    err1['train'].update(100.0 - batch_err[0].item(), inputs.size(0))
                elif phase == "val":
                    val_loss_mtr.update(batch_loss.item(), inputs.size(0))
                    err1['val'].update(100.0 - batch_err[0].item(), inputs.size(0))

    gen_gap = val_loss_mtr.avg - train_loss_mtr.avg
    print(gen_gap, err1['train'].avg, err1['val'].avg)
    quit()
    # compute base loss and gradients
    print("Computing the base loss and gradients")
    base_loss = 0.0
    for inp_data in dset_loaders['train']:
        inputs, targets = inp_data

        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            batch_loss = -1*criterion(outputs, targets) * inputs.size(0) / len(dset_loaders['train'].dataset)
            batch_loss.backward()
            base_loss += -1*batch_loss.item()

    lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay = 0.0)

    for itr in range(10000):
        optimizer.step()

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
                movement_grad.update(curr_loss - prev_curr_loss, 1)
            else:
                prev_curr_loss = curr_loss

            if (itr+1) % 10 == 0:
                if movement_grad.avg < 0.1:
                    lr *= 2
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay = 0.0)
                    print(f"increased learning rate to :{lr}, grad_movement:{movement_grad.avg}")
                    movement_grad = AverageMeter()

            print(f"Itr: {itr}, base_loss:{base_loss:0.7f}, curr_loss:{curr_loss:0.7f}")

    params1 = parameters_to_vector(model.parameters())
    model.load_state_dict(torch.load(f"{args.cp_dir}/trained_model.pth.tar"))
    params2 = parameters_to_vector(model.parameters())

    l2_norm = torch.norm(params1 - params2)
    with open(f"{args.cp_dir}/dist_bw_params.txt", 'w') as f:
        f.write(f"gen_gap, div, flatness\n")
        f.write(f"{gen_gap},{curr_loss/base_loss}, {l2_norm}\n")


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
