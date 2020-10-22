from args import get_args
import torch
import torch.nn as nn
from models import resnet18_narrow as resnet18
import numpy as np
import random
from utils import get_loader
import torch.optim as optim
from utils.train_utils import AverageMeter, accuracy
import copy
from hessian_eigenthings import compute_hessian_eigenthings


def load_grad(model, grads):
    for p, g in zip(model.parameters(), grads):
        p.grad.data = g


def alpha_flatness(args):

    dset_loaders = get_loader(args, training=True)

    print(f'loading model from {args.cp_dir}')
    model = resnet18(args)

    if args.use_cuda:
        model.cuda()

    model.load_state_dict(torch.load(f"{args.cp_dir}/trained_model.pth.tar"))
    criterion = nn.CrossEntropyLoss()

    # compute generalization gap
    print("Computing gradients")
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

    total_norm = 0.0
    grads = []
    for p in model.parameters():
        grads.append(copy.deepcopy(p.grad.data))
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    if total_norm == np.nan:
        print("Gradient norm is nan")
        quit()

    a = 0.001
    a_range = [float('-inf'), float('inf')]
    optimizer = optim.SGD(model.parameters(), a, 0.0, 0.0)

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

        print(f"Itr: {itr}, base_loss:{base_loss:0.7f}, curr_loss:{curr_loss:0.7f}, a:{a}, div:{curr_loss/base_loss:0.7f}")

        if curr_loss / base_loss < 1.99:
            a_range[0] = a
            if a_range[1] < float('inf'):
                a = np.array(a_range).mean()
            else:
                a *= 2
        elif curr_loss / base_loss > 2.001:
            a_range[1] = a
            if a_range[0] > float('-inf'):
                a = np.array(a_range).mean()
            else:
                a /= 2
                a += a/2
        else:
            print(f"Itr: {itr}, base_loss:{base_loss:0.7f}, curr_loss:{curr_loss:0.7f}, a:{a}, div:{curr_loss/base_loss:0.7f}")
            break
        model.load_state_dict(torch.load(f"{args.cp_dir}/trained_model.pth.tar"))
        load_grad(model, grads)
        optimizer = optim.SGD(model.parameters(), a, 0.0, 0.0)

    flatness = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        flatness += param_norm.item() ** 2
    flatness = flatness ** (1. / 2)
    flatness = a*flatness

    with open(f"{args.cp_dir}/alpha_flatness.txt", 'w') as f:
        f.write(f"train_loss, train_err, val_loss, val_err, flatness, div\n")
        f.write(f"{loss_mtr['train'].avg}, {err_mtr['train'].avg}, {loss_mtr['val'].avg}, {err_mtr['val'].avg}, {flatness}, {curr_loss/base_loss}\n")

    return 0


def eig_den(args):
    # data initialization
    dset_loaders = get_loader(args, training=True)

    # model initialization
    model = resnet18(args)
    if args.use_cuda:
        model.cuda()

    model.load_state_dict(torch.load(f"{args.cp_dir}/trained_model.pth.tar"))
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    t = time.time()
    eig_vals = compute_hessian_eigenthings(model, dset_loaders["train"], criterion, 100,
                                           mode="lanczos", full_dataset=True,
                                           use_gpu=args.use_cuda)
    print(eig_vals)
    print(f"{time.time() - t}")


if __name__ == '__main__':
    args = get_args(["--exp_num", "0"])

    # Random seed
    random.seed(args.ms)
    torch.manual_seed(args.ms)
    np.random.seed(args.ms)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.ms)
        torch.backends.cudnn.benchmark = True

    # Intialize directory and create path
    args.bs = 128
    args.cp_dir = f"{args.dir}/checkpoints/{args.n}/run_ms_{args.ms}"

    # alpha_flatness(args)
    import time
    t = time.time()
    eig_den(args)
    print(f"time:{time.time() -t }")
