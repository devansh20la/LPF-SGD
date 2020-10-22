import torch
import torch.nn as nn
from models import ResNet18, LeNet, test
import logging
from utils import get_loader
import argparse
import time
import numpy as np
from tqdm import trange


def eval_hessian(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in trange(l):
        grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian.cpu().data.numpy()


def main(args):
    dset_loaders = get_loader(args, training=False)
    model = LeNet()
    model.load_state_dict(torch.load("checkpoints/cifar10/sgd_0.01_0.9_64_0.0/run_ms_1/trained_model.pth.tar",map_location='cpu'))
    model.train()
    model.norm()

    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    for batch_idx, inp_data in enumerate(dset_loaders['test'], 1):

        inputs, targets = inp_data

        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        break

    outputs = model(inputs)
    batch_loss = criterion(outputs, targets)

    loss_grad = torch.autograd.grad(batch_loss, model.parameters(), create_graph=True)
    grads2 = eval_hessian(loss_grad, model)  # second order gradient
    np.save("hess_norm.npy", grads2)


def main2():
    model = test()
    cri = nn.MSELoss()
    target = torch.Tensor([10])

    x = torch.randn(1, 1)
    loss = cri(model(x), target)

    loss_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads2 = eval_hessian(loss_grad, model)  # second order gradient
    print(grads2)
    print(2*x**2, 2*x)


if __name__ == '__main__':
    main2()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dir', type=str, default='.')
    # parser.add_argument('--dtype', type=str, default="cifar10", help='Data type')
    # parser.add_argument('--bs', type=int, default=2, help='batch size')
    # parser.add_argument('--cp_dir', type=str)
    # parser.add_argument('--print_freq', type=int, default=50)

    # args = parser.parse_args()

    # if args.dtype == 'cifar10':
    #     args.num_classes = 10
    # elif args.dtype == 'cifar100':
    #     args.num_classes = 100
    # elif args.dtype == 'imagenet':
    #     args.num_classes = 1000
    # elif args.dtype == 'mnist':
    #     args.num_classes = 10
    # args.data_dir = f"{args.dir}/data/{args.dtype}"
    # args.use_cuda = torch.cuda.is_available()

    # # Logging tools
    # logger = logging.getLogger('my_log')
    # logger.setLevel(logging.INFO)

    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # logger.addHandler(console)
    # logger.info(args)
    # main(args)
