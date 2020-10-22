from args import get_args
import torch
import torch.nn as nn
from models import LeNet
import numpy as np
import random
from utils import get_loader, vector_to_parameter_tuple
from flat_meas import fro_norm, eig_trace, eps_flatness_model


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def make_functional(mod):
    orig_params = tuple(p.detach().requires_grad_() for p in mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names


def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)


class model_for_sharp():
    def __init__(self, model, dset_loaders, criterion, use_cuda=False):
        self.dset_loaders = dset_loaders
        self.criterion = criterion
        self.model = model
        self.functional = False
        self.use_cuda = use_cuda

    def hvp(self, v):
        model = self.model
        criterion = self.criterion

        if self.functional is not True:
            self.params, self.names = make_functional(model)
            self.functional = True

        v = vector_to_parameter_tuple(v, self.params)

        def f(*new_params):
            load_weights(model, self.names, new_params)
            loss = 0.0
            for batch_idx, inp_data in enumerate(self.dset_loaders['train'], 1):
                inputs, targets = inp_data
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, targets)
                    loss += batch_loss
            return loss

        return torch.cat([x.cpu().contiguous().view(-1, 1) for x in torch.autograd.functional.vhp(f, self.params, v)[1]])


def main(args):
    import copy
    mtr = {}

    dset_loaders = get_loader(args, training=True)
    criterion = nn.CrossEntropyLoss()

    # initialize model
    model = LeNet()
    model.load_state_dict(torch.load(f"{args.cp_dir}/trained_model.pth.tar", map_location='cpu'))
    if args.use_cuda:
        model = model.cuda()
    model.norm()

    # get parameter dimension
    d = 0
    for p in model.parameters(): d += p.numel()

    mtr['eps_flat'] = eps_flatness_model(model, criterion, dset_loaders['train'], 0.1,
                                         tol=1e-6, use_cuda=args.use_cuda, verbose=True)
    quit()
    model_func = model_for_sharp(model, dset_loaders, criterion, use_cuda=args.use_cuda)
    mtr['fro_norm'] = fro_norm(model_func, d, 1)
    mtr["eig_trace"] = eig_trace(model_func, d, 100, draws=100, use_cuda=args.use_cuda, verbose=True)

# "eps_flat", "pac_bayes", "local_entropy", "low_pass"


if __name__ == '__main__':
    args = get_args(["--exp_num", "0", "--dtype", "mnist"])

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

    main(args)
