from args import get_args
import torch
import torch.nn as nn
from models import resnet18_narrow
import numpy as np
import random
from utils import get_loader
from flat_meas import fro_norm, entropy_one_direc, \
                      entropy_grad, eig_trace, eps_flatness, \
                      pac_bayes, entropy, low_pass, low_pass2, fim, shannon_entropy
from utils.train_utils import AverageMeter, accuracy
import copy
import pickle
import time
import os
import logging


class model_for_sharp():
    def __init__(self, model, dset_loaders, criterion, use_cuda=False):
        self.dataloader = dset_loaders
        self.criterion = criterion
        self.model = model
        self.functional = False
        self.use_cuda = use_cuda
        self.dim = 0   # get parameter dimension
        for p in self.model.parameters():
            self.dim += p.numel()

        self.train_loss, self.train_acc = self.compute_loss()
        self.val_loss, self.val_acc = self.compute_loss(phase='val')

    def compute_loss(self, phase='train', ascent_stats= False):
        self.zero_grad()
        loss_mtr = AverageMeter()
        acc_mtr = AverageMeter()

        for inputs, targets in self.dataloader[phase]:
            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            with torch.set_grad_enabled(ascent_stats):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                batch_acc = accuracy(outputs, targets, topk=(1,))[0]
                acc_mtr.update(batch_acc, inputs.size(0))

            loss_mtr.update(loss, inputs.shape[0])

            if ascent_stats:
                loss *= (-inputs.shape[0] / len(self.dataloader[phase].dataset))
                loss.backward()

        if ascent_stats:
            theta_star_params = []
            theta_star_grads = []
            for p in self.model.parameters():
                theta_star_params.append(copy.deepcopy(p))
                theta_star_grads.append(copy.deepcopy(p.grad.data))

            return theta_star_params, theta_star_grads, loss_mtr.avg.item(), acc_mtr.avg.item()
        else:
            return loss_mtr.avg.item(), acc_mtr.avg.item()

    def hvp(self, vec):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        self.zero_grad()
        hessian_vec_prod = None
        phase = 'train'

        for inputs, targets in self.dataloader[phase]:
            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            grad_dict = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True
            )
            grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
            grad_grad = torch.autograd.grad(
                grad_vec, self.model.parameters(), grad_outputs=vec, only_inputs=True
            )
            if hessian_vec_prod is not None:
                hessian_vec_prod += torch.cat([g.contiguous().view(-1) for g in grad_grad])
            else:
                hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in grad_grad])

            self.zero_grad()

        return hessian_vec_prod/len(self.dataloader[phase])

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()


def main(args):
    logger = logging.getLogger('my_log')
    dset_loaders = get_loader(args, training=True)
    criterion = nn.CrossEntropyLoss()

    # initialize model
    model = resnet18_narrow(args)
    model.load_state_dict(torch.load(f"{args.cp_dir}/trained_model.pth.tar", map_location='cpu'))
    if args.use_cuda:
        model = model.cuda()
    model.eval()
    model.norm()

    model_func = model_for_sharp(model, dset_loaders, criterion, use_cuda=args.use_cuda)

    if os.path.isfile(f"{args.cp_dir}/measures.pkl"):
        with open(f"{args.cp_dir}/measures.pkl", 'rb') as f:
            mtr = pickle.load(f)
        logger.info(np.abs(mtr["train_loss"] - model_func.train_loss))
        if np.abs(mtr["train_loss"] - model_func.train_loss) > 1e-6:
            logger.info("STUPIDITY HAPPENDED HERE")
            quit()
    else:
        mtr = {}
        mtr["train_loss"] = model_func.train_loss
        mtr["val_loss"] = model_func.val_loss
        mtr["train_acc"] = model_func.train_acc
        mtr["val_acc"] = model_func.val_acc

    if mtr["train_loss"] > 0.01:
        save(mtr)
        quit()

    # compute various measures
    if 'shannon_entropy' not in mtr.keys():
        t = time.time()
        mtr['shannon_entropy'] = shannon_entropy(model_func)
        logger.info(f"time required for shannon_entropy:{time.time() - t}")
        save(mtr)

    if 'eps_flat' not in mtr.keys():
        t = time.time()
        mtr['eps_flat'] = eps_flatness(model_func, 0.1, tol=1e-6, use_cuda=args.use_cuda, verbose=True)
        logger.info(f"time required for eps_flat:{time.time() - t}")
        save(mtr)

    if 'pac_bayes' not in mtr.keys():
        t = time.time()
        model_init = resnet18_narrow(args)
        model_init.load_state_dict(torch.load(f"{args.cp_dir}/model_init.pth.tar", map_location='cpu'))
        if args.use_cuda:
            model_init = model_init.cuda()
        model_init.norm()
        mtr["pac_bayes"] = pac_bayes(model_func, 10, 0.1, theta_init=model_init.parameters(), tol=1e-6, verbose=True)
        logger.info(f"time required for pac_bayes:{time.time() - t}")
        del model_init
        save(mtr)

    if 'fro_norm' not in mtr.keys():
        t = time.time()
        mtr['fro_norm'] = fro_norm(model_func, 10)
        logger.info(f"time required for fro_norm:{time.time() - t}")
        save(mtr)

    if 'fim' not in mtr.keys():
        t = time.time()
        mtr['fim'] = fim(model_func)
        logger.info(f"time required for fim:{time.time() - t}")
        save(mtr)

    if 'local_entropy' not in mtr.keys():
        t = time.time()
        mtr['local_entropy'] = entropy(model_func, 100, 100)
        logger.info(f"time required for entropy:{time.time() - t}")
        save(mtr)

    if 'low_pass' not in mtr.keys():
        t = time.time()
        mtr['low_pass'] = low_pass(model_func, 0.01, 100)
        logger.info(f"time required for low pass:{time.time() - t}")
        save(mtr)

    if 'local_entropy_grad_norm' not in mtr.keys():
        t = time.time()
        e = entropy_grad(model_func)
        mtr["local_entropy_grad_norm"] = e
        logger.info(f"time required for entropy_grad:{time.time() - t}")
        save(mtr)


    if 'eig_trace' not in mtr.keys():
        t = time.time()
        e = eig_trace(model_func, 100, draws=2, use_cuda=args.use_cuda, verbose=True)
        mtr["eig_trace"] = e.sum()
        logger.info(f"time required for eig:{time.time() - t}")
        with open(f"{args.cp_dir}/eig_val.npy", 'wb') as f:
            np.save(f, e)

    save(mtr)
    logger.info(mtr)


def save(mtr):
    with open(f"{args.cp_dir}/measures.pkl", 'wb') as f:
        pickle.dump(mtr, f)


if __name__ == '__main__':
    args = get_args()

    # Random seed
    random.seed(args.ms)
    torch.manual_seed(args.ms)
    np.random.seed(args.ms)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.ms)
        torch.backends.cudnn.benchmark = True

    # Intialize directory and create path
    args.bs = 256
    args.cp_dir = f"{args.dir}/checkpoints/{args.n}/run_ms_{args.ms}"

    # Logging tools
    logger = logging.getLogger('my_log')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f'{args.cp_dir}/sharpness_logs.log')
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(args)
    main(args)
