import torch
import numpy as np
import copy
from tqdm import tqdm
from torch.optim.optimizer import Optimizer, required
from copy import deepcopy
from scipy.integrate import quad


class EntropySGD(Optimizer):
    def __init__(self, params, config={}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
                        weight_decay=0, nesterov=True,
                        L=0, eps=1e-4, g0=1e-2, g1=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(EntropySGD, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for Entropy-SGD, model and criterion'
        loss = closure()

        c = self.config
        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = int(c['L'])
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']

        params = self.param_groups[0]['params']

        state = self.state
        # initialize
        if not 't' in state:
            state['t'] = 0
            state['wc'], state['mdw'] = [], []
            for w in params:
                state['wc'].append(deepcopy(w.data))
                state['mdw'].append(deepcopy(w.grad.data))

            state['langevin'] = dict(mw=deepcopy(state['wc']),
                                     mdw=deepcopy(state['mdw']),
                                     eta=deepcopy(state['mdw']),
                                     lr=0.1,
                                     beta1=0.75)

        lp = state['langevin']
        for i, w in enumerate(params):
            state['wc'][i].copy_(w.data)
            lp['mw'][i].copy_(w.data)
            lp['mdw'][i].zero_()
            lp['eta'][i].normal_()

        state['debug'] = dict(wwpd=0, df=0, dF=0, g=0, eta=0)
        llr, beta1 = lp['lr'], lp['beta1']
        g = g0*(1+g1)**state['t']

        for i in range(L):
            loss = closure()
            for wc, w, mw, mdw, eta in zip(state['wc'], params, \
                                           lp['mw'], lp['mdw'], lp['eta']):
                dw = w.grad.data

                if wd > 0:
                    dw.add_(w.data, alpha=wd)
                if mom > 0:
                    mdw.mul_(mom).add_(dw, alpha=1-damp)
                    if nesterov:
                        dw.add_(mdw, alpha=mom)
                    else:
                        dw = mdw

                # add noise
                eta.normal_()
                dw.add_(wc-w.data, alpha=-g).add_(eta, alpha=eps/np.sqrt(0.5*llr))

                # update weights
                w.data.add_(dw, alpha=-llr)
                mw.mul_(beta1).add_(w.data, alpha=1-beta1)

        if L > 0:
            # copy model back
            for i, w in enumerate(params):
                w.data.copy_(state['wc'][i])
                w.grad.data.copy_(w.data-lp['mw'][i])

        for w, mdw, mw in zip(params, state['mdw'], lp['mw']):
            dw = w.grad.data

            if wd > 0:
                dw.add_(w.data, alpha=wd)
            if mom > 0:
                mdw.mul_(mom).add_(dw, alpha=1-damp)
                if nesterov:
                    dw.add_(mdw, alpha=mom)
                else:
                    dw = mdw

            w.grad.data = dw
            # data.add_(-lr, dw)

        return loss


def _entropy(func, theta_star, gamma, mcmc_itr):

    out = []
    for k in range(mcmc_itr):
        theta = theta_star + torch.randn((theta_star.shape)).normal_(0, 1/gamma)
        out += [-func(theta)]

    return -(torch.logsumexp(torch.cat(out), 0, False) + np.log(1/mcmc_itr) + theta_star.shape[0]/2 * np.log(2*np.pi) - np.log(gamma**0.5))


@torch.no_grad()
def load_weights(model, params):
    for mp, p in zip(model.parameters(), params):
        mp.copy_(p)


def entropy(model_func, gamma, mcmc_itr):
    with torch.no_grad():
        theta_star = [p.data.clone() for p in model_func.model.parameters()]

    out = []
    for k in tqdm(range(mcmc_itr)):
        for mp, p in zip(model_func.model.parameters(), theta_star):
            mp.data.copy_(p + torch.zeros(p.shape, device=mp.data.device).normal_(0, 1/gamma))
        out += [torch.Tensor([model_func.compute_loss()[0]])]

    load_weights(model_func.model, theta_star)
    return -(torch.logsumexp(torch.cat(out), 0, False) + np.log(1/mcmc_itr) + model_func.dim/2 * np.log(2*np.pi) - np.log(gamma**0.5)).item()


def entropy_one_direc(model_func):
    criterion = model_func.criterion
    model = model_func.model
    dset_loaders = model_func.dataloader
    with torch.no_grad():
        theta_star = [copy.deepcopy(p.data) for p in model.parameters()]

    for inputs, targets in dset_loaders['train']:

        if model_func.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets) * inputs.shape[0] / len(dset_loaders['train'].dataset)
        loss.backward()

    with torch.no_grad():
        g_norm = 0.0
        grads = []
        for p in model.parameters():
            t = copy.deepcopy(p.grad.data)
            grads += [t.div_(torch.norm(t) * (len(list(model.parameters()))**0.5))]
            g_norm += torch.norm(grads[-1].reshape(-1))**2
        g_norm = g_norm**0.5
    model.zero_grad()

    @torch.no_grad()
    def func(t):
        gamma = 0.1
        load_weights(model, theta_star)
        model.eval()

        for mp, g in zip(model.parameters(), grads):
            mp.add_(g, alpha=t)

        entropy_loss = 0.0
        for inputs, targets in tqdm(dset_loaders['train']):
            if model_func.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)
            entropy_loss += criterion(outputs, targets) * inputs.shape[0] / len(dset_loaders['train'].dataset)
        return torch.exp(-entropy_loss - np.abs(t)*gamma/2 * g_norm) * g_norm

    I, e = quad(func, -1, 1, epsabs=1e-2, epsrel=1e-2)
    return -np.log(I)


def entropy_grad(model_func):
    criterion = model_func.criterion
    model = model_func.model
    dset_loaders = model_func.dataloader
    opt = EntropySGD(model.parameters(), config=dict(lr=0.1, momentum=0.0,
                     nesterov=False, weight_decay=0.0, L=20, eps=1e-4,
                     g0=1e-4, g1=1e-3))
    all_grads = torch.empty((len(dset_loaders['train']), model_func.dim))
    for i, (inputs, targets) in enumerate(tqdm(dset_loaders['train'])):
        if model_func.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        def closure():
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            return loss

        loss = opt.step(closure, model, criterion)
        if torch.isnan(loss):
            print("ERROR")
            quit()

        grads = None
        for p in model.parameters():
            if grads is None:
                grads = copy.deepcopy(p.grad.data).reshape(-1)
            else:
                grads = torch.cat([grads, copy.deepcopy(p.grad.data).reshape(-1)])
        all_grads[i, :] = grads

    return torch.mean(torch.norm(all_grads, dim=1)).item()
