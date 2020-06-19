import torch
from torch.optim.optimizer import Optimizer, required
from functools import reduce
import copy


class SGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        initial_mu=momentum, initial_lr=lr)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

        self.d = []
        for group in self.param_groups:
            d = 0
            for p in group["params"]:
                d += reduce(lambda x, y: x * y, p.shape)
            self.d.append(d)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        self.cos_dist = []
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, group in enumerate(self.param_groups):
            base_mu = group["initial_mu"]
            base_lr = group["initial_lr"]

            a, b = torch.zeros(self.d[i]).fill_(float('nan')), torch.zeros(self.d[i]).fill_(float('nan'))
            j = 0
            for p in group["params"]:
                param_state = self.state[p]
                d_p = copy.deepcopy(p.grad).view(-1).cpu()

                if 'momentum_buffer' not in param_state:
                    avg_d_p = d_p
                else:
                    avg_d_p = copy.deepcopy(param_state["momentum_buffer"]).view(-1).cpu()

                a[j:j + d_p.shape[0]] = d_p
                b[j:j + avg_d_p.shape[0]] = avg_d_p
                j += avg_d_p.shape[0]

            cos_dist = 1 - torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), 0)
            self.cos_dist += [cos_dist]

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss


class AMSGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        initial_mu=momentum, initial_lr=lr)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(AMSGD, self).__init__(params, defaults)

        self.d = []
        for group in self.param_groups:
            d = 0
            for p in group["params"]:
                d += reduce(lambda x, y: x * y, p.shape)
            self.d.append(d)

    def __setstate__(self, state):
        super(AMSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        self.cos_dist = []
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, group in enumerate(self.param_groups):
            base_mu = group["initial_mu"]
            base_lr = group["initial_lr"]

            a, b = torch.zeros(self.d[i]).fill_(float('nan')), torch.zeros(self.d[i]).fill_(float('nan'))
            j = 0
            for p in group["params"]:
                param_state = self.state[p]
                d_p = copy.deepcopy(p.grad).view(-1).cpu()

                if 'momentum_buffer' not in param_state:
                    avg_d_p = d_p
                else:
                    avg_d_p = copy.deepcopy(param_state["momentum_buffer"]).view(-1).cpu()

                a[j:j+d_p.shape[0]] = d_p
                b[j:j+avg_d_p.shape[0]] = avg_d_p
                j += avg_d_p.shape[0]

            cos_dist = 1 - torch.nn.functional.cosine_similarity(a.reshape(-1),b.reshape(-1), 0)
            # group['momentum'] = base_mu * torch.exp(-0.05 * cos_dist).item()
            # group['lr'] = base_lr * torch.exp(-0.346 * cos_dist).item()
            self.cos_dist += [cos_dist]
            group['momentum'] = base_mu * max(1 - cos_dist/2, 1e-4)
            group['lr'] = base_lr * max(1 - cos_dist/2, 1e-4)

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss
