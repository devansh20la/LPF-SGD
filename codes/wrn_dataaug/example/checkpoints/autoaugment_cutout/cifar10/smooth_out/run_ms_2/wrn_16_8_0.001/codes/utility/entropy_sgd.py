from torch.optim.optimizer import Optimizer, required
import copy
from copy import deepcopy
import torch
import numpy as np


# class EntropySGD(Optimizer):

#     def __init__(self, params, wd, lr, momentum, gamma_0, gamma_1, L=5, eta_prime = 0.1, epsilon=0.0001, alpha=0.75, nesterov=False):
#         defaults = dict(lr=lr,  wd=wd, momentum=momentum, L=L, gamma_0=gamma_0, gamma_1=gamma_1, eta_prime=eta_prime, epsilon=epsilon, alpha=alpha, nesterov=nesterov)
#         super(EntropySGD, self).__init__(params, defaults)

#     def __setgroup__(self, group):
#         super(EntropySGD, self).__setgroup__(group)

#     @torch.no_grad()
#     def step(self, closure):

#         if 'step' not in self.group:
#             self.group['step'] = 0

#         # copy current parameters of network into x_groups
#         x_groups = []
#         for group in self.param_groups:
#             group_copy = dict()
#             group_copy['params'] = deepcopy(group['params'])
#             x_groups.append(group_copy)
#         mu_groups = deepcopy(x_groups)

#         L = self.param_groups[0]['L']
#         for l in range(L):
#             with torch.enable_grad():
#                 loss = closure()
#             for group_index, group in enumerate(self.param_groups):
#                 for param_index, x_prime in enumerate(group['params']):
#                     x = x_groups[group_index]['params'][param_index]
#                     dx_prime = x_prime.grad.data
#                     if group['wd'] > 0:
#                         dx_prime.add_(x.data, alpha=group['wd'])

#                     if group['momentum'] != 0:
#                         param_group = self.group[x_prime]
#                         if 'inner_loop_velocity' not in param_group:
#                             vel = param_group['inner_loop_velocity'] = torch.clone(dx_prime).detach()
#                         else:
#                             vel = param_group['inner_loop_velocity']
#                             if l == 0:
#                                 vel.fill_(0)
#                             vel.mul_(group['momentum']).add_(dx_prime)
#                         if group['nesterov']:
#                             dx_prime.add_(vel, alpha=group['momentum'])
#                         else:
#                             dx_prime = vel

#                     gamma = group['gamma_0'] * ((1+group['gamma_1']) ** self.group['step'])
#                     dx_prime.add_(x.data - x_prime.data, alpha=-gamma)
#                     x_prime.data.add_(dx_prime, alpha=-group['eta_prime'])
#                     gaussian_noise = torch.empty_like(x_prime)
#                     gaussian_noise.normal_()
#                     x_prime.data.add_(gaussian_noise, alpha=(np.sqrt(group['eta_prime']) * group['epsilon']))
#                     mu = mu_groups[group_index]['params'][param_index]
#                     mu.data.mul_(1-group['alpha'])
#                     mu.data.add_(x_prime.data, alpha=group['alpha'])

#         for group_index, group in enumerate(self.param_groups):
#             for param_index, p in enumerate(group['params']):
#                 x = x_groups[group_index]['params'][param_index]
#                 mu = mu_groups[group_index]['params'][param_index]
#                 p.data.copy_(x.data)
#                 gradient = x.data - mu.data

#                 if group['wd'] > 0:
#                     gradient.add_(x.data, alpha=group['wd'])

#                 if group['momentum'] != 0:
#                     param_group = self.group[p]
#                     if 'outer_loop_velocity' not in param_group:
#                         vel = param_group['outer_loop_velocity'] = torch.clone(gradient).detach()
#                     else:
#                         vel = param_group['outer_loop_velocity']
#                         vel.mul_(group['momentum']).add_(gradient)
#                     if group['nesterov']:
#                         gradient.add_(vel, alpha=group['momentum'])
#                     else:
#                         gradient = vel
#                 p.data.add_(gradient, alpha=-group['lr'])

#         self.group['step'] += 1
#         return loss

class EntropySGD(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, num_batches=0, gtime=1,
                        momentum=0, momentum_sgld=0, damp=0,
                        weight_decay_sgld=1e-4, weight_decay=0, nesterov=True,
                        L=0, eps=1e-4, g0=None, g1=None, gmax=1e4, epochs=None,
                        sgld_lr=0.1, alpha_arg=0.75, gscale=True)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(EntropySGD, self).__init__(params, config)

    def __setgroup__(self, group):
        super(EntropySGD, self).__setgroup__(group)

    @torch.no_grad()
    def step(self, closure=None):
        assert (closure is not None), \
                'attach closure for Entropy-SGD, model and criterion'

        group = self.param_groups[0]
        mom = group['momentum']
        mom_sgld = group['momentum_sgld']
        mom_wd = group['weight_decay_sgld']
        wd = group['weight_decay']
        damp = group['damp']
        nesterov = group['nesterov']
        L = int(group['L'])
        eps = group['eps']
        gmax = group['gmax']
        epochs = group['epochs']
        
        sgld_lr = group['sgld_lr']
        alpha_arg = group['alpha_arg']
        gscale = group['gscale']
        
        num_batches = group['num_batches']
        gtime = group['gtime']

        # initialize
        params = group['params']
        if 'step' not in group:
            group['step'] = 0
            group['wc'], group['mdw'] = [], []

            for w in params:
                group['wc'].append(deepcopy(w.data))
                group['mdw'].append(deepcopy(w.data))

            # momentum init.
            for i, w in enumerate(params):
                group['mdw'][i].zero_()
                
            group['langevin'] = dict(mw=deepcopy(group['wc']),
                                     mdw=deepcopy(group['mdw']),
                                     eta=deepcopy(group['mdw']),
                                     lr_in=sgld_lr,
                                     alpha=alpha_arg)

        # SGLD init.
        lp = group['langevin']
        for i, w in enumerate(params):
            group['wc'][i].copy_(w.data)
            lp['mw'][i].copy_(w.data)
            lp['mdw'][i].zero_()
            lp['eta'][i].normal_()

            
        llr, alpha = lp['lr_in'], lp['alpha']
        
        g = group['g0'] * (1 + group['g1']) ** group['step']
        if group['step'] % num_batches == 0:
            print("lr = %s, llr=%s, g=%s, step=%s" % (group['lr'], llr, g, group['step']))
                
        # SGLD loop
        for i in range(L):
            with torch.enable_grad():         
                mf = closure()
            
            # g scoping
            g = group['g0'] * (1 + group['g1']) ** group['step']
            g = min(g, gmax)
    
            for wc, w, mw, mdw, eta in zip(group['wc'], params, lp['mw'], lp['mdw'], lp['eta']):
                
                dw = w.grad.data
                
                # add interaction term
                dw.add_(wc - w.data, alpha=-g)

                # momentum and weight decay
                if mom_wd > 0:
                    dw.add_(w.data, alpha=mom_wd)
                
                if mom_sgld > 0:
                    mdw.mul_(mom_sgld).add_(dw, alpha=1-damp)
                    if nesterov:
                        dw.add_(mdw, alpha=mom_sgld)
                    else:
                        dw = mdw

                # add noise
                if eps > 0.:
                    eta.normal_()
                    dw.add_(eta, alpha=eps/np.sqrt(0.5*llr))

                # update weights
                w.data.add_(dw, alpha=-llr)
                mw.mul_(alpha).add_(w.data,alpha=1-alpha)

            # calculate g0 and g1 automatically (after 1 epoch)
            if group['step'] >= gtime:
                if group['g1'] == 0:
                    group['g1'] = group['gmax']**(1/(epochs*num_batches)) - 1
                if group['g0'] == 0 and i == L-1:
                    with torch.no_grad():
                        dist_0 = 0.
                        for w1, w2 in zip(group['wc'], params):
                            dist_0 += torch.sum((w1.data - w2.data)**2)
                    group['g0'] = mf.item() / (0.5*dist_0.item())
                    print(f"# COUPLING SCHEDULE  dist at step {group['step']}: {dist_0} g0: {group['g0']}  grate: {group['g1']}")

        # copy model back
        if L > 0:
            for i, w in enumerate(params):
                w.data.copy_(group['wc'][i])
                w.grad.data.copy_(w.data - lp['mw'][i])            
            
        # update parameters
        #if group['t'] > 0:
        for w, mdw in zip(params, group['mdw']):

            dw = w.grad.data

            # momentum and weight decay
            if wd > 0:
                dw.add_(w.data, alpha=wd)
            if mom > 0:
                mdw.mul_(mom).add_(dw, alpha=1-damp)
                if nesterov:
                    dw.add_(mdw, alpha=mom)
                else:
                    dw = mdw

            if gscale:
                w.data.add_(dw, alpha=-group['lr']*g) # learning rate rescaled by g
            else:
                w.data.add_(dw, alpha=-group['lr']) # learning rate rescaled by g
                    
        # increase time-step    
        group['step'] += 1
                    
        return g


# class EntropySGD(Optimizer):
#     def __init__(self, params, config = {}):

#         defaults = dict(lr=0.01, num_batches=0, gtime=1,
#                         momentum=0, momentum_sgld=0, damp=0,
#                         weight_decay_sgld=1e-4, weight_decay=0, nesterov=True,
#                         L=0, eps=1e-4, g0=None, g1=None, gmax=1e4, epochs=None,
#                         sgld_lr=0.1, alpha_arg=0.75, gscale=True)
#         for k in defaults:
#             if config.get(k, None) is None:
#                 config[k] = defaults[k]

#         super(EntropySGD, self).__init__(params, config)

#     def __setgroup__(self, group):
#         super(EntropySGD, self).__setgroup__(group)

#     @torch.no_grad()
#     def step(self, closure=None):
#         assert (closure is not None), \
#                 'attach closure for Entropy-SGD, model and criterion'

#         group = self.param_groups[0]
#         mom = group['momentum']
#         mom_sgld = group['momentum_sgld']
#         mom_wd = group['weight_decay_sgld']
#         wd = group['weight_decay']
#         damp = group['damp']
#         nesterov = group['nesterov']
#         L = int(group['L'])
#         eps = group['eps']
#         gmax = group['gmax']
#         epochs = group['epochs']
        
#         sgld_lr = group['sgld_lr']
#         alpha_arg = group['alpha_arg']
#         gscale = group['gscale']
        
#         num_batches = group['num_batches']
#         gtime = group['gtime']

#         # initialize
#         params = group['params']
#         if 'step' not in group:
#             group['step'] = 0
#             group['wc'], group['mdw'] = [], []

#             for w in params:
#                 group['wc'].append(deepcopy(w.data))
#                 group['mdw'].append(deepcopy(w.data))

#             # momentum init.
#             for i, w in enumerate(params):
#                 group['mdw'][i].zero_()
                
#             group['langevin'] = dict(mw=deepcopy(group['wc']),
#                                      mdw=deepcopy(group['mdw']),
#                                      eta=deepcopy(group['mdw']),
#                                      lr_in=sgld_lr,
#                                      alpha=alpha_arg)

#         # SGLD init.
#         lp = group['langevin']
#         for i, w in enumerate(params):
#             group['wc'][i].copy_(w.data)
#             lp['mw'][i].copy_(w.data)
#             lp['mdw'][i].zero_()
#             lp['eta'][i].normal_()

            
#         llr, alpha = lp['lr_in'], lp['alpha']
        
#         # g = (group['g0'] * (1 + group['g1']) ** group['step'])

#         # if group['step'] % L == 0:
#         g = group['g0']*((1 - 1/(2*128))**(group['step']))
#         g = 1 / g
#         g = min(g, 1.0)
                
#         # SGLD loop
#         for i in range(L):
#             with torch.enable_grad():         
#                 mf = closure()
            
#             # g scoping
#             # g = group['g0'] * (1 + group['g1']) ** group['step']
#             #g = min(g, gmax)
    
#             for wc, w, mw, mdw, eta in zip(group['wc'], params, lp['mw'], lp['mdw'], lp['eta']):
                
#                 dw = w.grad.data
                
#                 # add interaction term
#                 dw.add_(wc - w.data, alpha=-g)

#                 # momentum and weight decay
#                 if mom_wd > 0:
#                     dw.add_(w.data, alpha=mom_wd)
                
#                 if mom_sgld > 0:
#                     mdw.mul_(mom_sgld).add_(dw, alpha=1-damp)
#                     if nesterov:
#                         dw.add_(mdw, alpha=mom_sgld)
#                     else:
#                         dw = mdw

#                 # update weights
#                 w.data.add_(dw, alpha=-llr)
#                 mw.mul_(alpha).add_(w.data,alpha=1-alpha)

#         # copy model back
#         if L > 0:
#             for i, w in enumerate(params):
#                 w.data.copy_(group['wc'][i])
#                 w.grad.data.copy_(w.data - lp['mw'][i])            
            
#         # update parameters
#         #if group['t'] > 0:
#         for w, mdw in zip(params, group['mdw']):

#             dw = w.grad.data

#             # momentum and weight decay
#             if wd > 0:
#                 dw.add_(w.data, alpha=wd)

#             if mom > 0:
#                 mdw.mul_(mom).add_(dw, alpha=1-damp)
#                 if nesterov:
#                     dw.add_(mdw, alpha=mom)
#                 else:
#                     dw = mdw

#             if gscale:
#                 w.data.add_(dw, alpha=-group['lr']*g) # learning rate rescaled by g
#             else:
#                 w.data.add_(dw, alpha=-group['lr']) # learning rate rescaled by g
                    
#         # increase time-step    
#         group['step'] += 1
                    
#         return g
