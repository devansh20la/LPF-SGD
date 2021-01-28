from torch.optim import Optimizer
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

class EntropySGD(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, droplr=0., drop_mstones=None, num_batches=0,
                        momentum=0, momentum_sgld=0, damp=0,
                        weight_decay=0, nesterov=True,
                        L=0, eps=1e-4, g0=1e-2, g1=0, gmax=float("inf"),
                        sgld_lr=0.1, alpha_arg=0.75, gscale=True)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(EntropySGD, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, criterion=None, train_loader=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for Entropy-SGD, model and criterion'
        
        #mf, merr = closure()
        """data, target = next(iter(train_loader.single_loader()))
        target = target.to(model.master_device)
        model.zero_grad()
        output = model(data)
        l = criterion(output, target)
        l.backward()
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(target.view_as(pred)).sum().item()
        mf, merr = l.item(), acc
        """
        mf, merr = 0,0

        c = self.config
        mom = c['momentum']
        mom_sgld = c['momentum_sgld']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = int(c['L'])
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']
        gmax = c['gmax']
        
        sgld_lr = c['sgld_lr']
        alpha_arg = c['alpha_arg']
        gscale = c['gscale']
        
        num_batches = c['num_batches']
        droplr = c['droplr']
        if droplr > 1. and c['drop_mstones'] is not None:
            drop_mstones = [int(h) for h in c['drop_mstones'].split('_')[1:]]

        #params = self.param_groups[0]['params']
        #params = model.parameters()
        
        state = self.state
        # initialize
        if not 't' in state:
            state['t'] = 0
            state['lr'] = c['lr'] # initialize lr for scheduler

            state['xa'], state['mdw'] = [], []
            for w in model.parameters():
                state['xa'].append(deepcopy(w.data))
                state['mdw'].append(deepcopy(w.data))

            for i, w in enumerate(model.parameters()):
                state['mdw'][i].zero_()
                
            state['langevin'] = dict(za=deepcopy(state['xa']),
                                     mdw=deepcopy(state['mdw']),
                                     eta=deepcopy(state['mdw']),
                                     lr_in=sgld_lr,
                                     alpha=alpha_arg)

        # SGLD init.
        lp = state['langevin']
        for i, w in enumerate(model.parameters()):
            state['xa'][i].copy_(w.data)
            lp['za'][i].copy_(w.data)
            #lp['mdw'][i].zero_()
            lp['eta'][i].normal_()

        llr, alpha = lp['lr_in'], lp['alpha']
                
        # g scoping
        g = g0 * (1 + g1) ** state['t']
        #g = g0 * (1 + 0.5/num_batches) ** state['t']
        g = min(g, gmax)
        # increase time-step    
        state['t'] += 1

        # sgld loop
        for i in range(L):
            
            #for wg in model.parameters():
            #    wg.grad.data.zero_()
            #    #print("CIAOOOOO")
            
            #f, _ = closure()
            data, target = next(iter(train_loader.single_loader()))
            target = target.to(model.master_device)
            model.zero_grad()
            output = model(data)
            l = criterion(output, target)
            l.backward()
            #pred = output.argmax(dim=1, keepdim=True)
            #acc = pred.eq(target.view_as(pred)).sum().item()
            
            for xa, w, za, mdw, eta in zip(state['xa'], model.parameters(), lp['za'], lp['mdw'], lp['eta']):
                
                dw = w.grad.data

                #dw.add_(-g, xa-w.data).add_(eps/np.sqrt(0.5*llr), eta)
                dw.add_(-g, xa-w.data)
                # add noise
                if eps > 0.:
                    eta.normal_()
                    dw.add_(eps/np.sqrt(0.5*llr), eta)

                if mom_sgld > 0:
                    if wd > 0:
                        dw.add_(wd, w.data)
                    mdw.mul_(mom_sgld).add_(1-damp, dw)
                    if nesterov:
                        dw.add_(mom_sgld, mdw)
                    else:
                        dw = mdw
                    
                # update weights
                w.data.add_(-llr, dw)
                za.mul_(alpha).add_(1-alpha, w.data)

                
        with torch.no_grad():
            num_params = sum(p.numel() for p in model.parameters())
            #print("num_params1", num_params)
            norm_grad = 0.
            for ww in model.parameters():
                norm_grad += torch.sum((ww.grad.data)**2)
            norm_grad = np.sqrt((norm_grad.item()/num_params))

        with torch.no_grad():
            num_params = sum(p.numel() for p in state['xa'])
            #print("num_params2", num_params)
            dist = 0.
            for w1, w2 in zip(state['xa'], lp['za']):
                dist += torch.sum((w1 - w2)**2)
            dist = np.sqrt((dist.item()/num_params))

        if L > 0:
            # copy model back
            for i, w in enumerate(model.parameters()):
                w.data.copy_(state['xa'][i])
                w.grad.data.copy_(w.data - lp['za'][i])            
            
        for w, mdw in zip(model.parameters(), state['mdw']):

            dw = w.grad.data

            if mom > 0:
                if wd > 0:
                    dw.add_(wd, w.data)
                mdw.mul_(mom).add_(1-damp, dw)
                if nesterov:
                    dw.add_(mom, mdw)
                else:
                    dw = mdw

            if gscale:
                w.data.add_(-state['lr']*g, dw) # learning rate rescaled by g
            else:
                w.data.add_(-state['lr'], dw) # learning rate rescaled by g
                 
        #lr drop
        if droplr > 1. and (state['t']/num_batches) in drop_mstones:
            state['lr'] /= droplr
            lp['lr_in'] /= droplr
            print("New lr = %s, llr=%s, g=%s, step=%s" % (state['lr'], llr, g, state['t']))
            #print("Lr drop: new lr = %s (g=%s, state['t'])=%s" % (llr, g, state['t']))
        elif droplr < 1. and ( (state['t']/num_batches) % 2 == 0 ):
            state['lr'] *= droplr
            print("New lr = %s, llr=%s, g=%s, step=%s" % (state['lr'], llr, g, state['t']))

        if state['t'] % num_batches == 0:
            print("lr = %s, llr=%s, g=%s, step=%s" % (state['lr'], llr, g, state['t']))                
    
                    
        return mf, merr, dist, norm_grad, state['lr'], g
