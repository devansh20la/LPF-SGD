from torch.optim import Optimizer
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

def increase_g_cosine(self):
    self.g = -0.5 * self.gmax * (-1. + math.cos(math.pi * self.last_epoch / self.Tmax))
    self.last_epoch += 1

class EntropySGD(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, droplr=0., drop_mstones=None, num_batches=0,
                        momentum=0, momentum_sgld=0, damp=0,
                        weight_decay=0, nesterov=True,
                        L=0, eps=1e-4, g0=None, g1=None, gmax=1e4, epochs=None,
                        sgld_lr=0.1, alpha_arg=0.75, gscale=True)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(EntropySGD, self).__init__(params, config)
        self.config = config

    def step(self, closure=None, model=None, model2=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for Entropy-SGD, model and criterion'
        
        #mf, merr = closure()
        #mf, merr = 0,0

        # take params
        c = self.config
        mom = c['momentum']
        mom_sgld = c['momentum_sgld']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = int(c['L'])
        eps = c['eps']
        gmax = c['gmax']
        epochs = c['epochs']
        
        sgld_lr = c['sgld_lr']
        alpha_arg = c['alpha_arg']
        gscale = c['gscale']
        
        num_batches = c['num_batches']
        droplr = c['droplr']
        if droplr > 1. and c['drop_mstones'] is not None:
            drop_mstones = [int(h) for h in c['drop_mstones'].split('_')[1:]]

        # initialize
        params = self.param_groups[0]['params']
        state = self.state
        if not 't' in state:
            state['t'] = 0
            state['wc'], state['mdw'] = [], []
            state['lr'] = c['lr'] # initialize lr for scheduler

            for w in params:
                state['wc'].append(deepcopy(w.data))
                state['mdw'].append(deepcopy(w.data))

            # momentum init.
            for i, w in enumerate(params):
                state['mdw'][i].zero_()
                
            state['langevin'] = dict(mw=deepcopy(state['wc']),
                                     mdw=deepcopy(state['mdw']),
                                     eta=deepcopy(state['mdw']),
                                     lr_in=sgld_lr,
                                     alpha=alpha_arg)

        # SGLD init.
        lp = state['langevin']
        for i, w in enumerate(params):
            state['wc'][i].copy_(w.data)
            lp['mw'][i].copy_(w.data)
            lp['mdw'][i].zero_()
            lp['eta'][i].normal_()

        llr, alpha = lp['lr_in'], lp['alpha']
                
        # SGLD loop
        norm_grad = 0.
        for i in range(L):
                        
            mf, merr = closure()
            
            if c['g1'] == 0:
                c['g1'] = c['gmax']**(1/(epochs*L)) - 1
            if c['g0'] == 0 and i == 1:
                with torch.no_grad():
                    dist_0 = 0.
                    #for w1, w2 in zip(state['wc'], params):
                    for w1, w2 in zip(state['wc'], model2.parameters()):
                        dist_0 += torch.sum((w1.data - w2.data)**2)
                c['g0'] = mf / (0.5*dist_0.item())
                print(f"# COUPLING SCHEDULE  dist0: {dist_0} g0: {c['g0']}  grate: {c['g1']}")

            # g scoping
            g = c['g0'] * (1 + c['g1']) ** state['t']
            #g = min(g, gmax)
    
            #for wc, w, mw, mdw, eta in zip(state['wc'], params, lp['mw'], lp['mdw'], lp['eta']):
            for wc, w, mw, mdw, eta in zip(state['wc'], model2.parameters(), lp['mw'], lp['mdw'], lp['eta']):

                with torch.no_grad():
                    norm_grad += torch.sum((w.grad.data)**2)

                dw = w.grad.data
                
                # add interaction term
                dw.add_(-g, wc - w.data)

                # momentum and weight decay
                if wd > 0:
                    dw.add_(wd, w.data)
                if mom_sgld > 0:
                    mdw.mul_(mom_sgld).add_(1-damp, dw)
                    if nesterov:
                        dw.add_(mom_sgld, mdw)
                    else:
                        dw = mdw
                
                #dw.add_(-g, wc-w.data).add_(eps/np.sqrt(0.5*llr), eta)

                # add noise
                if eps > 0.:
                    eta.normal_()
                    dw.add_(eps/np.sqrt(0.5*llr), eta)

                # update weights
                w.data.add_(-llr, dw)
                mw.mul_(alpha).add_(1-alpha, w.data)

        # calculate gradient norm
        with torch.no_grad():
            num_params = sum(p.numel() for p in params)
            norm_grad = np.sqrt((norm_grad.item()/num_params/L))

        # calculate distance
        with torch.no_grad():
            #num_params = sum(p.numel() for p in state['wc'])
            #print("num_params2", num_params)
            dist = 0.
            for w1, w2 in zip(state['wc'], lp['mw']):
                dist += torch.sum((w1.data - w2.data)**2)
            dist = np.sqrt((dist.item()/num_params))

        # copy model back
        if L > 0:
            for i, w in enumerate(params):
                w.data.copy_(state['wc'][i])
                #w.grad.data.copy_(w.data - lp['mw'][i])            
                w.grad = deepcopy(w.data - lp['mw'][i])            

        # update parameters
        for w, mdw in zip(params, state['mdw']):

            dw = w.grad.data

            # momentum and weight decay
            if wd > 0:
                dw.add_(wd, w.data)
            if mom > 0:
                mdw.mul_(mom).add_(1-damp, dw)
                if nesterov:
                    dw.add_(mom, mdw)
                else:
                    dw = mdw

            if gscale:
                w.data.add_(-state['lr']*g, dw) # learning rate rescaled by g
            else:
                w.data.add_(-state['lr'], dw) # learning rate rescaled by g
                    
                    
        # increase time-step    
        state['t'] += 1

        if state['t'] % num_batches == 0:
            print("lr = %s, llr=%s, g=%s, step=%s" % (state['lr'], llr, g, state['t']))

        #lr drop
        if droplr:
            if droplr > 1. and (state['t']/num_batches) in drop_mstones:
                state['lr'] /= droplr
                #lp['lr_in'] /= droplr
                print("New lr = %s, llr=%s, g=%s, step=%s" % (state['lr'], llr, g, state['t']))
                #print("Lr drop: new lr = %s (g=%s, state['t'])=%s" % (llr, g, state['t']))
            elif droplr < 1. and ( (state['t']/num_batches) % 2 == 0 ):
                state['lr'] *= droplr
                print("New lr = %s, llr=%s, g=%s, step=%s" % (state['lr'], llr, g, state['t']))

                    
        return mf, merr, dist, norm_grad, state['lr'], g
