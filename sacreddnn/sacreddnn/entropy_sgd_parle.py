"""import torchnet as tnt
from torchvision.datasets.mnist import MNIST

def get_iterator(mode):
    ds = MNIST(root='~/data/parle/MNIST', download=True, train=mode)
    data = getattr(ds, 'data' if mode else 'test_data')
    labels = getattr(ds, 'targets' if mode else 'test_labels')
    tds = tnt.dataset.TensorDataset([data, labels])
    return tds.parallel(batch_size=128,
            num_workers=0, shuffle=mode, pin_memory=True)
"""
def train_esgd2(loss, model, train_loader, optimizer, args, file=None, s={}):
    model.train()
    t = tqdm(train_loader) # progress bar integration
    train_loss, accuracy, ndata = 0, 0, 0
    
    """train_ds = get_iterator(True)
    train_iter = train_ds.__iter__()
    t = tqdm(train_iter)
    """
    
    # ancora non inseriti - TODOS: capire differenza con esgd standard; piattezza resnet18; piattezza resnet110
    epsilon = args.sgld_noise
    sgld_lr = args.sgld_lr
    gscale = args.gscale
    
    # params
    L = args.L
    mom, alpha = args.mom_sgld, args.alpha

    droplr = args.droplr
    drop_mstones = [int(h) for h in args.drop_mstones.split('_')[1:]]
    num_batches = len(train_loader)

    for counter, (data1, target1) in enumerate(t):
                
        for l in range(L):
            data, target = next(iter(train_loader.single_loader()))
            target = target.to(model.master_device)
            #optimizer.zero_grad()
            model.zero_grad() # altra possibilità
            output = model(data)
            l = loss(output, target)
            l.backward()
            
            if not 't' in s:
                s['t'] = 0
                s['lr'] = args.lr

                for k in ['za', 'muy', 'mux', 'xa']: #, 'x', 'cache']:
                    s[k] = {}

                for p in model.parameters():
                    for k in ['xa', 'za', 'mux', 'muy']:
                        s[k][p] = p.data.clone()

                    s['muy'][p].zero_()
                    s['mux'][p].zero_()

                    #s['x'][p] = p.data.cpu().numpy()
                    #s['cache'][p] = p.data.cpu().numpy()
                    #za, muy, mux, xa = s['za'], s['muy'], s['mux'], s['xa']
                    
            time = s['t']
            lr = s['lr']
                                    
            za, muy, mux, xa = s['za'], s['muy'], s['mux'], s['xa']
            #x, cache = s['x'], s['cache']

            #gamma = args.g*(1 + 0.5/num_batches)**(time // L)
            gamma = args.g*(1 + args.grate)**(time // L)
            gamma = min(gamma, 1)

            # entropy-sgd iterations
            for p in model.parameters():
                p.grad.data.add_(gamma, p.data - xa[p])

                if mom > 0.:
                    muy[p].mul_(mom).add_(p.grad.data)
                    p.grad.data.add_(muy[p])
                    
                p.data.add_(-lr, p.grad.data)
    
                za[p].mul_(alpha).add_(1-alpha, p.data)

            # ATTENZIONE QUESTO t era nel loop !!!
            s['t'] += 1

        # calculate distance
        with torch.no_grad():
            num_params = sum(p.numel() for p in model.parameters())
            #print("num_params2", num_params)
            dist = 0.
            for p in model.parameters():
                dist += torch.sum((xa[p] - za[p])**2)
            dist = np.sqrt((dist.item()/num_params))
            print("t=%s, dist=%s, g=%s, nb=%d" % (s['t'], dist, gamma, num_batches))
            file.write('{}\n'.format(dist))
            
        for p in model.parameters():

            # elastic-sgd term
            
            p.data.zero_()
            p.data.add_(xa[p])
            
            p.grad.data.zero_()
            p.grad.data.add_(1, xa[p] - za[p])

            if mom > 0.:
                mux[p].mul_(mom).add_(p.grad.data)
                p.grad.data.add_(mux[p])
            p.data.add_(-lr, p.grad.data)

            xa[p].copy_(p.data)
            za[p].copy_(p.data)

        #lr drop
        if droplr > 1. and ((s['t']//L)/num_batches) in drop_mstones:
            s['lr'] /= droplr
            print("New lr = %s, g=%s, t=%s" % (lr, gamma, s['t']))


        ############
        train_loss += l.item()*len(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy += pred.eq(target.view_as(pred)).sum().item()
        ndata += len(data)

        t.set_postfix(loss=train_loss/ndata, err=100*(1-accuracy/ndata), dist=dist)
        
    return dist, 0, lr, gamma