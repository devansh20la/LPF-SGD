import torch
import argparse
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader
import time
import os
from tqdm import tqdm
# from collections import OrderedDict
# import json
import pandas as pd
from tabulate import tabulate

from sacred import Experiment
from sacred.commands import print_config

ex = Experiment('BinaryNet')

from models.binarynet.binary_lenet import BinaryLeNet
    
# from sacred.utils import apply_backspaces_and_linefeeds # for progress bar captured output
# ex.captured_out_filter = apply_backspaces_and_linefeeds # doesn't work: sacred/issues/440
from sacred import SETTINGS 
SETTINGS['CAPTURE_MODE'] = 'no' # don't capture output (avoid progress bar clutter)

from parse_args import get_dataset, get_loss_function, get_model_type, get_optimizer

def train(loss, model, device, train_loader, optimizer):
    model.train()
    t = tqdm(train_loader) # progress bar integration
    train_loss, accuracy, ndata = 0, 0, 0    
    for data, target in t:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        l = loss(output, target)
        l.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                # print("here")
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1, 1))


        train_loss += l.item()*len(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy += pred.eq(target.view_as(pred)).sum().item()
        ndata += len(data)
        
        t.set_postfix(loss=train_loss/ndata, err=100*(1-accuracy/ndata))

def eval_loss_and_error(loss, model, device, loader):
    # t0 = time.time()
    model.eval()
    l, accuracy, ndata = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            l += loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()
            ndata += len(data)
            
    # print(f"@eval time: {time.time()-t0:.4f}s")
    return l/ndata, (1-accuracy/ndata)*100

def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def l2_norm(model):
    return torch.sqrt(sum(p.norm()**2 for p in model.parameters())).item()
        

@ex.config  # Configuration is defined through local variables.
def cfg():
    batch_size = 128    # input batch size for training
    epochs = 100        # number of epochs to train
    lr = 0.1            # learning rate
    weight_decay = 5e-4    # weight decay param (=L2 reg. Good value is 5e-4)
    no_cuda = False     # disables CUDA training
    nthreads = 2        # number of threads
    save_model = ""     # save model to path
    load_model = ""     # load model from path
    droplr = 5         # learning rate drop factor (use 0 for no-drop)
    opt = "nesterov"         # optimizer type
    loss = "nll"        # classification loss [nll, mse]
    model = "lenet"     # model type  [lenet, densenet,...]  
    dataset = "fashion" # dataset  [mnist, fashion, cifar10, cifar100]
    logtime = 2         # report every logtime epochs
    exclude_class = -1  # exclude a class from the training
    preprocess = False  # data normalization
    gpu = 0             # gpu_id to use

@ex.automain
def main(_run, _config):
    args = argparse.Namespace(**_config)
    print_config(_run); print()

    if args.gpu >= 0:                                       # Avoid creating cuda context on all gpus. 
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # To be done before any call to torch.cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("USING CUDA =", use_cuda)
    device = torch.device(f"cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)
    # print(torch.get_)
    os.environ["OMP_NUM_THREADS"] = str(args.nthreads)
    # torch.set_num_threads(args.nthreads) # doesn't work!!!

    ## LOAD DATASET
    loader_args = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    DATAPATH = '~/data/'
    dtrain, dtest = get_dataset(args, DATAPATH)
    if args.exclude_class >= 0:
        train_idxs = [i for i in range(len(dtrain)) if dtrain.targets[i] != args.exclude_class]
        test_idxs = [i for i in range(len(dtest)) if dtest.targets[i] != args.exclude_class]
    else:
        train_idxs = list(range(len(dtrain)))
        test_idxs = list(range(len(dtest)))
    print(f"DATASET: {len(train_idxs)} Train and {len(test_idxs)} Test examples")

    train_loader = DataLoader(dtrain,
        sampler=SubsetRandomSampler(train_idxs),
        batch_size=args.batch_size, **loader_args)
    test_loader = DataLoader(dtest,
        sampler=SubsetRandomSampler(test_idxs),
        batch_size=args.batch_size, **loader_args)
    
    ## BUILD MODEL
    # Net = get_model_type(args)
    model = BinaryLeNet().to(device)
    if use_cuda and args.gpu < 0: 
        model = torch.nn.DataParallel(model)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model + ".pt"))
    ex.info["num_params"] = num_params(model)
    print(f"MODEL: {ex.info['num_params']} params")

    ## CREATE OPTIMIZER
    optimizer = get_optimizer(args, model)
    if args.droplr:
        gamma = 1/args.droplr if isinstance(args.droplr, int) else 0.1
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,\
                milestones=[args.epochs//2, args.epochs*3//4, args.epochs*15//16], gamma=gamma) 

    ## LOSS FUNCTION
    loss = get_loss_function(args)

    ## REPORT CALLBACK
    def report(epoch):
        model.eval()
 
        o = dict() # store observations
        o["epoch"] = epoch
        o["lr"] = optimizer.param_groups[0]["lr"]
        o["train_loss"], o["train_error"] = \
            eval_loss_and_error(loss, model, device, train_loader)
        o["test_loss"], o["test_error"] = \
            eval_loss_and_error(loss, model, device, test_loader)
        o["norm"]  = l2_norm(model)

        print("\n", pd.DataFrame({k:[o[k]] for k in o}), "\n")
        for k in o:
            ex.log_scalar(k, o[k], epoch)
        
    ## START TRAINING
    report(0)
    for epoch in range(1, args.epochs + 1):
        if args.droplr:
            scheduler.step()
        train(loss, model, device, train_loader, optimizer)
        # torch.cuda.empty_cache()
        if epoch % args.logtime == 0:
            report(epoch)
        if epoch % 10 and args.save_model:
            torch.save(model.state_dict(), args.save_model + ".pt")
            
    # Save model after training
    if args.save_model:
        ex.add_artifact(args.save_model + ".pt")
