import torch
import argparse
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader
import time
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import tempfile

from sacred import Experiment
from sacred.commands import print_config
ex = Experiment('NoisyDNN')
# from sacred.utils import apply_backspaces_and_linefeeds # for progress bar captured output
# ex.captured_out_filter = apply_backspaces_and_linefeeds # doesn't work: sacred/issues/440
from sacred import SETTINGS 
SETTINGS['CAPTURE_MODE'] = 'no' # don't capture output (avoid progress bar clutter)

# add sacreddnn dir to path
import os, sys
sacreddnn_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, sacreddnn_dir) 

from sacreddnn.hessian.hvp_operator import compute_hessian_eigenthings
from sacreddnn.parse_args import get_dataset, get_loss_function, get_model_type, get_optimizer
from sacreddnn.utils import num_params, l2_norm, run_and_config_to_path,\
                        file_observer_dir, to_gpuid_string


def train(loss, model, device, train_loader, optimizer):
    model.train()
    t = tqdm(train_loader) # progress bar integration
    train_loss, accuracy, ndata = 0, 0, 0
    for data, target in t:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        z = create_perturb(model)
        perturb(model, z, model.g)
        
        output = model(data)
        l = loss(output, target)
        l.backward()
        
        perturb(model, z, -model.g)
        optimizer.step()

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
        
def create_perturb(model):
    z = []
    for p in model.parameters():
        r = p.clone().detach().normal_()
        z.append(p.data * r)
    return z
    
def perturb(model, z, g):
    for i,p in enumerate(model.parameters()):
        p.data += g * z[i]

import dnn
@ex.config  # Configuration is defined through local variables.
def cfg():
    batch_size = 128    # input batch size for training
    epochs = 100        # number of epochs to train
    lr = 0.1            # learning rate
    weight_decay = 5e-4 # weight decay param (=L2 reg. Good value is 5e-4)
    no_cuda = False     # disables CUDA training
    nthreads = 2        # number of threads
    save_model = False  # save model at checkpoints
    load_model = ""     # load model from path
    droplr = 5          # learning rate drop factor (use 0 for no-drop)
    opt = "nesterov"    # optimizer type
    loss = "nll"        # classification loss [nll, mse]
    model = "lenet"     # model type  [lenet, densenet121, resnet18, ...]  
    dataset = "fashion" # dataset  [mnist, fashion, cifar10, cifar100]
    datapath = '~/data/'# folder containing the datasets (e.g. mnist will be in "data/MNIST")
    logtime = 2         # report every logtime epochs
    M = -1              # take only first M training and test examples 
    preprocess = False  # data normalization
    gpu = 0             # gpu_id(s) to use 
    # NOISY DNN SPECIFIC
    g = 0.01            # weight noise intensity
    dropg = 0           # g drop factor (use 0 or 1 for no-drop)

    # To be done before any call to torch.cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = to_gpuid_string(gpu) 

@ex.automain
def main(_run, _config):
    args = argparse.Namespace(**_config)
    print_config(_run); print()
    logdir = file_observer_dir(_run)
    if not logdir is None:
        from torch.utils.tensorboard import SummaryWriter 
        writer = SummaryWriter(log_dir=f"{logdir}/{run_and_config_to_path(_run, _config)}")

    if args.save_model: # make temp file. In the end, the model will be stored by the observers.
        save_path = tempfile.mkdtemp() + "/model.pt" 

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("USING CUDA =", use_cuda)
    device = torch.device(f"cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.nthreads)

    ## LOAD DATASET
    loader_args = {'pin_memory': True} if use_cuda else {}
    dtrain, dtest = get_dataset(args)
    train_idxs = list(range(len(dtrain) if args.M <= 0 else args.M))
    test_idxs = list(range(len(dtest) if args.M <= 0 else min(args.M,len(dtest))))
    print(f"DATASET {args.dataset}: {len(train_idxs)} Train and {len(test_idxs)} Test examples")

    train_loader = DataLoader(dtrain,
        sampler=SubsetRandomSampler(train_idxs),
        batch_size=args.batch_size, **loader_args)
    test_loader = DataLoader(dtest,
        sampler=SubsetRandomSampler(test_idxs),
        batch_size=args.batch_size, **loader_args)

    ## BUILD MODEL
    Net = get_model_type(args)
    model = Net().to(device)
    if use_cuda and torch.cuda.device_count() > 1: 
        model = torch.nn.DataParallel(model)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
    model.g = args.g
    ex.info["num_params"] = num_params(model)
    print(f"MODEL: {ex.info['num_params']} params")


    ## CREATE OPTIMIZER
    optimizer = get_optimizer(args, model)
    milestones = [args.epochs//2, args.epochs*3//4, args.epochs*15//16]
    if args.droplr:
        gamma_sched = 1/args.droplr if args.droplr > 0 else 1
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
            milestones, gamma=gamma_sched) 
    if args.dropg:
        g_sched = 1/args.dropg if args.dropg > 0 else 1
        
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
        o["gamma"]  = model.g

        print("\n", pd.DataFrame({k:[o[k]] for k in o}), "\n")
        for k in o:
            ex.log_scalar(k, o[k], epoch)
            if logdir:
                writer.add_scalar(k, o[k], epoch)

        # hessian_loader = DataLoader(dtrain, sampler=SubsetRandomSampler(train_idxs),
        #                         batch_size=200, **loader_args)
        # eigvals, eigvecs = compute_hessian_eigenthings(
        #     loss, model, device, hessian_loader,
        #     num_eigenthings=40, full_dataset=False, mode='lanczos', max_samples=10,
        #     # power_iter_err_threshold=1e-5, momentum=0, # power iter
        #     which='BE', max_steps=100, tol=1e-5, num_lanczos_vectors=None) # lanczos
                
        # print(eigvals)
        
    ## START TRAINING
    report(0)
    for epoch in range(1, args.epochs + 1):
        train(loss, model, device, train_loader, optimizer)
        # torch.cuda.empty_cache()
        if args.dropg and epoch in milestones:
            model.g *= g_sched
        if epoch % args.logtime == 0:
            report(epoch)
        if epoch % 10 and args.save_model:
            torch.save(model.state_dict(), save_path)
            ex.add_artifact(save_path, content_type="application/octet-stream")    
        if args.droplr:
            scheduler.step()
            
    # Save model after training
    if args.save_model:
        ex.add_artifact(save_path, content_type="application/octet-stream")
