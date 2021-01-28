## Basic script for NN training on standard classification datasets
import os
import torch

import argparse
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import tempfile

from sacred import Experiment
from sacred.commands import print_config
ex = Experiment('DNN') # NOTE: this name should reflect the script name
# from sacred.utils import apply_backspaces_and_linefeeds # for progress bar captured output
# ex.captured_out_filter = apply_backspaces_and_linefeeds # doesn't work: sacred/issues/440
from sacred import SETTINGS 
SETTINGS['CAPTURE_MODE'] = 'no' # don't capture output (avoid progress bar clutter)

# add sacreddnn dir to path
import os, sys
sacreddnn_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, sacreddnn_dir) 

# SacredDNN imports 
from sacreddnn.parse_args import get_dataset, get_loss_function, get_model_type, get_optimizer
from sacreddnn.utils import num_params, l2_norm, run_and_config_to_path,\
                            file_observer_dir, to_gpuid_string
from sacreddnn.activations import replace_relu_
import math

def get_std_cosine_schedule(num_epochs: int, std: float,
                            num_training_obs: int,
                            batch_size: int):
  steps_per_epoch = int(math.floor(num_training_obs / batch_size))
  halfwavelength_steps = num_epochs * steps_per_epoch

  def std_rate_fn(step):
    scale_factor = -np.cos(step * np.pi / halfwavelength_steps) * 0.5 + 0.5
    return std * (scale_factor * 9 + 1)

  return std_rate_fn


def train(loss, model, device, train_loader, optimizer, std_rate_fn, ep, args):
    model.train()
    t = tqdm(train_loader) # progress bar integration
    train_loss, accuracy, ndata = 0, 0, 0    
    for batch_itr, (data, target) in enumerate(t, 1):
        args.std = std_rate_fn(len(train_loader)*(ep-1) + batch_itr)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        itr = 1
        for _ in range(itr):
            # add noise to theta
            with torch.no_grad():
                noise = []
                for mp in model.parameters():
                    temp = torch.empty_like(mp, device=mp.data.device)
                    temp.normal_(0, args.std * mp.view(-1).norm().item() + 1e-16)
                    noise.append(temp)
                    mp.data.add_(noise[-1])

            # single sample convolution approximation
            with torch.set_grad_enabled(True):
                output = model(data)
                l = loss(output, target) * 1/itr
                l.backward()

            # going back to without theta
            with torch.no_grad():
                for mp, n in zip(model.parameters(), noise):
                    mp.data.sub_(n)        
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
            # print(f"SHAPE {data.shape}")
            data, target = data.to(device), target.to(device)
            output = model(data)
            l += loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()
            ndata += len(data)
            
    # print(f"@eval time: {time.time()-t0:.4f}s")
    return l/ndata, (1-accuracy/ndata)*100


def initialization_rescaling(model, gain_factor):
    for i,p in enumerate(model.parameters()):
        p.data *= gain_factor


@ex.config  # Configuration is defined through local variables.
def cfg():
    std = 0.001
    batch_size = 128       # input batch size for training
    epochs = 300           # number of epochs to train
    lr = 0.01               # learning rate
    weight_decay = 1e-4    # weight decay param (=L2 reg. Good value is 5e-4)
    dropout = 0.           # dropout
    no_cuda = False        # disables CUDA training
    nthreads = 4           # number of threads
    save_model = False     # save current model to path
    save_epoch = 100        # save every save_epoch model
    keep_models = False    # keep all saved models
    load_model = ""        # load model from path
    last_epoch = 0         # last_epoch for lr_scheduler
    droplr = 10             # learning rate drop factor (use 0 for no-drop)
    drop_mstones = "drop_150_225" # learning rate milestones (epochs at which applying the drop factor)
    opt = "momentum"       # optimizer type
    loss = "nll"           # classification loss [nll, mse]
    model = "lenet"        # model type  [lenet, densenet, resnet_cifar, efficientnet-b{1-7}(-pretrained)]  
    dataset = "cifar10"    # dataset  [mnist, fashion, cifar10, cifar100]
    datapath = f'data/{dataset}/'   # folder containing the datasets (e.g. mnist will be in "data/MNIST")
    logtime = 2            # report every logtime epochs
    #M = -1                # take only first M training examples 
    #Mtest = -1            # take only first Mtest test examples 
    #pclass = -1           # take only pclass training examples for each class 
    preprocess = 2         # data preprocessing level. preprocess=0 is no preproc. Check preproc_transforms.py
    gpu = 0                # which gpu(s) to use, if 'distribute' all gpus are used
    deterministic = False  # set deterministic mode on gpu
    noise_ampl = 0.        # noise amplitude (for perturbing model initialization)
    init_rescale = 0.     # rescaling of model initialization

    # fake
    use_center=False
    y=1
    
    # non-trivial data augmentations ((Fast)AutoAugment, CutOut)
    augmentation_type = 'autoaug_cifar10'
    cutout = 0

    # Entropy-SGD specific
    L = 1
    sgld_noise = 1e-4
    #sgld_g = 1e-4
    #sgld_grate = 1e-3
    
    ## CHANGE ACTIVATIONS
    activation = "relu"   # Change to e.g. "swish" to replace each relu of the model with a new activation
                        # ["swish", "quadu", "mish", ...]

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
    print(f"{logdir}/{run_and_config_to_path(_run, _config)}")
    if args.save_model: # make temp file. In the end, the model will be stored by the observers.
        save_path = tempfile.mkdtemp() + "/model.pt" 

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda" if use_cuda else "cpu")
    torch.set_num_threads(args.nthreads)
    print(f"USE_CUDA = {use_cuda},  DEVICE_COUNT={torch.cuda.device_count()}, NUM_CPU_THREADS={torch.get_num_threads()}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    ## LOAD DATASET
    loader_args = {'pin_memory': True} if use_cuda else {}
    
    dtrain, dtest = get_dataset(args)
    train_idxs = list(range(len(dtrain)))
    test_idxs = list(range(len(dtest)))
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
    ex.info["num_params"] = num_params(model)
    ex.info["num_learnable"] = num_params(model, learnable=True)
    print(f"MODEL: {ex.info['num_params']} params, {ex.info['num_learnable']} learnable")

    #replace activation function
    replace_relu_(model, args.activation)

    # rescale initialization
    if args.init_rescale > 0.:
        print("Initialization rescaling")
        initialization_rescaling(model, args.init_rescale)

    ## CREATE OPTIMIZER
    optimizer = get_optimizer(args, model)
    
    if args.droplr:
        
        if args.droplr == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0., last_epoch=-1)
            print("Cosine lr schedule")
            
        elif args.droplr < 1. and args.droplr != 0.:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=args.droplr)
        else:
            gamma_sched = 1. / args.droplr if args.droplr > 0 else 1
            if args.model.startswith('resnet'):
                #mstones = [82, 123]
                mstones = [int(h) for h in args.drop_mstones.split('_')[1:]]
                print("MileStones %s" % mstones)
                print("ResNet lr schedule")
            elif args.model.startswith('pyramidnet'):
                #mstones = [150, 225]
                mstones = [int(h) for h in args.drop_mstones.split('_')[1:]]
                print("MileStones %s" % mstones)
                print("PyramidNet lr schedule")
            else:
                mstones = [args.epochs//2, args.epochs*3//4, args.epochs*15//16]
                
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,\
                        milestones=mstones, gamma=gamma_sched)

    # dummy loop to make the schedulers progress
    for i in range(args.last_epoch):
        scheduler.step()

    ## LOSS FUNCTION
    loss = get_loss_function(args)

    std_rate_fn = get_std_cosine_schedule(args.epochs + 1, args.std,
                                          len(train_loader.dataset),
                                          args.batch_size)

    ## REPORT CALLBACK
    def report(epoch):
        model.eval()
 
        o = dict() # store observations
        o["epoch"] = epoch
        o["lr"] = optimizer.param_groups[0]["lr"]
        o["std"] = args.std
        o["train_loss"], o["train_error"] = \
            eval_loss_and_error(loss, model, device, train_loader)
        o["test_loss"], o["test_error"] = \
            eval_loss_and_error(loss, model, device, test_loader)
        o["norms"]  = l2_norm(model)

        print("\n", pd.DataFrame({k:[o[k]] for k in o}), "\n")
        for k in o:
            ex.log_scalar(k, o[k], epoch)
            if logdir:
                writer.add_scalar(k, o[k], epoch)
        
    ## START TRAINING
    report(args.last_epoch)
    for epoch in range(args.last_epoch + 1, args.epochs + 1):
        train(loss, model, device, train_loader, optimizer, std_rate_fn, epoch, args)
        # torch.cuda.empty_cache()
        if epoch % args.logtime == 0:
            report(epoch)
        if epoch % args.save_epoch and args.save_model:
            torch.save(model.state_dict(), save_path)
            if args.keep_models:
                kept_model_path = save_prefix+"_epoch_{}.pt".format(epoch)
                copyfile(model_path, kept_model_path)
                ex.add_artifact(kept_model_path, content_type="application/octet-stream")
            ex.add_artifact(save_path, content_type="application/octet-stream")   
            
        if args.droplr:
            print("lr=%s" % scheduler.get_lr())
            scheduler.step()
            
    # Save model after training
    if args.save_model:
        torch.save(model.state_dict(), save_path)
        ex.add_artifact(save_path, content_type="application/octet-stream")

        
