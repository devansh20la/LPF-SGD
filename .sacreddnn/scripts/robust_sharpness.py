## Compute sharpness
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
from shutil import copyfile
import copy

from sacred import Experiment
from sacred.commands import print_config
ex = Experiment('RobustDNN')
from sacred import SETTINGS 
SETTINGS['CAPTURE_MODE'] = 'no'

# add sacreddnn dir to path
import os, sys
sacreddnn_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, sacreddnn_dir) 

# SacredDNN imports
from sacreddnn.models.robust import RobustNet, RobustDataLoader
from sacreddnn.parse_args import get_dataset, get_loss_function, get_model_type, get_optimizer
from sacreddnn.utils import num_params, l2_norm, run_and_config_to_path,\
                            file_observer_dir, to_gpuid_string, take_n_per_class
from sacreddnn.activations import replace_relu_
        
def eval_loss_and_error(loss, model, loader, noise="add", sigma=0, M2=1, M3=-1):
    #model.eval()
    center_0 = model.get_or_build_center()
    #center_0.eval()

    mc_center_loss, mc_center_accuracy = 0., 0.
    mc_center2_loss, mc_center2_accuracy = 0., 0.

    for mc_step in range(M2):
        
        t0 = time.time()
        model_0 = copy.deepcopy(model)
        center = copy.deepcopy(center_0)
        
        # perturb model with amplitude sigma
        if sigma > 0.:
            with torch.no_grad():
                z = create_perturb(center, noise=noise)
                perturb(center, z, sigma)
                
        #center = model_0.get_or_build_center()
        center.eval()
  
        center_loss, center_accuracy = 0., 0.
        ndata = 0
        
        with torch.no_grad():
            for counter, (data, target) in enumerate(loader.single_loader()):

                # estimate loss only on M3 minibatches
                if counter == M3:
                    break

                target = target.to(model.master_device)
                
                # center
                output = center(data.to(model.master_device))
                run_center_loss = loss(output, target, reduction='sum').item()
                center_loss += run_center_loss
                pred = output.argmax(dim=1, keepdim=True)
                run_center_accuracy = pred.eq(target.view_as(pred)).sum().item()
                center_accuracy += run_center_accuracy
                
                ndata += len(data) 
                
        center_loss /= ndata
        center_accuracy /= ndata
        print(f"mcstep {mc_step+1} @eval time: {time.time()-t0:.2f}s error: {(1-center_accuracy)*100:.2f}")

        mc_center_loss += center_loss
        mc_center_accuracy += center_accuracy
        mc_center2_loss += center_loss**2
        mc_center2_accuracy += center_accuracy**2

        #if sigma > 0.:
        #    with torch.no_grad():
        #        perturb(center, z, -sigma)

    mc_center_loss /= M2
    mc_center_accuracy /= M2
    
    mc_center2_loss /= M2
    mc_center2_accuracy /= M2

    std_mc_center_loss = np.sqrt(mc_center2_loss - mc_center_loss**2)
    std_mc_center_accuracy = np.sqrt(mc_center2_accuracy - mc_center_accuracy**2)

    return mc_center_loss, (1-mc_center_accuracy)*100, std_mc_center_loss, std_mc_center_accuracy*100

def create_perturb(model, noise="add"):
    z = []
    for p in model.parameters():
        r = p.clone().detach().normal_()
        if noise == "add":
            z.append(r)           # additive noise 
        else:
            z.append(p.data * r)  # multiplicative noise
    return z
    
def perturb(model, z, noise_ampl):
    for i,p in enumerate(model.parameters()):
        p.data += noise_ampl * z[i]

def initialization_rescaling(model, gain_factor):
    for i,p in enumerate(model.parameters()):
        p.data *= gain_factor


@ex.config  # Configuration is defined through local variables.
def cfg():
    batch_size = 128      # input batch size for training
    epochs = 100          # number of epochs to train
    #weight_decay = 5e-4   # weight decay param (=L2 reg. Good value is 5e-4)
    dropout = 0.          # dropout
    no_cuda = False       # disables CUDA training
    nthreads = 2          # number of threads
    save_model = False    # save current model to path
    save_epoch = 10       # save every save_epoch model
    load_model = ""       # load model from path
    last_epoch = 0        # last_epoch for lr_scheduler
    opt = "nesterov"      # optimizer
    loss = "nll"          # classification loss [nll, mse]
    model = "lenet"       # model type  [lenet, densenet, resnet_cifar, efficientnet-b{1-7}(-pretrained)]  
    dataset = "fashion"   # dataset  [mnist, fashion, cifar10, cifar100]
    datapath = '~/data/'  # folder containing the datasets (e.g. mnist will be in "data/MNIST")
    logtime = 1           # report every logtime epochs
    preprocess = 0        # data preprocessing level. preprocess=0 is no preproc. Check preproc_transforms.py
    gpu=0                 # which gpu(s) to use, if 'distribute' all gpus are used
    deterministic = False # set deterministic mode on gpu
    noise_type = "add"
    #noise_ampl = 0.       # noise amplitude (for perturbing model initialization)
    #init_rescale = 0.     # rescaling of model initialization

    # non-trivial data augmentations ((Fast)AutoAugment, CutOut)
    augm_type = 'autoaug_cifar10'
    cutout = 0
    
    ## ROBUST ENSEMBLE SPECIFIC
    y = 3                         # number of replicas
    use_center = False            # use a central replica
    #g = 1e-3                      # initial coupling value
    #grate = 1e-1                  # coupling increase rate
    #gmax = float("inf")           # maximum coupling value
    #g_schedule = "exp"
    #g_time = 1
    #rescale_coupling_loss = False # rescale initial coupling loss at 1
    
    # Sharpness-specific
    # args.M1 is replaced by args.epochs (anyway it is for bisection)
    M2 = 100        # MonteCarlo iterations
    M3 = -1         # Number of minibatches for loss estimation
    s_max = 0.1     # max sigmavalue
    #s_min = 0.001   # min sigma value
    
    ## CHANGE ACTIVATIONS
    activation = None   # Change to e.g. "swish" to replace each relu of the model with a new activation
                        # ["swish", "quadu", "mish", ...]


@ex.automain
def main(_run, _config):
    ## SOME BOOKKEEPING
    args = argparse.Namespace(**_config)
    print_config(_run); print()
    logdir = file_observer_dir(_run)
    if not logdir is None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"{logdir}/{run_and_config_to_path(_run, _config)}")

    if args.save_model: # make temp file. In the end, the model will be stored by the observers.
        save_prefix = tempfile.mkdtemp() + "/model"

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        if args.gpu=="distribute":
            ndevices = torch.cuda.device_count()
            devices = [torch.device(y%ndevices) for y in range(args.y+1)]
        elif isinstance(args.gpu, int):
            devices = [torch.device(args.gpu)]*(args.y+1)
        else:
            if not isinstance(args.gpu, list):
                raise ValueError("please provide gpu as list")
            l=len(args.gpu)
            devices = [torch.device(args.gpu[r%l]) for r in range(args.y+1)]
    else:
        devices = [torch.device("cpu")]*(args.y+1)
    for r, device in enumerate(devices):
        print("{} on {}".format("replica {}".format(r) if r<len(devices)-1 else "master", device))


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    torch.set_num_threads(args.nthreads)

    ## LOAD DATASET
    #loader_args = {'pin_memory': True} if use_cuda else {}
    loader_args = {}
    dtrain, dtest = get_dataset(args)
    train_idxs = list(range(len(dtrain)))
    test_idxs = list(range(len(dtest)))

    print(f"DATASET {args.dataset}: {len(train_idxs)} Train and {len(test_idxs)} Test examples")

    train_loader = RobustDataLoader(dtrain,
        y=args.y, concatenate=True, num_workers=0,
        sampler=SubsetRandomSampler(train_idxs),
        batch_size=args.batch_size, **loader_args)
    test_loader = RobustDataLoader(dtest,
        y=args.y, concatenate=True, num_workers=0,
        sampler=SubsetRandomSampler(test_idxs),
        batch_size=args.batch_size, **loader_args)

    ## BUILD MODEL
    Net = get_model_type(args)
    model = RobustNet(Net, y=args.y, g=1e-3, grate=1e-1, devices=devices, gmax=10., use_center=args.use_center, Tmax=args.epochs)
    #replace_relu_(model, args.activation)
    
    # rescale initialization
    #if args.init_rescale > 0.:
    #    initialization_rescaling(model, args.init_rescale)

    # perturb initialization
    #if args.noise_ampl > 0:
    #    z = create_perturb(model)
    #    perturb(model, z, args.noise_ampl)
    #    #ampl = args.noise_ampl * l2_norm(model, mediate=True)
    #    #perturb(model, z, ampl)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model + ".pt"))
    ex.info["num_params"] = num_params(model)
    print(f"MODEL: {ex.info['num_params']} params")
    model.eval()
        
    ## LOSS FUNCTION
    loss = get_loss_function(args)

    ## REPORT CALLBACK
    def report(epoch, sigma, o0, M2=1, M3=-1):
        model.eval()
 
        o = dict() # store scalar observations
        oo = dict() # store array observations
        o["sigma"] = sigma

        if epoch > 0:
            o["train_center_loss"], o["train_center_error"], o["train_center_loss_std"], o["train_center_error_std"] = eval_loss_and_error(loss, model, train_loader, noise=args.noise_type, sigma=sigma, M2=M2, M3=M3)
            o["test_center_loss"], o["test_center_error"], o["test_center_loss_std"], o["test_center_error_std"] = eval_loss_and_error(loss, model, test_loader, noise=args.noise_type, sigma=sigma, M2=M2, M3=M3)
        else:
            o["train_center_loss"], o["train_center_error"] = o0["train_center_loss"], o0["train_center_error"]
            o["test_center_loss"], o["test_center_error"] = o0["test_center_loss"], o0["test_center_error"]
            o["train_center_loss_std"], o["train_center_error_std"] = 0., 0.
            o["test_center_loss_std"], o["test_center_error_std"] = 0., 0.
            
        # compute relative differences        
        o["train_center_loss_rel"] = (o["train_center_loss"]-o0["train_center_loss"])
        o["train_center_error_rel"] = (o["train_center_error"]-o0["train_center_error"])
        o["test_center_loss_rel"] = (o["test_center_loss"]-o0["test_center_loss"])
        o["test_center_error_rel"] = (o["test_center_error"]-o0["test_center_error"])
        
        print("\n", pd.DataFrame({k:[o[k]] for k in o}), "\n")
        for k in o:
            ex.log_scalar(k, o[k], epoch)
            if logdir:
                writer.add_scalar(k, o[k], epoch)
        for k in oo:
            print(f"{k}:\t{oo[k]}")
            ex.log_scalar(k, np.mean(oo[k]), epoch) # Ref. https://github.com/IDSIA/sacred/issues/465
            if logdir:
                writer.add_scalar(k, np.mean(oo[k]), epoch)
        print()

        
    ## START SHARPNESS MEASURE
    
    sigma = 0.
    sigma_step = args.s_max / args.epochs
    
    o0 = dict()
    o0["train_center_loss"], o0["train_center_error"], _, _ = \
                eval_loss_and_error(loss, model, train_loader, noise=args.noise_type, sigma=0, M2=1, M3=-1)
    o0["test_center_loss"], o0["test_center_error"], _, _ = \
                eval_loss_and_error(loss, model, test_loader, noise=args.noise_type, sigma=0, M2=1, M3=-1)


    report(0, 0., o0, M2=1, M3=-1)
    for epoch in range(1, args.epochs + 1):
        
        sigma += sigma_step
        
        if epoch % args.logtime == 0:
            report(epoch, sigma, o0, M2=args.M2, M3=args.M3)
            
        #save models
        if epoch % args.save_epoch == 0 and args.save_model:
            model_path = save_prefix+".pt"
            torch.save(model.state_dict(), model_path)
            if args.save_epoch > 0:
                kept_model_path = save_prefix+"_epoch_{}.pt".format(epoch)
                copyfile(model_path, kept_model_path)
                ex.add_artifact(kept_model_path, content_type="application/octet-stream")
            ex.add_artifact(model_path, content_type="application/octet-stream")
        
    # Save model after training
    if args.save_model:
        model_path = save_prefix+"_final.pt"
        torch.save(model.state_dict(), model_path)
        ex.add_artifact(model_path, content_type="application/octet-stream")
