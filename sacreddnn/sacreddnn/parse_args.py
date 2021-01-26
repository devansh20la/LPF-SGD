import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from .models import lenet, mlp, preactresnet, wideresnet, efficientnet
from .models import resnet_cifar, densenet_old
from .models import torchvision_models
from .utils import to_onehot
from .preproc_transforms import preproc_transforms
from .activations import ReQU, QuadU, Swish, Mish, QuadLin, QuadBald, SlowQuad

###### FAST-AA IMPORTS
from .fast_autoaugment import resnet_fastaa, pyramidnet, wideresnet_aa
from .entropy_sgd import EntropySGD
from .rmspropTF import RMSpropTF
######

def get_xsize(args):
    xsize = [28,28]  if args.dataset in ["mnist","fashion"] else\
            [32,32]  if args.dataset in ["cifar10","cifar100","svhn"] else\
            [64,64] if args.dataset=='tiny-imagenet' else\
            [224,224] if args.dataset=='imagenet' else args.xsize
    return xsize

def get_loss_function(args):
    if args.loss == 'nll':
        def loss(output, target, **kwargs):
            output = F.log_softmax(output, dim=1)
            return F.nll_loss(output, target, **kwargs)
    elif args.loss == 'mse':
        def loss(output, target, **kwargs):
            target = to_onehot(target, output.shape[1])
            return F.mse_loss(output, target, **kwargs)
    # elif losstype == 'hinge': #TODO
    return loss

def get_model_type(args):
    dset = args.dataset
    nclasses =  10  if args.dataset in ["mnist","fashion","cifar10", "svhn"] else\
                100 if args.dataset in ["cifar100"] else\
                200 if args.dataset=='tiny-imagenet' else\
                1000 if args.dataset=='imagenet' else args.nclasses

    nchannels = 1 if args.dataset in ["mnist", "fashion"] else\
                3 if args.dataset in ["cifar10", "cifar100", "svhn", "tiny-imagenet", "imagenet"] else 3

    xsize = get_xsize(args)
    
    new_act =   nn.ReLU() if args.activation == 'relu' else\
                nn.Tanh() if args.activation == 'tanh' else\
                ReQU() if args.activation == 'requ' else\
                QuadU() if args.activation == 'quadu' else\
                Swish() if args.activation == 'swish' else\
                QuadLin() if args.activation == 'quadlin' else\
                QuadBald() if args.activation == 'quadbald' else\
                SlowQuad() if args.activation == 'slowquad' else\
                Mish() if args.activation == 'mish' else None

    if args.model == 'lenet' and dset in ["mnist", "fashion", "cifar10", "cifar100"]:
        if xsize[0] != xsize[1]:
            raise NotImplementedError("input needs to be square for lenet")
        Net = lambda: lenet.LeNet(dim=xsize[0], in_channels=nchannels, nclasses=nclasses, activation=new_act, dropout=args.dropout)
    elif args.model == 'lenet_bn' and dset in ["mnist", "fashion", "cifar10", "cifar100"]:
        if xsize[0] != xsize[1]:
            raise NotImplementedError("input needs to be square for lenet")
        Net = lambda: lenet.LeNet_bn(dim=xsize[0], in_channels=nchannels, nclasses=nclasses, activation=new_act, dropout=args.dropout)
    elif args.model == 'lenet5' and dset in ["mnist", "fashion", "cifar10", "cifar100"]:
        if xsize[0] != xsize[1]:
            raise NotImplementedError("input needs to be square for lenet")
        Net = lambda: lenet.LeNet5(dim=xsize[0], in_channels=nchannels, nclasses=nclasses, activation=new_act, dropout=args.dropout)
    elif args.model == 'lenet5_bn' and dset in ["mnist", "fashion", "cifar10", "cifar100"]:
        if xsize[0] != xsize[1]:
            raise NotImplementedError("input needs to be square for lenet")
        Net = lambda: lenet.LeNet5_bn(dim=xsize[0], in_channels=nchannels, nclasses=nclasses, activation=new_act, dropout=args.dropout)
    #elif args.model == 'densenet': #deprecated -> use densenet121
    #    Net = lambda: densenet_old.densenet_cifar(nclasses)
    #elif args.model.startswith('wideresnet'): 
    #    depth = int(args.model.split('_')[1])
    #    widen_factor = int(args.model.split('_')[2])
    #    Net = lambda: wideresnet.WideResNet(depth, nclasses, widen_factor)
    elif args.model == 'preactresnet18':
        Net = lambda: preactresnet.PreActResNet18(nclasses)
    elif args.model == 'resnet110':
        #Net = lambda: resnet_cifar.ResNet110(num_classes=nclasses)
        Net = lambda: resnet_cifar.ResNet110(num_classes=nclasses, activation=new_act)

    # FAST-AA MODELS
    elif args.model == 'resnet110AA':
        Net = lambda: resnet_fastaa.ResNetAA(args.dataset, 110, nclasses, bottleneck=False)
    elif args.model == 'pyramidnetAA':
        Net = lambda: pyramidnet.PyramidNetAA(args.dataset, 272, 200, nclasses, bottleneck=True)
    elif args.model == 'wresnet28_10':
        Net = lambda: wideresnet_aa.WideResNet(28, 10, dropout_rate=0.0, num_classes=nclasses)
    ################
        
    elif args.model.startswith('efficientnet'):
        if args.model.endswith('pretrained'):
            Net = lambda: efficientnet.EfficientNet.from_pretrained(args.model.replace('-pretrained', ''), num_classes=nclasses)
        else:
            Net = lambda: efficientnet.EfficientNet.from_name(args.model, override_params={"num_classes": nclasses})
    elif args.model.startswith('mlp'):
        nin = np.prod(xsize)*nchannels
        nhs = [int(h) for h in args.model.split('_')[1:]]
        Net = lambda: mlp.MLP(nin, nhs, nclasses)
        
    elif args.model == 'pratiklenet':
        # pratik LeNet
        class View(nn.Module):
            def __init__(self,o):
                super(View, self).__init__()
                self.o = o
            def forward(self,x):
                return x.view(-1, self.o)
            
        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.BatchNorm2d(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.Dropout(p))
        
        Net = lambda : nn.Sequential(
            convbn(1,20,5,3,0.25),
            convbn(20,50,5,2,0.25),
            View(50*2*2),
            nn.Linear(50*2*2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(500,10),
            )

    else: # try if corresponds to one of the models import from torchvision 
        # see the file models/torchvision_models.py
        #print("CIAOOO")
        Net = lambda: eval(f"torchvision_models.{args.model}")(num_classes=nclasses)

    return Net

def get_dataset(args):
    if args.dataset == 'mnist':
        DSet = datasets.MNIST
    elif args.dataset == 'fashion':
        DSet = datasets.FashionMNIST
    elif args.dataset == 'cifar10':
        DSet = datasets.CIFAR10
    elif args.dataset == 'cifar100':
        DSet = datasets.CIFAR100
    elif args.dataset == 'svhn':
        DSet = datasets.SVHN
    elif args.dataset == 'tiny-imagenet':
        #train_dir = '~/data/tiny-imagenet-200/train'
        #val_dir = '~/data/tiny-imagenet-200/val'
        train_dir = '/mnt/disk1/Datasets/tiny-imagenet-200/train'
        val_dir = '/mnt/disk1/Datasets/tiny-imagenet-200/val'
        DSet = datasets.ImageFolder
    elif args.dataset == 'imagenet':
        train_dir = '~/data/imagenet/ILSVRC/Data/CLS-LOC/train'
        val_dir = '~/data/imagenet/ILSVRC/Data/CLS-LOC/val'
        DSet = datasets.ImageFolder

    if args.preprocess:
        transform_train, transform_test = preproc_transforms(args)
    else:
        transform_train = transforms.ToTensor()
        transform_test = transforms.ToTensor()
               
    if args.dataset not in ['tiny-imagenet', 'imagenet', "svhn"]:
        dtrain = DSet(args.datapath, train=True, download=True, transform=transform_train)
        dtest = DSet(args.datapath, train=False, download=True, transform=transform_test)  
    elif args.dataset == "svhn": 
        #trainset = DSet(args.datapath, split='train', download=True, transform=transform_train)
        #extraset = DSet(args.datapath, split='extra', download=True, transform=transform_train)
        #dtrain = ConcatDataset([trainset, extraset])
        dtrain = DSet(args.datapath, split='train', download=True, transform=transform_train)
        dtest = DSet(args.datapath, split='test', download=True, transform=transform_test)  
    else:
        dtrain = DSet(train_dir, transform=transform_train)
        dtest = DSet(val_dir, transform=transform_test)   
    
    return dtrain, dtest

def get_optimizer(args, model):
    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "nesterov":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                    momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    elif args.opt == "momentum":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                    momentum=0.9, nesterov=False, weight_decay=args.weight_decay)
    elif args.opt == "rmsprop": # optimizer TensorFlow style
        optimizer = RMSpropTF(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, alpha=0.9, momentum=0.9, eps=0.001)
    elif args.opt.startswith('entropy-sgd'):
        if args.dataset.startswith('cifar'):
            num_batches = int(5e4/args.batch_size)+1
        elif args.dataset in ["mnist", "fashion"]:
            num_batches = int(6e4/args.batch_size)+1
        elif args.dataset in ["svhn"]:
            num_batches = int(73257/args.batch_size)+1
        elif args.dataset in ["tiny-imagenet"]:
            num_batches = int(1e5/args.batch_size)+1
        optimizer = EntropySGD(model.parameters(),
                    config = dict(lr=args.lr, num_batches=num_batches, gtime=args.gtime, momentum=args.mom, momentum_sgld=args.mom_sgld, nesterov=True, weight_decay=args.weight_decay, L=args.L, eps=args.sgld_noise, g0=args.g, g1=args.grate, gmax=args.gmax, epochs=args.epochs, sgld_lr=args.sgld_lr, alpha_arg=args.alpha, gscale=args.gscale))
    return optimizer
