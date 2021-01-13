from __future__ import print_function
import argparse, math, random
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import torch.backends.cudnn as cudnn

from models import *
import loader, optim
import numpy as np
from utils import *
import os
import shutil 
import pickle


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        shutil.rmtree(path)


parser = argparse.ArgumentParser(description='PyTorch Entropy-SGD')
ap = parser.add_argument
ap('-m', help='mnistfc | mnistconv | allcnn', type=str, default='mnistconv')
ap('-b',help='Batch size', type=int, default=128)
ap('-B', help='Max epochs', type=int, default=150)
ap('--lr', help='Learning rate', type=float, default=0.1)
ap('--l2', help='L2', type=float, default=0.0)
ap('-L', help='Langevin iterations', type=int, default=0)
ap('--gamma', help='gamma', type=float, default=1e-4)
ap('--scoping', help='scoping', type=float, default=1e-3)
ap('--noise', help='SGLD noise', type=float, default=1e-4)
ap('-g', help='GPU idx.', type=int, default=0)
ap('-s', help='seed', type=int, default=42)
opt = vars(parser.parse_args())

th.set_num_threads(2)
opt['cuda'] = th.cuda.is_available()
if opt['cuda']:
    opt['g'] = -1
    th.cuda.set_device(opt['g'])
    th.cuda.manual_seed(opt['s'])
    cudnn.benchmark = True
random.seed(opt['s'])
np.random.seed(opt['s'])
th.manual_seed(opt['s'])

opt['dataset'] = 'cifar10'

train_loader, val_loader, test_loader = getattr(loader, opt['dataset'])(opt)
model = ResNet18(10)

criterion = nn.CrossEntropyLoss()
if opt['cuda']:
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.EntropySGD(model.parameters(),
                             config = dict(lr=opt['lr'], momentum=0.9, nesterov=True, weight_decay=opt['l2'],
                             L=opt['L'], eps=opt['noise'], g0=opt['gamma'], g1=opt['scoping']))
scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
print(opt)


def set_dropout(cache=None, p=0):
    if cache is None:
        cache = []
        for l in model.modules():
            if 'Dropout' in str(type(l)):
                cache.append(l.p)
                l.p = p
        return cache
    else:
        for l in model.modules():
            if 'Dropout' in str(type(l)):
                assert len(cache) > 0, 'cache is empty'
                l.p = cache.pop(0)


def dry_feed():
    cache = set_dropout()
    maxb = int(math.ceil(train_loader.n/opt['b']))
    for bi in range(maxb):
        x, y = next(train_loader)
        if opt['cuda']:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y.squeeze())
        yh = model(x)
    set_dropout(cache)


@th.no_grad()
def val(e, data_loader):
    dry_feed()
    model.eval()

    maxb = int(math.ceil(data_loader.n/opt['b']))

    fs, top1 = AverageMeter(), AverageMeter()
    for bi in range(maxb):
        x, y = next(data_loader)
        bsz = x.size(0)

        if opt['cuda']:
            x, y = x.cuda(), y.cuda()

        x, y = Variable(x), Variable(y.squeeze())
        yh = model(x)

        f = criterion.forward(yh, y).data.item()
        prec1, = accuracy(yh.data, y.data, topk=(1,))
        err = 100 - prec1.item()

        fs.update(f, bsz)
        top1.update(err, bsz)

    print('Test: [%2d] %2.4f %2.4f%%\n'%(e, fs.avg, top1.avg))
    print()
    return fs.avg, top1.avg


cp_dir = "checkpoints/cifar10/resnet18/esgd/run_ms_" + str(opt['s']) + '/run0/'
model.load_state_dict(th.load(cp_dir + 'best_model.pth.tar')['model'])

loss, valerr1 = val(0, val_loader)
mtr = {'err': valerr1}
print(mtr)
with open(cp_dir + 'measure.pkl', 'wb') as f:
    pickle.dump(mtr, f)
