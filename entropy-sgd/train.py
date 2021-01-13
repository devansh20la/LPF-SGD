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


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    else:
        print("deleting")
        shutil.rmtree(path)


parser = argparse.ArgumentParser(description='PyTorch Entropy-SGD')
ap = parser.add_argument
ap('-b',help='Batch size', type=int, default=128)
ap('-B', help='Max epochs', type=int, default=200)
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

scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1, verbose=True)
print(opt)


def train(e):
    model.train()

    fs, top1 = AverageMeter(), AverageMeter()
    ts = timer()

    bsz = opt['b']
    maxb = int(math.ceil(train_loader.n/bsz))

    for bi in range(maxb):
        def helper():
            def feval():
                x, y = next(train_loader)
                if opt['cuda']:
                    x, y = x.cuda(), y.cuda()

                x, y = Variable(x), Variable(y.squeeze())
                bsz = x.size(0)

                optimizer.zero_grad()
                yh = model(x)
                f = criterion.forward(yh, y)
                f.backward()

                prec1, = accuracy(yh.data, y.data, topk=(1,))
                err = 100.-prec1.item()
                return (f.data.item(), err)
            return feval

        f, err = optimizer.step(helper(), model, criterion)

        fs.update(f, bsz)
        top1.update(err, bsz)

        if bi % 100 == 0 and bi != 0:
            print('[%2d][%4d/%4d] %2.4f %2.2f%%'%(e,bi,maxb, fs.avg, top1.avg))

    print('Train: [%2d] %2.4f %2.2f%% [%.2fs]'% (e, fs.avg, top1.avg, timer()-ts))
    print()
    return fs.avg, top1.avg


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


data = {'trainloss': [], 'trainerr': [], 'valloss': [], 'valerr': []}
best_err = float('inf')
cp_dir = "checkpoints/cifar10/resnet18/esgd/run_ms_" + str(opt['s']) + '/run0/'
create_path(cp_dir)

for e in range(opt['B']):
    loss, err = train(e)
    data['trainloss'].append(loss)
    data['trainerr'].append(err)

    loss, valerr1 = val(e, val_loader)
    data['valloss'].append(loss)
    data['valerr'].append(valerr1)

    scheduler.step()

    th.save(data, cp_dir + 'all_data.pth.tar')
    if valerr1 < best_err:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': e,
            'best_err': best_err
        }
        th.save(state, cp_dir + 'best_model.pth.tar')
        best_err = valerr1
