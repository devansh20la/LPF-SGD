import torch as th
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
import os, sys, pdb


class sampler_t:
    def __init__(self, batch_size, x,y, train=True):
        self.n = x.size(0)
        self.x, self.y = x,y
        self.b = batch_size
        self.idx = th.arange(0, self.b).long()
        self.train = train
        self.sidx = 0

    def __next__(self):
        if self.train:
            self.idx.random_(0,self.n)
        else:
            s = self.sidx
            e = min(s+self.b-1, self.n)

            self.idx = th.arange(s, e).long()
            self.sidx += self.b
            if self.sidx >= self.n:
                self.sidx = 0

        x, y = th.index_select(self.x, 0, self.idx), \
            th.index_select(self.y, 0, self.idx)
        return x, y

    next = __next__

    def __iter__(self):
        return self


def mnist(opt):
    d1, d2 = datasets.MNIST('data', download=True, train=True), \
        datasets.MNIST('data', train=False)

    train = sampler_t(opt['b'], d1.train_data.view(-1, 1, 28, 28).float(),
                      d1.train_labels)
    val = sampler_t(opt['b'], d2.test_data.view(-1, 1, 28, 28).float(),
                    d2.test_labels, train=False)
    return train, val, val


def cifar10(opt):
    """ Function to load cifar10 data"""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    transform =  transforms.Compose([transforms.ToTensor(),
                                     normalize])
    dsets = {
        'train': datasets.CIFAR10(root='data/cifar10/', train=True,
                                  download=True, transform=transform),
        'val': datasets.CIFAR10(root='data/cifar10/', train=False,
                                download=True, transform=transform)
        }
    dset_loaders = {
        'train': torch.utils.data.DataLoader(dsets['train'], batch_size=128,
                                             shuffle=True, pin_memory=True,
                                             num_workers=4),
        'val': torch.utils.data.DataLoader(dsets['val'], batch_size=128,
                                           shuffle=False, pin_memory=True,
                                           num_workers=4)
        }
    x = []
    y = []
    for inputs, labels in dset_loaders['train']:
        x.append(inputs)
        y.append(labels)

    train = sampler_t(opt['b'], th.cat(x, dim=0), th.cat(y, dim=0))
    x = []
    y = []
    for inputs, labels in dset_loaders['val']:
        x.append(inputs)
        y.append(labels)
    val = sampler_t(opt['b'], th.cat(x, dim=0), th.cat(y, dim=0), train=False)

    return train, val, val
