import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, MNIST, ImageFolder
import os
import torch
import numpy as np
import dataflow as td
from io import BytesIO
from PIL import Image

def get_loader(args, training, augment=False):
    """ function to get data loader specific to different datasets
    """
    if args.dtype == 'cifar10':
        dsets = cifar10_dsets(args, training, augment)
    elif args.dtype == 'cifar100':
        dsets = cifar100_dsets(args, training, augment)
    elif args.dtype == 'imagenet':
        dsets = imagenet_dsets(args, training)
        return dsets
    elif args.dtype == 'tinyimagenet':
        dsets = tinyimagenet(args, training)
    elif args.dtype == 'mnist':
        dsets = mnist_dsets(args, training)
    else:
        print("dtype wrong")
        quit()
    if training is True:
        dset_loaders = {
            'train': DataLoader(dsets['train'], batch_size=args.bs,
                                shuffle=True, pin_memory=True,
                                num_workers=4),
            'val': DataLoader(dsets['val'], batch_size=128,
                              shuffle=False, pin_memory=True,
                              num_workers=4)
            }
    else:
        dset_loaders = {
            'test': DataLoader(dsets['test'], batch_size=128,
                               shuffle=False, pin_memory=True,
                               num_workers=4)
        }
    return dset_loaders


def cifar10_dsets(args, training, augment=False):
    """ Function to load cifar10 data"""
    tf = {}
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    if augment:
        tf['train'] = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        tf['train'] = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    tf['val'] = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    if training is True:
        dsets = {
            'train': CIFAR10(root=args.data_dir, train=True,
                             download=False, transform=tf['train']),
            'val': CIFAR10(root=args.data_dir, train=False,
                           download=False, transform=tf['val'])
            }
    else:
        dsets = {
            'test': CIFAR10(root=args.data_dir, train=False,
                            download=False, transform=tf['val'])
            }
    return dsets


def cifar100_dsets(args, training, augment=False):
    """ Function to load cifar10 data"""
    tf = {}
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    if augment:
        tf['train'] = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        tf['train'] = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    tf['val'] = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    if training is True:
        dsets = {
            'train': CIFAR100(root=args.data_dir, train=True,
                              download=False, transform=tf['train']),
            'val': CIFAR100(root=args.data_dir, train=False,
                            download=False, transform=tf['val'])
            }
    else:
        dsets = {
            'test': CIFAR100(root=args.data_dir, train=False,
                             download=False, transform=tf['val'])
            }
    return dsets


def tinyimagenet(args, training):
    """ Function to load cifar10 data"""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = {
        'train': transforms.Compose([transforms.ToTensor(),
                                     normalize]),
        'val': transforms.Compose([transforms.ToTensor(),
                                   normalize])
        }
    if training is True:
        dsets = {
            'train': ImageFolder(root=args.data_dir + '/train/',
                                 transform=transform['train']),
            'val': ImageFolder(root=args.data_dir + '/val/',
                               transform=transform['val'])
            }
    else:
        dsets = {
            'test': ImageFolder(root=args.data_dir + '/val/',
                                transform=transform['val'])
            }
    return dsets

def imagenet_dsets(args, training):
    """ Function to load imagenet data"""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = {
        'train': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize]),
        'val': transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   normalize])
        }
    if training is True:
        dsets = {
            'train': ImagenetLoader(args.data_dir, 'train', transform['train'],
                        batch_size=args.bs, num_workers=12, shuffle=True),
            'val': ImagenetLoader(args.data_dir, 'val', transform['val'],
                        batch_size=128, num_workers=12, shuffle=False)
        }
    else:
        dsets = {
            'test': ImagenetLoader(args.data_dir, 'val', transform['val'],
                        batch_size=args.bs, num_workers=12, shuffle=False)
            }
    return dsets

# def imagenet_dsets(args, training):
#     """ Function to load imagenet data"""

#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     transform = {
#         'train': transforms.Compose([transforms.Resize(256),
#                                    transforms.CenterCrop(224),
#                                    transforms.ToTensor(),
#                                    normalize]),
#         'val': transforms.Compose([transforms.Resize(256),
#                                    transforms.CenterCrop(224),
#                                    transforms.ToTensor(),
#                                    normalize])
#         }
#     if training is True:
#         dsets = {
#             'train': ImageNet(root=args.data_dir, split='train', transform=transform['train']),
#             'val': ImageNet(root=args.data_dir, split='val', transform=transform['val'])
#             }
#     else:
#         dsets = {
#             'test': ImageNet(root=args.data_dir, split='val',
#                              transform=transform['val'])
#             }
#     return dsets


def mnist_dsets(args, training):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    if training is True:
        dsets = {
            'train': MNIST(args.data_dir, train=True, download=True,
                           transform=transform),
            'val': MNIST(args.data_dir, train=False, download=True,
                         transform=transform)
            }
    else:
        dsets = {
            'test': MNIST(args.data_dir, train=False, download=True,
                          transform=transform)
            }
    return dsets

class ImagenetLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int): how many samples per batch to load
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 4)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

    def __init__(self, imagenet_dir, mode, transform, batch_size, shuffle=False, num_workers=4, cache=50000,
            drop_last=False):
        if drop_last:
            raise NotImplementedError("drop_last not implemented")
        # enumerate standard imagenet augmentors
        assert mode in ['train', 'val'], mode

        # open the lmdb file
        lmdb_loc = os.path.join(imagenet_dir, 'ILSVRC-%s.lmdb'%mode)
        ds = td.LMDBData(lmdb_loc, shuffle=False)
        if shuffle:
            ds = td.LocallyShuffleData(ds, cache)
        def f(x):
            img, label= td.LMDBSerializer._deserialize_lmdb(x)
            # img, label = x
            img = Image.open(BytesIO(img.tobytes())).convert('RGB')
            img = transform(img)
            return img, label
        # ds = td.MultiProcessMapDataZMQ(ds, num_proc=num_workers, map_func=f)
        ds = td.MultiThreadMapData(ds, num_thread=num_workers, map_func=f)
        # ds = td.MapData(ds, f)
        self.ds = td.BatchData(ds, batch_size, use_list=True, remainder=False)
        # self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.ds.reset_state()
        self.ds_iter = iter(self.ds)
        self.N = self.ds.size()
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if (self.i + 1) == self.N:
            raise StopIteration
        x, y = next(self.ds_iter)
        self.i += 1
        x, y = torch.stack(x), torch.tensor(y)
        return x, y

    def __len__(self):
        return self.N

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from args import get_args
    args = get_args(["--exp_num",'0', "--dtype", "mnist"])

    dsets = get_loader(args, True)
