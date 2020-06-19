import torch
from torchvision import datasets, transforms
from utils.data_loader_mnist import MNIST
from sklearn.metrics import average_precision_score


def get_loader(args, training, lp=1.0):
    """ function to get data loader specific to different datasets
    """
    if args.dtype == 'cifar10':
        dsets = cifar10_dsets(args, training)

    elif args.dtype == 'cifar100':
        dsets = cifar100_dsets(args, training)

    elif args.dtype == 'imagenet':
        dsets = imagenet_dsets(args, training)

    elif args.dtype == 'mnist':
        dsets = mnist_dsets(args, training, lp=lp)

    if training is True:
        dset_loaders = {
            'train': torch.utils.data.DataLoader(dsets['train'], batch_size=args.bs,
                                                 shuffle=True, pin_memory=True,
                                                 num_workers=8),
            'val': torch.utils.data.DataLoader(dsets['val'], batch_size=128,
                                               shuffle=False, pin_memory=True,
                                               num_workers=8)
        }

    else:
        dset_loaders = {
            'test': torch.utils.data.DataLoader(dsets['test'], batch_size=128,
                                                shuffle=False, pin_memory=True,
                                                num_workers=8)
        }

    return dset_loaders


def mnist_dsets(args, training, lp):
    """ Function to load mnist data
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    if training is True:
        dsets = {
            'train': MNIST(args.data_dir, train=True, download=False,
                           transform=transform, lp=lp),
            'val': MNIST(args.data_dir, train=False, download=False,
                         transform=transform, lp=lp)
        }
    else:
        dsets = {
            'test': MNIST(args.data_dir, train=False, download=False,
                          transform=transform, lp=lp)
        }

    return dsets


def cifar10_dsets(args, training):
    """ Function to load cifar10 data
    """
    transform = {
        'train': transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))]),
        'val': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))])
    }
    if training is True:
        dsets = {
            'train': datasets.CIFAR10(root=args.data_dir, train=True,
                                      download=False, transform=transform['train']),
            'val': datasets.CIFAR10(root=args.data_dir, train=False,
                                    download=False, transform=transform['val'])
        }
    else:
        dsets = {
            'test': datasets.CIFAR10(root=args.data_dir, train=False,
                                     download=False, transform=transform['val'])
        }

    return dsets


def cifar100_dsets(args, training):
    """ Function to load cifar100 data
    """
    transform = {
        'train': transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))]),
        'val': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))])
    }
    if training is True:
        dsets = {
            'train': datasets.CIFAR100(root=args.data_dir, train=True,
                                       download=True, transform=transform['train']),
            'val': datasets.CIFAR100(root=args.data_dir, train=False,
                                     download=True, transform=transform['val'])
        }
    else:
        dsets = {
            'test': datasets.CIFAR100(root=args.data_dir, train=False,
                                      download=True, transform=transform['val'])
        }

    return dsets


def imagenet_dsets(args, training):
    """ Function to load imagenet data
    """
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

    if training is True:
        dsets = {
            'train': datasets.ImageNet(root=args.data_dir, split='train',
                                       images_per_class=args.img_per_class,
                                       download=False,
                                       transform=transform),
            'val': datasets.ImageNet(root=args.data_dir, split='val',
                                     download=False, transform=transform)
        }

    else:
        dsets = {
            'test': datasets.ImageNet(root=args.data_dir, split='val',
                                      download=False, transform=transform)
        }

    return dsets


def accuracy(output, target, topk=(1,)):
    if output.shape[1] > 1:
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        return res
    else:
        batch_size = target.size(0)
        pred = output
        pred[pred >= 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct.view(-1).float().sum(0, keepdim=True)
        return (correct.mul_(100.0 / batch_size),)


def precision(output, target, topk=(1,)):
    out = output.clone()
    tar = target.clone()

    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        out = out.cpu().detach().numpy()
        tar = tar.cpu().detach().numpy()
        average_precision = average_precision_score(tar, out,average='micro')
        return torch.Tensor([average_precision])


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
    sys.path.append("..")
    from args import get_args
    args = get_args(["--dtype", "mnist"])

    dsets = mnist_dsets(args, True)

