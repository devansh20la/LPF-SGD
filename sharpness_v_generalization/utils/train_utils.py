import sys
sys.path.append("../")
import torch
from torchvision import datasets, transforms
from sklearn.metrics import average_precision_score
from utils.data_loader_mnist import MNIST
from utils.data_loader_cifar import CIFAR10, CIFAR10_NOISY


def get_loader(args, training, data_load_fraction=1.0, label_noise=0.0):
  """ function to get data loader specific to different datasets
  """
  if args.dtype == 'cifar10':
    dsets = cifar10_dsets(args, training, data_load_fraction, label_noise)
  elif args.dtype == 'cifar10_noisy':
    dsets = cifar10_noisy_dsets(args, training, args.dn)
  elif args.dtype == 'mnist':
    dsets = mnist_dsets(args, training, data_load_fraction, label_noise)

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


def mnist_dsets(args, training, data_load_fraction, label_noise):
  """ Function to load mnist data
  """
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])

  if training is True:
    dsets = {
        'train': MNIST(args.data_dir, train=True, download=True,
                       transform=transform, data_load_fraction=data_load_fraction,
                       label_noise=label_noise),
        'val': MNIST(args.data_dir, train=False, download=True,
                     transform=transform)
    }
  else:
    dsets = {
        'test': MNIST(args.data_dir, train=False, download=True,
                      transform=transform)
    }

  return dsets


def cifar10_noisy_dsets(args, training, data_noise):
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
      'train': CIFAR10_NOISY(root=args.data_dir, train=True, data_noise=args.dn),
      'val': CIFAR10(root=args.data_dir.split('cifar10_noisy')[0]+"cifar10/", train=False,
                     download=True, transform=transform['val'])
    }
  else:
    dsets = {
      'test': CIFAR10(root=args.data_dir.split('cifar10_noisy')[0]+"cifar10/", train=False,
                      download=True, transform=transform['val'])
    }

  return dsets


def cifar10_dsets(args, training, data_load_fraction, label_noise):
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
      'train': CIFAR10(root=args.data_dir, data_load_fraction=data_load_fraction,
                       label_noise=label_noise, train=True,
                       download=True, transform=transform['train']),
      'val': CIFAR10(root=args.data_dir, train=False,
                     download=True, transform=transform['val'])
    }
  else:
    dsets = {
      'test': CIFAR10(root=args.data_dir, train=False,
                      download=True, transform=transform['val'])
    }

  return dsets


def accuracy(output, target, topk=(1,)):
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


