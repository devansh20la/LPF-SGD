import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, MNIST, ImageFolder
import numpy as np

def _cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def get_cosine_annealing_scheduler(optimizer, epochs, steps_per_epoch, base_lr):
    lr_min = 0.0
    total_steps = epochs * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_annealing(
            step,
            total_steps,
            1,  # since lr_lambda computes multiplicative factor
            lr_min / base_lr))

    return scheduler
    
def get_loader(args, training):
    """ function to get data loader specific to different datasets
    """
    if args.dtype == 'cifar10':
        dsets = cifar10_dsets(args, training)
    else:
        print("Wring data type")
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

    # add noise to labels
    add_noise_cifar_w(dset_loaders['train'], noise_percentage=args.noise)

    return dset_loaders


def add_noise_cifar_w(loader, noise_percentage = 20):
    torch.manual_seed(2)
    np.random.seed(42)
    noisy_labels = [sample_i for sample_i in loader.sampler.data_source.targets]
    images = [sample_i for sample_i in loader.sampler.data_source.data]
    probs_to_change = torch.randint(100, (len(noisy_labels),))
    idx_to_change = probs_to_change >= (100.0 - noise_percentage)
    percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))

    for n, label_i in enumerate(noisy_labels):
        if idx_to_change[n] == 1:
            set_labels = list(set(range(10)))  # this is a set with the available labels (with the current label)
            set_index = np.random.randint(len(set_labels))
            noisy_labels[n] = set_labels[set_index]

    loader.sampler.data_source.data = images
    loader.sampler.data_source.targets = noisy_labels

    return noisy_labels


def cifar10_dsets(args, training):
    """ Function to load cifar10 data"""
    tf = {}
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    tf['train'] = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
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
    dsets = get_loader(args, True)
