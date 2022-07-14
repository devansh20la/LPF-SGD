import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utility.cutout import Cutout


class Cifar:
    def __init__(self, batch_size, threads, want_cifar100=False):
        self.want_cifar100 = want_cifar100
        mean, std = self._get_statistics()

        if self.want_cifar100:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            train_set = torchvision.datasets.CIFAR100(root='./cifar', train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR100(root='./cifar', train=False, download=True, transform=test_transform)
        else:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR10(root='./cifar', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads, pin_memory=True)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads, pin_memory=True)
    
    def _get_statistics(self):
        if self.want_cifar100:
            train_set = torchvision.datasets.CIFAR100(root='./cifar', train=True, download=True, transform=transforms.ToTensor())
        else:
            train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

