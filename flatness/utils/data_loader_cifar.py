from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import torch
from torchvision import transforms
import scipy.io


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_load_fraction=1.0, label_noise=0.0):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        if data_load_fraction > 1:
            raise RuntimeError("Stupid! You are trying to load more than 100\% data")
        else:
            self.data_load_fraction = data_load_fraction

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data)
        size_per_class, _ = np.histogram(self.targets, bins=np.arange(0,101))

        count = [0] * len(np.unique(self.targets))
        indexes = []
        for idx, class_label in enumerate(self.targets):
            if count[class_label] >= int(self.data_load_fraction * size_per_class[class_label]):
                pass
            else:
                count[class_label] += 1
                indexes.append(idx)

        print("Exs per label:{0}".format(count))
        self.data = self.data[indexes]
        self.targets = torch.Tensor([self.targets[x] for x in indexes]).type(torch.LongTensor).reshape(-1)
        if label_noise > 0.0:
            for i in range(self.targets.shape[0]):
                t = np.random.random()
                if t < label_noise:
                    self.targets[i] = torch.randint(0, 10, self.targets[i].shape)

        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR10_NOISY(data.Dataset):

    def __init__(self, root, train=True, data_noise=0.0):

        if train:
            data = torch.load(f"{root}/cifar10_data_train_{data_noise}.pt")
            self.data, self.targets = torch.cat(data["inputs"]), torch.cat(data["targets"])
        else:
            print("This is only for training data")

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    transform = {
        'train': transforms.Compose([transforms.ToTensor()]),
        'val': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))])
    }
    dsets = {
      'train': CIFAR10(root="../data/cifar10/", train=True, download=True, transform=transform['train']),
      'val': CIFAR10(root="../data/cifar10/", train=False, download=True, transform=transform['val'])
    }

    dset_loaders = {
        'train': torch.utils.data.DataLoader(dsets['train'], batch_size=10,
                                             shuffle=False, pin_memory=True,
                                             num_workers=8),
        'val': torch.utils.data.DataLoader(dsets['val'], batch_size=128,
                                           shuffle=False, pin_memory=True,
                                           num_workers=8)
    }

    def norm(x):
        normalize = transforms.Normalize(
                            (0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
        for i in range(x.shape[0]):
            x[i] = normalize(x[i])
        return x

    for noise in [i/10 for i in range(32, 61, 2)]:
        data = []
        labels = []
        for inputs in dset_loaders["train"]:
            data.append(norm(inputs[0] + torch.normal(torch.zeros(inputs[0].shape), noise*torch.ones(inputs[0].shape))))
            labels.append(inputs[1])

        torch.save({"inputs": data, "targets": labels}, f"../data/cifar10_noisy/cifar10_data_train_{noise}.pt")
        print(torch.cat(data).shape, torch.cat(labels).shape)
