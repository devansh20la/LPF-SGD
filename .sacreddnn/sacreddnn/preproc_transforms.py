import PIL
from torchvision import transforms
from .autoaugment import CIFAR10Policy
from .autoaugment_rwightman import auto_augment_transform

# FAST-AA
from .fast_autoaugment.archive import arsaug_policy, autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10, fa_reduced_svhn, fa_resnet50_rimagenet
from .fast_autoaugment.augmentations import *
#

def mean_std_dataset(dataset_name):
    if dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset_name == 'mnist':
        mean, std = (0.1307,), (0.3081,)
    elif dataset_name == 'fashion':
        mean, std = (0.2860,), (0.3530,)
    elif dataset_name == 'svhn':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset_name == 'tiny-imagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
    elif dataset_name == 'imagenet':
        mean = (0.485, 0.456, 0.406)  
        std = (0.229, 0.224, 0.225)

    return mean, std

def preproc_transforms(args):
     ## DATA PREPROCESSING
    if args.dataset.startswith('cifar'):
        transform_train, transform_test = preproc_cifar(args)
    elif args.dataset == 'mnist' or args.dataset == 'fashion':
        transform_train, transform_test = preproc_mnist(args)
    elif args.dataset == 'svhn':
        transform_train, transform_test = preproc_svhn(args)
    elif args.dataset =='tiny-imagenet':
        transform_train, transform_test = preproc_tinyimagenet(args)
    elif args.dataset =='imagenet':
        transform_train, transform_test = preproc_imagenet(args)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)
    return transform_train, transform_test

def preproc_imagenet(args):
    mean, std = mean_std_dataset(args.dataset)
        
    if args.preprocess == 1: # crop, hflip, normalization

        transform_train = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        transform_test = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        if args.augm_type == 'imagenet':
            transform_train.insert(0, Augmentation(fa_resnet50_rimagenet()))

        if args.cutout > 0:
            transform_train.append(CutoutDefault(args.cutout))
            print("CUTOUT: with length %s" % args.cutout)

    return transform_train, transform_test

def preproc_tinyimagenet(args):
    mean, std = mean_std_dataset(args.dataset)
        
    if args.preprocess == 1: # just normalization
        transform_train = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        
        transform_test = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        
    elif args.preprocess == 2: # crop, hflip, normalization
        width_crop = 64
        padding_crop = 8

        transform_train = [
            transforms.RandomCrop(width_crop, padding_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        transform_test = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        
        if args.augm_type == 'imagenet':
            transform_train.insert(0, Augmentation(fa_resnet50_rimagenet()))

        if args.cutout > 0:
            transform_train.append(CutoutDefault(args.cutout))
            print("CUTOUT: with length %s" % args.cutout)

    return transform_train, transform_test

def preproc_svhn(args):
    mean, std = mean_std_dataset(args.dataset)
        
    if args.preprocess == 1: # just normalization
        transform_train = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        
        transform_test = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        
    elif args.preprocess == 2: # crop, hflip, normalization
        width_crop = 32
        padding_crop = 4

        transform_train = [
            transforms.RandomCrop(width_crop, padding_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        transform_test = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        
        if args.augm_type == 'svhn':
            transform_train.insert(0, Augmentation(fa_reduced_svhn()))

        if args.cutout > 0:
            transform_train.append(CutoutDefault(args.cutout))
            print("CUTOUT: with length %s" % args.cutout)

    return transform_train, transform_test

def preproc_mnist(args):
    mean, std = mean_std_dataset(args.dataset)
    if args.preprocess == 0:
        transform_train = [transforms.ToTensor()]
        
        transform_test = [transforms.ToTensor()]
        
    elif args.preprocess == 1: # just normalization
        transform_train = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        
        transform_test = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

    return transform_train, transform_test

def preproc_cifar(args):
    mean, std = mean_std_dataset(args.dataset)
    if args.preprocess == 1: 
        # just normalization
        transform_train = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        transform_test = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

    elif args.preprocess == 2: # crop, hflip, normalization
        width_crop = 32
        padding_crop = 4 

        transform_train = [
            transforms.RandomCrop(width_crop, padding_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        transform_test = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

    elif args.preprocess == 3: # crop, hflip, normalization
        
        bin_norm = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        transform_train = pad_random_crop(32, scale_size=40, normalize=bin_norm)
        transform_test = scale_crop(input_size=32, scale_size=32, normalize=bin_norm)

    elif args.preprocess == 4: # resize, crop, hflip, autoaugment, normalization
        # TODO add paper reference (maybe autoaugment)
        size = 224
        width_crop = 224
        padding_crop = 32 

        # TODO check if normalization is correct
        transform_train = [
            transforms.Resize(size),
            transforms.RandomCrop(width_crop, padding_crop),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(), # TODO: check if applies also to Cifar100
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        transform_test = [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
    
    elif args.preprocess == 5: # resize, crop, hflip, autoaugment, normalization
        # TODO add paper reference (maybe autoaugment)
        # efficientnet b0
        size = 224
        width_crop = 224
        padding_crop = 32
        # efficientnet b7
        #size = 600
        #width_crop = 600
        #padding_crop = 75 

        # TODO check if normalization is correct
        transform_train = [
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
            transforms.RandomCrop(width_crop, padding_crop),
            #transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(), # TODO: check if applies also to Cifar100
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        transform_test = [
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
    
    elif args.preprocess == 6: # resize, crop, hflip, autoaugment, normalization
        # TODO add paper reference (maybe autougment)
        size = 224
        width_crop = 224
        padding_crop = 32 
        
        aa_params = dict(translate_const=int(size * 0.45),
                         img_mean=tuple([min(255, round(255 * x)) for x in (0.4914, 0.4822, 0.4465)]),
                         interpolation=PIL.Image.BICUBIC
                        )

        # TODO check if normalization is correct
        transform_train = [
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
            transforms.RandomCrop(width_crop, padding_crop),
            #transforms.RandomResizedCrop(size, interpolation=PIL.Image.BICUBIC),
            
            transforms.RandomHorizontalFlip(),
            
            # not working at the moment
            auto_augment_transform('v0', hparams=None), # TODO: this should be the policy from effnet
            
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        transform_test = [
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

    elif args.preprocess == 7: # resize, crop, hflip, autoaugment, normalization
        # from ShakeDrop: https://arxiv.org/abs/1802.02375
        # TODO: check AutoAugment and FastAutoaugment paper
        width_crop = 32
        padding_crop = 4

        # TODO check if normalization is correct
        transform_train = [
            CIFAR10Policy(),
            transforms.RandomCrop(width_crop, padding_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        transform_test = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

    elif args.preprocess == 8: # resize, crop, hflip, autoaugment, normalization
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        transform_test = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        
        if args.augm_type == 'fa_reduced_cifar10':
            transform_train.transforms.insert(0, Augmentation(fa_reduced_cifar10()))

        elif args.augm_type == 'fa_reduced_imagenet':
            transform_train.transforms.insert(0, Augmentation(fa_resnet50_rimagenet()))

        elif args.augm_type == 'fa_reduced_svhn':
            transform_train.transforms.insert(0, Augmentation(fa_reduced_svhn()))

        elif args.augm_type == 'arsaug':
            transform_train.transforms.insert(0, Augmentation(arsaug_policy()))
        elif args.augm_type == 'autoaug_cifar10':
            transform_train.insert(0, Augmentation(autoaug_paper_cifar10()))
        elif args.augm_type == 'autoaug_extend':
            transform_train.insert(0, Augmentation(autoaug_policy()))
        elif args.augm_type in ['default', 'inception', 'inception320']:
            pass
        else:
            raise ValueError('not found augmentations. %s' % args.augmentation_type)
        
        if args.cutout > 0:
            transform_train.append(CutoutDefault(args.cutout))
            print("CUTOUT: with length %s" % args.cutout)
    return transform_train, transform_test

    

# FAST-AA
class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img
    
# for binary_resnet

def scale_crop(input_size, scale_size=None, normalize={}):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Scale(scale_size)] + t_list

    return t_list

def pad_random_crop(input_size, scale_size=None, normalize={}):
    padding = int((scale_size - input_size) / 2)
    return [transforms.RandomCrop(input_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**normalize),
           ]
