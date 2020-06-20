import torch
import torch.nn as nn
from models import resnet18_narrow as resnet18
from utils import get_loader
from utils.train_utils import AverageMeter, accuracy
import argparse
from sklearn.model_selection import ParameterGrid
import pickle
from tqdm import tqdm
import glob


def main(args):
    dset_loaders = get_loader(args, training=True)
    model = resnet18(args)

    if args.use_cuda:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    model.load_state_dict(torch.load(f"{args.cp_dir}/trained_model.pth.tar"))
    model.eval()
    criterion = nn.CrossEntropyLoss()

    train_loss_mtr = AverageMeter()
    val_loss_mtr = AverageMeter()
    err1 = {'train': AverageMeter(), 'val': AverageMeter()}
    for phase in ["train", "val"]:
        for inp_data in dset_loaders[phase]:
            inputs, targets = inp_data

            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                batch_err = accuracy(outputs, targets, topk=(1, 5)) 
                if phase == "train":
                    train_loss_mtr.update(batch_loss.item(), inputs.size(0))
                    err1['train'].update(100.0 - batch_err[0].item(), inputs.size(0))
                elif phase == "val":
                    val_loss_mtr.update(batch_loss.item(), inputs.size(0))
                    err1['val'].update(100.0 - batch_err[0].item(), inputs.size(0))

    gen_gap = val_loss_mtr.avg - train_loss_mtr.avg
    return err1['train'].avg, err1['val'].avg, gen_gap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--dtype', type=str, default="cifar10", help='Data type')
    parser.add_argument('--optim', type=str, default="sgd")
    parser.add_argument('--ep', type=int, default=250, help='Epochs')

    args = parser.parse_args()

    param_grid = {'m': [0.9, 0.5, 0.0],
                  'wd': [0.0, 1e-2, 1e-4],
                  'ms': [1],
                  'lr_decay': ["True", "False"],
                  'lr': [0.01, 0.001, 0.005],
                  'bs': [16, 32, 64, 128],
                  'skip': ["True", "False"],
                  'regular': ["dropout", "batch_norm", "none"]}

    grid = list(ParameterGrid(param_grid))

    for exp_num, params in enumerate(tqdm(grid), 0):

        args.m = params['m']
        args.wd = params['wd']
        args.ms = params['ms']
        args.lr_decay = params['lr_decay']
        args.lr = params['lr']
        args.bs = params['bs']
        args.skip = params['skip']
        args.regular = params['regular']
        args.width = 1

        args.n = f"all_{args.dtype}/*_{args.optim}_{args.ep}_{args.lr}" \
                 f"_{args.wd}_{args.bs}_{args.m}_{args.regular}_{args.skip}_{args.lr_decay}"

        args.bs = 1024

        for fol in glob.glob(f"checkpoints/{args.n}"):
            args.cp_dir = f"{fol}/run_ms_1"

            args.data_dir = f"{args.dir}/data/{args.dtype}"
            args.use_cuda = torch.cuda.is_available()

            train_err, val_err, gen_gap = main(args)
            print(train_err, val_err)
