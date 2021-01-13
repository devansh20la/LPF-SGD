import argparse
import torch
from models import cifar_resnet50, cifar_resnet18, cifar_resnet101
from utils import get_loader
from utils.train_utils import AverageMeter, accuracy
import pickle
from tqdm import tqdm


def main(args):

    dset_loaders = get_loader(args, training=False)

    if args.mtype == 'resnet50':
        model = cifar_resnet50(num_classes=args.num_classes)
    elif args.mtype == 'resnet18':
        model = cifar_resnet18(num_classes=args.num_classes)
    elif args.mtype == 'resnet101':
        model = cifar_resnet101(num_classes=args.num_classes)
    else:
        print("define model")
        quit()

    state = torch.load(f"{args.cp_dir}/best_model.pth.tar")
    model.load_state_dict(state['model'])

    if args.use_cuda:
        model = model.cuda()
        torch.backends.cudnn.benchmark = True

    model.eval()

    err1 = AverageMeter()

    for batch_idx, inp_data in enumerate(tqdm(dset_loaders['test']), 1):

        inputs, targets = inp_data

        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            outputs = model(inputs)

        batch_err = accuracy(outputs, targets, topk=(1, 5))
        err1.update(float(100.0 - batch_err[0]), inputs.size(0))

    mtr = {'err': err1.avg}
    print(mtr)
    file_name = args.cp_dir.split("best_model")[0]
    with open(f"{file_name}/measure.pkl", 'wb') as f:
        pickle.dump(mtr, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--dtype', type=str, required=True)
    parser.add_argument('--mtype', type=str, required=True)
    parser.add_argument('--ms', type=int, required=True)
    parser.add_argument('--optim', type=str, required=True)

    args = parser.parse_args()

    if args.dtype == 'cifar10':
        args.num_classes = 10
    elif args.dtype == 'cifar100':
        args.num_classes = 100
    elif args.dtype == 'imagenet':
        args.num_classes = 1000
    else:
        print(f"BAD COMMAND dtype: {args.dtype}")

    args.n = f"{args.dtype}/{args.mtype}/{args.optim}"
    args.cp_dir = f"{args.dir}/checkpoints/{args.n}/run_ms_{args.ms}/run4/"
    args.data_dir = f"{args.dir}/data/{args.dtype}"
    args.use_cuda = torch.cuda.is_available()

    main(args)
