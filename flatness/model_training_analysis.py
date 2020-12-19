import argparse
from sklearn.model_selection import ParameterGrid
import pickle
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--dtype', type=str, default="cifar10", help='Data type')
    parser.add_argument('--optim', type=str, default="sgd")
    parser.add_argument('--ep', type=int, default=250, help='Epochs')

    args = parser.parse_args()

    param_grid = {'ms': [0],  # seed
                  'mo': [0.0, 0.5, 0.9],  # momentum
                  'width': [4, 6, 8],  # network width
                  'wd': [0.0, 1e-4, 5e-4],  # weight decay
                  'lr': [0.01, 0.0075, 0.005],  # learning rate
                  'bs': [32, 128, 512],  # batch size
                  'skip': [True, False],
                  'batchnorm': [True, False]  # batchnorm
                  }

    check_grid = {'mo': {'0.9': [0, 0], '0.5': [0, 0], '0.0': [0, 0]},
                  'width': {'4': [0, 0], '6': [0, 0], '8': [0, 0]},
                  'wd': {'0.0': [0, 0], '0.0005': [0, 0], '0.0001': [0, 0]},
                  'lr': {'0.01': [0, 0], '0.0075': [0, 0], '0.005': [0, 0]},
                  'bs': {'32': [0, 0], '128': [0, 0], '512': [0, 0]},
                  'skip': {"True": [0, 0], "False": [0, 0]},
                  'batchnorm': {"True": [0, 0], "False": [0, 0]}
                  }

    grid = list(ParameterGrid(param_grid))

    for exp_num, params in enumerate(tqdm(grid), 0):
        args.exp_num = exp_num
        args.ms = params['ms']
        args.mo = params['mo']
        args.width = params['width']
        args.wd = params['wd']
        args.lr = params['lr']
        args.bs = params['bs']
        args.skip = params['skip']
        args.batchnorm = params['batchnorm']

        args.n = f"{args.dtype}/resnet/" \
                 f"{args.exp_num}_{args.ms}_{args.mo}_{args.width}_{args.wd}_" \
                 f"{args.lr}_{args.bs}_{args.skip}_{args.batchnorm}"

        check_grid['mo'][f"{params['mo']}"][0] += 1
        check_grid['width'][f"{params['width']}"][0] += 1
        check_grid['wd'][f"{params['wd']}"][0] += 1
        check_grid['lr'][f"{params['lr']}"][0] += 1
        check_grid['bs'][f"{params['bs']}"][0] += 1
        check_grid['skip'][f"{params['skip']}"][0] += 1
        check_grid['batchnorm'][f"{params['batchnorm']}"][0] += 1
        args.cp_dir = f"checkpoints/{args.n}/run_ms_0/"

        with open(args.cp_dir + "measures.pkl", 'rb') as f:
            measure = pickle.load(f)
            train_err = 100 - measure['train_acc']
            val_err = 100 - measure['val_acc']

        if measure['train_loss'] < 0.01:
            check_grid['mo'][f"{params['mo']}"][1] += 1
            check_grid['width'][f"{params['width']}"][1] += 1
            check_grid['wd'][f"{params['wd']}"][1] += 1
            check_grid['lr'][f"{params['lr']}"][1] += 1
            check_grid['bs'][f"{params['bs']}"][1] += 1
            check_grid['skip'][f"{params['skip']}"][1] += 1
            check_grid['batchnorm'][f"{params['batchnorm']}"][1] += 1

    with open('results/summ_resnet_cifar10.csv', 'w') as f:
        f.write("hyper-parameter, good_exp, bad_exp\n")

        for k1, v1 in check_grid.items():
            for k2, v2 in v1.items():
                f.write(f"{k1}-{k2}, {v2[1]}, {v2[0]-v2[1]}\n")
