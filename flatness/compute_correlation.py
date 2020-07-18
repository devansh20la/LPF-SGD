import scipy
from sklearn.model_selection import ParameterGrid
import glob
from utils.train_utils import AverageMeter
import numpy as np
from sklearn.model_selection import ParameterGrid
import copy


mtr = AverageMeter()

param_grid = {
              'mo': [0.0, 0.5, 0.9],  # momentum
              'width': [4, 6, 8],  # network width
              'wd': [0.0, 1e-4, 1e-2],  # weight decay
              'lr': [5e-3, 1e-2, 5e-2],  # learning rate
              'bs': [16, 32, 64],  # batch size
              'lr_decay': [True, False],  # learning rate decay
              'skip': [True, False],  # skip connection
              'batchnorm': [True, False]  # batchnorm
              }

gen_gap_corr = []
for key, value in param_grid.items():
    grid = copy.deepcopy(param_grid)
    del grid[key]

    grid = list(ParameterGrid(grid))
    corr = []

    for params in grid:
        flat_measure = []
        hyper_param = []
        gen_gap = []
        for v in value:
            params[f"{key}"] = v

            name = f"checkpoints/all_new_cifar10/" \
                   f"*_{params['width']}_{params['batchnorm']}_{params['skip']}_" \
                   f"sgd_250_{params['lr']}_{params['wd']}_" \
                   f"{params['bs']}_{params['mo']}_{params['lr_decay']}"

            fol = glob.glob(name)[0]
            with open(f"{fol}/run_ms_0/dist_bw_params.txt", 'r') as f:
                header = next(f)
                row = next(f)

                if 'nan' in row:
                    continue

                if float(row.split(',')[1]) > 5:
                    continue

                flat_measure.append(float(row.split(',')[4]))
                gen_gap.append(float(row.split(',')[3]) - float(row.split(',')[1]))
                if v is True:
                    hyper_param.append(1)
                elif v is False:
                    hyper_param.append(0)
                else:
                    hyper_param.append(v)

        if len(flat_measure) > 1 and len(list(np.unique(flat_measure))) > 1:
            try:
                corr.append(scipy.stats.spearmanr(hyper_param, flat_measure)[0])
            except:
                pass
            try:
                gen_gap_corr.append(scipy.stats.spearmanr(hyper_param, flat_measure)[0])
            except:
                pass

    print(key, value, np.mean(corr))


grid = list(ParameterGrid(param_grid))
flat_measure = []
gen_gap = []

for params in grid:

    name = f"checkpoints/all_new_cifar10/" \
           f"*_{params['width']}_{params['batchnorm']}_{params['skip']}_" \
           f"sgd_250_{params['lr']}_{params['wd']}_" \
           f"{params['bs']}_{params['mo']}_{params['lr_decay']}"

    fol = glob.glob(name)[0]
    with open(f"{fol}/run_ms_0/dist_bw_params.txt", 'r') as f:
        header = next(f)
        row = next(f)

        if 'nan' in row:
            continue

        if float(row.split(',')[1]) > 5:
            continue

        flat_measure.append(float(row.split(',')[4]))
        gen_gap.append(float(row.split(',')[3]) - float(row.split(',')[1]))


print(scipy.stats.spearmanr(gen_gap, flat_measure)[0])
