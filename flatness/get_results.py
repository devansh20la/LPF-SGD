import scipy
from sklearn.model_selection import ParameterGrid
import glob
from utils.train_utils import AverageMeter
import numpy as np

mtr = AverageMeter()

param_grid = {'m': [0.9, 0.5, 0.0],
              'wd': [0.0, 1e-2, 1e-4],
              'ms': [1],
              'lr': [0.01, 0.001, 0.005],
              'bs': [16, 32, 64, 128],
              'lr_decay': [False, True],
              'skip': ["True", "False"],
              'regular': ["batch_norm", "dropout", "none"]
              }

grid = list(ParameterGrid(param_grid))
flat_measure = []
gen_gap = []

for params in grid:
    name = f"checkpoints/all_cifar10/*_sgd_250_{params['lr']}" \
           f"_{params['wd']}_{params['bs']}_{params['m']}_{params['regular']}_{params['skip']}_{params['lr_decay']}"

    if len(glob.glob(f"{name}/run_ms_1/dist_bw_params.txt")) > 1:
        print("Error")
        quit()
    for file in glob.glob(f"{name}/run_ms_1/dist_bw_params.txt"):
        with open(file, 'r') as f:
            header = next(f)
            row = next(f)
            flat_measure.append(float(row.split(',')[-1]))
            gen_gap.append(float(row.split(',')[0]))

print(scipy.stats.pearsonr(gen_gap, flat_measure)[0])
