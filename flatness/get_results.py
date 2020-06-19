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
              'skip': ["True", "False"],
              'regular': ["batch_norm", "dropout", "none"]}

grid = list(ParameterGrid(param_grid))

for params in grid:
    flat_measure = []
    for lr_decay in [False, True]:
        name = f"checkpoints/all_cifar10/*_sgd_250_{params['lr']}" \
               f"_{params['wd']}_{params['bs']}_{params['m']}_{params['regular']}_{params['skip']}_{lr_decay}"

        if len(glob.glob(f"{name}/run_ms_1/dist_bw_params.txt")) > 1:
            print("Error")
            quit()
        for file in glob.glob(f"{name}/run_ms_1/dist_bw_params.txt"):
            with open(file, 'r') as f:
                header = next(f)
                flat_measure.append(float(next(f).split(',')[-1]))

        # print(file, flat_measure[-1])
        # print(file)
    if scipy.stats.pearsonr([0, 1], flat_measure)[0] is np.nan:
        continue
    else:
        mtr.update(scipy.stats.pearsonr([0, 1], flat_measure)[0], 1)

print(mtr.avg)
