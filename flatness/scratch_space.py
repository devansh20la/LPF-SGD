# import glob

# for fol in glob.glob("*sgd_*"):
#     for file in glob.glob(fol + "/run_ms_0/*.log"):
#         f = open(file, 'r')
#         cont = f.read()
#         if 'nan' in cont:
#             print(file)

from sklearn.model_selection import ParameterGrid
import glob
import shutil

param_grid = {'ms': [0],  # seed
              'mo': [0.0, 0.5, 0.9],  # momentum
              'width': [4, 6, 8],  # network width
              'wd': [1e-2],  # weight decay
              'lr': [5e-3, 1e-2, 5e-2],  # learning rate
              'bs': [16, 32, 64],  # batch size
              'lr_decay': [True, False],  # learning rate decay
              'skip': [True, False],  # skip connection
              'batchnorm': [True, False]  # batchnorm
              }

grid = list(ParameterGrid(param_grid))
for param in grid:
    if param['wd'] == 0.01:
        n = f"{param['width']}_{param['batchnorm']}_{param['skip']}_" \
            f"sgd_250_{param['lr']}_{param['wd']}_" \
            f"{param['bs']}_{param['mo']}_{param['lr_decay']}"
        replace_fol = glob.glob(f"testing/*{n}")[0]
        print(replace_fol)
        replace_fol_num = replace_fol.split('/')[1].split('_')[0]
        n = f"{param['width']}_{param['batchnorm']}_{param['skip']}_" \
            f"sgd_250_{param['lr']}_{5e-4}_" \
            f"{param['bs']}_{param['mo']}_{param['lr_decay']}"
        replace_fol_with = glob.glob(f"all_new_new_cifar10/*{n}")[0]
        replace_fol_with_num = replace_fol_with.split('/')[1].split('_')[0]
        shutil.rmtree(replace_fol)
        shutil.copytree(replace_fol_with, f"testing/{replace_fol_num}_{n}")

# fols = glob.glob1("."/ "*sgd_*")

# for i,param in enumerate(grid,0):


#   with open(n + '')
#   print()


# import glob

# with open("results/resnet_cifar10.csv", 'w') as f:
#     f.write("exp_num, width,batchnorm,skip,opt, epoch, lr,wd,bs,mo,lr-decay,train_loss, train_err, val_loss,val_err, flatness, div\n")

# for fol in glob.glob("checkpoints/all_new_cifar10/*sgd_*"):
#     with open(fol + "/run_ms_0/dist_bw_params.txt", 'r') as f:
#         h = next(f)
#         cont = next(f)
#     with open("results/resnet_cifar10.csv", 'a') as f:
#         f.write(f"{','.join(fol.split('/')[-1].split('_'))}, {cont}")
