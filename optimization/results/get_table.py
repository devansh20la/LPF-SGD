import tensorflow as tf
import glob
import numpy as np
import os 
import pickle
from scipy import stats

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print("opt\t&\tval_err", flush=True)
for dtype in ["cifar10", "cifar100"]:
    for mtype in ["resnet18", "resnet50", "resnet101"]:
        print(f"{dtype}\t&\t {mtype} \t&\t", end='')
        for opt in ["sgd", "sam_sgd", "entropy_sgd", "fsgd"]:
            best_avg = []
            best_fol_avg = []
            if dtype == 'cifar100' and mtype == 'resnet101':
                all_seeds = [1,2,3,4]
            else:
                all_seeds = [0,1,2,3,4]
            for seed in all_seeds:
                all_folders =  glob.glob(f'../checkpoints/{dtype}/{mtype}/{opt}/run_ms_{seed}/run0/*event*', recursive=True)
                if opt == "entropy_sgd":
                    all_folders =  glob.glob(f'../checkpoints/entropy_sgd/{dtype}/{mtype}/{opt}/run_ms_{seed}/run0/*event*', recursive=True)

                curr_best = float('inf')
                for file in all_folders:
                    for e in tf.compat.v1.train.summary_iterator(file):
                        for v in e.summary.value:
                            if "Val/Val_Err1" in v.tag:
                                last = v.simple_value
                                if curr_best > v.simple_value:
                                    curr_best = v.simple_value
                                    best_fol = file
                best_avg.append(curr_best)
                best_fol_avg.append(best_fol)
            std = stats.norm.interval(.95, loc = np.mean(best_avg), scale = np.std(best_avg)/np.sqrt(len(best_avg)))
            std =  std[1] - np.mean(best_avg)
            print(f"${np.mean(best_avg):<02.1f}_{{\pm {std:<1.1f}}}$\t&\t", end='')
            # print('')
            # for x in range(len(best_avg)):
            #     print(best_avg[x], best_fol_avg[x])
        print('')

print("opt\t&\tval_err", flush=True)
for dtype in ["tinyimagenet", "imagenet"]:
    for mtype in ["resnet18"]:
        print(f"{dtype}\t&\t {mtype} \t&\t", end='')
        for opt in ["sgd", "sam_sgd", "fsgd"]:
            best_avg = []
            best_fol_avg = []
            if dtype == 'cifar100' and mtype == 'resnet101':
                all_seeds = [1,2,3,4]
            else:
                all_seeds = [0,1,2,3,4]
            for seed in all_seeds:
                all_folders =  glob.glob(f'../checkpoints/{dtype}/{mtype}/{opt}/run_ms_{seed}/run0/*event*', recursive=True)

                curr_best = float('inf')
                for file in all_folders:
                    for e in tf.compat.v1.train.summary_iterator(file):
                        for v in e.summary.value:
                            if "Val/Val_Err1" in v.tag:
                                last = v.simple_value
                                if curr_best > v.simple_value:
                                    curr_best = v.simple_value
                                    best_fol = file
                best_avg.append(curr_best)
                best_fol_avg.append(best_fol)
            std = stats.norm.interval(.95, loc = np.mean(best_avg), scale = np.std(best_avg)/np.sqrt(len(best_avg)))
            std =  std[1] - np.mean(best_avg)
            print(f"${np.mean(best_avg):<02.1f}_{{\pm {std:<1.1f}}}$\t&\t", end='')
            # print('')
            # for x in range(len(best_avg)):
            #     print(best_avg[x], best_fol_avg[x])
        print('')



