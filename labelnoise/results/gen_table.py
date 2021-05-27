import tensorflow as tf
import glob
import numpy as np
import os 
import pickle
from scipy import stats

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# print("opt\t&\tval_err", flush=True)
# for mtype in ["28_10"]:
#     for aug in ["autoaugment_cutout"]:
#         print(f"{mtype} \t&\t {aug} \t&\t", end='')
#         for dtype in ["cifar100"]:
#             for opt in ["ssgd"]:
#                 best_avg = []
#                 best_fol_avg = []
#                 for seed in [0,1,2,3,4]:
#                     all_folders =  glob.glob(f'../checkpoints/greene/{aug}/{dtype}*/{opt}/run_ms_{seed}/{mtype}*_0.0007/run0/*event*', recursive=True)
#                     curr_best = float('inf')
#                     for file in all_folders:
#                         for e in tf.compat.v1.train.summary_iterator(file):
#                             for v in e.summary.value:
#                                 if "val_err1" in v.tag:
#                                     last = v.simple_value
#                                     if curr_best > v.simple_value:
#                                         curr_best = v.simple_value
#                                         best_fol = file
#                     best_avg.append(curr_best)
#                     best_fol_avg.append(best_fol)

#                 std = stats.norm.interval(.95, loc = np.mean(best_avg), scale = np.std(best_avg)/np.sqrt(len(best_avg)))
#                 std =  std[1] - np.mean(best_avg)
#                 print(f"${np.mean(best_avg):<02.1f}_{{\pm {std:<1.1f}}}$\t&\t")
#                 for x in range(len(best_avg)):
#                     print(best_avg[x], best_fol_avg[x])
#         print('')

print("opt\t&\tval_err", flush=True)
for opt in  ["sgd", "sam", "ssgd"]:
    for noise in ["0.0", "20.0", "40.0", "60.0", "80.0"]:
        best_avg = []
        best_fol_avg = []
        for seed in [0, 1]:
            all_folders =  glob.glob(f'../checkpoints/cifar10/resnet32/{opt}/noise_{noise}/run*_ms_{seed}/*/*event*', recursive=True)
            curr_best = float('inf')
            for file in all_folders:
                for e in tf.compat.v1.train.summary_iterator(file):
                    for v in e.summary.value:
                        if "Val/Val_Err1" in v.tag:
                            last = v.simple_value
                            if curr_best >= v.simple_value:
                                curr_best = v.simple_value
                                best_fol = file
            best_avg.append(curr_best)
            best_fol_avg.append(best_fol) 
        std = stats.norm.interval(.95, loc = np.mean(best_avg), scale = np.std(best_avg)/np.sqrt(len(best_avg)))
        std =  std[1] - np.mean(best_avg)
        print(f"${100 - np.mean(best_avg):<02.1f}_{{\pm {std:<1.1f}}}$ \t&\t", end='')
        # print('')
        # for x in range(len(best_avg)):
        #     print(best_avg[x], best_fol_avg[x])
    print('')



