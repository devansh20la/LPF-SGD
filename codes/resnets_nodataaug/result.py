import tensorflow as tf
import glob
import numpy as np
import os 
import pickle
from scipy import stats
from matplotlib import pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# print("opt\t&\tval_err", flush=True)
# for dtype in ["cifar10", "cifar100"]:
#     for mtype in ["resnet18", "resnet50", "resnet101"]:
#         print(f"{dtype:8s}\t&\t {mtype} \t&\t", end='')
#         for opt in ["sgd", "entropy_sgd","smooth_out", "sam_sgd", "lpfsgd"]:
#             best_avg = []

#             for seed in [0,1,2,3,4]:
#                 all_folders =  glob.glob(f'checkpoints/{dtype}/{mtype}/{opt}/run_ms_{seed}/run0/*event*', recursive=True)
#                 curr_best = float('inf')
#                 for file in all_folders:
#                     for e in tf.compat.v1.train.summary_iterator(file):
#                         for v in e.summary.value:
#                             if "Val/Val_Err1" in v.tag:
#                                 if curr_best > v.simple_value:
#                                     curr_best = v.simple_value
#                 if curr_best != float('inf'):
#                     best_avg.append(curr_best)
#             print(f"{np.mean(best_avg):<02.1f}\t", end='')
#         print("")


# print("opt\t&\tval_err", flush=True)
# for dtype in ["imagenet"]:
#     for mtype in ["resnet18"]:
#         print(f"{dtype:8s}\t&\t {mtype} \t&\t", end='')
#         for opt in ["sgd", "entropy_sgd","smooth_out", "sam_sgd", "lpfsgd"]:
#             best_avg = []

#             for seed in [0,1,2,3,4]:
#                 all_folders =  glob.glob(f'checkpoints/{dtype}/{mtype}/{opt}/run_ms_{seed}/run0/*event*', recursive=True)
#                 curr_best = float('inf')
#                 for file in all_folders:
#                     for e in tf.compat.v1.train.summary_iterator(file):
#                         for v in e.summary.value:
#                             if "Val/Val_Err1" in v.tag:
#                                 if curr_best > v.simple_value:
#                                     curr_best = v.simple_value
#                 if curr_best != float('inf'):
#                     best_avg.append(curr_best)
#             print(f"{np.mean(best_avg):<02.1f}\t", end='')
#         print("")

print("opt\t&\tval_err", flush=True)
for dtype in ["tinyimagenet"]:
    for mtype in ["resnet18"]:
        print(f"{dtype:8s}\t&\t {mtype} \t&\t", end='')
        for opt in ["sgd", "entropy_sgd","smooth_out", "sam_sgd", "lpfsgd"]:
            best_avg = []

            for seed in [0,1,2,3,4]:
                all_folders =  glob.glob(f'checkpoints/{dtype}/{mtype}/{opt}/run_ms_{seed}/run0/*event*', recursive=True)
                curr_best = float('inf')
                for file in all_folders:
                    for e in tf.compat.v1.train.summary_iterator(file):
                        for v in e.summary.value:
                            if "Val/Val_Err1" in v.tag:
                                if curr_best > v.simple_value:
                                    curr_best = v.simple_value
                if curr_best != float('inf'):
                    best_avg.append(curr_best)
            print(f"{np.mean(best_avg):<02.1f}\t", end='')
        print("")
