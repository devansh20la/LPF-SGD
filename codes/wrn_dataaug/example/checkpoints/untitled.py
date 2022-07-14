import tensorflow as tf
import glob
import numpy as np
import os 
import pickle
from scipy import stats

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
####################################################################
import tensorflow as tf
import glob
import numpy as np
import os 
import pickle
from scipy import stats

###################################################################
                    # SGD SAM
###################################################################
# for mtype in ["16_8", "28_10", "shakeshake", "pyramidnet_110", "pyramidnet_272_bottleneck"]:
#     for aug in ["autoaugment_cutout"]:
#         print(f"{mtype} \t&\t {aug} \t&\t", end='')
#         for dtype in ["cifar10", "cifar100"]:
#             for opt in ["sgd", "sam"]:
#                 best = float('inf')
#                 seed_best = []
#                 for seed in [0, 1, 2, 3 ,4]:

#                     all_folders =  glob.glob(f'../checkpoints/{aug}/{dtype}/{opt}/run_ms_{seed}/{mtype}/run0/*event*')

#                     if len(all_folders) == 0: continue
#                     curr_best = float('inf')
#                     for e in tf.compat.v1.train.summary_iterator(all_folders[0]):
#                         for v in e.summary.value:
#                             if "val_err1" in v.tag:
#                                 last = v.simple_value
#                                 if v.simple_value < curr_best:
#                                     curr_best = v.simple_value
#                     seed_best.append(curr_best)

#                 if np.mean(seed_best) < best:
#                     best = np.mean(seed_best)
#                     std = stats.norm.interval(.95, loc = best, scale = np.std(seed_best)/np.sqrt(len(seed_best)))
#                     std =  std[1] - best

#                 print(f"${best:0.1f}_{{\pm {std:1.1f}}}$\t & \t", end='')
#                 # print(best_fol)
#         print('')


###################################################################
                    # SSGD
###################################################################
# for mtype in ["16_8", "28_10", "26_96_8_0.0005", "pyramidnet_110_8_8_0.0005",  "pyramidnet_272_bottleneck_8_8_0.0005"]:
#     for aug in ["autoaugment_cutout"]:
#         print(f"{mtype} \t&\t {aug} \t&\t", end='')
#         for dtype in ["cifar10", "cifar100"]:
#             for opt in ["ssgd"]:
#                 best = float('inf')
#                 seed_best = []
#                 for seed in [0, 1, 2, 3, 4]:
#                     if mtype == "16_8" or mtype == "28_10":
#                         if aug == 'autoaugment_cutout' and dtype == 'cifar100' and '28_10' in mtype:
#                             all_folders =  glob.glob(f'../checkpoints/{aug}/{dtype}/ssgd/run_ms_{seed}/{mtype}*0.0007/run0/*event*')
#                         else:
#                             all_folders =  glob.glob(f'../checkpoints/{aug}/{dtype}/ssgd/run_ms_{seed}/{mtype}*0.0005/run0/*event*', recursive=True)
#                     else:
#                         all_folders =  glob.glob(f'../checkpoints/{aug}/{dtype}/ssgd/run_ms_{seed}/{mtype}/run0/*event*', recursive=True)

#                     if len(all_folders) == 0:
#                         continue
                        
#                     curr_best = float('inf')
#                     for e in tf.compat.v1.train.summary_iterator(all_folders[0]):
#                         for v in e.summary.value:
#                             if "val_err1" in v.tag:
#                                 last = v.simple_value
#                                 if v.simple_value < curr_best:
#                                     curr_best = v.simple_value
#                     seed_best.append(curr_best)

#                 if np.mean(seed_best) < best:
#                     best = np.mean(seed_best)
#                     std = stats.norm.interval(.95, loc = best, scale = np.std(seed_best)/np.sqrt(len(seed_best)))
#                     std =  std[1] - best

#                 print(f"${best:0.1f}_{{\pm {std:1.1f}}}$\t & \t", end='')
#                 # print(best_fol)
#         print('')


###################################################################
                    # SMOOTH OUT
###################################################################
for mtype in ["wrn_16_8_0.009", "wrn_28_1_0.009", "shakeshake_26_96_0.009", "pyramidnet_110_8_0.009",  "pyramidnet_272_8_0.009"]:
    for aug in ["basic_none", "basic_cutout", "autoaugment_cutout"]:
        print(f"{mtype} \t&\t {aug} \t&\t", end='')
        for dtype in ["cifar10", "cifar100"]:
            for opt in ["smooth_out"]:
                best = float('inf')
                seed_best = []
                for seed in [0, 1, 2, 3, 4]:
                    all_folders =  glob.glob(f'../checkpoints_so/{aug}/{dtype}/{opt}/run_ms_{seed}/{mtype}/*event*')

                    curr_best = float('inf')
                    for e in tf.compat.v1.train.summary_iterator(all_folders[0]):
                        for v in e.summary.value:
                            if "val_err1" in v.tag:
                                last = v.simple_value
                                if v.simple_value < curr_best:
                                    curr_best = v.simple_value
                    seed_best.append(curr_best)

                if np.mean(seed_best) < best:
                    best = np.mean(seed_best)
                    std = stats.norm.interval(.95, loc = best, scale = np.std(seed_best)/np.sqrt(len(seed_best)))
                    std =  std[1] - best
                    best_fol = a

                print(f"${best:0.1f}_{{\pm {std:1.1f}}}$\t & \t", end='')
                # print(best_fol)
        print('')
