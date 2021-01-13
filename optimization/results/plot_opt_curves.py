import tensorflow as tf
import glob
import numpy as np


folders = sorted(glob.glob('../checkpoints/cifar10/resnet18/sgd/*'), key=lambda y: y.split('run_ms_')[1])

data = {
    'trainloss': {
        'values': np.zeros((1, 300, len(folders))),
        'walltime': np.zeros((1, 300, len(folders))),
    },
    'trainerr': {
        'values': np.zeros((1, 300, len(folders))),
        'walltime': np.zeros((1, 300, len(folders))),
    },
    'valloss': {
        'values': np.zeros((1, 300, len(folders))),
        'walltime': np.zeros((1, 300, len(folders))),
    },
    'valerr': {
        'values': np.zeros((1, 300, len(folders))),
        'walltime': np.zeros((1, 300, len(folders))),
    }
}

for k, f in enumerate(folders):
    file = glob.glob(f'{f}/*event*')[0]
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == "Train/Train_Loss":
                data['trainloss']['values'][0, e.step, k] = v.simple_value
                data['trainloss']['walltime'][0, e.step, k] = e.wall_time
            if v.tag == "Val/Val_Loss":
                data['valloss']['values'][0, e.step, k] = v.simple_value
                data['valloss']['walltime'][0, e.step, k] = e.wall_time
            if v.tag == "Train/Train_Err" or v.tag == "Train/Train_Err1":
                data['trainerr']['values'][0, e.step, k] = v.simple_value
                data['trainerr']['walltime'][0, e.step, k] = e.wall_time
            if v.tag == "Val/Val_Err" or v.tag == "Val/Val_Err1":
                data['valerr']['values'][0, e.step, k] = v.simple_value
                data['valerr']['walltime'][0, e.step, k] = e.wall_time
