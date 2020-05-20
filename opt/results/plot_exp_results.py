import tensorflow as tf
import glob
import numpy as np
import scipy.io
from natsort import natsorted

save_dict = {}

for folder in natsorted(glob.glob("../checkpoints/mnist/sgd*")):
    data = {
        'TrainErr': np.zeros((1, 200)),
        'ValErr': np.zeros((1, 200)),
        'TrainLoss': np.zeros((1, 200)),
        'ValLoss': np.zeros((1, 200))
    }
    txt_file = glob.glob(f"{folder}/run0/*.log")[0]

    with open(txt_file, 'r') as f:
        args = f.readline()

    lr = args.split("lr=")[1].split(',')[0]
    m = args.split("m=")[1].split(',')[0]
    bs = args.split("bs=")[1].split(',')[0]

    file = glob.glob(f"{folder}/run0/*event*")[0]
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == "Train/Train_Err" or v.tag == "Train/Train_Err1":
                data['TrainErr'][0, e.step] = v.simple_value
            if v.tag == "Val/Val_Err" or v.tag == "Val/Val_Err1":
                data['ValErr'][0, e.step] = v.simple_value
            if v.tag == "Train/Train_Loss" or v.tag == "Train/Train_Loss":
                data['TrainLoss'][0, e.step] = v.simple_value
            if v.tag == "Val/Val_Loss" or v.tag == "Val/Val_Loss":
                data['ValLoss'][0, e.step] = v.simple_value
    scipy.io.savemat(f"sgd/data/sgd_{lr}_{m}_{bs}.mat", data)
