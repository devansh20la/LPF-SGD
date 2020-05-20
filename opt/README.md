# Adaptive Momentum Stochastic gradient descent.

## Prerequisites
You need to install
```
pytorch
torchvision
PIL
sklearn
scikit-learn
tensorboard (for logging metrics)
tqdm (for pretty progress bars)
```

## Training
The following arguemnts are required to train lenet model on mnist data.

```
--dir: directory
--print_freq: frequency to print statistics

--dtype: data type
  supported arguments:
  "mnist"

--ms: manual seed
--ep: total number of epochs
--bs: batch size

-opt: optimizer
  supported arguments are:
  "amsgd"
  "sgd"
-lr: learning rate
-wd: weight decay
-m: momentum
```
Example to train lenet model on mnist data with AMSGD optimizer:
```
CUDA_VISIBLE_DEVICES=0 python train.py \
  -lr=0.01  \
  -wd=0.0  \
  -m=0.9  \
  --bs=128  \
  --print_freq=100  \
  --dtype="mnist" \
  --ep=200 \
  -opt="amsgd" &
```


## Authors
Devansh Bisla
