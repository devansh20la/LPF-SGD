# Low-Pass Filtering SGD for Recovering Flat Optima in the Deep Learning Optimization Landscape

This repository is the official implementation of Low-Pass Filtering SGD for Recovering Flat Optima in the Deep Learning Optimization Landscape. 


## Requirements

To install requirements:

```setup
create --name <env> --file requirement.txt
```


## Sharpness vs generalization:

All the training codes and computing sharpness measures lie in the folder sharpness_v_generalization:

```train
./run.sh
```

This shell script will perform the following operations:
1. Download all relevant datasets.
2. Training 2916 resnet18 models on cifar10 dataset with varying hyper-parameters and 3 random seeds.
3. Training 50 different models with 10 different level of label noise and 5 random seeds.
4. Training 100 different models with 20 different level of data noise and 5 random seeds.
5. Training 64 different models with 64 different width levels and 5 random seeds.
6. Compute sharpness measures for all the models and store it as a numpy array in .pkl file 


## Experiments 1

## Experiments 2


