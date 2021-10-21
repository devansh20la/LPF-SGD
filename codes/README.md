# Low-Pass Filtering SGD for Recovering Flat Optima in the Deep Learning Optimization Landscape

This repository is the official implementation of Low-Pass Filtering SGD for Recovering Flat Optima in the Deep Learning Optimization Landscape. 


## Requirements

Let us first install all the requirements for the code repository. We provide a conda requirements file which can be installed as follows:

```setup
conda create --name <env> --file requirements.txt
```


## Sharpness vs generalization:

All the training codes and computing sharpness measures lie in the folder sharpness_v_generalization:

```train
./run.sh
```

This shell script will perform the following operations:
1. Download all relevant datasets.
2. Train 2916 resnet18 models on cifar10 dataset with varying hyper-parameters and 3 random seeds.
3. Train 50 different models with 10 different level of label noise and 5 random seeds.
4. Train 100 different models with 20 different level of data noise and 5 random seeds.
5. Train 64 different models with 64 different width levels and 5 random seeds.
6. Compute sharpness measures for all the models and store it as a numpy array in .pkl file 


## Experiments 1
To re-create the results in Table 3 from the paper, change to resnets_dataaug directory 'cd resnets_dataaug'. Here are the arguments common to all training python scripts:

```
  --dir DIR 			Data directory
  --print_freq 			Print frequency
  --dtype DTYPE         data type [cifar10, cifar100]
  --ep EP               Epochs
  --mtype mtype 		model type [resnet18, resnet50, resnet101]
  --ms MS               manuel seed
  --mo MO               momentum
  --wd WD               weight decay
  --lr LR               learning rate
  --bs BS               batch size
```

### Training
We can train SGD, SAM and lpfsgd models on cifar10 data with resnet18 model as follow:

```
python sgd_train.py \
	--ep=200 \
	--bs=128 \
	--dtype='cifar10' \
	--mtype='resnet18' \
	--print_freq=100 \
	--mo=0.9 \
	--lr=0.1 \
	--ms=0 \
	--wd=5e-4
```

We can train SAM models as:

```
python sam_train.py \
	--ep=200 \
	--bs=128 \
	--dtype='cifar10' \
	--mtype='resnet18' \
	--print_freq=100 \
	--mo=0.9 \
	--lr=0.1 \
	--ms=0 \
	--wd=5e-4
```

We can train lpfsgd models as:


```
python lpf_train.py \
	--ep=200 \
	--bs=128 \
	--dtype='cifar10' \
	--mtype='resnet18' \
	--print_freq=100 \
	--mo=0.9 \
	--lr=0.1 \
	--ms=0 \
	--wd=5e-4 \
	--std=0.0002
```

### Hyper-parameters

<div id="tab:exp1_1">
Training hyper-parameters common to all optimizers:
momentum.

|   Dataset    |       Model        | BS  |         WD          | MO  | Epochs |           LR (Policy)            |
|:------------:|:------------------:|:---:|:-------------------:|:---:|:------:|:--------------------------------:|
|    MNIST     |       LeNet        | 128 | 5*e*<sup> − 4</sup> | 0.9 |  150   |  0.01 (x 0.1 at ep=\[50,100\])   |
| CIFAR10, 100 | ResNet-18, 50, 101 | 128 | 5*e*<sup> − 4</sup> | 0.9 |  200   |  0.1 (x 0.1 at ep=\[100,120\])   |
| TinyImageNet |     ResNet-18      | 128 | 1*e*<sup> − 4</sup> | 0.9 |  100   | 0.1 (x 0.1 at ep=\[30, 60, 90\]) |
|   ImageNet   |     ResNet-18      | 256 | 1*e*<sup> − 4</sup> | 0.9 |  100   | 0.1 (x 0.1 at ep=\[30, 60, 90\]) |


</div>

<div id="tab:exp1_2">
Summary of SAM hyper-parameters:

|   Dataset    |      Model      | *ρ* (policy) |
|:------------:|:---------------:|:------------:|
|    MNIST     |      LeNet      | 0.05 (fixed) |
|    CIFAR     | ResNet18,50,101 | 0.05 (fixed) |
| TinyImageNet |    ResNet18     | 0.05 (fixed) |
|   ImageNet   |    ResNet18     | 0.05 (fixed) |


</div>

<div id="tab:exp1_4">

Summary of LPF-SGD hyper-parameters:

|   Dataset    |      Model      |  M  |  *γ* (policy)  |
|:------------:|:---------------:|:---:|:--------------:|
|    MNIST     |      LeNet      |  1  |  0.001(fixed)  |
|    CIFAR     | ResNet18,50,101 |  1  | 0.002 (fixed)  |
| TinyImageNet |     ResNet      |  1  | 0.001 (fixed)  |
|   ImageNet   |     ResNet      |  1  | 0.0005 (fixed) |


</div>


## Experiments 2
To re-create the results in Table 5 from the paper, change to wrn_dataaug directory 'cd wrn_dataaug/example'. Here are the arguments common to all training scripts:

```
	--mtype 		        Model Type ["wrn", "shakeshake", "pyramidnet"] 
	--depth 		        Number of layers. 
	--width_factor 		  	width factor  
	--epochs 		       	Total number of epochs.
	--learning_rate 		base learning rate
	--momentum 				momentum
	--weight_decay 			weight decay
	--batch_size 			batch size
	--seed 	           		seed
	--dtype 	         	dtype
	--img_aug 			    augmentation scheme [basic_none, basic_cutout(basic+cut), autoaugment(basic+aa+cut)]
	--threads THREADS     	Number of CPU threads for dataloaders.
```

### Training
Here is an example to train shakeshake model with basic augmentation scheme and sgd optimizer:
```
python sgd_train.py \
	--mtype "shakeshake" \
	--dtype "cifar10" \
	--learning_rate 0.2 \
	--weight_decay 0.0001 \
	--epochs 1800 \
	--depth 26 \
	--width 96 \
	--seed 0 \
	--img_aug "basic_none" \
	--threads 8
```

Here is an example to train shakeshake model with basic augmentation scheme and SAM optimizer (default rho is set to 0.05):

```
python sam_train.py \
	--mtype "shakeshake" \
	--dtype "cifar10" \
	--learning_rate 0.2 \
	--weight_decay 0.0001 \
	--epochs 1800 \
	--depth 26 \
	--width 96 \
	--seed 0 \
	--img_aug "basic_none" \
	--threads 8
```

Here is an example to train shakeshake model with basic augmentation scheme and LPF-SGD optimizer (default std is set to 0.0005):
```
python ssgd_train.py \
	--mtype "shakeshake" \
	--dtype 'cifar10'
	--learning_rate 0.2 \
	--weight_decay 0.0001 \
	--epochs 1800 \
	--depth 26 \
	--width 96 \
	--seed 0 \
	--img_aug 'basic_none' \
	--std 0.0005 \
	--inc 15 \
	--M 8
```


### Hyper-parameters
<div id="tab:exp2_1">
Training hyper-parameters common to all optimizers utilized for
Table 5 in the paper. Here BS: batch size, WD: weight decay, MO: SGD
momentum.


|                       |     |                     |     |        |                              |                              |
|:----------------------|:---:|:-------------------:|:---:|:------:|:----------------------------:|:-------------------------:   |
| Model                 | BS  |         WD          | MO  | Epochs |                          LR(Policy)                         |
|                       |     |                     |     |        |           CIFAR-10           |         CIFAR-100            |
| WRN16-8               | 128 | 5*e*<sup> − 4</sup> | 0.9 |  200   | 0.1(× 0.2 at \[60,120,160\]) | 0.1(× 0.2 at \[60,120,160\]) |
| WRN28-10              | 128 | 5*e*<sup> − 4</sup> | 0.9 |  200   | 0.1(× 0.2 at \[60,120,160\]) | 0.1(× 0.2 at \[60,120,160\]) |
| ShakeShake (26 2x96d) | 128 | 1*e*<sup> − 4</sup> | 0.9 |  1800  |     0.2(cosine decrease)     | 0.2(cosine decrease)         |
| PyNet110              | 128 | 1*e*<sup> − 4</sup> | 0.9 |  200   |  0.1(× 0.1 at \[100,150\])   | 0.5(× 0.1 at \[100,150\])    |
| PyNet272              | 128 | 1*e*<sup> − 4</sup> | 0.9 |  200   |  0.1(× 0.1 at \[100,150\])   | 0.1(× 0.1 at \[100,150\])    |

</div>

<div id="tab:exp2_2">
Radius hyper-parameter for LPF-SGD optimizer. Here \* refers to all
kinds of augmentation schemes.


|                     |              |     |                 |              |                 |              |
|:-------------------:|:------------:|:---:|:---------------:|:------------:|:---------------:|:------------:|
|        Model        |     Aug      |  M  |             CIFAR-10           ||              CIFAR-100        ||
|                     |              |     | *γ*<sub>0</sub> | *α* (policy) | *γ*<sub>0</sub> | *α* (policy) |
|       WRN16-8       |      \*      |  8  |     0.0005      |      15      |     0.0005      |      15      |
|      WRN28-10       |    Basic     |  8  |     0.0005      |      35      |     0.0005      |      25      |
|                     |  Basic+Cut   |  8  |     0.0005      |      35      |     0.0005      |      25      |
|                     | Basic+Cut+AA |  8  |     0.0005      |      35      |     0.0007      |      15      |
| ShakeShake 26 2x96d |      \*      |  8  |     0.0005      |      15      |     0.0005      |      15      |
|      PyNet110       |      \*      |  8  |     0.0005      |      15      |     0.0005      |      15      |
|      PyNet272       |      \*      |  8  |     0.0005      |      15      |     0.0005      |      15      |


</div>

## Machine Translation
To re-create the results in Table 7 from the paper, change to exp1 directory 'cd machine_translation'. Here are the arguments common to all training scripts:

```
--ts 		  Total number of steps.
--lr 			base learning rate
--wd 			weight decay
--bs 			batch size
--seed		seed
--dp			dropout probability
```

### Training
Here is an example to train tranformer model with adam optimizer:
```
python _adamtrain.py \
	--wd 0.0001 \
	--lr 0.0005 \
	--bs 256 \
	--ts 100000 \
	--dp 0.1
```

### Hyper-parameters
<div id="tab:exp_mt1">
Training hyper-parameters common to all optimizers utilized for
Table 7 in the paper. Here BS: batch size, WD: weight decay.

| Dataset |   Model    | BS  |        WD         | (*β*<sub>1</sub>,*β*<sub>2</sub>) | Itr  |           LR (Policy)            |
|:-------:|:----------:|:---:|:-----------------:|:---------------------------------:|:----:|:--------------------------------:|
| WMT2014 | Mini-Trans | 256 | 1*e*<sup>−4</sup> |            (0.9,0.99)             | 100k | 0.0005 (× 0.1 ReduceLROnPlateau) |

</div>



<div id="tab:exp_mt2">
Summary of E-SGD, ASO, SAM and LPF-SGD hyper-parameters used for obtaining Table 7.


| Dataset |   Model    |                   E-Adam                   | ASO-Adam | SAM  |    LPF-Adam     |
|:-------:|:----------:|:------------------------------------------:|:--------:|:----:|:---------------:|
|         |            | (*γ*<sub>0</sub>, *γ*<sub>1</sub>, *η*, M) |    a     | *ρ*  | (M, *γ*, *α*))  |
| WMT2014 | Mini-Trans |           (0.5, 0.0001, 0.1, 5)            |  0.009   | 0.01 | (8, 0.0005, 15) |

</div>

## Adversarial
In order to compute adversarial robustness of the model, first train the required model and move the model file in ```adversarial/checkpooints/```.
You can now run `main.py` with same parameters as training but an additional parameter ```model_path``` that loads the saved model.
