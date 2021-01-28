# SacredDNN

Run deep neural network experiments with [pytorch](https://pytorch.org/) and [sacred](https://github.com/IDSIA/sacred).

## Installation

If you work within an anaconda/miniconda distribution, you can create a new virtual environment
and install all the required packages using the "conda_environment.yml" configuration file provided by this repo:

```bash
conda env create -f conda_environment.yml
```

## Usage

The `scripts/` folder contains script files for training neural netowrk  according to different (experimental or not) techniques. The entry point is the file "scripts/dnn.py", training NN for classification according to standard techniques.

The other scripts, implementing some non-standard algorithms, are built starting from the "dnn.py" template, therefore they typically accept the same command-line flags along with script-specific ones. See the section "Scripts" in this readme for a list of available scripts. In this section we focus on the usage of the "dnn.py" script.

Run the program with defaults flags with

```bash
python scripts/dnn.py
```

You can change the paremeters using the `with` syntax as in the following example:

```bash
python scripts/dnn.py with lr=1e-1 model=lenet dataset=fashion opt=sgd epochs=10
```

On multi-gpu machines, you can select a specific gpu or a subset of the available gpus (for multi-gpu training) with the `gpu` flag:

```bash
# default value
python scripts/dnn.py with gpu=0  

# train on gpus with ids 0,2 and 3
python scripts/dnn.py with gpu=0,2,3  
```

Data-parallel multi-gpu training should give a noticeable speedup for large models.

### Save results to MongoDB

Use the flag `-m some_collection` to save the results to a MongoDB collection

```bash
python scripts/dnn.py -m dnn with  model=resnet18 dataset=cifar10 preprocess=True
```

The mongodb server has to be installed and running to exploit this functionality.

[Omniboard](https://github.com/vivekratnavel/omniboard) provides a web dashboard that can be connected to a database containing sacred experiment
results (called `dnn` in this example). Once you install omniboard with

```bash
npm install -g omniboard
```

run the server with

```bash
omniboard -m localhost:27017:dnn
```

and browse to localhost:27017.

### Save results to folder + Tensorboard integration

As an alternative to MongoDB storage and to Omniboard visualization, you can
save results to a folder structure (using sacred's FileObserver) and at the same time activate tensorboard logging.
This can be done using the flag  `-F some_folder`:

```bash
python scripts/dnn.py -F runs/ with  model=resnet18 dataset=cifar10 preprocess=True
```

You can then display the results running tensorboard dashboard:

```bash
tensorboard --logdir=runs
```

You can also use both mongodb and the file observe at the same time,
but you should be carefull about conflicting id naming for your runs.

### Checkpoints

Setting the option `save_model=True`, you can save on disk your model
periodically during training. You need to have an observer (mongodb or the file observer)
attached to your experiment, e.g.

```bash
python scripts/dnn.py -F runs/ with save_model=True
```

## Robust Ensemble

Run a robust ensemble version of your DNN with

```bash
python scripts/robust_dnn.py with dataset=cifar10 model=densenet preprocess=True y=3
```

Available robust ensemble training specific options are

```python
y=3                 # number of replicas
use_center=False    # use a central replica
g=1e-3              # initial coupling value
grate=1e-1          # coupling increase rate
```

## Noisy DNN

Train the network in a "robust" fashion injecting  noisy
at each iteration on a scale controlled by a parameter `g`:

```bash
python scripts/noisy_dnn.py with g=1e-3
```
