import torch
from sacred.observers import FileStorageObserver
import random
import numpy as np

# https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/29
def to_onehot(idx, K):
    x = torch.zeros(len(idx), K, device=idx.device)
    return x.scatter_(1, idx.unsqueeze(1), 1.)

def num_params(model, learnable=False):
    if learnable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad==True)
    else:
        return sum(p.numel() for p in model.parameters())

def l2_norm(model, mean=False):
    if mean:
        return torch.sqrt(sum(p.norm()**2 for p in model.parameters()) / num_params(model)).item()
    else:
        return torch.sqrt(sum(p.norm()**2 for p in model.parameters())).item()


def run_and_config_to_path(_run, _config):
    path = f"{_run.experiment_info['name']}"
    exclude_list = ["logtime","gpu","nthreads","save_model","no_cuda","load_model"] 
    for k in _config:
        if k not in exclude_list:
            path += f"_{k}={_config[k]}"
    return path

def file_observer_dir(_run):
    for o in _run.observers:
        if isinstance(o, FileStorageObserver):
            return o.dir
    return None

def to_gpuid_string(x):
    if isinstance(x, list):
        return str(x)[1:-1]
    return str(x)


# Returns indexes of n examples for each class
def take_n_per_class(dataset, n):
    y = dataset.targets
    classes = y.unique()
    idxs = []
    print(f"Classes {len(classes)}")
    for c in classes:
        (idxs_c, ) = torch.where(y == c)
        idxs_c = np.random.permutation(idxs_c)[:n]
        # print(f"Class {c}: {len(idxs_c)}")
        idxs.extend(idxs_c.tolist())
    return idxs