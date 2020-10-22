from tqdm import tqdm
import torch


def fro_norm(func, dim, mcmc_itr):
    out = 0.0
    for i in tqdm(range(mcmc_itr)):
        v = torch.normal(mean=torch.zeros(dim), std=torch.ones(dim))
        out += torch.norm(func.hvp(v), p=2)**2

    return (out/mcmc_itr).item()**0.5

