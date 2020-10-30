from tqdm import tqdm
import torch


def fro_norm(model_func, mcmc_itr):
    r'''
        Frobenium norm approximation based on [put papaer here]
        Inputs:
            func       : model functional class
            mcmc_itr   : mcmc_itr
        Outputs:
			frobenius norm
	'''
    out = 0.0
    for i in tqdm(range(mcmc_itr)):
        v = torch.normal(mean=torch.zeros(model_func.dim), std=torch.ones(model_func.dim)).to(device=torch.device('cuda'))
        out += torch.norm(model_func.hvp(v), p=2)**2

    return (out/mcmc_itr).item()**0.5

def _fro_norm(model_func, mcmc_itr):
    r'''
        Frobenium norm approximation based on [put papaer here]
        Inputs:
            func       : model functional class
            mcmc_itr   : mcmc_itr
        Outputs:
            frobenius norm
    '''
    out = 0.0
    for i in tqdm(range(mcmc_itr)):
        v = torch.normal(mean=torch.zeros(model_func.dim), std=torch.ones(model_func.dim))
        out += torch.norm(model_func.hvp(v), p=2)**2

    return (out/mcmc_itr).item()**0.5
