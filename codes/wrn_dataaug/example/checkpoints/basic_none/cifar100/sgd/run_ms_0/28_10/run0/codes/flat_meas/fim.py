from tqdm import tqdm
import torch


def fim(model_func):
    r'''
        Fisher information matrix
        Inputs:
            func       : model functional class
        Outputs:
			frobenius norm
	'''
    out = 0.0
    theta_star = torch.cat([p.view(-1) for p in model_func.model.parameters()])

    return (theta_star.T @ model_func.hvp(theta_star)).item()


def _fim(model_func, theta_star):
    r'''
        Fisher information matrix
        Inputs:
            func       : model functional class
        Outputs:
            frobenius norm
    '''
    return (theta_star.T @ model_func.hvp(theta_star)).item()