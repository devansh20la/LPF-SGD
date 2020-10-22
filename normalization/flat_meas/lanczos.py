import torch
import numpy as np
import time
from utils.spectral_utils import tridiag_to_eigv


def lanczos(func, dim, max_itr, use_gpu=False, verbose=False):
    '''
        Lanczos iteration following the wikipedia article here
            https://en.wikipedia.org/wiki/Lanczos_algorithm
        Inputs:
            batch_loss: f(x)
            model: input model
            d: hessian dimension
            max_it: max iteration
        Outputs:
            eigven values
            weights
    '''
    float_dtype = torch.float64

    # Initializing empty arrays for storing
    tridiag = torch.zeros((max_itr, max_itr), dtype=float_dtype)
    vecs = torch.zeros((dim, max_itr), dtype=float_dtype)

    init_vec = torch.zeros((dim, 1), dtype=float_dtype).uniform_(-1, 1)
    init_vec /= torch.norm(init_vec)
    vecs[:, 0:1] = init_vec

    beta = 0.0
    v_old = torch.zeros((dim, 1), dtype=float_dtype)

    for k in range(max_itr):
        t = time.time()
        v = vecs[:, k:k+1]

        if use_gpu:
            v = v.type(torch.float32).cuda()

        time_mvp = time.time()
        w = func.hvp(v)

        if use_gpu:
            v = v.cpu()

        v = v.type(float_dtype)
        time_mvp = time.time() - time_mvp

        w -= beta * v_old
        alpha = w.T @ v
        tridiag[k, k] = alpha
        w -= alpha*v

        # Reorthogonalization
        for j in range(k):
            tau = vecs[:, j:j+1]
            coeff = w.T @ tau
            w -= coeff * tau

        beta = np.linalg.norm(w)

        if beta < 1e-6:
            print(beta)
            raise ZeroDivisionError
            quit()

        vecs[:, k+1:k+2] = w / beta
        if k < max_itr-1:
            tridiag[k, k+1] = beta
            tridiag[k+1, k] = beta

        v_old = v

        info = f"Iteration {k} / {max_itr} done in {time.time()-t:.2f}s (MVP: {time_mvp:.2f}s)"
        if verbose:
            print(info)

    return vecs, tridiag


def eig_trace(func, dim, max_itr, draws=100, use_gpu=False, verbose=False):
    tri = np.zeros((draws, max_itr, max_itr))
    for num_draws in range(draws):
        _, tridiag = lanczos(func, dim, max_itr, use_gpu, verbose)
        tri[num_draws, :, :] = tridiag.numpy()

    e, w = tridiag_to_eigv(tri)

    return e.sum()
