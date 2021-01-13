import numpy as np
import torch
from flat_meas import _fro_norm, _eps_flatness, _pac_bayes, _entropy, _eig_trace, _low_pass, _fim
import time
from utils import AverageMeter
from tqdm import tqdm, trange


class func(object):
    """docstring for func"""
    def __init__(self, H):
        super(func, self).__init__()
        self.H = H
        self.dim = H.shape[0]

    def __call__(self, x):
        return x.T @ (self.H @ x) / 2

    def gradient(self, x):
        return self.H @ x

    def hessian(self):
        return self.H

    def hvp(self, x):
        return self.H @ x


def main():
    mtr = {x: AverageMeter() for x in ["fro_norm", "eig_spec", "eps_flat", "pac_bayes", "local_entropy", "low_pass", "fim"]}

    d = 100
    total_exps = 1
    resolution = 5
    for seed in range(total_exps):
        np.random.seed(seed)

        all_data = np.zeros((8, len(list(range(0, d+2, resolution))), total_exps))

        H = np.random.randn(d, d)
        H = H.T @ H / 2
        E, V = np.linalg.eig(H)

        idx = E.argsort()
        E = E[idx]
        V = V[:, idx]

        g = torch.from_numpy(V[:, -1]).reshape(-1, 1).type(torch.float32)
        theta_init = torch.randn((d, 1))
        noise = torch.randn((d, 1))*1e-10
        for i in trange(0, d+2, resolution):

            if i > 0:
                E[i-resolution:i] = (1e-3 - 1e-5) * np.random.random(size=resolution) + 1e-5

            H = V @ (np.diag(E) @ V.T)

            H = torch.from_numpy(H).type(torch.float32)
            f = func(H)

            theta_star, _ = torch.solve(torch.zeros((d, 1)), f.H)

            t = time.time()
            all_data[0, i//resolution, seed] = _eps_flatness(f, theta_star, g, 0.01, verbose=False)
            mtr["eps_flat"].update(time.time() - t, 1)

            t = time.time()
            all_data[1, i//resolution, seed] = _pac_bayes(f, theta_init, theta_star, 100, 1, 0.01, verbose=False)
            mtr["pac_bayes"].update(time.time() - t, 1)

            t = time.time()
            all_data[2, i//resolution, seed] = _fro_norm(f, 100)
            mtr["fro_norm"].update(time.time() - t, 1)

            t = time.time()
            all_data[3, i//resolution, seed] = _fim(f, theta_star+noise)
            mtr["fim"].update(time.time() - t, 1)

            t = time.time()
            all_data[4, i//resolution, seed] = _entropy(f, theta_star, 5e-4, 10000)
            mtr["local_entropy"].update(time.time() - t, 1)

            t = time.time()
            e = _eig_trace(f, 95, use_cuda=False).sum()
            all_data[5, i//resolution, seed]  = np.sum(e)
            mtr["eig_spec"].update(time.time() - t, 1)

            t = time.time()
            all_data[6, i//resolution, seed] = _low_pass(f, theta_star, mcmc_itr=100, sigma=1)
            mtr["low_pass"].update(time.time() - t, 1)

    for x in ["fro_norm", "eig_spec", "eps_flat", "pac_bayes", "local_entropy", "low_pass"]:
        print(f"{x} took: {mtr[x].avg}s")

    with open("results/invest_cvx_func_first_set.npy", 'wb') as f:
        np.save(f, np.mean(all_data, 2))


if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)

    main()
