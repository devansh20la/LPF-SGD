import numpy as np
from utils.spectral_utils import tridiag_to_eigv
from tqdm import tqdm
from hessian_eigenthings import deflated_power_iteration
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import eigsh
from warnings import warn
import torch


# def lanczos(func, dim, max_itr, use_cuda=False, verbose=False):
#     r'''
#         Lanczos iteration following the wikipedia article here
#             https://en.wikipedia.org/wiki/Lanczos_algorithm
#         Inputs:
#             func   : model functional class
#             dim    : dimensions
#             max_itr: max iteration
#             use_gpu: Use Gpu
#             verbose: print extra details
#         Outputs:
#             eigven values
#             weights
#     '''
#     float_dtype = torch.float64

#     # Initializing empty arrays for storing
#     tridiag = torch.zeros((max_itr, max_itr), dtype=float_dtype)
#     vecs = torch.zeros((dim, max_itr), dtype=float_dtype)

#     # intialize a random unit norm vector
#     init_vec = torch.zeros((dim, 1), dtype=float_dtype).uniform_(-1, 1)
#     init_vec /= torch.norm(init_vec)
#     vecs[:, 0:1] = init_vec

#     # placeholders for data
#     beta = 0.0
#     v_old = torch.zeros((dim, 1), dtype=float_dtype)

#     for k in range(max_itr):
#         t = time.time()

#         v = vecs[:, k:k+1]
#         if use_cuda:
#             v = v.type(torch.float32).cuda()
#         time_mvp = time.time()
#         w = func.hvp(v)
#         if use_cuda:
#             v = v.cpu().type(float_dtype)
#             w = w.cpu().type(float_dtype)
#         time_mvp = time.time() - time_mvp

#         w -= beta * v_old
#         alpha = w.T @ v
#         tridiag[k, k] = alpha
#         w -= alpha*v

#         # Reorthogonalization
#         for j in range(k):
#             tau = vecs[:, j:j+1]
#             coeff = w.T @ tau
#             w -= coeff * tau

#         beta = np.linalg.norm(w)

#         if beta < 1e-6:
#             print(beta)
#             raise ZeroDivisionError
#             quit()

#         vecs[:, k+1:k+2] = w / beta
#         if k < max_itr-1:
#             tridiag[k, k+1] = beta
#             tridiag[k+1, k] = beta

#         v_old = v

#         info = f"Iteration {k} / {max_itr} done in {time.time()-t:.2f}s (MVP: {time_mvp:.2f}s)"
#         if verbose:
#             print(info)

#     return vecs, tridiag


# def eig_trace(model_func, max_itr, draws=100, use_cuda=False, verbose=False):
#     tri = np.zeros((draws, max_itr, max_itr))
#     for num_draws in tqdm(range(draws)):
#         _, tridiag = lanczos(model_func, model_func.dim, max_itr, use_cuda, verbose)
#         tri[num_draws, :, :] = tridiag.numpy()

#     e, w = tridiag_to_eigv(tri)
#     e = np.mean(e, 0)
#     return e, e.sum()

def eig_trace(
    operator,
    num_eigenthings=10,
    which="LM",
    max_steps=20,
    tol=1e-6,
    num_lanczos_vectors=None,
    init_vec=None,
    use_cuda=False,
):
    """
    Use the scipy.sparse.linalg.eigsh hook to the ARPACK lanczos algorithm
    to find the top k eigenvalues/eigenvectors.

    Parameters
    -------------
    operator: power_iter.Operator
        linear operator to solve.
    num_eigenthings : int
        number of eigenvalue/eigenvector pairs to compute
    which : str ['LM', SM', 'LA', SA']
        L,S = largest, smallest. M, A = in magnitude, algebriac
        SM = smallest in magnitude. LA = largest algebraic.
    max_steps : int
        maximum number of arnoldi updates
    tol : float
        relative accuracy of eigenvalues / stopping criterion
    num_lanczos_vectors : int
        number of lanczos vectors to compute. if None, > 2*num_eigenthings
    init_vec: [torch.Tensor, torch.cuda.Tensor]
        if None, use random tensor. this is the init vec for arnoldi updates.
    use_cuda: bool
        if true, use cuda tensors.

    Returns
    ----------------
    eigenvalues : np.ndarray
        array containing `num_eigenthings` eigenvalues of the operator
    eigenvectors : np.ndarray
        array containing `num_eigenthings` eigenvectors of the operator
    """

    shape = (operator.dim, operator.dim)

    if num_lanczos_vectors is None:
        num_lanczos_vectors = min(2 * num_eigenthings, operator.dim - 1)
    if num_lanczos_vectors < 2 * num_eigenthings:
        warn(
            "[lanczos] number of lanczos vectors should usually be > 2*num_eigenthings"
        )

    def _scipy_apply(x):
        x = torch.from_numpy(x)
        if use_cuda:
            x = x.cuda()
        return operator.hvp(x).cpu().numpy()

    scipy_op = ScipyLinearOperator(shape, _scipy_apply)
    if init_vec is None:
        init_vec = np.random.rand(operator.dim)
    elif isinstance(init_vec, torch.Tensor):
        init_vec = init_vec.cpu().numpy()
    eigenvals = eigsh(
        A=scipy_op,
        k=num_eigenthings,
        which=which,
        maxiter=max_steps,
        tol=tol,
        ncv=num_lanczos_vectors,
        return_eigenvectors=False,
    )
    return eigenvals


def _eig_trace(
    operator,
    num_eigenthings=10,
    which="LM",
    max_steps=20,
    tol=1e-6,
    num_lanczos_vectors=None,
    init_vec=None,
    use_cuda=False,
):
    """
    Use the scipy.sparse.linalg.eigsh hook to the ARPACK lanczos algorithm
    to find the top k eigenvalues/eigenvectors.

    Parameters
    -------------
    operator: power_iter.Operator
        linear operator to solve.
    num_eigenthings : int
        number of eigenvalue/eigenvector pairs to compute
    which : str ['LM', SM', 'LA', SA']
        L,S = largest, smallest. M, A = in magnitude, algebriac
        SM = smallest in magnitude. LA = largest algebraic.
    max_steps : int
        maximum number of arnoldi updates
    tol : float
        relative accuracy of eigenvalues / stopping criterion
    num_lanczos_vectors : int
        number of lanczos vectors to compute. if None, > 2*num_eigenthings
    init_vec: [torch.Tensor, torch.cuda.Tensor]
        if None, use random tensor. this is the init vec for arnoldi updates.
    use_cuda: bool
        if true, use cuda tensors.

    Returns
    ----------------
    eigenvalues : np.ndarray
        array containing `num_eigenthings` eigenvalues of the operator
    eigenvectors : np.ndarray
        array containing `num_eigenthings` eigenvectors of the operator
    """

    shape = (operator.dim, operator.dim)

    if num_lanczos_vectors is None:
        num_lanczos_vectors = min(2 * num_eigenthings, operator.dim - 1)
    if num_lanczos_vectors < 2 * num_eigenthings:
        warn(
            "[lanczos] number of lanczos vectors should usually be > 2*num_eigenthings"
        )

    def _scipy_apply(x):
        return operator.hvp(x)

    scipy_op = ScipyLinearOperator(shape, _scipy_apply)
    if init_vec is None:
        init_vec = np.random.rand(operator.dim)
    elif isinstance(init_vec, torch.Tensor):
        init_vec = init_vec.cpu().numpy()
    eigenvals = eigsh(
        A=scipy_op,
        k=num_eigenthings,
        which=which,
        maxiter=max_steps,
        tol=tol,
        ncv=num_lanczos_vectors,
        return_eigenvectors=False,
    )
    return eigenvals
