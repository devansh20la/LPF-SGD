""" Use scipy/ARPACK implicitly restarted lanczos to find top k eigenthings """
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import eigsh
from warnings import warn


def lanczos(operator,
            num_eigenthings=10,
            which='LM',
            max_steps=20,
            tol=1e-5,
            num_lanczos_vectors=None):
    """
    Use the scipy.sparse.linalg.eigsh hook to the ARPACK lanczos algorithm
    to find the top k eigenvalues/eigenvectors.

    Parameters
    -------------
    operator: power_iter.Operator
        linear operator to solve.
    num_eigenthings : int
        number of eigenvalue/eigenvector pairs to compute
    which : str ['LM', SM', 'LA', SA','BE']
        L,S = largest, smallest. M, A = in magnitude, algebriac
        SM = smallest in magnitude. LA = largest algebraic.
        BE = both ends.
    max_steps : int
        maximum number of arnoldi updates
    tol : float
        relative accuracy of eigenvalues / stopping criterion
    num_lanczos_vectors : int
        number of lanczos vectors to compute. If None, > 2*num_eigenthings

    Returns
    ----------------
    eigenvalues : np.ndarray
        array containing `num_eigenthings` eigenvalues of the operator
    eigenvectors : np.ndarray
        array containing `num_eigenthings` eigenvectors of the operator
    """
    size = operator.size if isinstance(operator.size, int) else operator.size[0]
    shape = (size, size)

    def _scipy_apply(x):
        x = torch.from_numpy(x)
        return operator.apply(x.float()).cpu().numpy()

    scipy_op = ScipyLinearOperator(shape, _scipy_apply)

    eigenvals, eigenvecs = eigsh(
        A=scipy_op,
        k=num_eigenthings,
        which=which,
        maxiter=max_steps,
        tol=tol,
        ncv=num_lanczos_vectors,
        return_eigenvectors=True)

    return eigenvals, eigenvecs.T
