from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from tqdm import tqdm
import math
import numpy as np


def get_hessian(batch_loss, model):
  grads = torch.autograd.grad(batch_loss, model.parameters(), create_graph=True)
  grads = torch.cat([x.view(-1) for x in grads])
  hessian = torch.zeros((grads.shape[0], grads.shape[0]))
  i = 0
  for grad_t in tqdm(grads):
    for g in grad_t.reshape(-1):
      hess = torch.autograd.grad(g, model.parameters(), retain_graph=True)
      hess = torch.cat([h.reshape(-1) for h in hess])
      hessian[i, :] = hess
      i += 1
  return hessian


def _check_param_device(param, old_param_device):
  r"""This helper function is to check if the parameters are located
  in the same device. Currently, the conversion between model parameters
  and single vector form is not supported for multiple allocations,
  e.g. parameters in different GPUs, or mixture of CPU/GPU.

  Arguments:
      param ([Tensor]): a Tensor of a parameter of a model
      old_param_device (int): the device where the first parameter of a
                              model is allocated.

  Returns:
      old_param_device (int): report device for the first time
  """

  # Meet the first parameter
  if old_param_device is None:
    old_param_device = param.get_device() if param.is_cuda else -1
  else:
    warn = False
    if param.is_cuda:  # Check if in same GPU
      warn = (param.get_device() != old_param_device)
    else:  # Check if in CPU
      warn = (old_param_device != -1)
    if warn:
      raise TypeError('Found two parameters on different devices, '
                        'this is currently not supported.')
  return old_param_device


def Rop(ys, xs, vs):
  if torch.cuda.is_available():
    ws = [torch.zeros(y.size(), requires_grad=True).cuda() for y in ys]
  else:
    ws = [torch.zeros(y.size(), requires_grad=True) for y in ys]

  gs = torch.autograd.grad(ys, xs, grad_outputs=ws, create_graph=True, retain_graph=True, allow_unused=True)

  re = torch.autograd.grad(gs, ws, grad_outputs=vs, create_graph=False, allow_unused=True)
  re = (r.contiguous() for r in re)

  return tuple([j.detach() for j in re])


def vector_to_parameter_tuple(vec, parameters):
  r"""Convert one vector to the parameter list

  Arguments:
      vec (Tensor): a single vector represents the parameters of a model.
      parameters (Iterable[Tensor]): an iterator of Tensors that are the
          parameters of a model.
  """
  # Ensure vec of type Tensor
  if not isinstance(vec, torch.Tensor):
    raise TypeError('expected torch.Tensor, but got: {}'
                    .format(torch.typename(vec)))
  params_new = []
  # Pointer for slicing the vector for each parameter
  pointer = 0
  for param in parameters:
    # The length of the parameter
    num_param = param.numel()
    # Slice the vector, reshape it, and replace the old data of the parameter
    param_new = vec[pointer:pointer + num_param].view_as(param).data.to(param.device)
    params_new.append(param_new)
    # Increment the pointer
    pointer += num_param

  return tuple(params_new)


def Lop(ys, xs, ws):
  vJ = torch.autograd.grad(ys, xs, grad_outputs=ws, create_graph=True, retain_graph=True, allow_unused=True)
  return tuple([j.detach() for j in vJ])


def HesssianVectorProduct(f, x, v):
  df_dx = torch.autograd.grad(f, x, create_graph=True, retain_graph=True)
  Hv = Rop(df_dx, x, v)
  return tuple([j.detach() for j in Hv])


def FisherVectorProduct(loss, output, model, vp):
  Jv = Rop(output, list(model.parameters()), vp)
  batch, dims = output.size(0), output.size(1)
  if loss.grad_fn.__class__.__name__ == 'NllLossBackward':
    outputsoftmax = torch.nn.functional.softmax(output, dim=1)
    M = torch.zeros(batch, dims, dims).cuda() if outputsoftmax.is_cuda else torch.zeros(batch, dims, dims)
    M.reshape(batch, -1)[:, ::dims + 1] = outputsoftmax
    H = M - torch.einsum('bi,bj->bij', (outputsoftmax, outputsoftmax))
    HJv = [torch.squeeze(H @ torch.unsqueeze(Jv[0], -1)) / batch]
  else:
    HJv = HesssianVectorProduct(loss, output, Jv)
  JHJv = Lop(output, list(model.parameters()), HJv)

  return torch.cat([torch.flatten(v) for v in JHJv])

def eigv_to_density(eig_vals, all_weights=None, grids=None,
                    grid_len=10000, sigma_squared=None, grid_expand=1e-2):
  """Compute the smoothed spectral density from a set of eigenvalues.
  Convolves the given eigenvalues with a Gaussian kernel, weighting the values
  by all_weights (or uniform weighting if all_weights is None). Example output
  can be seen in Figure 1 of https://arxiv.org/pdf/1901.10159.pdf. Visualizing
  the estimated density can be done by calling plt.plot(grids, density). There
  is likely not a best value of sigma_squared that works for all use cases,
  so it is recommended to try multiple values in the range [1e-5,1e-1].
  Args:
    eig_vals: Array of shape [num_draws, order]
    all_weights: Array of shape [num_draws, order], if None then weights will be
      taken to be uniform.
    grids: Array of shape [grid_len], the smoothed spectrum will be plotted
      in the interval [grids[0], grids[-1]]. If None then grids will be
      computed based on max and min eigenvalues and grid length.
    grid_len: Integer specifying number of grid cells to use, only used if
      grids is None
    sigma_squared: Scalar. Controls the smoothing of the spectrum estimate.
      If None, an appropriate value is inferred.
    grid_expand: Controls the window of values that grids spans.
      grids[0] = smallest eigenvalue - grid_expand.
      grids[-1] = largest_eigenvalue + grid_expand.
  Returns:
    density: Array of shape [grid_len], the estimated density, averaged over
      all draws.
    grids: Array of shape [grid_len]. The values the density is estimated on.
  """
  if all_weights is None:
    all_weights = np.ones(eig_vals.shape) * 1.0 / float(eig_vals.shape[1])
  num_draws = eig_vals.shape[0]

  lambda_max = np.nanmean(np.max(eig_vals, axis=1), axis=0) + grid_expand
  lambda_min = np.nanmean(np.min(eig_vals, axis=1), axis=0) - grid_expand

  if grids is None:
    assert grid_len is not None, 'grid_len is required if grids is None.'
    grids = np.linspace(lambda_min, lambda_max, num=grid_len)

  grid_len = grids.shape[0]
  if sigma_squared is None:
    sigma = 10 ** -5 * max(1, (lambda_max - lambda_min))
  else:
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

  density_each_draw = np.zeros((num_draws, grid_len))
  for i in range(num_draws):

    if np.isnan(eig_vals[i, 0]):
      raise ValueError('tridaig has nan values.')
    else:
      for j in range(grid_len):
        x = grids[j]
        vals = _kernel(eig_vals[i, :], x, sigma)
        density_each_draw[i, j] = np.sum(vals * all_weights[i, :])
  density = np.nanmean(density_each_draw, axis=0)
  norm_fact = np.sum(density) * (grids[1] - grids[0])
  density = density / norm_fact
  return density, grids


def tridiag_to_eigv(tridiag_list):
  """Preprocess the tridiagonal matrices for density estimation.
  Args:
    tridiag_list: Array of shape [num_draws, order, order] List of the
      tridiagonal matrices computed from running num_draws independent runs
      of lanczos. The output of this function can be fed directly into
      eigv_to_density.
  Returns:
    eig_vals: Array of shape [num_draws, order]. The eigenvalues of the
      tridiagonal matricies.
    all_weights: Array of shape [num_draws, order]. The weights associated with
      each eigenvalue. These weights are to be used in the kernel density
      estimate.
  """
  # Calculating the node / weights from Jacobi matrices.
  num_draws = len(tridiag_list)
  num_lanczos = tridiag_list[0].shape[0]
  eig_vals = np.zeros((num_draws, num_lanczos))
  all_weights = np.zeros((num_draws, num_lanczos))
  for i in range(num_draws):
    nodes, evecs = np.linalg.eigh(tridiag_list[i])
    index = np.argsort(nodes)
    nodes = nodes[index]
    evecs = evecs[:, index]
    eig_vals[i, :] = nodes
    all_weights[i, :] = evecs[0] ** 2
  return eig_vals, all_weights


def tridiag_to_density(tridiag_list, sigma_squared=1e-5, grid_len=10000):
  """This function estimates the smoothed density from the output of lanczos.
  Args:
    tridiag_list: Array of shape [num_draws, order, order] List of the
      tridiagonal matrices computed from running num_draws independent runs
      of lanczos.
    sigma_squared: Controls the smoothing of the density.
    grid_len: Controls the granularity of the density.
  Returns:
    density: Array of size [grid_len]. The smoothed density estimate averaged
      over all num_draws.
    grids: Array of size [grid_len]. The values the density estimate is on.
  """
  eig_vals, all_weights = tridiag_to_eigv(tridiag_list)
  density, grids = eigv_to_density(eig_vals, all_weights,
                                   grid_len=grid_len,
                                   sigma_squared=sigma_squared)
  return density, grids


def _kernel(x, x0, variance):
  """Point estimate of the Gaussian kernel.
  This function computes the Gaussian kernel for
  C exp(-(x - x0) ^2 /(2 * variance)) where C is the appropriate normalization.
  variance should be a list of length 1. Either x0 or x should be a scalar. Only
  one of the x or x0 can be a numpy array.
  Args:
    x: Can be either scalar or array of shape [order]. Points to estimate
      the kernel on.
    x0: Scalar. Mean of the kernel.
    variance: Scalar. Variance of the kernel.
  Returns:
    point_estimate: A scalar corresponding to
      C exp(-(x - x0) ^2 /(2 * variance)).
  """
  coeff = 1.0 / np.sqrt(2 * math.pi * variance)
  val = -(x0 - x) ** 2
  val = val / (2.0 * variance)
  val = np.exp(val)
  point_estimate = coeff * val
  return point_estimate