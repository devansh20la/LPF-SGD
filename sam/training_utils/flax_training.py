import functools
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as onp

from absl import flags
from absl import logging
import flax
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import lr_schedule
import jax
import jax.numpy as jnp
import numpy as np
from datasets import dataset_source as dataset_source_lib
from efficientnet import optim as efficientnet_optim
import tensorflow as tf
from tensorflow.io import gfile
import copy

FLAGS = flags.FLAGS


# Training hyper-parameters
flags.DEFINE_float('gradient_clipping', 5.0, 'Gradient clipping.')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_bool('use_learning_rate_schedule', True,
                  'Whether to use a cosine schedule or keep the learning rate '
                  'constant. Training on cifar should always use the schedule '
                  ', this flag is mostly for testing purpose.')
flags.DEFINE_bool('use_std_schedule', True,
                  'Whether to use a cosine schedule or keep the std rate '
                  'constant.')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay coefficient.')
flags.DEFINE_integer('run_seed', 0,
                     'Seed to use to generate pseudo random number during '
                     'training (for dropout for instance). Has no influence on '
                     'the dataset shuffling.')
flags.DEFINE_bool('use_rmsprop', False, 'If True, uses RMSprop instead of SGD')
flags.DEFINE_enum('lr_schedule', 'cosine', ['cosine', 'exponential', 'multistep'],
                  'Learning rate schedule to use.')
flags.DEFINE_enum('std_schedule', 'cosine', ['cosine', 'exponential'],
                  'std schedule to use.')

# Additional flags that don't affect the model.
flags.DEFINE_string('load_checkpoint', None, 'Load checkpoint')

flags.DEFINE_integer('save_progress_seconds', 3600, 'Save progress every...s')
flags.DEFINE_multi_integer(
    'additional_checkpoints_at_epochs', [10, 20, 50, 75, 100, 150],
    'Additional epochs when we should save the model for later analysis. '
    'No matter the value of this flag, the most recent version of the model '
    'will be saved regularly to resume training if needed.')
flags.DEFINE_bool('also_eval_on_training_set', False,
                  'If set to true, the model will also be evaluated on the '
                  '(non-augmented) training set at the end of each epoch.')
flags.DEFINE_bool('compute_top_5_error_rate', False,
                  'If true, will also compute top 5 error rate.')
flags.DEFINE_float('label_smoothing', 0.0, 'Label smoothing for cross entropy.')
flags.DEFINE_float('ema_decay', 0.0, 'If not zero, use EMA on all weights.')
flags.DEFINE_bool('no_weight_decay_on_bn', False,
                  'If set to True, will not apply weight decay on the batch '
                  'norm parameters.')
flags.DEFINE_integer('evaluate_every', 1,
                     'Evaluate on the test set every n epochs.')

# SSGD related flags:
flags.DEFINE_float('ssgd_std', 0.001, 'std for ssgd')
flags.DEFINE_integer('std_inc', 9, 'std inc')
flags.DEFINE_integer('M', 8, 'M')


# SAM related flags.
flags.DEFINE_float('sam_rho', -1,
                   'Size of the neighborhood considered for the SAM '
                   'perturbation. If set to zero, SAM will not be used.')
flags.DEFINE_bool('sync_perturbations', True,
                  'If set to True, sync the adversarial perturbation between '
                  'replicas.')
flags.DEFINE_integer('inner_group_size', None,
                     'Inner group size for syncing the adversarial gradients.'
                     'If None, we sync the adversarial perturbation across all '
                     'replicas. Else, we sync the perturbations inside groups '
                     'of inner_group_size replicas. Has no effect if '
                     'sync_perturbations is set to False.')

def restore_checkpoint(optimizer: flax.optim.Optimizer, 
  model_state: Any, directory: str) -> Tuple[flax.optim.Optimizer, flax.nn.Collection, int]:
  """Restores a model and its state from a given checkpoint.

  If several checkpoints are saved in the checkpoint directory, the latest one
  will be loaded (based on the `epoch`).

  Args:
    optimizer: The optimizer containing the model that we are training.
    model_state: Current state associated with the model.
    directory: Directory where the checkpoints should be saved.

  Returns:
    The restored optimizer and model state, along with the number of epochs the
      model was trained for.
  """
  train_state = dict(optimizer=optimizer, model_state=model_state, epoch=0)
  restored_state = checkpoints.restore_checkpoint(directory, train_state)
  return (restored_state['optimizer'],
          restored_state['model_state'],
          restored_state['epoch'])

def save_checkpoint(optimizer: flax.optim.Optimizer,
  model_state: Any,directory: str,epoch: int):
  """Saves a model and its state.

  Removes a checkpoint if it already exists for a given epoch. For multi-host
  training, only the first host will save the checkpoint.

  Args:
    optimizer: The optimizer containing the model that we are training.
    model_state: Current state associated with the model.
    directory: Directory where the checkpoints should be saved.
    epoch: Number of epochs the model has been trained for.
  """
  train_state = dict(optimizer=optimizer,
                     model_state=model_state,
                     epoch=epoch)
  if gfile.exists(os.path.join(directory, 'checkpoint_' + str(epoch))):
    gfile.remove(os.path.join(directory, 'checkpoint_' + str(epoch)))
  checkpoints.save_checkpoint(directory, train_state, epoch, keep=2)

def create_optimizer(model: flax.nn.Model, 
  learning_rate: float, beta: float = 0.9) -> flax.optim.Optimizer:
  """Creates an optimizer.

  Learning rate will be ignored when using a learning rate schedule.

  Args:
    model: The FLAX model to optimize.
    learning_rate: Learning rate for the gradient descent.
    beta: Momentum parameter.

  Returns:
    A SGD (or RMSProp) optimizer that targets the model.
  """
  if FLAGS.use_rmsprop:
    # We set beta2 and epsilon to the values used in the efficientnet paper.
    optimizer_def = efficientnet_optim.RMSProp(
        learning_rate=learning_rate, beta=beta, beta2=0.9, eps=0.001)
  else:
    optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta, nesterov=True)
  optimizer = optimizer_def.create(model)
  return optimizer

def cross_entropy_loss(logits: jnp.ndarray, 
  one_hot_labels: jnp.ndarray) -> jnp.ndarray:
  """Returns the cross entropy loss between some logits and some labels.

  Args:
    logits: Output of the model.
    one_hot_labels: One-hot encoded labels. Dimensions should match the logits.

  Returns:
    The cross entropy, averaged over the first dimension (samples).
  """
  if FLAGS.label_smoothing > 0:
    smoothing = jnp.ones_like(one_hot_labels) / one_hot_labels.shape[-1]
    one_hot_labels = ((1-FLAGS.label_smoothing) * one_hot_labels
                      + FLAGS.label_smoothing * smoothing)
  log_softmax_logits = jax.nn.log_softmax(logits)
  loss = -jnp.sum(one_hot_labels * log_softmax_logits) / logits.shape[0]
  return jnp.nan_to_num(loss)  # Set to zero if there is no non-masked samples.

def error_rate_metric(logits: jnp.ndarray,
  one_hot_labels: jnp.ndarray) -> jnp.ndarray:
  """Returns the error rate between some predictions and some labels.

  Args:
    logits: Output of the model.
    one_hot_labels: One-hot encoded labels. Dimensions should match the logits.
  Returns:
    The error rate (1 - accuracy), averaged over the first dimension (samples).
  """
  error_rate = (jnp.argmax(logits, -1) != jnp.argmax(one_hot_labels, -1)).sum() / logits.shape[0]
  return jnp.nan_to_num(error_rate)

def tensorflow_to_numpy(xs):
  """Converts a tree of tensorflow tensors to numpy arrays.

  Args:
    xs: A pytree (such as nested tuples, lists, and dicts) where the leaves are
      tensorflow tensors.

  Returns:
    A pytree with the same structure as xs, where the leaves have been converted
      to jax numpy ndarrays.
  """
  # Use _numpy() for zero-copy conversion between TF and NumPy.
  xs = jax.tree_map(lambda x: x._numpy(), xs)  # pylint: disable=protected-access
  return xs

def create_exponential_learning_rate_schedule(base_learning_rate: float,
  steps_per_epoch: int,
  lamba: float,
  warmup_epochs: int = 0) -> Callable[[int], float]:
  """Creates a exponential learning rate schedule with optional warmup.

  Args:
    base_learning_rate: The base learning rate.
    steps_per_epoch: The number of iterations per epoch.
    lamba: Decay is v0 * exp(-t / lambda).
    warmup_epochs: Number of warmup epoch. The learning rate will be modulated
      by a linear function going from 0 initially to 1 after warmup_epochs
      epochs.

  Returns:
    Function `f(step) -> lr` that computes the learning rate for a given step.
  """
  def learning_rate_fn(step):
    t = step / steps_per_epoch
    return base_learning_rate * jnp.exp(-t / lamba) * jnp.minimum(
        t / warmup_epochs, 1)

  return learning_rate_fn

def get_cosine_schedule(num_epochs: int, learning_rate: float, num_training_obs: int,
  batch_size: int) -> Callable[[int], float]:
  """Returns a cosine learning rate schedule, without warm up.

  Args:
    num_epochs: Number of epochs the model will be trained for.
    learning_rate: Initial learning rate.
    num_training_obs: Number of training observations.
    batch_size: Total batch size (number of samples seen per gradient step).

  Returns:
    A function that takes as input the current step and returns the learning
      rate to use.
  """
  steps_per_epoch = int(math.floor(num_training_obs / batch_size))
  learning_rate_fn = lr_schedule.create_cosine_learning_rate_schedule(
      learning_rate, steps_per_epoch // jax.host_count(), num_epochs,
      warmup_length=0)
  return learning_rate_fn

def get_std_cosine_schedule(num_epochs: int, std: float,
  num_training_obs: int,
  batch_size: int,
  inc: int) -> Callable[[int], float]:
  steps_per_epoch = int(math.floor(num_training_obs / batch_size))
  halfwavelength_steps = num_epochs * steps_per_epoch

  def std_rate_fn(step):
    scale_factor = -jnp.cos(step * jnp.pi / halfwavelength_steps) * 0.5 + 0.5
    return std * (scale_factor*inc + 1)

  return std_rate_fn

def get_std_exp_schedule(num_epochs: int, std: float,
  num_training_obs: int,
  batch_size: int) -> Callable[[int], float]:
  steps_per_epoch = int(math.floor(num_training_obs / batch_size))
  halfwavelength_steps = num_epochs * steps_per_epoch

  def std_rate_fn(step):
    scale_factor = (7)**(step/halfwavelength_steps)
    return std * scale_factor
  return std_rate_fn

def get_exponential_schedule(num_epochs: int, learning_rate: float,
  num_training_obs: int,
  batch_size: int) -> Callable[[int], float]:
  """Returns an exponential learning rate schedule, without warm up.

  Args:
    num_epochs: Number of epochs the model will be trained for.
    learning_rate: Initial learning rate.
    num_training_obs: Number of training observations.
    batch_size: Total batch size (number of samples seen per gradient step).

  Returns:
    A function that takes as input the current step and returns the learning
      rate to use.
  """
  steps_per_epoch = int(math.floor(num_training_obs / batch_size))
  # At the end of the training, lr should be 1.2% of original value
  # This mimic the behavior from the efficientnet paper.
  end_lr_ratio = 0.012
  lamba = - num_epochs / math.log(end_lr_ratio)
  learning_rate_fn = create_exponential_learning_rate_schedule(
      learning_rate, steps_per_epoch // jax.host_count(), lamba)
  return learning_rate_fn

def create_stepped_learning_rate_schedule(base_learning_rate, steps_per_epoch,
  lr_sched_steps, warmup_length=0.0):
  """Create a stepped learning rate schedule with optional warmup.
  A stepped learning rate schedule decreases the learning rate
  by specified amounts at specified epochs. The steps are given as
  the `lr_sched_steps` parameter. A common ImageNet schedule decays the
  learning rate by a factor of 0.1 at epochs 30, 60 and 80. This would be
  specified as:
  [
    [30, 0.1],
    [60, 0.01],
    [80, 0.001]
  ]
  This function also offers a learing rate warmup as per
  https://arxiv.org/abs/1706.02677, for the purpose of training with large
  mini-batches.
  Args:
    base_learning_rate: the base learning rate
    steps_per_epoch: the number of iterations per epoch
    lr_sched_steps: the schedule as a list of steps, each of which is
      a `[epoch, lr_factor]` pair; the step occurs at epoch `epoch` and
      sets the learning rate to `base_learning_rage * lr_factor`
    warmup_length: if > 0, the learning rate will be modulated by a warmup
      factor that will linearly ramp-up from 0 to 1 over the first
      `warmup_length` epochs
  Returns:
    Function `f(step) -> lr` that computes the learning rate for a given step.
  """
  def _piecewise_constant(boundaries, values, t):
      index = jnp.sum(boundaries < t)
      return jnp.take(values, index)

  boundaries = [step[0] for step in lr_sched_steps]
  decays = [step[1] for step in lr_sched_steps]
  boundaries = onp.array(boundaries) * steps_per_epoch
  boundaries = onp.round(boundaries).astype(int)
  values = onp.array([1.0] + decays) * base_learning_rate

  def learning_rate_fn(step):
    lr = _piecewise_constant(boundaries, values, step)
    if warmup_length > 0.0:
      lr = lr * jnp.minimum(1., step / float(warmup_length) / steps_per_epoch)
    return lr
  return learning_rate_fn

def get_multistep_schedule(num_epochs: int, learning_rate: float,
  num_training_obs: int, batch_size: int) -> Callable[[int], float]:
  """Returns an exponential learning rate schedule, without warm up.

  Args:
    num_epochs: Number of epochs the model will be trained for.
    learning_rate: Initial learning rate.
    num_training_obs: Number of training observations.
    batch_size: Total batch size (number of samples seen per gradient step).

  Returns:
    A function that takes as input the current step and returns the learning
      rate to use.
  """
  steps_per_epoch = int(math.floor(num_training_obs / batch_size))
  learning_rate_fn = create_stepped_learning_rate_schedule(learning_rate, steps_per_epoch,
                          [[60, 0.02], [120, 0.004], [160, 0.0008]], warmup_length=0.0)
  return learning_rate_fn

def global_norm(updates) -> jnp.ndarray:
  """Returns the l2 norm of the input.

  Args:
    updates: A pytree of ndarrays representing the gradient.
  """
  return jnp.sqrt(
      sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(updates)]))

def clip_by_global_norm(updates):
  """Clips the gradient by global norm.

  Will have no effect if FLAGS.gradient_clipping is set to zero (no clipping).

  Args:
    updates: A pytree of numpy ndarray representing the gradient.

  Returns:
    The gradient clipped by global norm.
  """
  if FLAGS.gradient_clipping > 0:
    g_norm = global_norm(updates)
    trigger = g_norm < FLAGS.gradient_clipping
    updates = jax.tree_multimap(
        lambda t: jnp.where(trigger, t, (t / g_norm) * FLAGS.gradient_clipping),
        updates)
  return updates

def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
  """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.

  Args:
    y: A pytree of numpy ndarray, vector y in the equation above.
  """
  gradient_norm = jnp.sqrt(sum(
      [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
  normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)
  return normalized_gradient

def eval_on_dataset(model: flax.nn.Model, state: flax.nn.Collection, dataset: tf.data.Dataset):
  """Evaluates the model on the whole dataset.

  Args:
    model: The model to evaluate.
    state: Current state associated with the model (contains the batch norm MA).
    dataset: Dataset on which the model should be evaluated. Should already
      being batched.
    pmapped_eval_step: A pmapped version of the `eval_step` function (see its
      documentation for more details).

  Returns:
    A dictionary containing the loss and error rate on the batch. These metrics
    are averaged over the samples.
  """
  eval_metrics = []
  total_num_samples = 0

  for batch in dataset:
    # Load and shard the TF batch.
    batch = tensorflow_to_numpy(batch)

    with flax.nn.stateful(state, mutable=False):
      logits = model(batch['image'], train=False)

    # Because we don't have a guarantee that all batches contains the same number
    # of samples, we can't average the metrics per batch and then average the
    # resulting values. To compute the metrics correctly, we sum them (error rate
    # and cross entropy returns means, thus we multiply by the number of samples),
    # and finally sum across replicas. These sums will be divided by the total
    # number of samples outside of this function.
    num_samples = batch['image'].shape[0]
    labels = batch['label']
    metrics = {
        'error_rate':
            error_rate_metric(logits, labels) * num_samples,
        'loss':
            cross_entropy_loss(logits, labels) * num_samples
    }

    eval_metrics.append(metrics)
    total_num_samples += batch['label'].shape[0]
  eval_metrics = jax.tree_multimap(lambda *args: np.stack(args), *eval_metrics)
  eval_summary = jax.tree_map(lambda x: x.sum() / total_num_samples, eval_metrics)
  return eval_summary


_EMAUpdateStep = Callable[[
    flax.optim.Optimizer, flax.nn.Collection, efficientnet_optim
    .ExponentialMovingAverage
], efficientnet_optim.ExponentialMovingAverage]


def train_for_one_epoch(dataset_source: dataset_source_lib.DatasetSource,
  optimizer: flax.optim.Optimizer, state: flax.nn.Collection,
  prng_key: jnp.ndarray, pmapped_update_ema: Optional[_EMAUpdateStep],
  moving_averages: Optional[efficientnet_optim.ExponentialMovingAverage],
  summary_writer: tensorboard.SummaryWriter,
  learning_rate_fn: Callable[[int], float],
  std_rate_fn: Callable[[int], float]) -> Tuple[flax.optim.Optimizer, flax.nn.Collection,
           Optional[efficientnet_optim.ExponentialMovingAverage]]:

  """Trains the model for one epoch.

  Args:
    dataset_source: Container for the training dataset.
    optimizer: The optimizer targeting the model to train.
    state: Current state associated with the model (contains the batch norm MA).
    prng_key: A PRNG key to use for stochasticity (e.g. for sampling an eventual
      dropout mask). Is not used for shuffling the dataset.
    pmapped_train_step: A pmapped version of the `train_step` function (see its
      documentation for more details).
    pmapped_update_ema: Function to update the parameter moving average. Can be
      None if we don't use EMA.
    moving_averages: Parameters moving average if used.
    summary_writer: A Tensorboard SummaryWriter to use to log metrics.

  Returns:
    The updated optimizer (with the associated updated model), state and PRNG
      key.
  """
  def stack_forest(forest):
    stack_args = lambda *args: np.stack(args)
    return jax.tree_multimap(stack_args, *forest)

  def forward_and_loss(model: flax.nn.Model, true_gradient: bool = False):
    """Returns the model's loss, updated state and predictions.

    Args:
      model: The model that we are training.
      true_gradient: If true, the same mixing parameter will be used for the
        forward and backward pass for the Shake Shake and Shake Drop
        regularization (see papers for more details).
    """
    with flax.nn.stateful(state) as new_state:
      with flax.nn.stochastic(prng_key):
        try:
          logits = model(
              batch['image'], train=True, true_gradient=true_gradient)
        except TypeError:
          logits = model(batch['image'], train=True)
    loss = cross_entropy_loss(logits, batch['label'])
    # We apply weight decay to all parameters, including bias and batch norm
    # parameters.
    weight_penalty_params = jax.tree_leaves(model.params)
    if FLAGS.no_weight_decay_on_bn:
      weight_l2 = sum(
          [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
    else:
      weight_l2 = sum([jnp.sum(x ** 2) for x in weight_penalty_params])
    weight_penalty = FLAGS.weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_state, logits)

  t = start_time = time.time()
  cnt = 0
  train_metrics = []
  acc_grad = None

  for batch_idx, batch in enumerate(dataset_source.get_train(use_augmentations=True), 1): 
    print(f"batch: {time.time() - t}")   
    t = time.time()
    # step
    step = optimizer.state.step * FLAGS.M + (batch_idx - 1) % FLAGS.M
    
    # step_key
    step_key = jax.random.fold_in(prng_key, step)

    # batch
    batch = tensorflow_to_numpy(batch)    

    # lr, rho, std
    lr = learning_rate_fn(step)
    rho = FLAGS.sam_rho
    std = std_rate_fn(step)

    (_, (state, logits)), grad = jax.value_and_grad(
        lambda m: forward_and_loss(m, true_gradient=True), has_aux=True)(optimizer.target)
    print(f"forward, backward: {time.time() - t}")
    # here is ssgd and sam magic
    # func = lambda a,b: a + jax.random.normal(step_key, shape=a.shape)*jnp.linalg.norm(b)*std
    # grad = jax.tree_multimap(func, grad, optimizer.target)

    metrics = {'train_error_rate': error_rate_metric(logits, batch['label']),
             'train_loss': cross_entropy_loss(logits, batch['label'])
    }

    if acc_grad is not None:
      batch_metric.append(metrics)
      acc_grad = jax.tree_multimap(lambda a,b: a + b, grad, acc_grad)
    else:
      batch_metric = [metrics]
      acc_grad = jax.tree_map(lambda a: a, grad)
    if batch_idx % (FLAGS.M) == 0:
      acc_grad = jax.tree_map(lambda a: a / FLAGS.M, acc_grad)

      # compute dual vector
      # here is sam ssgd magic
      # acc_grad = dual_vector(acc_grad)
      # noised_model = jax.tree_multimap(lambda a, b: a + rho * b,
      #                                  optimizer.target, acc_grad)
      # (_, (_, logits)), acc_grad = jax.value_and_grad(
      #     forward_and_loss, has_aux=True)(noised_model)

      # Gradient is clipped after being synchronized.
      acc_grad = clip_by_global_norm(acc_grad)

      # take gradient step
      optimizer = optimizer.apply_gradient(acc_grad, learning_rate=lr)

      # Compute some norms to log on tensorboard.
      batch_metric = stack_forest(batch_metric)
      batch_metric = jax.tree_map(lambda x: x.mean(), batch_metric)
      batch_metric["gradient_norm"] = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(acc_grad)]))
      batch_metric["param_norm"] = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(optimizer.target)]))
      
      train_metrics.append(batch_metric)
      acc_grad = None
    print(f"do something with grad: {time.time() - t}")
    t = time.time()

    cnt += 1

    if moving_averages is not None:
      moving_averages = pmapped_update_ema(optimizer, state, moving_averages)
  

  train_summary = stack_forest(train_metrics)
  train_summary = jax.tree_map(lambda x: x.mean(), train_summary)
    
  train_summary['learning_rate'] = lr
  train_summary['std_rate'] = std
  current_step = int(optimizer.state.step)
  info = 'Whole training step done in {} ({} steps)'.format(
      time.time()-start_time, cnt)
  logging.info(info)

  for metric_name, metric_value in train_summary.items():
    summary_writer.scalar(metric_name, metric_value, current_step)
    logging.info(f"{metric_name}: {metric_value:0.4f}")
  summary_writer.flush()
  return optimizer, state, moving_averages

def train(optimizer: flax.optim.Optimizer, state: flax.nn.Collection, 
  dataset_source: dataset_source_lib.DatasetSource, training_dir: str, num_epochs: int):
  """Trains the model.

  Args:
    optimizer: The optimizer targeting the model to train.
    state: Current state associated with the model (contains the batch norm MA).
    dataset_source: Container for the training dataset.
    training_dir: Parent directory where the tensorboard logs and model
      checkpoints should be saved.
   num_epochs: Number of epochs for which we want to train the model.
  """
  checkpoint_dir = os.path.join(training_dir, 'checkpoints')
  summary_writer = tensorboard.SummaryWriter(training_dir)
  prng_key = jax.random.PRNGKey(FLAGS.run_seed)

  if FLAGS.ema_decay:
    end_warmup_step = 1560
    moving_averages = efficientnet_optim.ExponentialMovingAverage(
        (optimizer.target, state), FLAGS.ema_decay, end_warmup_step)  # pytype:disable=wrong-arg-count

    def update_ema(optimizer, state, ema):
      step = optimizer.state.step
      return ema.update_moving_average((optimizer.target, state), step)

    pmapped_update_ema = jax.pmap(update_ema, axis_name='batch')
  else:
    pmapped_update_ema = moving_averages = None

  # Log initial results:
  if FLAGS.load_checkpoint is not None:
    if FLAGS.ema_decay:
      optimizer, (state,
                  moving_averages), epoch_last_checkpoint = restore_checkpoint(
                      optimizer, (state, moving_averages), FLAGS.load_checkpoint)
    else:
      optimizer, state, epoch_last_checkpoint = restore_checkpoint(
          optimizer, state, FLAGS.load_checkpoint)
    # If last checkpoint was saved at the end of epoch n, then the first
    # training epochs to do when we resume training is n+1.
    initial_epoch = epoch_last_checkpoint + 1
    info = 'Resuming training from epoch {}'.format(initial_epoch)
    logging.info(info)
  else:
    initial_epoch = jnp.array(0, dtype=jnp.int32)
    logging.info('Starting training from scratch.')

  if FLAGS.use_learning_rate_schedule:
    if FLAGS.lr_schedule == 'cosine':
      learning_rate_fn = get_cosine_schedule(num_epochs, FLAGS.learning_rate,
                                             dataset_source.num_training_obs,
                                             dataset_source.batch_size * FLAGS.M)
    elif FLAGS.lr_schedule == 'exponential':
      learning_rate_fn = get_exponential_schedule(
          num_epochs, FLAGS.learning_rate, dataset_source.num_training_obs,
          dataset_source.batch_size * FLAGS.M)
    elif FLAGS.lr_schedule == 'multistep':
      learning_rate_fn = get_multistep_schedule(
          num_epochs, FLAGS.learning_rate, dataset_source.num_training_obs,
          dataset_source.batch_size * FLAGS.M)      
    else:
      raise ValueError('Wrong schedule: ' + FLAGS.lr_schedule)
  else:
    learning_rate_fn = lambda step: FLAGS.learning_rate

  if FLAGS.use_std_schedule:
    if FLAGS.std_schedule == 'cosine':
      std_rate_fn = get_std_cosine_schedule(num_epochs, FLAGS.ssgd_std,
                                            dataset_source.num_training_obs,
                                            dataset_source.batch_size * FLAGS.M,
                                            FLAGS.std_inc - 1)
    elif FLAGS.std_schedule == 'exponential':
      std_rate_fn = get_std_exp_schedule(num_epochs, FLAGS.ssgd_std,
                                         dataset_source.num_training_obs,
                                         dataset_source.batch_size * FLAGS.M)
    else:
      raise ValueError('Wrong schedule: ' + FLAGS.std_schedule)
  else:
    std_rate_fn = lambda step: FLAGS.ssgd_std

  time_at_last_checkpoint = time.time()
  for epochs_id in range(initial_epoch, num_epochs):
    if epochs_id in FLAGS.additional_checkpoints_at_epochs:
      c_path = os.path.join(checkpoint_dir, 'additional_ckpt_' + str(epochs_id))
      save_checkpoint(optimizer, state, c_path, epochs_id)
    tick = time.time()

    optimizer, state, moving_averages = train_for_one_epoch(
        dataset_source, optimizer, state, prng_key, 
        pmapped_update_ema, moving_averages, summary_writer,
        learning_rate_fn, std_rate_fn)

    tock = time.time()
    info = 'Epoch {} finished in {:.2f}s.'.format(epochs_id, tock - tick)
    logging.info(info)

    # Evaluate the model on the test set, and optionally the training set.
    if (epochs_id + 1) % FLAGS.evaluate_every == 0:
      info = 'Evaluating at end of epoch {} (0-indexed)'.format(epochs_id)
      logging.info(info)
      tick = time.time()
      current_step = int(optimizer.state.step)
      if FLAGS.also_eval_on_training_set:
        train_ds = dataset_source.get_train(use_augmentations=False)
        train_metrics = eval_on_dataset(
            optimizer.target, state, train_ds, pmapped_eval_step)
        for metric_name, metric_value in train_metrics.items():
          summary_writer.scalar('eval_on_train_' + metric_name,
                                metric_value, current_step)
        summary_writer.flush()

      if FLAGS.ema_decay:
        logging.info('Evaluating with EMA.')
        ema_model, ema_state = moving_averages.param_ema  # pytype:disable=attribute-error
        test_ds = dataset_source.get_test()
        test_metrics = eval_on_dataset(
            ema_model, ema_state, test_ds, pmapped_eval_step)
        for metric_name, metric_value in test_metrics.items():
          summary_writer.scalar('ema_test_' + metric_name,
                                metric_value, current_step)
        summary_writer.flush()
      else:
        test_ds = dataset_source.get_test()
        test_metrics = eval_on_dataset(
            optimizer.target, state, test_ds)
        for metric_name, metric_value in test_metrics.items():
          summary_writer.scalar('test_' + metric_name,
                                metric_value, current_step)
        summary_writer.flush()
        info = f"Avgtest_loss: {test_metrics['loss']:0.3f}," \
        f"Avgtest_err:{test_metrics['error_rate']:0.3f},"
        logging.info(info)

        tock = time.time()
        info = 'Evaluated model in {:.2f}.'.format(tock - tick)
        logging.info(info)
    # Save new checkpoint if the last one was saved more than
    # `save_progress_seconds` seconds ago.
    sec_from_last_ckpt = time.time() - time_at_last_checkpoint
    if sec_from_last_ckpt > FLAGS.save_progress_seconds:
      if FLAGS.ema_decay:
        save_checkpoint(
            optimizer, (state, moving_averages), checkpoint_dir, epochs_id)
      else:
        save_checkpoint(optimizer, state, checkpoint_dir, epochs_id)
      time_at_last_checkpoint = time.time()
      logging.info('Saved checkpoint.')

  # Always save final checkpoint
  if FLAGS.ema_decay:
    save_checkpoint(
        optimizer, (state, moving_averages), checkpoint_dir, epochs_id)
  else:
    save_checkpoint(optimizer, state, checkpoint_dir, epochs_id)


