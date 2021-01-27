# Copyright 2020 The Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wide Resnet Model.

Reference:

Wide Residual Networks, Sergey Zagoruyko, Nikos Komodakis
https://arxiv.org/abs/1605.07146

Forked from
https://github.com/google-research/google-research/blob/master/flax_models/cifar/models/wide_resnet.py

This implementation mimics the one from
github.com/tensorflow/models/blob/master/research/autoaugment/wrn.py
that is widely used as a benchmark.

It uses idendity + zero padding skip connections, with kaiming normal
initialization for convolutional kernels (mode = fan_out, gain=2.0).
The final dense layer use a uniform distribution U[-scale, scale] where
scale = 1 / sqrt(num_classes) as per the autoaugment implementation.

Using the default initialization instead gives error rates approximately 0.5%
greater on cifar100, most likely between the parameters used in the literature
where finetuned for this particular initialization.

Finally, the autoaugment implementation adds more residual connections between
the groups (instead of just between the blocks as per the original paper and
most implementations). It is possible to safely remove those connections without
degrading the performances, which we do by default to match the original
wideresnet paper. Setting `use_additional_skip_connections` to True will add
them back and then reproduces exactly the model used in autoaugment.
"""

from typing import Tuple

from absl import flags
from flax import nn
from jax import numpy as jnp
import jax
from models import utils


FLAGS = flags.FLAGS

class LeNet(nn.Module):
  def apply(self,
            x: jnp.ndarray,
            num_outputs: int,
            train: bool = True) -> jnp.ndarray:
    x = nn.Conv(
        x,
        20, (5, 5),
        padding='VALID',
        name='init_conv',
        kernel_init=utils.conv_kernel_init_fn,
        bias=False)
    x = jax.nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2,2), padding="VALID")

    x = nn.Conv(
        x,
        50, (5, 5),
        padding='VALID',
        name='conv1',
        kernel_init=utils.conv_kernel_init_fn,
        bias=False)
    x = jax.nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2,2), padding="VALID")

    x = nn.Conv(
        x,
        20, (5, 5),
        padding='SAME',
        name='conv2',
        kernel_init=utils.conv_kernel_init_fn,
        bias=False)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(x, 500, kernel_init=utils.dense_layer_init_fn, bias=True)
    x = jax.nn.relu(x)
    x = nn.dropout(x, 0.1, deterministic=not train)
    x = nn.Dense(x, num_outputs, kernel_init=utils.dense_layer_init_fn, bias=True)
    
    return x

