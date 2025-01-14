# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Solvers for Langevin diffusions."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import ArrayTree
from blackjax.util import generate_gaussian_noise

__all__ = ["overdamped_langevin"]


class DiffusionState(NamedTuple):
    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree


def overdamped_langevin(logdensity_grad_fn):
    """Euler solver for overdamped Langevin diffusion."""

    def one_step(rng_key, state: DiffusionState, step_size: float, batch: tuple = ()):
        position, _, logdensity_grad = state
        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree_util.tree_map(
            lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
            position,
            logdensity_grad,
            noise,
        )
        logdensity, logdensity_grad = logdensity_grad_fn(position, *batch)
        return DiffusionState(position, logdensity, logdensity_grad)

    return one_step

class CriticallyDampedState(NamedTuple):
    position: ArrayTree
    momentum: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree

def criticallydamped_langevin(logdensity_grad_fn, mass=1.0, gamma=2.0):
    """Euler-Maruyama solver for critically damped Langevin diffusion."""

    def one_step(rng_key, state: CriticallyDampedState, step_size: float, batch: tuple = ()):
        position, momentum, _, logdensity_grad = state
        
        # Generate noise for the momentum update
        noise = generate_gaussian_noise(rng_key, momentum)

        # Update momentum
        momentum = jax.tree_util.tree_map(
            lambda p, g, n: p - step_size * g - gamma * p * step_size + jnp.sqrt(2 * gamma * mass * step_size) * n,
            momentum,
            logdensity_grad,
            noise,
        )

        # Update position
        position = jax.tree_util.tree_map(
            lambda q, p: q + step_size * p / mass,
            position,
            momentum
        )

        # Compute new log-density and its gradient
        logdensity, logdensity_grad = logdensity_grad_fn(position, *batch)
        
        return CriticallyDampedState(position, momentum, logdensity, logdensity_grad)

    return one_step