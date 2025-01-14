# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""Public API for the Critically Damped Hamiltonian Monte Carlo (CDHMC) Kernel"""

from typing import Callable, NamedTuple, Union

import jax
import jax.numpy as jnp

import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.trajectory as trajectory
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.proposal import safe_energy_diff, static_binomial_sampling
from blackjax.mcmc.trajectory import hmc_energy
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "CLDState",
    "CLDInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
]

class CLDState(NamedTuple):
    """State of the CDHMC algorithm.

    The CDHMC algorithm maintains both position and momentum, as well as the
    log-density and its gradient.
    """

    position: ArrayTree
    momentum: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree


class CLDInfo(NamedTuple):
    """Additional information on the CDHMC transition."""

    acceptance_rate: float
    is_accepted: bool
    is_divergent: bool
    energy: float
    proposal: trajectory.IntegratorState
    num_integration_steps: int


def init(position: ArrayLikeTree, logdensity_fn: Callable):
    """Initialize the CLDMC state."""
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    momentum = jax.tree_util.tree_map(jnp.zeros_like, position)
    return CLDState(position, momentum, logdensity, logdensity_grad)


def build_kernel(
    integrator: Callable,
    divergence_threshold: float = 1000,
    mass: float = 1.0,
    gamma: float = 2.0,
):
    """Build a CLDMC kernel.

    Parameters
    ----------
    integrator
        The integrator function implementing the critically damped dynamics.
    divergence_threshold
        Threshold for considering a transition as divergent.
    mass
        Mass parameter for the dynamics.
    gamma
        Damping coefficient, set to achieve critical damping (gamma = 2 * sqrt(mass)).
    """

    def kernel(
        rng_key: PRNGKey,
        state: CLDState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: metrics.ArrayLikeTree,
        num_integration_steps: int,
    ) -> tuple[CLDState, CLDInfo]:
        """Generate a new sample with the CLDMC kernel."""

        metric = metrics.positive_definite_matrix(inverse_mass_matrix)
        symplectic_integrator = integrator(logdensity_fn, mass=mass, gamma=gamma)
        proposal_generator = cld_proposal(
            symplectic_integrator,
            metric,
            step_size,
            num_integration_steps,
            divergence_threshold,
        )

        key_integrator = rng_key  # CLD dynamics include noise internally

        position, momentum, logdensity, logdensity_grad = state

        integrator_state = trajectory.IntegratorState(
            position, momentum, logdensity, logdensity_grad
        )
        proposal, info, _ = proposal_generator(key_integrator, integrator_state)

        new_state = CLDState(
            proposal.position, proposal.momentum, proposal.logdensity, proposal.logdensity_grad
        )

        return new_state, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    inverse_mass_matrix: metrics.ArrayLikeTree,
    num_integration_steps: int,
    *,
    divergence_threshold: int = 1000,
    mass: float = 1.0,
    gamma: float = 2.0,
) -> SamplingAlgorithm:
    """User interface for the CDHMC kernel.

    Parameters
    ----------
    logdensity_fn
        The log-density function to sample from.
    step_size
        Integration step size.
    inverse_mass_matrix
        Inverse mass matrix for sampling momentum and computing kinetic energy.
    num_integration_steps
        Number of integration steps in each proposal.
    divergence_threshold
        Threshold for considering a transition as divergent.
    mass
        Mass parameter for the dynamics.
    gamma
        Damping coefficient.
    """

    from blackjax.mcmc.integrators import cd_langevin_integrator

    kernel = build_kernel(
        cd_langevin_integrator,
        divergence_threshold,
        mass=mass,
        gamma=gamma,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            inverse_mass_matrix,
            num_integration_steps,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def cld_proposal(
    integrator: Callable,
    metric: metrics.EuclideanMetric,
    step_size: float,
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
    *,
    sample_proposal: Callable = static_binomial_sampling,
) -> Callable:
    """Critically Damped HMC proposal function.

    Parameters
    ----------
    integrator
        Integrator function for critically damped dynamics.
    metric
        Metric object to compute kinetic energy.
    step_size
        Integration step size.
    num_integration_steps
        Number of integration steps.
    divergence_threshold
        Threshold for considering a transition as divergent.
    sample_proposal
        Function to sample the proposal (default: static binomial sampling).
    """

    build_trajectory = trajectory.static_integration(integrator)
    total_energy = hmc_energy(metric.kinetic_energy)

    def generate(
        rng_key, state: trajectory.IntegratorState
    ) -> tuple[trajectory.IntegratorState, CLDInfo, ArrayTree]:
        """Generate a new chain state."""
        end_state = build_trajectory(state, step_size, num_integration_steps)

        proposal_energy = total_energy(state)
        new_energy = total_energy(end_state)
        delta_energy = safe_energy_diff(proposal_energy, new_energy)
        is_diverging = -delta_energy > divergence_threshold
        sampled_state, info = sample_proposal(rng_key, delta_energy, state, end_state)
        do_accept, p_accept, other_proposal_info = info

        info = CLDInfo(
            p_accept,
            do_accept,
            is_diverging,
            new_energy,
            end_state,
            num_integration_steps,
        )

        return sampled_state, info, other_proposal_info

    return generate