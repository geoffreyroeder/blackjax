"""Module implementing the Reparameterized Metropolis-Hastings Langevin Monte Carlo algorithm."""
import operator
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

import blackjax.mcmc.diffusions as diffusions
import blackjax.mcmc.proposal as proposal
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["ReMALAState", "ReMALAInfo", "init", "build_kernel", "as_top_level_api"]


class ReMALAState(NamedTuple):
    """State of the Reparameterized MALA algorithm."""

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree


class ReMALAInfo(NamedTuple):
    """Additional information for the Reparameterized MALA transition."""

    acceptance_rate: float
    is_accepted: bool
    estimator: ArrayTree


def init(position: ArrayLikeTree, logdensity_fn: Callable) -> ReMALAState:
    """Initialize the state for the Reparameterized MALA sampler."""
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return ReMALAState(position, logdensity, logdensity_grad)


def build_kernel(f: Callable = lambda x: x):
    """Build the kernel for the Reparameterized MALA algorithm.

    Parameters:
        f: Function whose expectation is being estimated.
           Defaults to the identity function.
    """

    def transition_energy(state, new_state, step_size):
        """Transition energy to go from `state` to `new_state`"""
        theta = jax.tree_util.tree_map(
            lambda x, new_x, g: x - new_x - step_size * g,
            state.position,
            new_state.position,
            new_state.logdensity_grad,
        )
        theta_dot = jax.tree_util.tree_reduce(
            operator.add, jax.tree_util.tree_map(lambda x: jnp.sum(x * x), theta)
        )  
        return -new_state.logdensity + 0.25 * (1.0 / step_size) * theta_dot

    compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
        transition_energy
    )
    sample_proposal = proposal.static_binomial_sampling

    def kernel(
        rng_key: PRNGKey,
        state: ReMALAState,
        logdensity_fn: Callable,
        step_size: float,
    ) -> tuple[ReMALAState, ReMALAInfo]:
        """Generate a new sample with the Reparameterized MALA kernel."""
        grad_fn = jax.value_and_grad(logdensity_fn)
        integrator = diffusions.overdamped_langevin(grad_fn)
        

        key_integrator, key_rmh = jax.random.split(rng_key)
        
        proposed_state = integrator(key_integrator, state, step_size)
        proposed_state = ReMALAState(*proposed_state)
        log_p_accept = compute_acceptance_ratio(state, proposed_state, step_size=step_size)
        accepted_state, info = sample_proposal(key_rmh, log_p_accept, state, proposed_state)
        do_accept, p_accept, _ = info

        # Compute the reMALA estimate of f by marginalizing out the Metropolis-Hastings accept-reject step
        # estimator = p_accept * f(proposed_state.position) + (1 - p_accept) * f(state.position) 
        #         # Compute the weighted estimator
        estimator = jax.tree_util.tree_map(
            lambda curr, prop: p_accept * f(prop) + (1 - p_accept) * f(curr),
            state.position,
            proposed_state.position,
        )


        info = ReMALAInfo(p_accept, do_accept, estimator)
        return accepted_state, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    f: Callable = lambda x: x,
) -> SamplingAlgorithm:
    """Top-level API for the Reparameterized MALA algorithm.

    Parameters:
        logdensity_fn: The log-density function of the target distribution.
        step_size: The step size for the Langevin proposal.
        f: Function whose expectation is being estimated.
           Defaults to the identity function.
    """
    kernel = build_kernel(f)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key  # Unused
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state: ReMALAState):
        return kernel(rng_key, state, logdensity_fn, step_size)

    return SamplingAlgorithm(init_fn, step_fn)