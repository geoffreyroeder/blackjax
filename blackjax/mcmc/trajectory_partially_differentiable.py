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
"""Procedures to build trajectories for algorithms in the HMC family.

To propose a new state, algorithms in the HMC family generally proceed by
:cite:p:`betancourt2017conceptual`:

1. Sampling a trajectory starting from the initial point;
2. Sampling a new state from this sampled trajectory.

Step (1) ensures that the process is reversible and thus that detailed balance
is respected. The traditional implementation of HMC does not sample a
trajectory, but instead takes a fixed number of steps in the same direction and
flips the momentum of the last state.

We distinguish here between two different methods to sample trajectories: static
and dynamic sampling. In the static setting we sample trajectories with a fixed
number of steps, while in the dynamic setting the total number of steps is
determined by a dynamic termination criterion. Traditional HMC falls in the
former category, NUTS in the latter.

There are also two methods to sample proposals from these trajectories. In the
static setting we first build the trajectory and then sample a proposal from
this trajectory. In the progressive setting we update the proposal as the
trajectory is being sampled. While the former is faster, we risk saturating the
memory by keeping states that will subsequently be discarded.

"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.mcmc.integrators import IntegratorState
from blackjax.mcmc.proposal import (
    Proposal,
    progressive_biased_sampling,
    progressive_uniform_sampling,
    proposal_generator,
)
from blackjax.mcmc.termination import IterativeUTurnState
from blackjax.types import ArrayTree, PRNGKey


class Trajectory(NamedTuple):
    leftmost_state: IntegratorState
    rightmost_state: IntegratorState
    momentum_sum: ArrayTree
    num_states: int


def append_to_trajectory(trajectory: Trajectory, state: IntegratorState) -> Trajectory:
    """Append a state to the (right of the) trajectory to form a new trajectory."""
    momentum_sum = jax.tree_util.tree_map(
        jnp.add, trajectory.momentum_sum, state.momentum
    )
    return Trajectory(
        trajectory.leftmost_state, state, momentum_sum, trajectory.num_states + 1
    )


def reorder_trajectories(
    direction: int, trajectory: Trajectory, new_trajectory: Trajectory
) -> tuple[Trajectory, Trajectory]:
    """Order the two trajectories depending on the direction."""
    return jax.lax.cond(
        direction > 0,
        lambda _: (
            trajectory,
            new_trajectory,
        ),
        lambda _: (
            new_trajectory,
            trajectory,
        ),
        operand=None,
    )


def merge_trajectories(left_trajectory: Trajectory, right_trajectory: Trajectory):
    momentum_sum = jax.tree_util.tree_map(
        jnp.add, left_trajectory.momentum_sum, right_trajectory.momentum_sum
    )
    return Trajectory(
        left_trajectory.leftmost_state,
        right_trajectory.rightmost_state,
        momentum_sum,
        left_trajectory.num_states + right_trajectory.num_states,
    )


# -------------------------------------------------------------------
#                             Integration
#
# Generating samples by choosing a direction and running the integrator
# several times along this direction. Distinct from sampling.
# -------------------------------------------------------------------


def static_integration(
    integrator: Callable,
    direction: int = 1,
) -> Callable:
    """Generate a trajectory by integrating several times in one direction."""

    def integrate(
        initial_state: IntegratorState, step_size, num_integration_steps
    ) -> IntegratorState:
        directed_step_size = jax.tree_util.tree_map(
            lambda step_size: direction * step_size, step_size
        )

        def one_step(_, state):
            return integrator(state, directed_step_size)

        return jax.lax.fori_loop(0, num_integration_steps, one_step, initial_state)

    return integrate


class DynamicIntegrationState(NamedTuple):
    step: int
    proposal: Proposal
    trajectory: Trajectory
    termination_state: NamedTuple


def dynamic_progressive_integration(
    integrator: Callable,
    kinetic_energy: Callable,
    update_termination_state: Callable,
    is_criterion_met: Callable,
    divergence_threshold: float,
    max_scan_length: int = 10,
):
    """Integrate a trajectory and update the proposal sequentially in one direction
    until the termination criterion is met.

    Parameters
    ----------
    integrator
        The symplectic integrator used to integrate the Hamiltonian trajectory.
    kinetic_energy
        Function to compute the current value of the kinetic energy.
    update_termination_state
        Updates the state of the termination mechanism.
    is_criterion_met
        Determines whether the termination criterion has been met.
    divergence_threshold
        Value above which a transition is considered divergent.
    max_num_steps
        The maximum number of integration steps to perform.
    """
    _, generate_proposal = proposal_generator(hmc_energy(kinetic_energy))
    sample_proposal = progressive_uniform_sampling

    def integrate(
        rng_key: PRNGKey,
        initial_state: IntegratorState,
        direction: int,
        termination_state,
        max_num_steps: int,  # Set a reasonable maximum number of steps
        step_size,
        initial_energy,
    ):
        # Initialize 'done' flag and include it in the carry
        done = jnp.array(False)
        step = jnp.array(0)
        proposal = generate_proposal(initial_energy, initial_state)
        trajectory = Trajectory(
            leftmost_state=initial_state,
            rightmost_state=initial_state,
            momentum_sum=initial_state.momentum,
            num_states=0,
        )
        # Initialize integration state
        integration_state = DynamicIntegrationState(
            step, proposal, trajectory, termination_state
        )
        carry = (integration_state, rng_key, done)

        def scan_body(carry, _):
            integration_state, rng_key, done = carry
            step, proposal, trajectory, termination_state = integration_state

            # Define functions to execute when not done
            def do_integration(carry):
                integration_state, rng_key, _ = carry
                step, proposal, trajectory, termination_state = integration_state

                proposal_key = jax.random.fold_in(rng_key, step)
                rng_key, subkey = jax.random.split(rng_key)

                # Integrate forward one step
                new_state = integrator(trajectory.rightmost_state, direction * step_size)
                new_proposal = generate_proposal(initial_energy, new_state)
                is_diverging = -new_proposal.weight > divergence_threshold

                # Determine if we've reached the maximum number of steps
                # (Already handled by the fixed length of lax.scan)

                # Update the proposal
                updated_proposal = jax.lax.cond(
                    step == 0,
                    lambda _: new_proposal,  # Accept new proposal at step 0
                    lambda _: sample_proposal(proposal_key, proposal, new_proposal),
                    operand=None,
                )

                # Append the new state to the trajectory
                new_trajectory = append_to_trajectory(trajectory, new_state)

                # Update termination state
                new_termination_state = update_termination_state(
                    termination_state,
                    new_trajectory.momentum_sum,
                    new_state.momentum,
                    step,
                )

                # Check termination criterion
                has_terminated = is_criterion_met(
                    new_termination_state, new_trajectory.momentum_sum, new_state.momentum
                )

                # Update 'done' flag if stopping condition is met
                new_done = is_diverging | has_terminated

                # Update integration state
                new_integration_state = DynamicIntegrationState(
                    step + 1,
                    updated_proposal,
                    new_trajectory,
                    new_termination_state,
                )

                new_carry = (new_integration_state, rng_key, new_done)
                outputs = (updated_proposal, is_diverging, has_terminated)
                return new_carry, outputs

            # Define a function to execute when done
            def do_nothing(carry):
                # Carry forward the same state without changes
                integration_state, rng_key, done = carry
                outputs = (integration_state.proposal, False, False)
                new_carry = (integration_state, rng_key, done)
                return new_carry, outputs

            # Use lax.cond to select between the two branches
            new_carry, outputs = jax.lax.cond(
                done,
                do_nothing,
                do_integration,
                operand=carry,
            )

            return new_carry, outputs

        # Use lax.scan for a fixed number of steps
        (final_carry, outputs) = jax.lax.scan(
            scan_body,
            carry,
            xs=None,
            length=max_scan_length,
        )

        final_integration_state, rng_key, done = final_carry
        final_proposal = final_integration_state.proposal

        # Extract the stopping condition flags from outputs
        is_diverging_flags = outputs[1]
        has_terminated_flags = outputs[2]

        # Determine if any divergence or termination occurred
        is_diverging = is_diverging_flags.any()
        has_terminated = has_terminated_flags.any()

        return (
            final_integration_state.proposal,
            final_integration_state.trajectory,
            final_integration_state.termination_state,
            is_diverging,
            has_terminated,
        )

    return integrate


def dynamic_recursive_integration(
    integrator: Callable,
    kinetic_energy: Callable,
    uturn_check_fn: Callable,
    divergence_threshold: float,
    use_robust_uturn_check: bool = False,
    max_tree_depth: int = 10,
):
    """Iteratively integrate a trajectory using a loop to replace recursion,
    and update the proposal until the termination criterion is met.

    Parameters
    ----------
    integrator
        The symplectic integrator used to integrate the Hamiltonian trajectory.
    kinetic_energy
        Function to compute the current value of the kinetic energy.
    uturn_check_fn
        Function used to check the U-Turn criterion.
    divergence_threshold
        Value above which a transition is considered divergent.
    use_robust_uturn_check
        Whether to perform additional U-turn checks between subtrees.
    max_tree_depth
        The maximum depth of the tree to simulate recursion with iteration.
    """
    _, generate_proposal = proposal_generator(hmc_energy(kinetic_energy))
    proposal_sampler = progressive_uniform_sampling

    def integrate(
        rng_key: PRNGKey,
        initial_state: IntegratorState,
        direction: int,
        step_size,
        initial_energy: float,
        max_num_steps: int = 2 ** max_tree_depth,
    ):
        # Initialize the carry
        carry = (
            rng_key,
            DynamicIntegrationState(
                step=0,
                proposal=generate_proposal(initial_energy, initial_state),
                trajectory=Trajectory(
                    leftmost_state=initial_state,
                    rightmost_state=initial_state,
                    momentum_sum=initial_state.momentum,
                    num_states=1,
                ),
                is_diverging=jnp.array(False),
                is_turning=jnp.array(False),
            ),
            jnp.array(False),  # done flag
        )

        def body_fn(carry, depth):
            rng_key, integration_state, done = carry
            step = integration_state.step

            # Define functions to execute when not done
            def do_integration(carry):
                rng_key, integration_state, _ = carry
                sub_rng_key, rng_key = jax.random.split(rng_key)
                # Simulate the tree expansion
                (
                    rng_key,
                    proposal,
                    trajectory,
                    is_diverging,
                    is_turning,
                ) = build_tree(
                    sub_rng_key,
                    integration_state,
                    direction,
                    depth,
                    step_size,
                    initial_energy,
                )

                new_done = is_diverging | is_turning
                new_integration_state = DynamicIntegrationState(
                    step + 1, proposal, trajectory, is_diverging, is_turning
                )

                return (rng_key, new_integration_state, new_done)

            # When done, carry forward the state without changes
            def do_nothing(carry):
                return carry

            carry = jax.lax.cond(
                done,
                do_nothing,
                do_integration,
                operand=carry,
            )
            return carry, None

        def build_tree(
            rng_key,
            integration_state,
            direction,
            tree_depth,
            step_size,
            initial_energy,
        ):
            """Simulate the tree building process iteratively."""
            # Initialize the tree traversal
            def tree_body(carry, _):
                rng_key, state = carry

                # Integrate forward a single step
                next_state = integrator(state, direction * step_size)
                new_proposal = generate_proposal(initial_energy, next_state)
                is_diverging = -new_proposal.weight > divergence_threshold

                trajectory = Trajectory(
                    leftmost_state=next_state,
                    rightmost_state=next_state,
                    momentum_sum=next_state.momentum,
                    num_states=1,
                )

                return (rng_key, next_state), (new_proposal, trajectory, is_diverging)

            # Run the tree traversal for 2 ** tree_depth steps
            num_steps = 2 ** tree_depth

            _, (proposals, trajectories, is_diverging_flags) = jax.lax.scan(
                tree_body,
                (rng_key, integration_state.trajectory.rightmost_state),
                xs=None,
                length=num_steps,
            )

            # Combine proposals and trajectories
            combined_proposal = proposals[-1]
            combined_trajectory = trajectories[-1]
            is_diverging = is_diverging_flags.any()

            # Check for U-turn
            is_turning = uturn_check_fn(
                combined_trajectory.leftmost_state.momentum,
                combined_trajectory.rightmost_state.momentum,
                combined_trajectory.momentum_sum,
            )

            # Sample a new proposal
            proposal_key = jax.random.fold_in(rng_key, tree_depth)
            new_proposal = proposal_sampler(
                proposal_key, integration_state.proposal, combined_proposal
            )

            return (
                rng_key,
                new_proposal,
                combined_trajectory,
                is_diverging,
                is_turning,
            )

        # Run the main loop over tree depths
        depths = jnp.arange(0, max_tree_depth)
        (rng_key, final_integration_state, done), _ = jax.lax.scan(
            body_fn, carry, depths
        )

        return (
            final_integration_state.proposal,
            final_integration_state.trajectory,
            final_integration_state.is_diverging,
            final_integration_state.is_turning,
        )

    return integrate


# -------------------------------------------------------------------
#                             Sampling
#
# Sampling a trajectory by choosing a direction at random and integrating
# the trajectory in this direction. In the simplest case we perform one
# integration step, but can also perform several as is the case in the
# NUTS algorithm.
# -------------------------------------------------------------------


class DynamicExpansionState(NamedTuple):
    step: int
    proposal: Proposal
    trajectory: Trajectory
    termination_state: NamedTuple


def dynamic_multiplicative_expansion(
    trajectory_integrator: Callable,
    uturn_check_fn: Callable,
    max_num_expansions: int = 10,
    rate: int = 2,
) -> Callable:
    """Sample a trajectory and update the proposal sequentially
    until the termination criterion is met.

    This version avoids using traced integers in indexing and control flow that
    requires static shapes or values.

    Parameters
    ----------
    trajectory_integrator
        A function that runs the symplectic integrators and returns a new proposal
        and the integrated trajectory.
    uturn_check_fn
        Function used to check the U-Turn criterion.
    max_num_expansions
        The maximum number of trajectory expansions until the proposal is returned.
    rate
        The rate of the geometrical expansion. Typically 2 in NUTS.
    """
    proposal_sampler = progressive_biased_sampling

    def expand(
        rng_key: PRNGKey,
        initial_expansion_state: DynamicExpansionState,
        initial_energy: float,
        step_size: float,
    ):
        # Initialize 'done' flag and include it in the carry
        done = jnp.array(False)
        carry = (initial_expansion_state, rng_key, done)

        def scan_body(carry, _):
            expansion_state, rng_key, done = carry
            step, proposal, trajectory, termination_state = expansion_state

            # Define a function to execute when not done
            def do_expansion(carry):
                expansion_state, rng_key, _ = carry
                step, proposal, trajectory, termination_state = expansion_state

                subkey = jax.random.fold_in(rng_key, step)
                direction_key, trajectory_key, proposal_key = jax.random.split(subkey, 3)

                # Randomly choose direction
                direction = jnp.where(jax.random.bernoulli(direction_key), 1, -1)
                start_state = jax.lax.cond(
                    direction > 0,
                    lambda _: trajectory.rightmost_state,
                    lambda _: trajectory.leftmost_state,
                    operand=None,
                )

                # Integrate trajectory
                (
                    new_proposal,
                    new_trajectory,
                    termination_state,
                    is_diverging,
                    is_turning_subtree,
                ) = trajectory_integrator(
                    trajectory_key,
                    start_state,
                    direction,
                    termination_state,
                    rate ** step,
                    step_size,
                    initial_energy,
                )

                # Update the proposal
                def update_sum_log_p_accept():
                    return Proposal(
                        proposal.state,
                        proposal.energy,
                        proposal.weight,
                        jnp.logaddexp(proposal.sum_log_p_accept, new_proposal.sum_log_p_accept),
                    )

                updated_proposal = jax.lax.cond(
                    is_diverging | is_turning_subtree,
                    lambda: update_sum_log_p_accept(),
                    lambda: proposal_sampler(proposal_key, proposal, new_proposal),
                )

                # Merge the trajectories and check for U-turn
                left_trajectory, right_trajectory = reorder_trajectories(
                    direction, trajectory, new_trajectory
                )

                merged_trajectory = merge_trajectories(left_trajectory, right_trajectory)

                is_turning = uturn_check_fn(
                    merged_trajectory.leftmost_state.momentum,
                    merged_trajectory.rightmost_state.momentum,
                    merged_trajectory.momentum_sum,
                )

                # Update 'done' flag if stopping condition is met
                new_done = is_diverging | is_turning
                
                    # Make sure idx_min and idx_max have consistent dtype
                new_termination_state = IterativeUTurnState(
                    momentum=termination_state.momentum,
                    momentum_sum=termination_state.momentum_sum,
                    idx_min=jnp.asarray(termination_state.idx_min, dtype=jnp.int32),
                    idx_max=jnp.asarray(termination_state.idx_max, dtype=jnp.int32),
                )

                # Update the expansion state
                new_expansion_state = DynamicExpansionState(
                    step + 1, updated_proposal, merged_trajectory, new_termination_state
                )

                new_carry = (new_expansion_state, rng_key, new_done)
                outputs = (updated_proposal, is_diverging, is_turning)
                return new_carry, outputs

            # Define a function to execute when done
            def do_nothing(carry):
                # Carry forward the same state without changes
                expansion_state, rng_key, done = carry
                # Ensure termination_state has consistent dtype
                termination_state = IterativeUTurnState(
                    momentum=expansion_state.termination_state.momentum,
                    momentum_sum=expansion_state.termination_state.momentum_sum,
                    idx_min=jnp.asarray(expansion_state.termination_state.idx_min, dtype=jnp.int32),
                    idx_max=jnp.asarray(expansion_state.termination_state.idx_max, dtype=jnp.int32),
                )

                # Update the expansion state with consistent termination_state
                expansion_state = DynamicExpansionState(
                    expansion_state.step,
                    expansion_state.proposal,
                    expansion_state.trajectory,
                    termination_state,
                )
                            

                outputs = (expansion_state.proposal, False, False)
                new_carry = (expansion_state, rng_key, done)
                return new_carry, outputs

            # Use lax.cond to select between the two branches
            new_carry, outputs = jax.lax.cond(
                done,
                do_nothing,
                do_expansion,
                operand=carry,
            )

            return new_carry, outputs

        # Use lax.scan for a fixed number of expansions
        (final_carry, outputs) = jax.lax.scan(
            scan_body,
            carry,
            xs=None,
            length=max_num_expansions,
        )

        final_expansion_state, rng_key, done = final_carry
        final_proposal = final_expansion_state.proposal

        # Extract the stopping condition flags from outputs
        # The last true value of 'done' indicates the stopping point
        is_diverging_flags = outputs[1]
        is_turning_flags = outputs[2]

        # Since we are carrying 'done' in the scan, we don't need to use traced indices
        # The final 'done' flag tells us if we have stopped

        return final_expansion_state, (is_diverging_flags[-1], is_turning_flags[-1])

    return expand


def hmc_energy(kinetic_energy):
    def energy(state):
        return -state.logdensity + kinetic_energy(
            state.momentum, position=state.position
        )

    return energy
