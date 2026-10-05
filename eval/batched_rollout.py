"""Lockstep RL search with shared inference and independent per-start accounting."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import heapq
import math

import numpy as np

from eval.algorithm.base import SearchNode
from eval.algorithm.rl import RLAlgorithm
from eval.policy import PolicyScorer
from eval.rollout import RolloutResult, _RolloutSession
from reward_functions import get_reward


@dataclass(frozen=True)
class _ScoredNode:
    node: SearchNode
    score: float = 0.0


@dataclass
class _Trajectory:
    session: _RolloutSession
    algorithm: RLAlgorithm
    initial: SearchNode
    frontier: list[_ScoredNode]  # Current beam, or the single selected BeFS parent.
    seen: set[str]
    pending_frontier: list[tuple[float, int, _ScoredNode]] = field(default_factory=list)
    reason: str | None = None
    result: RolloutResult | None = None


@dataclass
class _Expansion:
    trajectory: _Trajectory
    parent: _ScoredNode
    actions: tuple
    transitions: dict
    log_probabilities: object = None
    selected_indices: list[int] = field(default_factory=list)
    states: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)


def _prepare_layer(trajectories, engine, policy):
    """Batch free neighbor enumeration/scoring, then admit a budgeted prefix.

    Enumeration may prefetch later beam parents. Only admitted parents become
    logical expansions; no objective is queried for a parent beyond its budget.
    Transition snapshots survive cache eviction while the layer is consumed.
    BeFS supplies only its selected parent; its other frontier entries stay queued.
    """
    entries = [(trajectory, parent) for trajectory in trajectories
               if trajectory.result is None for parent in trajectory.frontier]
    actions, _ = engine.candidate_actions_for_states([parent.node.state for _, parent in entries])
    expansions = [_Expansion(trajectory, parent, tuple(action_list),
                             dict(engine.nodes_by_key[parent.node.state.key].transitions))
                  for (trajectory, parent), action_list in zip(entries, actions)]
    scored = [item for item in expansions if item.trajectory.algorithm.requires_policy and item.actions]
    if scored:
        probabilities = policy.score_actions([item.parent.node.state for item in scored],
                                             [item.actions for item in scored])
        if len(probabilities) != len(scored):
            raise ValueError("Policy returned the wrong number of action-score rows.")
        for item, row in zip(scored, probabilities):
            row = np.asarray(row, dtype=np.float64)
            if row.shape != (len(item.actions),) or not np.isfinite(row).all():
                raise ValueError(f"Invalid policy scores for state {item.parent.node.state.key}.")
            item.log_probabilities = row

    scheduled = []
    previous = None
    for item in expansions:
        trajectory = item.trajectory
        if trajectory is not previous:
            remaining = trajectory.session.remaining_budget
            reserved = set()
            previous = trajectory
        if remaining <= 0:
            continue
        next_keys = [item.transitions[action].next_key for action in item.actions]
        item.selected_indices = trajectory.algorithm.propose(
            next_keys, item.log_probabilities, trajectory.seen, trajectory.session.rng, reserved,
        )
        for index in item.selected_indices:
            action = item.actions[index]
            state = engine.materialize_transition(item.parent.node.state, item.transitions[action])
            item.states[action] = state
            reserved.add(state.key)
        remaining -= len(item.selected_indices)
        scheduled.append(item)
    return scheduled


def run_batched_rollouts(
    initial_states, algorithms, engine, *, policy: PolicyScorer,
    objective_function, objective_goal, objective_budget, seeds, start_indices,
    objective_name="max_kcup", reward_function=None, batch_objective_function=None,
    on_query=None, on_transition=None, on_expansion=None, on_rollout=None,
) -> list[RolloutResult]:
    """Advance every start together; physical chunks never own search state.

    Batch objective providers must yield values in input order. They implement
    physical work only; each result still passes through _RolloutSession.
    Without a batch provider the scalar callback remains supported.
    """
    if type(objective_budget) is not int or objective_budget < 0:
        raise ValueError("objective_budget must be a non-negative integer.")
    if objective_goal not in ("min", "max"):
        raise ValueError("objective_goal must be 'min' or 'max'.")
    if any(len(items) != len(initial_states) for items in (algorithms, seeds, start_indices)):
        raise ValueError("Each initial state needs an algorithm, seed and start_index.")
    if not all(isinstance(algorithm, RLAlgorithm) for algorithm in algorithms):
        raise TypeError("run_batched_rollouts requires RLAlgorithm instances.")
    for algorithm in algorithms:
        if algorithm.supported_objectives is not None and objective_name not in algorithm.supported_objectives:
            raise ValueError(f"{algorithm.name} supports only {algorithm.supported_objectives}.")
    if policy is None:
        raise ValueError("RL evaluation requires a shared policy scorer.")
    if not initial_states:
        return []
    reward = reward_function if reward_function is not None else get_reward(objective_name)
    batch_objectives = (batch_objective_function if batch_objective_function is not None else
                        lambda states: (objective_function(state) for state in states))
    trajectories = []
    current_session = None

    @contextmanager
    def objective_values(states):
        values = iter(batch_objectives(states))
        try:
            yield values
        finally:
            close = getattr(values, "close", None)
            if close is not None:
                close()

    def finish(trajectory):
        trajectory.result = trajectory.session.result(trajectory.initial, trajectory.reason)
        trajectory.frontier.clear()
        trajectory.pending_frontier.clear()
        trajectory.seen.clear()
        if on_rollout is not None:
            on_rollout(trajectory.result)

    try:
        with objective_values(initial_states) as initial_values:
            for state, algorithm, seed, start_index in zip(initial_states, algorithms, seeds, start_indices):
                session = _RolloutSession(
                    state, engine, identity=dict(algorithm=algorithm.name, polytope_index=state.point_config_index,
                                                start_index=start_index, seed=seed),
                    objective_function=objective_function, objective_goal=objective_goal,
                    objective_name=objective_name, objective_budget=objective_budget,
                    on_query=on_query, on_transition=on_transition, on_expansion=on_expansion,
                )
                current_session = session
                value = session.evaluate(state, depth=0, objective_function=lambda _: next(initial_values))
                initial = SearchNode(state, value, 0, 0)
                trajectory = _Trajectory(session, algorithm, initial, [_ScoredNode(initial)], {state.key})
                trajectories.append(trajectory)
                if session.remaining_budget == 0:
                    trajectory.reason = "budget_exhausted"
                    finish(trajectory)
        engine.release_active_states()

        while any(trajectory.result is None for trajectory in trajectories):
            active = [trajectory for trajectory in trajectories if trajectory.result is None]
            current_session = active[0].session
            try:
                scheduled = _prepare_layer(active, engine, policy)
                value_inputs = [(item, action, state) for item in scheduled
                                if item.trajectory.algorithm.requires_values
                                for action, state in item.states.items()]
                if value_inputs:
                    values = policy.score_values([state for _, _, state in value_inputs])
                    if len(values) != len(value_inputs):
                        raise ValueError("Policy returned the wrong number of critic values.")
                    for (item, action, _), value in zip(value_inputs, values):
                        if not math.isfinite(float(value)):
                            raise ValueError("Nonfinite critic value in evaluation batch.")
                        item.values[action] = float(value)

                candidates = [state for item in scheduled for state in item.states.values()]
                next_frontiers = {id(trajectory): [] for trajectory in active}
                with objective_values(candidates) as values:
                    for item in scheduled:
                        trajectory = item.trajectory
                        session, algorithm = trajectory.session, trajectory.algorithm
                        current_session = session
                        with session.expansion(
                            item.parent.node, prepared=(item.actions, item.transitions),
                            materialized_states=item.states, objective_function=lambda _: next(values),
                            release_active_states=False,
                        ) as (context, _):
                            if algorithm.is_walk and not item.selected_indices:
                                trajectory.reason = "no_unvisited_neighbors" if item.actions else "no_neighbors"
                            queries_before = session.queries
                            for index in item.selected_indices:
                                action = item.actions[index]
                                evaluated = context.evaluate_action(action)
                                trajectory.seen.add(evaluated.state.key)
                                score = algorithm.score_candidate(
                                    item.parent.score,
                                    item.log_probabilities[index] if algorithm.requires_policy else 0.0,
                                    item.parent.node.objective, evaluated.objective, reward,
                                    item.values.get(action, 0.0),
                                )
                                if not math.isfinite(score):
                                    raise ValueError("Nonfinite RL search score.")
                                child = SearchNode(evaluated.state, evaluated.objective, evaluated.query_index,
                                                   item.parent.node.depth + 1)
                                next_frontiers[id(trajectory)].append(_ScoredNode(child, float(score)))
                                if algorithm.is_walk:
                                    session.move(item.parent.node, evaluated, queries_before)

                        # Preserve completed starts even if a later start in this
                        # physical batch fails. Nothing remains after its budget.
                        if session.remaining_budget == 0:
                            trajectory.reason = "budget_exhausted"
                        if trajectory.reason is not None:
                            finish(trajectory)

                for trajectory in active:
                    if trajectory.result is not None:
                        continue
                    children = next_frontiers[id(trajectory)]
                    if trajectory.algorithm.is_best_first:
                        for child in children:
                            heapq.heappush(trajectory.pending_frontier,
                                           (-child.score, child.node.query_index, child))
                        trajectory.frontier = ([heapq.heappop(trajectory.pending_frontier)[2]]
                                               if trajectory.pending_frontier else [])
                    else:
                        trajectory.frontier = sorted(children,
                                                     key=lambda item: (-item.score, item.node.query_index))[
                                                         :trajectory.algorithm.beam_width]
                    if not trajectory.frontier:
                        trajectory.reason = "frontier_exhausted"
                        finish(trajectory)
            finally:
                engine.release_active_states()
        return [trajectory.result for trajectory in trajectories]
    except Exception as exc:
        if current_session is not None:
            raise current_session.failure(exc) from exc
        raise
    finally:
        engine.release_active_states()
