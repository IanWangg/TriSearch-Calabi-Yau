"""Shared accounting for single-path rollouts and frontier searches."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Callable, Iterator

import numpy as np

from eval.algorithm.base import (
    Algorithm, EvaluatedAction, EvaluationAlgorithm, EvaluationContext,
    FrontierSearchAlgorithm, PopulationAlgorithm, SearchNode,
)
from mdp.cy_rollout import CYRandomRolloutEngine
from mdp.cy_state_record import CyStateRecord


RESULT_FORMAT_VERSION = 2


class ObjectiveBudgetExhausted(Exception):
    """Normal interruption of an upstream population optimizer at an exact budget."""


@dataclass(frozen=True)
class RolloutResult:
    algorithm: str
    polytope_index: int
    start_index: int
    seed: int
    objective_name: str
    objective_goal: str
    objective_budget: int
    objective_queries: int
    transition_count: int
    expansion_count: int
    budget_overshoot: int
    initial_state_key: str
    best_state_key: str
    initial_objective: float
    best_objective: float
    termination_reason: str


class _RolloutSession:
    """One budget and best tracker, shared by both algorithm interfaces."""

    def __init__(self, initial_state, engine, *, identity, objective_function, objective_goal,
                 objective_name, objective_budget, on_query, on_transition, on_expansion):
        self.engine = engine
        self.identity = identity
        self.objective_function = objective_function
        self.goal, self.objective_name = objective_goal, objective_name
        self.budget = objective_budget
        self.on_query, self.on_transition, self.on_expansion = on_query, on_transition, on_expansion
        self.rng = np.random.default_rng(identity["seed"])
        self.queries = self.moves = self.expansions = 0
        self.source = initial_state
        self.best_key = initial_state.key
        self.best_value = None

    @property
    def remaining_budget(self) -> int:
        return max(0, self.budget - self.queries)

    def evaluate(self, state, *, depth: int | None, action=None, objective_function=None, metadata=None) -> float:
        event = {**self.identity, "query_index": self.queries, "round_index": self.expansions,
                 "expansion_index": self.expansions, "depth": depth,
                 "is_initial": action is None, "source_key": self.source.key,
                 "state_key": state.key, "action": action}
        if metadata is not None:
            event.update(metadata)
        try:
            value = float((objective_function or self.objective_function)(state))
            if not math.isfinite(value) or (self.objective_name == "max_kcup" and value <= 0):
                raise ValueError(f"Invalid {self.objective_name} objective: {value}.")
        except Exception as exc:
            if self.on_query is not None:
                self.on_query({**event, "status": "failed", "error": str(exc)})
            raise
        if self.best_value is None or (value > self.best_value if self.goal == "max" else value < self.best_value):
            self.best_key, self.best_value = state.key, value
        if self.on_query is not None:
            self.on_query({**event, "status": "ok", "objective": value,
                           "best_objective": self.best_value, "best_state_key": self.best_key})
        return value

    @contextmanager
    def expansion(self, node: SearchNode, *, prepared=None, objective_function=None,
                  materialized_states=None, release_active_states=True):
        """Account for one logical parent expansion, including cache hits/dead ends."""
        if self.remaining_budget <= 0:
            raise RuntimeError("Cannot start a parent expansion after exhausting the objective budget.")
        self.source = node.state
        self.expansions += 1
        queries_before = self.queries
        event = {**self.identity, "expansion_index": self.expansions, "depth": node.depth,
                 "state_key": node.state.key, "objective": node.objective,
                 "queries_before": queries_before}
        try:
            if prepared is None:
                actions_by_state, _ = self.engine.candidate_actions_for_states([node.state])
                actions = tuple(actions_by_state[0])
                transitions = self.engine.nodes_by_key[node.state.key].transitions
            else:
                actions, transitions = prepared
            event["candidate_count"] = len(actions)

            def evaluate_action(action) -> EvaluatedAction:
                transition = transitions[action]
                self.queries += 1
                candidate = (materialized_states[action] if materialized_states is not None else
                             self.engine.materialize_transition(node.state, transition))
                value = self.evaluate(candidate, depth=node.depth + 1, action=action,
                                      objective_function=objective_function)
                self.engine.prune_runtime_caches()
                return EvaluatedAction(action, candidate, value, self.queries)

            yield EvaluationContext(
                current_state=node.state, current_objective=node.objective, actions=actions,
                goal=self.goal, rng=self.rng, remaining_budget=self.remaining_budget,
                evaluate_action=evaluate_action,
            ), transitions
            event["status"] = "ok"
        except Exception as exc:
            event.update(status="failed", error=str(exc))
            raise
        finally:
            try:
                if self.on_expansion is not None:
                    self.on_expansion({**event, "objective_queries": self.queries,
                                       "round_queries": self.queries - queries_before})
            finally:
                if release_active_states:
                    self.engine.release_active_states()

    def move(self, current: SearchNode, selected: EvaluatedAction, queries_before: int) -> SearchNode:
        self.moves += 1
        if self.on_transition is not None:
            self.on_transition({**self.identity, "transition_index": self.moves,
                                "expansion_index": self.expansions, "objective_queries": self.queries,
                                "round_queries": self.queries - queries_before, "source_key": current.state.key,
                                "state_key": selected.state.key, "action": selected.action,
                                "objective": selected.objective, "best_objective": self.best_value,
                                "best_state_key": self.best_key})
        return SearchNode(selected.state, selected.objective, selected.query_index, current.depth + 1)

    def result(self, initial: SearchNode, reason: str) -> RolloutResult:
        return RolloutResult(
            **self.identity, objective_name=self.objective_name, objective_goal=self.goal,
            objective_budget=self.budget, objective_queries=self.queries, transition_count=self.moves,
            expansion_count=self.expansions, budget_overshoot=max(0, self.queries - self.budget),
            initial_state_key=initial.state.key, best_state_key=self.best_key,
            initial_objective=initial.objective, best_objective=self.best_value, termination_reason=reason,
        )

    def failure(self, exc) -> RuntimeError:
        return RuntimeError(
            f"Rollout failed: algorithm={self.identity['algorithm']}, polytope={self.identity['polytope_index']}, "
            f"start={self.identity['start_index']}, queries={self.queries}, transitions={self.moves}, "
            f"expansions={self.expansions}, state={self.source.key}: {exc}"
        )

    def walk(self, initial: SearchNode, algorithm: EvaluationAlgorithm) -> str:
        current = initial
        while self.remaining_budget > 0:
            with self.expansion(current) as (context, transitions):
                if not context.actions:
                    return "no_neighbors"
                queries_before = self.queries
                selected = algorithm.select_action(context)
                if (selected.action not in transitions or not queries_before < selected.query_index <= self.queries
                        or selected.state.key != transitions[selected.action].next_key):
                    raise ValueError("Algorithm must return an action evaluated in the current round.")
                current = self.move(current, selected, queries_before)
        return "budget_exhausted"


class _SearchContext:
    """Per-start graph deduplication, independent of the shared engine's history."""

    def __init__(self, session: _RolloutSession, initial: SearchNode):
        self._session = session
        self.initial = initial
        self.goal = session.goal
        self._seen_keys = {initial.state.key}

    @property
    def remaining_budget(self) -> int:
        return self._session.remaining_budget

    def priority(self, node: SearchNode) -> tuple[float, int]:
        return (-node.objective if self.goal == "max" else node.objective, node.query_index)

    def expand(self, node: SearchNode) -> Iterator[SearchNode]:
        with self._session.expansion(node) as (context, transitions):
            for action in context.actions:
                key = transitions[action].next_key
                if key in self._seen_keys:
                    continue
                evaluated = context.evaluate_action(action)
                self._seen_keys.add(key)
                yield SearchNode(evaluated.state, evaluated.objective, evaluated.query_index, node.depth + 1)


class _PopulationContext:
    """Account for fitness queries without inventing flip edges or parent depths."""

    def __init__(self, session, initial):
        self._session, self.initial = session, initial
        self.goal, self.seed = session.goal, session.identity["seed"]
        self._event = None

    @property
    def remaining_budget(self):
        return self._session.remaining_budget

    @contextmanager
    def generation(self, index, **metadata):
        session = self._session
        if not self.remaining_budget:
            raise ObjectiveBudgetExhausted
        session.expansions += 1
        before = session.queries
        self._event = dict(**session.identity, kind="population", generation_index=index,
                           expansion_index=session.expansions, depth=None, state_key=None, objective=None,
                           queries_before=before, candidate_count=0, rejected_candidates=[], **metadata)
        try:
            yield
            self._event["status"] = "ok"
        except ObjectiveBudgetExhausted:
            self._event.update(status="ok", stopped_at_budget=True)
            raise
        except Exception as exc:
            self._event.update(status="failed", error=str(exc))
            raise
        finally:
            try:
                if session.on_expansion is not None:
                    session.on_expansion({**self._event, "objective_queries": session.queries,
                                          "round_queries": session.queries - before})
            finally:
                self._event = None
                session.engine.release_active_states()

    def reject(self, dna, reason):
        self._event["candidate_count"] += 1
        self._event["rejected_candidates"].append(dict(dna=list(dna), reason=reason))

    def materialize_candidate(self, encoding, dna):
        """Store encoding aliases in the existing bounded hot-state cache.

        Aliases have a distinct prefix and share the exact byte/entry limits and
        pressure/close lifecycle of normal state keys; no extra cache is created.
        The start's DNA always resolves to its supplied full triangulation.
        """
        initial = self.initial.state
        if dna == encoding.initial_dna[initial.key]:
            return initial
        cache = self._session.engine.state_cache
        if cache.mode == "none":
            return encoding.decode(dna, initial)
        key = f"cyopt_dna|{initial.point_config_index}|{encoding.metadata['codebook_sha256']}|{dna}"
        state = cache.hot_states.get(key)
        if state is None:
            state = encoding.decode(dna, initial)
            if state is not None:
                cache.hot_states[key] = state
        return state

    def evaluate_candidate(self, state, *, dna):
        if not self.remaining_budget:
            raise ObjectiveBudgetExhausted
        session = self._session
        self._event["candidate_count"] += 1
        session.queries += 1
        state.bind_objective_provider(session.engine.objective_value)
        value = session.evaluate(state, depth=None, metadata=dict(
            is_initial=False, source_key=None, candidate_kind="cyopt_dna", dna=list(dna),
            generation_index=self._event["generation_index"]))
        session.engine.prune_runtime_caches()
        return value


def run_rollout(
    initial_state: CyStateRecord,
    algorithm: Algorithm,
    engine: CYRandomRolloutEngine,
    *,
    objective_function: Callable[[CyStateRecord], float],
    objective_goal: str,
    objective_budget: int,
    seed: int,
    start_index: int = 0,
    objective_name: str = "max_kcup",
    on_query: Callable[[dict], None] | None = None,
    on_transition: Callable[[dict], None] | None = None,
    on_expansion: Callable[[dict], None] | None = None,
    policy=None,
    reward_function=None,
    batch_objective_function=None,
) -> RolloutResult:
    """Run one independent start, scoring all queried states toward the best.

    Initial evaluation is free. Each action query costs one even on a cache hit.
    Frontier algorithms skip previously discovered states before querying, and
    complete one parent before checking the budget. Frontier switches are not
    trajectory moves; their work is represented by expansion events instead.
    """
    if type(objective_budget) is not int or objective_budget < 0:
        raise ValueError("objective_budget must be a non-negative integer.")
    if objective_goal not in ("min", "max"):
        raise ValueError("objective_goal must be 'min' or 'max'.")
    from eval.algorithm.rl import RLAlgorithm

    if isinstance(algorithm, RLAlgorithm):
        from eval.batched_rollout import run_batched_rollouts

        return run_batched_rollouts(
            [initial_state], [algorithm], engine, policy=policy,
            objective_function=objective_function, objective_goal=objective_goal,
            objective_budget=objective_budget, seeds=[seed], start_indices=[start_index],
            objective_name=objective_name, reward_function=reward_function,
            batch_objective_function=batch_objective_function,
            on_query=on_query, on_transition=on_transition, on_expansion=on_expansion,
        )[0]
    identity = {"algorithm": algorithm.name, "polytope_index": initial_state.point_config_index,
                "start_index": start_index, "seed": seed}
    session = _RolloutSession(
        initial_state, engine, identity=identity, objective_function=objective_function,
        objective_goal=objective_goal, objective_name=objective_name, objective_budget=objective_budget,
        on_query=on_query, on_transition=on_transition, on_expansion=on_expansion,
    )
    try:
        initial = SearchNode(initial_state, session.evaluate(initial_state, depth=0), 0, 0)
        if isinstance(algorithm, PopulationAlgorithm):
            reason = algorithm.run_population(_PopulationContext(session, initial))
        elif isinstance(algorithm, FrontierSearchAlgorithm):
            algorithm.search(_SearchContext(session, initial))
            reason = "budget_exhausted" if session.remaining_budget == 0 else "frontier_exhausted"
        else:
            reason = session.walk(initial, algorithm)
        return session.result(initial, reason)
    except Exception as exc:
        raise session.failure(exc) from exc
    finally:
        engine.release_active_states()
