"""Algorithms choose from transitions and query objectives through one callback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, runtime_checkable

import numpy as np

from mdp.cy_graph import CanonicalAction
from mdp.cy_state_record import CyStateRecord
from eval.algorithm.rl import RLAlgorithm


@dataclass(frozen=True)
class EvaluatedAction:
    action: CanonicalAction
    state: CyStateRecord
    objective: float
    query_index: int


@dataclass(frozen=True)
class EvaluationContext:
    current_state: CyStateRecord
    current_objective: float
    actions: tuple[CanonicalAction, ...]
    goal: str
    rng: np.random.Generator
    remaining_budget: int
    evaluate_action: Callable[[CanonicalAction], EvaluatedAction]


class EvaluationAlgorithm(Protocol):
    name: str

    def select_action(self, context: EvaluationContext) -> EvaluatedAction:
        """Return a candidate obtained from this context's evaluate_action.

        Every callback invocation costs one query, even on a cache hit. A round
        may exceed remaining_budget; the runner stops before the next round.
        """
        ...


@dataclass(frozen=True)
class SearchNode:
    """A geometry-free frontier entry, independent of engine cache eviction."""

    state: CyStateRecord
    objective: float
    query_index: int
    depth: int


class SearchContext(Protocol):
    initial: SearchNode
    goal: str

    @property
    def remaining_budget(self) -> int: ...

    def priority(self, node: SearchNode) -> tuple[float, int]:
        """Smaller priorities win; ties retain first discovery order."""
        ...

    def expand(self, node: SearchNode) -> Iterator[SearchNode]:
        """Yield newly discovered, evaluated neighbors in canonical order.

        Fully consume each expansion before checking the budget or expanding
        another parent. One parent may exceed the remaining query budget.
        """
        ...


@runtime_checkable
class FrontierSearchAlgorithm(Protocol):
    name: str

    def search(self, context: SearchContext) -> None:
        """Search until the query budget or this algorithm's frontier is exhausted."""
        ...


@runtime_checkable
class PopulationAlgorithm(Protocol):
    name: str

    def run_population(self, context) -> str:
        """Evaluate arbitrary FRST candidates through the budgeted population context."""
        ...


Algorithm = EvaluationAlgorithm | FrontierSearchAlgorithm | RLAlgorithm | PopulationAlgorithm
