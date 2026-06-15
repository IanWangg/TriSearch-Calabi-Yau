from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Optional, Tuple


CanonicalSimplex = Tuple[int, ...]
CanonicalSimplices = Tuple[CanonicalSimplex, ...]
CanonicalAction = Tuple[int, ...]


def _sorted_simplices_tuple(simplices: Iterable[Iterable[int]]) -> CanonicalSimplices:
    return tuple(sorted(tuple(sorted(int(vertex) for vertex in simplex)) for simplex in simplices))


@dataclass(frozen=True)
class CYGraphTransition:
    next_key: str
    next_simplices: CanonicalSimplices
    next_is_target: Optional[bool] = None


@dataclass
class CYGraphNode:
    key: str
    point_config_index: int
    simplices: CanonicalSimplices
    candidate_actions: Tuple[CanonicalAction, ...] = ()
    ambiguous_actions: FrozenSet[CanonicalAction] = frozenset()
    transitions: Dict[CanonicalAction, CYGraphTransition] = field(default_factory=dict)
    expanded: bool = False


@dataclass(frozen=True)
class CYStateExpansion:
    key: str
    point_config_index: int
    simplices: CanonicalSimplices
    candidate_actions: Tuple[CanonicalAction, ...]
    ambiguous_actions: FrozenSet[CanonicalAction]
    transitions: Tuple[Tuple[CanonicalAction, CYGraphTransition], ...]


@dataclass
class CYRolloutCollection:
    base_states: Dict[str, object]
    initial_states: list
    polytope_by_index: Dict[int, object]
    vertices_by_polytope: Dict[int, list]
    polytope_indices: list


@dataclass
class RandomRolloutStepResult:
    input_states: list
    transitioned_states: list
    next_states: list
    rewards: list
    dones: list
    chosen_actions: list
    terminal_reasons: list
    reset_count: int
    frt_hits: int
    collapsed_hits: int
    dead_end_hits: int
    expanded_states: int
    discovered_states: int
    used_multiprocessing: bool


@dataclass(frozen=True)
class ExpandSummary:
    expanded_count: int
    discovered_count: int
    used_multiprocessing: bool
