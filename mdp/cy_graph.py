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
    next_simplices: CanonicalSimplices = ()
    next_is_target: Optional[bool] = None
    removed_simplices: CanonicalSimplices = ()
    added_simplices: CanonicalSimplices = ()
    next_is_frst: Optional[bool] = None

    def simplices_from(self, source) -> CanonicalSimplices:
        if self.removed_simplices or self.added_simplices:
            return tuple(sorted((set(source) - set(self.removed_simplices)) | set(self.added_simplices)))
        return self.next_simplices

    def num_next_simplices(self, source) -> int:
        if self.removed_simplices or self.added_simplices:
            return len(source) - len(self.removed_simplices) + len(self.added_simplices)
        return len(self.next_simplices)


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
    transition_pool: object = None
    owns_transition_pool: bool = False

    def close(self):
        if self.owns_transition_pool and self.transition_pool is not None:
            self.transition_pool.shutdown()
            self.transition_pool = None
            self.owns_transition_pool = False


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
    candidate_actions: list | None = None


@dataclass(frozen=True)
class ExpandSummary:
    expanded_count: int
    discovered_count: int
    used_multiprocessing: bool
