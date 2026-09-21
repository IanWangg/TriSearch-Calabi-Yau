"""Geometry-free descriptions shared by the trainer and geometry workers.

Keep this module limited to the standard library: importing a rollout descriptor
must not initialize Sage, CYTools, NumPy, or Torch in the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterable

from core.cy_bounded_cache import BoundedLRU, retained_size

NEIGHBOR_MODES = ("regular", "two_neighbors")


def canonical_simplices(simplices: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(int(vertex) for vertex in simplex)) for simplex in simplices))


def normalize_neighbor_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in NEIGHBOR_MODES:
        raise ValueError(f"Unknown neighbor_mode '{mode}'. Expected one of: regular, two_neighbors.")
    return mode


def state_key(index: int, simplices: Iterable[Iterable[int]], neighbor_mode: str = "regular") -> str:
    key = f"{index}:{canonical_simplices(simplices)}"
    return key if neighbor_mode == "regular" else f"{neighbor_mode}|{key}"


@dataclass(frozen=True)
class CYPointConfiguration:
    index: int
    input_vertices: tuple[tuple[int, ...], ...]
    points: tuple[tuple[int, ...], ...]
    labels: tuple[int, ...]
    include_points_interior_to_facets: bool


@dataclass
class CyStateRecord:
    configuration: CYPointConfiguration
    simplices: frozenset[tuple[int, ...]]
    neighbor_mode: str = "regular"
    is_frst: bool = False
    is_target: bool = False
    visitation: int = 0
    key: str = field(init=False)
    _edges: frozenset[tuple[int, ...]] | None = field(default=None, init=False, repr=False)
    _objective_provider: Callable[["CyStateRecord", str], float] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.neighbor_mode = normalize_neighbor_mode(self.neighbor_mode)
        self.simplices = frozenset(canonical_simplices(self.simplices))
        self.key = state_key(self.point_config_index, self.simplices, self.neighbor_mode)

    @property
    def point_config_index(self) -> int:
        return self.configuration.index

    @property
    def vertices(self) -> tuple[tuple[int, ...], ...]:
        return self.configuration.input_vertices

    @property
    def edges(self) -> frozenset[tuple[int, ...]]:
        if self._edges is None:
            self._edges = frozenset(edge for simplex in self.simplices for edge in combinations(simplex, 2))
        return self._edges

    @property
    def is_frt(self) -> bool:
        return self.is_target

    @property
    def reward(self) -> int:
        return int(self.is_frst)

    @property
    def terminal(self) -> bool:
        return self.is_target or len(self.simplices) <= 1

    def to_payload(self) -> tuple:
        """An immutable request with no runtime provider or duplicate coordinates."""
        return (self.point_config_index, canonical_simplices(self.simplices), self.neighbor_mode,
                bool(self.is_frst), bool(self.is_target))

    def bind_objective_provider(self, provider: Callable[["CyStateRecord", str], float] | None) -> None:
        self._objective_provider = provider

    def objective_value(self, name: str) -> float:
        if self._objective_provider is None:
            raise RuntimeError("This state is not attached to a managed geometry engine.")
        return float(self._objective_provider(self, str(name)))

    def __getstate__(self) -> dict:
        # Bound methods can otherwise pickle the entire engine, graph and pool.
        state = self.__dict__.copy()
        state["_objective_provider"] = None
        return state


# Accommodate callers that follow the existing all-capital CY class prefix.
CYStateRecord = CyStateRecord


class BoundedCache(BoundedLRU):
    """Worker cache defaults over the shared insertion-bounded LRU."""

    def __init__(self, *, max_entries: int = 2048, max_bytes: int = 128 * 1024**2):
        super().__init__(max_bytes=max_bytes, max_entries=max_entries)
