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


def two_face_restrictions(configuration: "CYPointConfiguration", simplices: Iterable[Iterable[int]]) -> tuple:
    """Return canonical immutable restrictions to the ambient 2-faces.

    Face labels come from CYTools in the worker. Intersecting each full simplex
    with these labels is the geometry-free equivalent of Triangulation.restrict.
    Keep faces separate: the full triangulation's arbitrary interior triangles
    are not part of this representation.
    """
    if not configuration.two_face_labels:
        raise ValueError("two_face_state requires ambient 2-face labels from the geometry worker.")
    simplices = tuple(frozenset(simplex) for simplex in simplices)
    restrictions = []
    for labels in canonical_simplices(configuration.two_face_labels):
        face = frozenset(labels)
        triangles = {tuple(sorted(intersection)) for simplex in simplices
                     if len(intersection := face.intersection(simplex)) == 3}
        if not triangles:
            raise ValueError(f"Missing triangulation restriction for 2-face {labels}.")
        restrictions.append((tuple(sorted(labels)), canonical_simplices(triangles)))
    return tuple(restrictions)


def two_face_state_key(configuration: "CYPointConfiguration", simplices: Iterable[Iterable[int]]) -> str:
    restrictions = two_face_restrictions(configuration, simplices)
    return f"two_face|{configuration.index}:{restrictions}"


def evaluation_state_key(state: "CyStateRecord", two_face_state: bool = False) -> str:
    return state.two_face_key if two_face_state else state.key


@dataclass(frozen=True)
class CYPointConfiguration:
    index: int
    input_vertices: tuple[tuple[int, ...], ...]
    points: tuple[tuple[int, ...], ...]
    labels: tuple[int, ...]
    include_points_interior_to_facets: bool
    two_face_labels: tuple[tuple[int, ...], ...] = ()


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
    _two_face_key: str | None = field(default=None, init=False, repr=False, compare=False)
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
    def two_face_key(self) -> str:
        if self._two_face_key is None:
            self._two_face_key = two_face_state_key(self.configuration, self.simplices)
        return self._two_face_key

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
