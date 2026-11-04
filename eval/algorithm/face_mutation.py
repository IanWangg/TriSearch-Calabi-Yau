"""No-learning single-face mutation search over the shared 2-face codebook.

Uses the same CyoptEncoding, start DNA and logical query accounting as cyopt_ga.
A DNA has one position per "interesting" 2-face (codebook size > 1); a mutation
replaces the index at one position with another codebook entry of that face.
Only objective queries are budgeted; non-solid-cone children are free rejects.
Each child is logged with its face, flip distance and log-volume change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
import math

import numpy as np

from eval.algorithm.cyopt_ga import CyoptEncoding


def _better(goal: str, a: float, b: float) -> bool:
    return a > b if goal == "max" else a < b


class _MutationRun:
    """Per-start state: known DNAs (evaluated or rejected) are never queried again."""

    def __init__(self, algorithm, context):
        self.algorithm = algorithm
        self.context = context
        self.encoding: CyoptEncoding = algorithm.encoding
        self.rng = np.random.default_rng(context.seed)
        codebook = self.encoding.metadata["face_triangulations"]
        self.codebook = codebook
        self.point_counts = self.encoding.metadata["face_point_counts"]
        # DNA position i indexes the i-th face whose codebook has more than one entry.
        self.faces = [f for f, entries in enumerate(codebook) if len(entries) > 1]
        self.bounds = self.encoding.bounds
        if len(self.faces) != len(self.bounds):
            raise ValueError("Codebook interesting faces do not match the DNA bounds.")
        self.known: set[tuple[int, ...]] = set()
        self.generation = 0

    def start_dna(self):
        initial = self.context.initial.state
        return tuple(int(x) for x in self.encoding.initial_dna[initial.key])

    def draw(self, parent, count):
        """Up to `count` distinct unseen single-position mutations of `parent`.

        Position is uniform among positions with >1 entries; the new index is
        uniform among that position's other entries. Falls back to exhaustive
        enumeration so that a nearly saturated neighborhood is still complete.
        """
        chosen, local = [], set()
        if not self.bounds:
            return chosen

        def add(position, new):
            child = list(parent)
            child[position] = new
            key = tuple(child)
            if key in self.known or key in local:
                return
            local.add(key)
            chosen.append((position, new, key))

        attempts, limit = 0, 50 * count
        while len(chosen) < count and attempts < limit:
            attempts += 1
            position = int(self.rng.integers(len(self.bounds)))
            lo, hi = self.bounds[position]
            new = int(self.rng.integers(lo, hi))  # [lo, hi-1]
            if new >= parent[position]:
                new += 1  # uniform over the other entries
            add(position, new)
        if len(chosen) < count:
            pool = [(p, n) for p, (lo, hi) in enumerate(self.bounds)
                    for n in range(lo, hi + 1) if n != parent[p]]
            order = self.rng.permutation(len(pool))
            for index in order:
                if len(chosen) >= count:
                    break
                add(*pool[index])
        return chosen

    def record(self, parent, parent_value, position, new, key, feasible, value, generation, extra):
        face = self.faces[position]
        symdiff = len(set(self.codebook[face][parent[position]]) ^ set(self.codebook[face][new]))
        dlnv = None
        if feasible and value is not None and parent_value and value > 0 and parent_value > 0:
            dlnv = math.log(value) - math.log(parent_value)
        self.algorithm.candidate_log.append(dict(
            algorithm=self.algorithm.name, polytope_index=self.context.initial.state.point_config_index,
            start_hash=hashlib.sha1(self.context.initial.state.key.encode()).hexdigest()[:12],
            generation_index=generation,
            parent_dna=list(parent), child_dna=list(key), dna_position=position, face=face,
            face_point_count=self.point_counts[face], parent_index=parent[position], child_index=new,
            face_symdiff=symdiff, flip_distance=symdiff / 2, feasible=feasible,
            objective=value, parent_objective=parent_value, dlnV=dlnv, **extra))

    def evaluate(self, parent, parent_value, children, generation, extra):
        """Evaluate children in draw order; returns the list of (key, value) feasible ones."""
        feasible_children = []
        for position, new, key in children:
            self.known.add(key)
            candidate = self.context.materialize_candidate(self.encoding, key)
            if candidate is None:
                self.context.reject(key, "non_solid_cone")
                self.record(parent, parent_value, position, new, key, False, None, generation, extra)
                continue
            value = self.context.evaluate_candidate(candidate, dna=key)
            self.record(parent, parent_value, position, new, key, True, value, generation, extra)
            feasible_children.append((key, value, position, new))
        return feasible_children


@dataclass
class FaceMutationHillClimb:
    """Move to the best of M single-face children when it improves the objective."""

    mutations_per_parent: int = 16
    encoding: CyoptEncoding | None = field(default=None, repr=False)
    candidate_log: list = field(default_factory=list, repr=False)
    name: str = field(default="face_mutation_hc", init=False)

    def run_population(self, context) -> str:
        from eval.rollout import ObjectiveBudgetExhausted

        if context.remaining_budget == 0:
            return "budget_exhausted"
        if self.encoding is None:
            raise RuntimeError("face_mutation_hc requires a prepared CyoptEncoding.")
        run = _MutationRun(self, context)
        current = run.start_dna()
        current_value = context.initial.objective
        run.known.add(current)
        try:
            while context.remaining_budget:
                children = run.draw(current, self.mutations_per_parent)
                if not children:
                    return "neighborhood_exhausted"
                with context.generation(run.generation, mutations_drawn=len(children)):
                    evaluated = run.evaluate(current, current_value, children, run.generation,
                                             extra=dict(mode="hc"))
                    best = None
                    for key, value, _, _ in evaluated:
                        if best is None or _better(context.goal, value, best[1]):
                            best = (key, value)
                    moved = best is not None and _better(context.goal, best[1], current_value)
                    if moved:
                        for entry in self.candidate_log[-len(children):]:
                            if entry["child_dna"] == list(best[0]):
                                entry["moved"] = True
                        current, current_value = best
                run.generation += 1
        except ObjectiveBudgetExhausted:
            pass
        return "budget_exhausted"


@dataclass
class FaceMutationBeFS:
    """Best-first over DNAs: pop the best, expand M single-face children, push feasible ones."""

    mutations_per_parent: int = 16
    encoding: CyoptEncoding | None = field(default=None, repr=False)
    candidate_log: list = field(default_factory=list, repr=False)
    name: str = field(default="face_mutation_befs", init=False)

    def run_population(self, context) -> str:
        from eval.rollout import ObjectiveBudgetExhausted

        if context.remaining_budget == 0:
            return "budget_exhausted"
        if self.encoding is None:
            raise RuntimeError("face_mutation_befs requires a prepared CyoptEncoding.")
        run = _MutationRun(self, context)
        start = run.start_dna()
        run.known.add(start)
        sign = -1.0 if context.goal == "max" else 1.0
        counter = 0
        frontier = [(sign * context.initial.objective, counter, start, context.initial.objective)]
        try:
            while context.remaining_budget:
                if not frontier:
                    return "frontier_exhausted"
                _, _, parent, parent_value = heapq.heappop(frontier)
                children = run.draw(parent, self.mutations_per_parent)
                if not children:
                    continue  # A dead parent costs nothing and is not an expansion.
                with context.generation(run.generation, mutations_drawn=len(children)):
                    for key, value, _, _ in run.evaluate(parent, parent_value, children, run.generation,
                                                         extra=dict(mode="befs")):
                        counter += 1
                        heapq.heappush(frontier, (sign * value, counter, key, value))
                run.generation += 1
        except ObjectiveBudgetExhausted:
            pass
        return "budget_exhausted"
