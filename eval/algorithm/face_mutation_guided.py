"""Best-first over DNAs with a learned single-face move scorer, plus a random-selection control.

Each expansion draws K = pool_size distinct unseen single-face mutations of the parent with the same rule as
FaceMutationBeFS, and scores all of them with FaceMoveScorer (no objective query, no feasibility check).
"score" keeps the M highest-scoring moves and evaluates them in score order; "random" keeps M uniformly at
random (same draw, same scoring for logging) and evaluates them in draw order. Kept children are then handled
exactly as in FaceMutationBeFS: non-solid-cone children are free rejects, feasible children cost one query.
Only kept (evaluated or rejected) children are logged; the other K - M draws cost nothing and are not logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq

import numpy as np

from eval.algorithm.cyopt_ga import CyoptEncoding
from eval.algorithm.face_move_scorer import DEFAULT_CHECKPOINT, FaceMoveScorer
from eval.algorithm.face_mutation import _MutationRun

GUIDED_NAME = "face_mutation_befs_guided"
RANDOM_NAME = "face_mutation_befs_random64"
SCORED_ALGORITHMS = (GUIDED_NAME, RANDOM_NAME)


@dataclass
class FaceMutationGuidedBeFS:
    selection: str = "score"
    pool_size: int = 64
    mutations_per_parent: int = 16
    aggregation: str = "mean2"
    checkpoint: str = str(DEFAULT_CHECKPOINT)
    encoding: CyoptEncoding | None = field(default=None, repr=False)
    candidate_log: list = field(default_factory=list, repr=False)
    scorer_log: list = field(default_factory=list, repr=False)
    scorer: FaceMoveScorer | None = field(default=None, repr=False)
    name: str = field(default="", init=False)

    def __post_init__(self):
        if self.selection not in ("score", "random"):
            raise ValueError("selection must be 'score' or 'random'.")
        if self.aggregation not in ("mean2", "sum"):
            raise ValueError("aggregation must be 'mean2' or 'sum'.")
        if not 0 < self.mutations_per_parent <= self.pool_size:
            raise ValueError("Need 0 < mutations_per_parent <= pool_size.")
        self.name = GUIDED_NAME if self.selection == "score" else RANDOM_NAME

    def _face_label_sets(self, run):
        faces = self.encoding.polytope.faces(2)
        if len(faces) != len(run.codebook):
            raise ValueError("Codebook faces do not match the polytope's 2-faces.")
        return [tuple(sorted(int(label) for label in face.labels)) for face in faces]

    def _score(self, run, parent, pool, face_labels):
        moves = [(face_labels[run.faces[position]], run.codebook[run.faces[position]][new])
                 for position, new, _ in pool]
        scores = self.scorer.score(run.states[parent], moves)
        self.scorer_log.append(dict(algorithm=self.name, generation_index=run.generation, moves=len(moves),
                                    seconds=self.scorer.last_seconds, polytope_index=run.context.initial.state.point_config_index))
        return scores

    def run_population(self, context) -> str:
        from eval.rollout import ObjectiveBudgetExhausted

        if context.remaining_budget == 0:
            return "budget_exhausted"
        if self.encoding is None:
            raise RuntimeError(f"{self.name} requires a prepared CyoptEncoding.")
        if self.scorer is None:
            self.scorer = FaceMoveScorer(self.checkpoint, aggregation=self.aggregation)
        run = _MutationRun(self, context)
        face_labels = self._face_label_sets(run)
        start = run.start_dna()
        run.known.add(start)
        run.states[start] = context.initial.state
        sign = -1.0 if context.goal == "max" else 1.0
        counter = 0
        frontier = [(sign * context.initial.objective, counter, start, context.initial.objective)]
        try:
            while context.remaining_budget:
                if not frontier:
                    return "frontier_exhausted"
                _, _, parent, parent_value = heapq.heappop(frontier)
                pool = run.draw(parent, self.pool_size)
                if not pool:
                    continue  # A dead parent costs nothing and is not an expansion.
                with context.generation(run.generation, mutations_drawn=len(pool)):
                    scores = self._score(run, parent, pool, face_labels)
                    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
                    rank = {index: position + 1 for position, index in enumerate(order)}
                    if self.selection == "score":
                        kept = order[:self.mutations_per_parent]
                    else:
                        picked = run.rng.choice(len(pool), size=min(self.mutations_per_parent, len(pool)),
                                                replace=False)
                        kept = sorted(int(i) for i in picked)
                    children = [pool[i] for i in kept]
                    extras = [dict(selection=self.selection, aggregation=self.aggregation,
                                   score=float(scores[i]), score_rank=rank[i], pool_size=len(pool),
                                   kept=len(kept)) for i in kept]
                    for key, value, _, _ in run.evaluate(parent, parent_value, children, run.generation,
                                                         extra=dict(mode=self.name), extras=extras):
                        counter += 1
                        heapq.heappush(frontier, (sign * value, counter, key, value))
                run.generation += 1
        except ObjectiveBudgetExhausted:
            pass
        return "budget_exhausted"
