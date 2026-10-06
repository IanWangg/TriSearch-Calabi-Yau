"""Use upstream cyopt.GA with shared FRST starts and logical query accounting.

Only initialization and the fitness callback are adapted. Selection, crossover,
mutation, elitism and within-generation DNA uniqueness remain upstream code.
Imports are lazy so other algorithms do not require the optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import unquote, urlparse

from mdp.cy_state_record import CyStateRecord, canonical_simplices


def cyopt_metadata():
    package = distribution("cyopt")
    digest = hashlib.sha256()
    for entry in sorted(package.files or [], key=str):
        if str(entry).startswith("cyopt/") and str(entry).endswith(".py"):
            digest.update(str(entry).encode())
            digest.update(package.locate_file(entry).read_bytes())
    origin = json.loads(package.read_text("direct_url.json") or "{}")
    metadata = dict(version=package.version, installed_source_sha256=digest.hexdigest(), origin=origin)
    url = urlparse(origin.get("url", ""))
    if url.scheme == "file":
        source = Path(unquote(url.path))
        if source.is_dir():
            for key, args in (("source_commit", ["rev-parse", "HEAD"]),
                              ("source_status", ["status", "--porcelain"])):
                result = subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True)
                metadata[key] = result.stdout.strip() if result.returncode == 0 else None
    return metadata


def _label_face_triangulations(face, triangulations):
    """Restore ambient labels after grow_frt creates a local 2D polytope."""
    points = dict(zip((tuple(point) for point in face.points()), face.labels))
    optimal_points = dict(zip((tuple(point) for point in face.points(optimal=True)), face.labels))
    result = []
    for triang in triangulations:
        coordinates = [tuple(point) for point in triang.points()]
        lookup = points if set(coordinates) == set(points) else optimal_points
        if set(coordinates) != set(lookup):
            raise ValueError("A sampled triangulation does not contain the requested face points.")
        labels = {int(label): int(lookup[point]) for label, point in zip(triang.labels, coordinates)}
        if any(label != ambient for label, ambient in labels.items()):
            triang = face.as_poly().triangulate(
                simplices=[[labels[int(label)] for label in simplex] for simplex in triang.simplices()],
                include_points_interior_to_facets=True, make_star=False, check_input_simplices=False,
            )
        result.append(triang)
    return result


class CyoptEncoding:
    """One sorted, audited 2-face codebook shared by this polytope's starts."""

    def __init__(self, starts, *, seed: int, max_points: int, samples: int):
        import cyopt.frst  # noqa: F401; installs the upstream encoding methods
        from mdp.cy_geometry_worker import _get_polytope, _triangulate

        started = time.perf_counter()
        self.configuration = starts[0].configuration
        if any(state.configuration != self.configuration for state in starts):
            raise ValueError("A cyopt codebook must contain exactly one point configuration.")
        self.polytope = _get_polytope(self.configuration)
        # _get_polytope may retain a prepped object from an earlier evaluation.
        # Rebuild the codebook for this run's declared sampling parameters.
        self.polytope._cyopt_prepped = False
        faces = self.polytope.face_triangs(max_npts=max_points, N_face_triangs=samples, seed=seed)
        faces = [_label_face_triangulations(face, triangulations)
                 for face, triangulations in zip(self.polytope.faces(2), faces)]
        face_maps = [{canonical_simplices(tri.simplices()): tri for tri in face} for face in faces]
        initial_triangs = [_triangulate(self.polytope, state.configuration, canonical_simplices(state.simplices))
                          for state in starts]
        # Sampling large faces must never make a supplied start unrepresentable.
        # Include restrictions from inputs only, before querying any objectives.
        for triang in initial_triangs:
            for index, restriction in enumerate(triang.restrict(as_poly=True)):
                face_maps[index].setdefault(canonical_simplices(restriction.simplices()), restriction)
        faces = [[mapping[key] for key in sorted(mapping)] for mapping in face_maps]
        self.polytope.prep_for_optimizers(face_triangs=faces)
        self.bounds = self.polytope._cyopt_bounds
        self.initial_dna = {state.key: self.polytope.triang_to_dna(triang)
                            for state, triang in zip(starts, initial_triangs)}
        codebook = [sorted(mapping) for mapping in face_maps]
        self.metadata = dict(
            format_version=1, polytope_index=self.configuration.index, seed=seed,
            max_points=max_points, samples=samples, bounds=self.bounds,
            face_label_mapping="ambient_labels_by_coordinates",
            face_point_counts=[len(face.points()) for face in self.polytope.faces(2)],
            face_triangulations=codebook,
            codebook_sha256=hashlib.sha256(json.dumps(codebook, separators=(",", ":")).encode()).hexdigest(),
            starts=[dict(state_key=state.key, dna=self.initial_dna[state.key]) for state in starts],
            preparation_wall_sec=time.perf_counter() - started,
        )

    def decode(self, dna, initial):
        from core.cytools_config import REGULARITY_BACKEND

        if dna == self.initial_dna[initial.key]:
            return initial  # Preserve the exact supplied full FRST for this DNA.
        triang = self.polytope.dna_to_frst(dna)
        if triang is None:
            return None  # Upstream's explicit non-solid-cone result, not an exception.
        if not (triang.is_fine() and triang.is_star() and triang.is_regular(backend=REGULARITY_BACKEND)):
            raise ValueError("cyopt decoded a candidate that is not an FRST.")
        if self.polytope.triang_to_dna(triang) != dna:
            raise ValueError("cyopt DNA round trip changed the requested face triangulations.")
        # simplices() returns polytope labels, matching the shared worker protocol.
        return CyStateRecord(self.configuration, frozenset(canonical_simplices(triang.simplices())),
                             initial.neighbor_mode, True, True)


@dataclass
class CyoptGAAlgorithm:
    population_size: int = 50
    mutation_rate: float = 0.1
    elitism: int = 1
    max_stalled_generations: int = 20
    encoding: CyoptEncoding | None = field(default=None, repr=False)
    name: str = field(default="cyopt_ga", init=False)

    def run_population(self, context) -> str:
        from cyopt import GA, TupleSpace
        from eval.rollout import ObjectiveBudgetExhausted

        if context.remaining_budget == 0:
            return "budget_exhausted"
        if self.encoding is None:
            raise RuntimeError("cyopt_ga requires a prepared CyoptEncoding.")
        initial_dna = self.encoding.initial_dna[context.initial.state.key]
        if not self.encoding.bounds:
            return "dna_space_singleton"
        size = 1
        for lo, hi in self.encoding.bounds:
            size *= hi - lo + 1
        # Upstream can shrink populations below 4 after the first generation.
        # Keep its constructor minimum while preventing an oversized elite slice.
        effective_elitism = min(self.elitism, size - 1)

        class SeededSpace(TupleSpace):
            first = True

            def random(self, rng):
                if self.first:
                    self.first = False
                    return initial_dna
                return super().random(rng)

        first_fitness = True

        def fitness(dna):
            nonlocal first_fitness
            if first_fitness:
                first_fitness = False
                if dna != initial_dna:
                    raise RuntimeError("cyopt did not seed its first population member from the shared start.")
                value = context.initial.objective  # Known q=0 value; no second query.
            else:
                if context.remaining_budget == 0:
                    raise ObjectiveBudgetExhausted
                candidate = context.materialize_candidate(self.encoding, dna)
                if candidate is None:
                    context.reject(dna, "non_solid_cone")
                    return float("inf")  # GA fitness only, never an objective/logged volume.
                value = context.evaluate_candidate(candidate, dna=dna)
            return -value if context.goal == "max" else value

        optimizer = GA(fitness_fn=fitness, space=SeededSpace(self.encoding.bounds),
                       population_size=self.population_size, mutation_rate=self.mutation_rate,
                       elitism=effective_elitism, seed=context.seed, cache_size=0,
                       selection="tournament", crossover="npoint", mutation_k=1)
        generation, stalled = 0, 0
        try:
            while context.remaining_budget:
                before = context.remaining_budget
                with context.generation(generation, effective_elitism=effective_elitism):
                    # run(0) performs upstream population initialization only.
                    optimizer.run(0 if generation == 0 else 1)
                stalled = stalled + 1 if context.remaining_budget == before else 0
                if stalled >= self.max_stalled_generations:
                    return "no_feasible_offspring"
                generation += 1
        except ObjectiveBudgetExhausted:
            pass
        return "budget_exhausted"
