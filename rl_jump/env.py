"""Single-face jump environment over the shared CyoptEncoding codebook.

State: a DNA (one codebook index per interesting 2-face) and its decoded FRST. Action: one of the K
feasible single-face mutations drawn for that state. Reward: reward_fn(next) - reward_fn(current) with
reward_fn = reward_functions.get_reward(objective).from_objectives (log volume difference for
max_kcup). Geometry runs in-process (CPU), so the caller decides the process layout (see vec.py).

Candidate proposals are a pure function of (seed, DNA): revisiting a state gives the same candidates,
so a cached objective is reused on any path. Failures (non-solid cone, tip solver, timeout, reward
error) mark a candidate infeasible or the step failed; they never raise out of step().
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import signal
import threading
import time
from types import SimpleNamespace

import numpy as np

from eval.algorithm.cyopt_ga import CyoptEncoding
from mdp.cy_geometry_worker import _triangulate, execute_geometry_request
from reward_functions import get_objective, get_reward, infer_goal
from rl_jump.obs import JumpObservation, timed_build

DEFAULT_OBJECTIVE = "max_kcup"


class GeometryFailure(Exception):
    """A candidate or state whose objective or reward could not be computed."""


class GeometryTimeout(Exception):
    """A geometry call exceeded the environment's time limit."""


@contextmanager
def _time_limit(seconds):
    """SIGALRM-based limit; active only on the main thread (each env process runs its env there)."""
    usable = (seconds and seconds > 0 and hasattr(signal, "SIGALRM")
              and threading.current_thread() is threading.main_thread())
    if not usable:
        yield
        return

    def _raise(signum, frame):
        raise GeometryTimeout(f"geometry call exceeded {seconds} s")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _reason(exc):
    return f"{type(exc).__name__}: {exc}".replace("\n", " ")[:300]


@dataclass(frozen=True)
class Candidate:
    dna: tuple
    position: int  # DNA position (interesting-face index)
    index: int  # new codebook index on that face
    face: int  # codebook face index (polytope.faces(2) order)
    face_labels: tuple  # sorted ambient labels of that 2-face
    new_triangles: tuple  # the face's new triangulation in ambient labels
    symdiff: int  # triangle symmetric difference to the current entry (4 = flip)
    state: object = field(repr=False, compare=False)  # decoded CyStateRecord (FRST)


class JumpEnv:
    def __init__(self, polytope_row, *, objective=DEFAULT_OBJECTIVE, K=32, near_fraction=0.5, near_k=8,
                 seed=0, horizon=20, codebook_seed=0, max_points=12, samples=1000, geometry_timeout=300.0):
        if K < 1 or horizon < 1 or near_k < 1:
            raise ValueError("K, horizon and near_k must be positive.")
        if not 0.0 <= near_fraction <= 1.0:
            raise ValueError("near_fraction must lie in [0, 1].")
        self.objective_name = objective
        self.reward = get_reward(objective)
        self.metric = get_objective(objective, reward=self.reward)
        self.goal = infer_goal(objective)
        self.K, self.near_fraction, self.near_k = int(K), float(near_fraction), int(near_k)
        self.seed, self.horizon = int(seed), int(horizon)
        self.geometry_timeout = geometry_timeout

        started = time.perf_counter()
        built = execute_geometry_request({
            "operation": "build_collection", "row": polytope_row, "neighbor_mode": "two_neighbors",
            "include_points_interior_to_facets": False, "two_face_state": False,
            "include_two_face_metadata": True,
        })
        self.configuration = built["configuration"]
        self.starts = tuple(built["base_states"])
        if not self.starts:
            raise ValueError("The polytope row has no FRST starts.")
        # Same codebook as the GA evaluation: all starts of the polytope, eval defaults for sampling.
        self.encoding = CyoptEncoding(list(self.starts), seed=codebook_seed, max_points=max_points,
                                      samples=samples)
        self.polytope = self.encoding.polytope
        self.codebook = self.encoding.metadata["face_triangulations"]
        self.face_labels = tuple(tuple(sorted(int(label) for label in face.labels))
                                 for face in self.polytope.faces(2))
        if len(self.face_labels) != len(self.codebook):
            raise ValueError("Codebook faces do not match the polytope's 2-faces.")
        self.faces = tuple(f for f, entries in enumerate(self.codebook) if len(entries) > 1)
        if len(self.faces) != len(self.encoding.bounds):
            raise ValueError("Codebook interesting faces do not match the DNA bounds.")
        for position, face in enumerate(self.faces):
            lo, hi = self.encoding.bounds[position]
            if lo != 0 or hi != len(self.codebook[face]) - 1:
                raise ValueError("DNA bounds are not the full codebook range of their face.")
        self._entry_sets = {face: [frozenset(entry) for entry in self.codebook[face]] for face in self.faces}
        self.setup_seconds = time.perf_counter() - started

        self._decode_cache = {}  # dna -> (state or None, reason or None)
        self._objective_cache = {}  # dna -> (value or None, reason or None)
        self.counters = Counter()
        self.counters.update(dict.fromkeys(("queries", "objective_requests", "objective_cache_hits",
                                            "objective_failures", "decode_calls", "decode_cache_hits",
                                            "decode_infeasible", "drawn", "infeasible", "steps",
                                            "step_failures", "obs_builds"), 0))
        self.counters.update(dict.fromkeys(("objective_sec", "decode_sec", "obs_sec"), 0.0))
        self.log = []
        self.dna = None
        self.state = None
        self.value = None
        self.candidates = []
        self.visited = set()
        self.t = 0
        self.done = True
        self.episode_best = None

    # ------------------------------------------------------------------ public API

    def reset(self, start_index=0):
        if not 0 <= start_index < len(self.starts):
            raise IndexError(f"start_index {start_index} outside 0..{len(self.starts) - 1}.")
        start = self.starts[start_index]
        dna = tuple(int(x) for x in self.encoding.initial_dna[start.key])
        try:
            value = self._objective(dna, start)
        except GeometryFailure as exc:
            raise RuntimeError(f"The start objective failed: {exc}") from exc
        self.dna, self.state, self.value = dna, start, value
        self.episode_best = value
        self.visited = {dna}
        self.t = 0
        self.log = []
        obs = self._enter()
        self.done = not self.candidates
        return obs

    def step(self, action):
        if self.done:
            raise RuntimeError("The episode is finished; call reset().")
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < len(self.candidates):
            raise IndexError(f"action {action} is not one of {len(self.candidates)} candidates.")
        action = int(action)
        cand = self.candidates[action]
        before = (self.counters["queries"], self.counters["objective_cache_hits"],
                  self.counters["objective_sec"])
        self.t += 1
        self.counters["steps"] += 1
        info = dict(t=self.t, action=action, chosen_dna=list(cand.dna), symdiff=cand.symdiff,
                    flip_distance=cand.symdiff / 2, revisit=cand.dna in self.visited, failure=None)
        try:
            value_next = self._objective(cand.dna, cand.state)
            reward = float(self.reward.from_objectives(self.value, value_next))
        except (GeometryFailure, GeometryTimeout, ValueError, RuntimeError, ArithmeticError) as exc:
            # The candidate is dropped from this state; the state stays and the step still counts.
            reward, value_next = 0.0, None
            info["failure"] = _reason(exc)
            self.counters["step_failures"] += 1
            del self.candidates[action]
        if value_next is not None:
            self.dna, self.state, self.value = cand.dna, cand.state, value_next
            self.visited.add(cand.dna)
            if (value_next > self.episode_best) if self.goal == "max" else (value_next < self.episode_best):
                self.episode_best = value_next
            obs = self._enter()
        else:
            obs = self._observe(drawn=0, infeasible=0, reasons={}, decode_sec=0.0, decode_calls=0, decode_hits=0)
        self.done = self.t >= self.horizon or not self.candidates
        info.update(
            reward=reward, value=self.value, episode_best=self.episode_best, moved=value_next is not None,
            objective_queries=self.counters["queries"] - before[0],
            objective_cache_hit=self.counters["objective_cache_hits"] > before[1],
            objective_sec=self.counters["objective_sec"] - before[2],
            done_reason=("horizon" if self.t >= self.horizon else "dead_end") if self.done else None,
        )
        self.log.append(info)
        return obs, reward, self.done, info

    def stats(self):
        out = dict(self.counters)
        out["setup_seconds"] = self.setup_seconds
        out["codebook_faces"] = len(self.codebook)
        out["dna_positions"] = len(self.faces)
        return out

    # ------------------------------------------------------------------ candidates

    def _seed_rng(self, dna):
        return np.random.default_rng(np.random.SeedSequence([self.seed, *dna]))

    def _near_index(self, face, value, rng):
        """Uniform among the near_k codebook entries closest in triangle symmetric difference (ties random)."""
        entries = self._entry_sets[face]
        current = entries[value]
        count = len(entries)
        distance = np.fromiter((len(entry ^ current) for entry in entries), dtype=np.int64, count=count)
        distance[value] = np.iinfo(np.int64).max  # excluded: the current entry
        order = np.lexsort((rng.random(count), distance))
        pool = min(self.near_k, count - 1)
        return int(order[int(rng.integers(pool))])

    def _draw(self, dna):
        """Up to K distinct (position, new_index, child_dna) proposals; the first near_fraction*K are near."""
        if not self.faces:
            return []
        rng = self._seed_rng(dna)
        n_near = int(round(self.K * self.near_fraction))
        seen, out = {dna}, []
        positions = len(self.faces)
        attempts, limit = 0, 20 * self.K
        while len(out) < self.K and attempts < limit:
            attempts += 1
            position = int(rng.integers(positions))
            if len(out) < n_near:
                new = self._near_index(self.faces[position], dna[position], rng)
            else:
                lo, hi = self.encoding.bounds[position]
                new = int(rng.integers(lo, hi))  # [lo, hi-1] ...
                if new >= dna[position]:
                    new += 1  # ... shifted to the other entries of [lo, hi]
            child = dna[:position] + (new,) + dna[position + 1:]
            if child in seen:
                continue
            seen.add(child)
            out.append((position, new, child))
        if len(out) < self.K:  # small neighbourhood: complete it from the full enumeration
            pool = [(p, n) for p, (lo, hi) in enumerate(self.encoding.bounds)
                    for n in range(lo, hi + 1) if n != dna[p]]
            for index in rng.permutation(len(pool)):
                if len(out) >= self.K:
                    break
                position, new = pool[index]
                child = dna[:position] + (new,) + dna[position + 1:]
                if child not in seen:
                    seen.add(child)
                    out.append((position, new, child))
        return out

    def _decode(self, dna):
        self.counters["decode_calls"] += 1
        cached = self._decode_cache.get(dna)
        if cached is not None:
            self.counters["decode_cache_hits"] += 1
            return cached
        started = time.perf_counter()
        try:
            with _time_limit(self.geometry_timeout):
                state = self.encoding.decode(dna, self.starts[0])
            result = (state, None) if state is not None else (None, "non_solid_cone")
        except (GeometryTimeout, ValueError, RuntimeError, ArithmeticError, IndexError, KeyError) as exc:
            result = (None, _reason(exc))
        self.counters["decode_sec"] += time.perf_counter() - started
        if result[0] is None:
            self.counters["decode_infeasible"] += 1
        self._decode_cache[dna] = result
        return result

    def _enumerate(self, dna):
        proposals = self._draw(dna)
        candidates, reasons = [], Counter()
        for position, new, child in proposals:
            state, reason = self._decode(child)
            if state is None:
                reasons[reason] += 1
                continue
            face = self.faces[position]
            symdiff = len(self._entry_sets[face][dna[position]] ^ self._entry_sets[face][new])
            candidates.append(Candidate(child, position, new, face, self.face_labels[face],
                                        self.codebook[face][new], symdiff, state))
        self.counters["drawn"] += len(proposals)
        self.counters["infeasible"] += len(proposals) - len(candidates)
        return candidates, len(proposals), reasons

    def _enter(self):
        """Enumerate candidates for the current state and build its observation."""
        calls, hits, dsec = (self.counters["decode_calls"], self.counters["decode_cache_hits"],
                             self.counters["decode_sec"])
        self.candidates, drawn, reasons = self._enumerate(self.dna)
        return self._observe(drawn=drawn, infeasible=drawn - len(self.candidates), reasons=dict(reasons),
                             decode_sec=self.counters["decode_sec"] - dsec,
                             decode_calls=self.counters["decode_calls"] - calls,
                             decode_hits=self.counters["decode_cache_hits"] - hits)

    def _observe(self, *, drawn, infeasible, reasons, decode_sec, decode_calls, decode_hits):
        data, obs_sec = timed_build(self.state, self.candidates)
        self.counters["obs_builds"] += 1
        self.counters["obs_sec"] += obs_sec
        stats = dict(drawn=drawn, infeasible=infeasible, infeasible_reasons=reasons,
                     n_candidates=len(self.candidates), candidate_revisits=sum(
                         1 for c in self.candidates if c.dna in self.visited),
                     decode_sec=decode_sec, decode_calls=decode_calls, decode_cache_hits=decode_hits,
                     obs_sec=obs_sec)
        meta = [dict(dna=list(c.dna), position=c.position, index=c.index, face=c.face, symdiff=c.symdiff)
                for c in self.candidates]
        return JumpObservation(data=data, dna=self.dna, value=self.value, t=self.t, candidates=meta,
                               stats=stats)

    # ------------------------------------------------------------------ objective

    def _objective(self, dna, state):
        """Objective of the FRST `state` (DNA `dna`): cached by DNA, every real computation is one query."""
        self.counters["objective_requests"] += 1
        if dna in self._objective_cache:
            self.counters["objective_cache_hits"] += 1
            value, reason = self._objective_cache[dna]
        else:
            started = time.perf_counter()
            self.counters["queries"] += 1
            value, reason = None, None
            try:
                with _time_limit(self.geometry_timeout):
                    triangulation = _triangulate(self.polytope, self.configuration, state.simplices)
                    value = float(self.metric(SimpleNamespace(
                        key=state.key, simplices=state.simplices, cy_triangulation=triangulation)))
                if not math.isfinite(value):
                    raise ValueError("nonfinite objective")
            except (GeometryTimeout, ValueError, RuntimeError, ArithmeticError, IndexError, KeyError) as exc:
                value, reason = None, _reason(exc)
            self.counters["objective_sec"] += time.perf_counter() - started
            if reason is not None:
                self.counters["objective_failures"] += 1
            self._objective_cache[dna] = (value, reason)
        if reason is not None:
            raise GeometryFailure(reason)
        return value
