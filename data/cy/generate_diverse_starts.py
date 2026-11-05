"""Generate FRST starts that are distinct up to 2-face equivalence (NTFE classes).

Two FRSTs with the same 2-face restriction give the same Calabi-Yau, so starts
are deduplicated by their 2-face triangulation. Input is a polytope JSONL from
`select_polytopes.py`; output rows load with `mdp.cy_rollout.load_cy_sample_rows`
(vertices = all lattice points, simplices in that index space, no non-fine lists).

Train starts (k per polytope): about half come from `random_triangulations_fast`
(the evaluation sampler). The rest are uniform over all NTFE classes when the
full enumeration finishes within the time limit, otherwise picked by
farthest-point selection from a pool of fast, fair, and two_neighbors random-walk
FRSTs (distance = number of 2-faces whose triangulation differs).
Dev/test starts come from `random_triangulations_fast` only.

```bash
python -m data.cy.generate_diverse_starts --polytopes data/cy/favorable_v1/train/polytopes_h11_15.jsonl \
  --output data/cy/favorable_v1/train/h11_15.samples.jsonl --starts 4 --min-starts 3 --mode train
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

TwoFaceParts = Tuple[Tuple[Tuple[int, ...], ...], ...]


def derive_seed(seed: int, *components: Any) -> int:
    encoded = json.dumps([int(seed), *components], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")


def two_face_parts(triangulation) -> TwoFaceParts:
    """Restriction of a triangulation to each 2-face, in the polytope's 2-face order."""
    parts = triangulation.simplices(on_faces_dim=2, split_by_face=True)
    return tuple(tuple(sorted(tuple(sorted(int(v) for v in simplex)) for simplex in face)) for face in parts)


def two_face_key(parts: TwoFaceParts) -> Tuple[Tuple[int, ...], ...]:
    return tuple(sorted(simplex for face in parts for simplex in face))


def two_face_hash(parts: TwoFaceParts) -> str:
    return hashlib.sha1(json.dumps(two_face_key(parts)).encode("utf-8")).hexdigest()[:16]


def two_face_distance(left: TwoFaceParts, right: TwoFaceParts) -> int:
    return sum(a != b for a, b in zip(left, right))


def farthest_point_order(candidates: Sequence[TwoFaceParts], chosen: Sequence[TwoFaceParts],
                         count: int, rng: np.random.Generator) -> List[int]:
    """Greedy max-min selection of `count` candidate indices, ties broken at random."""
    picked: List[int] = []
    anchors = list(chosen)
    remaining = list(range(len(candidates)))
    while remaining and len(picked) < count:
        if anchors:
            scores = [min(two_face_distance(candidates[i], a) for a in anchors) for i in remaining]
        else:
            scores = [0] * len(remaining)
        best = max(scores)
        options = [i for i, s in zip(remaining, scores) if s == best]
        choice = options[int(rng.integers(len(options)))]
        picked.append(choice)
        anchors.append(candidates[choice])
        remaining.remove(choice)
    return picked


def _is_frst(triangulation) -> bool:
    return bool(triangulation.is_fine() and triangulation.is_star() and triangulation.is_regular())


def _kcup_log_volume(triangulation, key: str) -> float:
    from reward_functions.max_kcup import MaxKcupReward

    class _State:
        def __init__(self):
            self.key, self.cy_triangulation = key, triangulation

    return float(math.log(MaxKcupReward().metric(_State())))


class _Pool:
    """Distinct 2-face classes with their sampler of origin."""

    def __init__(self):
        self.items: Dict[Tuple, Dict[str, Any]] = {}

    def add(self, triangulation, sampler: str, seed: Optional[int]) -> bool:
        parts = two_face_parts(triangulation)
        key = two_face_key(parts)
        if key in self.items:
            return False
        self.items[key] = {"triangulation": triangulation, "parts": parts, "sampler": sampler, "seed": seed}
        return True

    def by_sampler(self, prefix: str) -> List[Dict[str, Any]]:
        return [item for item in self.items.values() if item["sampler"].startswith(prefix)]


def _fast_pool(polytope, pool: _Pool, *, count: int, seed: int) -> Dict:
    from data.cy.pipeline import _generate_frst_seeds_with_random_triangulations_fair

    starts, diagnostic = _generate_frst_seeds_with_random_triangulations_fair(
        polytope, target_count=count, seed=seed, fast_only=True,
        include_points_interior_to_facets=False, make_star=True)
    for triangulation in starts:
        pool.add(triangulation, "fast", seed)
    return {"fast_returned": len(starts), "fast_status": diagnostic.get("status")}


def _enumerate_ntfe(polytope, pool: _Pool, *, needed: int, seed: int, time_limit: float,
                    upper_bound_limit: int) -> Dict:
    """Reservoir-sample NTFE classes uniformly if full enumeration finishes in time."""
    upper = int(polytope.n_2face_triangs())
    info = {"n_2face_triangs": upper, "ntfe_complete": False, "ntfe_count": None}
    if upper > upper_bound_limit:
        info["ntfe_skipped"] = "n_2face_triangs above limit"
        return info
    rng = np.random.default_rng(seed)
    reservoir: List[np.ndarray] = []
    count, start = 0, time.perf_counter()
    for heights in polytope.ntfe_frsts(N=None, as_generator=True, heights_only=True):
        count += 1
        if len(reservoir) < needed:
            reservoir.append(np.asarray(heights))
        else:
            slot = int(rng.integers(count))
            if slot < needed:
                reservoir[slot] = np.asarray(heights)
        if time.perf_counter() - start > time_limit:
            info.update(ntfe_skipped="time limit", ntfe_partial_count=count)
            return info
    info.update(ntfe_complete=True, ntfe_count=count, ntfe_seconds=time.perf_counter() - start)
    # ntfe heights cover polytope.points(); keep only points not interior to facets.
    keep = polytope.points_not_interior_to_facets(as_indices=True)
    for heights in reservoir:
        triangulation = polytope.triangulate(heights=heights[keep], include_points_interior_to_facets=False,
                                             make_star=True)
        pool.add(triangulation, "ntfe_uniform", seed)
    return info


def _fair_pool(polytope, pool: _Pool, *, count: int, seed: int, timeout: float) -> Dict:
    from data.cy.pipeline import _call_with_optional_timeout

    try:
        starts = _call_with_optional_timeout(lambda: polytope.random_triangulations_fair(
            N=count, make_star=True, include_points_interior_to_facets=False, as_list=True,
            progress_bar=False, seed=seed), timeout_seconds=timeout)
    except Exception as exc:
        return {"fair_status": f"error:{type(exc).__name__}"}
    added = sum(pool.add(t, "fair", seed) for t in starts if _is_frst(t))
    return {"fair_status": "ok", "fair_returned": len(starts), "fair_added": added}


def _walk_pool(pool: _Pool, *, walks: int, min_steps: int, max_steps: int, seed: int, time_limit: float) -> Dict:
    rng = np.random.default_rng(seed)
    sources = [item["triangulation"] for item in pool.by_sampler("fast")]
    added, steps, start = 0, 0, time.perf_counter()
    for walk in range(walks if sources else 0):
        triangulation = sources[int(rng.integers(len(sources)))]
        for _ in range(int(rng.integers(min_steps, max_steps + 1))):
            neighbors = list(triangulation.neighbor_triangulations(two_neighbors=True))
            if not neighbors:
                break
            triangulation = neighbors[int(rng.integers(len(neighbors)))]
            steps += 1
        added += pool.add(triangulation, "walk", seed)
        if time.perf_counter() - start > time_limit:
            break
    return {"walk_steps": steps, "walk_added": added, "walk_seconds": time.perf_counter() - start}


def process_polytope(spec: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cytools_config import configure_cytools

    configure_cytools(worker=True)
    from cytools import Polytope
    from data.cy.pipeline import _build_polytope_jsonl_rows, _serialize_fetched_n_polytope, serialize_triangulation

    start = time.perf_counter()
    index = int(spec["polytope_index"])
    seed = derive_seed(params["seed"], "starts", index)
    rng = np.random.default_rng(seed)
    polytope = Polytope(spec["vertices"])
    pool, info = _Pool(), {}
    k = int(params["starts"])
    info.update(_fast_pool(polytope, pool, count=params["fast_pool"], seed=derive_seed(seed, "fast")))
    if params["mode"] == "train":
        need_extra = k - min(k // 2, len(pool.items))
        if params.get("skip_ntfe_enumeration"):
            info.update(n_2face_triangs=None, ntfe_complete=False, ntfe_count=None,
                        ntfe_skipped="disabled by flag")
        else:
            info.update(_enumerate_ntfe(polytope, pool, needed=need_extra + k, seed=derive_seed(seed, "ntfe"),
                                        time_limit=params["ntfe_time_limit"],
                                        upper_bound_limit=params["ntfe_upper_bound_limit"]))
        if not info["ntfe_complete"]:
            info.update(_fair_pool(polytope, pool, count=params["fair_pool"], seed=derive_seed(seed, "fair"),
                                   timeout=params["fair_timeout"]))
            info.update(_walk_pool(pool, walks=params["walks"], min_steps=params["walk_min"],
                                   max_steps=params["walk_max"], seed=derive_seed(seed, "walk"),
                                   time_limit=params["walk_time_limit"]))
    info["pool_classes"] = len(pool.items)
    info["pool_by_sampler"] = {s: len(pool.by_sampler(s)) for s in ("fast", "ntfe_uniform", "fair", "walk")}

    # Candidate order: half fast (evaluation distribution), then uniform NTFE or farthest-point.
    fast = pool.by_sampler("fast")
    order = fast[: (k // 2 if params["mode"] == "train" else k)]
    if params["mode"] == "train":
        if info.get("ntfe_complete"):
            chosen_keys = {two_face_key(item["parts"]) for item in order}
            rest = [item for item in pool.by_sampler("ntfe_uniform") if two_face_key(item["parts"]) not in chosen_keys]
            rest = [rest[i] for i in rng.permutation(len(rest))] + fast[len(order):]
        else:
            taken = {id(item) for item in order}
            rest_pool = [item for item in pool.items.values() if id(item) not in taken]
            picked = farthest_point_order([item["parts"] for item in rest_pool], [item["parts"] for item in order],
                                          len(rest_pool), rng)
            rest = [rest_pool[i] for i in picked]
        order = order + rest
    else:
        order = order + fast[len(order):]

    selected, dropped = [], {"not_frst": 0, "no_two_neighbors": 0, "kcup_failed": 0}
    for item in order:
        if len(selected) >= k:
            break
        triangulation = item["triangulation"]
        if not _is_frst(triangulation):
            dropped["not_frst"] += 1
            continue
        num_neighbors = len(list(triangulation.neighbor_triangulations(two_neighbors=True)))
        if num_neighbors == 0:
            dropped["no_two_neighbors"] += 1
            continue
        try:
            kcup = _kcup_log_volume(triangulation, f"{index}:{two_face_hash(item['parts'])}")
        except Exception as exc:
            dropped["kcup_failed"] += 1
            info.setdefault("kcup_errors", []).append(f"{type(exc).__name__}: {exc}"[:200])
            continue
        selected.append({**item, "num_two_neighbors": num_neighbors, "kcup_log_volume": kcup})

    result = {"polytope_index": index, "seconds": time.perf_counter() - start, "info": info,
              "dropped_starts": dropped, "num_starts": len(selected)}
    if len(selected) < int(params["min_starts"]):
        result["dropped"] = f"only {len(selected)} distinct valid 2-face classes (min {params['min_starts']})"
        return result

    entry = _serialize_fetched_n_polytope({**spec, "polytope_source": "hugging_face:calabi-yau-data/polytopes-4d"},
                                          polytope)
    entry["frst_seeds"] = [serialize_triangulation(item["triangulation"]) for item in selected]
    row = _build_polytope_jsonl_rows({"polytopes": [entry]})[0]
    parts = [item["parts"] for item in selected]
    row.update(source_id=spec["source_id"], split=spec["split"], vertex_count=spec["vertex_count"],
               start_metadata=[{"sampler": item["sampler"], "seed": item["seed"],
                                "two_face_hash": two_face_hash(item["parts"]),
                                "num_two_neighbors": item["num_two_neighbors"],
                                "kcup_log_volume": item["kcup_log_volume"]} for item in selected])
    result["row"] = row
    result["pairwise_two_face_distance"] = [two_face_distance(parts[i], parts[j])
                                            for i in range(len(parts)) for j in range(i + 1, len(parts))]
    result["num_two_faces"] = len(parts[0])
    return result


def _worker(payload):
    spec, params = payload
    try:
        return process_polytope(spec, params)
    except Exception as exc:
        return {"polytope_index": int(spec["polytope_index"]), "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:]}


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    kept = [r for r in results if "row" in r]

    def stats(values):
        values = [float(v) for v in values]
        if not values:
            return None
        return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
                "min": float(np.min(values)), "max": float(np.max(values))}

    spreads = [max(m["kcup_log_volume"] for m in r["row"]["start_metadata"])
               - min(m["kcup_log_volume"] for m in r["row"]["start_metadata"]) for r in kept]
    samplers: Dict[str, int] = {}
    for r in kept:
        for meta in r["row"]["start_metadata"]:
            samplers[meta["sampler"]] = samplers.get(meta["sampler"], 0) + 1
    vertex_counts: Dict[str, int] = {}
    for r in kept:
        vertex_counts[str(r["row"]["vertex_count"])] = vertex_counts.get(str(r["row"]["vertex_count"]), 0) + 1
    return {
        "polytopes_in": len(results), "polytopes_kept": len(kept),
        "dropped_polytopes": [{"polytope_index": r["polytope_index"], "reason": r.get("dropped", r.get("error"))}
                              for r in results if "row" not in r],
        "starts_total": sum(r["num_starts"] for r in kept),
        "starts_per_polytope": stats([r["num_starts"] for r in kept]),
        "distinct_two_face_classes_per_polytope": stats([len({m["two_face_hash"] for m in r["row"]["start_metadata"]})
                                                         for r in kept]),
        "pairwise_two_face_distance": stats([d for r in kept for d in r["pairwise_two_face_distance"]]),
        "pairwise_two_face_distance_fraction": stats([d / r["num_two_faces"] for r in kept
                                                      for d in r["pairwise_two_face_distance"]]),
        "kcup_log_volume_spread_within_polytope": stats(spreads),
        "kcup_log_volume": stats([m["kcup_log_volume"] for r in kept for m in r["row"]["start_metadata"]]),
        "samplers": samplers,
        "ntfe_complete_fraction": float(np.mean([bool(r["info"].get("ntfe_complete")) for r in kept])) if kept else None,
        "ntfe_count": stats([r["info"]["ntfe_count"] for r in kept if r["info"].get("ntfe_count") is not None]),
        "seconds_per_polytope": stats([r["seconds"] for r in results if "seconds" in r]),
        "vertex_count_distribution": dict(sorted(vertex_counts.items(), key=lambda item: int(item[0]))),
        "dropped_starts": {key: sum(r.get("dropped_starts", {}).get(key, 0) for r in results)
                           for key in ("not_frst", "no_two_neighbors", "kcup_failed")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--polytopes", required=True)
    parser.add_argument("--output", required=True, help="Output .samples.jsonl path.")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--min-starts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N polytopes (pilot runs).")
    parser.add_argument("--fast-pool", type=int, default=20)
    parser.add_argument("--fair-pool", type=int, default=10)
    parser.add_argument("--fair-timeout", type=float, default=60.0)
    parser.add_argument("--walks", type=int, default=20)
    parser.add_argument("--walk-min", type=int, default=10)
    parser.add_argument("--walk-max", type=int, default=30)
    parser.add_argument("--walk-time-limit", type=float, default=120.0)
    parser.add_argument("--ntfe-time-limit", type=float, default=300.0)
    parser.add_argument("--ntfe-upper-bound-limit", type=int, default=600_000,
                        help="Skip full NTFE enumeration when n_2face_triangs exceeds this.")
    parser.add_argument("--skip-ntfe-enumeration", action="store_true",
                        help="Do not count 2-face triangulations or enumerate NTFEs; go straight to fair/walk pools.")
    parser.add_argument("--num-workers", type=int, default=12)
    args = parser.parse_args()

    specs = [json.loads(line) for line in Path(args.polytopes).expanduser().read_text().splitlines() if line.strip()]
    specs = specs[: args.limit] if args.limit else specs
    params = {key: getattr(args, key) for key in (
        "mode", "starts", "min_starts", "seed", "fast_pool", "fair_pool", "fair_timeout", "walks", "walk_min",
        "walk_max", "walk_time_limit", "ntfe_time_limit", "ntfe_upper_bound_limit", "skip_ntfe_enumeration")}
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    results = []
    with mp.get_context("spawn").Pool(args.num_workers, maxtasksperchild=20) as pool:
        for result in pool.imap_unordered(_worker, [(spec, params) for spec in specs]):
            results.append(result)
            status = "ok" if "row" in result else result.get("dropped", result.get("error"))
            print(f"[{len(results)}/{len(specs)}] polytope {result['polytope_index']}: {status} "
                  f"({result.get('seconds', 0):.1f}s, elapsed {time.time() - start:.0f}s)", flush=True)
    results.sort(key=lambda r: r["polytope_index"])
    with output.open("w") as handle:
        for result in results:
            if "row" in result:
                handle.write(json.dumps(result["row"]) + "\n")
    summary = {"params": params, "polytopes_file": str(args.polytopes), "wall_seconds": time.time() - start,
               "num_workers": args.num_workers, **summarize(results),
               "errors": [{"polytope_index": r["polytope_index"], "traceback": r["traceback"]}
                          for r in results if "error" in r]}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("errors", "dropped_polytopes")}, indent=1))


if __name__ == "__main__":
    main()
