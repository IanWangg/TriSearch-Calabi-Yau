"""A4: per-observation overhead of the two_face_invariant observation builder on real states.

Builds the observations of the first real base states from data/cy/two_neighbors_h11_12.samples.jsonl with
(a) create_two_face_data_from_state (default) and (b) create_invariant_two_face_data_from_state, first pass (cache
misses) and second pass (cache hits). Single process, single thread.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "test"))
torch.set_num_threads(1)

from core.cytools_config import configure_cytools  # noqa: E402
from core.cy_two_face_data import create_two_face_data_from_state  # noqa: E402
from core.two_face_invariant_features import (  # noqa: E402
    clear_invariant_topology_cache, create_invariant_two_face_data_from_state, invariant_topology_cache_stats)
from mdp import cy_geometry_worker as worker  # noqa: E402


def load_states(n_max=40):
    configure_cytools(worker=True)
    path = HERE / "data/cy/two_neighbors_h11_12.samples.jsonl"
    states = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        row["frst_list"] = row["frst_list"][:1]
        collection = worker.execute_geometry_request(dict(operation="build_collection", row=row,
            neighbor_mode="two_neighbors", include_points_interior_to_facets=False, include_two_face_metadata=True))
        states.extend(collection["base_states"])
        if len(states) >= n_max:
            break
    return states[:n_max]


def timed(builder, states):
    times = []
    for state in states:
        start = time.perf_counter()
        builder(state, ())
        times.append(time.perf_counter() - start)
    return times


def main():
    states = load_states()
    default = timed(create_two_face_data_from_state, states)
    clear_invariant_topology_cache()
    miss = timed(create_invariant_two_face_data_from_state, states)
    stats_after_miss = invariant_topology_cache_stats()
    hit = timed(create_invariant_two_face_data_from_state, states)
    stats_after_hit = invariant_topology_cache_stats()
    ms = lambda xs: 1000 * statistics.median(xs)  # noqa: E731
    result = dict(
        n_states=len(states),
        default_ms_median=ms(default),
        invariant_first_pass_ms_median=ms(miss),
        invariant_cached_pass_ms_median=ms(hit),
        invariant_overhead_uncached_ms=ms(miss) - ms(default),
        invariant_overhead_cached_ms=ms(hit) - ms(default),
        cache_after_first_pass=stats_after_miss,
        cache_after_second_pass=stats_after_hit,
    )
    print(json.dumps(result, indent=2))
    out = HERE / "runs/inv_encoder_20261104/obs_overhead.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
