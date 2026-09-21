"""Record and replay a bounded CY expansion trace, without policy training.

Run from the repository root, using the Sage environment. The reference is the
current rich-state compatibility implementation, not a historical checkout.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import statistics
import sys
import threading
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _initialize_benchmark_worker(reference):
    from core.cytools_config import configure_cytools

    configure_cytools(worker=True)
    if reference:
        # Preserve the supported CYTools-before-Sage import order.
        from mdp.cy_triangulation_state import CYTriangulationState  # noqa: F401


def _reference_expansion(request):
    """Time the current rich-state path inside an isolated, managed process."""
    from mdp.cy_geometry_worker import _get_polytope, _triangulate
    from mdp.cy_rollout import _expand_cy_state_worker
    from mdp.cy_triangulation_state import CYTriangulationState

    started = time.perf_counter()
    configuration = request["configuration"]
    index, simplices, mode, _is_frst, _is_target = request["state"]
    triangulation = _triangulate(_get_polytope(configuration), configuration, simplices)
    state = CYTriangulationState(
        vertices=configuration.input_vertices, point_config_index=index,
        simplices=simplices, cy_triangulation=triangulation, neighbor_mode=mode,
    )
    construction_sec = time.perf_counter() - started
    # This measures the old style rich input, without transferring geometry to
    # the benchmark parent. Pickling is timed separately from geometry compute.
    pickle_started = time.perf_counter()
    input_bytes = len(pickle.dumps((state, bool(request["objective_mode"])), protocol=pickle.HIGHEST_PROTOCOL))
    pickle_sec = time.perf_counter() - pickle_started
    started = time.perf_counter()
    expansion = _expand_cy_state_worker((state, bool(request["objective_mode"])))
    geometry_sec = construction_sec + time.perf_counter() - started
    pickle_started = time.perf_counter()
    output_bytes = len(pickle.dumps(expansion, protocol=pickle.HIGHEST_PROTOCOL))
    pickle_sec += time.perf_counter() - pickle_started
    return {"expansion": expansion, "geometry_sec": geometry_sec,
            "payload_input_pickle_bytes": input_bytes, "payload_output_pickle_bytes": output_bytes,
            "payload_pickle_sec": pickle_sec}


def _compact_expansion(request):
    from mdp.cy_geometry_worker import execute_geometry_request

    pickle_started = time.perf_counter()
    # Configurations are separately interned by the real transport. Its actual
    # wire bytes, including registration, are reported from pool statistics.
    input_bytes = len(pickle.dumps({key: value for key, value in request.items() if key != "configuration"},
                                  protocol=pickle.HIGHEST_PROTOCOL))
    pickle_sec = time.perf_counter() - pickle_started
    started = time.perf_counter()
    expansion = execute_geometry_request(request)
    geometry_sec = time.perf_counter() - started
    pickle_started = time.perf_counter()
    output_bytes = len(pickle.dumps(expansion, protocol=pickle.HIGHEST_PROTOCOL))
    pickle_sec += time.perf_counter() - pickle_started
    return {"expansion": expansion, "geometry_sec": geometry_sec,
            "payload_input_pickle_bytes": input_bytes, "payload_output_pickle_bytes": output_bytes,
            "payload_pickle_sec": pickle_sec}


def _semantic_signature(expansion):
    """Ignore backend order while comparing all available transition geometry."""
    payload = {
        "key": expansion.key,
        "point_config_index": expansion.point_config_index,
        "simplices": sorted(expansion.simplices),
        "actions": sorted(expansion.candidate_actions),
        "ambiguous_actions": sorted(expansion.ambiguous_actions),
        "transitions": sorted(
            (action, transition.next_key, transition.simplices_from(expansion.simplices), transition.next_is_target)
            for action, transition in expansion.transitions
        ),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


class _JobMemorySampler:
    """Sample aggregate RSS of this benchmark and its owned descendants."""

    def __init__(self):
        self.peak_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def _sample(self):
        from core.cy_process_runtime import _memory_snapshot

        snapshot = _memory_snapshot(os.getpid(), 0, ())
        self.peak_bytes = max(self.peak_bytes, snapshot["rss_bytes"])
        self.samples += 1

    def _sample_loop(self):
        while not self._stop.wait(.1):
            self._sample()

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self._stop.set()
        self._thread.join()
        self._sample()


def _bounded_rows(rows, max_states, neighbor_mode):
    """Stratify before geometry loading, so unused initial states stay unloaded."""
    from mdp.cy_rollout import _iter_row_initial_simplices

    selected = [[] for _ in rows]
    if neighbor_mode == "two_neighbors":
        streams = [iter(entry for entry in row.get("frst_list", ()) if entry.get("simplices")) for row in rows]
    else:
        streams = [iter(_iter_row_initial_simplices(row)) for row in rows]
    total = 0
    while total < max_states:
        advanced = False
        for index, stream in enumerate(streams):
            value = next(stream, None)
            if value is not None:
                selected[index].append(value)
                total += 1
                advanced = True
                if total == max_states:
                    break
        if not advanced:
            break
    result = []
    for row, chosen in zip(rows, selected):
        if not chosen:
            continue
        reduced = {"polytope_index": row["polytope_index"], "vertices": row["vertices"]}
        if neighbor_mode == "two_neighbors":
            reduced["frst_list"] = [{"simplices": entry["simplices"]} for entry in chosen]
        else:
            reduced["frst_list"] = []
            reduced["non_fine_triangulation_list"] = [{"simplices": simplices} for simplices in chosen]
        result.append(reduced)
    if not result:
        raise ValueError("No initial states were found in the selected dataset rows.")
    return result


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_trace(args):
    from mdp.cy_rollout import build_cy_rollout_collection, create_transition_pool, load_cy_sample_rows

    rows = load_cy_sample_rows(args.dataset_path, max_rows=args.max_rows)
    reduced = _bounded_rows(rows, args.max_states, args.neighbor_mode)
    del rows
    with create_transition_pool(num_workers=1, task_timeout_sec=args.task_timeout_sec,
                                memory_budget_gb=args.memory_budget_gb) as pool:
        collection = build_cy_rollout_collection(
            reduced, include_points_interior_to_facets=args.include_points_interior_to_facets,
            neighbor_mode=args.neighbor_mode, transition_pool=pool,
        )
        states = list(collection.initial_states)
        random.Random(args.seed).shuffle(states)
        trace = {
            "schema_version": 1,
            "dataset_path": str(Path(args.dataset_path).resolve()),
            "dataset_sha256": _file_sha256(args.dataset_path),
            "seed": args.seed,
            "selection": "round_robin_first_initial_states_then_seeded_shuffle",
            "configurations": [asdict(configuration) for _, configuration in sorted(collection.polytope_by_index.items())],
            "requests": [{"operation": "expand", "configuration_index": state.point_config_index,
                          "state": state.to_payload(), "objective_mode": True, "action_order": "native"}
                         for state in states[:args.max_states]],
        }
    return trace


def _trace_requests(trace, max_states):
    from mdp.cy_state_record import CYPointConfiguration

    if trace.get("schema_version") != 1:
        raise ValueError("Unsupported expansion trace schema_version.")
    rows = trace["requests"]
    if not rows or len(rows) > max_states:
        raise ValueError("Trace request count must be between 1 and --max_states.")
    configurations = {}
    for row in trace["configurations"]:
        configuration = CYPointConfiguration(
            index=int(row["index"]), input_vertices=tuple(tuple(point) for point in row["input_vertices"]),
            points=tuple(tuple(point) for point in row["points"]), labels=tuple(row["labels"]),
            include_points_interior_to_facets=bool(row["include_points_interior_to_facets"]),
        )
        if configuration.index in configurations:
            raise ValueError("Trace contains duplicate point configuration indices.")
        configurations[configuration.index] = configuration
    requests = []
    for row in rows:
        index, simplices, mode, is_frst, is_target = row["state"]
        configuration = configurations[int(row["configuration_index"])]
        if int(index) != configuration.index or row["operation"] != "expand":
            raise ValueError("Trace request has inconsistent configuration or operation.")
        requests.append({"operation": "expand", "configuration": configuration,
                         "state": (int(index), tuple(tuple(simplex) for simplex in simplices), mode, bool(is_frst), bool(is_target)),
                         "objective_mode": bool(row["objective_mode"]), "action_order": row.get("action_order", "native")})
    return requests


def _run_phase(pool, requests, function, golden):
    before = pool.stats
    metrics = {"geometry_sec_sum": 0., "payload_pickle_sec_sum": 0.,
               "payload_input_pickle_bytes": 0, "payload_output_pickle_bytes": 0}
    signatures = []
    actions_total = 0
    mismatch_indices = []
    with _JobMemorySampler() as memory:
        started = time.perf_counter()
        for index, result in enumerate(pool.imap(function, iter(requests), chunksize=1)):
            expansion = result["expansion"]
            signature = _semantic_signature(expansion)
            signatures.append(signature)
            if golden is not None and signature != golden[index]:
                mismatch_indices.append(index)
            actions_total += len(expansion.candidate_actions)
            metrics["geometry_sec_sum"] += result["geometry_sec"]
            metrics["payload_pickle_sec_sum"] += result["payload_pickle_sec"]
            for name in ("payload_input_pickle_bytes", "payload_output_pickle_bytes"):
                metrics[name] += result[name]
            del result, expansion
        wall_sec = time.perf_counter() - started
    after = pool.stats
    metrics.update({"wall_sec": wall_sec, "requests": len(requests),
                    "requests_per_sec": len(requests) / wall_sec, "actions_total": actions_total,
                    "sampled_peak_job_rss_bytes": memory.peak_bytes, "memory_samples": memory.samples,
                    "geometry_equal_to_reference": not mismatch_indices, "mismatch_indices": mismatch_indices})
    for name in ("request_bytes_sent", "configuration_bytes_sent", "configuration_registrations", "configuration_reuses"):
        metrics[name] = after.get(name, 0) - before.get(name, 0)
    return metrics, signatures


def run_benchmark(args):
    from mdp.cy_rollout import create_transition_pool

    output = Path(args.output_path).resolve()
    if not output.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("--output_path must be inside this repository.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.replay_trace_path:
        with Path(args.replay_trace_path).open(encoding="utf-8") as handle:
            trace = json.load(handle)
    else:
        trace = _record_trace(args)
    requests = _trace_requests(trace, args.max_states)
    trace_path = output.with_name(output.stem + "_trace.json")
    trace_text = json.dumps(trace, sort_keys=True, separators=(",", ":")) + "\n"
    trace_path.write_text(trace_text, encoding="utf-8")
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "trace_path": str(trace_path), "trace_sha256": hashlib.sha256(trace_text.encode()).hexdigest(),
        "request_count": len(requests), "python": sys.version,
        "source_sha256": {path: _file_sha256(REPOSITORY_ROOT / path) for path in (
            "mdp/cy_rollout.py", "mdp/cy_triangulation_state.py", "mdp/cy_geometry_worker.py",
            "core/cy_process_runtime.py", "tools/benchmark_cy_runtime.py")},
        "reference": "current_rich_state_compatibility_path_in_one_managed_process",
        "comparison": "candidate_sets_ambiguity_destination_simplices_and_target_flags; backend_order_ignored",
        "notes": ["This is an expansion microbenchmark, not historical-baseline or whole-training throughput.",
                  "Cold uses a fresh worker pool with library imports completed; warm repeats the trace in that pool.",
                  "Wall time includes payload-size measurement and semantic validation; geometry_sec_sum excludes both.",
                  "Reference rich input is serialized for measurement inside its worker; the parent sends compact descriptions.",
                  "RSS is sampled every 0.1 seconds and may miss shorter peaks; it includes the parent, guardian, and workers."],
        "runs": [],
    }
    golden = None
    backends = [("rich_reference", 1, _reference_expansion)] + [("managed_compact", count, _compact_expansion) for count in args.workers]
    for backend, workers, function in backends:
        for repetition in range(args.repetitions):
            started = time.perf_counter()
            with create_transition_pool(num_workers=workers, task_timeout_sec=args.task_timeout_sec,
                                        memory_budget_gb=args.memory_budget_gb,
                                        initializer=_initialize_benchmark_worker, initargs=(backend == "rich_reference",)) as pool:
                startup_sec = time.perf_counter() - started
                for phase in ("cold", "warm"):
                    metrics, signatures = _run_phase(pool, requests, function, golden)
                    if golden is None:
                        golden = signatures
                        report["reference_semantic_signatures"] = golden
                    entry = {"backend": backend, "workers": workers, "repetition": repetition,
                             "phase": phase, "startup_sec": startup_sec, **metrics}
                    report["runs"].append(entry)
                    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                    print(f"{backend} workers={workers} repetition={repetition + 1} {phase}: "
                          f"{metrics['wall_sec']:.3f}s {metrics['requests_per_sec']:.2f} requests/s "
                          f"rss={metrics['sampled_peak_job_rss_bytes'] / 1024**3:.3f} GiB "
                          f"equal={metrics['geometry_equal_to_reference']}", flush=True)
    summaries = []
    for backend, workers, _function in backends:
        for phase in ("cold", "warm"):
            runs = [entry for entry in report["runs"] if entry["backend"] == backend and entry["workers"] == workers and entry["phase"] == phase]
            reference = [entry["wall_sec"] for entry in report["runs"] if entry["backend"] == "rich_reference" and entry["phase"] == phase]
            median_wall = statistics.median(entry["wall_sec"] for entry in runs)
            summaries.append({"backend": backend, "workers": workers, "phase": phase,
                              "median_wall_sec": median_wall, "median_requests_per_sec": len(requests) / median_wall,
                              "speedup_vs_current_rich_reference": statistics.median(reference) / median_wall,
                              "sampled_peak_job_rss_bytes": max(entry["sampled_peak_job_rss_bytes"] for entry in runs)})
    report["summaries"] = summaries
    report["geometry_equal_to_reference"] = all(entry["geometry_equal_to_reference"] for entry in report["runs"])
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {output}\nTrace: {trace_path}", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", default="data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl")
    parser.add_argument("--replay_trace_path", default=None, help="Replay an existing JSON trace without loading the dataset.")
    parser.add_argument("--max_rows", type=int, default=20)
    parser.add_argument("--max_states", "--num_states", dest="max_states", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--neighbor_mode", choices=("regular", "two_neighbors"), default="regular")
    interior = parser.add_mutually_exclusive_group()
    interior.add_argument("--include_points_interior_to_facets", dest="include_points_interior_to_facets", action="store_true", default=None)
    interior.add_argument("--exclude_points_interior_to_facets", dest="include_points_interior_to_facets", action="store_false")
    parser.add_argument("--memory_budget_gb", type=float, default=64.)
    parser.add_argument("--task_timeout_sec", type=float, default=300.)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--output_path", default=f"runs/benchmark_cy_runtime/{timestamp}.json")
    args = parser.parse_args(argv)
    if min(args.max_rows, args.max_states, args.repetitions, *args.workers) <= 0:
        parser.error("row/state/worker counts and repetitions must be positive.")
    args.workers = list(dict.fromkeys(args.workers))
    if args.include_points_interior_to_facets is None:
        args.include_points_interior_to_facets = args.neighbor_mode == "regular"
    if args.neighbor_mode == "two_neighbors" and args.include_points_interior_to_facets:
        parser.error("two_neighbors requires --exclude_points_interior_to_facets.")
    return args


def main(argv=None):
    report = run_benchmark(parse_args(argv))
    return 0 if report["geometry_equal_to_reference"] else 1


if __name__ == "__main__":
    # Worker tasks must be importable by module name, not from __main__.
    from tools.benchmark_cy_runtime import main as _module_main

    raise SystemExit(_module_main())
