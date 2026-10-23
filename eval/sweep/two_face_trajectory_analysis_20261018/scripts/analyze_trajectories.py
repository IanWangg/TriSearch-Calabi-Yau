"""Offline analysis of completed two_face searches and their observed query graphs.

Uses native validation and canonical two_face identity. Never queries geometry,
loads a policy, or treats unqueried actions / GA mutations as flip graph edges.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict, deque
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np

from eval.results.plotting import read_comparison, TWO_FACE_METRIC_REL_TOL
from eval.setup import load_eval_setup
from eval.research_paths import resolve_archived_path
from mdp.cy_state_record import CYPointConfiguration, state_key, two_face_state_key

SOURCES = {
    30: ROOT / "eval/results/runs/h11_30_two_face_checkpoint_sweep_20261011_202310",
    50: ROOT / "eval/results/runs/h11_50_two_face_checkpoint_sweep_20261012_000945",
}
ARCHIVE_ROOT = ROOT
BUDGET = 1000
TOL = TWO_FACE_METRIC_REL_TOL


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def sha(key):
    return hashlib.sha256(key.encode()).hexdigest()


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def configuration(raw):
    return CYPointConfiguration(
        index=raw["index"], input_vertices=tuple(map(tuple, raw["input_vertices"])),
        points=tuple(map(tuple, raw["points"])), labels=tuple(raw["labels"]),
        include_points_interior_to_facets=raw["include_points_interior_to_facets"],
        two_face_labels=tuple(map(tuple, raw["two_face_labels"])),
    )


def source_jobs(study):
    plan = json.loads((study / "run_plan.json").read_text())
    jobs = [dict(j, source_run=j["output_dir"], reused=False)
            for j in plan["planned_jobs"] if j["status"] == "complete"]
    for algorithm, entries in plan["reused_baselines"].items():
        jobs.extend(dict(e, algorithm=algorithm, group="favorable", iteration=None,
                         family="baseline", reused=True) for e in entries)
    for job in jobs:
        job["source_run"] = str(resolve_archived_path(job["source_run"], historical_root=ARCHIVE_ROOT))
    return plan, jobs


def ingest(destination):
    """Validate raw runs and write small, independently inspectable derived data."""
    destination.mkdir(parents=True, exist_ok=False)
    graphs, trajectories, provenance, availability = {}, [], [], []
    for h11, study in SOURCES.items():
        plan, jobs = source_jobs(study)
        configurations = {}
        expected_initials = {}
        for group, entry in plan["group_setups"].items():
            setup = load_eval_setup(resolve_archived_path(entry["path"], historical_root=ARCHIVE_ROOT))
            expected_initials[group] = {r["polytope_index"]: state_key(
                r["polytope_index"], r["frst_list"][0]["simplices"], "two_neighbors") for r in setup.rows}
            raw = json.loads((study / "diagnostics" / f"configurations_{group}.json").read_text())
            configurations[group] = {int(k): configuration(v) for k, v in raw.items()}
        for j in plan["planned_jobs"]:
            availability.append(dict(h11=h11, name=j["name"], status=j["status"], group=j["group"],
                                     checkpoint=j.get("iteration"), algorithm=j["algorithm"]))
        for job_index, job in enumerate(jobs):
            run, group = Path(job["source_run"]), job["group"]
            config = json.loads((run / "config.json").read_text())
            data = read_comparison(run)
            assert data.spec["h11"] == h11 and data.spec["num_starts"] == 1
            assert data.spec["objective_budget"] >= BUDGET
            algorithm = job["algorithm"]
            assert data.spec["algorithms"] == [algorithm]
            checkpoint = job.get("iteration")
            if checkpoint is not None:
                assert config["policy"]["checkpoint_sha256"] == plan["checkpoints"][str(checkpoint)]["sha256"]
                assert config["policy"]["value_score_definitions"][algorithm] == "ln_objective_plus_discounted_value"
                assert config["policy"]["policy_preselection_mode"] == job["policy_preselection_mode"]
            local = {}
            for (_, index, start), rollout in sorted(data.rollouts.items()):
                assert start == 0 and rollout["initial_state_key"] == expected_initials[group][index]
                pair = (h11, group, index)
                graph = graphs.setdefault(pair, dict(node_ids={}, nodes=[], edges=set(), root=None))
                t = dict(trajectory_id=len(trajectories), h11=h11, group=group, polytope_index=index,
                         checkpoint=checkpoint, algorithm=algorithm,
                         mode=job.get("policy_preselection_mode", "not_applicable"),
                         proposal_count=job.get("policy_proposal_count") if checkpoint is not None else None,
                         source_run=str(run.relative_to(ARCHIVE_ROOT)), reused=job["reused"],
                         source_queries=rollout["objective_queries"], source_budget=data.spec["objective_budget"],
                         source_overshoot=rollout["budget_overshoot"], source_expansions=rollout["expansion_count"],
                         termination_reason=rollout["termination_reason"], queries=[], expansions=[],
                         curve=data.curves[algorithm, index, start][:BUDGET + 1].tolist())
                trajectories.append(t)
                local[index] = t
            full_to_face = {}

            def node_id(index, key, value=None, full_key=None):
                graph = graphs[h11, group, index]
                digest = sha(key)
                n = graph["node_ids"].get(digest)
                if n is None:
                    n = len(graph["nodes"])
                    graph["node_ids"][digest] = n
                    graph["nodes"].append(dict(digest=digest, min_volume=None, max_volume=None,
                                               observations=0, flip_observed=False))
                node = graph["nodes"][n]
                if value is not None:
                    node["min_volume"] = value if node["min_volume"] is None else min(value, node["min_volume"])
                    node["max_volume"] = value if node["max_volume"] is None else max(value, node["max_volume"])
                    node["observations"] += 1
                    node["flip_observed"] |= algorithm != "cyopt_ga"
                return n

            raw_hashes = {}
            for filename in ["queries.jsonl", "expansions.jsonl"]:
                path = run / filename
                before = path.stat()
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for line in stream:
                        digest.update(line)
                        event = json.loads(line)
                        index = event["polytope_index"]
                        t = local[index]
                        if filename == "queries.jsonl":
                            q = event["query_index"]
                            if q > BUDGET:
                                continue
                            key = event.get("evaluation_state_key")
                            if key is None:
                                assert algorithm == "cyopt_ga"
                                full_digest = sha(event["state_key"])
                                key = full_to_face.get(full_digest)
                                if key is None:
                                    simplices = ast.literal_eval(event["state_key"].split(":", 1)[1])
                                    key = two_face_state_key(configurations[group][index], simplices)
                                    full_to_face[full_digest] = key
                            n = node_id(index, key, event["objective"])
                            parent = -1
                            if q == 0:
                                graph = graphs[h11, group, index]
                                assert graph["root"] in (None, n)
                                graph["root"] = n
                            elif algorithm != "cyopt_ga":
                                parent = node_id(index, event["source_evaluation_state_key"])
                                assert parent != n
                                graphs[h11, group, index]["edges"].add(tuple(sorted((parent, n))))
                            t["queries"].append([q, n, parent, event["objective"], event.get("depth"),
                                                 event["expansion_index"]])
                        elif algorithm != "cyopt_ga" and event["queries_before"] < BUDGET:
                            assert event["status"] == "ok"
                            n = node_id(index, event["evaluation_state_key"])
                            t["expansions"].append([event["expansion_index"], n, event["queries_before"],
                                                    event["round_queries"], event["candidate_count"],
                                                    event["depth"], event["objective"]])
                after = path.stat()
                assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                raw_hashes[filename] = digest.hexdigest()
                if job["reused"]:
                    assert raw_hashes[filename] == job["raw_sha256"][filename]
            for t in local.values():
                assert len(t["queries"]) == min(t["source_queries"], BUDGET) + 1
                assert math.isclose(max(q[3] for q in t["queries"]), t["curve"][-1], rel_tol=1e-12)
            provenance.append(dict(h11=h11, source_run=str(run.relative_to(ARCHIVE_ROOT)),
                                   native_validation="passed", raw_sha256=raw_hashes,
                                   checkpoint_sha256=data.policy_checkpoint_sha256, git_commit=config["environment"].get("git_commit")))
            if job_index % 5 == 0 or job_index == len(jobs) - 1:
                print(f"h11={h11}: {job_index + 1}/{len(jobs)} sources validated; {len(trajectories)} trajectories", flush=True)
    with (destination / "trajectories.jsonl").open("w") as stream:
        for t in trajectories:
            stream.write(json.dumps(t, separators=(",", ":"), allow_nan=False) + "\n")
    for pair, graph in sorted(graphs.items()):
        h11, group, index = pair
        name = f"h11_{h11}_{group}_polytope_{index}"
        write_json(destination / f"{name}_graph.json", dict(h11=h11, group=group, polytope_index=index,
                   root=graph["root"], nodes=graph["nodes"], edges=sorted(graph["edges"])))
    write_json(destination / "provenance.json", provenance)
    write_csv(destination / "configuration_availability.csv", availability)
    return trajectories


def analyze(destination):
    trajectories = list(rows(destination / "trajectories.jsonl"))
    graphs, graph_summary = {}, []
    for path in sorted(destination.glob("*_graph.json")):
        g = json.loads(path.read_text())
        pair = g["h11"], g["group"], g["polytope_index"]
        adjacency = [[] for _ in g["nodes"]]
        for a, b in g["edges"]:
            adjacency[a].append(b)
            adjacency[b].append(a)
        distances = {g["root"]: 0}
        queue = deque([g["root"]])
        while queue:
            a = queue.popleft()
            for b in adjacency[a]:
                if b not in distances:
                    distances[b] = distances[a] + 1
                    queue.append(b)
        flip_nodes = {i for i, n in enumerate(g["nodes"]) if n["flip_observed"]}
        assert set(distances) == flip_nodes
        g["distances"] = distances
        graphs[pair] = g
        graph_summary.append(dict(h11=pair[0], group=pair[1], polytope_index=pair[2],
            observed_nodes=len(g["nodes"]), observed_flip_nodes=len(flip_nodes),
            observed_flip_edges=len(g["edges"]), observed_cycle_rank=len(g["edges"]) - len(flip_nodes) + 1,
            observed_radius=max(distances.values()), median_observed_degree=float(np.median([len(adjacency[n]) for n in flip_nodes])),
            metric_conflict_nodes=sum(n["max_volume"] / n["min_volume"] > 1 + TOL for n in g["nodes"]),
            best_observed_volume=max(n["max_volume"] for n in g["nodes"])))
    metrics, paired, initial_selection = [], [], []
    all_root_values = defaultdict(dict)
    for t in trajectories:
        if t["algorithm"] == "cyopt_ga":
            continue
        pair = t["h11"], t["group"], t["polytope_index"]
        for q, n, parent, v, depth, ex in t["queries"]:
            if q and ex == 1:
                all_root_values[pair][n] = v
    for t in trajectories:
        pair = t["h11"], t["group"], t["polytope_index"]
        g = graphs[pair]
        qs, exs = t["queries"], t["expansions"]
        curve = np.asarray(t["curve"])
        values = np.array([q[3] for q in qs])
        best_q = int(np.argmax(values))
        node_to_query = {q[1]: q for q in qs}
        ids = {q[1] for q in qs}
        graph_search = t["algorithm"] != "cyopt_ga"
        if graph_search:
            assert len(ids) == len(qs), "Graph searches must never requery a two_face state."
            for q, n, parent, v, depth, ex in qs[1:]:
                assert parent in node_to_query and node_to_query[parent][0] < q
                assert depth == node_to_query[parent][4] + 1
        path = []
        if graph_search:
            current = qs[best_q]
            while True:
                path.append(current)
                if current[0] == 0:
                    break
                current = node_to_query[current[2]]
            path.reverse()
        path_logs = np.log10([q[3] for q in path]) if path else np.array([])
        downhill = sum(q[3] < node_to_query[q[2]][3] * (1 - TOL) for q in qs[1:]) if graph_search else None
        inversions, regrets = [], []
        if t["algorithm"] in ("rl_value_best_first", "best_first"):
            frontier, removed, pos = [], set(), 0
            for ex, parent, before, count, degree, depth, obj in exs:
                while pos < len(qs) and qs[pos][0] <= before:
                    q = qs[pos]
                    heapq.heappush(frontier, (-q[3], q[1]))
                    pos += 1
                while frontier and frontier[0][1] in removed:
                    heapq.heappop(frontier)
                assert parent not in removed
                assert frontier
                gap = max(0., math.log10(-frontier[0][0] / obj))
                regrets.append(gap)
                inversions.append(gap > math.log10(1 + TOL))
                removed.add(parent)
            if t["algorithm"] == "best_first":
                assert not any(inversions), "Ordinary BeFS must expand the maximum-volume frontier node."
        row = {k: t[k] for k in ["trajectory_id", "h11", "group", "polytope_index", "checkpoint", "algorithm", "mode", "proposal_count", "source_run", "source_queries", "source_budget", "source_overshoot", "termination_reason"]}
        row.update(initial_volume=values[0], endpoint_volume=float(curve[-1]), best_query=best_q,
            queries_to_90pct_endpoint=int(np.flatnonzero(curve >= .9 * curve[-1])[0]),
            unique_states=len(ids), duplicate_query_fraction=1 - (len(ids) - 1) / max(1, len(qs) - 1),
            expansions=len(exs) if graph_search else None,
            max_depth=max(q[4] for q in qs) if graph_search else None,
            best_depth=qs[best_q][4] if graph_search else None,
            observed_shortest_distance_to_best=g["distances"][qs[best_q][1]] if graph_search else None,
            best_path_downhill_steps=int(np.sum(np.diff(path_logs) < math.log10(1 - TOL))) if path else None,
            best_path_max_drawdown_dex=float(np.max(np.maximum.accumulate(path_logs) - path_logs)) if path else None,
            downhill_query_fraction=downhill / (len(qs) - 1) if graph_search and len(qs) > 1 else None,
            low_parent_expansion_fraction=float(np.mean([obj < curve[min(before, BUDGET)] / 10 for ex, parent, before, count, degree, depth, obj in exs])) if exs else None,
            frontier_inversion_fraction=float(np.mean(inversions)) if inversions else None,
            frontier_regret_median_dex=float(np.median(regrets)) if regrets else None,
            frontier_regret_max_dex=max(regrets) if regrets else None,
            raw_candidate_count_median=float(np.median([e[4] for e in exs])) if exs else None,
            queries_per_expansion=(len(qs) - 1) / len(exs) if exs else None,
            gain_after_500_dex=float(math.log10(curve[-1] / curve[500])),
            observed_pool_gap_dex=math.log10(max(n["max_volume"] for n in g["nodes"]) / curve[-1]),
            **{f"best_at_{q}": float(curve[q]) for q in [100, 250, 500, 1000]})
        metrics.append(row)
        if graph_search:
            first = [q for q in qs if q[0] and q[5] == 1]
            pool = all_root_values[pair]
            best_root = max(pool.values())
            initial_selection.append(dict(trajectory_id=t["trajectory_id"], h11=t["h11"], group=t["group"],
                polytope_index=t["polytope_index"], checkpoint=t["checkpoint"], algorithm=t["algorithm"],
                mode=t["mode"], proposal_count=t["proposal_count"], selected_count=len(first),
                pooled_root_neighbor_count=len(pool),
                best_root_neighbor_hit=any(q[3] >= best_root * (1 - TOL) for q in first),
                root_selected_max_volume=max(q[3] for q in first), root_all_max_volume=best_root,
                root_best_gap_dex=math.log10(best_root / max(q[3] for q in first)),
                selected_mean_percentile=float(np.mean([np.mean(np.array(list(pool.values())) <= q[3] * (1 + TOL)) for q in first]))))
    by_id = {r["trajectory_id"]: r for r in metrics}
    lookup = {(t["h11"], t["group"], t["polytope_index"], t["checkpoint"], t["algorithm"], t["mode"], t["proposal_count"]): t for t in trajectories}
    for t in trajectories:
        if t["checkpoint"] is None:
            continue
        prefix = t["h11"], t["group"], t["polytope_index"]
        references = {"ordinary_befs": (*prefix, None, "best_first", "not_applicable", None),
                      "ga": (*prefix, None, "cyopt_ga", "not_applicable", None),
                      "all_neighbors_same_checkpoint": (*prefix, t["checkpoint"], t["algorithm"], "none", -1),
                      "greedy_same_k": (*prefix, t["checkpoint"], t["algorithm"], "greedy", t["proposal_count"]),
                      "beam_same_settings": (*prefix, t["checkpoint"], "rl_value_beam_search", t["mode"], t["proposal_count"]),
                      "checkpoint_400_same_settings": (*prefix, 400, t["algorithm"], t["mode"], t["proposal_count"])}
        for comparison, key in references.items():
            ref = lookup.get(key)
            if ref is None or ref is t:
                continue
            a, b = by_id[t["trajectory_id"]], by_id[ref["trajectory_id"]]
            ids_a, ids_b = {q[1] for q in t["queries"]}, {q[1] for q in ref["queries"]}
            ratio = a["endpoint_volume"] / b["endpoint_volume"]
            paired.append(dict(h11=t["h11"], group=t["group"], polytope_index=t["polytope_index"],
                checkpoint=t["checkpoint"], algorithm=t["algorithm"], mode=t["mode"], proposal_count=t["proposal_count"],
                comparison=comparison, trajectory_id=t["trajectory_id"], reference_id=ref["trajectory_id"],
                endpoint_ratio=ratio, log10_endpoint_ratio=math.log10(ratio),
                outcome="win" if ratio > 1 + TOL else "loss" if ratio < 1 / (1 + TOL) else "tie",
                state_jaccard=len(ids_a & ids_b) / len(ids_a | ids_b),
                endpoint_a=a["endpoint_volume"], endpoint_b=b["endpoint_volume"]))
    write_csv(destination / "trajectory_metrics.csv", metrics)
    write_csv(destination / "paired_comparisons.csv", paired)
    write_csv(destination / "initial_actor_selection.csv", initial_selection)
    write_csv(destination / "graph_summary.csv", graph_summary)
    write_json(destination / "validation.json", dict(
        source_runs=len(json.loads((destination / "provenance.json").read_text())),
        trajectories=len(trajectories), polytopes=len(graphs), comparison_budget=BUDGET,
        objective_queries_performed=0, policy_inferences=0, geometry_calls=0,
        native_source_validation="passed", raw_logs_modified=False,
        cross_source_metric_conflict_nodes=sum(g["metric_conflict_nodes"] for g in graph_summary),
        metric_relative_tolerance=TOL, graph_definition="undirected union of observed query source/target two_face edges; excludes GA mutations",
        graph_scope="partial observed graph; no claim of full graph coverage, shortest distance, or global optimum",
        checks=["saved full FRST initial pairing", "checkpoint hashes and recorded score definitions",
                "native result counts, cumulative best, budgets and source success", "raw source hashes",
                "query-prefix endpoints", "graph search uniqueness and parent-depth consistency",
                "connected observed flip graph", "ordinary BeFS frontier replay"]))
    print(f"Analyzed {len(trajectories)} trajectories on {len(graphs)} polytopes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reuse_derived", action="store_true")
    parser.add_argument("--historical_root", type=Path, default=ROOT)
    args = parser.parse_args()
    ARCHIVE_ROOT = args.historical_root.resolve()
    SOURCES = {h11: ARCHIVE_ROOT / source.relative_to(ROOT) for h11, source in SOURCES.items()}
    if not args.reuse_derived:
        ingest(args.output_dir)
    analyze(args.output_dir)
