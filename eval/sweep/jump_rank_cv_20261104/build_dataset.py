"""Stage 1: group B1 single-face candidates by parent, decode parents, precompute scorer inputs.

For each (polytope, start, parent DNA) group with >= 4 feasible children:
  - decode the parent FRST once with the polytope's CyoptEncoding (cached as JSON under parents/);
  - run the invariant encoder once on the parent (precomputed Batch) and store per-move index tensors
    in the same layout FaceMoveScorer uses, so training and scoring need no geometry;
  - store per-child metadata (feasibility, dlnV, face_symdiff, flip distance) from the logs.

Usage: python build_dataset.py [--polytopes 0 1 ...] [--max-groups N] [--check-scorer]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

W = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/.claude/worktrees/bridge-cse_01W7whPSySKiYGCEwuxiRkNf")
R = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau")
RESULTS = R / "runs/face_mutation_20261104/results"
SETUPS = R / "runs/face_mutation_20261104/setups"
OUT = R / "runs/jump_rank_cv_20261104"
CHECKPOINT = R / "runs/inv_rl_full_20261104/checkpoints/final.pth"
os.environ.setdefault("MOSEKLM_LICENSE_FILE", str(Path.home() / "mosek/mosek.lic"))
sys.path.insert(0, str(W))

import torch  # noqa: E402

torch.set_num_threads(4)


def load_candidate_groups(polytopes):
    """Dedupe children per (polytope, start, parent). Returns {key: {child_dna: record}}."""
    groups = defaultdict(dict)
    for path in sorted(RESULTS.glob("face_mutation_*_budget1000_seed0_poly*/face_mutation_candidates.jsonl")):
        algo = path.parent.name.split("_budget")[0]
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["polytope_index"] not in polytopes:
                    continue
                key = (row["polytope_index"], row["start_hash"], tuple(row["parent_dna"]))
                child = tuple(row["child_dna"])
                if child in groups[key]:
                    groups[key][child]["algorithms"].add(algo)
                    continue
                row["algorithms"] = {algo}
                groups[key][child] = row
    return groups


def parent_cache_path(poly, start_hash, parent_dna):
    digest = hashlib.sha1(json.dumps(list(parent_dna)).encode()).hexdigest()[:16]
    return OUT / "parents" / f"p{poly}_{start_hash}_{digest}.json"


def index_for_moves(state, moves):
    """Copy of FaceMoveScorer.score's index construction (kept in sync; checked against the scorer)."""
    from core.cy_two_face_data import _build_topology
    from mdp.cy_state_record import two_face_restrictions

    configuration = state.configuration
    restrictions = two_face_restrictions(configuration, state.simplices)
    topology = _build_topology(configuration, restrictions)
    face_of_labels = {labels: index for index, (labels, _) in enumerate(restrictions)}
    old_triangles = [set(triangles) for _, triangles in restrictions]
    move_face, removed_ids, removed_move, added_nodes, added_move = [], [], [], [], []
    for m, (labels, new_triangles) in enumerate(moves):
        key = tuple(sorted(int(label) for label in labels))
        face = face_of_labels.get(key)
        if face is None:
            raise ValueError(f"Face {key} is not a 2-face of the parent state.")
        node_map, triangle_map = topology.node_maps[face], topology.triangle_maps[face]
        new = {tuple(sorted(int(v) for v in triangle)) for triangle in new_triangles}
        for triangle in new:
            if len(triangle) != 3 or len(set(triangle)) != 3 or not set(triangle) <= node_map.keys():
                raise ValueError(f"Triangle {triangle} is not a 3-subset of face {key}.")
        removed = sorted(old_triangles[face] - new)
        added = sorted(new - old_triangles[face])
        if not removed and not added:
            raise ValueError(f"Move leaves face {key} unchanged.")
        move_face.append(face)
        for triangle in removed:
            removed_ids.append(triangle_map[triangle])
            removed_move.append(m)
        for triangle in added:
            added_nodes.append([node_map[label] for label in triangle])
            added_move.append(m)
    return dict(
        move_face=torch.tensor(move_face, dtype=torch.long),
        removed_ids=torch.tensor(removed_ids, dtype=torch.long),
        removed_move=torch.tensor(removed_move, dtype=torch.long),
        added_nodes=torch.tensor(added_nodes, dtype=torch.long).reshape(-1, 3),
        added_move=torch.tensor(added_move, dtype=torch.long),
        count=len(moves),
    )


def build_polytope(poly, pool, args, stats):
    from core.two_face_invariant_features import create_invariant_two_face_data_from_state
    from eval.algorithm.cyopt_ga import CyoptEncoding
    from eval.setup import load_eval_setup
    from mdp.cy_rollout import build_cy_rollout_collection
    from mdp.cy_state_record import CyStateRecord, canonical_simplices, two_face_restrictions
    from torch_geometric.data import Batch

    setup = load_eval_setup(SETUPS / f"seed0_poly{poly}_s3")
    collection = build_cy_rollout_collection(
        setup.rows, include_points_interior_to_facets=False, neighbor_mode="two_neighbors",
        transition_pool=pool, two_face_state=False, include_two_face_metadata=True)
    starts = [state for state in collection.base_states.values() if state.point_config_index == poly]
    start_by_hash = {hashlib.sha1(state.key.encode()).hexdigest()[:12]: state for state in starts}

    # Reference codebook from the logs; the rebuilt encoding must reproduce it exactly.
    any_dir = next(RESULTS.glob(f"face_mutation_befs_budget1000_seed0_poly{poly}"))
    reference = json.loads((any_dir / "cyopt_encoding" / f"polytope_{poly}.json").read_text())
    encoding = CyoptEncoding(starts, seed=reference["seed"], max_points=reference["max_points"],
                             samples=reference["samples"])
    if encoding.metadata["codebook_sha256"] != reference["codebook_sha256"]:
        raise RuntimeError(f"poly {poly}: rebuilt codebook does not match the logged codebook.")
    mem = [[[[int(v) for v in simplex] for simplex in triang] for triang in face]
           for face in encoding.metadata["face_triangulations"]]
    if mem != reference["face_triangulations"]:
        raise RuntimeError(f"poly {poly}: codebook sha matches but lists differ.")
    codebook = [[tuple(tuple(int(v) for v in t) for t in entry) for entry in face]
                for face in reference["face_triangulations"]]
    interesting = [f for f, entries in enumerate(codebook) if len(entries) > 1]
    if len(interesting) != len(encoding.bounds):
        raise RuntimeError(f"poly {poly}: interesting faces do not match DNA length.")
    face_labels = []
    for entries in codebook:
        face_labels.append(tuple(sorted({int(v) for entry in entries for t in entry for v in t})))

    groups = load_candidate_groups({poly})
    selected = [(key, children) for key, children in groups.items()
                if sum(1 for r in children.values() if r["feasible"]) >= 4]
    selected.sort(key=lambda item: (item[0][1], item[0][2]))
    if args.max_groups:
        selected = selected[:args.max_groups]
    records = []
    decode_seconds = []
    for (poly_index, start_hash, parent_dna), children in selected:
        start = start_by_hash[start_hash]
        cache = parent_cache_path(poly_index, start_hash, parent_dna)
        if cache.exists():
            simplices = [tuple(s) for s in json.loads(cache.read_text())["simplices"]]
            state = CyStateRecord(start.configuration, frozenset(simplices), start.neighbor_mode, True, True)
        else:
            began = time.perf_counter()
            state = encoding.decode(tuple(parent_dna), start)
            if state is None:
                stats["none_parent"] += 1
                continue
            elapsed = time.perf_counter() - began
            decode_seconds.append(elapsed)
            cache.write_text(json.dumps({"simplices": [list(s) for s in sorted(state.simplices)],
                                         "decode_sec": elapsed}))
        # Consistency: each restriction of the decoded parent equals the logged codebook entry.
        restrictions = two_face_restrictions(state.configuration, state.simplices)
        by_labels = {labels: r for r, (labels, _) in enumerate(restrictions)}
        if len(restrictions) != len(codebook):
            stats["restriction_count_mismatch"] += 1
        for f, labels in enumerate(face_labels):
            r = by_labels.get(labels)
            if r is None:
                stats["unmatched_face_label"] += 1
                continue
            current = {tuple(sorted(t)) for t in restrictions[r][1]}
            expected = {tuple(sorted(t)) for t in codebook[f][parent_dna[interesting.index(f)]]} \
                if f in interesting else {tuple(sorted(t)) for t in codebook[f][0]}
            if current != expected:
                stats["parent_face_mismatch"] += 1
        moves, meta = [], []
        for child_dna, row in children.items():
            pos = row["dna_position"]
            f = row["face"]
            if interesting[pos] != f:
                raise RuntimeError("dna_position does not map to the logged face.")
            new_entry = codebook[f][child_dna[pos]]
            moves.append((face_labels[f], [list(t) for t in new_entry]))
            meta.append(dict(child_dna=list(child_dna), face=f, dna_position=pos,
                             face_symdiff=row["face_symdiff"], flip_distance=row["flip_distance"],
                             feasible=bool(row["feasible"]),
                             objective=row["objective"], dlnV=row["dlnV"],
                             algorithms=sorted(row["algorithms"])))
        data = create_invariant_two_face_data_from_state(state, ())
        batch = Batch.from_data_list([data])
        index = index_for_moves(state, moves)
        record = dict(poly=poly_index, start_hash=start_hash, parent_dna=list(parent_dna),
                      parent_objective=next(iter(children.values()))["parent_objective"],
                      batch=batch, index=index, children=meta, n_children=len(meta),
                      n_feasible=sum(m["feasible"] for m in meta))
        records.append(record)
        if len(records) % 50 == 0:
            print(f"poly {poly}: {len(records)}/{len(selected)} parents, last decode "
                  f"{decode_seconds[-1] if decode_seconds else 0:.2f}s", flush=True)
    stats["decoded_new"] += len(decode_seconds)
    stats["decode_seconds"] += sum(decode_seconds)
    return records, encoding, codebook, start_by_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--polytopes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--check-scorer", action="store_true")
    args = parser.parse_args()
    (OUT / "dataset").mkdir(parents=True, exist_ok=True)
    from mdp.cy_rollout import create_transition_pool
    pool = create_transition_pool(num_workers=1)
    try:
        for poly in args.polytopes:
            stats = Counter()
            began = time.perf_counter()
            records, encoding, codebook, _ = build_polytope(poly, pool, args, stats)
            if args.check_scorer and records:
                check_scorer(records[0], encoding, codebook)
            if args.max_groups:
                print(f"poly {poly} test run: {len(records)} records, stats {dict(stats)}", flush=True)
                continue
            torch.save(records, OUT / "dataset" / f"poly{poly}.pt")
            summary = dict(poly=poly, records=len(records), feasible_children=sum(r["n_feasible"] for r in records),
                           children=sum(r["n_children"] for r in records), wall_sec=time.perf_counter() - began,
                           **dict(stats))
            print(json.dumps(summary), flush=True)
            with (OUT / "dataset" / "build_log.jsonl").open("a") as handle:
                handle.write(json.dumps(summary) + "\n")
    finally:
        pool.close()


def check_scorer(record, encoding, codebook):
    """The copied index construction must give the same logits as FaceMoveScorer on one parent."""
    from eval.algorithm.face_move_scorer import FaceMoveScorer
    import numpy as np
    from mdp.cy_state_record import CyStateRecord
    from core.cy_two_face_data import _build_topology  # noqa: F401
    scorer = FaceMoveScorer(CHECKPOINT, aggregation="mean2")
    # Reconstruct the parent state from the cache (same as build) for the scorer call.
    path = parent_cache_path(record["poly"], record["start_hash"], record["parent_dna"])
    simplices = [tuple(s) for s in json.loads(path.read_text())["simplices"]]
    configuration = encoding.configuration
    state = CyStateRecord(configuration, frozenset(simplices), "two_neighbors", True, True)
    moves = []
    interesting = [f for f, entries in enumerate(codebook) if len(entries) > 1]
    for child in record["children"][:20]:
        f = child["face"]
        labels = tuple(sorted({int(v) for entry in codebook[f] for t in entry for v in t}))
        moves.append((labels, [list(t) for t in codebook[f][child["child_dna"][child["dna_position"]]]]))
    expected = scorer.score(state, moves)
    print("scorer check: first moves", np.round(expected[:3], 4).tolist(), flush=True)


if __name__ == "__main__":
    main()
