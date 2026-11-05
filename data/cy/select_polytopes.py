"""Select favorable 4D polytopes uniformly at random from calabi-yau-data/polytopes-4d.

The source files are ordered by vertex count, so taking the first matches gives
only small polytopes. This script indexes the h12 column of every vertex-count
file, draws rows uniformly at random among rows with h12 == target h11 (source
h12 is the N-lattice CY h11), and keeps rows that CYTools confirms as
h11(lattice="N") == target and favorable. All splits are disjoint.

```bash
python -m data.cy.select_polytopes \
  --splits train:15:3000 train:20:1000 dev:15:30 dev:20:30 test:25:30 test:30:30 \
  --seed 0 --index-dir ~/.cache/trisearch_h12_index --output-dir data/cy/favorable_v1
```
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

HF_REPO_ID = "calabi-yau-data/polytopes-4d"
HF_REVISION = "60c0e119a03608418df538191f65da3f43b5b819"
SPLIT_ORDER = ("train", "dev", "test")


def parse_split_specs(specs: Sequence[str]) -> List[Tuple[str, int, int]]:
    parsed = []
    for spec in specs:
        split, h11, count = spec.split(":")
        if split not in SPLIT_ORDER:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_ORDER}.")
        parsed.append((split, int(h11), int(count)))
    if len({(split, h11) for split, h11, _ in parsed}) != len(parsed):
        raise ValueError("Each split:h11 pair may appear only once.")
    return parsed


def _parquet_files(fs, revision: str) -> List[str]:
    root = f"datasets/{HF_REPO_ID}@{revision}"
    return sorted(path for path in fs.ls(root, detail=False) if path.endswith("-vertices.parquet"))


def build_h12_index(h_values: Iterable[int], *, revision: str, index_path: Path) -> Dict:
    """Index rows by h12 for every source file; reuses index_path when it covers h_values."""
    h_values = sorted({int(h) for h in h_values})
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["revision"] == revision and set(h_values) <= {int(h) for h in index["h_values"]}:
            return index
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files: Dict[str, Dict] = {}
    start = time.time()
    for path in _parquet_files(fs, revision):
        parquet = pq.ParquetFile(fs.open(path, block_size=8 * 2**20))
        row_groups = [parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups)]
        rows = {str(h): [] for h in h_values}
        offset = 0
        for group, count in enumerate(row_groups):
            h12 = parquet.read_row_group(group, columns=["h12"]).column("h12").to_numpy()
            for h in h_values:
                rows[str(h)].extend((offset + np.nonzero(h12 == h)[0]).tolist())
            offset += count
        name = path.rsplit("/", 1)[-1]
        files[name] = {"num_rows": offset, "row_groups": row_groups, "rows": rows}
        print(f"[index] {name}: {offset} rows, " + ", ".join(
            f"h12={h}: {len(rows[str(h)])}" for h in h_values) + f" ({time.time() - start:.0f}s)", flush=True)
    index = {"repo_id": HF_REPO_ID, "revision": revision, "h_values": h_values, "files": files,
             "created_at": datetime.now(timezone.utc).isoformat()}
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index))
    return index


def candidate_rows(index: Mapping, h11: int, *, exclude_first: int) -> List[Tuple[str, int]]:
    """All (file, row) with h12 == h11 in source order, minus the first `exclude_first`.

    Existing evaluation setups take matches in source order, so dropping the
    head of that order keeps random splits disjoint from them.
    """
    rows = [(name, int(row)) for name in sorted(index["files"])
            for row in index["files"][name]["rows"].get(str(h11), [])]
    return rows[int(exclude_first):]


def permuted_candidates(rows: Sequence[Tuple[str, int]], *, seed: int, h11: int) -> List[Tuple[str, int]]:
    rng = np.random.default_rng([int(seed), int(h11)])
    return [rows[i] for i in rng.permutation(len(rows))]


def row_group_of(index: Mapping, name: str, row: int) -> Tuple[int, int]:
    bounds = np.cumsum([0] + list(index["files"][name]["row_groups"]))
    group = int(np.searchsorted(bounds, row, side="right") - 1)
    return group, int(bounds[group])


def vertex_key(vertices) -> Tuple[Tuple[int, ...], ...]:
    return tuple(sorted(tuple(int(c) for c in vertex) for vertex in vertices))


def hull_key(vertices) -> Tuple[Tuple[int, ...], ...]:
    """Canonical polytope key: sorted convex-hull vertices, independent of which lattice points are listed."""
    from scipy.spatial import ConvexHull

    points = np.asarray(vertices, dtype=np.int64)
    corners = ConvexHull(points.astype(float)).vertices
    return tuple(sorted(tuple(int(c) for c in points[i]) for i in corners))


def load_exclusions(paths: Sequence[str]):
    """Polytopes to exclude: JSONL rows with `vertices` (and optional `source_id`) or a setup.json."""
    keys, ids, report = set(), set(), []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.suffix == ".json":
            items = [{"vertices": spec["vertices"]} for spec in json.loads(path.read_text())["metadata"]["polytope_specs"]]
        else:
            items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for item in items:
            keys.add(hull_key(item["vertices"]))
            if item.get("source_id"):
                ids.add(item["source_id"])
        report.append({"path": str(path), "polytopes": len(items)})
    return keys, ids, report


def assign_splits(accepted: Sequence[Dict], requests: Sequence[Tuple[str, int]], *, seed: int, h11: int):
    """Shuffle accepted polytopes of one h11 and cut them into the requested splits."""
    rng = np.random.default_rng([int(seed), int(h11), 1])
    order = [accepted[i] for i in rng.permutation(len(accepted))]
    out, start = {}, 0
    for split, count in sorted(requests, key=lambda item: SPLIT_ORDER.index(item[0])):
        out[split] = order[start:start + count]
        start += count
    return out


def _read_row_groups(index: Mapping, wanted: Mapping[Tuple[str, int], List[int]], revision: str, threads: int):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    root = f"datasets/{HF_REPO_ID}@{revision}/"

    def read(key):
        name, group = key
        table = pq.ParquetFile(fs.open(root + name, block_size=8 * 2**20)).read_row_group(
            group, columns=["vertices", "vertex_count", "point_count", "h12"])
        offset = row_group_of(index, name, wanted[key][0])[1]
        return {row: table.slice(row - offset, 1).to_pylist()[0] for row in wanted[key]}

    records = {}
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for key, rows in zip(wanted, pool.map(read, list(wanted))):
            for row, record in rows.items():
                records[(key[0], row)] = record
    return records


def select_bucket(index: Mapping, h11: int, needed: int, *, seed: int, exclude_first: int,
                  revision: str, threads: int, seen_keys: set, oversample: float = 1.3,
                  excluded_keys: frozenset = frozenset(), excluded_ids: frozenset = frozenset()):
    from cytools import Polytope

    order = permuted_candidates(candidate_rows(index, h11, exclude_first=exclude_first), seed=seed, h11=h11)
    accepted, position = [], 0
    rejected = {"h11_mismatch": 0, "not_favorable": 0, "duplicate": 0, "excluded_prior": 0}
    while len(accepted) < needed:
        if position >= len(order):
            raise ValueError(f"Only {len(accepted)} favorable h11={h11} polytopes available; needed {needed}.")
        batch = order[position:position + max(16, int((needed - len(accepted)) * oversample))]
        position += len(batch)
        wanted: Dict[Tuple[str, int], List[int]] = {}
        for name, row in batch:
            wanted.setdefault((name, row_group_of(index, name, row)[0]), []).append(row)
        start = time.time()
        records = _read_row_groups(index, wanted, revision, threads)
        for name, row in batch:
            if len(accepted) >= needed:
                break
            record = records[(name, row)]
            if int(record["h12"]) != h11:
                raise ValueError(f"Index mismatch at {name}:{row}.")
            key = vertex_key(record["vertices"])
            if key in seen_keys:
                rejected["duplicate"] += 1
                continue
            if f"{revision}:{name}:{row}" in excluded_ids or hull_key(record["vertices"]) in excluded_keys:
                rejected["excluded_prior"] += 1
                continue
            polytope = Polytope(record["vertices"])
            if int(polytope.h11(lattice="N")) != h11:
                rejected["h11_mismatch"] += 1
                continue
            if not polytope.is_favorable(lattice="N"):
                rejected["not_favorable"] += 1
                continue
            seen_keys.add(key)
            accepted.append({
                "source_id": f"{revision}:{name}:{row}", "source_file": name, "source_row": row,
                "h11": h11, "favorable": True, "vertices": record["vertices"],
                "vertex_count": int(record["vertex_count"]), "source_point_count": int(record["point_count"]),
                "num_points_not_interior_to_facets": int(len(polytope.points_not_interior_to_facets())),
            })
        print(f"[select] h11={h11}: {len(accepted)}/{needed} accepted after {position} candidates "
              f"({len(wanted)} row groups, {time.time() - start:.0f}s)", flush=True)
    return accepted, {"candidates_examined": position, "rejected": rejected,
                      "eligible_rows": len(order)}


def vertex_count_distribution(index: Mapping, h11: int) -> Dict[str, int]:
    counts = {}
    for name, entry in index["files"].items():
        count = len(entry["rows"].get(str(h11), []))
        if count:
            counts[name.split("-")[2]] = count
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", required=True, help="split:h11:count, e.g. train:15:3000.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--revision", default=HF_REVISION)
    parser.add_argument("--index-dir", default="~/.cache/trisearch_h12_index")
    parser.add_argument("--exclude-first-source-order", type=int, default=2000,
                        help="Skip the first N h12-matching rows in source order (used by existing eval setups).")
    parser.add_argument("--threads", type=int, default=4, help="Parallel row-group downloads.")
    parser.add_argument("--exclude-from", nargs="*", default=[],
                        help="Polytopes that must not be selected: JSONL files with `vertices` (and `source_id`) "
                             "or a setup.json with metadata.polytope_specs. Matched by convex-hull vertices.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from core.cytools_config import configure_cytools

    configure_cytools()
    specs = parse_split_specs(args.splits)
    h_values = sorted({h11 for _, h11, _ in specs})
    index_path = Path(args.index_dir).expanduser() / f"h12_index_{args.revision[:12]}.json"
    index = build_h12_index(h_values, revision=args.revision, index_path=index_path)
    excluded_keys, excluded_ids, exclusion_sources = load_exclusions(args.exclude_from)

    output_dir = Path(args.output_dir).expanduser()
    seen_keys: set = set()
    manifest = {"repo_id": HF_REPO_ID, "revision": args.revision, "seed": args.seed,
                "splits": args.splits, "exclude_first_source_order": args.exclude_first_source_order,
                "selection": "uniform over h12==h11 rows of all vertex-count files, then CYTools h11(N) and "
                             "favorable check; accepted polytopes shuffled and cut into splits",
                "exclusions": {"sources": exclusion_sources, "hull_keys": len(excluded_keys),
                               "source_ids": len(excluded_ids), "match_rule": "convex-hull vertex set "
                               "(scipy ConvexHull on the listed lattice points), or source_id equality"},
                "buckets": {}, "created_at": datetime.now(timezone.utc).isoformat()}
    next_index = 0
    for h11 in h_values:
        requests = [(split, count) for split, h, count in specs if h == h11]
        accepted, stats = select_bucket(index, h11, sum(count for _, count in requests), seed=args.seed,
                                        exclude_first=args.exclude_first_source_order,
                                        revision=args.revision, threads=args.threads, seen_keys=seen_keys,
                                        excluded_keys=frozenset(excluded_keys), excluded_ids=frozenset(excluded_ids))
        # Independent post-check on the accepted set: recompute keys and compare with the exclusion sets.
        stats["exclusion_check"] = {
            "accepted": len(accepted),
            "accepted_matching_excluded_hull_key": sum(hull_key(r["vertices"]) in excluded_keys for r in accepted),
            "accepted_matching_excluded_source_id": sum(r["source_id"] in excluded_ids for r in accepted),
        }
        for split, rows in assign_splits(accepted, requests, seed=args.seed, h11=h11).items():
            for row in rows:
                row.update(polytope_index=next_index, split=split)
                next_index += 1
            path = output_dir / split / f"polytopes_h11_{h11}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            stats.setdefault("outputs", {})[split] = {"path": str(path.relative_to(output_dir)), "count": len(rows)}
        stats["vertex_count_of_selected"] = dict(sorted(
            {str(v): sum(r["vertex_count"] == v for r in accepted) for v in {r["vertex_count"] for r in accepted}}.items(),
            key=lambda item: int(item[0])))
        stats["h12_rows_per_vertex_file"] = vertex_count_distribution(index, h11)
        manifest["buckets"][str(h11)] = stats
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"Wrote {next_index} polytopes to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
