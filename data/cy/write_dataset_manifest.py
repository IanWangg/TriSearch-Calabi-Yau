"""Combine selection and generation summaries into one dataset manifest.

```bash
python -m data.cy.write_dataset_manifest --dataset-dir data/cy/favorable_v1 \
  --index ~/.cache/trisearch_h12_index/h12_index_60c0e119a036.json
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def _version(package: str):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _git(*args: str):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--index", required=True, help="h12 index JSON written by select_polytopes.py.")
    parser.add_argument("--name", default="trisearch-cy-favorable-v1", help="Dataset name written to manifest.json.")
    args = parser.parse_args()

    root = Path(args.dataset_dir).expanduser()
    selection = json.loads((root / "selection_manifest.json").read_text())
    index = json.loads(Path(args.index).expanduser().read_text())
    files = {}
    for path in sorted(root.glob("*/h11_*.samples.jsonl")):
        summary = json.loads(path.with_suffix(".summary.json").read_text())
        summary.pop("errors", None)
        summary.pop("polytopes_file", None)
        files[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": sum(1 for line in path.open() if line.strip()),
            "generation": summary,
        }
    population = {}
    for h11 in selection["buckets"]:
        counts = {}
        for name, entry in index["files"].items():
            # Index file names look like polytopes-4d-14-vertices.parquet.
            count = len(entry["rows"].get(str(h11), []))
            if count:
                counts[str(int(name.split("-")[2]))] = count
        population[h11] = counts
    manifest = {
        "name": args.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"repo_id": selection["repo_id"], "revision": selection["revision"]},
        "selection": selection,
        "h12_rows_per_vertex_count_in_source": population,
        "files": files,
        "versions": {"cytools": _version("cytools"), "mosek": _version("Mosek") or _version("mosek"),
                     "numpy": _version("numpy"), "pyarrow": _version("pyarrow")},
        "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD")},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"Wrote {root / 'manifest.json'} with {len(files)} sample files.")


if __name__ == "__main__":
    main()
