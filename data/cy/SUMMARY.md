# CY Data Generation Pipeline Summary

Implemented folder: `cy_data_generation/`

## What Was Added

- `cy_data_generation/pipeline.py`
  - Full dataset generation pipeline for CY triangulation data.
  - Core steps:
    1. Load `n` reflexive polytopes from `k3.txt`.
    2. Convert each from M-lattice to N-lattice (`dual_polytope`).
    3. Generate FRST seeds via `random_triangulations_fair`.
    4. Run BFS from FRST seeds to sample regular triangulations and compute distance from the source FRST.
       - Optional depth-filtered collection mode:
         - collect only non-FRST states at specified BFS depths
         - either keep all collected states or randomly subsample
    5. Store polytope data, FRST seeds, sampled triangulations, nearest/source FRST, and BFS distance/statistics.
  - Includes robust fallback for neighbor-BFS backend incompatibilities in `cytools`.

- `cy_data_generation/generate_dataset.py`
  - CLI entry point for running the full pipeline and saving outputs.
  - BFS controls are exposed via both:
    - `--bfs-max-depth` (aliases: `--max-depth`, `--max_depth`)
    - `--bfs-max-nodes` (aliases: `--max-nodes`, `--max_nodes`, `--max-node`, `--max_node`)
  - Depth-filtered collection controls:
    - `--collection-depths` / `--collection_depths` (list of depths, e.g. `1 2 3`)
    - `--collection-all` / `--collection_all` (collect all matching states instead of subsampling)

- `cy_data_generation/__init__.py`
  - Exports key pipeline functions.

## Dataset Contents

For each polytope and each random sample, the dataset stores:

- Polytope information:
  - source header
  - M-space vertices
  - N-space points
  - lattice space marker (`"N"`)
- Generated regular triangulation:
  - simplices
  - points used
  - regular/fine/star/FRST flags
  - canonical triangulation signature
- Random heights:
  - In FRST-first mode, `random_triangulations_fair` does not expose explicit height vectors.
  - Dataset keeps `heights: null` with `height_generation_method` metadata.
- BFS results:
  - nearest FRST triangulation snapshot (if found)
  - distance to nearest FRST
  - visited/expanded node counts
  - truncated flag and stop reason

Saved artifacts:

- `<output_name>.json`: full nested dataset
- `<output_name>.samples.jsonl`: flattened sample-level records

## How To Run

From repo root (in `sage` env):

```bash
python cy_data_generation/generate_dataset.py \
  --k3-path cy_data/k3.txt \
  --num-polytopes 5 \
  --triangulations-per-polytope 4 \
  --frsts-per-polytope 2 \
  --max-depth 5 \
  --max-node 500 \
  --collection-depths 1 2 3 \
  --collection_all \
  --seed 2026 \
  --output-dir cy_data_generation/output \
  --output-name cy_reflexive_dataset
```

## Testing

New tests:

- `test/test_cy_data_generation_pipeline.py`
  - Verifies BFS returns distance 0 for a known FRST start.
  - Validates FRST-first dataset structure and saved output files.
  - Runs a larger reproducibility integration test (`4 polytopes x 4 triangulations`).

Test result in `sage` environment:

- `pytest -q test/test_cy_data_generation_pipeline.py` -> `6 passed`
- Full suite: `pytest -q` -> all tests pass.
