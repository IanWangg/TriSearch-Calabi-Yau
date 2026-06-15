# Data Generation Revision Write-up

## What I changed

I revised `cy_data_generation/pipeline.py` to follow the pseudocode and notes in `cy_data_generation/INSTRUCTION.md`.

1. FRST seed generation flow

- Per polytope, the pipeline now requests `M` fine+regular seeds using `random_triangulations_fair`.
- If fair sampling under-delivers or fails, it falls back to `random_triangulations_fast(..., only_fine=True)`.
- Diagnostics now record fair/fast round outcomes and failures.

1. BFS collection logic (regular subspace)

- BFS now expands neighbors with:
  - `only_regular=True`
  - `only_fine=False`
  - `only_star=False`
- BFS tracks shortest depth from each seed (standard first-visit BFS depth).
- The collected pool includes only **non-fine** triangulations, consistent with the note that fine+regular states are effectively FRST-reachable.

1. Per-FRST selection and global dedup with min distance

- For each FRST seed, BFS produces a neighborhood collection.
- If `collection_all=False`, up to `K` states are sampled from that seed’s collection; otherwise all are taken.
- Repeated states across different seeds are merged by triangulation signature with:
  - `distance = min(distance_seen_so_far, new_distance)`
- This implements the required global nearest-seed distance labeling.

1. Multi-processing

- Added per-polytope parallel collection using `ProcessPoolExecutor`.
- New `num_workers` controls worker count; default is `min(cpu_count, num_polytopes)`.
- Output ordering remains deterministic by sorting worker results by `polytope_index`.

1. CLI and tests

- Added `--num-workers` to `cy_data_generation/generate_dataset.py` and wired it into the pipeline call.
- Updated `test/test_cy_data_generation_pipeline.py` assertions for:
  - new collection semantics (non-fine collected states)
  - variable sample counts after global dedup
  - deterministic single-worker test mode (`num_workers=1`)

## Problem notes encountered

### 1. "Couldn't find wall" in `random_triangulations_fair`

- This warning can cause fair sampling to return fewer triangulations than requested.
- Since this behavior is inside `cytools`, the pipeline-level mitigation is fallback to:
  - `random_triangulations_fast(..., only_fine=True)`
- Implemented this fallback automatically and recorded diagnostics in output metadata.
- I reproduced this warning during a small smoke run (`num_polytopes=1`, low attempt budget), and fallback handling kept the pipeline from crashing.

### 2. TOPCOM neighbor issue (empty neighbors)

- Sometimes neighbor search may return no neighbors unexpectedly.
- Implemented a sanity note in BFS when:
  - `neighbor_triangulations(...)` returns empty **and**
  - triangulation has more than one simplex.
- This is logged in per-source BFS reports (`sanity_notes`) for downstream inspection.
- In tests, the trivial-one-simplex TOPCOM warning also appears; this is expected and now distinguishable from the non-trivial empty-neighbor case.

## Files revised

- `cy_data_generation/pipeline.py`
- `cy_data_generation/generate_dataset.py`
- `test/test_cy_data_generation_pipeline.py`
- `cy_data_generation/REIVISION.md` (this file)

