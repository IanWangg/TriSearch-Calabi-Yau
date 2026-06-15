# CY `.samples.jsonl` Data Description

## Context
- Source: reflexive polytopes from `k3.txt` (loaded in file order, 0-based `polytope_index`).
- Generation: FRST seeds are found per polytope, then non-fine regular triangulations are collected from each FRST (BFS mode or `--random_flip` mode).
- File: `<output-name>.samples.jsonl` contains one JSON object per polytope.
- Note: line order is not guaranteed when multi-processing is used; use `polytope_index` as the stable key.

## Row Format (one line)
```json
{
  "polytope_index": 0,
  "vertices": [[0, 0, 0], [1, 0, 0], ...],
  "frst_list": [
    {
      "frst_index": 0,
      "simplices": [[0, 1, 2, 3], ...],
      "triangulation_list": [
        {
          "distance": 2,
          "simplices": [[0, 1, 2, 3], ...]
        }
      ]
    }
  ],
  "non_fine_triangulation_count": 25
}
```

## Field Semantics
- `polytope_index`: index of the polytope in loaded `k3.txt` sequence (0-based).
- `vertices`: polytope points in **N-lattice** coordinates.
- `frst_list`: FRST seeds found for this polytope.
- `frst_list[*].simplices`: FRST signature in simplex form.
- `triangulation_list`: non-fine regular triangulations associated with that FRST.
- `distance`:
  - BFS mode: flip-graph distance from the FRST.
  - `--random_flip` mode: equals the configured `--max-depth` used in `random_flips(N=depth, only_regular=True)`.
- `simplices`: each simplex entry uses indices into `vertices`.
- `non_fine_triangulation_count`: total number of collected non-fine triangulations for the polytope (sum of all `triangulation_list` lengths).

## Practical Notes
- Requested counts are upper bounds: some polytopes may have fewer samples if FRST generation under-delivers or if unique non-fine samples cannot be collected within the attempt budget.
