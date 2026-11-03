# Favorable v1 dataset

Private Hugging Face dataset `ywang3/trisearch-cy-favorable-v1`. All polytopes
are favorable 4D reflexive polytopes from `calabi-yau-data/polytopes-4d`
(revision `60c0e119a03608418df538191f65da3f43b5b819`).

| Split | h11 | Polytopes | Starts per polytope | Start samplers |
|---|---|---|---|---|
| train | 15 | 3000 | 4 | fast + uniform NTFE (or farthest-point) |
| train | 20 | 1000 | 4 | fast + farthest-point (or uniform NTFE) |
| dev | 15, 20 | 30 each | 1 | fast |
| test | 25, 30 | 29, 30 | 2 | fast |

The test split is for final results only. Do not use it for training or
checkpoint selection.

## How it was made

```bash
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
python -m data.cy.select_polytopes \
  --splits train:15:3000 train:20:1000 dev:15:30 dev:20:30 test:25:30 test:30:30 \
  --seed 0 --output-dir data/cy/favorable_v1
R="python -m data.cy.generate_diverse_starts --num-workers 12 --seed 0"
$R --polytopes data/cy/favorable_v1/train/polytopes_h11_15.jsonl \
   --output data/cy/favorable_v1/train/h11_15.samples.jsonl --mode train --starts 4 --min-starts 3
$R --polytopes data/cy/favorable_v1/test/polytopes_h11_25.jsonl \
   --output data/cy/favorable_v1/test/h11_25.samples.jsonl --mode eval --starts 2 --min-starts 2
# (same for train h11=20, dev with --starts 1, test h11=30)
python -m data.cy.write_dataset_manifest --dataset-dir data/cy/favorable_v1 \
  --index ~/.cache/trisearch_h12_index/h12_index_60c0e119a036.json
```

- **Polytopes:** uniform over all source rows with `h12 == h11` (the source
  `h12` is the N-lattice CY h11), then checked with CYTools for
  `h11(lattice="N")` and favorability. The first 2000 matching rows in source
  order are skipped, because older evaluation setups took polytopes in that
  order. The splits are disjoint.
- **Starts:** FRSTs with `make_star=True` and no facet-interior points,
  distinct by 2-face restriction (NTFE class). Each start has `start_metadata`:
  sampler, seed, 2-face hash, `num_two_neighbors`, and the initial `max_kcup`
  log volume.
- **Format:** `.samples.jsonl` rows load with `load_cy_sample_rows` and
  `build_cy_rollout_collection(..., neighbor_mode="two_neighbors",
  include_points_interior_to_facets=False)`. There are no non-fine
  triangulations.

`manifest.json` in the dataset has the parameters, versions, file hashes, and
diversity statistics.
