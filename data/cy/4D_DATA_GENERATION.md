# Instruction on obtaining 4D reflexive polytopes

## Data format
The data format should be the same as the 3D dataset, with example data files:
- `cy_reflexive_dataset_random_flip.checkpoint.json`
- `cy_reflexive_dataset_random_flip.json`
- `cy_reflexive_dataset_random_flip.samples.jsonl`
The core data file is `cy_reflexive_dataset_random_flip.samples.jsonl`, which is the data file passed to training script. It has the following structure per line for example:
```json
{
  "polytope_index": 0,
  "h11": ...,
  "vertices": [[0, 0, 0, 0], [1, 0, 0, 0], ...],
  "frst_list": [
    {
      "frst_index": 0,
      "simplices": [[0, 1, 2, 3, 4], ...],
      "triangulation_list": [
        {
          "distance": 2,
          "simplices": [[0, 1, 2, 3, 4], ...]
        }
      ]
    }
  ],
  "non_fine_triangulation_count": ...
}
```

Your task is to write a script called `generate_4d_dataset.py`.

## CYTools API
Below is an instruction of loading 4D reflexive polytopes in N-lattice with specific h11 number and number of vertices.
```python
from cytools import fetch_polytopes # Note that it can directly be imported from the root
num_vertices: int = ...
h11: int= ...
dim: int = 4
favorable: bool = ...

polytope_lists = fetch_polytopes(h11=h11, dim=dim, favorable=favorable, lattice="N", as_list=True) # Constructs a list of polytopes

polytope_generator = fetch_polytopes(h11=h11, dim=dim, favorable=favorable, lattice="N", as_list=True) # Constructs a list of polytopes
next(polytope_generator)
# A 4-dimensional reflexive lattice polytope in ZZ^4
```

## API of `generate_4d_dataset.py`
Your implementation should allow the following options:
1. Number of polytopes to sample (must be given)
2. h11 number (must be given)
3. number of vertices (can be none)
4. Favorable (default to False)
The rest options can follow `generate_dataset.py`.

## Hugging Face source

`generate_4d_dataset_hugging_face.py` has the same generation options and
output format as `generate_4d_dataset.py`, but reads vertices from
`calabi-yau-data/polytopes-4d`:

```bash
conda activate sage
python data/cy/generate_4d_dataset_hugging_face.py \
  --num-polytopes 100 \
  --h11 12 \
  --num-vertices 8 \
  --favorable \
  --num-triangulations-per-frst 10 \
  --frsts-per-polytope 10 \
  --random-flip \
  --fast \
  --no-include-points-interior-to-facets \
  --compact-output \
  --output-dir data/cy/output4d_hugging_face \
  --output-name cy_4d_h11_12
```

The source table describes its normal-form vertices with the mirror Hodge
convention relative to using those vertices directly in the N lattice.
Consequently, the script filters source `h12` for the requested N-lattice CY
`h11`, then verifies every selected row with `Polytope.h11(lattice="N")`.
Favorability is also evaluated with CYTools because it is not a source column.
Parquet filtering is performed while streaming; the full 15.8 GB dataset is
not loaded into memory. The resolved Hugging Face revision is saved in the
checkpoint/full-dataset metadata.

The `.samples.jsonl` output is accepted directly by TriSearch. For toric CY
volume optimization, generate without facet-interior points as above and use
the FRST two-neighbor mode:

```bash
python scripts/train_cy.py \
  --dataset_path data/cy/output4d_hugging_face/cy_4d_h11_12.samples.jsonl \
  --reward max_toric_cy_volume \
  --neighbor_mode two_neighbors \
  --no-include_points_interior_to_facets
```

For reproducible long or resumed jobs, pass an immutable commit SHA through
`--hf-revision`. `--hf-cache-dir` can be used to select a shared local cache.
