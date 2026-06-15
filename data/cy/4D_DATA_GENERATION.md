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