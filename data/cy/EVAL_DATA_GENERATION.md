# Evaluation Data Generate Instruction

The generation pipeline should take the following arguments:
- num_polytopes : number of polytopes
- h11: the value of h11 to sample from
- num_vertices: optional, the number of vertices
- num_tri: maximal number of triangulations to generate per polytope
- max_tries: maximal number of random heights sampling

The goal is the generate a set of non-fine regular triangulations by assigning random heights to the vertices.

The triangulation api in `cytools` is the following:
```python
from cytools.polytope import Polytope
import numpy as np

p: Polytope = ...
heights = np.random.rand(len(p.points()))
tri_with_heights = p.triangulate(include_points_interior_to_facets=True, heights=..., check_heights=True)
```

Each generated triangulation should be unique, disgard duplicate ones. 

If a generated triangulation is fine, disgard it. 

Generate at most `max_retries` times. 

The data should in the following format.

```
- Polytope List:
    - Polytope 1
        - Vertices: ... # The vertices in N-Lattice space
        - Non-Fine Triangulation List:
            - Triangulation 1:
                - Heights: ...
                - Signiture: ...
            - Triangulation 2:
                - Heights: ...
                - Signiture: ...
            ...
    ...

```