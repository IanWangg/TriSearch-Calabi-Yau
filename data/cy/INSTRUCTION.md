# General Instruction
Inspect the implementation in this folder. Do the following things:
1. Check whether the data collection logic follows the following piece of pseudo code. Previous implementation summary is in `SUMMARY.md`.
2. If not, re-implement it to follow the following logic.

Note:
- Triangulation that is fine and regular (FRT) is effectively FRST, turning a triangulation into a star triangulation is very easy. One can simply assign the height of the origin to be sufficiently low and obtain a star triangulation.
- When doing BFS, we only search in the regular triangulation space by using `only_regular=True` when searching neighbor triangulations. Regular flips do not break the connectivity of the graph. 

You may find the code of `cytools` in `external_reference/cytools/`. It is only there for reference, the `cytools` package is already installed in `sage` environment.

Summarize your implementation, your problem encountered into a write-up called: REIVISION.md.

## Problems to note
Keep the following problem in mind when you developing. Include them in your write-up if you encounter the problems.

- Can't find wall problem: The `random_triangulations_fair` can sometimes cause some problem, throwing warning: "Couldn't find wall.". If this function causes the problem, investigate the error. If there is no fix without touching the `cytools` package, use `random_triangulations_fast` instead. 
- TOPCOM can't find neighbor problem: Sometimes the `neighbor_triangulations` throws warning that there is no neighbor trianguations. If you encounter this, do a sanity check: The current triangulation must have only one simplex. Otherwise, a regular triangulation must have at least one regular neighbor triangulation. 

## Purpose of the dataset
The dataset is to prepare a set of initial regular state to train an RL algorithm to navigate within the regular triangulation space to find the nearest FRST. The collection mechanism should have the following guarantees:
- There is an FRST (FRT) nearby. So it is guaranteed to find one within certain steps.
- The shortest path is known (from BFS), so we can measure the performance of the RL policy.

## Core implementation requirement:
- Use multi-processing to accelerate the collection.

The logic I want the data collection mechanism to have is the following:
```python
from cytools.polytope import Polytope
from cytools.triangulation import Triangulation
import numpy as np
from collections import deque
from typing import List, Dict

# Set N, M and K for example
N: int = ...
M: int = ...
K: int = ...
max_depths: int = ...
depths_to_collect: int = ...
collect_all: bool = ...

# Suppose BFS is a function to perform BFS on the regular subspace of the flip graph
def BFS(triangulation, depth):
    ...
    collection: Dict[Triangulation, int] = {} # key: triangulation, value: known distance to FRST (depth)
    queue = deque([triangulation])
    while ...:
        t = queue.popleft()
        neighbors = t.neighbor_triangulations(only_regular=True, only_fine=False, only_star=False)
        ...
        # This loop should keep track of the depth of the BFS
        # NOTE: The triangulation that is fine and regular can be easily converted to FRST.
        #       Thus, the collection should only contain the triangulations that are not fine. 

    return collection

# Load N lattice polytopes
m_polytopes: List[Polytope] = load_n_polytopes(N)
n_polytopes: List[Polytope] = [m_polytope.dual_polytope() for m_polytope in m_polytopes]

# For each polytope, get M FRST
# Use multi-processing to accelerate the following part
frsts = []
for polytope in n_polytopes:
    frst_ = polytope.random_triangulations_fair(M)
    # NOTE: Check the length of frst_ and the output of the line above.
    #       If it does not work, we may use: frst_ = polytope.random_triangulations_fast(M, only_fine=True) instead. 
    frsts.extend(frst_)

collected_states = {}

for frst in frsts:
    neighborhood_triangulations = BFS(frst, depth)

    if collect_all:
        selected_keys = list(neighborhood_triangulations.keys) 
    else:
        selected_keys = np.random.choice(list(neighborhood_triangulations.keys), size=K)

    for key in selected_keys:
        # be careful about the repeatitive states
        collected_states[key] = min(neighborhood_triangulations[key], collected_states.get(key, float('inf')))

# return collected_states
```