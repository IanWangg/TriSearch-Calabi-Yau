# Format of the dataset:

```
- Polytope List:
    - Polytope 1
        - Vertices: ... # The vertices in N-Lattice space
        - FRST List:
            - FRST 1:
                - Signiture: ... # The simplices, the order should align with the polytope vertices array.
                - Triangulation List: # Found by using bfs starting from FRST 1
                    - Triangulation 1:
                        - Distance: ...
                        - Signiture: ...
                    - Triangulation 2:
                        - Distance: ...
                        - Signiture: ...
                        ...
            - FRST 2:
                ...
            ...
        

```

The dataset will be loaded to train RL policies, with the mdp class in `cy_triangulation_state.py`. 