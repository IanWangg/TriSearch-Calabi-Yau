# FRSTs Optimization with `two_neighbors` Function

## Context (What is in this folder)

This folder contains the application of algorithm `TriSearch` on Calabi-Yau manifold sampling through finding FRSTs. There are two usage types:
- The initial version is to conduct uniform sampling : to sample as many different FRST as possible.
- The second version is to optimize an objective (e.g. number of simplices) for triangulation of a reflexive polytope. But not limited to FRSTs.

## New Usage: Find Near-Optimal FRSTs for Some Objective

We want to find a "near-optimal" FRSTs according to some metrics. 

This is similar to optimize the objective for triangulation in general , the difference is that we now only operate in the FRST space.

To do this, instead of list all the neighbors , we list the `two_neighbors` , which are "Non-two-face-equivalent" neighbors. If we start from an FRST, we will always get FRST neighbors through this function in cytools. 
- `TriSearch` will only step in the FRST space , and find the FRST that has the good quality according to our objective function.

### Objective to start with
To start with, we want to optimize the volume of the volume of the Calabi-Yau Manifold:
```python
Kcup = cy.mori_cone_cap(in_basis=True).dual()
t = Kcup.tip_of_stretched_cone(c=1)
V = cy.compute_cy_volume(t) # maximize this
```
or 
```python
t = cy.toric_kahler_cone().tip_of_stretched_cone(c=1)
V = cy.compute_cy_volume(t) # maximize this
```
I need you to investigate these two objectives:
- Whether they are equivalent?
- Which one is easier (cheaper) to start with?

### Notes:
I need you to look into the following things:
- Are the neighbors found by `two_neighbors` method is only different from the current triangulation by one circuit : i.e. Is it compatible with our current model. Our model select the neighbors based on one circuit of difference.
- Perform an end-to-end experiment for 50 iterations. Using the `two_neighbors` as well as the objective we mentioned above (pick the cheaper one). Check whether the objective is improving.