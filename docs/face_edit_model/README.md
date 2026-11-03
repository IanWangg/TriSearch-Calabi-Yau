# FaceEdit model (planning)

Hierarchical policy for CY triangulation search: pick one 2-face, then regenerate
that face's FRT with a dualGNN-style autoregressive editor conditioned on a
set-fused representation of all faces. The environment glues the new 2-face
tuple into an FRST with CYTools `Polytope.ntfe_frts(face_triangs=..., N=1, make_star=True)`.

Living plan (architecture, MDP, training stages, acceptance criteria, open questions):
https://claude.ai/artifact/TDrngRQaoUfDJ1rbGPti7s

Status 2026-11-03: plan under discussion. No implementation yet.

Face statistics used for the plan (CYTools enumeration, fine+regular, faces <= 10 points):

| h11 | polytopes | faces/polytope | mean points/face | max face points | editable faces |
|-----|-----------|----------------|------------------|-----------------|----------------|
| 15  | 15 (dev)  | 49             | 3.5              | 6.5 (max 9)     | 9.8            |
| 20  | 15 (dev)  | 49             | 3.7              | 8.0 (max 11)    | 13.0           |
| 50  | 5         | 15             | 9.7              | 25 (max 42)     | 5.2            |

Notes: do not copy dualGNN code (GPL-3); implement from the paper's ideas.
