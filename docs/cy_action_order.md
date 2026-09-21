# Geometry action ordering and reproducibility

Managed geometry requests accept `action_order="native"` or
`action_order="canonical"`. Both return the same valid actions, ambiguity
filtering, exact simplex transitions, and geometry objectives. The request-level
default is `native`; managed training, evaluation, rollout entrypoints, and the
rollout engine default to `canonical`. Pass `--action_order native` to an
entrypoint to retain backend ordering explicitly.

Native order retains the order returned by the installed geometry backend,
including the existing convention that regular add actions precede regular
remove actions. That backend order is not reproducible across worker histories.
In the local triangulumancer source,
`/home/yiranwang/combinartorics/triangulumancer/src/triangulumancer/TOPCOM.cpp:115`
iterates `MarkedFlips` directly when collecting neighbours. Its definition at
`/home/yiranwang/combinartorics/triangulumancer/extern/topcom/lib-src/MarkedFlips.hh:27`
uses either `PlainHashMap` or `std::unordered_map` depending on build settings.
Neither establishes a canonical geometric ordering.

This affected a real reproduction using polytope 11, all five initial states
from `data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl`,
and NumPy generator seed 100000. One worker and two workers returned equal
candidate sets with different candidate orderings. The fifth chosen action
changed from `(0, 1, 2, 3, 4, 8)` to `(1, 2, 3, 5, 6, 8)`. Rebuilding the exact
original point configuration and reducing the parent cache independently
confirmed that the mismatch came from backend ordering, not cache eviction.

Canonical order sorts by circuit and destination identity inside the existing
regular add/remove groups. It sorts two-neighbour actions by circuit and
destination identity without introducing add/remove groups. This makes action
indices independent of which worker performed an expansion. It can change
trajectories from earlier runs that sampled native action indices; model weights,
action features, and geometry remain compatible.

`test/test_cy_managed_geometry.py` checks canonical expansion equality across
one and two workers, before and after worker cache reclamation, using the same
five real initial states. Legacy rich state APIs retain native backend ordering.
