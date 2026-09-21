# CY runtime contract

This document records the behavior of the CY runtime. The policy refactor makes
`snn_simplex` the default subcomplex policy. The managed rollout runtime also
introduces bounded retention, geometry isolation, and canonical action ordering.
See [resource controls](cy_managed_runtime.md) and
[action ordering](cy_action_order.md) for intentional compatibility changes.

## Policy selection

- Omitting `subcomplex_actor_type`, or passing `default`, selects
  `snn_simplex` for training, evaluation, policy rollout, model factories, and
  low-level policy constructors.
- SNN policies use simplex-topology features for both action logits and value
  estimates. Data construction must therefore include cached simplex topology.
- `gnn`, `mlp`, and `circuit_pool` remain explicit supported choices.
- Checkpoints contain weights only. Existing GNN checkpoints require
  `--subcomplex_actor_type gnn`; optimizer state and iteration counters are not
  restored.

## Rollout and training parity

- The canonical rollout engine is `mdp.cy_rollout.CYRandomRolloutEngine`.
  Training, evaluation, and both rollout tools use that implementation.
- Reward evaluation, terminal reasons, reset timing, action masking,
  first-episode statistics, full-horizon returns, count bonuses, and objective
  best-value tracking retain their existing semantics.
- PPO padding, GAE, clipping, entropy/value losses, gradient clipping, and
  valid-action masking retain their existing tensor order and formulas.
- Canonical action ordering preserves the candidate set and exact transitions,
  and makes action indices independent of worker scheduling and cache eviction.
  It does not preserve historical native-order seeded trajectories. Native
  backend ordering is available explicitly but is not worker-independent.
- Timing, memory, and throughput measurements are observational and are excluded
  from numeric parity comparisons.

## Persistence and observability

- `latest.pth` has checkpoint discovery priority, followed by the largest
  numeric or `oom_guard_iterN` checkpoint, then modification time.
- Policy saves remain atomic through a temporary sibling file and `os.replace`.
- Existing iteration JSONL schema version 1 fields and W&B keys remain available.
  JSONL additionally records `memory` and `action_order`. Cumulative graph counts
  are distinct from bounded resident graph sizes.
- Final/latest saves remain atomic. A handled memory guard or SIGTERM after
  policy initialization saves `oom_guard_interrupted.pth` and `latest.pth`.
  SIGKILL cannot save trainer state; the independent guardian cleans descendants.
- Exact disk-backed discovery and visitation counts survive cache eviction, but
  are run-local. Checkpoint loading remains weights-only, not history resume.
- `scripts/train_cy.py` must not import Torch or the training stack when loaded
  as a multiprocessing spawn worker.

## Validation environment

Run commands from the repository root with the complete Sage environment on
`PATH`:

```bash
PATH=/home/yiranwang/anaconda3/envs/sage/bin:$PATH \
  /home/yiranwang/anaconda3/envs/sage/bin/pytest -q
```
