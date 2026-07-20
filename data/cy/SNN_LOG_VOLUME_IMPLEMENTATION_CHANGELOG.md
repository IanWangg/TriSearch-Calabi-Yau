# SNN + Log-Volume Implementation Changelog

## 2026-07-18

### Added

- Added the `max_cy_volume` objective and reward implementation in
  `reward_functions/max_cy_volume.py`.
  - Computes raw Calabi-Yau threefold volume at the stretched Kcup cone tip.
  - Caches raw volumes by triangulation-state key.
  - Validates that CYTools returns a threefold and that a materialized
    triangulation is available.
- Added `--cy_volume_reward_transform {raw,log}` to training and evaluation.
  - `raw`, the compatibility default, returns `V_next - V_current`.
  - `log` returns `log(V_next) - log(V_current)`.
  - Log rewards fail explicitly when either volume is nonpositive.
  - The log transform is rejected unless `max_cy_volume` is selected.
- Added `--iteration_metrics_path` for opt-in per-iteration JSONL telemetry.
  Each flushed record contains:
  - train and deterministic-evaluation return distributions;
  - all per-slot raw initial, final, and best volumes;
  - all per-slot best-volume improvements;
  - aggregate raw-volume means and improved fractions;
  - PPO losses and sample counts;
  - rollout, bootstrap, optimization, evaluation, and iteration timing.
- Added `neighbor_mode=two_neighbors` support to training, evaluation, rollout
  collection, state materialization, and cache keys.
  - Initial-state pools contain validated fine, star, regular triangulations.
  - Actions are the four vertices of the changed two-face diagonal-flip
    circuit.
  - The mode requires `--no-include_points_interior_to_facets`.
- Added the compact 12-polytope h11=12 fixture at
  `data/cy/two_neighbors_h11_12.samples.jsonl`.

### Changed

- Configured the experiment policy as an EGNN state encoder with the
  `snn_simplex` action head instead of the GNN action head.
- Kept `MaxCYVolumeReward.metric(state)` in raw-volume units even when PPO uses
  log-volume rewards. Console output, JSONL metrics, and evaluation summaries
  therefore remain geometrically interpretable.
- Reused the same `MaxCYVolumeReward` instance for transition rewards and raw
  objective reporting so both paths share the volume cache.
- Appended the two-neighbor mode to generated training checkpoint variant
  names; the selected SNN actor type remains part of the actor variant suffix.
- Updated README examples and documentation for FRST-only navigation,
  log-volume training, and iteration JSONL output.

### Fixed

- Fixed SNN candidate-to-simplex membership for lower-dimensional circuit
  actions. A four-vertex two-face circuit cannot contain a five-vertex ambient
  FRST simplex, so the SNN head now pools incident top-simplex cofaces of the
  circuit's current source faces. Existing full-simplex containment behavior
  is unchanged for ordinary actions.
- Isolated regular and two-neighbor state/action caches by including the
  neighbor mode in state keys.
- Ensured two-neighbor transitions retain the complete CYTools FRST
  representative while exposing the local four-vertex circuit to the policy.

### Validation

- Added tests for raw reward compatibility, exact log-volume differences,
  shared caching, and nonpositive-volume failures.
- Added CLI parsing and invalid-objective validation tests for the volume
  transform.
- Added JSONL schema, 32-slot aggregate, and per-record flush tests.
- Added SNN lower-dimensional coface-membership and real CYTools two-neighbor
  topology tests.
- Passed 61 focused reward, training, SNN, and two-neighbor tests.
- Passed the complete test suite: 128 tests.
- Verified both `scripts/train_cy.py --help` and `scripts/eval_cy.py --help`.

### Experiment Artifacts

- Completed 25 PPO iterations with 32 stochastic training states and a strict
  five-step horizon.
- Evaluated 32 deterministic held-out slots for five steps after every
  iteration.
- Wrote `latest.pth` every five iterations, `25.pth` at iteration 25, and
  `final.pth` after normal completion. All three terminal checkpoints contain
  identical model tensors.
- Evaluated `final.pth` once on each of the four held-out initial FRSTs for
  exactly five deterministic steps.
- Recorded the full configuration, 25-row learning table, and final held-out
  results in `data/cy/SNN_LOG_VOLUME_FIVE_STEP_EXPERIMENT.md`.
- Stored run outputs under `runs/snn_log_volume_five_step_seed0/` and terminal
  checkpoints under
  `ckpt/runs/snn_log_volume_five_step_seed0/checkpoints/`.

No matched untrained SNN baseline was run as part of this implementation.
