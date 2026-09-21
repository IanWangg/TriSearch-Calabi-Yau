# Calabi-Yau Runtime Layout

This document records module ownership after the runtime refactor. Public entry
points remain stable; implementation details live in focused modules named for
the responsibility they own.

## Entry Points and Compatibility Facades

| User-facing command or import | Stable facade | Implementation owner |
| --- | --- | --- |
| `scripts/train_cy.py` | `core/train_cy.py` | `core/cy_training_runner.py` |
| `scripts/eval_cy.py` | `core/evaluate_rl_cy.py` | `core/cy_evaluation.py` |
| `tools/rollout_cy_random.py` | `core/rollout_cy_random.py` | `mdp/cy_rollout.py` |
| `tools/rollout_cy_policy.py` | `core/rollout_cy_policy.py` | `core/cy_policy_rollout.py` |
| Existing policy rollout imports | `core/cy_policy_rollout_utils.py` | Policy modules listed below |

The facades are intentionally retained. They preserve public imports while
keeping command wiring separate from training, evaluation, and rollout logic.

## Training Responsibilities

- `core/cy_training_config.py`: typed training configuration, argument parsing,
  validation, run suffixes, dry-run behavior, and thread settings.
- `core/cy_training_runner.py`: training lifecycle orchestration.
- `core/cy_checkpointing.py`: checkpoint save and restore behavior.
- `core/cy_training_metrics.py`: training metric aggregation and reporting.
- `core/cy_runtime_utils.py`: runtime helpers shared by training and rollout.

## Policy Responsibilities

- `models/subcomplex_policy_config.py`: canonical policy kind names and the SNN
  default.
- `models/subcomplex_policy_factory.py`: policy construction without CLI or
  training concerns.
- `core/cy_policy_inference.py`: policy action selection and inference helpers.
- `core/cy_ppo.py`: PPO-specific update logic.
- `core/cy_policy_rollout.py`: policy rollout collection and summaries.

## Evaluation and Geometry Responsibilities

- `core/cy_evaluation_config.py`: evaluation arguments and validation.
- `core/cy_evaluation.py`: evaluation orchestration and result aggregation.
- `mdp/cy_rollout.py`: canonical random rollout engine.
- `mdp/cy_state_record.py`: exact lightweight state and immutable point
  configuration records used by the trainer; no native geometry objects.
- `mdp/cy_geometry_worker.py`: worker-owned CYTools geometry, exact objectives,
  compact expansion deltas, and bounded native-object caches.
- `mdp/cy_triangulation_state.py`: retained rich-state geometry compatibility API.

## Resource and Process Ownership

- `core/cy_managed_runtime.py`: shared lifecycle and cache-budget allocation for
  training, evaluation, and standalone rollouts.
- `core/cy_process_runtime.py`: independent Linux guardian, managed workers,
  bounded ordered IPC, deadlines, memory monitoring, and descendant cleanup.
- `core/cy_bounded_cache.py`: byte- and entry-bounded admission-time LRU caches.
- `core/cy_state_history.py`: exact SQLite-backed identities and visit counts.
- `tools/benchmark_cy_runtime.py`: saved expansion traces and cold/warm replay.

See [managed rollout runtime](cy_managed_runtime.md) for operating limits and
the distinction between resident caches and cumulative exploration history.

## Deferred Removal

`test/test_cy_policy_training.py` is the canonical policy-training test module.
Its legacy-named duplicate,
`test/test_main_rl_egnn_subcomplex_cy_improved_fixed.py`, remains present until
the repository-root `remove_redundant_files.sh` script is run. The script uses
an exact allowlist, performs no recursive deletion, and is safe to run again.

Framework migrations, dependency upgrades, public API changes, and broader
architecture moves are outside this refactor and should be handled as separate
migration tasks with their own parity checks.
