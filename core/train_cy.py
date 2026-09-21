from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cy_checkpointing import (
    find_latest_policy_checkpoint,
    save_iteration_checkpoints,
    save_policy_checkpoint,
)
from core.cy_training_config import (
    CYTrainingConfig,
    apply_dry_run_overrides,
    build_training_variant_suffix,
    configure_torch_cpu_threads,
    format_float_suffix,
    parse_args,
    validate_count_bonus_args,
    validate_cy_volume_reward_transform_args,
    validate_neighbor_mode_args,
    validate_similarity_aug_args,
)
from core.cy_training_metrics import (
    build_iteration_metrics_record,
    build_raw_volume_metrics,
    build_wandb_run_name,
    init_wandb_run,
    write_iteration_metrics_record,
)
from core.cy_training_runner import main, maybe_filter_initial_state_pool
from mdp.cy_rollout import get_cy_shared_cache_sizes, prune_cy_shared_caches
from models.subcomplex_policy_config import (
    normalize_subcomplex_actor_type,
    value_feature_source_for_subcomplex_actor,
)

__all__ = [
    "CYTrainingConfig",
    "apply_dry_run_overrides",
    "build_iteration_metrics_record",
    "build_raw_volume_metrics",
    "build_training_variant_suffix",
    "build_wandb_run_name",
    "configure_torch_cpu_threads",
    "find_latest_policy_checkpoint",
    "format_float_suffix",
    "get_cy_shared_cache_sizes",
    "init_wandb_run",
    "main",
    "maybe_filter_initial_state_pool",
    "normalize_subcomplex_actor_type",
    "parse_args",
    "prune_cy_shared_caches",
    "save_iteration_checkpoints",
    "save_policy_checkpoint",
    "validate_count_bonus_args",
    "validate_cy_volume_reward_transform_args",
    "validate_neighbor_mode_args",
    "validate_similarity_aug_args",
    "value_feature_source_for_subcomplex_actor",
    "write_iteration_metrics_record",
]


if __name__ == "__main__":
    main(parse_args())
