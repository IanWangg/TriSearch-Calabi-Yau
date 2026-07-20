import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.train_cy as train_cy
from core.train_cy import (
    build_iteration_metrics_record,
    build_training_variant_suffix,
    build_wandb_run_name,
    configure_torch_cpu_threads,
    normalize_subcomplex_actor_type,
    parse_args,
    validate_count_bonus_args,
    validate_similarity_aug_args,
    write_iteration_metrics_record,
)
from core.training_types import PPOTrainStats, PolicyRolloutSummary


def test_parse_args_accepts_circuit_pool_actor_type():
    args = parse_args(["--subcomplex_actor_type", "circuit_pool"])

    assert args.subcomplex_actor_type == "circuit_pool"


def test_parse_args_accepts_snn_simplex_actor_type():
    args = parse_args(["--subcomplex_actor_type", "snn_simplex"])

    assert args.subcomplex_actor_type == "snn_simplex"
    assert normalize_subcomplex_actor_type("snn_simplex") == "snn_simplex"


def test_normalize_subcomplex_actor_type_treats_default_as_gnn():
    assert normalize_subcomplex_actor_type("default") == "gnn"


def test_parse_args_defaults_to_gnn_actor_type():
    args = parse_args([])

    assert args.subcomplex_actor_type == "gnn"


def test_training_variant_suffix_includes_non_mlp_actor_type():
    args = parse_args(["--subcomplex_actor_type", "gnn"])

    assert build_training_variant_suffix(args) == "_actor_gnn"


def test_training_variant_suffix_omits_mlp_actor_alias():
    args = parse_args(["--subcomplex_actor_type", "mlp"])

    assert build_training_variant_suffix(args) == ""


def test_parse_args_accepts_count_bonus_options():
    args = parse_args(["--count_bonus_coef", "0.25", "--count_bonus_exponent", "0.75"])

    assert args.count_bonus_coef == 0.25
    assert args.count_bonus_exponent == 0.75
    assert build_training_variant_suffix(args) == "_actor_gnn_count_bonus0p25_exp0p75"


def test_parse_args_accepts_vertex_aug_options():
    args = parse_args(
        [
            "--vertex_aug_enable",
            "--vertex_aug_prob",
            "0.5",
            "--vertex_aug_scale_min",
            "0.8",
            "--vertex_aug_scale_max",
            "1.2",
            "--vertex_aug_shift_std",
            "0.03",
            "--vertex_aug_reflect_prob",
            "0.2",
        ]
    )

    assert args.vertex_aug_enable is True
    assert args.vertex_aug_prob == 0.5
    assert args.vertex_aug_scale_min == 0.8
    assert args.vertex_aug_scale_max == 1.2
    assert args.vertex_aug_shift_std == 0.03
    assert args.vertex_aug_reflect_prob == 0.2
    assert build_training_variant_suffix(args) == "_actor_gnn_rollout_aug"


def test_parse_args_accepts_torch_thread_options():
    args = parse_args(["--torch_num_threads", "2", "--torch_num_interop_threads", "3"])

    assert args.torch_num_threads == 2
    assert args.torch_num_interop_threads == 3


def test_parse_args_accepts_name_suffix():
    args = parse_args(["--name_suffix", "torch1_workers16"])

    assert args.name_suffix == "torch1_workers16"


def _volume_rollout_summary(*, offset=0.0):
    initial_values = [float(index + 1) + offset for index in range(32)]
    best_values = [
        value + (2.0 if index % 2 == 0 else 0.0)
        for index, value in enumerate(initial_values)
    ]
    final_values = [value + 1.0 for value in initial_values]
    return PolicyRolloutSummary(
        final_states=[],
        rollout_buffer=None,
        success_rate=0.0,
        discounted_reward=0.25,
        finished_fraction=0.0,
        finished_count=0,
        frt_hits=0,
        collapsed_hits=0,
        dead_end_hits=0,
        all_step_reset_count=0,
        all_step_frt_hits=0,
        all_step_collapsed_hits=0,
        all_step_dead_end_hits=0,
        expanded_states=1,
        discovered_states=1,
        multiprocessing_steps=0,
        total_candidates=32,
        total_valid_actions=32,
        candidate_expand_sec=0.1,
        policy_data_build_sec=0.1,
        policy_batch_transfer_sec=0.1,
        policy_value_inference_sec=0.1,
        policy_action_inference_sec=0.1,
        transition_apply_sec=0.1,
        objective_name="max_cy_volume",
        objective_goal="max",
        objective_initial_values=initial_values,
        objective_final_values=final_values,
        objective_best_values=best_values,
        return_mean=0.5,
        return_std=0.2,
        return_min=-0.1,
        return_max=0.9,
        training_return_mean=0.5,
        training_discounted_reward=0.25,
    )


def test_iteration_metrics_schema_records_all_volume_slots_and_aggregates():
    train_summary = _volume_rollout_summary()
    eval_summary = _volume_rollout_summary(offset=100.0)
    train_stats = PPOTrainStats(
        total_loss=1.0,
        policy_loss=2.0,
        value_loss=3.0,
        entropy_loss=4.0,
        explained_variance=5.0,
        clip_ratio=6.0,
        num_samples=160,
        num_valid_action_samples=150,
    )

    record = build_iteration_metrics_record(
        iteration=0,
        reward_function="max_cy_volume",
        cy_volume_reward_transform="log",
        rollout_summary=train_summary,
        eval_summary=eval_summary,
        train_stats=train_stats,
        deterministic_rollout=False,
        deterministic_eval=True,
        rollout_sec=1.0,
        bootstrap_sec=2.0,
        prepare_sec=3.0,
        train_sec=4.0,
        eval_sec=5.0,
        iteration_sec=15.0,
    )

    assert record["schema_version"] == 1
    assert record["iteration"] == 1
    assert record["cy_volume_reward_transform"] == "log"
    assert record["train"]["deterministic"] is False
    assert record["eval"]["deterministic"] is True
    assert record["train"]["return"] == {
        "mean": 0.5,
        "std": 0.2,
        "min": -0.1,
        "max": 0.9,
        "discounted_mean": 0.25,
        "training_mean": 0.5,
        "training_discounted_mean": 0.25,
    }
    train_volume = record["train"]["raw_volume"]
    eval_volume = record["eval"]["raw_volume"]
    assert len(train_volume["slots"]) == 32
    assert len(eval_volume["slots"]) == 32
    assert train_volume["slots"][0] == {
        "slot": 0,
        "initial_volume": 1.0,
        "final_volume": 2.0,
        "best_volume": 3.0,
        "best_volume_improvement": 2.0,
    }
    assert train_volume["initial_mean"] == 16.5
    assert train_volume["final_mean"] == 17.5
    assert train_volume["best_mean"] == 17.5
    assert train_volume["mean_best_volume_improvement"] == 1.0
    assert train_volume["improved_fraction"] == 0.5
    assert record["ppo"]["policy_loss"] == 2.0
    assert record["timing"]["iteration_sec"] == 15.0


class _FlushTrackingStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1
        super().flush()


def test_iteration_metrics_writer_emits_one_jsonl_record_and_flushes():
    stream = _FlushTrackingStream()

    write_iteration_metrics_record(stream, {"iteration": 1, "value": 2.0})

    assert stream.flush_count == 1
    assert json.loads(stream.getvalue()) == {"iteration": 1, "value": 2.0}
    assert stream.getvalue().endswith("\n")


def test_name_suffix_is_appended_to_checkpoint_dir(monkeypatch):
    captured_default_dirs = []

    def fake_build_checkpoint_dir(*, checkpoint_path, default_dir):
        captured_default_dirs.append(default_dir)
        raise RuntimeError("stop after checkpoint dir construction")

    monkeypatch.setattr(train_cy, "load_cy_sample_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(train_cy, "infer_dataset_coordinate_dim", lambda rows: 3)
    monkeypatch.setattr(train_cy, "resolve_policy_in_channels", lambda rows, requested: 3)
    monkeypatch.setattr(
        train_cy,
        "split_rows_by_vertex_count",
        lambda rows, num_eval_polytopes: SimpleNamespace(
            train_polytope_indices=[1],
            eval_polytope_indices=[2],
            train_rows=[],
            eval_rows=[],
        ),
    )
    monkeypatch.setattr(train_cy, "mean_vertex_count", lambda rows: 0.0)
    monkeypatch.setattr(
        train_cy,
        "build_cy_rollout_collection",
        lambda *args, **kwargs: SimpleNamespace(
            base_states={},
            initial_states=[],
            polytope_by_index={},
            vertices_by_polytope={},
        ),
    )

    args = parse_args(
        [
            "--name_suffix",
            "torch1_workers16",
            "--dataset_path",
            "dummy.jsonl",
            "--force_cpu",
            "--torch_num_threads",
            "0",
            "--torch_num_interop_threads",
            "0",
        ]
    )
    monkeypatch.setattr("core.cy_runtime_utils.build_checkpoint_dir", fake_build_checkpoint_dir)

    with pytest.raises(RuntimeError, match="stop after checkpoint dir construction"):
        train_cy.main(args)

    assert captured_default_dirs == [
        "ckpt/cy_subcomplex_ppo_improved_128state_20rollout_actor_gnn_torch1_workers16"
    ]


def test_name_suffix_is_appended_to_wandb_run_name():
    args = parse_args(
        [
            "--name_suffix",
            "torch1_workers16",
            "--num_eval_polytopes",
            "7",
            "--num_epochs",
            "2",
        ]
    )

    assert (
        build_wandb_run_name(args)
        == "algo-cy-egnn-subcomplex-ppo-improved__hardest-eval-7__epochs-per-iter-2__torch1_workers16"
    )


class _FakeTorch:
    def __init__(self):
        self.num_threads = 44
        self.num_interop_threads = 44
        self.calls = []

    def set_num_threads(self, value):
        self.calls.append(("set_num_threads", int(value)))
        self.num_threads = int(value)

    def get_num_threads(self):
        return self.num_threads

    def set_num_interop_threads(self, value):
        self.calls.append(("set_num_interop_threads", int(value)))
        self.num_interop_threads = int(value)

    def get_num_interop_threads(self):
        return self.num_interop_threads


def test_configure_torch_cpu_threads_sets_positive_values(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(train_cy, "torch", fake_torch)

    result = configure_torch_cpu_threads(
        SimpleNamespace(torch_num_threads=2, torch_num_interop_threads=3)
    )

    assert fake_torch.calls == [
        ("set_num_threads", 2),
        ("set_num_interop_threads", 3),
    ]
    assert result == {
        "torch_num_threads": 2,
        "torch_num_interop_threads": 3,
    }


def test_configure_torch_cpu_threads_skips_non_positive_values(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(train_cy, "torch", fake_torch)

    result = configure_torch_cpu_threads(
        SimpleNamespace(torch_num_threads=0, torch_num_interop_threads=-1)
    )

    assert fake_torch.calls == []
    assert result == {
        "torch_num_threads": 44,
        "torch_num_interop_threads": 44,
    }


def test_train_cy_entrypoint_does_not_import_training_stack_as_spawn_worker():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "train_cy.py"
    code = (
        "import runpy, sys; "
        "sys.modules.pop('core.train_cy', None); "
        "sys.modules.pop('torch', None); "
        f"runpy.run_path({str(script)!r}, run_name='__mp_main__'); "
        "print('core.train_cy' in sys.modules, 'torch' in sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False False"


def test_validate_count_bonus_args_rejects_invalid_values():
    with pytest.raises(ValueError, match="count_bonus_coef"):
        validate_count_bonus_args(parse_args(["--count_bonus_coef", "-0.1"]))

    with pytest.raises(ValueError, match="count_bonus_exponent"):
        validate_count_bonus_args(parse_args(["--count_bonus_exponent", "0.0"]))


def test_validate_similarity_aug_args_rejects_invalid_values():
    with pytest.raises(ValueError, match="vertex_aug_prob"):
        validate_similarity_aug_args(
            name="vertex_aug",
            aug_prob=1.1,
            scale_min=0.9,
            scale_max=1.1,
            shift_std=0.05,
            reflect_prob=0.1,
        )

    with pytest.raises(ValueError, match="vertex_aug_scale_max"):
        validate_similarity_aug_args(
            name="vertex_aug",
            aug_prob=1.0,
            scale_min=1.2,
            scale_max=1.1,
            shift_std=0.05,
            reflect_prob=0.1,
        )
