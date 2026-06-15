import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.train_cy as train_cy
from core.train_cy import (
    build_training_variant_suffix,
    build_wandb_run_name,
    configure_torch_cpu_threads,
    normalize_subcomplex_actor_type,
    parse_args,
    validate_count_bonus_args,
    validate_similarity_aug_args,
)


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
