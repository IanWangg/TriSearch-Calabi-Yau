"""Shared inference, PPO, strict checkpoints, and real training/search integration."""

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from core.cy_checkpointing import save_policy_checkpoint
from core.cy_data_utils import configure_cy_data_tensor_caches
from core.cy_policy_inference import (
    _copy_data_with_subcomplex_vertices, batched_policy_action_selection, build_cy_data_list,
    configure_policy_execution, evaluate_policy_actions, evaluate_policy_actions_from_data_list,
    evaluate_policy_scores, evaluate_policy_values,
)
from core.cy_ppo import PPORolloutBuffer, train_policy_from_rollout
from core.cy_runtime_utils import load_policy_checkpoint
from core.cy_training_config import CYTrainingConfig, parse_args, validate_two_face_training_args
from core.cy_two_face_data import TwoFaceData
from eval.config import EvaluationSpec
from eval.policy import EvaluationPolicy
from test_cy_bounded_ppo import _step
from test_cy_two_face_data import square_state, square_actions
from test_two_face_agent import two_face_policy


@pytest.fixture(autouse=True)
def execution_limits():
    configure_policy_execution()
    yield
    configure_policy_execution()
    configure_cy_data_tensor_caches(max_bytes=0)
    configure_cy_data_tensor_caches()


def test_inference_routes_four_labels_and_ppo_preserves_log_probs():
    model = two_face_policy()
    states = [square_state(), square_state(one_face=True), square_state()]
    actions = [square_actions(states[0]), [], list(reversed(square_actions(states[2])))]
    selection = batched_policy_action_selection(states, actions, model, device=torch.device("cpu"))
    assert selection.subcomplex_width == 4
    assert all(isinstance(data, TwoFaceData) for data in selection.data_list)
    copied = _copy_data_with_subcomplex_vertices(selection.data_list[0], selection.data_list[0].subcomplex_vertices.clone())
    assert isinstance(copied, TwoFaceData) and set(copied.keys()) == set(selection.data_list[0].keys())
    buffer = PPORolloutBuffer()
    buffer.append(_step(states, selection))
    configure_cy_data_tensor_caches(max_bytes=0)
    prepared = buffer.prepare(bootstrap_value=torch.zeros(3), gamma=.9, gae_lambda=.95, device=torch.device("cpu"))
    assert prepared.observation_kind == "two_face"
    assert prepared.state_buffer_list == [None] * 3
    scored = evaluate_policy_actions_from_data_list(prepared.data_buffer_list, prepared.action_index_buffer_flat,
                                                   model, device=torch.device("cpu"))
    rebuilt = evaluate_policy_actions(states, actions, selection.action_index_tensor, model, device=torch.device("cpu"))
    torch.testing.assert_close(scored.log_prob_tensor, selection.log_prob_tensor)
    torch.testing.assert_close(rebuilt.log_prob_tensor, selection.log_prob_tensor)
    torch.testing.assert_close(evaluate_policy_values(states, [()] * 3, model, device=torch.device("cpu")).value_tensor,
                               selection.value_tensor)
    scorer = EvaluationPolicy(model, device="cpu", max_graph_size=1)
    assert scorer.metadata["policy_observation_kind"] == "two_face"
    assert scorer.metadata["policy_observation_schema_version"] == 1
    torch.testing.assert_close(torch.tensor(scorer.score_values(states), dtype=torch.float32), selection.value_tensor)
    scores = scorer.score_actions(states, actions)
    assert len(scores[0]) == 2 and len(scores[1]) == 0
    buffer.clear()
    assert buffer.observation_kind is None and not buffer.data_buffer


@pytest.mark.parametrize("saved_first", [True, False])
def test_mixed_saved_and_rebuilt_buffer_uses_two_face_schema(saved_first):
    state = square_state()
    selection = batched_policy_action_selection([state], [square_actions(state)], two_face_policy(), device=torch.device("cpu"))
    saved, rebuilt = _step([state], selection), _step([state], selection)
    rebuilt.data_list = None
    rebuilt.observation_kind = "two_face"
    buffer = PPORolloutBuffer()
    for step in (saved, rebuilt) if saved_first else (rebuilt, saved):
        buffer.append(step)
    prepared = buffer.prepare(bootstrap_value=torch.zeros(1), gamma=.9, gae_lambda=.95, device=torch.device("cpu"))
    assert all(isinstance(data, TwoFaceData) for data in prepared.data_buffer_list)
    assert torch.equal(prepared.data_buffer_list[0].removed_triangle_ids, prepared.data_buffer_list[1].removed_triangle_ids)
    bad = _step([state], selection)
    bad.observation_kind = "full_triangulation"
    with pytest.raises(ValueError, match="mix observation schemas"):
        buffer.append(bad)


@pytest.mark.parametrize("save_data", [True, False])
def test_ppo_chunks_preserve_losses_gradients_and_update(save_data):
    model = two_face_policy()
    chunked = copy.deepcopy(model)
    states = [square_state(), square_state(one_face=True), square_state()]
    actions = [square_actions(states[0]), [], square_actions(states[2])]
    selection = batched_policy_action_selection(states, actions, model, device=torch.device("cpu"))
    step = _step(states, selection)
    step.observation_kind = "two_face"
    if not save_data:
        step.data_list = None
    buffer = PPORolloutBuffer()
    buffer.append(step)
    prepared = buffer.prepare(bootstrap_value=torch.tensor([.1, -.2, .3]), gamma=.9, gae_lambda=.95, device=torch.device("cpu"))
    kwargs = dict(prepared_rollout=prepared, device=torch.device("cpu"), num_epochs=2, batch_size=3,
                  clip_coef=.2, value_coef=.5, entropy_coef=.01, max_grad_norm=.5)
    torch.manual_seed(39)
    full_stats = train_policy_from_rollout(policy=model, optimizer=torch.optim.SGD(model.parameters(), lr=.01),
                                           max_graph_size=10**9, **kwargs)
    torch.manual_seed(39)
    chunk_stats = train_policy_from_rollout(policy=chunked, optimizer=torch.optim.SGD(chunked.parameters(), lr=.01),
                                            max_graph_size=1, **kwargs)
    assert chunk_stats.total_loss == pytest.approx(full_stats.total_loss, abs=2e-6)
    for first, second in zip(model.parameters(), chunked.parameters()):
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-6)
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=2e-6)
    with pytest.raises(ValueError, match="schema"):
        train_policy_from_rollout(policy=model, optimizer=torch.optim.SGD(model.parameters(), lr=.01),
            **{**kwargs, "prepared_rollout": replace(prepared, observation_kind="full_triangulation")})


def test_strict_checkpoint_metadata_round_trip_and_mismatch(tmp_path):
    model = two_face_policy(4)
    path = tmp_path / "latest.pth"
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match="Missing model_config"):
        load_policy_checkpoint(model, str(path))
    # A new directory creates the sidecar and keeps the pure state_dict format.
    path = tmp_path / "new" / "latest.pth"
    save_policy_checkpoint(model, str(path))
    spec = EvaluationSpec(1, 12, 1, 1, policy_checkpoint=str(path.parent), force_cpu=True,
        subcomplex_actor_type="two_face_deep_sets", in_channels=4, hidden_channels=8, out_channels=8, num_layers=2)
    restored = EvaluationPolicy.from_spec(spec)
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(weight, restored.model.state_dict()[name])
    sidecar = path.parent / "model_config.json"
    before = sidecar.read_bytes()
    save_policy_checkpoint(model, str(path))
    assert sidecar.read_bytes() == before
    with pytest.raises(ValueError, match="model_config mismatch"):
        EvaluationPolicy.from_spec(replace(spec, num_layers=3))
    changed = json.loads(before)
    changed["observation_schema_version"] = 2
    sidecar.write_text(json.dumps(changed))
    checkpoint_before = path.read_bytes()
    with pytest.raises(ValueError, match="model_config mismatch"):
        save_policy_checkpoint(model, str(path))
    assert path.read_bytes() == checkpoint_before


def test_v1_training_constraints_and_builder_validation():
    args = parse_args(["--subcomplex_actor_type", "two_face_deep_sets", "--neighbor_mode", "two_neighbors",
                       "--no-include_points_interior_to_facets", "--reward_function", "max_kcup", "--no-vertex_aug_enable"])
    validate_two_face_training_args(args)
    assert CYTrainingConfig.from_namespace(args).observation_kind == "two_face"
    for name, value in (("vertex_aug_enable", True), ("count_bonus_coef", .1), ("neighbor_mode", "regular"),
                        ("reward_function", "min_tri"), ("in_channels", 3)):
        invalid = copy.copy(args)
        setattr(invalid, name, value)
        with pytest.raises(ValueError):
            validate_two_face_training_args(invalid)
    with pytest.raises(ValueError, match="width 4"):
        build_cy_data_list([square_state()], [[]], observation_kind="two_face", subcomplex_width=6)
    with pytest.raises(ValueError, match="augmentation"):
        build_cy_data_list([square_state()], [[]], observation_kind="two_face", trajectory_transforms=[None])
    # Critic-only evaluation ignores even a supplied invalid candidate list.
    value = evaluate_policy_scores([square_state()], [[(999,)]], two_face_policy(),
                                    device=torch.device("cpu"), value_only=True).value_tensor
    assert torch.isfinite(value).all()


def test_real_training_checkpoint_and_search_smoke(tmp_path):
    from eval.pipeline import run_evaluation
    from eval.setup import EvaluationSetup
    from eval.results.plotting import read_comparison

    root = Path(__file__).resolve().parents[1]
    fixture = root / "data/cy/two_neighbors_h11_12.samples.jsonl"
    checkpoint_dir = tmp_path / "checkpoints"
    command = [sys.executable, "scripts/train_cy.py", "--dataset_path", str(fixture), "--max_rows", "2",
        "--num_eval_polytopes", "1", "--subcomplex_actor_type", "two_face_deep_sets",
        "--neighbor_mode", "two_neighbors", "--no-include_points_interior_to_facets", "--reward_function", "max_kcup",
        "--no-vertex_aug_enable", "--count_bonus_coef", "0", "--force_cpu", "--num_iterations", "1",
        "--num_states", "2", "--rollout_length", "2", "--batch_size", "4", "--num_eval_states", "1",
        "--eval_steps", "1", "--eval_interval", "1", "--save_interval", "1", "--latest_checkpoint_interval", "1",
        "--hidden_channels", "8", "--out_channels", "8", "--num_layers", "2",
        "--transition_num_workers", "1", "--memory_budget_gb", "8", "--runtime_cache_gb", ".05",
        "--checkpoint_path", str(checkpoint_dir), "--iteration_metrics_path", str(tmp_path / "metrics.jsonl")]
    logs = tmp_path / "logs"
    logs.mkdir()
    with (logs / "training.log").open("w") as stream:
        result = subprocess.run(command, cwd=root, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                                stdout=stream, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, (logs / "training.log").read_text()
    assert (checkpoint_dir / "1.pth").is_file() and (checkpoint_dir / "model_config.json").is_file()
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["iteration"] == 1
    row = json.loads(fixture.read_text().splitlines()[0])
    row["frst_list"] = row["frst_list"][:1]
    spec = EvaluationSpec(1, 12, 1, 2, algorithms=("rl_value_best_first",),
        policy_checkpoint=str(checkpoint_dir), subcomplex_actor_type="two_face_deep_sets", force_cpu=True,
        hidden_channels=8, out_channels=8, num_layers=2, two_face_state=True, policy_proposal_count=2,
        runtime_cache_gb=.05, memory_budget_gb=8)
    setup = EvaluationSetup([row], {"parameters": spec.setup_parameters(), "source_metadata": {"fixture": fixture.name}},
                            tmp_path / "setup")
    warm = run_evaluation(spec, setup, output_dir=tmp_path / "warm")
    cold = run_evaluation(replace(spec, cache_states=False), setup, output_dir=tmp_path / "cold")
    assert warm.rollouts == cold.rollouts
    for output in (warm.output_dir, cold.output_dir):
        read_comparison(output)
        config = json.loads((output / "config.json").read_text())
        assert config["policy"]["policy_observation_kind"] == "two_face"
        assert config["policy"]["policy_observation_schema_version"] == 1
    # Metadata is requested even with full-representative search identity.
    critic_spec = replace(spec, two_face_state=False, policy_proposal_count=-1)
    critic_policy = EvaluationPolicy.from_spec(critic_spec)
    def forbidden(*args, **kwargs):
        raise AssertionError("Critic-only search must skip actor inference.")
    critic_policy.score_actions = forbidden
    run_evaluation(critic_spec, setup, output_dir=tmp_path / "critic_only", policy=critic_policy)
