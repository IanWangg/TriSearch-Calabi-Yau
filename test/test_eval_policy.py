"""Shared SNN evaluation inference, checkpoint and pure-reward contracts."""

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from core.cy_policy_inference import build_cy_data_list
from eval.config import EvaluationSpec
from eval.policy import EvaluationPolicy
from models.egnn_subcomplex_predictor import EGNNSubcomplexAgent
from models.gcn_subcomplex_predictor import GCNSubcomplexAgent
from reward_functions import get_objective, get_reward, SUPPORTED_REWARDS
from test_cy_bounded_ppo import GeometryState, _policy


def forbidden(*args, **kwargs):
    raise AssertionError("Critic-only inference must not compute candidate scores or logits.")


@pytest.mark.parametrize("model_type", [EGNNSubcomplexAgent, GCNSubcomplexAgent])
@pytest.mark.parametrize("cached", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_critic_only_matches_joint_and_skips_actor(monkeypatch, model_type, cached, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = _policy(model_type=model_type).to(device).eval()
    states = [GeometryState("one"), GeometryState("two")]
    actions = [[(0, 1, 2), (0, 2, 3)], []]
    data = build_cy_data_list(states, actions, include_simplex_topology=True)
    if not cached:
        for entry in data:
            for name in model.snn_simplex_actor._CACHED_TOPOLOGY_ATTRS:
                del entry[name]
    batch = Batch.from_data_list(data).to(device)
    with torch.inference_mode():
        expected, _ = model.get_value_and_logits(batch)
        monkeypatch.setattr(model, "get_value_and_logits", forbidden)
        monkeypatch.setattr(model.subcomplex_decoder_head, "forward", forbidden)
        monkeypatch.setattr(model, "_build_padded_logits", forbidden)
        monkeypatch.setattr(model.snn_simplex_actor, "_pool_candidate_simplex_embeddings", forbidden)
        actual = model.get_value(batch)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_scorer_chunks_preserve_input_order_and_values(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = _policy().to(device).eval()
    full = EvaluationPolicy(model, device=device, max_graph_size=250000)
    chunked = EvaluationPolicy(model, device=device, max_graph_size=1)
    states = [GeometryState(str(index)) for index in range(5)]
    # Different point configurations and graph sizes exercise PyG offsets and
    # prevent an accidentally per-polytope scorer from passing this check.
    states[-1].point_config_index = 704
    states[-1].vertices += ((0., 0., 1.),)
    states[-1].simplices += ((0, 1, 4),)
    states[-1].edges = tuple(sorted({edge for simplex in states[-1].simplices for edge in combinations(simplex, 2)}))
    actions = [[(0, 1, 2), (0, 2, 3)], [], [(0, 1, 2)], [], [(0, 2, 3), (0, 1, 2)]]
    expected, actual = full.score_actions(states, actions), chunked.score_actions(states, actions)
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(full.score_values(states), chunked.score_values(states), atol=1e-6, rtol=1e-5)
    assert full.stats()["physical_batches"] == 2
    assert chunked.stats()["physical_batches"] == 10
    assert full.stats()["max_logical_batch_states"] == 5
    assert full.score_actions([], []) == [] and len(full.score_values([])) == 0


@pytest.mark.parametrize("name", SUPPORTED_REWARDS)
def test_pure_reward_matches_training_without_queries(name):
    calls = []

    def state(value):
        def objective(requested):
            calls.append(requested)
            return value
        return SimpleNamespace(key=str(value), simplices=[()] * int(value), objective_value=objective)

    reward = get_reward(name)
    current, following = state(2), state(5)
    objective = get_objective(name, reward=reward)
    left, right = objective(current), objective(following)
    expected = reward(current, following)
    count = len(calls)
    assert reward.from_objectives(left, right) == pytest.approx(expected)
    assert len(calls) == count


def test_spec_cli_and_setup_separation():
    from scripts.eval_cy import parse_args

    spec = EvaluationSpec(1, 12, 2, 5, beam_width=3)
    assert spec.resolved_policy_proposal_count == 3 and spec.value_discount == 0.9
    changed = replace(spec, policy_proposal_count=-1, value_discount=0.1, force_cpu=True)
    assert changed.setup_parameters() == spec.setup_parameters()
    args = parse_args(["--num_polytopes", "1", "--h11", "12", "--num_starts", "2",
                       "--objective_budget", "5", "--algorithms", "rl_stochastic_policy",
                       "rl_policy_beam_search", "rl_value_beam_search", "rl_value_best_first",
                       "--policy_proposal_count", "-1"])
    assert args.policy_proposal_count == -1 and args.value_discount == 0.9
    assert args.subcomplex_actor_type == "snn_simplex"


@pytest.mark.parametrize("count,expected", [(None, 4), (1, 1), (8, 8), (-1, -1)])
def test_value_best_first_config_defaults_and_setup_separation(count, expected):
    from eval.algorithm import get_algorithm

    spec = EvaluationSpec(1, 12, 2, 5)
    changed = replace(spec, algorithms=("rl_value_best_first",), beam_width=8, policy_proposal_count=count)
    algorithm = get_algorithm("rl_value_best_first", beam_width=changed.beam_width,
                              policy_proposal_count=changed.policy_proposal_count)
    assert algorithm.proposal_count == expected and algorithm.value_discount == 0.9
    assert algorithm.requires_policy == (count != -1)
    assert changed.setup_parameters() == spec.setup_parameters()


@pytest.mark.parametrize("field,value", [("policy_proposal_count", 0), ("policy_proposal_count", -2),
                                         ("policy_proposal_count", True), ("value_discount", -0.1),
                                         ("value_discount", 1.1), ("value_discount", float("nan")),
                                         ("policy_max_graph_size", 0), ("gpu_index", -1)])
def test_invalid_rl_settings(field, value):
    with pytest.raises(ValueError):
        replace(EvaluationSpec(1, 12, 2, 5), **{field: value})


def test_checkpoint_directory_resolution_and_strict_architecture(tmp_path):
    model = _policy()
    torch.save(model.state_dict(), tmp_path / "latest.pth")
    spec = EvaluationSpec(1, 12, 1, 1, policy_checkpoint=str(tmp_path), force_cpu=True,
                          in_channels=3, out_channels=8, hidden_channels=8, num_layers=2)
    # The fixture uses projection width 8 rather than the production width 64;
    # incompatible weights must be rejected, never partially loaded.
    with pytest.raises(RuntimeError, match="size mismatch"):
        EvaluationPolicy.from_spec(spec)
    from models.subcomplex_policy_factory import build_subcomplex_agent

    compatible = build_subcomplex_agent(model_type="egnn", in_channels=3, out_channels=8,
                                        hidden_channels=8, num_layers=2, device="cpu")
    torch.save(compatible.state_dict(), tmp_path / "latest.pth")
    loaded = EvaluationPolicy.from_spec(spec)
    assert loaded.metadata["checkpoint"] == str((tmp_path / "latest.pth").resolve())
    assert len(loaded.metadata["checkpoint_sha256"]) == 64
    assert not loaded.model.training
    for key, value in compatible.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[key])
    (tmp_path / "latest.pth").rename(tmp_path / "200.pth")
    assert Path(EvaluationPolicy.from_spec(spec).metadata["checkpoint"]).name == "200.pth"


def test_missing_checkpoint_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No policy checkpoint"):
        EvaluationPolicy.from_spec(EvaluationSpec(1, 12, 1, 1, policy_checkpoint=str(tmp_path)))
