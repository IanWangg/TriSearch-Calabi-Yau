"""Tests for multi-file datasets, a separate eval file, h11 weights and KL early stop."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.cy_data_utils import (
    combine_dataset_rows,
    h11_by_polytope_index,
    parse_h11_sampling_weights,
    split_rows_with_eval_dataset,
)
from core.cy_policy_inference import batched_policy_action_selection
from core.cy_ppo import PPORolloutBuffer, train_policy_from_rollout
from core.cy_training_config import parse_args
from mdp.cy_rollout import CYRandomRolloutEngine
from test_cy_bounded_ppo import GeometryState, _policy, _step


def _row(index, h11=15):
    return {"polytope_index": index, "h11": h11, "vertices": [[0, 0, 0, 0]], "frst_list": []}


def test_combine_rows_rejects_polytope_in_two_files():
    rows = combine_dataset_rows([[_row(0), _row(1)], [_row(2)]])
    assert [row["polytope_index"] for row in rows] == [0, 1, 2]
    with pytest.raises(ValueError, match="more than one file"):
        combine_dataset_rows([[_row(0)], [_row(0)]])


def test_eval_dataset_split_keeps_all_training_rows_and_rejects_overlap():
    split = split_rows_with_eval_dataset([_row(0), _row(1)], [_row(5)])
    assert split.train_polytope_indices == [0, 1]
    assert split.eval_polytope_indices == [5]
    with pytest.raises(ValueError, match="share polytope_index"):
        split_rows_with_eval_dataset([_row(0), _row(1)], [_row(1)])


def test_parse_h11_sampling_weights():
    assert parse_h11_sampling_weights(None) is None
    assert parse_h11_sampling_weights("15=3, 20=1") == {15: 3.0, 20: 1.0}
    for bad in ("15", "15=-1", "15=0", "15=1,15=2", "15=nan"):
        with pytest.raises(ValueError):
            parse_h11_sampling_weights(bad)


def test_cli_accepts_several_dataset_files_eval_file_weights_and_target_kl():
    args = parse_args(["--dataset_path", "a.jsonl", "b.jsonl", "--eval_dataset_path", "dev.jsonl",
                       "--h11_sampling_weights", "15=3,20=1", "--target_kl", "0.015"])
    assert args.dataset_path == ["a.jsonl", "b.jsonl"]
    assert args.eval_dataset_path == ["dev.jsonl"]
    assert args.h11_sampling_weights == "15=3,20=1"
    assert args.target_kl == 0.015
    defaults = parse_args([])
    assert isinstance(defaults.dataset_path, list) and defaults.eval_dataset_path is None
    assert defaults.target_kl is None


def _engine_with_pool(h11_by_polytope):
    engine = CYRandomRolloutEngine.__new__(CYRandomRolloutEngine)
    engine.initial_states = []
    engine._initial_state_h11_by_polytope = None
    engine._initial_state_weight_by_h11 = None
    return engine


def test_h11_weights_set_share_per_h11_and_uniform_within_h11():
    # 6 starts at h11=15, 2 at h11=20.
    pool = [SimpleNamespace(point_config_index=index) for index in (0, 0, 1, 1, 2, 2, 10, 11)]
    h11 = {0: 15, 1: 15, 2: 15, 10: 20, 11: 20}
    engine = _engine_with_pool(h11)
    # No weights: uniform over starts, and the old integers() RNG path is unchanged.
    uniform = engine.sample_initial_states(5, rng=np.random.default_rng(3), initial_state_pool=pool)
    expected = [pool[int(i)] for i in np.random.default_rng(3).integers(0, len(pool), size=5)]
    assert uniform == expected
    engine.set_initial_state_h11_weights(h11, {15: 1.0, 20: 1.0})
    probabilities = engine.initial_state_probabilities(pool)
    assert probabilities[:6] == pytest.approx([1 / 12] * 6)
    assert probabilities[6:] == pytest.approx([1 / 4] * 2)
    draws = engine.sample_initial_states(20000, rng=np.random.default_rng(0), initial_state_pool=pool)
    share_20 = sum(h11[state.point_config_index] == 20 for state in draws) / len(draws)
    assert share_20 == pytest.approx(0.5, abs=0.02)
    engine.set_initial_state_h11_weights(h11, {15: 1.0})
    with pytest.raises(KeyError, match="h11=20"):
        engine.initial_state_probabilities(pool)
    with pytest.raises(ValueError):
        engine.set_initial_state_h11_weights(h11, None)


def test_h11_by_polytope_index_requires_h11():
    assert h11_by_polytope_index([_row(0, 15), _row(1, 20)]) == {0: 15, 1: 20}
    with pytest.raises(ValueError, match="no h11"):
        h11_by_polytope_index([{"polytope_index": 3}])


def _prepared_rollout(model):
    states = [GeometryState(str(index)) for index in range(8)]
    actions = [[(0, 1, 2), (0, 2, 3)]] * 8
    selection = batched_policy_action_selection(states, actions, model, device=torch.device("cpu"))
    buffer = PPORolloutBuffer()
    step = _step(states, selection)
    step.rewards = [float(index % 2) * 5.0 for index in range(8)]
    buffer.append(step)
    return buffer.prepare(bootstrap_value=torch.zeros(8), gamma=.9, gae_lambda=.95, device=torch.device("cpu"))


def test_target_kl_none_keeps_old_update_and_small_target_stops_early():
    model = _policy()
    prepared = _prepared_rollout(model)
    kwargs = dict(prepared_rollout=prepared, device=torch.device("cpu"), num_epochs=6, batch_size=2,
                  clip_coef=.2, value_coef=.5, entropy_coef=.0, max_grad_norm=10.0)
    free_model, stopped_model = copy.deepcopy(model), copy.deepcopy(model)
    torch.manual_seed(5)
    free = train_policy_from_rollout(policy=free_model, optimizer=torch.optim.SGD(free_model.parameters(), lr=2.0),
                                     **kwargs)
    assert free.num_minibatch_updates == 6 * 4 and not free.early_stopped
    assert free.approx_kl > 0
    torch.manual_seed(5)
    stopped = train_policy_from_rollout(policy=stopped_model,
                                        optimizer=torch.optim.SGD(stopped_model.parameters(), lr=2.0),
                                        target_kl=1e-9, **kwargs)
    assert stopped.early_stopped
    # The first minibatch is on-policy (KL 0), so at least one update happens.
    assert 1 <= stopped.num_minibatch_updates < free.num_minibatch_updates
    with pytest.raises(ValueError, match="target_kl"):
        train_policy_from_rollout(policy=model, optimizer=torch.optim.SGD(model.parameters(), lr=.1),
                                  target_kl=0.0, **kwargs)
