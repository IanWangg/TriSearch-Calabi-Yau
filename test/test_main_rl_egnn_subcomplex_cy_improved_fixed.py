from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import core.cy_policy_rollout_utils as cy_policy_rollout_utils
from core.training_types import FirstEpisodeTracker
from core.cy_policy_rollout_utils import collect_policy_rollout
from core.cy_data_utils import resolve_policy_in_channels


class DummyEngine:
    def sample_initial_states(self, num_states, *, rng, initial_state_pool=None):
        pool = list(initial_state_pool or [])
        return pool[: int(num_states)]


def _state(key: str) -> SimpleNamespace:
    return SimpleNamespace(key=key, visitation=0)


def test_resolve_policy_in_channels_infers_dataset_dimension():
    rows = [
        {"polytope_index": 0, "vertices": [[0, 0, 0, 0], [1, 0, 0, 0]]},
        {"polytope_index": 1, "vertices": [[0, 0, 0, 0], [0, 1, 0, 0]]},
    ]

    assert resolve_policy_in_channels(rows, None) == 4

    with pytest.raises(ValueError, match="--in_channels must match"):
        resolve_policy_in_channels(rows, 3)


def test_first_episode_tracker_records_terminal_reason_counts_before_reset():
    tracker = FirstEpisodeTracker.create(num_envs=2, gamma=0.5)

    tracker.update(
        rewards=[1.0, 0.0],
        dones=[True, False],
        terminal_reasons=["frt_or_frst", "continue"],
        step_index=0,
    )
    tracker.update(
        rewards=[-1.0, -1.0],
        dones=[True, True],
        terminal_reasons=["single_simplex", "single_simplex"],
        step_index=1,
    )

    assert tracker.success_rate() == 0.5
    assert tracker.mean_discounted_reward() == 0.25
    assert tracker.finished_fraction() == 1.0
    assert tracker.success_count() == 1
    assert tracker.collapsed_count() == 1
    assert tracker.dead_end_count() == 0
    assert tracker.finished_count() == 2


def test_collect_policy_rollout_uses_first_episode_counts_for_public_metrics(monkeypatch):
    initial_states = [_state("env0"), _state("env1")]
    step_results = [
        SimpleNamespace(
            rewards=[-1.0, 0.0],
            dones=[True, False],
            terminal_reasons=["single_simplex", "continue"],
            next_states=[_state("reset0"), _state("env1_step1")],
            frt_hits=0,
            collapsed_hits=1,
            dead_end_hits=0,
            reset_count=1,
            expanded_states=3,
            discovered_states=5,
            used_multiprocessing=False,
            action_candidates=[((1,),), ((2,),)],
            valid_action_mask=torch.tensor([True, True]),
            candidate_expand_sec=0.1,
            policy_data_build_sec=0.2,
            policy_batch_transfer_sec=0.3,
            policy_value_inference_sec=0.4,
            policy_action_inference_sec=0.5,
            transition_apply_sec=0.6,
        ),
        SimpleNamespace(
            rewards=[0.0, 1.0],
            dones=[True, True],
            terminal_reasons=["dead_end_current", "frt_or_frst"],
            next_states=[_state("reset1"), _state("reset2")],
            frt_hits=1,
            collapsed_hits=0,
            dead_end_hits=1,
            reset_count=2,
            expanded_states=7,
            discovered_states=11,
            used_multiprocessing=True,
            action_candidates=[tuple(), ((3,),)],
            valid_action_mask=torch.tensor([False, True]),
            candidate_expand_sec=1.0,
            policy_data_build_sec=1.1,
            policy_batch_transfer_sec=1.2,
            policy_value_inference_sec=1.3,
            policy_action_inference_sec=1.4,
            transition_apply_sec=1.5,
        ),
    ]

    def fake_rollout_step_with_policy(*args, **kwargs):
        return step_results.pop(0)

    monkeypatch.setattr(cy_policy_rollout_utils, "rollout_step_with_policy", fake_rollout_step_with_policy)

    summary = collect_policy_rollout(
        engine=DummyEngine(),
        policy=object(),
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
        initial_state_pool=initial_states,
        num_envs=2,
        rollout_length=2,
        gamma=0.5,
        deterministic=False,
        use_multiprocessing=False,
        transition_pool=None,
        transition_mp_chunksize=1,
        transition_mp_min_batch=1,
        store_buffer=False,
        report_every=0,
        label="rollout",
    )

    assert summary.success_rate == 0.5
    assert summary.discounted_reward == -0.25
    assert summary.finished_fraction == 1.0
    assert summary.finished_count == 2
    assert summary.frt_hits == 1
    assert summary.collapsed_hits == 1
    assert summary.dead_end_hits == 0
    assert summary.all_step_reset_count == 3
    assert summary.all_step_frt_hits == 1
    assert summary.all_step_collapsed_hits == 1
    assert summary.all_step_dead_end_hits == 1
