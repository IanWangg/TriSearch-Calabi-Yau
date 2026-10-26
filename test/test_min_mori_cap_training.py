import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from core.cy_episode_context import BEST_SO_FAR, MinimizationEpisodeContext, best_so_far_min_reward
from core.cy_policy_inference import (
    build_cy_data_list, configure_policy_execution, evaluate_policy_actions_from_data_list,
    evaluate_policy_scores,
)
from core.cy_policy_rollout import collect_policy_rollout
from core.cy_ppo import train_policy_from_rollout
from core.cy_training_config import parse_args
from models.subcomplex_policy_factory import build_subcomplex_agent
from reward_functions import get_reward
from test_cy_bounded_ppo import GeometryState
from test_cy_best_so_far_reward import EpisodeEngine


def _policy():
    torch.manual_seed(17)
    return build_subcomplex_agent(
        model_type="egnn", subcomplex_actor_type="snn_simplex", in_channels=3,
        hidden_channels=8, out_channels=8, num_layers=1,
        training_reward_mode=BEST_SO_FAR, episode_horizon=5,
    ).eval()


def test_minimum_records_telescope_without_rewarding_revisited_values():
    best, rewards = 8.0, []
    for value in [4, 6, 4, 2, 5, 0]:
        best, reward = best_so_far_min_reward(best, value)
        rewards.append(reward)
    assert rewards == [4, 0, 0, 2, 0, 2]
    assert sum(rewards) == 8 - best
    assert MinimizationEpisodeContext(6, 2, 4).features() == pytest.approx([6, 4, math.log(5)])
    with pytest.raises(ValueError):
        MinimizationEpisodeContext(2, 4, 1).features()


def test_best_so_far_cli_allows_requested_bonus_and_requires_minimization_snn():
    flags = ["--reward", "min_mori_cap", "--training_reward_mode", BEST_SO_FAR,
             "--count_bonus_coef", ".001", "--no-vertex_aug_enable", "--rollout_length", "64"]
    args = parse_args(flags)
    assert args.gamma == 1 and args.rollout_length == 64 and args.count_bonus_coef == .001
    for extra in (["--gamma", ".95"], ["--subcomplex_actor_type", "mlp"],
                  ["--reward", "max_kcup"], ["--vertex_aug_enable"]):
        with pytest.raises(ValueError):
            parse_args(flags + extra)


def test_snn_context_is_local_and_preserved_in_chunking_actor_critic_and_ppo():
    model, state = _policy(), GeometryState()
    contexts = [MinimizationEpisodeContext(8, 8, 5), MinimizationEpisodeContext(8, 2, 1)]
    states, actions = [state, state], [[(0, 1, 2), (0, 2, 3)]] * 2
    data = build_cy_data_list(states, actions, include_simplex_topology=True, episode_contexts=contexts)
    legacy = build_cy_data_list([state], [actions[0]], include_simplex_topology=True)
    assert getattr(legacy[0], "episode_context", None) is None
    batch = Batch.from_data_list(data)
    values, logits = model.get_value_and_logits(batch)
    torch.testing.assert_close(model.get_value(batch), values)
    assert not torch.isclose(values[0], values[1])
    # Different contexts must change action preferences, not just add a common logit shift.
    assert not torch.isclose(logits[0, 0] - logits[0, 1], logits[1, 0] - logits[1, 1], atol=1e-8, rtol=1e-6)
    try:
        configure_policy_execution(max_graph_size=1)
        chunked = evaluate_policy_scores(states, actions, model, device=torch.device("cpu"), episode_contexts=contexts)
        torch.testing.assert_close(chunked.value_tensor, values.detach())
    finally:
        configure_policy_execution()
    scored = evaluate_policy_actions_from_data_list(data, torch.tensor([0, 1]), model, device=torch.device("cpu"))
    (scored.value_tensor.sum() + scored.log_prob_tensor.sum()).backward()
    assert model.context_mlp[0].weight.grad[:, -3:].abs().sum() > 0
    with pytest.raises(ValueError, match="schema"):
        evaluate_policy_scores(states, actions, model, device=torch.device("cpu"))


class CountEpisodeEngine(EpisodeEngine):
    def __init__(self):
        super().__init__()
        geometry = GeometryState()
        for state in self.states:
            state.vertices, state.edges, state.simplices = geometry.vertices, geometry.edges, geometry.simplices
        self.action = (0, 1, 2)
        for state, target in zip(self.states, [1, 2, 3, 0, None]):
            transitions = {} if target is None else {self.action: SimpleNamespace(
                next_key=str(target), next_simplices=geometry.simplices)}
            self.nodes_by_key[state.key] = SimpleNamespace(transitions=transitions, candidate_actions=tuple(transitions))
        self.reward_function = get_reward("min_mori_cap")

    def objective(self, state):
        self.objective_calls.append(state.key)
        return [8., 4., 6., 2., 10.][int(state.key)]


def test_best_count_rollout_adds_bonus_masks_padding_and_stops_at_horizon():
    engine, model = CountEpisodeEngine(), _policy()
    summary = collect_policy_rollout(
        engine=engine, policy=model, rng=np.random.default_rng(0), device=torch.device("cpu"),
        initial_state_pool=engine.states, num_envs=2, rollout_length=5, gamma=1, deterministic=True,
        use_multiprocessing=False, transition_pool=None, transition_mp_chunksize=1, transition_mp_min_batch=1,
        store_buffer=True, report_every=0, label="test", objective_function=engine.objective,
        objective_name="min_mori_cap", objective_goal="min", training_reward_mode=BEST_SO_FAR,
        count_bonus_coef=.001, visit_counts_by_key={},
    )
    assert summary.episode_lengths == [5, 1]
    assert summary.episode_terminal_reasons == ["budget_exhausted", "dead_end_current"]
    assert summary.objective_best_values == [2, 10] and summary.objective_final_values == [4, 10]
    assert summary.return_mean == 3 and summary.training_return_mean > 3
    assert summary.training_return_mean == pytest.approx(summary.return_mean + summary.intrinsic_bonus_mean * 5)
    buffer = summary.rollout_buffer
    assert torch.stack(buffer.reward_buffer).sum(0)[1] == 0  # No bonus leaks into inactive padding.
    assert buffer.data_buffer[2][0].episode_context.tolist()[0] == pytest.approx([6, 2, math.log(4)])
    prepared = buffer.prepare(bootstrap_value=torch.full((2,), 1e6), gamma=1, gae_lambda=1, device=torch.device("cpu"))
    assert prepared.sample_valid_mask_flat.sum() == 6
    assert prepared.value_targets[0, 0] < 7  # Terminal masks exclude the poisoned bootstrap.
    stats = train_policy_from_rollout(
        policy=model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-4), prepared_rollout=prepared,
        device=torch.device("cpu"), num_epochs=1, batch_size=6, clip_coef=.1, value_coef=.5,
        entropy_coef=.001, max_grad_norm=1,
    )
    assert stats.num_samples == 6 and math.isfinite(stats.total_loss)
