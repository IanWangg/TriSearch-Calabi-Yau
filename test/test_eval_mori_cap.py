"""Mori minimization, contextual queue refresh, and raw-count comparisons."""

from dataclasses import replace
import json

import numpy as np
import pytest

from core.cy_episode_context import MinimizationEpisodeContext
from eval.algorithm import BestFirstAlgorithm, RLValueBestFirst
from eval.batched_rollout import run_batched_rollouts
from eval.config import EvaluationSpec
from eval.policy import EvaluationPolicy
from eval.results.plotting import (
    read_comparison, plot_evaluation, combine_comparisons, combine_labeled_comparisons,
)
from mdp.cy_rollout import CYRandomRolloutEngine
from reward_functions import get_reward
from test_eval_rl import GraphPolicy, run_rl_graph, indices
from test_eval_rollout import GraphPool, run_graph
from test_eval_comparison import make_run, write_json, write_jsonl


def test_mori_zero_value_weight_matches_minimizing_vanilla_befs(tmp_path):
    kwargs = dict(adjacency=[[1, 2], [3], [4], [], []], values=[30, 20, 25, 10, 9], budget=3)
    baseline, pool, queries, _ = run_graph(tmp_path / "base", BestFirstAlgorithm(), goal="min", **kwargs)
    results, new_pool, _, events = run_rl_graph(
        tmp_path / "rl", lambda: RLValueBestFirst(-1, 0), goal="min", objective_name="min_mori_cap",
        forbid_actor=True, **kwargs)
    assert results[0].best_objective == baseline.best_objective == 10
    assert indices(pool, queries) == indices(new_pool, events["queries"])
    algorithm = RLValueBestFirst(value_discount=1)
    assert algorithm.score_candidate(0, 0, 30, 20, get_reward("min_mori_cap"), 3) == -17
    EvaluationSpec(1, 30, 1, 3, algorithms=("rl_value_best_first",), reward_function="min_mori_cap")


@pytest.mark.parametrize("cache", [False, True])
def test_mori_context_refreshes_known_frontier_without_objective_requeries(tmp_path, cache):
    pool = GraphPool([[1, 2], [3], [4], [5], [], []], [30, 20, 25, 10, 9, 8])
    contexts = []

    class ContextPolicy(GraphPolicy):
        requires_episode_context = True

        def score_actions(self, states, action_lists, *, episode_contexts=None):
            assert all(isinstance(context, MinimizationEpisodeContext) for context in episode_contexts)
            return super().score_actions(states, action_lists)

        def score_values(self, states, *, episode_contexts=None):
            contexts.extend(episode_contexts)
            result = []
            for state, context in zip(states, episode_contexts):
                index = self.index(state)
                assert index in pool.objective_calls
                assert context.current_objective == pool.values[index]
                result.append(3 if index == 1 else (5 if index == 2 and context.best_objective <= 10
                                                    else 1 if index == 2 else 0))
            return result

    initial = pool.states[0]
    engine = CYRandomRolloutEngine(
        base_states={initial.key: initial}, initial_states=[initial],
        polytope_by_index={0: initial.configuration}, neighbor_mode="two_neighbors",
        include_points_interior_to_facets=False, reward_function=get_reward("min_mori_cap"),
        transition_pool=pool, state_cache_mode="lru" if cache else "none",
        cache_budget_bytes=1000000 if cache else 0, history_path=str(tmp_path / "states.sqlite3"))
    expansions, observations = [], []
    try:
        result = run_batched_rollouts(
            [initial], [RLValueBestFirst(2, 1)], engine, policy=ContextPolicy(pool),
            objective_function=lambda state: pool.values[pool.by_simplices[state.simplices]],
            objective_name="min_mori_cap", objective_goal="min", objective_budget=5,
            # The toy geometry pool exposes a single objective channel.
            batch_objective_function=lambda states: engine.objective_values(states, "max_kcup"),
            seeds=[0], start_indices=[0], on_expansion=expansions.append,
            search_observer=observations.append)[0]
        assert result.best_objective == 8 and result.objective_queries == 5
        assert indices(pool, expansions) == [0, 1, 2, 3]
        assert MinimizationEpisodeContext(25, 10, 2) in contexts
        assert len(pool.objective_calls) == 6
        assert any(row["event"] == "frontier_rescore" for row in observations)
        assert next(row for row in observations if row["event"] == "candidate_score")["score"] == -17
    finally:
        engine.close()


def test_contextual_mori_checkpoint_round_trip_and_horizon_cap(tmp_path):
    from core.cy_checkpointing import save_policy_checkpoint
    from models.subcomplex_policy_factory import build_subcomplex_agent

    model = build_subcomplex_agent(model_type="egnn", in_channels=4, out_channels=8, hidden_channels=8,
                                  num_layers=1, training_reward_mode="best_so_far", episode_horizon=64)
    save_policy_checkpoint(model, str(tmp_path / "400.pth"))
    spec = EvaluationSpec(1, 30, 1, 1000, algorithms=("rl_value_best_first",),
                          reward_function="min_mori_cap", force_cpu=True, policy_checkpoint=str(tmp_path),
                          out_channels=8, hidden_channels=8, num_layers=1)
    policy = EvaluationPolicy.from_spec(spec)
    assert policy.requires_episode_context and policy.device.type == "cpu"
    assert policy.episode_context(30, 20, 1000) == MinimizationEpisodeContext(30, 20, 64)
    assert policy.episode_context(30, 20, 3).remaining_steps == 3
    with pytest.raises(ValueError, match="min_mori_cap"):
        EvaluationPolicy.from_spec(replace(spec, reward_function="max_kcup"))


def test_mori_reader_validates_minimum_and_excludes_overshoot(tmp_path):
    run = tmp_path / "run"
    make_run(run)
    config = json.loads((run / "config.json").read_text())
    config["spec"]["reward_function"] = "min_mori_cap"
    write_json(run / "config.json", config)
    traces = [json.loads(line) for line in (run / "queries.jsonl").read_text().splitlines()]
    for row in traces:
        row["objective"] = row["best_objective"] = 1000 - row["objective"]
    write_jsonl(run / "queries.jsonl", traces)
    rows = [json.loads(line) for line in (run / "rollouts.jsonl").read_text().splitlines()]
    for row in rows:
        row.update(objective_name="min_mori_cap", objective_goal="min",
                   initial_objective=1000-row["initial_objective"], best_objective=1000-row["best_objective"])
    write_jsonl(run / "rollouts.jsonl", rows)
    data = read_comparison(run)
    np.testing.assert_array_equal(data.curves["greedy", 0, 0], [999, 998, 997, 996])
    output = plot_evaluation(run)
    assert (output / "best_mori_cap_by_polytope.png").is_file()
    assert json.loads((output / "plot_config.json").read_text())["objective_goal"] == "min"


def test_explicit_labels_preserve_different_checkpoint_provenance_and_pair_checks(tmp_path):
    paths = {"incremental": tmp_path / "incremental", "best_so_far": tmp_path / "best_so_far"}
    for label, path in paths.items():
        make_run(path, algorithms=("rl_value_best_first",))
        config = json.loads((path / "config.json").read_text())
        config["policy"] = {"checkpoint_sha256": label}
        write_json(path / "config.json", config)
    with pytest.raises(ValueError, match="duplicate algorithms"):
        combine_comparisons(paths.values())
    data = combine_labeled_comparisons(paths)
    assert data.spec["algorithms"] == ["incremental", "best_so_far"]
    output = plot_evaluation(paths["incremental"], tmp_path / "comparison", labeled_run_dirs=paths)
    metadata = json.loads((output / "plot_config.json").read_text())
    assert metadata["labeled_sources"]["best_so_far"]["policy"]["checkpoint_sha256"] == "best_so_far"
    config_path = paths["best_so_far"] / "config.json"
    config = json.loads(config_path.read_text())
    config["spec"]["mori_cap_backend"] = "mosek"
    write_json(config_path, config)
    with pytest.raises(ValueError, match="scientific settings"):
        combine_labeled_comparisons(paths)
