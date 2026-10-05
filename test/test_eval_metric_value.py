"""RL value search uses absolute log metric; legacy metric names are aliases."""

import math

import pytest

from eval.algorithm import (
    BeamSearchAlgorithm, BestFirstAlgorithm, RLMetricValueBeamSearch, RLMetricValueBestFirst,
    RLValueBeamSearch, RLValueBestFirst, get_algorithm,
)
from eval.config import EvaluationSpec
from test_eval_rl import indices, run_rl_graph
from test_eval_rollout import run_graph


@pytest.mark.parametrize("factory", [RLValueBeamSearch, RLValueBestFirst, RLMetricValueBeamSearch, RLMetricValueBestFirst])
def test_absolute_score_uses_natural_log_and_ignores_parent_and_reward(factory):
    class ForbiddenReward:
        def from_objectives(self, *args):
            raise AssertionError("Absolute scoring must not compute a transition reward")

    algorithm = factory(value_discount=0.9)
    for parent in (1, 1000):
        assert algorithm.score_candidate(123, -5, parent, 10, ForbiddenReward(), 2) == pytest.approx(
            math.log(10) + 0.9 * 2)


@pytest.mark.parametrize("factory,discount,expected", [
    (lambda discount: RLMetricValueBeamSearch(2, -1, discount), 0, [0, 1, 2, 3, 4]),
    (lambda discount: RLMetricValueBeamSearch(2, -1, discount), 0.9, [0, 1, 2, 4, 3]),
    (lambda discount: RLMetricValueBestFirst(-1, discount), 0, [0, 1, 3, 2, 4]),
    (lambda discount: RLMetricValueBestFirst(-1, discount), 0.9, [0, 1, 2, 4, 3]),
    (lambda discount: RLMetricValueBeamSearch(2, -1, discount), 3.0, [0, 1, 2, 4, 3]),
    (lambda discount: RLMetricValueBestFirst(-1, discount), 3.0, [0, 1, 2, 4, 3]),
])
def test_absolute_metric_changes_cross_parent_order_and_discount(tmp_path, factory, discount, expected):
    _, pool, policy, events = run_rl_graph(
        tmp_path, lambda: factory(discount), adjacency=[[1, 2], [3], [4], [], []],
        values=[1, 10, 2, 11, 5], critic={3: -2}, forbid_actor=True,
    )
    assert indices(pool, events["expansions"]) == expected
    assert not policy.action_batches


@pytest.mark.parametrize("factory,expected", [
    (lambda: RLValueBeamSearch(2, -1, 0), [0, 1, 2, 3, 4]),
    (lambda: RLValueBestFirst(-1, 0), [0, 1, 3, 2, 4]),
])
def test_value_names_rank_by_absolute_metric_across_parents(tmp_path, factory, expected):
    _, pool, _, events = run_rl_graph(
        tmp_path, factory, adjacency=[[1, 2], [3], [4], [], []],
        values=[1, 10, 2, 11, 5], forbid_actor=True,
    )
    assert indices(pool, events["expansions"]) == expected


@pytest.mark.parametrize("baseline,factory", [
    (BeamSearchAlgorithm(2), lambda: RLMetricValueBeamSearch(2, -1, 0)),
    (BestFirstAlgorithm(), lambda: RLMetricValueBestFirst(-1, 0)),
])
def test_zero_discount_all_neighbors_matches_objective_baseline(tmp_path, baseline, factory):
    kwargs = dict(adjacency=[[0, 1, 2], [0, 3, 4], [0, 3, 5], [], [6, 7], [7], [], []],
                  values=[1, 10, 2, 11, 12, 3, 15, 14], budget=6)
    original, pool, queries, _ = run_graph(tmp_path / "base", baseline, **kwargs)
    results, new_pool, _, events = run_rl_graph(tmp_path / "new", factory, forbid_actor=True, **kwargs)
    for field in ("objective_queries", "budget_overshoot", "expansion_count", "transition_count",
                  "best_objective", "termination_reason"):
        assert getattr(original, field) == getattr(results[0], field)
    assert indices(pool, queries) == indices(new_pool, events["queries"])
    assert indices(pool, pool.expansion_events) == indices(new_pool, events["expansions"])


@pytest.mark.parametrize("name,count", [("rl_value_beam_search", 8), ("rl_value_best_first", 4),
                                        ("rl_metric_value_beam_search", 8), ("rl_metric_value_best_first", 4)])
def test_registration_defaults_and_reject_unscaled_objectives(tmp_path, name, count):
    from scripts.eval_cy import parse_args

    args = parse_args(["--num_polytopes", "1", "--h11", "23", "--num_starts", "1",
                       "--objective_budget", "5", "--algorithms", name, "--beam_width", "8"])
    algorithm = get_algorithm(name, beam_width=args.beam_width, policy_proposal_count=args.policy_proposal_count)
    assert algorithm.proposal_count == count and algorithm.value_discount == 0.9
    with pytest.raises(ValueError, match="only max_kcup"):
        EvaluationSpec(1, 23, 1, 5, algorithms=(name,), reward_function="min_tri")
    with pytest.raises(ValueError, match="supports only"):
        run_rl_graph(tmp_path, lambda: get_algorithm(name), adjacency=[[]], values=[1],
                     objective_name="min_tri", goal="min")


@pytest.mark.parametrize("name", ["rl_value_beam_search", "rl_value_best_first",
                                  "rl_metric_value_beam_search", "rl_metric_value_best_first"])
def test_metric_coefficient_above_one_config_cli_and_score(name):
    from scripts.eval_cy import parse_args

    args = parse_args(["--num_polytopes", "1", "--h11", "30", "--num_starts", "1",
                       "--objective_budget", "5", "--algorithms", name, "--value_discount", "3"])
    spec = EvaluationSpec(1, 30, 1, 5, algorithms=(name,), value_discount=args.value_discount)
    algorithm = get_algorithm(name, value_discount=spec.value_discount)
    assert algorithm.score_candidate(0, 0, 1, math.e, None, 2) == pytest.approx(7)
    assert spec.setup_parameters() == EvaluationSpec(1, 30, 1, 5).setup_parameters()
    for invalid in (-1, math.nan, math.inf):
        with pytest.raises(ValueError, match="value_discount"):
            EvaluationSpec(1, 30, 1, 5, algorithms=(name,), value_discount=invalid)
        with pytest.raises(ValueError, match="value_discount"):
            get_algorithm(name, value_discount=invalid)
    EvaluationSpec(1, 30, 1, 5, algorithms=("rl_value_beam_search", "rl_value_best_first"), value_discount=3)
