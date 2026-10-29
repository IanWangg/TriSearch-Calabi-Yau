"""MOSEK extremality certificates, exact polyhedral references and backend routing."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from core.mori_extremal_rays import (
    ExtremalRaySolveError, NonPointedConeError, _solve_checked, mosek_extremal_rays,
)
from core.cy_training_config import CYTrainingConfig, parse_args
from mdp.cy_rollout import CYRandomRolloutEngine
from mdp.cy_state_record import BoundedCache, CYPointConfiguration, CyStateRecord
from reward_functions import get_objective, get_reward
from reward_functions.min_mori_cap import mori_cap_target


def _canonical(rays):
    rays = np.asarray(rays, dtype=np.int64)
    rays = rays[np.any(rays != 0, axis=1)]
    if not len(rays):
        return set()
    return {tuple(row) for row in rays // np.gcd.reduce(rays, axis=1)[:, None]}


@pytest.mark.parametrize("optimizer", ["primal_simplex", "dual_simplex"])
@pytest.mark.parametrize("rays,expected", [
    ([[1, 0], [0, 1], [1, 1]], {(1, 0), (0, 1)}),
    ([[2, 0], [3, 0], [0, 0], [0, 2], [2, 2]], {(1, 0), (0, 1)}),
    ([[1, 0, 0], [0, 1, 0], [1, 1, 0]], {(1, 0, 0), (0, 1, 0)}),
    ([[-1, 0], [-2, 0]], {(-1, 0)}),
    ([[0, 0]], set()),
])
def test_extremal_rays_handles_redundancy_scaling_zeros_and_lower_dimensional_cones(optimizer, rays, expected):
    stats = {}
    result = mosek_extremal_rays(rays, optimizer=optimizer, diagnostics=stats)
    assert _canonical(result) == expected
    assert stats["retries"] == 0
    assert stats["extremal_rays"] == len(expected)


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
def test_random_pointed_cones_match_exact_ppl_extremal_generators(dimension):
    from cytools.cone import dualize

    rng = np.random.default_rng(170 + dimension)
    base = rng.integers(-4, 5, size=(12, dimension))
    base[:, 0] = rng.integers(1, 4, size=len(base))  # A known strictly positive functional.
    interior = rng.integers(1, 4, size=(8, len(base))) @ base
    rays = np.vstack((base, interior, 3 * base[:3]))
    exact = _canonical(dualize(dualize(rays)))
    result = mosek_extremal_rays(rays)
    assert _canonical(result) == exact
    # Permutation and positive rescaling must not change the geometry or count.
    permuted = rays[rng.permutation(len(rays))] * rng.integers(1, 5, size=(len(rays), 1))
    assert _canonical(mosek_extremal_rays(permuted)) == exact


def test_floating_generators_are_not_rounded_to_integer_rays():
    result = mosek_extremal_rays([[.5, 0.], [0., .25], [.25, .25]])
    assert {tuple(ray) for ray in result} == {(.5, 0.), (0., .25)}


@pytest.mark.parametrize("optimizer", ["primal_simplex", "dual_simplex"])
def test_real_degenerate_mori_cap_preserves_fixed_variable_bounds(optimizer):
    # Captured from the first full-size h11=15 training rollout. With presolve
    # disabled, MOSEK 11.1.10 could report optimal with a fixed weight of -0.625.
    fixture = Path(__file__).parent / "fixtures/mori_cap_fixed_bounds_rays.npz"
    with np.load(fixture) as data:
        rays = data["rays"]
        expected = data["expected_extremal_rays"]
    stats = {}
    result = mosek_extremal_rays(rays, optimizer=optimizer, diagnostics=stats)
    assert len(result) == 27
    assert _canonical(result) == _canonical(expected)


@pytest.mark.parametrize("rays", [[[1, 0], [-1, 0]], [[1, 0], [-1, 0], [0, 1]]])
def test_non_pointed_cones_require_lineality_decomposition(rays):
    with pytest.raises(NonPointedConeError):
        mosek_extremal_rays(rays)


@pytest.mark.parametrize("rays", [[[np.nan, 0]], [[np.inf, 1]], [1, 2], [[]]])
def test_invalid_rays_are_rejected(rays):
    with pytest.raises(ValueError):
        mosek_extremal_rays(rays)


@pytest.mark.parametrize("termination_name", ["ok", "trm_max_time"])
def test_unknown_or_timed_out_solves_fail_after_one_retry(termination_name):
    import mosek

    task = Mock()
    task.getintparam.return_value = mosek.optimizertype.dual_simplex
    task.getsolsta.return_value = mosek.solsta.unknown
    task.getdouinf.return_value = 0.0
    task.optimize.return_value = getattr(mosek.rescode, termination_name)
    stats = dict(retries=0, lp_solves=0, optimizer_sec=0.)
    with pytest.raises(ExtremalRaySolveError, match="two attempts"):
        _solve_checked(task, lambda status: None, stats)
    assert task.optimize.call_count == 2
    assert stats["retries"] == 1


def test_unverified_infeasibility_certificate_cannot_increase_extremal_count(monkeypatch):
    import mosek

    monkeypatch.setattr(mosek.Task, "gety", lambda task, *args: np.zeros(task.getnumcon()))
    with pytest.raises(ExtremalRaySolveError, match="Ray"):
        mosek_extremal_rays([[1, 0], [0, 1], [1, 1]])


def test_mosek_reward_counts_rays_without_calling_slow_extremal_method():
    cap = SimpleNamespace(rays=lambda: np.array([[1, 0], [0, 1], [1, 1]]),
                          extremal_rays=Mock(side_effect=AssertionError("Unexpected fallback")))
    cy = SimpleNamespace(mori_cone_cap=lambda **kwargs: cap)
    state = SimpleNamespace(key="pointed", cy_triangulation=SimpleNamespace(get_cy=lambda: cy))
    reward = get_reward("min_mori_cap", mori_cap_backend="mosek")
    assert get_objective("min_mori_cap", reward=reward)(state) == 2.
    assert reward.from_objectives(5., 2.) == 3.
    cap.extremal_rays.assert_not_called()


def test_non_pointed_reward_preserves_cytools_minimal_generator_convention():
    from cytools import Cone

    cap = Cone([[1, 0], [-1, 0], [0, 1]])
    cy = SimpleNamespace(mori_cone_cap=lambda **kwargs: cap)
    assert mori_cap_target(cy, backend="mosek") == float(len(cap.extremal_rays()))


def test_invalid_backend_and_wrong_objective_are_rejected():
    with pytest.raises(ValueError, match="backend"):
        get_reward("min_mori_cap", mori_cap_backend="unknown")
    with pytest.raises(ValueError, match="only valid"):
        get_reward("max_kcup", mori_cap_backend="mosek")
    args = parse_args(["--reward", "min_mori_cap", "--mori_cap_backend", "mosek"])
    assert CYTrainingConfig.from_namespace(args).mori_cap_backend == "mosek"


def test_evaluation_backend_cli_and_config_preserve_setup_identity():
    from dataclasses import replace
    from eval.config import EvaluationSpec
    from scripts.eval_cy import parse_args as eval_args

    spec = EvaluationSpec(1, 12, 1, 2, reward_function="min_mori_cap")
    changed = replace(spec, mori_cap_backend="mosek")
    assert changed.setup_parameters() == spec.setup_parameters()
    args = eval_args(["--num_polytopes", "1", "--h11", "12", "--num_starts", "1",
                      "--objective_budget", "2", "--reward_function", "min_mori_cap",
                      "--mori_cap_backend", "mosek"])
    assert args.mori_cap_backend == "mosek"
    with pytest.raises(ValueError, match="only valid"):
        replace(changed, reward_function="max_kcup")


def _record():
    points = ((0, 0), (1, 0), (0, 1))
    configuration = CYPointConfiguration(971, points, points, (0, 1, 2), True)
    return CyStateRecord(configuration, frozenset({(0, 1, 2)}))


def test_worker_objective_cache_separates_backends(monkeypatch):
    from mdp import cy_geometry_worker as worker

    monkeypatch.setattr(worker, "_OBJECTIVES", BoundedCache(max_entries=10))
    cap = SimpleNamespace(rays=lambda: np.array([[1, 0], [0, 1], [1, 1]]),
                          extremal_rays=Mock(return_value=np.ones((7, 2))))
    cy = SimpleNamespace(mori_cone_cap=lambda **kwargs: cap)
    monkeypatch.setattr(worker, "_triangulate", lambda *args: SimpleNamespace(get_cy=lambda: cy))
    state = _record()
    for backend, expected in (("cytools", 7.), ("mosek", 2.), ("cytools", 7.)):
        request = dict(reward_name="min_mori_cap", mori_cap_backend=backend)
        assert worker._objective(request, state.configuration, state.to_payload(), None) == expected
    assert len(worker._OBJECTIVES) == 2
    cap.extremal_rays.assert_called_once()


def test_engine_routes_backend_for_scalar_batch_and_cached_objectives():
    class Pool:
        def __init__(self):
            self.requests = []

        def imap(self, function, requests, **kwargs):
            for request in requests:
                self.requests.append(request)
                yield 2. if request["mori_cap_backend"] == "mosek" else 7.

    state, pool = _record(), Pool()
    engine = CYRandomRolloutEngine(base_states={state.key: state}, initial_states=[state],
        reward_function=get_reward("min_mori_cap"), transition_pool=pool, cache_budget_bytes=1000000)
    try:
        assert engine.objective_value(state, "min_mori_cap") == 7.
        engine.reward_function = get_reward("min_mori_cap", mori_cap_backend="mosek")
        assert list(engine.objective_values([state, state], "min_mori_cap")) == [2., 2.]
        assert engine.objective_value(state, "min_mori_cap") == 2.
        engine.reward_function = get_reward("min_mori_cap")
        assert list(engine.objective_values([state], "min_mori_cap")) == [7.]
        assert [request["mori_cap_backend"] for request in pool.requests] == ["cytools", "mosek"]
    finally:
        engine.close()


@pytest.mark.parametrize("mode", ["incremental", "best_so_far"])
def test_real_h11_15_training_with_mosek_backend_saves_checkpoints_and_metrics(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    checkpoints = tmp_path / "checkpoints"
    metrics_path = tmp_path / "metrics.jsonl"
    logs = tmp_path / "logs"
    logs.mkdir()
    command = [sys.executable, "scripts/train_cy.py", "--dataset_path",
        "data/cy/output4d_hugging_face/cy_4d_h11_15_favorable_1000_frst_10.samples.jsonl",
        "--max_rows", "2", "--num_eval_polytopes", "1", "--reward", "min_mori_cap",
        "--mori_cap_backend", "mosek", "--training_reward_mode", mode,
        "--neighbor_mode", "two_neighbors", "--no-include_points_interior_to_facets",
        "--force_cpu", "--num_iterations", "1", "--num_states", "2", "--rollout_length", "3",
        "--batch_size", "4", "--num_eval_states", "1", "--eval_steps", "1", "--eval_interval", "1",
        "--hidden_channels", "8", "--out_channels", "8", "--num_layers", "1",
        "--entropy_coef", ".001", "--count_bonus_coef", ".001",
        "--use_multiprocessing", "--transition_num_workers", "2",
        "--memory_budget_gb", "8", "--runtime_cache_gb", ".05",
        "--save_interval", "1", "--latest_checkpoint_interval", "1",
        "--checkpoint_path", str(checkpoints), "--iteration_metrics_path", str(metrics_path)]
    with (logs / "train.log").open("w") as log:
        result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                 "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert result.returncode == 0, (logs / "train.log").read_text()
    record = json.loads(metrics_path.read_text())
    assert record["mori_cap_backend"] == "mosek"
    assert (checkpoints / "latest.pth").is_file() and (checkpoints / "final.pth").is_file()
    assert all(math.isfinite(record["ppo"][key]) for key in ("total_loss", "policy_loss", "value_loss"))
    objectives = record["train"]["objective"]
    ending = objectives["best_values"] if mode == "best_so_far" else objectives["final_values"]
    expected = np.mean(np.asarray(objectives["initial_values"]) - np.asarray(ending))
    assert record["train"]["return"]["mean"] == pytest.approx(expected)
    assert record["train"]["return"]["training_mean"] >= expected
