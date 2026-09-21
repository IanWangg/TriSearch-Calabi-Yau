"""Integration checks for the shared memory guard and cache allocations."""

import builtins
import io
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

import core.cy_policy_rollout as policy_rollout
import core.cy_ppo as cy_ppo
from core.cy_data_utils import configure_cy_data_tensor_caches, get_cy_data_tensor_cache_stats
from core.cy_managed_runtime import managed_rollout_runtime
from core.cy_policy_inference import configure_policy_execution
from core.cy_process_runtime import MemoryBudgetExceeded
from mdp.cy_rollout import CYRandomRolloutEngine
from mdp.cy_state_record import CYPointConfiguration, CyStateRecord


class FakePool:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.checks = 0
        self.closed = False
        self.snapshot = {"rss_bytes": 123, "trainer_rss_bytes": 100}

    def check_memory(self):
        self.checks += 1
        return self.snapshot

    def shutdown(self):
        self.closed = True


def _arguments(**overrides):
    values = dict(
        memory_budget_gb=64.0, runtime_cache_gb=16.0,
        transition_num_workers=1, use_multiprocessing=True,
        transition_mp_start_method="spawn", transition_task_timeout_sec=300,
        shared_cache_max_entries=10, policy_max_graph_size=250000,
        profile_cuda_timing=False, max_rss_gb=None, action_order="canonical",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def _restore_execution_defaults():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    configure_cy_data_tensor_caches(max_bytes=0)
    yield
    configure_cy_data_tensor_caches(max_bytes=0)
    configure_cy_data_tensor_caches()
    configure_policy_execution()
    torch.set_num_threads(old_threads)


def _engine(index, cache_bytes, pool, history_path):
    points = ((0, 0), (1, 0), (0, 1), (1, 1))
    configuration = CYPointConfiguration(index, points, points, (0, 1, 2, 3), True)
    state = CyStateRecord(configuration, frozenset(((0, 1, 2), (1, 2, 3))))
    return CYRandomRolloutEngine(
        base_states={state.key: state}, initial_states=[state],
        polytope_by_index={index: configuration},
        vertices_by_polytope={index: points}, cache_budget_bytes=cache_bytes,
        history_path=str(history_path), transition_pool=pool,
    )


@pytest.mark.parametrize("cache_gb", [0.0, 0.001, 16.0])
@pytest.mark.parametrize("workers", [1, 4])
def test_combined_cache_limits_include_history_objectives_and_ipc(tmp_path, cache_gb, workers):
    args = _arguments(runtime_cache_gb=cache_gb, transition_num_workers=workers)
    with managed_rollout_runtime(args, FakePool) as (pool, engine_bytes, register):
        engines = [_engine(index, engine_bytes, pool, tmp_path / f"history_{index}.sqlite3")
                   for index in range(2)]
        for engine in engines:
            register(engine)
        tensor_bytes = sum(cache["max_bytes"] for cache in get_cy_data_tensor_cache_stats().values())
        worker_bytes = pool.options["initargs"][0]
        ipc_bytes = pool.options["configuration_cache_bytes"]
        allocated_bytes = 2 * engine_bytes + tensor_bytes + workers * worker_bytes + (workers + 1) * ipc_bytes
        assert allocated_bytes <= int(cache_gb * 1024**3)

        for engine in engines:
            page_cache_kb = engine.history.connection.execute("PRAGMA cache_size").fetchone()[0]
            assert page_cache_kb <= 0
            assert engine.history.connection.execute("PRAGMA mmap_size").fetchone()[0] == 0
            charged_bytes = (engine.state_cache.hot_states.max_bytes + engine._graph_max_bytes +
                             engine._objective_cache.max_bytes + abs(page_cache_kb) * 1024)
            assert charged_bytes <= engine_bytes
            engine._objective_cache[("volume", "state")] = 1.0
            engine.release_active_states()
            if cache_gb == 0:
                assert engine.memory_stats()["resident_graph_bytes"] == 0
                assert engine.memory_stats()["hot_state_bytes"] == 0
                assert engine.memory_stats()["objective_cache_bytes"] == 0
        if cache_gb == 0:
            assert allocated_bytes == ipc_bytes == worker_bytes == tensor_bytes == 0
    assert pool.closed
    assert all(engine.history is None for engine in engines)


def test_stricter_trainer_rss_limit_wraps_common_pool_guard(monkeypatch):
    args = _arguments(max_rss_gb=1.0)
    pages = [1]
    original_open = builtins.open

    def memory_file(path, *positional, **kwargs):
        if path == "/proc/self/statm":
            return io.StringIO(f"0 {pages[0]} 0 0 0 0 0")
        return original_open(path, *positional, **kwargs)

    monkeypatch.setattr(builtins, "open", memory_file)
    with managed_rollout_runtime(args, FakePool) as (pool, _, _):
        assert pool.check_memory() is pool.snapshot
        pages[0] = 1024**3 // os.sysconf("SC_PAGE_SIZE")
        with pytest.raises(MemoryBudgetExceeded, match="max_rss_gb=1"):
            pool.check_memory()
        assert pool.checks == 2
    assert pool.closed


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf")])
def test_invalid_trainer_limit_still_closes_created_pool(limit):
    pools = []

    def create_pool(**kwargs):
        pools.append(FakePool(**kwargs))
        return pools[-1]

    with pytest.raises(ValueError, match="max_rss_gb"):
        with managed_rollout_runtime(_arguments(max_rss_gb=limit), create_pool):
            pytest.fail("Invalid limit was accepted.")
    assert len(pools) == 1 and pools[0].closed


def test_setup_failure_closes_registered_engines_before_the_pool():
    events = []

    class OrderedPool(FakePool):
        def shutdown(self):
            events.append("pool")
            super().shutdown()

    with pytest.raises(RuntimeError, match="model setup failed"):
        with managed_rollout_runtime(_arguments(), OrderedPool) as (_, _, register):
            register(SimpleNamespace(close=lambda: events.append("train_engine")))
            register(SimpleNamespace(close=lambda: events.append("eval_engine")))
            raise RuntimeError("model setup failed")
    assert events == ["eval_engine", "train_engine", "pool"]


@pytest.mark.parametrize("explicit_pool", [True, False])
def test_cached_rollout_checks_budget_before_retaining_another_observation(monkeypatch, explicit_pool):
    pool = FakePool()
    state = SimpleNamespace(key="cached", visitation=0)
    engine = SimpleNamespace(
        transition_pool=pool,
        sample_initial_states=lambda *args, **kwargs: [state],
    )
    retained = []
    steps = []

    def check_memory():
        pool.checks += 1
        if pool.checks == 3:
            raise MemoryBudgetExceeded("cached rollout budget exhausted")
        return pool.snapshot

    def cached_step(*args, **kwargs):
        steps.append(True)
        return SimpleNamespace(
            input_states=[state], transitioned_states=[state], next_states=[state],
            rewards=[0.0], dones=[False], terminal_reasons=["continue"],
            frt_hits=0, collapsed_hits=0, dead_end_hits=0, reset_count=0,
            expanded_states=0, discovered_states=0, used_multiprocessing=False,
            action_candidates=[((0, 1),)], valid_action_mask=torch.tensor([True]),
            candidate_expand_sec=0.0, policy_data_build_sec=0.0,
            policy_batch_transfer_sec=0.0, policy_value_inference_sec=0.0,
            policy_action_inference_sec=0.0, transition_apply_sec=0.0,
        )

    pool.check_memory = check_memory
    monkeypatch.setattr(policy_rollout, "rollout_step_with_policy", cached_step)
    monkeypatch.setattr(policy_rollout, "PPORolloutBuffer", lambda: SimpleNamespace(append=retained.append))
    with pytest.raises(MemoryBudgetExceeded, match="cached rollout"):
        policy_rollout.collect_policy_rollout(
            engine=engine, policy=object(), rng=np.random.default_rng(0), device=torch.device("cpu"),
            initial_state_pool=[state], num_envs=1, rollout_length=1000, gamma=0.99,
            deterministic=True, use_multiprocessing=False,
            transition_pool=pool if explicit_pool else None,
            transition_mp_chunksize=1, transition_mp_min_batch=1,
            store_buffer=True, report_every=0, label="cached_rollout",
        )
    assert pool.checks == 3
    assert len(steps) == len(retained) == state.visitation == 2


def _prepared_batch(num_samples=5):
    observations = [Data(x=torch.ones(1, 1), edge_index=torch.empty(2, 0, dtype=torch.long),
                         subcomplex_vertices=torch.zeros(1, 1, dtype=torch.long),
                         num_available_subcomplexes=1) for _ in range(num_samples)]
    zeros = torch.zeros(1, num_samples)
    return cy_ppo.PreparedPPORolloutBatch(
        state_buffer_list=[None] * num_samples, candidate_buffer_list=[()] * num_samples,
        action_buffer_flat=torch.zeros(num_samples, 1, dtype=torch.long),
        action_index_buffer_flat=torch.zeros(num_samples, dtype=torch.long),
        log_prob_buffer_flat=zeros.reshape(-1), entropy_buffer_flat=zeros.reshape(-1),
        reward_buffer_tensor=zeros, value_buffer_tensor=zeros, done_buffer_tensor=zeros,
        valid_mask_flat=torch.ones(num_samples, dtype=torch.bool),
        advantages=torch.arange(num_samples, dtype=torch.float).reshape(1, -1),
        value_targets=torch.ones(1, num_samples), data_buffer_list=observations,
    )


def _train_with_guard(monkeypatch, guard, events):
    policy = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    original_step = optimizer.step

    def optimizer_step(*args, **kwargs):
        events.append("step")
        return original_step(*args, **kwargs)

    def evaluate(observations, action_indices, model, **kwargs):
        events.append("evaluate")
        values = model.weight.reshape(()).expand(len(observations))
        return SimpleNamespace(value_tensor=values, log_prob_tensor=values,
                               entropy_tensor=values * 0 + 0.5,
                               valid_action_mask=torch.ones(len(observations), dtype=torch.bool))

    monkeypatch.setattr(optimizer, "step", optimizer_step)
    monkeypatch.setattr(cy_ppo, "evaluate_policy_actions_from_data_list", evaluate)
    return cy_ppo.train_policy_from_rollout(
        policy=policy, optimizer=optimizer, prepared_rollout=_prepared_batch(),
        device=torch.device("cpu"), num_epochs=2, batch_size=3,
        clip_coef=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
        max_graph_size=1, memory_guard=guard,
    )


def test_ppo_checks_budget_before_each_logical_and_physical_batch(monkeypatch):
    events = []
    result = _train_with_guard(monkeypatch, lambda: events.append("guard"), events)
    assert result.num_samples == 5
    assert events.count("step") == 4  # Two logical batches in each of two epochs.
    assert events.count("evaluate") == 10
    assert events.count("guard") == 14
    assert all(index > 0 and events[index - 1] == "guard"
               for index, event in enumerate(events) if event == "evaluate")


@pytest.mark.parametrize("fail_on_check, expected_evaluations, expected_steps", [(1, 0, 0), (3, 1, 0), (5, 3, 1)])
def test_ppo_memory_failure_does_not_apply_an_incomplete_logical_update(
        monkeypatch, fail_on_check, expected_evaluations, expected_steps):
    events = []

    def memory_guard():
        events.append("guard")
        if events.count("guard") == fail_on_check:
            raise MemoryBudgetExceeded("PPO budget exhausted")

    with pytest.raises(MemoryBudgetExceeded, match="PPO budget"):
        _train_with_guard(monkeypatch, memory_guard, events)
    assert events.count("evaluate") == expected_evaluations
    assert events.count("step") == expected_steps
