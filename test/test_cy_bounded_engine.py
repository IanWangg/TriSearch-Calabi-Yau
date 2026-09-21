from __future__ import annotations

import gc
from pathlib import Path
import weakref

import numpy as np
import pytest

from core.cy_bounded_cache import BoundedLRU
from core.cy_state_history import StateHistory
from mdp.cy_graph import CYGraphTransition, CYStateExpansion
from mdp.cy_rollout import (
    CYRandomRolloutEngine,
    build_cy_rollout_collection,
    create_runtime_state_cache,
    create_transition_pool,
    register_runtime_state,
    runtime_cache_total_unique_states,
)
from mdp.cy_state_record import CYPointConfiguration, CyStateRecord, canonical_simplices


def _ring_records(mode="regular", count=9):
    vertices = tuple((index, 0) for index in range(count + 3))
    configuration = CYPointConfiguration(7401, vertices, vertices, tuple(range(len(vertices))), mode == "regular")
    return [CyStateRecord(configuration, frozenset({(0, 1, index + 3), (0, 2, index + 3)}), mode,
                          is_frst=bool(index % 2), is_target=bool(index % 2))
            for index in range(count)]


class RingExpansionPool:
    """Exact ordered worker protocol without needing a valid geometric ring."""

    def __init__(self, records):
        self.records = records
        self.by_simplices = {canonical_simplices(record.simplices): index for index, record in enumerate(records)}
        self.worker_pids = (101, 102)
        self.returned = []

    def imap(self, function, requests, chunksize=1):
        assert chunksize == 1
        for request in requests:
            assert request["operation"] == "expand"
            source = self.records[self.by_simplices[request["state"][1]]]
            assert request["configuration"] == source.configuration
            assert request["objective_mode"] is True
            index = self.by_simplices[canonical_simplices(source.simplices)]
            actions = ((0, 2, 7), (0, 1, 4, 6))
            transitions = []
            for offset, action in enumerate(actions, 1):
                destination = self.records[(index + offset) % len(self.records)]
                transitions.append((action, CYGraphTransition(
                    next_key=destination.key, next_is_target=destination.is_target,
                    removed_simplices=canonical_simplices(source.simplices - destination.simplices),
                    added_simplices=canonical_simplices(destination.simplices - source.simplices),
                    next_is_frst=destination.is_frst,
                )))
            expansion = CYStateExpansion(source.key, source.point_config_index, canonical_simplices(source.simplices),
                                         actions, frozenset({(0, 3, 8)}), tuple(transitions))
            self.returned.append(expansion)
            yield expansion


def _ring_engine(records, *, history_path, cache_budget_bytes=100_000, max_hot_states=1):
    pool = RingExpansionPool(records)
    engine = CYRandomRolloutEngine(
        base_states={records[0].key: records[0]}, initial_states=[records[0]],
        polytope_by_index={records[0].point_config_index: records[0].configuration},
        vertices_by_polytope={records[0].point_config_index: records[0].vertices},
        include_points_interior_to_facets=records[0].configuration.include_points_interior_to_facets,
        neighbor_mode=records[0].neighbor_mode,
        reward_function=lambda before, after: float(len(before.simplices) - len(after.simplices)),
        transition_pool=pool, history_path=str(history_path),
        cache_budget_bytes=cache_budget_bytes, max_hot_states=max_hot_states,
    )
    return engine, pool


def test_oversized_runtime_state_bypasses_cache_without_losing_unique_count():
    state = _ring_records()[1]
    cache = create_runtime_state_cache(mode="lru", base_states={}, max_hot_states=1, max_bytes=0)
    assert register_runtime_state(cache, state) is True
    assert register_runtime_state(cache, state) is False
    assert runtime_cache_total_unique_states(cache) == 1
    assert len(cache.hot_states) == 0


def test_engine_rejects_reused_history_without_corrupting_exact_keys(tmp_path):
    path = tmp_path / "history.sqlite3"
    engine, _pool = _ring_engine(_ring_records(), history_path=path)
    engine.candidate_actions_for_states(engine.initial_states)
    discovered_count = engine.graph_node_count()
    engine.close()
    with pytest.raises(ValueError, match="history is run-local"):
        _ring_engine(_ring_records(), history_path=path)
    history = StateHistory(path)
    try:
        assert len(history.keys("discovered")) == discovered_count
    finally:
        history.close()


@pytest.mark.parametrize("mode", ["regular", "two_neighbors"])
@pytest.mark.parametrize("budget", [0, 100_000])
def test_more_active_lanes_than_cache_capacity_preserve_deltas_and_metadata(tmp_path, mode, budget):
    records = _ring_records(mode)
    engine, pool = _ring_engine(records, history_path=tmp_path / "history.sqlite3", cache_budget_bytes=budget)
    states = records[:5]
    try:
        for _ in range(4):
            actions, _summary = engine.candidate_actions_for_states(states)
            assert all(candidates == ((0, 2, 7), (0, 1, 4, 6)) for candidates in actions)
            next_states = []
            for source, candidates in zip(states, actions):
                transition = engine.nodes_by_key[source.key].transitions[candidates[0]]
                assert transition.next_simplices == ()
                destination = engine.materialize_transition(source, transition)
                assert destination.key == transition.next_key
                assert canonical_simplices(destination.simplices) == transition.simplices_from(source.simplices)
                assert destination.is_target == transition.next_is_target
                assert destination.is_frst == transition.next_is_frst
                assert destination.neighbor_mode == mode
                assert destination.configuration is records[0].configuration
                next_states.append(destination)
            states = next_states
            assert len(engine.state_cache.hot_states) <= 1
        assert len({state.key for state in states}) == 5
        engine._active_keys.clear()
        engine.prune_runtime_caches()
        assert len(engine.nodes_by_key) <= (0 if budget == 0 else 1)
        assert engine.memory_stats()["resident_graph_bytes"] <= budget * 3 // 4
    finally:
        engine.close()


def test_random_rollout_with_tiny_caches_matches_full_cache_and_exact_history(tmp_path):
    tiny_records, full_records = _ring_records(), _ring_records()
    tiny, tiny_pool = _ring_engine(tiny_records, history_path=tmp_path / "tiny.sqlite3")
    full, _full_pool = _ring_engine(full_records, history_path=tmp_path / "full.sqlite3", max_hot_states=100)
    tiny_states, full_states = tiny_records[:5], full_records[:5]
    tiny_rng, full_rng = np.random.default_rng(72), np.random.default_rng(72)
    try:
        for step in range(20):
            small = tiny.rollout_step(tiny_states, rng=tiny_rng, initial_state_pool=tiny.initial_states)
            large = full.rollout_step(full_states, rng=full_rng, initial_state_pool=full.initial_states)
            assert small.chosen_actions == large.chosen_actions
            assert small.rewards == large.rewards
            assert small.dones == large.dones
            assert small.terminal_reasons == large.terminal_reasons
            assert [state.key for state in small.next_states] == [state.key for state in large.next_states]
            tiny_states, full_states = small.next_states, large.next_states
            assert tiny.graph_node_count() == full.graph_node_count()
            assert tiny.graph_edge_count() == full.graph_edge_count()
            if step % 3 == 0:
                tiny._active_keys.clear()
                tiny.prune_runtime_caches(pressure=True)
        unique_expansions = {expansion.key: expansion for expansion in tiny_pool.returned}
        discovered = {record.key for record in tiny.base_states.values()}
        for expansion in unique_expansions.values():
            discovered.add(expansion.key)
            discovered.update(transition.next_key for _, transition in expansion.transitions)
        assert len(tiny_pool.returned) > len(unique_expansions)
        assert tiny.graph_node_count() == len(discovered)
        assert tiny.graph_edge_count() == sum(len(expansion.transitions) for expansion in unique_expansions.values())
        assert tiny.history is not None
        assert type(tiny.state_cache.runtime_unique_keys).__name__ == "HistorySet"
    finally:
        tiny.close()
        full.close()


@pytest.mark.parametrize("remaining", [0, 1])
def test_shrinking_active_batch_releases_previous_pins_without_new_expansion(tmp_path, remaining):
    records = _ring_records()
    engine, pool = _ring_engine(records, history_path=tmp_path / "pins.sqlite3")
    try:
        engine.candidate_actions_for_states(records[:5])
        assert len(engine.nodes_by_key) == 5
        requests_before = len(pool.returned)
        engine.expand_states(records[:remaining])
        assert len(pool.returned) == requests_before
        assert len(engine.nodes_by_key) <= 1
    finally:
        engine.close()


def test_compaction_does_not_repopulate_all_base_nodes_beyond_cache_limit(tmp_path):
    records = _ring_records()
    engine = CYRandomRolloutEngine(
        base_states={record.key: record for record in records}, initial_states=records,
        polytope_by_index={records[0].point_config_index: records[0].configuration},
        max_hot_states=1, cache_budget_bytes=100_000, history_path=str(tmp_path / "base.sqlite3"),
    )
    try:
        assert len(engine.nodes_by_key) <= 1
        assert engine.graph_node_count() == len(records)
        engine.compact_runtime_graph_to_base()
        assert len(engine.nodes_by_key) <= 1
        assert engine.graph_node_count() == len(records)
    finally:
        engine.close()


def test_disk_visitation_counts_survive_eviction_and_reopening(tmp_path):
    from core.cy_policy_rollout import compute_cy_state_count_bonus

    records = _ring_records()
    path = tmp_path / "counts.sqlite3"
    engine, _pool = _ring_engine(records, history_path=path)
    counts = engine.history.counts("visits")
    counts[records[1].key] = 8
    reference = {records[1].key: 8}
    try:
        for _ in range(3):
            engine.candidate_actions_for_states(records[:5])
            engine._active_keys.clear()
            engine.prune_runtime_caches(pressure=True)
            engine.compact_runtime_graph_to_base()
            actual = compute_cy_state_count_bonus(input_states=[records[0]], transitioned_states=[records[1]],
                                                  visit_counts_by_key=counts, coef=3., exponent=.5)
            expected = compute_cy_state_count_bonus(input_states=[records[0]], transitioned_states=[records[1]],
                                                    visit_counts_by_key=reference, coef=3., exponent=.5)
            assert actual == expected == [1.]
        assert len(counts) == 1
    finally:
        engine.close()
    reopened = StateHistory(path)
    try:
        assert dict(reopened.counts("visits")) == reference
    finally:
        reopened.close()


def test_history_namespaces_do_not_share_counts(tmp_path):
    history = StateHistory(tmp_path / "namespaces.sqlite3")
    try:
        counts = history.counts("visits")
        discovered = history.keys("discovered")
        counts["state"] = 91
        assert discovered.add("state") is True
        assert discovered.add("state") is False
        assert counts["state"] == 91
        counts.clear()
        assert "state" in discovered
        assert len(discovered) == 1
    finally:
        history.close()


def test_base_records_and_pickle_do_not_retain_geometry(tmp_path):
    import pickle

    records = _ring_records()
    engine, _pool = _ring_engine(records, history_path=tmp_path / "lightweight.sqlite3")
    try:
        assert not any(hasattr(state, "cy_triangulation") for state in engine.base_states.values())
        assert all(isinstance(polytope, CYPointConfiguration) for polytope in engine.polytope_by_index.values())
        payload = pickle.loads(pickle.dumps(records[0]))
        assert payload._objective_provider is None
        assert payload.configuration == records[0].configuration
        engine_ref = weakref.ref(engine)
    finally:
        engine.close()
    del engine
    gc.collect()
    assert engine_ref() is None


def _square_rows():
    return [{
        "polytope_index": 7450 + index,
        "vertices": [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
        "frst_list": [{"simplices": [[0, 1, 2], [0, 2, 4], [0, 3, 4], [0, 1, 3]],
                       "triangulation_list": [{"simplices": [[1, 2, 4], [1, 3, 4]]}]}],
    } for index in range(3)]


def test_real_managed_serial_parallel_rollouts_agree_under_eviction(tmp_path):
    import importlib.util

    if importlib.util.find_spec("cytools") is None:
        pytest.skip("CYTools is required for the managed geometry parity check.")
    signatures = []
    for num_workers in (1, 2):
        with create_transition_pool(num_workers=num_workers, task_timeout_sec=90) as pool:
            collection = build_cy_rollout_collection(_square_rows(), include_points_interior_to_facets=True, transition_pool=pool)
            engine = CYRandomRolloutEngine(collection=collection, max_hot_states=1, cache_budget_bytes=100_000,
                                          history_path=str(tmp_path / f"workers_{num_workers}.sqlite3"),
                                          reward_function=lambda before, after: float(len(before.simplices) - len(after.simplices)))
            try:
                states = list(collection.initial_states)
                rng = np.random.default_rng(119)
                trajectory = []
                for _ in range(4):
                    result = engine.rollout_step(states, rng=rng, initial_state_pool=collection.initial_states,
                                                 use_multiprocessing=num_workers > 1)
                    trajectory.append((result.chosen_actions, result.rewards, result.dones, result.terminal_reasons,
                                       [(state.key, state.is_target, state.is_frst) for state in result.next_states]))
                    states = result.next_states
                    assert len(engine.state_cache.hot_states) <= 1
                signatures.append(trajectory)
                assert engine.memory_stats()["graph_evictions"] > 0
            finally:
                engine.close()
    assert signatures[0] == signatures[1]
