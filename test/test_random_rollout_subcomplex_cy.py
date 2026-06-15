from typing import Dict, Iterable, Tuple

import numpy as np

import mdp.cy_rollout as cy_rollout
from mdp.cy_rollout import CYRandomRolloutEngine, create_transition_pool


def _simplices_key(simplices: Iterable[Tuple[int, ...]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(simplex)) for simplex in simplices))


class FakeCYState:
    def __init__(
        self,
        *,
        key: str,
        point_config_index: int,
        simplices: Iterable[Tuple[int, ...]],
        transition_by_action: Dict[Tuple[int, ...], tuple],
        ambiguous_actions: Iterable[Tuple[int, ...]] = (),
        is_target: bool = False,
    ):
        self.key = key
        self.point_config_index = point_config_index
        self.simplices = _simplices_key(simplices)
        self.actions_ready = False
        self.available_subcomplex_actions = tuple(transition_by_action.keys())
        self.ambiguous_subcomplex_actions = frozenset(ambiguous_actions)
        self._transition_by_action = dict(transition_by_action)
        self.find_calls = 0
        self.visitation = 0
        self.is_target = bool(is_target)

    def find_available_actions(self):
        self.find_calls += 1
        self.actions_ready = True

    def get_available_subcomplex_actions(self):
        return tuple(self.available_subcomplex_actions)

    def get_transition_output_from_subcomplex_action(self, action):
        canonical = tuple(int(v) for v in action)
        return self._transition_by_action[canonical]


def _build_fake_state_factory(state_by_simplices):
    def _factory(point_config_index, simplices):
        key = (int(point_config_index), _simplices_key(simplices))
        return state_by_simplices[key]

    return _factory


def test_rollout_step_reuses_cached_state_expansion():
    point_config_index = 7
    a_simplices = ((0, 1, 2), (0, 2, 3))
    b_simplices = ((0, 1, 3), (0, 2, 3))
    c_simplices = ((0, 1, 4), (0, 3, 4))

    state_c = FakeCYState(
        key="c",
        point_config_index=point_config_index,
        simplices=c_simplices,
        transition_by_action={},
        is_target=True,
    )
    state_b = FakeCYState(
        key="b",
        point_config_index=point_config_index,
        simplices=b_simplices,
        transition_by_action={
            (1, 2, 3, 4): (frozenset(c_simplices), frozenset(), "c"),
        },
    )
    state_a = FakeCYState(
        key="a",
        point_config_index=point_config_index,
        simplices=a_simplices,
        transition_by_action={
            (0, 1, 2, 3): (frozenset(b_simplices), frozenset(), "b"),
        },
    )

    state_factory = _build_fake_state_factory(
        {
            (point_config_index, _simplices_key(a_simplices)): state_a,
            (point_config_index, _simplices_key(b_simplices)): state_b,
            (point_config_index, _simplices_key(c_simplices)): state_c,
        }
    )
    engine = CYRandomRolloutEngine(
        base_states={"a": state_a},
        initial_states=[state_a],
        state_factory=state_factory,
        is_target_state_fn=lambda state: bool(getattr(state, "is_target", False)),
    )

    rng = np.random.default_rng(0)

    step1 = engine.rollout_step(
        [state_a],
        rng=rng,
        initial_state_pool=[state_a],
    )
    assert step1.dones == [False]
    assert step1.next_states[0].key == "b"
    assert state_a.find_calls == 1

    step2 = engine.rollout_step(
        step1.next_states,
        rng=rng,
        initial_state_pool=[state_a],
    )
    assert step2.dones == [True]
    assert step2.rewards == [1.0]
    assert step2.terminal_reasons == ["frt_or_frst"]
    assert step2.next_states[0].key == "a"
    assert state_b.find_calls == 1

    step3 = engine.rollout_step(
        step2.next_states,
        rng=rng,
        initial_state_pool=[state_a],
    )
    assert step3.next_states[0].key == "b"
    assert state_a.find_calls == 1
    assert engine.graph_node_count() == 3
    assert engine.graph_edge_count() == 2


def test_filter_actionable_initial_states_and_dead_end_reset():
    point_config_index = 11
    live_simplices = ((0, 1, 2), (0, 2, 3))
    dead_simplices = ((0, 1, 3), (1, 2, 3))
    target_simplices = ((0, 1, 4), (1, 3, 4))

    live_state = FakeCYState(
        key="live",
        point_config_index=point_config_index,
        simplices=live_simplices,
        transition_by_action={(0, 1, 2, 3): (frozenset(dead_simplices), frozenset(), "dead")},
    )
    dead_state = FakeCYState(
        key="dead",
        point_config_index=point_config_index,
        simplices=dead_simplices,
        transition_by_action={},
    )
    target_state = FakeCYState(
        key="target",
        point_config_index=point_config_index,
        simplices=target_simplices,
        transition_by_action={},
        is_target=True,
    )

    state_factory = _build_fake_state_factory(
        {
            (point_config_index, _simplices_key(live_simplices)): live_state,
            (point_config_index, _simplices_key(dead_simplices)): dead_state,
            (point_config_index, _simplices_key(target_simplices)): target_state,
        }
    )
    engine = CYRandomRolloutEngine(
        base_states={
            "live": live_state,
            "dead": dead_state,
            "target": target_state,
        },
        initial_states=[live_state, dead_state, target_state],
        state_factory=state_factory,
        is_target_state_fn=lambda state: bool(getattr(state, "is_target", False)),
    )

    actionable = engine.filter_actionable_initial_states()
    assert [state.key for state in actionable] == ["live"]
    assert live_state.find_calls == 1
    assert dead_state.find_calls == 1
    assert target_state.find_calls == 0

    rng = np.random.default_rng(1)
    step = engine.rollout_step(
        [dead_state],
        rng=rng,
        initial_state_pool=[live_state],
    )
    assert step.dones == [True]
    assert step.rewards == [0.0]
    assert step.terminal_reasons == ["dead_end_current"]
    assert step.next_states[0].key == "live"


def test_expand_states_multiprocessing_matches_sequential():
    point_config_index = 19
    a1_simplices = ((0, 1, 2), (0, 2, 3))
    b1_simplices = ((0, 1, 3), (0, 3, 4))
    a2_simplices = ((0, 1, 4), (1, 2, 4))
    b2_simplices = ((0, 2, 4), (2, 3, 4))

    seq_state_a1 = FakeCYState(
        key="a1",
        point_config_index=point_config_index,
        simplices=a1_simplices,
        transition_by_action={(0, 1, 2, 3): (frozenset(b1_simplices), frozenset(), "b1")},
    )
    seq_state_a2 = FakeCYState(
        key="a2",
        point_config_index=point_config_index,
        simplices=a2_simplices,
        transition_by_action={(0, 1, 2, 4): (frozenset(b2_simplices), frozenset(), "b2")},
    )
    seq_engine = CYRandomRolloutEngine(
        base_states={"a1": seq_state_a1, "a2": seq_state_a2},
        initial_states=[seq_state_a1, seq_state_a2],
        state_factory=_build_fake_state_factory({}),
        is_target_state_fn=lambda state: bool(getattr(state, "is_target", False)),
    )

    seq_actions, seq_summary = seq_engine.candidate_actions_for_states([seq_state_a1, seq_state_a2])

    mp_state_a1 = FakeCYState(
        key="a1",
        point_config_index=point_config_index,
        simplices=a1_simplices,
        transition_by_action={(0, 1, 2, 3): (frozenset(b1_simplices), frozenset(), "b1")},
    )
    mp_state_a2 = FakeCYState(
        key="a2",
        point_config_index=point_config_index,
        simplices=a2_simplices,
        transition_by_action={(0, 1, 2, 4): (frozenset(b2_simplices), frozenset(), "b2")},
    )
    mp_engine = CYRandomRolloutEngine(
        base_states={"a1": mp_state_a1, "a2": mp_state_a2},
        initial_states=[mp_state_a1, mp_state_a2],
        state_factory=_build_fake_state_factory({}),
        is_target_state_fn=lambda state: bool(getattr(state, "is_target", False)),
    )

    pool = create_transition_pool(num_workers=2, start_method="spawn")
    try:
        mp_actions, mp_summary = mp_engine.candidate_actions_for_states(
            [mp_state_a1, mp_state_a2],
            use_multiprocessing=True,
            transition_pool=pool,
            transition_mp_chunksize=8,
            transition_mp_min_batch=1,
        )
    finally:
        pool.shutdown()

    assert seq_actions == mp_actions
    assert seq_summary.expanded_count == 2
    assert mp_summary.expanded_count == 2
    assert mp_summary.used_multiprocessing is True
    assert seq_engine.graph_node_count() == mp_engine.graph_node_count() == 4
    assert seq_engine.graph_edge_count() == mp_engine.graph_edge_count() == 2


def test_expand_states_falls_back_when_multiprocessing_pool_breaks(monkeypatch):
    point_config_index = 23
    a_simplices = ((0, 1, 2), (0, 2, 3))
    b_simplices = ((0, 1, 3), (0, 3, 4))

    state_a = FakeCYState(
        key="a",
        point_config_index=point_config_index,
        simplices=a_simplices,
        transition_by_action={(0, 1, 2, 3): (frozenset(b_simplices), frozenset(), "b")},
    )
    engine = CYRandomRolloutEngine(
        base_states={"a": state_a},
        initial_states=[state_a],
        state_factory=_build_fake_state_factory({}),
        is_target_state_fn=lambda state: bool(getattr(state, "is_target", False)),
    )

    class BrokenPool:
        def __init__(self):
            self.calls = 0

        def map(self, func, iterable, chunksize=1):
            self.calls += 1
            raise RuntimeError("simulated pool failure")

    pool = BrokenPool()
    monkeypatch.setattr(cy_rollout, "_CY_ROLLOUT_MP_DISABLED", False)

    actions, summary = engine.candidate_actions_for_states(
        [state_a],
        use_multiprocessing=True,
        transition_pool=pool,
        transition_mp_chunksize=8,
        transition_mp_min_batch=1,
    )

    assert actions == [((0, 1, 2, 3),)]
    assert summary.expanded_count == 1
    assert summary.discovered_count == 1
    assert summary.used_multiprocessing is False
    assert pool.calls == 1
    assert cy_rollout._CY_ROLLOUT_MP_DISABLED is True
