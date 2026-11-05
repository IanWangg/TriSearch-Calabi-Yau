"""rl_jump.JumpEnv: reward, candidate mapping, flip-path agreement, caching, failure handling, batching."""

import math
from pathlib import Path

import numpy as np
import pytest
from torch_geometric.data import Batch

pytest.importorskip("cyopt")

DATA = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/data/cy/favorable_v1/train/h11_15.samples.jsonl")
CHECKPOINT = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/runs/inv_rl_full_20261104/checkpoints/final.pth")


@pytest.fixture(scope="module")
def env():
    if not DATA.exists():
        pytest.skip("favorable_v1 h11=15 rows not available")
    from rl_jump.env import JumpEnv
    from rl_jump.vec import load_row

    return JumpEnv(load_row(DATA, 0), K=32, seed=0, horizon=6)


def _restrictions(state):
    from mdp.cy_state_record import two_face_restrictions
    return two_face_restrictions(state.configuration, state.simplices)


def _independent_volume(env, state):
    """Objective through the geometry worker's own request path (separate from the env's objective code)."""
    from mdp.cy_geometry_worker import execute_geometry_request
    return execute_geometry_request({"operation": "objective", "state": state.to_payload(),
                                     "configuration": env.configuration, "reward_name": "max_kcup"})


def _move_fields(obs, owner):
    """Removed triangle ids and added node triples of candidate `owner` in an observation."""
    data = obs.data
    removed = sorted(int(i) for i, o in zip(data.move_removed_ids.tolist(), data.move_removed_owner.tolist())
                     if o == owner)
    added = sorted(tuple(t) for t, o in zip(data.move_added_nodes.tolist(), data.move_added_owner.tolist())
                   if o == owner)
    return removed, added


def test_reset_step_reward_is_log_volume_difference(env):
    obs = env.reset(0)
    assert obs.candidates, "the start must have feasible single-face candidates"
    assert obs.value == pytest.approx(_independent_volume(env, env.starts[0]), rel=1e-9)
    for _ in range(3):
        if env.done:
            break
        current_state, current_value = env.state, env.value
        obs, reward, done, info = env.step(0)
        assert info["failure"] is None
        independent_next = _independent_volume(env, env.state)
        assert info["value"] == pytest.approx(independent_next, rel=1e-9)
        expected = math.log(independent_next) - math.log(_independent_volume(env, current_state))
        assert reward == pytest.approx(expected, abs=1e-9)
        assert current_value == pytest.approx(_independent_volume(env, current_state), rel=1e-9)


def test_candidate_moves_map_to_codebook_faces_and_decoded_dna(env):
    from core.cy_two_face_data import _build_topology

    obs = env.reset(0)
    restrictions = _restrictions(env.state)
    index_of = {labels: i for i, (labels, _) in enumerate(restrictions)}
    topology = _build_topology(env.state.configuration, restrictions)
    for owner, cand in enumerate(env.candidates):
        face = index_of[cand.face_labels]
        old = set(restrictions[face][1])
        new = set(cand.new_triangles)
        removed, added = old - new, new - old
        assert (old - removed) | added == new  # applying the move gives the candidate FRT of that face
        child = dict(_restrictions(cand.state))
        assert child[cand.face_labels] == tuple(sorted(new))
        for position, face_id in enumerate(env.faces):  # every 2-face equals its decoded DNA entry
            assert child[env.face_labels[face_id]] == tuple(sorted(env.codebook[face_id][cand.dna[position]]))
        rem, add = _move_fields(obs, owner)
        assert rem == sorted(topology.triangle_maps[face][t] for t in removed)
        assert add == sorted(tuple(sorted(topology.node_maps[face][label] for label in t)) for t in added)


def test_flip_candidate_matches_existing_flip_action_path(env):
    from core.cy_two_face_data import build_two_face_data

    obs = env.reset(0)
    restrictions = _restrictions(env.state)
    index_of = {labels: i for i, (labels, _) in enumerate(restrictions)}
    flips = [(owner, cand) for owner, cand in enumerate(env.candidates) if cand.symdiff == 4]
    assert flips, "expected a flip-sized candidate on the h11=15 start"
    for owner, cand in flips[:6]:
        old = set(restrictions[index_of[cand.face_labels]][1])
        new = set(cand.new_triangles)
        circuit = tuple(sorted({v for t in (old ^ new) for v in t}))
        assert len(circuit) == 4
        flip = build_two_face_data(env.state.configuration, restrictions, actions=(circuit,))
        flip_removed = sorted(int(i) for i in flip.removed_triangle_ids[0].tolist())
        flip_added = sorted(tuple(v) for v in flip.added_triangle_vertices[0].tolist())
        assert _move_fields(obs, owner) == (flip_removed, flip_added)


def test_stepping_into_known_dna_costs_no_query(env):
    env.reset(0)
    first = env.candidates[0]
    queries_before = env.counters["queries"]
    _, _, _, info = env.step(0)
    assert info["failure"] is None
    queries_after = env.counters["queries"]
    assert queries_after - queries_before <= 1
    # Same start again: the start objective is cached and the same candidate is proposed again.
    env.reset(0)
    assert env.counters["queries"] == queries_after
    assert env.candidates[0].dna == first.dna
    _, _, _, info = env.step(0)
    assert tuple(info["chosen_dna"]) == first.dna
    assert info["objective_cache_hit"] is True
    assert info["objective_queries"] == 0
    assert env.counters["queries"] == queries_after


def test_repeated_objective_request_is_one_query(env):
    env.reset(0)
    cand = env.candidates[0]
    before = env.counters["queries"]
    first = env._objective(cand.dna, cand.state)
    second = env._objective(cand.dna, cand.state)
    assert first == second
    assert env.counters["queries"] - before <= 1


def test_geometry_failure_is_isolated_to_the_candidate(env, monkeypatch):
    env.reset(0)
    # A candidate whose objective is not cached yet, so the injected failure is actually reached.
    action = next(i for i, c in enumerate(env.candidates) if c.dna not in env._objective_cache)
    bad = env.candidates[action]
    n_before, dna_before = len(env.candidates), env.dna
    original = env.metric

    def failing(state):
        if state.key == bad.state.key:
            raise ValueError("injected tip solver failure")
        return original(state)

    monkeypatch.setattr(env, "metric", failing)
    obs, reward, done, info = env.step(action)
    assert info["failure"] and "injected" in info["failure"]
    assert reward == 0.0
    assert env.dna == dna_before
    assert len(env.candidates) == n_before - 1 and len(obs.candidates) == n_before - 1


def test_batched_logits_match_face_move_scorer(env):
    if not CHECKPOINT.exists():
        pytest.skip("invariant checkpoint not available")
    from eval.algorithm.face_move_scorer import FaceMoveScorer
    from rl_jump.policy import batched_move_logits

    scorer = FaceMoveScorer(CHECKPOINT, aggregation="mean2")
    obs_a = env.reset(0)
    moves_a = [(c.face_labels, c.new_triangles) for c in env.candidates]
    expected_a = scorer.score(env.starts[0], moves_a)
    obs_b = env.reset(1)
    moves_b = [(c.face_labels, c.new_triangles) for c in env.candidates]
    expected_b = scorer.score(env.starts[1], moves_b)
    logits = batched_move_logits(scorer.model, Batch.from_data_list([obs_a.data, obs_b.data]),
                                 aggregation="mean2").numpy()
    assert logits.shape == (len(moves_a) + len(moves_b),)
    np.testing.assert_allclose(logits[:len(moves_a)], expected_a, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(logits[len(moves_a):], expected_b, rtol=1e-5, atol=1e-6)
