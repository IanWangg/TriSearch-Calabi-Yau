"""FaceMoveScorer must reproduce the trained invariant actor's own logits for flips on a real state."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

pytest.importorskip("cyopt")

SETUP = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/runs/face_mutation_20261104/setups/seed0_poly0_s3")
CHECKPOINT = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/runs/inv_rl_full_20261104/checkpoints/final.pth")


@pytest.fixture(scope="module")
def parent_state():
    if not (SETUP / "samples.jsonl").exists() or not CHECKPOINT.exists():
        pytest.skip("B1 setup or invariant checkpoint not available")
    from eval.setup import load_eval_setup
    from mdp.cy_rollout import build_cy_rollout_collection, create_transition_pool

    setup = load_eval_setup(SETUP)
    pool = create_transition_pool(num_workers=1)
    try:
        collection = build_cy_rollout_collection(
            setup.rows, include_points_interior_to_facets=False, neighbor_mode="two_neighbors",
            transition_pool=pool, two_face_state=False, include_two_face_metadata=True)
        yield next(iter(collection.base_states.values()))
    finally:
        pool.close()


@pytest.fixture(scope="module")
def scorer():
    from eval.algorithm.face_move_scorer import FaceMoveScorer
    return FaceMoveScorer(CHECKPOINT, aggregation="mean2")


def _faces(state):
    from mdp.cy_state_record import two_face_restrictions
    return two_face_restrictions(state.configuration, state.simplices)


def _flips(state, limit):
    """Four-point circuits on one 2-face with exactly two present triangles (a flip, as the actor sees it)."""
    flips = []
    for face_index, (labels, triangles) in enumerate(_faces(state)):
        present = set(triangles)
        found = 0
        for quad in combinations(labels, 4):
            triples = list(combinations(quad, 3))
            removed = [t for t in triples if t in present]
            added = [t for t in triples if t not in present]
            if len(removed) == 2 and len(added) == 2:
                new = (present - set(removed)) | set(added)
                flips.append((face_index, labels, quad, tuple(sorted(new))))
                found += 1
                if found == 2:  # at most two per face, so several faces are exercised
                    break
    return flips[:limit]


def _model_logit(scorer, state, quad):
    from core.two_face_invariant_features import create_invariant_two_face_data_from_state
    data = create_invariant_two_face_data_from_state(state, actions=(tuple(sorted(quad)),))
    with torch.no_grad():
        _, logits = scorer.model.get_value_and_logits(Batch.from_data_list([data]))
    assert logits.shape[1] == 1
    return float(logits[0, 0])


def test_flip_logits_match_model_actor(parent_state, scorer):
    flips = _flips(parent_state, limit=12)
    assert len(flips) >= 6, "expected several flips on a h11=50 parent"
    faces_seen = {face for face, *_ in flips}
    assert len(faces_seen) >= 3
    for face_index, labels, quad, new in flips:
        expected = _model_logit(scorer, parent_state, quad)
        moves = [(labels, new)]
        got = scorer.score(parent_state, moves)[0]
        assert abs(got - expected) < 1e-5, (labels, quad, got, expected)


def test_sum_and_mean2_agree_on_flips(parent_state):
    from eval.algorithm.face_move_scorer import FaceMoveScorer
    mean2 = FaceMoveScorer(CHECKPOINT, aggregation="mean2")
    total = FaceMoveScorer(CHECKPOINT, aggregation="sum")
    moves = [(labels, new) for _, labels, _, new in _flips(parent_state, limit=6)]
    assert np.allclose(mean2.score(parent_state, moves), total.score(parent_state, moves), atol=1e-6)


def test_face_matched_by_labels_and_batch_order_invariant(parent_state, scorer):
    flips = _flips(parent_state, limit=5)
    moves = [(labels, new) for _, labels, _, new in flips]
    forward = scorer.score(parent_state, moves)
    backward = scorer.score(parent_state, moves[::-1])
    assert np.allclose(forward, backward[::-1], atol=1e-6)
    assert np.isfinite(forward).all()


def test_unmatched_face_and_non_move_raise(parent_state, scorer):
    labels, triangles = _faces(parent_state)[0]
    with pytest.raises(ValueError, match="not a 2-face"):
        scorer.score(parent_state, [((-1, -2, -3), triangles)])
    with pytest.raises(ValueError, match="unchanged"):
        scorer.score(parent_state, [(labels, triangles)])
    with pytest.raises(ValueError, match="not a 3-subset"):
        scorer.score(parent_state, [(labels, tuple(triangles) + ((-1, -2, -3),))])
