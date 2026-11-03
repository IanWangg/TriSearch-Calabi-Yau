import numpy as np
import pytest

from data.cy.generate_diverse_starts import (
    _Pool,
    farthest_point_order,
    two_face_distance,
    two_face_key,
    two_face_parts,
)
from data.cy.select_polytopes import (
    assign_splits,
    candidate_rows,
    parse_split_specs,
    permuted_candidates,
    row_group_of,
)

# A favorable h11=15 polytope from calabi-yau-data/polytopes-4d (16-vertex file).
H11_15_VERTICES = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1],
                   [0, -1, -1, 0], [-1, 1, 0, 0], [-1, -1, 0, 0], [0, -1, 0, 1], [0, -1, -1, -1],
                   [-1, 0, 1, -1], [-1, -1, 0, -1], [-2, 0, 1, 0], [-2, 1, 1, -1], [-2, 0, 1, -1]]


def _index():
    return {"files": {
        "polytopes-4d-05-vertices.parquet": {"row_groups": [10], "rows": {"15": [1, 4, 7], "20": [2]}},
        "polytopes-4d-06-vertices.parquet": {"row_groups": [5, 5], "rows": {"15": [0, 3, 6, 9], "20": [8]}},
    }}


def test_parse_split_specs_rejects_duplicates():
    assert parse_split_specs(["train:15:3", "dev:15:1"]) == [("train", 15, 3), ("dev", 15, 1)]
    with pytest.raises(ValueError):
        parse_split_specs(["train:15:3", "train:15:1"])


def test_candidate_rows_skip_source_order_head():
    rows = candidate_rows(_index(), 15, exclude_first=2)
    assert rows == [("polytopes-4d-05-vertices.parquet", 7)] + [
        ("polytopes-4d-06-vertices.parquet", r) for r in (0, 3, 6, 9)]


def test_selection_is_deterministic_and_splits_are_disjoint():
    rows = candidate_rows(_index(), 15, exclude_first=0)
    first = permuted_candidates(rows, seed=0, h11=15)
    assert first == permuted_candidates(rows, seed=0, h11=15)
    assert sorted(first) == sorted(rows)
    accepted = [{"source_id": f"{name}:{row}"} for name, row in first]
    splits = assign_splits(accepted, [("dev", 2), ("train", 4)], seed=0, h11=15)
    assert [len(splits["train"]), len(splits["dev"])] == [4, 2]
    ids = [item["source_id"] for items in splits.values() for item in items]
    assert len(ids) == len(set(ids))
    assert splits == assign_splits(accepted, [("dev", 2), ("train", 4)], seed=0, h11=15)


def test_row_group_of_uses_row_group_bounds():
    assert row_group_of(_index(), "polytopes-4d-06-vertices.parquet", 6) == (1, 5)
    assert row_group_of(_index(), "polytopes-4d-06-vertices.parquet", 4) == (0, 0)


def test_farthest_point_order_prefers_far_candidates():
    anchor = (((0, 1, 2),), ((3, 4, 5),), ((6, 7, 8),))
    near = (((0, 1, 2),), ((3, 4, 5),), ((6, 7, 9),))
    far = (((0, 1, 3),), ((3, 4, 6),), ((6, 7, 9),))
    assert two_face_distance(anchor, far) == 3
    order = farthest_point_order([near, far], [anchor], 2, np.random.default_rng(0))
    assert order == [1, 0]


def test_pool_deduplicates_by_two_face_class():
    pytest.importorskip("cytools.polytope")
    from cytools import Polytope

    polytope = Polytope(H11_15_VERTICES)
    triangulations = polytope.random_triangulations_fast(
        N=10, as_list=True, progress_bar=False, seed=0, make_star=True, include_points_interior_to_facets=False)
    pool = _Pool()
    for triangulation in triangulations:
        pool.add(triangulation, "fast", 0)
    keys = {tuple(sorted(tuple(sorted(int(v) for v in s)) for s in t.simplices(on_faces_dim=2)))
            for t in triangulations}
    assert len(pool.items) == len(keys)
    assert not pool.add(triangulations[0], "fast", 0)
    for item in pool.items.values():
        assert two_face_key(two_face_parts(item["triangulation"])) == two_face_key(item["parts"])
