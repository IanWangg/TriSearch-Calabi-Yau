import json

from data.cy.generate_eval_dataset import _build_parser
from data.cy.pipeline import (
    _generate_random_height_non_fine_triangulations,
    generate_and_save_cy_4d_random_height_eval_dataset,
)


class _FakeEvalTriangulation:
    def __init__(self, signature, *, point_indices=None, is_regular=True, is_fine=False):
        self._signature = signature
        self._point_indices = (
            list(point_indices)
            if point_indices is not None
            else list(range(max(max(simplex) for simplex in signature) + 1))
        )
        self._is_regular = is_regular
        self._is_fine = is_fine

    def simplices(self, as_indices=True):
        assert as_indices is True
        return self._signature

    def points(self, as_poly_indices=False):
        assert as_poly_indices is True
        return self._point_indices

    def is_regular(self):
        return self._is_regular

    def is_fine(self):
        return self._is_fine


class _FakeRandomHeightPolytope:
    def __init__(self, outcomes, *, num_points):
        self._outcomes = list(outcomes)
        self._num_points = int(num_points)
        self.calls = []

    def points(self):
        return [[0, 0, 0, 0] for _ in range(self._num_points)]

    def triangulate(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeFetchedPolytope:
    def __init__(self, vertices):
        self._vertices = vertices

    def vertices(self):
        return self._vertices


def test_generate_eval_dataset_cli_parser():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--num-polytopes",
            "2",
            "--h11",
            "19",
            "--num-vertices",
            "8",
            "--favorable",
            "--num-tri",
            "3",
            "--max-tries",
            "21",
            "--compact-output",
            "--no-include-points-interior-to-facets",
            "--make-star",
            "--num-workers",
            "4",
        ]
    )

    assert args.num_polytopes == 2
    assert args.h11 == 19
    assert args.num_vertices == 8
    assert args.favorable is True
    assert args.num_triangulations == 3
    assert args.max_tries == 21
    assert args.compact_output is True
    assert args.include_points_interior_to_facets is False
    assert args.make_star is True
    assert args.num_workers == 4


def test_generate_random_height_non_fine_triangulations_filters_and_deduplicates():
    fine = _FakeEvalTriangulation([[0, 1, 2, 3, 4]], is_fine=True)
    non_fine_a = _FakeEvalTriangulation(
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]],
        point_indices=[0, 1, 2, 3, 4, 5],
    )
    non_fine_a_duplicate = _FakeEvalTriangulation(
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]],
        point_indices=[0, 1, 2, 3, 4, 5],
    )
    non_fine_b = _FakeEvalTriangulation(
        [[0, 1, 2, 4, 5]],
        point_indices=[0, 1, 2, 3, 4, 5],
    )
    polytope = _FakeRandomHeightPolytope(
        [
            fine,
            non_fine_a,
            non_fine_a_duplicate,
            RuntimeError("boom"),
            non_fine_b,
        ],
        num_points=6,
    )

    result = _generate_random_height_non_fine_triangulations(
        polytope,
        num_triangulations=2,
        max_tries=5,
        seed=7,
        triangulation_backend="qhull",
        include_points_interior_to_facets=True,
        make_star=False,
        triangulation_verbosity=3,
    )

    assert len(result["triangulations"]) == 2
    assert result["diagnostics"]["attempt_count"] == 5
    assert result["diagnostics"]["generated_count"] == 2
    assert result["diagnostics"]["fine_discard_count"] == 1
    assert result["diagnostics"]["duplicate_count"] == 1
    assert result["diagnostics"]["error_count"] == 1
    assert result["diagnostics"]["last_error"] == "RuntimeError"
    assert result["diagnostics"]["target_reached"] is True

    for call in polytope.calls:
        assert call["check_heights"] is True
        assert call["backend"] == "qhull"
        assert call["include_points_interior_to_facets"] is True
        assert call["make_star"] is False
        assert call["verbosity"] == 3
        assert len(call["heights"]) == 6

    signatures = [item["signature"] for item in result["triangulations"]]
    assert signatures == [
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]],
        [[0, 1, 2, 4, 5]],
    ]
    assert all(len(item["heights"]) == 6 for item in result["triangulations"])


def test_generate_and_save_4d_random_height_eval_dataset(monkeypatch, tmp_path):
    captured_jobs = []

    def _stub_fetch_polytopes(**kwargs):
        assert kwargs["h11"] == 12
        assert kwargs["dim"] == 4
        assert kwargs["lattice"] == "N"
        assert kwargs["n_vertices"] == 6
        assert kwargs["favorable"] is False
        assert kwargs["limit"] == 2
        return [
            _FakeFetchedPolytope(
                [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
            ),
            _FakeFetchedPolytope(
                [[0, 0, 0, 0], [2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [-2, -2, -2, -2]]
            ),
        ]

    def _stub_process(job):
        captured_jobs.append(job)
        polytope_spec = job["polytope_spec"]
        polytope_index = int(polytope_spec["polytope_index"])
        return {
            "polytope_index": polytope_index,
            "polytope_entry": {
                "polytope_index": polytope_index,
                "h11": int(polytope_spec["h11"]),
                "favorable": bool(polytope_spec["favorable"]),
                "n_points": polytope_spec["vertices"],
                "non_fine_triangulations": [
                    {
                        "heights": [0.1, 0.2, 0.3],
                        "signature": [[0, 1, 2, 3, 4]],
                    }
                ],
            },
            "num_samples": 1,
            "underfilled_polytope_count": int(polytope_index == 1),
        }

    monkeypatch.setattr("data.cy.pipeline.fetch_polytopes", _stub_fetch_polytopes)
    monkeypatch.setattr(
        "data.cy.pipeline._process_fetched_polytope_random_height_eval_job",
        _stub_process,
    )

    result = generate_and_save_cy_4d_random_height_eval_dataset(
        num_polytopes=2,
        h11=12,
        num_vertices=6,
        favorable=False,
        num_triangulations=3,
        max_tries=9,
        seed=5,
        num_workers=1,
        output_dir=str(tmp_path),
        output_name="eval_unit",
    )

    assert len(captured_jobs) == 2
    assert captured_jobs[0]["num_triangulations"] == 3
    assert captured_jobs[0]["max_tries"] == 9
    assert result["metadata"]["generation_mode"] == "random_height_non_fine_regular"
    assert result["metadata"]["num_samples"] == 2
    assert result["metadata"]["underfilled_polytope_count"] == 1
    assert result["summary"]["num_samples"] == 2
    assert result["summary"]["underfilled_polytope_count"] == 1

    dataset_path = tmp_path / "eval_unit.json"
    samples_path = tmp_path / "eval_unit.samples.jsonl"
    assert dataset_path.exists()
    assert samples_path.exists()

    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    assert dataset["metadata"]["h11"] == 12
    assert dataset["metadata"]["num_triangulations_per_polytope"] == 3
    assert len(dataset["polytopes"]) == 2
    assert dataset["polytopes"][0]["non_fine_triangulation_list"][0]["signature"] == [[0, 1, 2, 3, 4]]

    with samples_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 2
    assert rows[0]["non_fine_triangulation_count"] == 1
    assert rows[0]["non_fine_triangulation_list"][0]["heights"] == [0.1, 0.2, 0.3]
