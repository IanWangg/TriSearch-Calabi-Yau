import json

import pytest

pytest.importorskip("cytools.polytope")

from data.cy.generate_4d_dataset import _build_parser
from data.cy.pipeline import generate_and_save_cy_4d_reflexive_dataset_incremental


class _FakeFetchedPolytope:
    def __init__(self, vertices):
        self._vertices = vertices

    def vertices(self):
        return self._vertices


def test_generate_4d_dataset_cli_parser():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--num-polytopes",
            "2",
            "--h11",
            "27",
            "--num-vertices",
            "9",
            "--favorable",
            "--triangulations-per-polytope",
            "3",
            "--max-depth",
            "5",
            "--max-node",
            "88",
            "--collection-depths",
            "2",
            "4",
            "--collection_all",
            "--num-frsts-per-polytope",
            "4",
            "--resume",
            "--log-every",
            "6",
            "--random_flip",
            "--fast",
        ]
    )

    assert args.num_polytopes == 2
    assert args.h11 == 27
    assert args.num_vertices == 9
    assert args.favorable is True
    assert args.num_triangulations_per_frst == 3
    assert args.bfs_max_depth == 5
    assert args.bfs_max_nodes == 88
    assert args.collection_depths == [2, 4]
    assert args.collection_all is True
    assert args.frsts_per_polytope == 4
    assert args.resume is True
    assert args.log_every == 6
    assert args.random_flip is True
    assert args.fast is True


def test_generate_4d_dataset_cli_parser_default_favorable_is_unset():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--num-polytopes",
            "1",
            "--h11",
            "2",
            "--k",
            "1",
        ]
    )

    assert args.favorable is None


def test_generate_4d_dataset_cli_parser_supports_polytope_file_without_fetch_args(tmp_path):
    parser = _build_parser()
    polytope_file = tmp_path / "saved_polytopes.jsonl"
    polytope_file.write_text(
        json.dumps(
            {
                "polytope_index": 0,
                "vertices": [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = parser.parse_args(
        [
            "--polytope-file",
            str(polytope_file),
            "--k",
            "1",
        ]
    )

    assert args.polytope_file == str(polytope_file)
    assert args.num_polytopes is None
    assert args.h11 is None


def test_generate_4d_dataset_incremental_writes_h11_rows(monkeypatch, tmp_path):
    captured_jobs = []

    def _stub_fetch_polytopes(**kwargs):
        assert kwargs["h11"] == 12
        assert kwargs["dim"] == 4
        assert kwargs["lattice"] == "N"
        assert kwargs["n_vertices"] == 6
        assert kwargs["favorable"] is True
        assert kwargs["limit"] == 2
        return [
            _FakeFetchedPolytope(
                [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
            ),
            _FakeFetchedPolytope(
                [[0, 0, 0, 0], [2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [-2, -2, -2, -2]]
            ),
        ]

    def _stub_process_fetched_polytope_collection_job(job):
        captured_jobs.append(job)
        polytope_spec = job["polytope_spec"]
        polytope_index = int(polytope_spec["polytope_index"])
        simplices = [[0, 1, 2, 3, 4]]
        point_indices = [0, 1, 2, 3, 4, 5]
        polytope_entry = {
            "polytope_index": polytope_index,
            "h11": int(polytope_spec["h11"]),
            "favorable": bool(polytope_spec["favorable"]),
            "n_points": polytope_spec["vertices"],
            "frst_seeds": [
                {
                    "point_indices": point_indices,
                    "simplices": simplices,
                }
            ],
            "samples": [
                {
                    "source_frst_index": 0,
                    "distance_to_nearest_frst": polytope_index + 1,
                    "generated_triangulation": {
                        "point_indices": point_indices,
                        "simplices": simplices,
                    },
                }
            ],
        }
        return {
            "polytope_index": polytope_index,
            "polytope_entry": polytope_entry,
            "num_samples": 1,
            "frst_found_count": 1,
            "truncated_search_count": 0,
        }

    monkeypatch.setattr("data.cy.pipeline.fetch_polytopes", _stub_fetch_polytopes)
    monkeypatch.setattr(
        "data.cy.pipeline._process_fetched_polytope_collection_job",
        _stub_process_fetched_polytope_collection_job,
    )

    result = generate_and_save_cy_4d_reflexive_dataset_incremental(
        num_polytopes=2,
        h11=12,
        num_vertices=6,
        favorable=True,
        num_triangulations_per_frst=1,
        num_workers=1,
        compact_output=True,
        output_dir=str(tmp_path),
        output_name="fourfold_unit",
        log_every=1,
    )

    assert len(captured_jobs) == 2
    assert captured_jobs[0]["polytope_spec"]["h11"] == 12
    assert captured_jobs[0]["polytope_spec"]["requested_num_vertices"] == 6
    assert result["metadata"]["dataset_dimension"] == 4
    assert result["metadata"]["polytope_source"] == "cytools.fetch_polytopes"
    assert result["metadata"]["h11"] == 12
    assert result["metadata"]["num_vertices"] == 6
    assert result["metadata"]["favorable"] is True
    assert result["summary"]["num_samples"] == 2

    samples_path = tmp_path / "fourfold_unit.samples.jsonl"
    checkpoint_path = tmp_path / "fourfold_unit.checkpoint.json"
    assert samples_path.exists()
    assert checkpoint_path.exists()

    with samples_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 2
    assert sorted(int(row["polytope_index"]) for row in rows) == [0, 1]
    assert all(int(row["h11"]) == 12 for row in rows)
    assert all(bool(row["favorable"]) is True for row in rows)
    assert all(int(row["non_fine_triangulation_count"]) == 1 for row in rows)


def test_generate_4d_dataset_incremental_uses_polytope_file(monkeypatch, tmp_path):
    polytope_file = tmp_path / "saved_polytopes.jsonl"
    polytope_rows = [
        {
            "polytope_index": 10,
            "h11": 9,
            "vertices": [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        },
        {
            "polytope_index": 11,
            "h11": 9,
            "vertices": [[0, 0, 0, 0], [2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]],
        },
    ]
    with polytope_file.open("w", encoding="utf-8") as handle:
        for row in polytope_rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    def _fail_fetch_polytopes(**_kwargs):
        raise AssertionError("fetch_polytopes should not be called when polytope_file is provided")

    def _stub_process_fetched_polytope_collection_job(job):
        polytope_spec = job["polytope_spec"]
        polytope_index = int(polytope_spec["polytope_index"])
        simplices = [[0, 1, 2, 3, 4]]
        point_indices = [0, 1, 2, 3, 4]
        return {
            "polytope_index": polytope_index,
            "polytope_entry": {
                "polytope_index": polytope_index,
                "h11": int(polytope_spec["h11"]),
                "n_points": polytope_spec["vertices"],
                "frst_seeds": [
                    {
                        "point_indices": point_indices,
                        "simplices": simplices,
                    }
                ],
                "samples": [],
            },
            "num_samples": 0,
            "frst_found_count": 0,
            "truncated_search_count": 0,
        }

    monkeypatch.setattr("data.cy.pipeline.fetch_polytopes", _fail_fetch_polytopes)
    monkeypatch.setattr(
        "data.cy.pipeline._process_fetched_polytope_collection_job",
        _stub_process_fetched_polytope_collection_job,
    )

    result = generate_and_save_cy_4d_reflexive_dataset_incremental(
        polytope_file=str(polytope_file),
        num_triangulations_per_frst=1,
        num_workers=1,
        compact_output=True,
        output_dir=str(tmp_path),
        output_name="fourfold_from_file",
        log_every=1,
    )

    assert result["metadata"]["polytope_source"] == "polytope_file"
    assert result["metadata"]["polytope_file"] == str(polytope_file)
    assert result["summary"]["num_polytopes"] == 2

    samples_path = tmp_path / "fourfold_from_file.samples.jsonl"
    with samples_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert sorted(int(row["polytope_index"]) for row in rows) == [10, 11]
    assert all(int(row["h11"]) == 9 for row in rows)
    assert all(int(row["non_fine_triangulation_count"]) == 0 for row in rows)


def test_generate_4d_dataset_incremental_warns_and_uses_available_polytopes(monkeypatch, tmp_path):
    def _stub_fetch_polytopes(**kwargs):
        assert kwargs["as_list"] is False
        return iter(
            [
                _FakeFetchedPolytope(
                    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                ),
                _FakeFetchedPolytope(
                    [[0, 0, 0, 0], [2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]
                ),
            ]
        )

    def _stub_process_fetched_polytope_collection_job(job):
        polytope_spec = job["polytope_spec"]
        polytope_index = int(polytope_spec["polytope_index"])
        simplices = [[0, 1, 2, 3, 4]]
        point_indices = [0, 1, 2, 3, 4]
        return {
            "polytope_index": polytope_index,
            "polytope_entry": {
                "polytope_index": polytope_index,
                "h11": int(polytope_spec["h11"]),
                "n_points": polytope_spec["vertices"],
                "frst_seeds": [
                    {
                        "point_indices": point_indices,
                        "simplices": simplices,
                    }
                ],
                "samples": [],
            },
            "num_samples": 0,
            "frst_found_count": 0,
            "truncated_search_count": 0,
        }

    monkeypatch.setattr("data.cy.pipeline.fetch_polytopes", _stub_fetch_polytopes)
    monkeypatch.setattr(
        "data.cy.pipeline._process_fetched_polytope_collection_job",
        _stub_process_fetched_polytope_collection_job,
    )

    with pytest.warns(UserWarning, match="returned only 2"):
        result = generate_and_save_cy_4d_reflexive_dataset_incremental(
            num_polytopes=5,
            h11=2,
            num_triangulations_per_frst=1,
            num_workers=1,
            compact_output=True,
            output_dir=str(tmp_path),
            output_name="fourfold_partial",
            log_every=1,
        )

    assert result["metadata"]["requested_num_polytopes"] == 5
    assert result["metadata"]["num_polytopes"] == 2
    assert result["summary"]["num_polytopes"] == 2

    samples_path = tmp_path / "fourfold_partial.samples.jsonl"
    with samples_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 2
