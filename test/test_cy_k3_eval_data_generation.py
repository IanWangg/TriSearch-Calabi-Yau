import json
from types import SimpleNamespace

import pytest

from data.cy.generate_k3_eval_dataset import _build_parser
from data.cy.pipeline import generate_and_save_cy_k3_random_height_eval_dataset


def test_generate_k3_eval_dataset_cli_parser():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--polytope-index",
            "7",
            "3",
            "--k3-path",
            "cy_data/k3.txt",
            "--num-tri",
            "4",
            "--max-tries",
            "25",
            "--compact-output",
            "--no-include-points-interior-to-facets",
            "--make-star",
            "--num-workers",
            "2",
        ]
    )

    assert args.polytope_indices == [7, 3]
    assert args.k3_path == "cy_data/k3.txt"
    assert args.num_triangulations == 4
    assert args.max_tries == 25
    assert args.compact_output is True
    assert args.include_points_interior_to_facets is False
    assert args.make_star is True
    assert args.num_workers == 2


def test_generate_k3_random_height_eval_dataset_selects_requested_indices(monkeypatch, tmp_path):
    captured_jobs = []

    def _stub_load_k3_records(k3_path=None, max_polytopes=None):
        assert k3_path == "custom_k3.txt"
        assert max_polytopes == 8
        return [
            SimpleNamespace(
                record_index=i,
                header=f"header-{i}",
                ambient_dim=3,
                m_vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            )
            for i in range(max_polytopes)
        ]

    def _stub_process(job):
        captured_jobs.append(job)
        record = job["record"]
        return {
            "polytope_index": int(record.record_index),
            "polytope_entry": {
                "polytope_index": int(record.record_index),
                "source_header": record.header,
                "n_points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                "non_fine_triangulations": [
                    {
                        "heights": [0.2, 0.4, 0.6],
                        "signature": [[0, 1, 2, 3]],
                    }
                ],
            },
            "num_samples": 1,
            "underfilled_polytope_count": int(record.record_index == 7),
        }

    monkeypatch.setattr("data.cy.pipeline.load_k3_records", _stub_load_k3_records)
    monkeypatch.setattr(
        "data.cy.pipeline._process_k3_polytope_random_height_eval_job",
        _stub_process,
    )

    result = generate_and_save_cy_k3_random_height_eval_dataset(
        polytope_indices=[7, 3, 7],
        seed=9,
        k3_path="custom_k3.txt",
        num_triangulations=2,
        max_tries=11,
        num_workers=1,
        output_dir=str(tmp_path),
        output_name="k3_eval_unit",
    )

    assert [int(job["record"].record_index) for job in captured_jobs] == [3, 7]
    assert captured_jobs[0]["num_triangulations"] == 2
    assert captured_jobs[0]["max_tries"] == 11

    assert result["metadata"]["polytope_source"] == "k3.txt"
    assert result["metadata"]["dataset_dimension"] == 3
    assert result["metadata"]["polytope_indices"] == [3, 7]
    assert result["metadata"]["k3_path"] == "custom_k3.txt"
    assert result["metadata"]["num_samples"] == 2
    assert result["summary"]["underfilled_polytope_count"] == 1

    dataset_path = tmp_path / "k3_eval_unit.json"
    samples_path = tmp_path / "k3_eval_unit.samples.jsonl"
    assert dataset_path.exists()
    assert samples_path.exists()

    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    assert len(dataset["polytopes"]) == 2
    assert dataset["polytopes"][0]["polytope_index"] == 3
    assert dataset["polytopes"][1]["polytope_index"] == 7
    assert dataset["polytopes"][0]["non_fine_triangulation_list"][0]["heights"] == [0.2, 0.4, 0.6]

    with samples_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert [int(row["polytope_index"]) for row in rows] == [3, 7]
    assert all(int(row["non_fine_triangulation_count"]) == 1 for row in rows)


def test_generate_k3_random_height_eval_dataset_rejects_missing_index(monkeypatch):
    def _stub_load_k3_records(k3_path=None, max_polytopes=None):
        assert max_polytopes == 5
        return [
            SimpleNamespace(
                record_index=i,
                header=f"header-{i}",
                ambient_dim=3,
                m_vertices=((0, 0, 0),),
            )
            for i in range(4)
        ]

    monkeypatch.setattr("data.cy.pipeline.load_k3_records", _stub_load_k3_records)

    with pytest.raises(ValueError, match="out of range"):
        generate_and_save_cy_k3_random_height_eval_dataset(
            polytope_indices=[4],
            num_triangulations=1,
            max_tries=5,
            num_workers=1,
        )
