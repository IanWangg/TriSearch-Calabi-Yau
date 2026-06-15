import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cytools.polytope")

from cytools.polytope import Polytope

from data.cy.pipeline import (
    find_nearest_frst_bfs,
    generate_and_save_cy_reflexive_dataset_incremental,
    generate_cy_reflexive_dataset,
    save_cy_dataset,
)
from data.cy.generate_dataset import _build_parser
from core.cy_data_utils import load_k3_records


def _collect_sample_summaries(dataset):
    rows = []
    for polytope in dataset["polytopes"]:
        poly_idx = int(polytope["polytope_index"])
        for sample in polytope["samples"]:
            rows.append(
                {
                    "polytope_index": poly_idx,
                    "sample_index": int(sample["sample_index"]),
                    "sample_seed": int(sample["sample_seed"]),
                    "distance": sample["distance_to_nearest_frst"],
                    "tri_signature": sample["generated_triangulation"]["signature"],
                    "nearest_frst_signature": sample["nearest_frst"]["signature"],
                    "source_frst_index": int(sample["source_frst_index"]),
                    "heights": sample["heights"],
                }
            )
    return rows


def test_find_nearest_frst_distance_zero_for_known_frst():
    record = load_k3_records("cy_data/k3.txt", max_polytopes=1)[0]
    m_polytope = Polytope(np.asarray(record.m_vertices, dtype=np.int64))
    n_polytope = m_polytope.dual_polytope()

    # This construction is known to be FRST for the sample shown in cytools_api.ipynb.
    triangulation = n_polytope.triangulate(
        include_points_interior_to_facets=True,
        verbosity=0,
    )

    result = find_nearest_frst_bfs(
        triangulation,
        max_depth=3,
        max_nodes=40,
        neighbor_backend="qhull",
    )
    assert result["found"] is True
    assert result["distance"] == 0
    assert result["nearest_frst"] is not None
    assert result["nearest_frst"]["is_frst"] is True


def test_generate_cy_reflexive_dataset_structure_and_save(tmp_path):
    dataset = generate_cy_reflexive_dataset(
        num_polytopes=2,
        num_triangulations_per_frst=3,
        seed=11,
        k3_path="cy_data/k3.txt",
        triangulation_backend="qhull",
        neighbor_backend="qhull",
        include_points_interior_to_facets=True,
        make_star=True,
        frsts_per_polytope=2,
        fair_backend="cgal",
        fair_backend_fallback="qhull",
        fair_max_retries=10,
        fair_max_attempt_rounds=2,
        max_depth=2,
        max_node=80,
        triangulation_verbosity=0,
        num_workers=1,
    )

    assert dataset["metadata"]["num_polytopes"] == 2
    assert dataset["metadata"]["bfs_max_depth"] == 2
    assert dataset["metadata"]["bfs_max_nodes"] == 80
    assert dataset["metadata"]["num_workers"] == 1
    assert len(dataset["polytopes"]) == 2
    assert dataset["metadata"]["num_samples"] == sum(
        len(polytope["samples"]) for polytope in dataset["polytopes"]
    )

    for polytope in dataset["polytopes"]:
        assert polytope["lattice_space"] == "N"
        assert len(polytope["frst_seeds"]) >= 1
        assert polytope["frst_generation"]["diagnostics"]["obtained_frst_count"] >= 1
        assert len(polytope["samples"]) >= 1
        frst_signatures = {tuple(tuple(simplex) for simplex in frst["signature"]) for frst in polytope["frst_seeds"]}
        for sample in polytope["samples"]:
            assert sample["heights"] is None
            assert sample["height_generation_method"] == "not_available_from_random_triangulations_fair"
            assert sample["generated_triangulation"]["is_regular"] is True
            assert sample["generated_triangulation"]["is_fine"] is False

            distance = sample["distance_to_nearest_frst"]
            nearest_frst = sample["nearest_frst"]
            assert isinstance(distance, int)
            assert distance >= 1
            assert nearest_frst is not None
            assert nearest_frst["is_regular"] is True
            assert nearest_frst["is_fine"] is True
            nearest_signature = tuple(tuple(simplex) for simplex in nearest_frst["signature"])
            assert nearest_signature in frst_signatures

    paths = save_cy_dataset(
        dataset,
        output_dir=str(tmp_path),
        output_name="test_dataset",
    )
    dataset_path = paths["dataset_json"]
    samples_path = paths["samples_jsonl"]
    assert tmp_path.joinpath("test_dataset.json").exists()
    assert tmp_path.joinpath("test_dataset.samples.jsonl").exists()
    assert dataset_path.endswith("test_dataset.json")
    assert samples_path.endswith("test_dataset.samples.jsonl")

    with open(dataset_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["metadata"]["num_samples"] == dataset["metadata"]["num_samples"]
    assert len(loaded["polytopes"]) == len(dataset["polytopes"])
    assert "vertices" in loaded["polytopes"][0]
    assert "frst_list" in loaded["polytopes"][0]
    assert "polytope_description" not in loaded["polytopes"][0]

    with open(samples_path, "r", encoding="utf-8") as handle:
        jsonl_rows = [json.loads(line) for line in handle if line.strip()]
    assert len(jsonl_rows) == len(dataset["polytopes"])
    for row in jsonl_rows:
        assert "vertices" in row
        assert "polytope_description" not in row
        assert "frst_list" in row
        tri_count = 0
        for frst in row["frst_list"]:
            assert "simplices" in frst
            assert "triangulation" not in frst
            assert "triangulation_list" in frst
            tri_count += len(frst["triangulation_list"])
            for tri in frst["triangulation_list"]:
                assert "distance" in tri
                assert "simplices" in tri
                assert "triangulation" not in tri
        assert row["non_fine_triangulation_count"] == tri_count


def test_generate_dataset_larger_sample_reproducible_seed():
    # Larger integration sample to stress-test the full pipeline on more generated triangulations.
    dataset_a = generate_cy_reflexive_dataset(
        num_polytopes=4,
        num_triangulations_per_frst=4,
        seed=2026,
        k3_path="cy_data/k3.txt",
        triangulation_backend="qhull",
        neighbor_backend="qhull",
        include_points_interior_to_facets=True,
        make_star=True,
        frsts_per_polytope=1,
        fair_backend="cgal",
        fair_backend_fallback="qhull",
        fair_max_retries=8,
        fair_max_attempt_rounds=2,
        bfs_max_depth=1,
        bfs_max_nodes=40,
        triangulation_verbosity=0,
        num_workers=1,
    )
    dataset_b = generate_cy_reflexive_dataset(
        num_polytopes=4,
        num_triangulations_per_frst=4,
        seed=2026,
        k3_path="cy_data/k3.txt",
        triangulation_backend="qhull",
        neighbor_backend="qhull",
        include_points_interior_to_facets=True,
        make_star=True,
        frsts_per_polytope=1,
        fair_backend="cgal",
        fair_backend_fallback="qhull",
        fair_max_retries=8,
        fair_max_attempt_rounds=2,
        bfs_max_depth=1,
        bfs_max_nodes=40,
        triangulation_verbosity=0,
        num_workers=1,
    )

    assert dataset_a["metadata"]["num_samples"] == dataset_b["metadata"]["num_samples"]

    samples_a = _collect_sample_summaries(dataset_a)
    samples_b = _collect_sample_summaries(dataset_b)
    assert samples_a == samples_b


def test_generate_dataset_cli_aliases_for_bfs_limits():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--num-polytopes",
            "1",
            "--triangulations-per-polytope",
            "1",
            "--max-depth",
            "7",
            "--max-node",
            "123",
            "--collection-depths",
            "1",
            "2",
            "--collection_all",
            "--num-frsts-per-polytope",
            "3",
            "--resume",
            "--log-every",
            "7",
            "--random_flip",
            "--fast",
        ]
    )

    assert args.bfs_max_depth == 7
    assert args.bfs_max_nodes == 123
    assert args.collection_depths == [1, 2]
    assert args.collection_all is True
    assert args.num_triangulations_per_frst == 1
    assert args.frsts_per_polytope == 3
    assert args.resume is True
    assert args.log_every == 7
    assert args.random_flip is True
    assert args.fast is True


def test_generate_dataset_random_flip_ignores_collection_depths(monkeypatch):
    captured_jobs = []

    def _stub_load_k3_records(k3_path=None, max_polytopes=1):
        return [SimpleNamespace(record_index=i) for i in range(max_polytopes)]

    def _stub_process_polytope_collection_job(job):
        captured_jobs.append(job)
        return {
            "polytope_index": int(job["record"].record_index),
            "polytope_entry": {
                "polytope_index": int(job["record"].record_index),
                "samples": [],
            },
            "num_samples": 0,
            "frst_found_count": 0,
            "truncated_search_count": 0,
        }

    monkeypatch.setattr("data.cy.pipeline.load_k3_records", _stub_load_k3_records)
    monkeypatch.setattr(
        "data.cy.pipeline._process_polytope_collection_job",
        _stub_process_polytope_collection_job,
    )

    dataset = generate_cy_reflexive_dataset(
        num_polytopes=1,
        num_triangulations_per_frst=2,
        seed=1,
        bfs_max_depth=2,
        bfs_max_nodes=50,
        collection_depths=[5],
        collection_all=True,
        random_flip=True,
        fast=True,
        num_workers=1,
    )

    assert len(captured_jobs) == 1
    job = captured_jobs[0]
    assert job["random_flip"] is True
    assert job["fast"] is True
    assert job["collection_depths"] is None
    assert job["max_depth"] == 2

    assert dataset["metadata"]["random_flip"] is True
    assert dataset["metadata"]["fast"] is True
    assert dataset["metadata"]["frst_seed_sampler"] == "fast"
    assert dataset["metadata"]["generation_mode"] == "frst_random_flip_collection"
    assert dataset["metadata"]["collection_depths"] is None
    assert dataset["metadata"]["effective_bfs_max_depth"] == 2


def test_generate_dataset_depth_collection_all_non_frst():
    dataset = generate_cy_reflexive_dataset(
        num_polytopes=1,
        num_triangulations_per_frst=2,
        seed=123,
        k3_path="cy_data/k3.txt",
        triangulation_backend="qhull",
        neighbor_backend="qhull",
        include_points_interior_to_facets=True,
        make_star=True,
        frsts_per_polytope=1,
        fair_backend="cgal",
        fair_backend_fallback="qhull",
        fair_max_retries=8,
        fair_max_attempt_rounds=2,
        bfs_max_depth=1,
        bfs_max_nodes=80,
        collection_depths=[1],
        collection_all=True,
        triangulation_verbosity=0,
        num_workers=1,
    )

    assert dataset["metadata"]["collection_depths"] == [1]
    assert dataset["metadata"]["collection_all"] is True
    assert dataset["metadata"]["effective_bfs_max_depth"] >= 1
    samples = dataset["polytopes"][0]["samples"]
    assert len(samples) >= 1
    for sample in samples:
        assert sample["distance_to_nearest_frst"] == 1
        assert sample["generated_triangulation"]["is_fine"] is False


def test_incremental_generation_writes_checkpoint_and_jsonl(monkeypatch, tmp_path):
    def _stub_load_k3_records(k3_path=None, max_polytopes=1):
        return [SimpleNamespace(record_index=i) for i in range(max_polytopes)]

    def _stub_process_polytope_collection_job(job):
        polytope_index = int(job["record"].record_index)
        simplices = [[0, 1, 2, 3]]
        point_indices = [0, 1, 2, 3]
        polytope_entry = {
            "polytope_index": polytope_index,
            "n_points": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "frst_seeds": [
                {
                    "point_indices": point_indices,
                    "simplices": simplices,
                }
            ],
            "samples": [
                {
                    "source_frst_index": 0,
                    "distance_to_nearest_frst": 1,
                    "generated_triangulation": {
                        "point_indices": point_indices,
                        "simplices": simplices,
                    },
                },
                {
                    "source_frst_index": 0,
                    "distance_to_nearest_frst": 2,
                    "generated_triangulation": {
                        "point_indices": point_indices,
                        "simplices": simplices,
                    },
                },
            ],
        }
        return {
            "polytope_index": polytope_index,
            "polytope_entry": polytope_entry,
            "num_samples": 2,
            "frst_found_count": 2,
            "truncated_search_count": polytope_index % 2,
        }

    monkeypatch.setattr("data.cy.pipeline.load_k3_records", _stub_load_k3_records)
    monkeypatch.setattr(
        "data.cy.pipeline._process_polytope_collection_job",
        _stub_process_polytope_collection_job,
    )

    result = generate_and_save_cy_reflexive_dataset_incremental(
        num_polytopes=3,
        num_triangulations_per_frst=2,
        num_workers=1,
        compact_output=True,
        output_dir=str(tmp_path),
        output_name="incremental_unit",
        log_every=1,
    )

    samples_path = tmp_path / "incremental_unit.samples.jsonl"
    checkpoint_path = tmp_path / "incremental_unit.checkpoint.json"

    assert result["summary"]["num_samples"] == 6
    assert result["summary"]["completed_polytope_count"] == 3
    assert samples_path.exists()
    assert checkpoint_path.exists()
    assert "dataset_json" not in result["paths"]

    with samples_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 3
    assert sorted(int(row["polytope_index"]) for row in rows) == [0, 1, 2]

    with checkpoint_path.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    progress = checkpoint["progress"]
    assert progress["completed_polytope_count"] == 3
    assert progress["completed_polytope_indices"] == [0, 1, 2]
    assert progress["num_samples"] == 6
    assert set(progress["truncated_by_polytope"]) == {"0", "1", "2"}


def test_incremental_resume_uses_existing_jsonl_without_duplicates(monkeypatch, tmp_path):
    def _stub_load_k3_records(k3_path=None, max_polytopes=1):
        return [SimpleNamespace(record_index=i) for i in range(max_polytopes)]

    def _stub_process_polytope_collection_job(job):
        polytope_index = int(job["record"].record_index)
        simplices = [[0, 1, 2, 3]]
        point_indices = [0, 1, 2, 3]
        polytope_entry = {
            "polytope_index": polytope_index,
            "n_points": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "frst_seeds": [
                {
                    "point_indices": point_indices,
                    "simplices": simplices,
                }
            ],
            "samples": [
                {
                    "source_frst_index": 0,
                    "distance_to_nearest_frst": 1,
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

    monkeypatch.setattr("data.cy.pipeline.load_k3_records", _stub_load_k3_records)
    monkeypatch.setattr(
        "data.cy.pipeline._process_polytope_collection_job",
        _stub_process_polytope_collection_job,
    )

    generate_and_save_cy_reflexive_dataset_incremental(
        num_polytopes=3,
        num_triangulations_per_frst=1,
        num_workers=1,
        compact_output=True,
        output_dir=str(tmp_path),
        output_name="incremental_resume",
        log_every=1,
    )

    samples_path = tmp_path / "incremental_resume.samples.jsonl"
    checkpoint_path = tmp_path / "incremental_resume.checkpoint.json"

    with checkpoint_path.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    checkpoint["progress"]["completed_polytope_count"] = 1
    checkpoint["progress"]["completed_polytope_indices"] = [0]
    checkpoint["progress"]["num_samples"] = 1
    checkpoint["progress"]["frst_found_count"] = 1
    checkpoint["progress"]["truncated_search_count"] = 0
    checkpoint["progress"]["truncated_by_polytope"] = {"0": 0}
    with checkpoint_path.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, indent=2)

    result = generate_and_save_cy_reflexive_dataset_incremental(
        num_polytopes=3,
        num_triangulations_per_frst=1,
        num_workers=1,
        compact_output=True,
        output_dir=str(tmp_path),
        output_name="incremental_resume",
        resume=True,
        log_every=1,
    )

    with samples_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 3
    assert result["summary"]["completed_polytope_count"] == 3
    assert result["summary"]["num_samples"] == 3


def test_generate_dataset_depth_collection_subsample():
    dataset = generate_cy_reflexive_dataset(
        num_polytopes=1,
        num_triangulations_per_frst=2,
        seed=1234,
        k3_path="cy_data/k3.txt",
        triangulation_backend="qhull",
        neighbor_backend="qhull",
        include_points_interior_to_facets=True,
        make_star=True,
        frsts_per_polytope=1,
        fair_backend="cgal",
        fair_backend_fallback="qhull",
        fair_max_retries=8,
        fair_max_attempt_rounds=2,
        bfs_max_depth=2,
        bfs_max_nodes=80,
        collection_depths=[1, 2],
        collection_all=False,
        triangulation_verbosity=0,
        num_workers=1,
    )

    poly_entry = dataset["polytopes"][0]
    sample_count = len(poly_entry["samples"])
    candidate_pool_size = int(poly_entry["depth_collection"]["candidate_pool_size"])
    expected_count = min(2, candidate_pool_size)
    assert sample_count == expected_count
    for sample in poly_entry["samples"]:
        assert sample["distance_to_nearest_frst"] in {1, 2}
        assert sample["generated_triangulation"]["is_fine"] is False


def test_save_cy_dataset_compact_output_only_jsonl(tmp_path):
    dataset = generate_cy_reflexive_dataset(
        num_polytopes=1,
        num_triangulations_per_frst=1,
        seed=17,
        k3_path="cy_data/k3.txt",
        triangulation_backend="qhull",
        neighbor_backend="qhull",
        include_points_interior_to_facets=True,
        make_star=True,
        frsts_per_polytope=1,
        fair_backend="cgal",
        fair_backend_fallback="qhull",
        fair_max_retries=8,
        fair_max_attempt_rounds=2,
        bfs_max_depth=1,
        bfs_max_nodes=40,
        triangulation_verbosity=0,
        num_workers=1,
    )

    paths = save_cy_dataset(
        dataset,
        output_dir=str(tmp_path),
        output_name="compact_only",
        compact_output=True,
    )

    assert "dataset_json" not in paths
    assert "samples_jsonl" in paths
    assert not tmp_path.joinpath("compact_only.json").exists()
    assert tmp_path.joinpath("compact_only.samples.jsonl").exists()

    with open(paths["samples_jsonl"], "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == len(dataset["polytopes"])
    if rows:
        assert "vertices" in rows[0]
        assert "frst_list" in rows[0]
