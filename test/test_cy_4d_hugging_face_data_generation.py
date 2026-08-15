import json

import pytest

pytest.importorskip("cytools.polytope")

from data.cy.generate_4d_dataset import _build_parser as _build_original_parser
from data.cy.generate_4d_dataset_hugging_face import (
    HUGGING_FACE_DATASET_REPO,
    _build_parser,
    _load_hugging_face_parquet_rows,
    _resolve_hugging_face_snapshot,
    _select_parquet_files,
    load_hugging_face_4d_n_lattice_polytope_specs,
)
from data.cy.pipeline import generate_and_save_cy_4d_reflexive_dataset_incremental
from mdp.cy_rollout import load_cy_sample_rows


def _quintic_n_polytope_source_row():
    return {
        "vertices": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, -1, -1, -1],
        ],
        "vertex_count": 5,
        "facet_count": 5,
        "point_count": 6,
        "dual_point_count": 126,
        "h11": 101,
        "h12": 1,
        "euler_characteristic": 200,
    }


def test_hugging_face_cli_preserves_original_generator_api():
    original_actions = {
        action.dest: action
        for action in _build_original_parser()._actions
        if action.dest != "help"
    }
    hugging_face_actions = {
        action.dest: action
        for action in _build_parser()._actions
        if action.dest != "help"
    }

    assert set(original_actions).issubset(hugging_face_actions)
    for dest, original_action in original_actions.items():
        hugging_face_action = hugging_face_actions[dest]
        assert hugging_face_action.option_strings == original_action.option_strings
        assert hugging_face_action.required == original_action.required
        assert hugging_face_action.nargs == original_action.nargs
        assert hugging_face_action.default == original_action.default

    args = _build_parser().parse_args(
        [
            "--num-polytopes",
            "2",
            "--h11",
            "1",
            "--num-vertices",
            "5",
            "--favorable",
            "--k",
            "3",
            "--hf-revision",
            "test-revision",
        ]
    )
    assert args.num_polytopes == 2
    assert args.h11 == 1
    assert args.num_vertices == 5
    assert args.favorable is True
    assert args.num_triangulations_per_frst == 3
    assert args.hf_repo_id == HUGGING_FACE_DATASET_REPO
    assert args.hf_revision == "test-revision"


def test_select_parquet_files_uses_vertex_partition():
    filenames = [
        "README.md",
        "polytopes-4d-06-vertices.parquet",
        "polytopes-4d-05-vertices.parquet",
    ]
    assert _select_parquet_files(filenames, num_vertices=None) == [
        "polytopes-4d-05-vertices.parquet",
        "polytopes-4d-06-vertices.parquet",
    ]
    assert _select_parquet_files(filenames, num_vertices=6) == [
        "polytopes-4d-06-vertices.parquet"
    ]
    with pytest.raises(ValueError, match="no 4D polytope file with 7 vertices"):
        _select_parquet_files(filenames, num_vertices=7)


def test_pinned_canonical_revision_uses_static_partition_manifest(monkeypatch):
    def _fail_dataset_info(**_kwargs):
        raise AssertionError("A pinned canonical revision must not query dataset metadata")

    monkeypatch.setattr("huggingface_hub.HfApi.dataset_info", _fail_dataset_info)
    revision = "60c0e119a03608418df538191f65da3f43b5b819"
    filenames, resolved_revision = _resolve_hugging_face_snapshot(
        repo_id=HUGGING_FACE_DATASET_REPO,
        revision=revision,
    )

    assert resolved_revision == revision
    assert filenames[0] == "polytopes-4d-05-vertices.parquet"
    assert filenames[-1] == "polytopes-4d-36-vertices.parquet"


def test_parquet_stream_pushes_down_n_lattice_h11_filter(monkeypatch):
    captured = {}

    def _fake_hf_hub_url(**kwargs):
        captured["url_kwargs"] = kwargs
        return "https://example.invalid/source.parquet"

    def _fake_load_dataset(builder_name, **kwargs):
        captured["builder_name"] = builder_name
        captured["load_kwargs"] = kwargs
        return iter([_quintic_n_polytope_source_row()])

    monkeypatch.setattr("huggingface_hub.hf_hub_url", _fake_hf_hub_url)
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    rows = list(
        _load_hugging_face_parquet_rows(
            repo_id=HUGGING_FACE_DATASET_REPO,
            revision="resolved-sha",
            filename="polytopes-4d-05-vertices.parquet",
            h11=1,
            cache_dir=None,
        )
    )

    assert len(rows) == 1
    assert captured["builder_name"] == "parquet"
    assert captured["load_kwargs"]["streaming"] is True
    assert captured["load_kwargs"]["filters"] == [("h12", "=", 1)]
    assert captured["url_kwargs"]["revision"] == "resolved-sha"


def test_hugging_face_loader_maps_source_h12_to_n_lattice_h11(monkeypatch):
    captured = {}

    def _fake_resolve_hugging_face_snapshot(**kwargs):
        captured["snapshot_kwargs"] = kwargs
        return ["polytopes-4d-05-vertices.parquet"], "resolved-sha"

    def _fake_load_hugging_face_parquet_rows(**kwargs):
        captured["load_kwargs"] = kwargs
        return iter([_quintic_n_polytope_source_row()])

    monkeypatch.setattr(
        "data.cy.generate_4d_dataset_hugging_face._resolve_hugging_face_snapshot",
        _fake_resolve_hugging_face_snapshot,
    )
    monkeypatch.setattr(
        "data.cy.generate_4d_dataset_hugging_face._load_hugging_face_parquet_rows",
        _fake_load_hugging_face_parquet_rows,
    )

    specs, metadata = load_hugging_face_4d_n_lattice_polytope_specs(
        num_polytopes=1,
        h11=1,
        num_vertices=5,
        favorable=True,
        revision="main",
    )

    assert len(specs) == 1
    assert specs[0]["h11"] == 1
    assert specs[0]["favorable"] is True
    assert specs[0]["vertices"] == _quintic_n_polytope_source_row()["vertices"]
    assert specs[0]["source_metadata"]["source_h11"] == 101
    assert specs[0]["source_metadata"]["source_h12"] == 1
    assert metadata["source_lattice_interpretation"] == "N"
    assert metadata["source_hodge_filter"] == {"column": "h12", "value": 1}
    assert metadata["resolved_revision"] == "resolved-sha"
    assert captured["load_kwargs"]["revision"] == "resolved-sha"


def test_hugging_face_specs_use_training_compatible_incremental_output(
    monkeypatch,
    tmp_path,
):
    source_row = _quintic_n_polytope_source_row()
    polytope_spec = {
        "polytope_index": 0,
        "h11": 1,
        "favorable": True,
        "requested_num_vertices": 5,
        "vertices": source_row["vertices"],
        "polytope_source": f"hugging_face:{HUGGING_FACE_DATASET_REPO}",
        "source_metadata": {"source_h11": 101, "source_h12": 1},
    }

    def _fail_fetch_polytopes(**_kwargs):
        raise AssertionError("fetch_polytopes must not be used for provided Hugging Face specs")

    def _stub_process_job(job):
        spec = job["polytope_spec"]
        n_points = [[0, 0, 0, 0], *spec["vertices"]]
        simplices = [[0, 1, 2, 3, 4]]
        point_indices = list(range(len(n_points)))
        return {
            "polytope_index": 0,
            "polytope_entry": {
                "polytope_index": 0,
                "h11": 1,
                "favorable": True,
                "n_points": n_points,
                "frst_seeds": [
                    {"point_indices": point_indices, "simplices": simplices}
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
            },
            "num_samples": 1,
            "frst_found_count": 1,
            "truncated_search_count": 0,
        }

    monkeypatch.setattr("data.cy.pipeline.fetch_polytopes", _fail_fetch_polytopes)
    monkeypatch.setattr(
        "data.cy.pipeline._process_fetched_polytope_collection_job",
        _stub_process_job,
    )

    result = generate_and_save_cy_4d_reflexive_dataset_incremental(
        num_polytopes=1,
        h11=1,
        num_vertices=5,
        favorable=True,
        num_triangulations_per_frst=1,
        polytope_specs=[polytope_spec],
        polytope_source_metadata={
            "polytope_source": f"hugging_face:{HUGGING_FACE_DATASET_REPO}",
            "source_hodge_filter": {"column": "h12", "value": 1},
        },
        num_workers=1,
        compact_output=True,
        output_dir=str(tmp_path),
        output_name="hugging_face_unit",
        log_every=1,
    )

    assert result["metadata"]["polytope_source"] == (
        f"hugging_face:{HUGGING_FACE_DATASET_REPO}"
    )
    assert result["metadata"]["polytope_source_metadata"]["source_hodge_filter"] == {
        "column": "h12",
        "value": 1,
    }
    rows = load_cy_sample_rows(result["paths"]["samples_jsonl"])
    assert len(rows) == 1
    assert rows[0]["h11"] == 1
    assert rows[0]["favorable"] is True
    assert rows[0]["non_fine_triangulation_count"] == 1
    assert rows[0]["frst_list"][0]["triangulation_list"][0]["distance"] == 1

    with open(result["paths"]["samples_jsonl"], "r", encoding="utf-8") as handle:
        assert json.loads(handle.readline()) == rows[0]
