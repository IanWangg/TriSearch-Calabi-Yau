from dataclasses import replace
import json

import pytest

from eval.config import EvaluationSpec
from eval.data.loader import load_eval_polytopes
from eval.setup import load_eval_setup, prepare_eval_setup


VERTICES = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]


def source_spec():
    return {"polytope_index": 0, "h11": 1, "vertices": VERTICES,
            "source_metadata": {"source_h11": 101, "source_h12": 1}, "polytope_source": "hugging_face:test"}


def test_eval_loader_delegates_source_filters_and_requires_exact_distinct_count(monkeypatch, tmp_path):
    captured = {}

    def fake_loader(**kwargs):
        captured.update(kwargs)
        return [source_spec()], {"resolved_revision": "pinned_sha"}

    monkeypatch.setattr("data.cy.generate_4d_dataset_hugging_face.load_hugging_face_4d_n_lattice_polytope_specs", fake_loader)
    spec = EvaluationSpec(1, 1, 1, 4, hf_cache_dir=str(tmp_path / "cache"), favorable=True, num_vertices=5)
    rows, metadata = load_eval_polytopes(spec)
    assert captured["h11"] == 1 and captured["favorable"] is True and captured["num_vertices"] == 5
    assert rows[0]["source_metadata"]["source_h12"] == 1
    assert metadata["resolved_revision"] == "pinned_sha"
    with pytest.raises(ValueError, match="Requested 2 polytopes"):
        load_eval_polytopes(replace(spec, num_polytopes=2))

    monkeypatch.setattr("data.cy.generate_4d_dataset_hugging_face.load_hugging_face_4d_n_lattice_polytope_specs",
                        lambda **kw: ([source_spec(), {**source_spec(), "polytope_index": 1}], {}))
    with pytest.raises(ValueError, match="duplicate polytopes"):
        load_eval_polytopes(replace(spec, num_polytopes=2))


def test_setup_uses_shared_sampler_serialization_and_reloads_offline(monkeypatch, tmp_path):
    from data.cy import pipeline

    monkeypatch.setattr("eval.setup.load_eval_polytopes", lambda spec: ([source_spec()], {"resolved_revision": "sha"}))
    captured = {}

    def sampler(polytope, **kwargs):
        captured.update(kwargs)
        return [polytope.triangulate(include_points_interior_to_facets=False, make_star=True)], {"obtained_frst_count": 1}

    monkeypatch.setattr(pipeline, "_generate_frst_seeds_with_random_triangulations_fair", sampler)
    spec = EvaluationSpec(1, 1, 1, 3)
    setup = prepare_eval_setup(spec, output_dir=tmp_path / "setup")
    assert captured["fast_only"] is True and captured["make_star"] is True
    assert captured["include_points_interior_to_facets"] is False
    row = setup.rows[0]
    polytope = pipeline.Polytope(row["vertices"])
    reconstructed = polytope.triangulate(simplices=row["frst_list"][0]["simplices"],
                                        include_points_interior_to_facets=False)
    assert reconstructed.is_fine() and reconstructed.is_star() and reconstructed.is_regular()

    def no_network(spec):
        raise AssertionError("Offline setup load must not access the source")

    monkeypatch.setattr("eval.setup.load_eval_polytopes", no_network)
    loaded = load_eval_setup(setup.path)
    assert loaded.rows == setup.rows and loaded.setup_id == setup.setup_id
    loaded.validate(replace(spec, objective_budget=20, algorithms=("best_first", "beam_search"),
                            beam_width=8, cache_states=False))
    with pytest.raises(ValueError, match="parameters"):
        loaded.validate(replace(spec, num_starts=2))
    manifest_path = setup.path / "samples.jsonl"
    changed = dict(row, h11=2)
    manifest_path.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_eval_setup(setup.path)


def test_local_polytope_setup_validates_geometry_and_reloads_without_source(monkeypatch, tmp_path):
    import hashlib

    def no_network(**kwargs):
        raise AssertionError("Local inputs must not access Hugging Face")

    monkeypatch.setattr("data.cy.generate_4d_dataset_hugging_face.load_hugging_face_4d_n_lattice_polytope_specs",
                        no_network)
    path = tmp_path / "polytope.json"
    path.write_text(json.dumps({"polytopes": [{"vertices": VERTICES, "h11": 1}]}))
    spec = EvaluationSpec(1, 1, 1, 3, polytope_file=str(path), num_vertices=5)
    rows, metadata = load_eval_polytopes(spec)
    assert metadata["source"] == "polytope_file" and metadata["selection"] == "first_n_in_file_order"
    assert metadata["polytope_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert rows[0]["h11"] == 1 and rows[0]["source_metadata"]["cytools_h21"] == 101
    setup = prepare_eval_setup(spec, output_dir=tmp_path / "local_setup")
    assert len(setup.rows[0]["frst_list"]) == 1
    path.unlink()
    loaded = load_eval_setup(setup.path)
    loaded.validate(replace(spec, objective_budget=20))
    assert loaded.setup_id == setup.setup_id
    with pytest.raises(ValueError, match="parameters"):
        loaded.validate(replace(spec, polytope_file=str(tmp_path / "other.json")))
    assert "polytope_file" not in replace(spec, polytope_file=None).setup_parameters()


@pytest.mark.parametrize("changes,match", [({"h11": 2}, "CY h11=1"),
                                        ({"num_vertices": 6}, "5 vertices"),
                                        ({"favorable": False}, "favorable")])
def test_local_polytope_rejects_mismatched_geometry(tmp_path, changes, match):
    path = tmp_path / "polytope.jsonl"
    path.write_text(json.dumps({"vertices": VERTICES}) + "\n")
    spec = EvaluationSpec(1, 1, 1, 3, polytope_file=str(path))
    with pytest.raises(ValueError, match=match):
        load_eval_polytopes(replace(spec, **changes))


def test_local_polytope_rejects_false_hodge_metadata_duplicates_and_replacement(tmp_path):
    path = tmp_path / "polytope.json"
    spec = EvaluationSpec(1, 1, 1, 3, polytope_file=str(path))
    path.write_text(json.dumps([{"vertices": VERTICES, "h11": 2}]))
    with pytest.raises(ValueError, match="file h11=2"):
        load_eval_polytopes(spec)
    path.write_text(json.dumps([{"vertices": VERTICES}, {"vertices": VERTICES}]))
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_polytopes(replace(spec, num_polytopes=2))
    with pytest.raises(ValueError, match="supplied polytopes must be retained"):
        replace(spec, skip_insufficient_starts=True)


@pytest.mark.parametrize("duplicate", [False, True])
def test_setup_refuses_insufficient_or_duplicate_starts(monkeypatch, tmp_path, duplicate):
    from data.cy import pipeline

    monkeypatch.setattr("eval.setup.load_eval_polytopes", lambda spec: ([source_spec()], {}))

    def sampler(polytope, **kwargs):
        start = polytope.triangulate(include_points_interior_to_facets=False, make_star=True)
        return [start] * (2 if duplicate else 1), {}

    monkeypatch.setattr(pipeline, "_generate_frst_seeds_with_random_triangulations_fair", sampler)
    with pytest.raises((ValueError, RuntimeError), match="(distinct FRST|Requested 2)"):
        prepare_eval_setup(EvaluationSpec(1, 1, 2, 5), output_dir=tmp_path / "setup")
    assert not (tmp_path / "setup" / "setup.json").exists()


@pytest.mark.parametrize("field,value", [("num_polytopes", 0), ("num_starts", 0), ("objective_budget", -1),
                                         ("seed", -1), ("runtime_cache_gb", float("nan")),
                                         ("beam_width", 0), ("beam_width", -1), ("beam_width", 1.5)])
def test_invalid_spec_is_rejected_before_data_loading(field, value):
    with pytest.raises(ValueError):
        replace(EvaluationSpec(1, 12, 1, 3), **{field: value})


def _mock_ordered_candidates(monkeypatch):
    """Two distinct coordinate configurations with the same real quintic geometry."""
    from test_cy_4d_hugging_face_data_generation import _quintic_n_polytope_source_row

    first = _quintic_n_polytope_source_row()
    second = {**first, "vertices": [[-v[0], *v[1:]] for v in first["vertices"]]}
    monkeypatch.setattr("data.cy.generate_4d_dataset_hugging_face._resolve_hugging_face_snapshot",
                        lambda **kwargs: (["polytopes-4d-05-vertices.parquet"], "resolved_sha"))
    monkeypatch.setattr("data.cy.generate_4d_dataset_hugging_face._load_hugging_face_parquet_rows",
                        lambda **kwargs: iter([first, second]))


def test_skip_shortfalls_preserves_source_indices_seeds_and_full_audit(monkeypatch, tmp_path):
    from data.cy import pipeline
    from eval.config import derive_seed

    _mock_ordered_candidates(monkeypatch)
    seeds = []

    def sampler(polytope, **kwargs):
        seeds.append(kwargs["seed"])
        if len(seeds) == 1:
            return [], {"obtained_frst_count": 0, "fast_rounds": [{"status": "short_return"}]}
        return [polytope.triangulate(include_points_interior_to_facets=False, make_star=True)], {"obtained_frst_count": 1}

    monkeypatch.setattr(pipeline, "_generate_frst_seeds_with_random_triangulations_fair", sampler)
    spec = EvaluationSpec(1, 1, 1, 3, skip_insufficient_starts=True, hf_cache_dir=str(tmp_path / "cache"))
    setup = prepare_eval_setup(spec, output_dir=tmp_path / "setup")
    assert [row["polytope_index"] for row in setup.rows] == [1]
    assert seeds == [derive_seed(0, "frst", 0), derive_seed(0, "frst", 1)]
    source = setup.metadata["source_metadata"]
    assert source["selection"] == "first_n_with_enough_distinct_frst_in_source_order"
    assert source["candidate_count"] == 2 and source["rejected_candidate_rows"] == 1
    skipped = setup.metadata["skipped_polytopes"]
    assert len(skipped) == 1 and skipped[0]["reason"] == "insufficient_distinct_frst_starts"
    assert skipped[0]["polytope_spec"]["source_metadata"]["filtered_row_index"] == 0
    assert skipped[0]["seed"] == seeds[0] and skipped[0]["obtained_num_starts"] == 0
    assert skipped[0]["sampling_diagnostics"]["fast_rounds"] == [{"status": "short_return"}]
    loaded = load_eval_setup(setup.path)
    loaded.validate(replace(spec, objective_budget=1000))
    assert loaded.setup_id == setup.setup_id
    with pytest.raises(ValueError, match="parameters"):
        loaded.validate(replace(spec, skip_insufficient_starts=False))
    assert "skip_insufficient_starts" not in replace(spec, skip_insufficient_starts=False).setup_parameters()


def test_skip_shortfalls_does_not_hide_sampler_errors(monkeypatch, tmp_path):
    from data.cy import pipeline

    _mock_ordered_candidates(monkeypatch)
    calls = []

    def sampler(polytope, **kwargs):
        calls.append(kwargs["seed"])
        raise RuntimeError("unexpected sampler error")

    monkeypatch.setattr(pipeline, "_generate_frst_seeds_with_random_triangulations_fair", sampler)
    with pytest.raises(RuntimeError):
        prepare_eval_setup(EvaluationSpec(1, 1, 1, 3, skip_insufficient_starts=True,
                                         hf_cache_dir=str(tmp_path / "cache")), output_dir=tmp_path / "setup")
    assert len(calls) == 1
    assert not (tmp_path / "setup" / "setup.json").exists()


def test_skip_shortfalls_still_requires_exact_final_polytope_count(monkeypatch, tmp_path):
    from data.cy import pipeline

    _mock_ordered_candidates(monkeypatch)
    monkeypatch.setattr(pipeline, "_generate_frst_seeds_with_random_triangulations_fair", lambda *args, **kwargs: ([], {}))
    with pytest.raises(ValueError, match="No matching"):
        prepare_eval_setup(EvaluationSpec(1, 1, 1, 3, skip_insufficient_starts=True,
                                         hf_cache_dir=str(tmp_path / "cache")), output_dir=tmp_path / "setup")
    assert not (tmp_path / "setup" / "setup.json").exists()
