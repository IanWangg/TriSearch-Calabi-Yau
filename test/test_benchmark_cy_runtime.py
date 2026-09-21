from dataclasses import asdict
import json

from mdp.cy_graph import CYGraphTransition, CYStateExpansion
from mdp.cy_state_record import CYPointConfiguration, CyStateRecord
from tools.benchmark_cy_runtime import _bounded_rows, _semantic_signature, _trace_requests, parse_args


def test_benchmark_signature_accepts_order_and_representation_changes_but_checks_geometry():
    source = ((0, 1, 2), (0, 2, 3))
    destination = ((0, 1, 3), (1, 2, 3))
    full = CYGraphTransition("destination", destination, False)
    delta = CYGraphTransition("destination", next_is_target=False, removed_simplices=source, added_simplices=destination)
    first = CYStateExpansion("source", 1, source, ((3, 4), (1, 2)), frozenset(), (((3, 4), full), ((1, 2), full)))
    second = CYStateExpansion("source", 1, source, ((1, 2), (3, 4)), frozenset(), (((1, 2), delta), ((3, 4), delta)))
    assert _semantic_signature(first) == _semantic_signature(second)
    wrong = CYStateExpansion("source", 1, source, ((1, 2), (3, 4)), frozenset({(7, 8)}), second.transitions)
    assert _semantic_signature(first) != _semantic_signature(wrong)


def test_trace_json_roundtrip_preserves_exact_requests():
    configuration = CYPointConfiguration(3, ((0, 0), (1, 0), (0, 1)), ((0, 0), (1, 0), (0, 1)), (0, 1, 2), True)
    state = CyStateRecord(configuration, frozenset({(0, 1, 2)}))
    trace = {"schema_version": 1, "configurations": [asdict(configuration)],
             "requests": [{"configuration_index": 3, "operation": "expand", "state": state.to_payload(), "objective_mode": True}]}
    request = _trace_requests(json.loads(json.dumps(trace)), 1)[0]
    assert request["configuration"] == configuration
    assert request["state"] == state.to_payload()


def test_trace_selection_limits_geometry_before_collection_loading():
    rows = [{"polytope_index": index, "vertices": [[0, 0]], "frst_list": [],
             "non_fine_triangulation_list": [{"simplices": [[0, 1, vertex]]} for vertex in range(2, 9)]}
            for index in range(3)]
    selected = _bounded_rows(rows, 5, "regular")
    assert [len(row["non_fine_triangulation_list"]) for row in selected] == [2, 2, 1]
    assert sum(len(row["frst_list"]) for row in selected) == 0


def test_benchmark_neighbor_mode_defaults_and_worker_deduplication():
    args = parse_args(["--neighbor_mode", "two_neighbors", "--workers", "1", "2", "1"])
    assert args.include_points_interior_to_facets is False
    assert args.workers == [1, 2]
