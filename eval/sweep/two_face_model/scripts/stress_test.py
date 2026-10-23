"""Reproducible CPU stress checks; write disposable artifacts under eval/smoke_tests."""

import argparse
import copy
from dataclasses import replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import weakref

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch_geometric.data import Batch

from core.cy_data_utils import configure_cy_data_tensor_caches, get_cy_data_tensor_cache_stats
from core.cy_policy_inference import (
    _forward_policy_data, batched_policy_action_selection, configure_policy_execution,
    evaluate_policy_actions_from_data_list,
)
from core.cy_ppo import PPORolloutBuffer, train_policy_from_rollout
from core.cy_two_face_data import build_two_face_data, create_two_face_data_from_state
from mdp.cy_state_record import CYPointConfiguration, CyStateRecord, canonical_simplices, two_face_restrictions
from models.two_face_agent import TwoFaceAgent


def rss_bytes():
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def grid_faces(num_faces, side):
    """Synthetic manifold square grids, not claimed to be reflexive polytopes."""
    points, labels, restrictions, actions = [], [], [], []
    for face in range(num_faces):
        local = {}
        for row in range(side):
            for col in range(side):
                label = 10 + 7 * len(labels)
                labels.append(label)
                local[row, col] = label
                points.append((row / side, col / side, face % 4 - 2, face // 4 - 4))
        triangles = []
        for row in range(side - 1):
            for col in range(side - 1):
                a, b = local[row, col], local[row + 1, col]
                c, d = local[row + 1, col + 1], local[row, col + 1]
                triangles.extend(((a, b, c), (a, c, d)))
                actions.append(tuple(sorted((a, b, c, d))))
        restrictions.append((tuple(sorted(local.values())), canonical_simplices(triangles)))
    configuration = CYPointConfiguration(7000, tuple(points), tuple(points), tuple(labels), False,
                                         tuple(face for face, _ in restrictions))
    return configuration, tuple(restrictions), tuple(actions)


def permute_data(data):
    result = data.clone()
    nodes = torch.randperm(data.num_nodes)
    inverse_nodes = torch.argsort(nodes)
    result.x, result.node_face = data.x[nodes], data.node_face[nodes]
    for name in ("edge_index", "triangle_vertices", "added_triangle_vertices"):
        result[name] = inverse_nodes[result[name]]
    faces = torch.randperm(data.num_faces)
    for name in ("node_face", "triangle_face", "action_face"):
        result[name] = faces[result[name]]
    triangles = torch.randperm(data.num_triangles)
    inverse_triangles = torch.argsort(triangles)
    result.triangle_vertices = result.triangle_vertices[triangles].flip(-1)
    result.triangle_face = result.triangle_face[triangles]
    for name in ("triangle_edge_index", "removed_triangle_ids"):
        result[name] = inverse_triangles[result[name]]
    actions = torch.randperm(data.num_available_subcomplexes)
    for name in ("action_face", "removed_triangle_ids", "added_triangle_vertices", "subcomplex_vertices"):
        result[name] = result[name][actions]
    return result, actions


def check_close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)


def stress_tensors(args, record):
    model = TwoFaceAgent(in_channels=4).eval()
    if args.checkpoint:
        from core.cy_runtime_utils import load_policy_checkpoint
        load_policy_checkpoint(model, args.checkpoint, map_location=torch.device("cpu"))
        record("loaded_checkpoint", path=args.checkpoint,
               sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
    configure_cy_data_tensor_caches(max_bytes=2**20, max_entries=8)
    templates = []
    for faces, side in ((1, 2), (4, 3), (8, 4), (16, 5)):
        configuration, restrictions, actions = grid_faces(faces, side)
        templates.extend(build_two_face_data(configuration, restrictions, subset)
                         for subset in (actions, actions[:1], ()))
    with torch.inference_mode():
        expected = [model.get_value_and_logits(Batch.from_data_list([data])) for data in templates]
        for size in (1, 8, 32, 128, 512):
            data = [templates[index % len(templates)] for index in range(size)]
            start = time.perf_counter()
            full_value, full_logits, *_ = _forward_policy_data(data, model, device=torch.device("cpu"),
                                                              max_graph_size=10**9)
            full_sec = time.perf_counter() - start
            start = time.perf_counter()
            chunk_value, chunk_logits, *_ = _forward_policy_data(data, model, device=torch.device("cpu"),
                                                                max_graph_size=20000)
            check_close(full_value, chunk_value)
            check_close(full_logits, chunk_logits)
            for index, item in enumerate(data):
                value, logits = expected[index % len(templates)]
                check_close(full_value[index:index + 1], value)
                check_close(full_logits[index, :item.num_available_subcomplexes], logits[0])
            record("batch", states=size, nodes=sum(d.num_nodes for d in data),
                   triangles=sum(d.num_triangles for d in data), actions=sum(d.num_available_subcomplexes for d in data),
                   full_sec=full_sec, chunk_sec=time.perf_counter() - start, rss_bytes=rss_bytes())
        for iteration in range(100):
            index = iteration % len(templates)
            data, order = permute_data(templates[index])
            value, logits = model.get_value_and_logits(Batch.from_data_list([data]))
            check_close(value, expected[index][0])
            check_close(logits, expected[index][1][:, order])
        record("permutations", cases=100)
    configuration, restrictions, actions = grid_faces(32, 16)
    large = build_two_face_data(configuration, restrictions, actions)
    model.train()
    value, logits = model.get_value_and_logits(Batch.from_data_list([large]))
    loss = value.square().mean() + logits.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    # EGNN returns final coordinates, which the policy does not consume. Its
    # last coordinate-update head is therefore intentionally outside this loss.
    unused = {name for name, parameter in model.named_parameters() if parameter.grad is None}
    assert unused == {f"point_egnn.gcl_2.coord_mlp.{name}" for name in ("0.weight", "0.bias", "2.weight")}, unused
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    record("large_backward", nodes=large.num_nodes, triangles=large.num_triangles,
           actions=large.num_available_subcomplexes, loss=float(loss.detach()), rss_bytes=rss_bytes(),
           unused_parameter_names=sorted(unused))
    del large, value, logits, loss
    model.zero_grad(set_to_none=True)
    model.eval()
    configuration, restrictions, actions = grid_faces(4, 3)
    reference = build_two_face_data(configuration, restrictions, actions)
    with torch.inference_mode():
        expected_value, expected_logits = model.get_value_and_logits(Batch.from_data_list([reference]))
        samples = []
        before = get_cy_data_tensor_cache_stats()
        for iteration in range(2000):
            # Distinct identities force bounded-cache churn without changing the observation.
            item = build_two_face_data(replace(configuration, index=iteration), restrictions, actions)
            if iteration % 10 == 0:
                value, logits = model.get_value_and_logits(Batch.from_data_list([item] * 16))
                check_close(value, expected_value.expand(16))
                check_close(logits, expected_logits.expand(16, -1))
                assert sum(c["bytes"] for c in get_cy_data_tensor_cache_stats().values()) <= 2**20
            if iteration % 100 == 0:
                gc.collect()
                samples.append(rss_bytes())
        after = get_cy_data_tensor_cache_stats()
    evictions = sum(after[name]["evictions"] - before[name]["evictions"] for name in after)
    assert evictions > 1000
    assert max(samples[3:]) - min(samples[3:]) < 128 * 2**20, samples
    configure_cy_data_tensor_caches(max_bytes=0)
    rebuilt = build_two_face_data(configuration, restrictions, actions)
    for key in reference.keys():
        if isinstance(reference[key], torch.Tensor):
            assert torch.equal(reference[key], rebuilt[key]), key
    record("cache_churn", observations=2000, repeated_inference_batches=200, evictions=evictions,
           rss_samples=samples, cache_stats=after)


def selected_rows(args):
    from mdp.cy_rollout import load_cy_sample_rows
    rows = load_cy_sample_rows(str(ROOT / "data/cy/two_neighbors_h11_12.samples.jsonl"))
    if args.extra_dataset:
        extra = load_cy_sample_rows(args.extra_dataset)
        extra = sorted(extra, key=lambda row: (-len(row["vertices"]), row["polytope_index"]))[:8]
        rows += [dict(row, polytope_index=100000 + row["polytope_index"]) for row in extra]
    return [dict(row, frst_list=[dict(row["frst_list"][0], triangulation_list=[])]) for row in rows]


def stress_ppo(args, record):
    model = TwoFaceAgent(in_channels=4)
    states, candidates = [], []
    for index in range(48):
        configuration, restrictions, actions = grid_faces(1 + index % 8, 2 + index % 4)
        triangles = frozenset(triangle for _, face in restrictions for triangle in face)
        states.append(CyStateRecord(configuration, triangles, "two_neighbors", True, True))
        candidates.append(actions if index % 5 else ())
    configure_policy_execution(max_graph_size=20000)
    buffer = PPORolloutBuffer()
    for iteration in range(6):
        selection = batched_policy_action_selection(states, candidates, model, device=torch.device("cpu"))
        step = SimpleNamespace(input_states=states, action_candidates=selection.action_lists,
            data_list=selection.data_list, actions_tensor=selection.actions_tensor,
            action_index_tensor=selection.action_index_tensor, log_prob_tensor=selection.log_prob_tensor,
            entropy_tensor=selection.entropy_tensor, value_tensor=selection.value_tensor,
            valid_action_mask=selection.valid_action_mask, observation_kind="two_face", training_rewards=None,
            rewards=[math.sin(index + iteration) for index in range(len(states))],
            dones=[(index + iteration) % 7 == 0 for index in range(len(states))])
        buffer.append(step)
        configure_cy_data_tensor_caches(max_bytes=0)
    prepared = buffer.prepare(bootstrap_value=torch.zeros(len(states)), gamma=.95, gae_lambda=.95,
                              device=torch.device("cpu"))
    with torch.inference_mode():
        replay = evaluate_policy_actions_from_data_list(prepared.data_buffer_list,
            prepared.action_index_buffer_flat, model, device=torch.device("cpu"))
    check_close(replay.log_prob_tensor, prepared.log_prob_buffer_flat)
    chunked = copy.deepcopy(model)
    parameters = dict(prepared_rollout=prepared, device=torch.device("cpu"), num_epochs=2,
                      batch_size=96, clip_coef=.1, value_coef=.5, entropy_coef=.001, max_grad_norm=1)
    torch.manual_seed(41)
    full = train_policy_from_rollout(policy=model, optimizer=torch.optim.SGD(model.parameters(), lr=.0001),
                                    max_graph_size=10**9, **parameters)
    torch.manual_seed(41)
    chunks = train_policy_from_rollout(policy=chunked, optimizer=torch.optim.SGD(chunked.parameters(), lr=.0001),
                                      max_graph_size=2000, **parameters)
    assert math.isclose(full.total_loss, chunks.total_loss, rel_tol=5e-5, abs_tol=5e-5)
    max_parameter_difference = max_gradient_difference = 0
    for first, second in zip(model.parameters(), chunked.parameters(), strict=True):
        check_close(first, second)
        max_parameter_difference = max(max_parameter_difference, float((first - second).abs().max().detach()))
        if first.grad is not None:
            check_close(first.grad, second.grad)
            max_gradient_difference = max(max_gradient_difference, float((first.grad - second.grad).abs().max()))
    references = [weakref.ref(data) for data in prepared.data_buffer_list]
    del prepared, parameters, replay, step, selection
    buffer.clear()
    gc.collect()
    assert all(reference() is None for reference in references)
    record("ppo_chunk_equivalence", samples=288, zero_action_samples=60, epochs=2, minibatch_size=96,
           full_loss=full.total_loss, chunk_loss=chunks.total_loss,
           max_parameter_difference=max_parameter_difference, max_gradient_difference=max_gradient_difference,
           observation_objects_released=True, rss_bytes=rss_bytes())


def stress_geometry(args, record):
    from mdp.cy_geometry_worker import configure_geometry_worker, execute_geometry_request
    from mdp.cy_rollout import build_cy_rollout_collection, create_transition_pool
    rows = selected_rows(args)
    record("geometry_inputs", polytopes=[dict(index=row["polytope_index"], vertices=len(row["vertices"]))
                                         for row in rows])
    configure_cy_data_tensor_caches(max_bytes=2**20, max_entries=4)
    checked_actions, observed = 0, set()
    model = TwoFaceAgent(in_channels=4).eval()
    with create_transition_pool(num_workers=2, memory_budget_gb=12, task_timeout_sec=180,
                                initializer=configure_geometry_worker, initargs=(2**20,)) as pool:
        collection = build_cy_rollout_collection(rows, neighbor_mode="two_neighbors",
            include_points_interior_to_facets=False, include_two_face_metadata=True, transition_pool=pool)
        states = list(collection.initial_states)
        for step in range(12):
            requests = [dict(operation="expand", state=s.to_payload(), configuration=s.configuration,
                             objective_mode=True, action_order="canonical") for s in states]
            expansions = list(pool.imap(execute_geometry_request, requests))
            data_list, next_states = [], []
            for state, expansion in zip(states, expansions, strict=True):
                observed.add(state.key)
                assert not expansion.ambiguous_actions
                data = create_two_face_data_from_state(state, expansion.candidate_actions)
                data_list.append(data)
                before = dict(two_face_restrictions(state.configuration, state.simplices))
                labels = [label for face in before for label in face]
                for index, (action, transition) in enumerate(expansion.transitions):
                    assert action == expansion.candidate_actions[index]
                    after = dict(two_face_restrictions(state.configuration, transition.simplices_from(state.simplices)))
                    changed = [face for face in before if before[face] != after[face]]
                    assert len(changed) == 1
                    face = changed[0]
                    assert int(data.action_face[index]) == list(before).index(face)
                    removed = {tuple(sorted(labels[node] for node in triangle))
                               for triangle in data.triangle_vertices[data.removed_triangle_ids[index]].tolist()}
                    added = {tuple(sorted(labels[node] for node in triangle))
                             for triangle in data.added_triangle_vertices[index].tolist()}
                    assert removed == set(before[face]) - set(after[face])
                    assert added == set(after[face]) - set(before[face])
                    checked_actions += 1
                if expansion.transitions:
                    transition = expansion.transitions[(step * 7 + state.point_config_index) % len(expansion.transitions)][1]
                    next_states.append(replace(state, simplices=frozenset(transition.simplices_from(state.simplices))))
                else:
                    next_states.append(state)
            with torch.inference_mode():
                values, logits, *_ = _forward_policy_data(data_list, model, device=torch.device("cpu"), max_graph_size=20000)
            assert torch.isfinite(values).all()
            for index, data in enumerate(data_list):
                assert torch.isfinite(logits[index, :data.num_available_subcomplexes]).all()
            # Clear geometry caches, then require byte-for-byte equal immutable expansions.
            if step in (0, 5, 11):
                list(pool.imap(execute_geometry_request, [dict(operation="trim")] * 2))
                assert list(pool.imap(execute_geometry_request, requests)) == expansions
            record("geometry_walk", step=step, polytopes=len(states), unique_states=len(observed),
                   checked_actions=checked_actions, max_nodes=max(d.num_nodes for d in data_list),
                   max_actions=max(d.num_available_subcomplexes for d in data_list), rss_bytes=rss_bytes())
            states = next_states
        workers = pool.stats
        record("geometry_complete", checked_actions=checked_actions, unique_states=len(observed),
               workers=workers, tensor_caches=get_cy_data_tensor_cache_stats())
    assert pool._closed
    assert not any(Path(f"/proc/{pid}").exists() for pid in workers["worker_pids"])
    record("geometry_cleanup", worker_processes_released=True)


def stress_search(args, record):
    from eval.config import EvaluationSpec
    from eval.pipeline import run_evaluation
    from eval.policy import EvaluationPolicy
    from eval.results.plotting import read_comparison
    from eval.setup import EvaluationSetup, save_eval_setup
    rows = selected_rows(replace_args(args, extra_dataset=None))[:3]
    spec = EvaluationSpec(3, 12, 1, args.objective_budget, seed=23, two_face_state=True,
        algorithms=("rl_stochastic_policy", "rl_policy_beam_search", "rl_value_beam_search", "rl_value_best_first"),
        policy_checkpoint=args.checkpoint, subcomplex_actor_type="two_face_deep_sets", force_cpu=True,
        transition_num_workers=2, memory_budget_gb=12, runtime_cache_gb=.005, max_hot_states=8,
        policy_max_graph_size=20000, policy_proposal_count=4, beam_width=3)
    setup = EvaluationSetup(rows, {"parameters": spec.setup_parameters(),
                                  "source_metadata": {"fixture": "two_neighbors_h11_12.samples.jsonl"}})
    save_eval_setup(setup, args.output_dir / "setup")
    for name, proposal in (("actor", 4), ("critic", -1)):
        active = replace(spec, policy_proposal_count=proposal,
                         algorithms=spec.algorithms if name == "actor" else ("rl_value_best_first",))
        paths = []
        for cached in (True, False):
            current = replace(active, cache_states=cached)
            policy = EvaluationPolicy.from_spec(current)
            if name == "critic":
                def forbidden(*unused_args, **unused_kwargs):
                    raise AssertionError("Critic-only evaluation invoked actor scoring.")
                policy.score_actions = forbidden
            path = args.output_dir / f"{name}_{'cached' if cached else 'uncached'}"
            start = time.perf_counter()
            run_evaluation(current, setup, output_dir=path, policy=policy)
            read_comparison(path)
            paths.append(path)
            summary = json.loads((path / "summary.json").read_text())
            record("search_run", mode=name, cache_states=cached, elapsed_sec=time.perf_counter() - start,
                   summary=summary, rss_bytes=rss_bytes())
        for filename in ("queries.jsonl", "transitions.jsonl", "expansions.jsonl", "rollouts.jsonl"):
            first = [json.loads(line) for line in (paths[0] / filename).read_text().splitlines()]
            second = [json.loads(line) for line in (paths[1] / filename).read_text().splitlines()]
            assert len(first) == len(second), filename
            for left, right in zip(first, second, strict=True):
                assert left.keys() == right.keys()
                for key in left:
                    if isinstance(left[key], float):
                        assert math.isclose(left[key], right[key], rel_tol=1e-6, abs_tol=1e-9), (filename, key)
                    else:
                        assert left[key] == right[key], (filename, key)
        record("search_cache_equivalence", mode=name, matched_files=4)


def replace_args(args, **kwargs):
    return argparse.Namespace(**{**vars(args), **kwargs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tensors", "ppo", "geometry", "search"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--extra_dataset", help="Optional h11=15 training JSONL; test its 8 largest vertex counts.")
    parser.add_argument("--checkpoint", help="Default-size two_face checkpoint; required by search, optional for tensors.")
    parser.add_argument("--objective_budget", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "search" and not args.checkpoint:
        parser.error("search requires --checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(23)
    report = dict(mode=args.mode, seed=23, device="cpu", torch_version=torch.__version__,
                  arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), events=[], status="running")
    def record(event, **values):
        report["events"].append(dict(event=event, **values))
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(event=event, **values)), flush=True)
    start = time.perf_counter()
    try:
        {"tensors": stress_tensors, "ppo": stress_ppo, "geometry": stress_geometry,
         "search": stress_search}[args.mode](args, record)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        record("finished", elapsed_sec=time.perf_counter() - start,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


if __name__ == "__main__":
    main()
