"""Run algorithms against the same prepared inputs using training's runtime."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import time
from typing import Callable, Mapping
import uuid

from core.cy_managed_runtime import managed_rollout_runtime
from eval.algorithm import Algorithm, RLAlgorithm, get_algorithm
from eval.batched_rollout import run_batched_rollouts
from eval.policy import EvaluationPolicy, PolicyScorer
from eval.config import EVAL_ROOT, EvaluationSpec, derive_seed
from eval.results.writer import ResultWriter
from eval.rollout import RESULT_FORMAT_VERSION, RolloutResult, run_rollout
from eval.setup import EvaluationSetup, prepare_eval_setup, save_eval_setup
from mdp.cy_rollout import CYRandomRolloutEngine, build_cy_rollout_collection, create_transition_pool
from mdp.cy_state_record import state_key
from reward_functions import get_objective, get_reward, infer_goal


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    setup_id: str
    rollouts: list[RolloutResult]


def _environment_metadata() -> dict:
    packages = {}
    for name in ("cytools", "cyopt", "numpy", "torch", "torch_geometric", "datasets", "huggingface_hub"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=EVAL_ROOT.parent,
                            capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=EVAL_ROOT.parent,
                            capture_output=True, text=True, check=False)
    return {"packages": packages, "git_commit": commit.stdout.strip() or None,
            "git_dirty": bool(status.stdout.strip())}


def run_evaluation(
    spec: EvaluationSpec,
    setup: EvaluationSetup | None = None,
    *,
    output_dir: str | Path | None = None,
    algorithm_factories: Mapping[str, Callable[[], Algorithm]] | None = None,
    policy: PolicyScorer | None = None,
    search_observer=None,
) -> EvaluationResult:
    """Each algorithm owns its workers/caches; starts share caches within it.

    New algorithms can be supplied as zero-argument factories. A fresh instance
    is created for each start, so algorithm-local search state cannot leak.
    """
    factories = dict(algorithm_factories or {})
    algorithm_options = dict(beam_width=spec.beam_width, policy_proposal_count=spec.policy_proposal_count,
                             value_discount=spec.value_discount, ga_population_size=spec.ga_population_size,
                             ga_mutation_rate=spec.ga_mutation_rate, ga_elitism=spec.ga_elitism,
                             ga_max_stalled_generations=spec.ga_max_stalled_generations)
    for name in spec.algorithms:
        if name not in factories:
            get_algorithm(name, **algorithm_options)  # Reject misspellings before loading data.
    if setup is None:
        setup = prepare_eval_setup(spec)
    setup.validate(spec)
    save_eval_setup(setup, setup.path or EVAL_ROOT / "data" / "setups" / setup.setup_id)
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = EVAL_ROOT / "results" / "runs" / f"eval_{stamp}_{uuid.uuid4().hex[:8]}"
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime_args = Namespace(
        **{**spec.to_dict(), "runtime_cache_gb": spec.runtime_cache_gb if spec.cache_states else 0.0},
        use_multiprocessing=spec.transition_num_workers > 1,
        transition_mp_start_method="spawn", action_order="canonical",
    )
    rollouts = []
    with ResultWriter(output_dir) as writer:
        config = {
            "format_version": RESULT_FORMAT_VERSION, "spec": spec.to_dict(), "setup_id": setup.setup_id,
            "setup_path": str(setup.path), "neighbor_mode": "two_neighbors",
            "budget_unit": "logical_objective_query", "initial_objective_is_free": True,
            "budget_tail": "complete_parent_expansion", "environment": _environment_metadata(),
            "state_representation": "two_face_restrictions" if spec.two_face_state else "full_simplices",
        }
        writer.write_json("config", config)
        if "cyopt_ga" in spec.algorithms:
            from eval.algorithm.cyopt_ga import cyopt_metadata

            (output_dir / "cyopt_encoding").mkdir()
            config["cyopt"] = {
                **cyopt_metadata(), "representation": "sorted_2_face_triangulations",
                "initial_population": "shared_start_then_uniform_dna",
                "fitness": "negative_raw_objective", "selection": "tournament_k_3",
                "crossover": "npoint_1", "mutation_k": 1, "fitness_cache_size": 0,
                "budget_tail": "stop_before_next_objective_query",
                "infeasible_dna": "free_geometry_rejection_logged_in_expansion",
                "preparation": "codebook_generation_excluded_from_objective_budget_wall_time_recorded",
                "expansion_unit": "population_initialization_or_generation",
            }
            writer.write_json("config", config)
        runtime_stats, policy_stats = {}, {}
        policy_proposal_counts = {}
        value_score_definitions = {}
        for name in spec.algorithms:
            with managed_rollout_runtime(runtime_args, create_transition_pool) as (pool, cache_bytes, register_engine):
                collection = build_cy_rollout_collection(
                    setup.rows, include_points_interior_to_facets=False,
                    neighbor_mode="two_neighbors", transition_pool=pool,
                    two_face_state=spec.two_face_state,
                    include_two_face_metadata=(getattr(policy, "observation_kind", spec.policy_observation_kind) == "two_face"),
                )
                reward = get_reward(spec.reward_function)
                objective = get_objective(spec.reward_function, reward=reward)
                goal = infer_goal(spec.reward_function)
                engine = CYRandomRolloutEngine(
                    collection=collection, include_points_interior_to_facets=False,
                    neighbor_mode="two_neighbors", reward_function=reward,
                    state_cache_mode="lru" if spec.cache_states else "none",
                    max_hot_states=spec.max_hot_states, cache_budget_bytes=cache_bytes,
                    history_path=str(output_dir / "runtime" / name / "states.sqlite3"),
                    action_order="canonical",
                    two_face_state=spec.two_face_state,
                )
                register_engine(engine)
                starts, algorithms, seeds, start_indices = [], [], [], []
                for row in setup.rows:
                    for start_index, entry in enumerate(row["frst_list"]):
                        key = state_key(row["polytope_index"], entry["simplices"], "two_neighbors")
                        algorithm = factories[name]() if name in factories else get_algorithm(name, **algorithm_options)
                        if algorithm.name != name:
                            raise ValueError(f"Algorithm factory {name!r} returned {algorithm.name!r}.")
                        starts.append(collection.base_states[key])
                        algorithms.append(algorithm)
                        seeds.append(derive_seed(spec.seed, "rollout", name, row["polytope_index"], start_index))
                        start_indices.append(start_index)
                if all(isinstance(algorithm, RLAlgorithm) for algorithm in algorithms):
                    if policy is None:
                        policy = EvaluationPolicy.from_spec(spec)
                    policy_proposal_counts[name] = algorithms[0].proposal_count
                    definition = getattr(algorithms[0], "value_score_definition", None)
                    if definition is not None:
                        value_score_definitions[name] = definition
                    config["policy"] = {
                        **getattr(policy, "metadata", {"injected": True}),
                        "resolved_policy_proposal_count": spec.resolved_policy_proposal_count,
                        "resolved_policy_proposal_counts": dict(policy_proposal_counts),
                        "value_score_definitions": dict(value_score_definitions),
                        "value_discount": spec.value_discount,
                    }
                    writer.write_json("config", config)
                    if isinstance(policy, EvaluationPolicy):
                        policy.reset_stats()
                    started = time.perf_counter()
                    results = run_batched_rollouts(
                        starts, algorithms, engine, policy=policy,
                        objective_function=objective, objective_goal=goal,
                        objective_budget=spec.objective_budget, objective_name=spec.reward_function,
                        seeds=seeds, start_indices=start_indices, reward_function=reward,
                        batch_objective_function=lambda states: engine.objective_values(states, spec.reward_function),
                        on_query=writer.query, on_transition=writer.transition, on_expansion=writer.expansion,
                        on_rollout=writer.rollout,
                        two_face_state=spec.two_face_state,
                        search_observer=search_observer,
                    )
                    rollouts.extend(results)
                    policy_stats[name] = {**(policy.stats() if isinstance(policy, EvaluationPolicy) else {}),
                                          "wall_sec": time.perf_counter() - started}
                else:
                    if any(isinstance(algorithm, RLAlgorithm) for algorithm in algorithms):
                        raise ValueError("An algorithm factory must consistently return the same algorithm family.")
                    encoding = None
                    for state, algorithm, seed, start_index in zip(starts, algorithms, seeds, start_indices):
                        from eval.algorithm.cyopt_ga import CyoptEncoding, CyoptGAAlgorithm

                        if isinstance(algorithm, CyoptGAAlgorithm) and spec.objective_budget:
                            if encoding is None or encoding.configuration != state.configuration:
                                encoding = CyoptEncoding(
                                    [item for item in starts if item.point_config_index == state.point_config_index],
                                    seed=derive_seed(spec.seed, "cyopt_encoding", state.point_config_index),
                                    max_points=spec.ga_face_max_points, samples=spec.ga_face_samples,
                                )
                                writer.write_json(f"cyopt_encoding/polytope_{state.point_config_index}", encoding.metadata)
                                print(f"cyopt_ga: polytope={state.point_config_index} "
                                      f"DNA dimensions={len(encoding.bounds)} "
                                      f"face counts={[len(face) for face in encoding.metadata['face_triangulations']]}", flush=True)
                            algorithm.encoding = encoding
                        result = run_rollout(
                            state, algorithm, engine,
                            objective_function=objective, objective_goal=goal,
                            batch_objective_function=lambda states: engine.objective_values(states, spec.reward_function),
                            objective_budget=spec.objective_budget,
                            seed=seed,
                            start_index=start_index, objective_name=spec.reward_function,
                            on_query=writer.query, on_transition=writer.transition, on_expansion=writer.expansion,
                            two_face_state=spec.two_face_state,
                        )
                        rollouts.append(result)
                        writer.rollout(result)
                runtime_stats[name] = engine.memory_stats()
        writer.write_json("summary", {
            "format_version": RESULT_FORMAT_VERSION,
            "status": "complete", "setup_id": setup.setup_id, "num_rollouts": len(rollouts),
            "objective_queries": sum(result.objective_queries for result in rollouts),
            "transition_count": sum(result.transition_count for result in rollouts),
            "expansion_count": sum(result.expansion_count for result in rollouts),
            "budget_overshoot": sum(result.budget_overshoot for result in rollouts),
            "runtime_stats": runtime_stats,
            "policy_stats": policy_stats,
        })
    return EvaluationResult(output_dir, setup.setup_id, rollouts)
