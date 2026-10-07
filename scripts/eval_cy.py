"""Run from the repository root with the sage environment."""

from __future__ import annotations

import argparse
from dataclasses import fields
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.config import EvaluationSpec, load_evaluation_config
from eval.algorithm import RL_ALGORITHM_NAMES


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate CY search by logical objective-query budget.")
    parser.add_argument("--config", help="JSON EvaluationSpec defaults; explicit CLI arguments take precedence.")
    required = {
        "num_polytopes": "Exact number of matching polytopes, in source order.",
        "h11": "CY h11 in the N lattice (the Hugging Face source h12 column).",
        "num_starts": "Exact number of distinct FRST starts per polytope, shared by all algorithms.",
        "objective_budget": "Logical objective queries per algorithm/start; an already started parent expansion may exceed it.",
    }
    for name, help_text in required.items():
        parser.add_argument(f"--{name}", type=int, help=help_text)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algorithms", nargs="+", choices=("random", "greedy", "best_first", "beam_search", "cyopt_ga", *RL_ALGORITHM_NAMES),
                        default=["random", "greedy"])
    parser.add_argument("--beam_width", type=int, default=4,
                        help="Positive number of next-layer states retained by each beam algorithm (default: 4).")
    parser.add_argument("--policy_proposal_count", type=int,
                        help="Value search proposals per parent: beam defaults to beam_width, BeFS to 4. -1 uses all unseen neighbors without actor logits.")
    parser.add_argument("--value_discount", type=float, default=0.9,
                        help="Critic weight for ln(volume) + weight * V (default: 0.9). RL value-search-only runs accept any finite nonnegative weight; other runs require [0, 1].")
    for name in ("ga_population_size", "ga_elitism", "ga_max_stalled_generations", "ga_face_max_points", "ga_face_samples"):
        parser.add_argument(f"--{name}", type=int, default=getattr(EvaluationSpec, name))
    parser.add_argument("--ga_mutation_rate", type=float, default=EvaluationSpec.ga_mutation_rate)
    parser.add_argument("--policy_checkpoint", default=EvaluationSpec.policy_checkpoint,
                        help="Policy weights file or directory (latest.pth preferred).")
    from models.subcomplex_policy_config import SUPPORTED_SUBCOMPLEX_ACTOR_TYPES

    parser.add_argument("--subcomplex_actor_type", choices=SUPPORTED_SUBCOMPLEX_ACTOR_TYPES, default="snn_simplex")
    for name in ("in_channels", "out_channels", "hidden_channels", "num_layers", "gpu_index", "policy_max_graph_size"):
        parser.add_argument(f"--{name}", type=int, default=getattr(EvaluationSpec, name))
    cpu = parser.add_mutually_exclusive_group()
    cpu.add_argument("--force_cpu", action="store_true")
    cpu.add_argument("--no_force_cpu", dest="force_cpu", action="store_false")
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--profile_cuda_timing", action="store_true",
                       help="Synchronize CUDA for detailed policy timings (slower).")
    timing.add_argument("--no_profile_cuda_timing", dest="profile_cuda_timing", action="store_false")
    parser.set_defaults(force_cpu=False, profile_cuda_timing=False)
    from reward_functions import SUPPORTED_REWARDS

    parser.add_argument("--reward_function", choices=SUPPORTED_REWARDS, default="max_kcup")
    representation = parser.add_mutually_exclusive_group()
    representation.add_argument("--two_face_state", action="store_true",
                                help="Use 2-face equivalence for search deduplication and max_kcup caching/validation; keep full FRSTs for geometry and policy inference.")
    representation.add_argument("--no_two_face_state", dest="two_face_state", action="store_false")
    parser.set_defaults(two_face_state=False)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--cache_states", dest="cache_states", action="store_true")
    cache.add_argument("--no_cache_states", dest="cache_states", action="store_false",
                       help="Disable state, transition graph and objective caches; queries still cost one each.")
    parser.set_defaults(cache_states=True)
    parser.add_argument("--runtime_cache_gb", type=float, default=1.0)
    parser.add_argument("--memory_budget_gb", type=float, default=64.0)
    parser.add_argument("--transition_num_workers", type=int, default=1)
    parser.add_argument("--transition_task_timeout_sec", type=float, default=300.0)
    parser.add_argument("--max_hot_states", type=int, default=100000)
    parser.add_argument("--num_vertices", type=int)
    favorability = parser.add_mutually_exclusive_group()
    favorability.add_argument("--favorable", dest="favorable", action="store_const", const=True)
    favorability.add_argument("--non_favorable", dest="favorable", action="store_const", const=False)
    parser.add_argument("--hf_revision", default="main")
    parser.add_argument("--hf_cache_dir", default=EvaluationSpec.hf_cache_dir)
    parser.add_argument("--polytope_file", help="Local JSON/JSONL N-lattice vertices, using the shared training loader instead of HF.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--skip_insufficient_starts", action="store_true",
                           help="Skip and record candidates whose bounded sampler cannot supply num_starts distinct FRSTs.")
    selection.add_argument("--no_skip_insufficient_starts", dest="skip_insufficient_starts", action="store_false")
    parser.set_defaults(skip_insufficient_starts=False)
    parser.add_argument("--setup_path", help="Load an existing setup offline; its selection parameters must match.")
    parser.add_argument("--setup_only", action="store_true", help="Download and prepare starts without running search.")
    parser.add_argument("--output_dir", help="New results directory, or setup directory with --setup_only.")
    parser.add_argument("--plot_results", action="store_true", help="Generate max_kcup comparison plots after a successful evaluation.")
    parser.add_argument("--parallel_resources", help="JSON per-algorithm CPU/worker/RAM/GPU allocations; run algorithms concurrently.")
    parser.add_argument("--cpu_ids", nargs="+", type=int, help="Restrict this process and geometry workers to these CPUs.")
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    config_path = bootstrap.parse_known_args(argv)[0].config
    if config_path:
        try:
            config = load_evaluation_config(config_path)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        # argparse does not validate choices supplied through set_defaults.
        for action in parser._actions:
            if action.dest in config and action.choices is not None:
                value = config[action.dest]
                values = value if isinstance(value, list) else [value]
                if any(item not in action.choices for item in values):
                    parser.error(f"Invalid config choice for {action.dest}: {value!r}.")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"Required in CLI or config: {', '.join(missing)}.")
    try:
        spec_args = {field.name: getattr(args, field.name) for field in fields(EvaluationSpec)}
        spec_args["algorithms"] = tuple(spec_args["algorithms"])
        EvaluationSpec(**spec_args)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.plot_results and (args.setup_only or args.reward_function != "max_kcup"):
        parser.error("--plot_results requires a search evaluation with reward_function=max_kcup.")
    if args.parallel_resources and args.setup_only:
        parser.error("--parallel_resources is for search, not --setup_only.")
    return args


def restrict_cpu_affinity(cpu_ids):
    """Pin existing library threads as well as the main thread and future workers."""
    if not set(cpu_ids) <= os.sched_getaffinity(0):
        raise ValueError("--cpu_ids includes unavailable CPUs.")
    os.sched_setaffinity(0, cpu_ids)
    for task in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(task.name), cpu_ids)
        except ProcessLookupError:
            pass  # A background thread may finish during enumeration.


def main(argv=None):
    args = vars(parse_args(argv))
    args.pop("config")
    plot_results = args.pop("plot_results")
    parallel_resources = args.pop("parallel_resources")
    cpu_ids = args.pop("cpu_ids")
    if cpu_ids is not None:
        restrict_cpu_affinity(cpu_ids)
    setup_path, setup_only, output_dir = (args.pop(name) for name in ("setup_path", "setup_only", "output_dir"))
    args["algorithms"] = tuple(args["algorithms"])
    spec = EvaluationSpec(**args)
    from eval.setup import load_eval_setup, prepare_eval_setup, save_eval_setup

    setup = load_eval_setup(setup_path) if setup_path else None
    if setup is not None:
        setup.validate(spec)
    if setup_only:
        if setup is None:
            setup = prepare_eval_setup(spec, output_dir=output_dir)
        elif output_dir is not None:
            save_eval_setup(setup, output_dir)
        print(f"Setup: {setup.path}")
        return
    from eval.pipeline import run_evaluation

    if parallel_resources:
        from eval.parallel import run_parallel_evaluation

        result = run_parallel_evaluation(spec, setup, resources_path=parallel_resources, output_dir=output_dir)
    else:
        result = run_evaluation(spec, setup, output_dir=output_dir)
    print(f"Results: {result.output_dir}")
    for rollout in result.rollouts:
        print(f"{rollout.algorithm}: polytope={rollout.polytope_index} start={rollout.start_index} "
              f"queries={rollout.objective_queries} moves={rollout.transition_count} "
              f"expansions={rollout.expansion_count} "
              f"best={rollout.best_objective:.8g} stop={rollout.termination_reason}")
    if plot_results:
        from eval.results.plotting import plot_evaluation

        print(f"Plots: {plot_evaluation(result.output_dir)}")


if __name__ == "__main__":
    main()
