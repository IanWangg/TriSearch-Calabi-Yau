"""Evaluate the h11=15 toric-volume checkpoint on one fixed 4D polytope.

The stochastic policy runs freely over all CYTools ``two_neighbors`` actions,
including transitions to previously visited states.  Enumerating a neighborhood
is free, and only evaluating the selected destination's metric consumes one step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

for _message in (
    r"builtin type SwigPyPacked has no __module__ attribute",
    r"builtin type SwigPyObject has no __module__ attribute",
    r"builtin type swigvarlink has no __module__ attribute",
):
    warnings.filterwarnings("ignore", message=_message, category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=r"\n\*+\nWarning: You have enabled experimental features of CYTools\.",
    category=UserWarning,
)

# The supplied vertices are the columns of the user's 4 x 10 matrix.  CYTools
# expects one vertex per row.
POLYTOPE_VERTICES = (
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, -1, 0),
    (0, 0, 1, 2),
    (2, 2, -1, -2),
    (-2, -1, 0, 2),
    (-1, -2, 2, 2),
    (0, 1, 0, -2),
    (1, 0, -2, -2),
)
POLYTOPE_INDEX = 0
NEIGHBOR_MODE = "two_neighbors"
OBJECTIVE_NAME = "max_toric_cy_volume"
WORKER_RESULT_PREFIX = "WORKER_RESULT="

CanonicalAction = tuple[int, ...]
CanonicalSimplices = tuple[tuple[int, ...], ...]
_CY_RUNTIME_LOADED = False


def load_cy_runtime() -> None:
    """Load Sage/CYTools only inside a geometry-owning subprocess."""

    global _CY_RUNTIME_LOADED
    global Batch
    global CYTriangulationState
    global EGNNSubcomplexAgent
    global Polytope
    global build_cy_data_list
    global get_objective
    global load_policy_checkpoint
    global torch
    if _CY_RUNTIME_LOADED:
        return

    import torch as torch_module
    from torch_geometric.data import Batch as BatchClass

    # Keep CYTools before CYTriangulationState, matching mdp.cy_rollout's safe
    # Sage/FLINT import order.
    from cytools import Polytope as PolytopeClass
    from core.cytools_config import configure_cytools

    configure_cytools()

    from core.cy_policy_rollout_utils import build_cy_data_list as build_data
    from core.cy_runtime_utils import load_policy_checkpoint as load_checkpoint
    from mdp.cy_triangulation_state import CYTriangulationState as StateClass
    from models.egnn_subcomplex_predictor import EGNNSubcomplexAgent as AgentClass
    from reward_functions import get_objective as get_objective_function

    torch = torch_module
    Batch = BatchClass
    Polytope = PolytopeClass
    build_cy_data_list = build_data
    load_policy_checkpoint = load_checkpoint
    CYTriangulationState = StateClass
    EGNNSubcomplexAgent = AgentClass
    get_objective = get_objective_function
    _CY_RUNTIME_LOADED = True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a free stochastic policy and a random-walk baseline on the "
            "fixed 4D reflexive polytope."
        )
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="ckpt/cy_train_h15/latest.pth",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num_initial_states", type=int, default=5)
    parser.add_argument("--initial_state_seed", type=int, default=314159)
    parser.add_argument("--policy_temperature", type=float, default=1.0)
    parser.add_argument(
        "--num_workers",
        "--transition_num_workers",
        dest="num_workers",
        type=int,
        default=20,
        help="Number of independent subprocess trials evaluated concurrently.",
    )
    parser.add_argument("--report_every", type=int, default=5)
    parser.add_argument(
        "--worker_mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if int(args.steps) <= 0:
        raise ValueError("--steps must be positive.")
    if int(args.num_initial_states) <= 0:
        raise ValueError("--num_initial_states must be positive.")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed.")
    if len(set(int(seed) for seed in args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be distinct.")
    if not math.isfinite(float(args.policy_temperature)) or float(args.policy_temperature) <= 0.0:
        raise ValueError("--policy_temperature must be finite and positive.")
    if int(args.num_workers) <= 0:
        raise ValueError("--num_workers must be positive.")


def canonical_simplices(simplices: Any) -> CanonicalSimplices:
    return tuple(
        sorted(tuple(sorted(int(vertex) for vertex in simplex)) for simplex in simplices)
    )


def state_digest(state_key: str) -> str:
    return hashlib.sha256(str(state_key).encode("utf-8")).hexdigest()[:16]


def triangulation_is_frst(triangulation: Any) -> bool:
    return bool(
        triangulation.is_fine()
        and triangulation.is_star()
        and triangulation.is_regular()
    )


def generate_initial_simplices(
    polytope: Polytope,
    *,
    num_initial_states: int,
    seed: int,
) -> list[CanonicalSimplices]:
    """Generate exactly five distinct deterministic FRST starts by default."""

    seed_rng = np.random.default_rng(int(seed))
    triangulations_by_signature: dict[CanonicalSimplices, Any] = {}
    for _ in range(100):
        if len(triangulations_by_signature) >= int(num_initial_states):
            break
        round_seed = int(seed_rng.integers(0, np.iinfo(np.uint32).max))
        needed = int(num_initial_states) - len(triangulations_by_signature)
        triangulations = polytope.random_triangulations_fast(
            N=needed,
            as_list=True,
            progress_bar=False,
            seed=round_seed,
            backend="cgal",
            make_star=True,
            only_fine=True,
            include_points_interior_to_facets=False,
            max_retries=1000,
        )
        for triangulation in triangulations:
            if triangulation_is_frst(triangulation):
                signature = canonical_simplices(triangulation.simplices())
                triangulations_by_signature.setdefault(signature, triangulation)

    if len(triangulations_by_signature) < int(num_initial_states):
        raise RuntimeError(
            "Could not generate the requested number of distinct FRST initial states: "
            f"requested={num_initial_states}, obtained={len(triangulations_by_signature)}."
        )
    return list(triangulations_by_signature)[: int(num_initial_states)]


def materialize_state(
    polytope: Polytope,
    simplices: CanonicalSimplices,
) -> CYTriangulationState:
    triangulation = polytope.triangulate(
        simplices=[list(simplex) for simplex in simplices],
        include_points_interior_to_facets=False,
        check_input_simplices=False,
    )
    points = [[int(coordinate) for coordinate in point] for point in triangulation.points()]
    return CYTriangulationState(
        vertices=points,
        point_config_index=POLYTOPE_INDEX,
        simplices=simplices,
        cy_triangulation=triangulation,
        is_frst=True,
        neighbor_mode=NEIGHBOR_MODE,
    )


def candidate_actions(state: CYTriangulationState) -> tuple[CanonicalAction, ...]:
    state.find_available_actions()
    ambiguous = set(state.ambiguous_subcomplex_actions)
    return tuple(
        tuple(int(vertex) for vertex in action)
        for action in state.get_available_subcomplex_actions()
        if action not in ambiguous
    )


def build_policy(checkpoint_path: str) -> EGNNSubcomplexAgent:
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    device = torch.device("cpu")
    policy = EGNNSubcomplexAgent(
        in_channels=4,
        out_channels=64,
        hidden_channels=64,
        num_layers=3,
        share_encoder=True,
        mlp_hidden_channel_list=[64],
        act="silu",
        subcomplex_actor_type="gnn",
        device=str(device),
    )
    return load_policy_checkpoint(policy, str(checkpoint), map_location=device)


def policy_logits(
    state: CYTriangulationState,
    actions: Sequence[CanonicalAction],
    policy: EGNNSubcomplexAgent,
) -> list[float]:
    data = build_cy_data_list([state], [actions])[0]
    batch = Batch.from_data_list([data])
    with torch.inference_mode():
        _value, logits_padded = policy.get_value_and_logits(batch)
    return [float(value) for value in logits_padded[0, : len(actions)].tolist()]


def stochastic_policy_index(
    logits: Sequence[float],
    available_indices: Sequence[int],
    *,
    temperature: float,
    generator: torch.Generator,
) -> int:
    available_logits = torch.tensor(
        [float(logits[index]) for index in available_indices],
        dtype=torch.float64,
    )
    probabilities = torch.softmax(available_logits / float(temperature), dim=0)
    if not bool(torch.isfinite(probabilities).all()) or float(probabilities.sum()) <= 0.0:
        raise RuntimeError("Policy produced invalid action probabilities.")
    local_index = int(torch.multinomial(probabilities, 1, generator=generator).item())
    return int(available_indices[local_index])


def run_one_trial(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one self-contained trial without transporting CYTools objects."""

    load_cy_runtime()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    method = str(payload["method"])
    seed = int(payload["seed"])
    initial_state_index = int(payload["initial_state_index"])
    max_steps = int(payload["steps"])
    temperature = float(payload["policy_temperature"])
    start_time = time.perf_counter()

    numpy_rng = np.random.default_rng(seed)
    torch_rng = torch.Generator(device="cpu")
    torch_rng.manual_seed(seed)
    polytope = Polytope([list(vertex) for vertex in POLYTOPE_VERTICES])
    state = materialize_state(
        polytope,
        canonical_simplices(payload["initial_simplices"]),
    )
    policy = build_policy(str(payload["checkpoint_path"])) if method == "policy" else None
    objective = get_objective(OBJECTIVE_NAME)

    initial_metric = float(objective(state))
    current_metric = initial_metric
    best_metric = initial_metric
    best_step = 0
    steps_completed = 0
    neighborhood_queries = 0
    neighbors_enumerated = 0
    metric_state_keys = {str(state.key)}
    if method not in {"policy", "random_walk"}:
        raise ValueError(f"Unknown method '{method}'.")
    visited = {str(state.key): 1}
    stopped_reason = "running"

    while steps_completed < max_steps:
        actions = candidate_actions(state)
        neighborhood_queries += 1
        neighbors_enumerated += len(actions)
        if not actions:
            stopped_reason = "dead_end"
            break

        if method == "policy":
            if policy is None:
                raise RuntimeError("Policy trial did not load its checkpoint.")
            logits = policy_logits(state, actions, policy)
            action_index = stochastic_policy_index(
                logits,
                range(len(actions)),
                temperature=temperature,
                generator=torch_rng,
            )
        else:
            action_index = int(numpy_rng.integers(0, len(actions)))

        selected_action = actions[action_index]
        next_simplices, _next_edges, next_key = state.get_transition_output_from_subcomplex_action(
            selected_action
        )
        next_triangulation = state.get_next_cy_triangulation_from_subcomplex_action(
            selected_action
        )
        next_state = CYTriangulationState(
            vertices=state.vertices,
            point_config_index=POLYTOPE_INDEX,
            simplices=next_simplices,
            cy_triangulation=next_triangulation,
            is_frst=True,
            neighbor_mode=NEIGHBOR_MODE,
        )
        if str(next_state.key) != str(next_key):
            raise RuntimeError("Two-neighbor transition key mismatch.")

        visited[str(next_key)] = int(visited.get(str(next_key), 0)) + 1

        # The selected destination metric evaluation is exactly one budgeted
        # step.  The initial metric and all neighborhood work are unbudgeted.
        current_metric = float(objective(next_state))
        steps_completed += 1
        metric_state_keys.add(str(next_key))
        state = next_state
        if current_metric > best_metric:
            best_metric = current_metric
            best_step = steps_completed

    if steps_completed >= max_steps:
        stopped_reason = "step_budget"

    return {
        "method": method,
        "initial_state_index": initial_state_index,
        "initial_state_digest": state_digest(str(payload["initial_state_key"])),
        "seed": seed,
        "steps_completed": steps_completed,
        "budgeted_metric_evaluations": steps_completed,
        "initial_metric_evaluation_is_unbudgeted": True,
        "neighborhood_queries": neighborhood_queries,
        "neighbors_enumerated": neighbors_enumerated,
        "visited_dict_size": len(visited),
        "unique_metric_states": len(metric_state_keys),
        "initial_log10_volume": initial_metric,
        "final_log10_volume": current_metric,
        "best_log10_volume": best_metric,
        "best_volume": float(10.0**best_metric),
        "improvement_log10_volume": best_metric - initial_metric,
        "best_step": best_step,
        "stopped_reason": stopped_reason,
        "runtime_sec": time.perf_counter() - start_time,
    }


def run_setup(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the polytope and generate starts in one isolated process."""

    load_cy_runtime()
    polytope = Polytope([list(vertex) for vertex in POLYTOPE_VERTICES])
    if int(polytope.dim()) != 4 or not bool(polytope.is_reflexive()):
        raise RuntimeError("The supplied vertices did not construct a 4D reflexive polytope.")
    initial_simplices = generate_initial_simplices(
        polytope,
        num_initial_states=int(payload["num_initial_states"]),
        seed=int(payload["initial_state_seed"]),
    )
    initial_states = [materialize_state(polytope, simplices) for simplices in initial_simplices]
    if len({state.key for state in initial_states}) != int(payload["num_initial_states"]):
        raise RuntimeError("Initial-state generation produced duplicate state keys.")
    return {
        "polytope": {
            "vertices": [list(vertex) for vertex in POLYTOPE_VERTICES],
            "dimension": int(polytope.dim()),
            "is_reflexive": bool(polytope.is_reflexive()),
            "h11_N": int(polytope.h11(lattice="N")),
            "num_lattice_points": len(polytope.points()),
        },
        "initial_states": [
            {
                "index": index,
                "key": str(state.key),
                "digest": state_digest(state.key),
                "num_simplices": len(state.simplices),
                "simplices": [list(simplex) for simplex in state.simplices],
            }
            for index, state in enumerate(initial_states)
        ],
    }


def worker_main() -> None:
    payload = json.loads(sys.stdin.read())
    if payload.get("job_type") == "setup":
        result = run_setup(payload)
    else:
        result = run_one_trial(payload)
    print(WORKER_RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)


def launch_trial_subprocess(payload: dict[str, Any]) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    conda_executable = environment.get("CONDA_EXE") or shutil.which("conda")
    if conda_executable is None:
        raise RuntimeError("Could not locate conda for the isolated sage trial process.")
    completed = subprocess.run(
        [
            conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            "sage",
            "python",
            "-c",
            (
                "from scripts.eval_cy_h15_polytope import worker_main; "
                "worker_main()"
            ),
        ],
        cwd=str(_REPO_ROOT),
        env=environment,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        job_label = (
            "setup"
            if payload.get("job_type") == "setup"
            else (
                f"method={payload['method']}, "
                f"initial_state_index={payload['initial_state_index']}, seed={payload['seed']}"
            )
        )
        raise RuntimeError(
            f"Worker subprocess failed ({job_label}, returncode={completed.returncode}).\n"
            f"stdout tail:\n{completed.stdout[-4000:]}\n"
            f"stderr tail:\n{completed.stderr[-4000:]}"
        )
    result_lines = [
        line[len(WORKER_RESULT_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if len(result_lines) != 1:
        raise RuntimeError(
            "Trial subprocess did not emit exactly one result line. "
            f"stdout tail:\n{completed.stdout[-4000:]}"
        )
    return json.loads(result_lines[0])


def aggregate_results(
    results: Sequence[dict[str, Any]],
    *,
    requested_steps: int,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for method in ("policy", "random_walk"):
        rows = [row for row in results if row["method"] == method]
        best = np.asarray([row["best_log10_volume"] for row in rows], dtype=float)
        final = np.asarray([row["final_log10_volume"] for row in rows], dtype=float)
        improvement = np.asarray([row["improvement_log10_volume"] for row in rows], dtype=float)
        steps = np.asarray([row["steps_completed"] for row in rows], dtype=float)
        aggregate[method] = {
            "num_trials": len(rows),
            "completed_requested_steps": sum(
                row["steps_completed"] == int(requested_steps) for row in rows
            ),
            "mean_steps_completed": float(np.mean(steps)),
            "mean_best_log10_volume": float(np.mean(best)),
            "std_best_log10_volume": float(np.std(best)),
            "min_best_log10_volume": float(np.min(best)),
            "max_best_log10_volume": float(np.max(best)),
            "mean_final_log10_volume": float(np.mean(final)),
            "std_final_log10_volume": float(np.std(final)),
            "mean_improvement_log10_volume": float(np.mean(improvement)),
            "std_improvement_log10_volume": float(np.std(improvement)),
        }

    policy_by_pair = {
        (row["initial_state_index"], row["seed"]): row
        for row in results
        if row["method"] == "policy"
    }
    random_by_pair = {
        (row["initial_state_index"], row["seed"]): row
        for row in results
        if row["method"] == "random_walk"
    }
    shared_pairs = sorted(set(policy_by_pair).intersection(random_by_pair))
    differences = np.asarray(
        [
            policy_by_pair[pair]["best_log10_volume"]
            - random_by_pair[pair]["best_log10_volume"]
            for pair in shared_pairs
        ],
        dtype=float,
    )
    aggregate["paired_policy_minus_random"] = {
        "num_pairs": len(shared_pairs),
        "mean_best_log10_volume_difference": float(np.mean(differences)),
        "std_best_log10_volume_difference": float(np.std(differences)),
        "policy_wins": int(np.sum(differences > 0.0)),
        "ties": int(np.sum(differences == 0.0)),
        "random_walk_wins": int(np.sum(differences < 0.0)),
    }
    return aggregate


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    setup = launch_trial_subprocess(
        {
            "job_type": "setup",
            "num_initial_states": int(args.num_initial_states),
            "initial_state_seed": int(args.initial_state_seed),
        }
    )
    polytope_metadata = dict(setup["polytope"])
    initial_states = list(setup["initial_states"])

    print(
        f"checkpoint={checkpoint_path}\n"
        f"polytope=dimension:{polytope_metadata['dimension']} "
        f"reflexive:{polytope_metadata['is_reflexive']} "
        f"h11_N:{polytope_metadata['h11_N']} "
        f"lattice_points:{polytope_metadata['num_lattice_points']}\n"
        f"configuration=steps:{args.steps} seeds:{list(map(int, args.seeds))} "
        f"initial_states:{args.num_initial_states} trials_per_method:"
        f"{len(args.seeds) * int(args.num_initial_states)} neighbor_mode:{NEIGHBOR_MODE} "
        f"policy:stochastic temperature:{args.policy_temperature} workers:{args.num_workers}\n"
        "initial_states="
        + json.dumps(
            [
                {
                    "index": state["index"],
                    "digest": state["digest"],
                    "num_simplices": state["num_simplices"],
                }
                for state in initial_states
            ]
        ),
        flush=True,
    )

    payloads: list[dict[str, Any]] = []
    for method in ("policy", "random_walk"):
        for state in initial_states:
            for seed in args.seeds:
                payloads.append(
                    {
                        "method": method,
                        "initial_state_index": int(state["index"]),
                        "initial_state_key": str(state["key"]),
                        "initial_simplices": state["simplices"],
                        "seed": int(seed),
                        "steps": int(args.steps),
                        "policy_temperature": float(args.policy_temperature),
                        "checkpoint_path": checkpoint_path,
                    }
                )

    results: list[dict[str, Any]] = []
    evaluation_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(int(args.num_workers), len(payloads))) as executor:
        future_to_payload = {
            executor.submit(launch_trial_subprocess, payload): payload for payload in payloads
        }
        for future in as_completed(future_to_payload):
            payload = future_to_payload[future]
            result = future.result()
            results.append(result)
            completed_count = len(results)
            if (
                int(args.report_every) > 0
                and (
                    completed_count == 1
                    or completed_count % int(args.report_every) == 0
                    or completed_count == len(payloads)
                )
            ):
                print(
                    f"completed_trials={completed_count}/{len(payloads)} "
                    f"last_method={payload['method']} "
                    f"last_initial_state={payload['initial_state_index']} "
                    f"last_seed={payload['seed']} "
                    f"last_steps={result['steps_completed']}",
                    flush=True,
                )

    results.sort(key=lambda row: (row["method"], row["initial_state_index"], row["seed"]))
    initial_metrics_by_index: dict[int, set[float]] = {}
    for result in results:
        initial_metrics_by_index.setdefault(int(result["initial_state_index"]), set()).add(
            float(result["initial_log10_volume"])
        )
    if any(len(values) != 1 for values in initial_metrics_by_index.values()):
        raise RuntimeError("Repeated trials disagreed on an initial state's objective value.")

    payload = {
        "configuration": {
            "checkpoint_path": checkpoint_path,
            "objective": OBJECTIVE_NAME,
            "objective_semantics": "log10(CY volume at toric Kahler-cone stretched-cone tip)",
            "goal": "max",
            "neighbor_mode": NEIGHBOR_MODE,
            "include_points_interior_to_facets": False,
            "policy_style": "free_stochastic_policy",
            "policy_stochastic": True,
            "policy_temperature": float(args.policy_temperature),
            "steps": int(args.steps),
            "seeds": [int(seed) for seed in args.seeds],
            "num_initial_states": int(args.num_initial_states),
            "num_trials_per_method": len(args.seeds) * int(args.num_initial_states),
            "initial_state_seed": int(args.initial_state_seed),
            "step_semantics": (
                "one selected destination metric evaluation is one step; "
                "neighborhood queries and the initial metric are unbudgeted"
            ),
            "policy_visited_semantics": (
                "a per-trial dict counts selected-state visits; revisits are allowed"
            ),
            "random_walk_visited_semantics": (
                "a per-trial dict counts selected-state visits; revisits are allowed"
            ),
        },
        "polytope": polytope_metadata,
        "initial_states": [
            {
                "index": int(state["index"]),
                "digest": str(state["digest"]),
                "num_simplices": int(state["num_simplices"]),
                "initial_log10_volume": next(
                    iter(initial_metrics_by_index[int(state["index"])])
                ),
            }
            for state in initial_states
        ],
        "aggregate": aggregate_results(results, requested_steps=int(args.steps)),
        "evaluation_wall_time_sec": time.perf_counter() - evaluation_start,
        "trials": results,
    }
    print("RESULT_JSON_START", flush=True)
    print(json.dumps(payload, sort_keys=True), flush=True)
    print("RESULT_JSON_END", flush=True)


if __name__ == "__main__":
    _args = parse_args()
    if _args.worker_mode:
        worker_main()
    else:
        main(_args)
