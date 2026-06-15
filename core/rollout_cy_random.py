import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    # Import cytools before CYTriangulationState. The reverse order can segfault
    # in the Sage/FLINT stack when Polytope(...) is constructed later.
    from cytools.polytope import Polytope
except ModuleNotFoundError:
    Polytope = None

from mdp.cy_triangulation_state import CYTriangulationState
from mdp.cy_graph import (
    CanonicalAction,
    CanonicalSimplices,
    CYGraphNode,
    CYGraphTransition,
    CYRolloutCollection,
    ExpandSummary,
    RandomRolloutStepResult,
    _sorted_simplices_tuple,
)
from mdp.cy_rollout import (
    RuntimeStateCache,
    TransitionPool,
    build_cy_rollout_collection,
    create_runtime_state_cache,
    create_transition_pool,
    default_is_target_state,
    get_state_from_runtime_cache,
    load_cy_sample_rows,
    register_runtime_state,
    runtime_cache_hot_size,
    runtime_cache_total_unique_states,
)
from core.cy_runtime_utils import read_process_memory_gb


def _safe_bool_method_call(obj: Any, method_name: str) -> bool:
    from core.cytools_config import REGULARITY_BACKEND
    method = getattr(obj, method_name, None)
    if method is None or not callable(method):
        return False
    try:
        if method_name == "is_regular":
            return bool(method(backend=REGULARITY_BACKEND))
        return bool(method())
    except Exception:
        return False


def _expand_cy_state_worker(state: CYTriangulationState):
    simplices = _sorted_simplices_tuple(state.simplices)
    if default_is_target_state(state) or len(simplices) <= 1:
        return str(state.key), int(state.point_config_index), simplices, tuple(), tuple()

    if not state.actions_ready:
        state.find_available_actions()

    candidate_actions = tuple(
        action
        for action in state.get_available_subcomplex_actions()
        if action not in getattr(state, "ambiguous_subcomplex_actions", frozenset())
    )
    transitions = []
    for action in candidate_actions:
        next_simplices, _next_edges, next_key = state.get_transition_output_from_subcomplex_action(action)
        next_tri = state.get_next_cy_triangulation_from_subcomplex_action(action)
        transitions.append(
            (
                tuple(int(v) for v in action),
                CYGraphTransition(
                    next_key=str(next_key),
                    next_simplices=_sorted_simplices_tuple(next_simplices),
                    next_is_target=(
                        _safe_bool_method_call(next_tri, "is_fine") and _safe_bool_method_call(next_tri, "is_regular")
                    )
                    if next_tri is not None
                    else None,
                ),
            )
        )
    return str(state.key), int(state.point_config_index), simplices, candidate_actions, tuple(transitions)


class CYRandomRolloutEngine:
    def __init__(
        self,
        *,
        collection: CYRolloutCollection,
        include_points_interior_to_facets: bool,
        state_cache_mode: str,
        max_hot_states: int,
    ):
        self.collection = collection
        self.include_points_interior_to_facets = bool(include_points_interior_to_facets)
        self.state_cache = create_runtime_state_cache(
            mode=state_cache_mode,
            base_states=collection.base_states,
            max_hot_states=max_hot_states,
        )
        self.nodes_by_key: Dict[str, CYGraphNode] = {}
        self.graph_by_polytope: Dict[int, Dict[str, CYGraphNode]] = {}
        for state in collection.base_states.values():
            self._register_node(
                key=str(state.key),
                point_config_index=int(state.point_config_index),
                simplices=_sorted_simplices_tuple(state.simplices),
            )

    def _register_node(
        self,
        *,
        key: str,
        point_config_index: int,
        simplices: CanonicalSimplices,
    ) -> Tuple[CYGraphNode, bool]:
        node = self.nodes_by_key.get(key)
        if node is not None:
            return node, False
        node = CYGraphNode(key=key, point_config_index=point_config_index, simplices=simplices)
        self.nodes_by_key[key] = node
        self.graph_by_polytope.setdefault(point_config_index, {})[key] = node
        return node, True

    def _store_expansion(self, payload) -> int:
        key, point_config_index, simplices, candidate_actions, transitions = payload
        node, _ = self._register_node(
            key=key,
            point_config_index=point_config_index,
            simplices=simplices,
        )
        node.candidate_actions = tuple(candidate_actions)
        node.transitions = {action: transition for action, transition in transitions}
        node.expanded = True

        discovered = 0
        for transition in node.transitions.values():
            _, is_new = self._register_node(
                key=transition.next_key,
                point_config_index=point_config_index,
                simplices=transition.next_simplices,
            )
            discovered += int(is_new)
        return discovered

    def expand_states(
        self,
        states: Sequence[CYTriangulationState],
        *,
        use_multiprocessing: bool,
        transition_pool: TransitionPool | None,
        transition_mp_chunksize: int,
        transition_mp_min_batch: int,
    ) -> ExpandSummary:
        unique_unexpanded: Dict[str, CYTriangulationState] = {}
        for state in states:
            self._register_node(
                key=str(state.key),
                point_config_index=int(state.point_config_index),
                simplices=_sorted_simplices_tuple(state.simplices),
            )
            if not self.nodes_by_key[str(state.key)].expanded:
                unique_unexpanded.setdefault(str(state.key), state)

        if not unique_unexpanded:
            return ExpandSummary(0, 0, False)

        pending = list(unique_unexpanded.values())
        use_mp = (
            bool(use_multiprocessing)
            and transition_pool is not None
            and len(pending) >= max(1, int(transition_mp_min_batch))
        )
        if use_mp:
            outputs = transition_pool.map(
                _expand_cy_state_worker,
                pending,
                chunksize=max(1, int(transition_mp_chunksize)),
            )
        else:
            outputs = [_expand_cy_state_worker(state) for state in pending]

        discovered = 0
        for output in outputs:
            discovered += self._store_expansion(output)
        return ExpandSummary(len(outputs), discovered, use_mp)

    def materialize_state(self, key: str) -> CYTriangulationState:
        cached = get_state_from_runtime_cache(self.state_cache, key)
        if cached is not None:
            return cached

        node = self.nodes_by_key[key]
        polytope = self.collection.polytope_by_index[node.point_config_index]
        triangulation = polytope.triangulate(
            simplices=[list(simplex) for simplex in node.simplices],
            include_points_interior_to_facets=self.include_points_interior_to_facets,
            check_input_simplices=False,
        )
        state = CYTriangulationState(
            vertices=self.collection.vertices_by_polytope[node.point_config_index],
            point_config_index=node.point_config_index,
            simplices=node.simplices,
            cy_triangulation=triangulation,
        )
        register_runtime_state(self.state_cache, state)
        return state

    def filter_actionable_initial_states(
        self,
        states: Sequence[CYTriangulationState],
        *,
        use_multiprocessing: bool,
        transition_pool: TransitionPool | None,
        transition_mp_chunksize: int,
        transition_mp_min_batch: int,
    ) -> List[CYTriangulationState]:
        self.expand_states(
            states,
            use_multiprocessing=use_multiprocessing,
            transition_pool=transition_pool,
            transition_mp_chunksize=transition_mp_chunksize,
            transition_mp_min_batch=transition_mp_min_batch,
        )
        return [state for state in states if len(self.nodes_by_key[str(state.key)].candidate_actions) > 0]

    def sample_initial_states(
        self,
        num_states: int,
        *,
        rng: np.random.Generator,
        initial_state_pool: Sequence[CYTriangulationState],
    ) -> List[CYTriangulationState]:
        indices = rng.integers(0, len(initial_state_pool), size=int(num_states))
        return [initial_state_pool[int(idx)] for idx in indices]

    def graph_node_count(self) -> int:
        return len(self.nodes_by_key)

    def graph_edge_count(self) -> int:
        return sum(len(node.transitions) for node in self.nodes_by_key.values())

    def graph_stats_by_polytope(self) -> Dict[int, Dict[str, int]]:
        return {
            polytope_index: {
                "nodes": len(nodes),
                "edges": sum(len(node.transitions) for node in nodes.values()),
                "expanded_nodes": sum(int(node.expanded) for node in nodes.values()),
            }
            for polytope_index, nodes in self.graph_by_polytope.items()
        }

    def rollout_step(
        self,
        states: Sequence[CYTriangulationState],
        *,
        rng: np.random.Generator,
        initial_state_pool: Sequence[CYTriangulationState],
        use_multiprocessing: bool,
        transition_pool: TransitionPool | None,
        transition_mp_chunksize: int,
        transition_mp_min_batch: int,
    ) -> RandomRolloutStepResult:
        current_states = list(states)
        current_expand = self.expand_states(
            current_states,
            use_multiprocessing=use_multiprocessing,
            transition_pool=transition_pool,
            transition_mp_chunksize=transition_mp_chunksize,
            transition_mp_min_batch=transition_mp_min_batch,
        )

        transitioned_states: List[CYTriangulationState] = []
        next_states = list(current_states)
        rewards = [0.0 for _ in current_states]
        dones = [False for _ in current_states]
        chosen_actions: List[Optional[CanonicalAction]] = []
        terminal_reasons = ["continue" for _ in current_states]
        frt_hits = 0
        collapsed_hits = 0
        dead_end_hits = 0
        pending_next_keys: Dict[str, None] = {}

        for idx, state in enumerate(current_states):
            node = self.nodes_by_key[str(state.key)]
            if len(node.candidate_actions) == 0:
                transitioned_states.append(state)
                dones[idx] = True
                terminal_reasons[idx] = "dead_end_current"
                chosen_actions.append(None)
                dead_end_hits += 1
                continue

            action = node.candidate_actions[int(rng.integers(0, len(node.candidate_actions)))]
            chosen_actions.append(action)
            transition = node.transitions[action]

            if transition.next_is_target is True:
                transitioned_states.append(state)
                rewards[idx] = 1.0
                dones[idx] = True
                terminal_reasons[idx] = "frt_or_frst"
                frt_hits += 1
                continue

            if len(transition.next_simplices) <= 1:
                transitioned_states.append(state)
                rewards[idx] = -1.0
                dones[idx] = True
                terminal_reasons[idx] = "single_simplex"
                collapsed_hits += 1
                continue

            next_state = self.materialize_state(transition.next_key)
            transitioned_states.append(next_state)
            next_states[idx] = next_state
            if default_is_target_state(next_state):
                rewards[idx] = 1.0
                dones[idx] = True
                terminal_reasons[idx] = "frt_or_frst"
                frt_hits += 1
            else:
                pending_next_keys[next_state.key] = None

        future_states = [self.materialize_state(key) for key in pending_next_keys]
        next_expand = self.expand_states(
            future_states,
            use_multiprocessing=use_multiprocessing,
            transition_pool=transition_pool,
            transition_mp_chunksize=transition_mp_chunksize,
            transition_mp_min_batch=transition_mp_min_batch,
        )

        for idx, state in enumerate(next_states):
            if dones[idx]:
                continue
            if len(self.nodes_by_key[str(state.key)].candidate_actions) == 0:
                dones[idx] = True
                terminal_reasons[idx] = "dead_end_next"
                dead_end_hits += 1

        reset_indices = [idx for idx, done in enumerate(dones) if done]
        if reset_indices:
            reset_states = self.sample_initial_states(
                len(reset_indices),
                rng=rng,
                initial_state_pool=initial_state_pool,
            )
            for idx, reset_state in zip(reset_indices, reset_states):
                next_states[idx] = reset_state

        return RandomRolloutStepResult(
            input_states=current_states,
            transitioned_states=transitioned_states,
            next_states=next_states,
            rewards=rewards,
            dones=dones,
            chosen_actions=chosen_actions,
            terminal_reasons=terminal_reasons,
            reset_count=len(reset_indices),
            frt_hits=frt_hits,
            collapsed_hits=collapsed_hits,
            dead_end_hits=dead_end_hits,
            expanded_states=current_expand.expanded_count + next_expand.expanded_count,
            discovered_states=current_expand.discovered_count + next_expand.discovered_count,
            used_multiprocessing=current_expand.used_multiprocessing or next_expand.used_multiprocessing,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl",
        help="Path to CY .samples.jsonl file.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap on the number of polytopes loaded from the JSONL file.",
    )
    parser.add_argument(
        "--include_points_interior_to_facets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to cytools Polytope.triangulate(...).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel rollout states.")
    parser.add_argument("--rollout_steps", type=int, default=100, help="Number of random-policy rollout steps.")
    parser.add_argument(
        "--filter_actionable_initial_states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expand the dataset initial states once and keep only states with at least one valid regular flip.",
    )
    parser.add_argument(
        "--use_multiprocessing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a process pool when expanding previously unseen states.",
    )
    parser.add_argument(
        "--transition_num_workers",
        type=int,
        default=0,
        help="Number of worker processes. <=0 uses os.cpu_count() in TransitionPool.",
    )
    parser.add_argument(
        "--transition_mp_start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method for expansion workers.",
    )
    parser.add_argument(
        "--transition_mp_chunksize",
        type=int,
        default=16,
        help="Chunksize for worker expansion batches.",
    )
    parser.add_argument(
        "--transition_mp_min_batch",
        type=int,
        default=1,
        help="Minimum number of unseen states before using multiprocessing.",
    )
    parser.add_argument(
        "--state_cache_mode",
        type=str,
        default="lru",
        choices=["full", "lru", "none"],
        help="Object-cache policy for materialized runtime states.",
    )
    parser.add_argument(
        "--max_hot_states",
        type=int,
        default=100000,
        help="Maximum runtime state objects kept in memory when --state_cache_mode=lru.",
    )
    parser.add_argument(
        "--report_every",
        type=int,
        default=10,
        help="Print rollout progress every N steps. <=0 disables periodic progress logs.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a very small rollout for a quick smoke test.",
    )
    return parser.parse_args()


def increment_visitation(states) -> None:
    for state in states:
        if hasattr(state, "visitation"):
            state.visitation += 1


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    if args.dry_run:
        args.max_rows = 2 if args.max_rows is None else min(int(args.max_rows), 2)
        args.num_envs = min(int(args.num_envs), 8)
        args.rollout_steps = min(int(args.rollout_steps), 5)
        args.report_every = 1
        print(
            "Dry-run overrides: "
            f"max_rows={args.max_rows}, num_envs={args.num_envs}, rollout_steps={args.rollout_steps}"
        )

    dataset_path = str(Path(args.dataset_path).expanduser())
    print(f"Loading CY rollout dataset from {dataset_path}")
    rows = load_cy_sample_rows(dataset_path, max_rows=args.max_rows)
    print(f"Loaded {len(rows)} polytopes")

    build_start = time.perf_counter()
    collection = build_cy_rollout_collection(
        rows,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
    )
    build_sec = time.perf_counter() - build_start
    print(
        "Built rollout collection: "
        f"base_states={len(collection.base_states)}, "
        f"initial_states={len(collection.initial_states)}, "
        f"polytopes={len(collection.polytope_indices)}, "
        f"time={build_sec:.2f}s"
    )

    engine = CYRandomRolloutEngine(
        collection=collection,
        include_points_interior_to_facets=args.include_points_interior_to_facets,
        state_cache_mode=args.state_cache_mode,
        max_hot_states=args.max_hot_states,
    )

    transition_pool = None
    if args.use_multiprocessing:
        transition_pool = create_transition_pool(
            num_workers=args.transition_num_workers,
            start_method=args.transition_mp_start_method,
        )
        print(
            "Multiprocessing enabled: "
            f"workers={args.transition_num_workers}, "
            f"start_method={args.transition_mp_start_method}, "
            f"chunksize={args.transition_mp_chunksize}, "
            f"min_batch={args.transition_mp_min_batch}"
        )

    try:
        initial_state_pool = list(collection.initial_states)
        if args.filter_actionable_initial_states:
            filter_start = time.perf_counter()
            initial_state_pool = engine.filter_actionable_initial_states(
                initial_state_pool,
                use_multiprocessing=args.use_multiprocessing,
                transition_pool=transition_pool,
                transition_mp_chunksize=args.transition_mp_chunksize,
                transition_mp_min_batch=args.transition_mp_min_batch,
            )
            filter_sec = time.perf_counter() - filter_start
            print(
                "Filtered initial state pool: "
                f"actionable={len(initial_state_pool)}/{len(collection.initial_states)}, "
                f"time={filter_sec:.2f}s"
            )

        if not initial_state_pool:
            raise ValueError("The initial state pool is empty after filtering.")

        states = engine.sample_initial_states(args.num_envs, rng=rng, initial_state_pool=initial_state_pool)

        total_frt_hits = 0
        total_collapsed_hits = 0
        total_dead_end_hits = 0
        total_resets = 0
        total_expanded_states = 0
        total_discovered_states = 0
        total_mp_steps = 0

        rollout_start = time.perf_counter()
        for step_idx in range(int(args.rollout_steps)):
            increment_visitation(states)
            step_result = engine.rollout_step(
                states,
                rng=rng,
                initial_state_pool=initial_state_pool,
                use_multiprocessing=args.use_multiprocessing,
                transition_pool=transition_pool,
                transition_mp_chunksize=args.transition_mp_chunksize,
                transition_mp_min_batch=args.transition_mp_min_batch,
            )
            states = step_result.next_states

            total_frt_hits += int(step_result.frt_hits)
            total_collapsed_hits += int(step_result.collapsed_hits)
            total_dead_end_hits += int(step_result.dead_end_hits)
            total_resets += int(step_result.reset_count)
            total_expanded_states += int(step_result.expanded_states)
            total_discovered_states += int(step_result.discovered_states)
            total_mp_steps += int(step_result.used_multiprocessing)

            should_report = args.report_every > 0 and (
                step_idx == 0 or (step_idx + 1) % int(args.report_every) == 0 or (step_idx + 1) == int(args.rollout_steps)
            )
            if should_report:
                rss_gb, hwm_gb = read_process_memory_gb()
                avg_reward = float(np.mean(step_result.rewards)) if step_result.rewards else 0.0
                done_fraction = float(np.mean(step_result.dones)) if step_result.dones else 0.0
                print(
                    f"step={step_idx + 1}/{args.rollout_steps} "
                    f"avg_reward={avg_reward:.4f} "
                    f"done_fraction={done_fraction:.4f} "
                    f"resets={step_result.reset_count} "
                    f"expanded={step_result.expanded_states} "
                    f"discovered={step_result.discovered_states} "
                    f"graph_nodes={engine.graph_node_count()} "
                    f"graph_edges={engine.graph_edge_count()} "
                    f"cached_states={runtime_cache_total_unique_states(engine.state_cache)} "
                    f"hot_cache={runtime_cache_hot_size(engine.state_cache)} "
                    f"rss_gb={rss_gb:.2f} "
                    f"hwm_gb={hwm_gb:.2f}"
                )

        rollout_sec = time.perf_counter() - rollout_start
        rss_gb, hwm_gb = read_process_memory_gb()
        env_steps = int(args.num_envs) * int(args.rollout_steps)
        env_steps_per_sec = env_steps / rollout_sec if rollout_sec > 0 else 0.0
        graph_stats = engine.graph_stats_by_polytope()
        expanded_nodes = sum(stats["expanded_nodes"] for stats in graph_stats.values())

        print("Rollout summary")
        print(
            f"env_steps={env_steps} "
            f"rollout_sec={rollout_sec:.2f} "
            f"env_steps_per_sec={env_steps_per_sec:.2f}"
        )
        print(
            f"graph_nodes={engine.graph_node_count()} "
            f"graph_edges={engine.graph_edge_count()} "
            f"expanded_nodes={expanded_nodes}"
        )
        print(
            f"frt_hits={total_frt_hits} "
            f"collapsed_hits={total_collapsed_hits} "
            f"dead_end_hits={total_dead_end_hits} "
            f"resets={total_resets}"
        )
        print(
            f"expanded_states={total_expanded_states} "
            f"discovered_states={total_discovered_states} "
            f"mp_steps={total_mp_steps}/{args.rollout_steps}"
        )
        print(
            f"cached_states={runtime_cache_total_unique_states(engine.state_cache)} "
            f"hot_cache={runtime_cache_hot_size(engine.state_cache)} "
            f"rss_gb={rss_gb:.2f} "
            f"hwm_gb={hwm_gb:.2f}"
        )
    finally:
        if transition_pool is not None:
            transition_pool.shutdown()


if __name__ == "__main__":
    main(parse_args())
