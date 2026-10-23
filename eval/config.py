"""Evaluation specifications, independent of geometry and CLI parsing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin, get_type_hints


EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY_CHECKPOINT = str(
    EVAL_ROOT.parent / "runs" / "cy_snn_kcup_h11_15_20260921_123000_1584461" / "checkpoints"
)


@dataclass(frozen=True)
class EvaluationSpec:
    num_polytopes: int
    h11: int
    num_starts: int
    objective_budget: int
    seed: int = 0
    algorithms: tuple[str, ...] = ("random", "greedy")
    reward_function: str = "max_kcup"
    cache_states: bool = True
    runtime_cache_gb: float = 1.0
    memory_budget_gb: float = 64.0
    transition_num_workers: int = 1
    transition_task_timeout_sec: float = 300.0
    max_hot_states: int = 100000
    num_vertices: int | None = None
    favorable: bool | None = None
    hf_revision: str = "main"
    hf_cache_dir: str = str(EVAL_ROOT / "data" / "cache")
    polytope_file: str | None = None
    beam_width: int = 4
    policy_proposal_count: int | None = None
    value_discount: float = 0.9
    policy_checkpoint: str = DEFAULT_POLICY_CHECKPOINT
    subcomplex_actor_type: str = "snn_simplex"
    in_channels: int = 4
    out_channels: int = 64
    hidden_channels: int = 64
    num_layers: int = 3
    gpu_index: int = 0
    force_cpu: bool = False
    policy_max_graph_size: int = 250000
    profile_cuda_timing: bool = False
    skip_insufficient_starts: bool = False
    ga_population_size: int = 50
    ga_mutation_rate: float = 0.1
    ga_elitism: int = 1
    ga_max_stalled_generations: int = 20
    ga_face_max_points: int = 12
    ga_face_samples: int = 1000
    two_face_state: bool = False

    def __post_init__(self) -> None:
        if type(self.two_face_state) is not bool:
            raise ValueError("two_face_state must be a boolean.")
        if self.two_face_state and self.reward_function != "max_kcup":
            raise ValueError("two_face_state currently supports only max_kcup.")
        for name in ("num_polytopes", "num_starts", "transition_num_workers", "max_hot_states", "beam_width",
                     "in_channels", "out_channels", "hidden_channels", "num_layers", "policy_max_graph_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("h11", "objective_budget", "seed", "gpu_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.num_vertices is not None and (type(self.num_vertices) is not int or self.num_vertices <= 0):
            raise ValueError("num_vertices must be a positive integer when specified.")
        for name in ("runtime_cache_gb", "memory_budget_gb", "transition_task_timeout_sec"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "runtime_cache_gb" and value == 0):
                raise ValueError(f"Invalid {name}: {value!r}.")
        if not self.algorithms or len(set(self.algorithms)) != len(self.algorithms):
            raise ValueError("algorithms must be nonempty and distinct.")
        if any(not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)
               for name in self.algorithms):
            raise ValueError("Algorithm names must use lowercase letters, digits and underscores.")
        if type(self.cache_states) is not bool or (self.favorable is not None and type(self.favorable) is not bool):
            raise ValueError("cache_states and favorable must be booleans (favorable may be None).")
        if not self.hf_revision:
            raise ValueError("hf_revision must be nonempty.")
        count = self.policy_proposal_count
        if count is not None and (type(count) is not int or (count != -1 and count <= 0)):
            raise ValueError("policy_proposal_count must be -1 or a positive integer when specified.")
        value_search_names = {"rl_value_beam_search", "rl_value_best_first",
                              "rl_metric_value_beam_search", "rl_metric_value_best_first"}
        metric_value_only = set(self.algorithms) <= value_search_names
        value_upper_bound = math.inf if metric_value_only else 1.0
        if not math.isfinite(self.value_discount) or not 0 <= self.value_discount <= value_upper_bound:
            raise ValueError(f"value_discount must be finite and in [0, {value_upper_bound}].")
        if type(self.force_cpu) is not bool or type(self.profile_cuda_timing) is not bool:
            raise ValueError("force_cpu and profile_cuda_timing must be booleans.")
        if type(self.skip_insufficient_starts) is not bool:
            raise ValueError("skip_insufficient_starts must be a boolean.")
        if self.polytope_file is not None:
            if not isinstance(self.polytope_file, str) or not self.polytope_file.strip():
                raise ValueError("polytope_file must be a nonempty path when specified.")
            if self.skip_insufficient_starts:
                raise ValueError("polytope_file cannot use skip_insufficient_starts; supplied polytopes must be retained.")
        for name in ("ga_population_size", "ga_max_stalled_generations", "ga_face_max_points", "ga_face_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.ga_population_size < 4:
            raise ValueError("ga_population_size must be at least 4 (cyopt requirement).")
        if type(self.ga_elitism) is not int or not 1 <= self.ga_elitism < self.ga_population_size:
            raise ValueError("ga_elitism must be in [1, ga_population_size) to retain the feasible start.")
        if not math.isfinite(self.ga_mutation_rate) or not 0 <= self.ga_mutation_rate <= 1:
            raise ValueError("ga_mutation_rate must be finite and in [0, 1].")
        if "cyopt_ga" in self.algorithms and self.reward_function != "max_kcup":
            raise ValueError("cyopt_ga currently supports max_kcup, an objective on 2-face triangulation classes.")
        if (set(self.algorithms) & value_search_names
                and self.reward_function != "max_kcup"):
            raise ValueError("Metric-value search currently supports only max_kcup, using natural log volume.")
        from models.subcomplex_policy_config import normalize_subcomplex_actor_type

        normalize_subcomplex_actor_type(self.subcomplex_actor_type)
        if self.policy_observation_kind == "two_face" and (self.reward_function != "max_kcup" or self.in_channels != 4):
            raise ValueError("two_face_deep_sets evaluation requires 4D max_kcup data.")
        from reward_functions import infer_goal

        infer_goal(self.reward_function)

    @property
    def policy_observation_kind(self) -> str:
        from models.subcomplex_policy_config import observation_kind_for_subcomplex_actor

        return observation_kind_for_subcomplex_actor(self.subcomplex_actor_type)

    @property
    def resolved_policy_proposal_count(self) -> int:
        """Legacy value-beam default; BeFS resolves its default independently."""
        return self.beam_width if self.policy_proposal_count is None else self.policy_proposal_count

    def setup_parameters(self) -> dict:
        """Exclude search settings so the same starts can serve multiple budgets."""
        parameters = {
            "num_polytopes": self.num_polytopes, "h11": self.h11,
            "num_starts": self.num_starts, "seed": self.seed,
            "num_vertices": self.num_vertices, "favorable": self.favorable,
            "hf_revision": self.hf_revision, "neighbor_mode": "two_neighbors",
            "sampler": "fast", "include_points_interior_to_facets": False,
        }
        # Preserve the checksum/validation contract of existing strict v1 setups.
        if self.skip_insufficient_starts:
            parameters["skip_insufficient_starts"] = True
        if self.polytope_file is not None:
            parameters["polytope_file"] = str(Path(self.polytope_file).expanduser().resolve())
        return parameters

    def to_dict(self) -> dict:
        return asdict(self)


def derive_seed(seed: int, *components: str | int) -> int:
    """Stable across processes, algorithm order, and Python hash seeds."""
    encoded = json.dumps([seed, *components], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")


def load_evaluation_config(path: str | Path) -> dict:
    """Read a partial EvaluationSpec; CLI overrides are applied by the caller."""
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation config must be a JSON object.")
    annotations = get_type_hints(EvaluationSpec)
    unknown = payload.keys() - annotations.keys()
    if unknown:
        raise ValueError(f"Unknown evaluation config fields: {', '.join(sorted(unknown))}.")
    for name, value in payload.items():
        annotation = annotations[name]
        variants = get_args(annotation) if get_origin(annotation) is UnionType else (annotation,)
        valid = False
        for expected in variants:
            if get_origin(expected) is tuple:
                valid |= isinstance(value, list) and all(type(item) is str for item in value)
            elif expected is float:
                valid |= type(value) in (int, float)
            else:
                valid |= type(value) is expected
        if not valid:
            raise ValueError(f"Invalid JSON type for {name}: expected {annotation}, got {value!r}.")
    return payload
