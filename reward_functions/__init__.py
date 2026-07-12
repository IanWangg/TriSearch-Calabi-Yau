"""Registry for triangulation training rewards and evaluation objectives."""

from typing import Callable, Literal

from reward_functions.common import Reward
from reward_functions.min_tri import MinTriangulationReward
from reward_functions.max_tri import MaxTriangulationReward

Goal = Literal["min", "max"]
SUPPORTED_REWARDS = ("min_tri", "max_tri")


def _normalize_reward_name(name: str) -> str:
    return str(name).strip().lower()


def _validate_reward_name(name: str) -> str:
    normalized = _normalize_reward_name(name)
    if normalized not in SUPPORTED_REWARDS:
        raise ValueError(
            f"Unknown reward function '{name}'. "
            f"Available options: {', '.join(SUPPORTED_REWARDS)}"
        )
    return normalized


def get_reward(name: str) -> Reward:
    normalized = _validate_reward_name(name)
    if normalized == "min_tri":
        return MinTriangulationReward()
    if normalized == "max_tri":
        return MaxTriangulationReward()
    raise AssertionError(f"Unhandled reward function '{normalized}'.")


def get_objective(name: str) -> Callable:
    normalized = _validate_reward_name(name)
    if normalized == "min_tri":
        return lambda state: float(len(state.simplices))
    if normalized == "max_tri":
        return lambda state: float(len(state.simplices))
    raise AssertionError(f"Unhandled objective '{normalized}'.")


def infer_goal(name: str) -> Goal:
    normalized = _validate_reward_name(name)
    if "min_" in normalized:
        return "min"
    if "max_" in normalized:
        return "max"
    raise AssertionError(f"Unhandled objective goal '{normalized}'.")
