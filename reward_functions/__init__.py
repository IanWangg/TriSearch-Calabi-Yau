"""Registry for triangulation training rewards and evaluation objectives."""

from typing import Callable, Literal

from reward_functions.common import Reward
from reward_functions.min_tri import MinTriangulationReward
from reward_functions.max_tri import MaxTriangulationReward
from reward_functions.max_cy_volume import MaxCYVolumeReward

Goal = Literal["min", "max"]
SUPPORTED_REWARDS = ("min_tri", "max_tri", "max_cy_volume")
CY_VOLUME_REWARD_TRANSFORMS = MaxCYVolumeReward.supported_transforms


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


def get_reward(name: str, *, cy_volume_reward_transform: str = "raw") -> Reward:
    normalized = _validate_reward_name(name)
    resolved_transform = str(cy_volume_reward_transform).strip().lower()
    if normalized != "max_cy_volume" and resolved_transform != "raw":
        raise ValueError(
            "cy_volume_reward_transform is only valid with the max_cy_volume reward."
        )
    if normalized == "min_tri":
        return MinTriangulationReward()
    if normalized == "max_tri":
        return MaxTriangulationReward()
    if normalized == "max_cy_volume":
        return MaxCYVolumeReward(transform=resolved_transform)
    raise AssertionError(f"Unhandled reward function '{normalized}'.")


def get_objective(name: str, *, reward: Reward | None = None) -> Callable:
    normalized = _validate_reward_name(name)
    if normalized == "min_tri":
        return lambda state: float(len(state.simplices))
    if normalized == "max_tri":
        return lambda state: float(len(state.simplices))
    if normalized == "max_cy_volume":
        volume_reward = reward if reward is not None else MaxCYVolumeReward()
        if not isinstance(volume_reward, MaxCYVolumeReward):
            raise TypeError(
                "max_cy_volume objective requires a MaxCYVolumeReward instance."
            )
        return volume_reward.metric
    raise AssertionError(f"Unhandled objective '{normalized}'.")


def infer_goal(name: str) -> Goal:
    normalized = _validate_reward_name(name)
    if "min_" in normalized:
        return "min"
    if "max_" in normalized:
        return "max"
    raise AssertionError(f"Unhandled objective goal '{normalized}'.")
