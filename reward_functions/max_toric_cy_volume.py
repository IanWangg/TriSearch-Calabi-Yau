import math
from typing import TYPE_CHECKING, Any, Dict

from reward_functions.common import Reward
from core.cy_bounded_cache import BoundedLRU

if TYPE_CHECKING:
    from mdp.cy_triangulation_state import CYTriangulationState
else:
    CYTriangulationState = Any


class MaxToricCYVolumeReward(Reward):
    """Maximize log10 CY volume at the stretched toric Kahler-cone tip."""

    reward_name = "max_toric_cy_volume"

    def __init__(self) -> None:
        self._objective_by_state_key = BoundedLRU(max_bytes=16 * 1024**2, max_entries=8192)

    def metric(self, state: CYTriangulationState) -> float:
        provider = getattr(state, "objective_value", None)
        if callable(provider):
            return float(provider(self.reward_name))
        state_key = str(state.key)
        cached_objective = self._objective_by_state_key.get(state_key)
        if cached_objective is not None:
            return cached_objective

        triangulation = getattr(state, "cy_triangulation", None)
        if triangulation is None:
            raise ValueError(
                "max_toric_cy_volume requires a materialized CYTools triangulation."
            )

        cy = triangulation.get_cy()
        dimension = int(cy.dimension())
        if dimension != 3:
            raise ValueError(
                "max_toric_cy_volume requires a Calabi-Yau threefold, "
                f"but CYTools returned dimension {dimension}."
            )

        tip = cy.toric_kahler_cone().tip_of_stretched_cone(c=1)
        volume = float(cy.compute_cy_volume(tip))
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError(
                "max_toric_cy_volume requires a finite, strictly positive CY volume, "
                f"but got {volume} for state '{state.key}'."
            )

        objective = math.log10(volume)
        self._objective_by_state_key[state_key] = objective
        return objective

    def __call__(
        self,
        state: CYTriangulationState,
        next_state: CYTriangulationState,
    ) -> float:
        return self.metric(next_state) - self.metric(state)
