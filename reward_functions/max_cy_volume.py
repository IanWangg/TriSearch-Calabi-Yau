import math
from typing import TYPE_CHECKING, Any, Dict

from reward_functions.common import Reward

if TYPE_CHECKING:
    from mdp.cy_triangulation_state import CYTriangulationState
else:
    CYTriangulationState = Any


class MaxCYVolumeReward(Reward):
    """Maximize the CY threefold volume at the stretched Kcup cone tip."""

    reward_name = "max_cy_volume"

    supported_transforms = ("raw", "log")

    def __init__(self, transform: str = "raw") -> None:
        resolved_transform = str(transform).strip().lower()
        if resolved_transform not in self.supported_transforms:
            raise ValueError(
                f"Unknown CY volume reward transform '{transform}'. "
                f"Expected one of: {', '.join(self.supported_transforms)}."
            )
        self.transform = resolved_transform
        self._volume_by_state_key: Dict[str, float] = {}

    def metric(self, state: CYTriangulationState) -> float:
        state_key = str(state.key)
        cached_volume = self._volume_by_state_key.get(state_key)
        if cached_volume is not None:
            return cached_volume

        triangulation = getattr(state, "cy_triangulation", None)
        if triangulation is None:
            raise ValueError(
                "max_cy_volume requires a materialized CYTools triangulation."
            )

        cy = triangulation.get_cy()
        dimension = int(cy.dimension())
        if dimension != 3:
            raise ValueError(
                "max_cy_volume requires a Calabi-Yau threefold, "
                f"but CYTools returned dimension {dimension}."
            )

        stretched_cone = cy.mori_cone_cap(in_basis=True).dual()
        tip = stretched_cone.tip_of_stretched_cone(c=1)
        volume = float(cy.compute_cy_volume(tip))
        self._volume_by_state_key[state_key] = volume
        return volume

    def __call__(
        self,
        state: CYTriangulationState,
        next_state: CYTriangulationState,
    ) -> float:
        current_volume = self.metric(state)
        next_volume = self.metric(next_state)
        if self.transform == "raw":
            return next_volume - current_volume

        if current_volume <= 0.0 or next_volume <= 0.0:
            raise ValueError(
                "log CY volume reward requires strictly positive volumes, "
                f"but got V_current={current_volume} for state '{state.key}' and "
                f"V_next={next_volume} for state '{next_state.key}'."
            )
        return math.log(next_volume) - math.log(current_volume)
