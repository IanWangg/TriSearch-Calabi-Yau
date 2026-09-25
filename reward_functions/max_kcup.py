import math
import warnings
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from reward_functions.common import Reward
from core.cy_bounded_cache import BoundedLRU

if TYPE_CHECKING:
    from mdp.cy_triangulation_state import CYTriangulationState
else:
    CYTriangulationState = Any


class MaxKcupReward(Reward):
    """Maximize log CY volume at the stretched Kcup cone tip."""

    reward_name = "max_kcup"

    def __init__(self) -> None:
        self._volume_by_state_key = BoundedLRU(max_bytes=16 * 1024**2, max_entries=8192)

    def metric(self, state: CYTriangulationState) -> float:
        provider = getattr(state, "objective_value", None)
        if callable(provider):
            return float(provider(self.reward_name))
        state_key = str(state.key)
        cached_volume = self._volume_by_state_key.get(state_key)
        if cached_volume is not None:
            return cached_volume

        triangulation = getattr(state, "cy_triangulation", None)
        if triangulation is None:
            raise ValueError("max_kcup requires a materialized CYTools triangulation.")

        cy = triangulation.get_cy()
        dimension = int(cy.dimension())
        if dimension != 3:
            raise ValueError(
                "max_kcup requires a Calabi-Yau threefold, "
                f"but CYTools returned dimension {dimension}."
            )

        kcup = cy.mori_cone_cap(in_basis=True).dual()
        tip = kcup.tip_of_stretched_cone(c=1, backend="mosek")
        if tip is None or not np.isfinite(tip).all():
            warnings.warn(
                f"max_kcup mosek tip solver failed for state '{state_key}'; retrying with osqp.",
                RuntimeWarning,
            )
            tip = kcup.tip_of_stretched_cone(c=1, backend="osqp")
        if tip is None or not np.isfinite(tip).all():
            warnings.warn(
                f"max_kcup tip solver failed for state '{state_key}'; retrying with cvxopt.",
                RuntimeWarning,
            )
            # CVXOPT solves the same minimum-norm QP, unlike the LP backends.
            tip = kcup.tip_of_stretched_cone(c=1, backend="cvxopt")
        if tip is None or not np.isfinite(tip).all():
            raise ValueError(f"max_kcup could not solve the stretched cone tip for state '{state_key}'.")
        volume = float(cy.compute_cy_volume(tip))
        if not math.isfinite(volume):
            raise ValueError(f"max_kcup returned a nonfinite volume for state '{state_key}'.")
        self._volume_by_state_key[state_key] = volume
        return volume

    def __call__(
        self,
        state: CYTriangulationState,
        next_state: CYTriangulationState,
    ) -> float:
        current_volume = self.metric(state)
        next_volume = self.metric(next_state)
        if current_volume <= 0.0 or next_volume <= 0.0:
            raise ValueError(
                "max_kcup log volume reward requires strictly positive volumes, "
                f"but got V_current={current_volume} for state '{state.key}' and "
                f"V_next={next_volume} for state '{next_state.key}'."
            )
        return math.log(next_volume) - math.log(current_volume)
