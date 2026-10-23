from typing import TYPE_CHECKING, Any

from core.cy_bounded_cache import BoundedLRU
from reward_functions.common import Reward

if TYPE_CHECKING:
    from mdp.cy_triangulation_state import CYTriangulationState
else:
    CYTriangulationState = Any


def mori_cap_target(cy) -> float:
    cap = cy.mori_cone_cap(in_basis=True)
    return float(len(cap.extremal_rays()))


class MinMoriCapReward(Reward):
    """Minimize the number of extremal rays of the Mori cone cap."""

    reward_name = "min_mori_cap"

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
            raise ValueError("min_mori_cap requires a materialized CYTools triangulation.")

        objective = mori_cap_target(triangulation.get_cy())
        self._objective_by_state_key[state_key] = objective
        return objective

    def from_objectives(self, current: float, following: float) -> float:
        return float(current - following)

    def __call__(
        self,
        state: CYTriangulationState,
        next_state: CYTriangulationState,
    ) -> float:
        return self.from_objectives(self.metric(state), self.metric(next_state))
