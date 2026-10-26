from typing import TYPE_CHECKING, Any

from core.cy_bounded_cache import BoundedLRU
from reward_functions.common import Reward

if TYPE_CHECKING:
    from mdp.cy_triangulation_state import CYTriangulationState
else:
    CYTriangulationState = Any

MORI_CAP_BACKENDS = ("cytools", "mosek")


def mori_cap_target(cy, *, backend="cytools") -> float:
    if backend not in MORI_CAP_BACKENDS:
        raise ValueError(f"Unknown Mori cap backend {backend!r}; expected {MORI_CAP_BACKENDS}.")
    cap = cy.mori_cone_cap(in_basis=True)
    if backend == "mosek":
        from core.mori_extremal_rays import NonPointedConeError, mosek_extremal_rays

        try:
            return float(len(mosek_extremal_rays(cap.rays())))
        except NonPointedConeError:
            # Preserve minimal=True semantics for cones with lineality. Solver
            # failures and unverified certificates are deliberately not caught.
            pass
    return float(len(cap.extremal_rays()))


class MinMoriCapReward(Reward):
    """Minimize the number of extremal rays of the Mori cone cap."""

    reward_name = "min_mori_cap"

    def __init__(self, *, backend="cytools") -> None:
        if backend not in MORI_CAP_BACKENDS:
            raise ValueError(f"Unknown Mori cap backend {backend!r}; expected {MORI_CAP_BACKENDS}.")
        self.mori_cap_backend = backend
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

        objective = mori_cap_target(triangulation.get_cy(), backend=self.mori_cap_backend)
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
