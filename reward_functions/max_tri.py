from typing import TYPE_CHECKING, Any

from reward_functions.common import Reward

if TYPE_CHECKING:
    from mdp.triangulation_state import TriangulationState
else:
    TriangulationState = Any


class MaxTriangulationReward(Reward):
    reward_name = "max_tri"

    def metric(self, state: TriangulationState) -> float:
        return float(len(state.simplices))

    def __call__(
        self,
        state: TriangulationState,
        next_state: TriangulationState,
    ) -> float:
        return self.metric(next_state) - self.metric(state)
