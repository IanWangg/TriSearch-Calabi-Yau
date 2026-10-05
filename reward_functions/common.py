"""Common reward interface for triangulation objectives."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mdp.triangulation_state import TriangulationState
else:
    TriangulationState = Any


class Reward:
    """Return positive reward when a transition improves the objective."""

    def from_objectives(self, current: float, following: float) -> float:
        """Compute reward from recorded registry objectives, without geometry queries."""
        return float(following - current)

    def __call__(
        self,
        state: TriangulationState,
        next_state: TriangulationState,
    ) -> float:
        raise NotImplementedError("Subclasses must implement __call__().")
