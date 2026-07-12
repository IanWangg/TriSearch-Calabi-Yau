"""Common reward interface for triangulation objectives."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mdp.triangulation_state import TriangulationState
else:
    TriangulationState = Any


class Reward:
    """Return positive reward when a transition improves the objective."""

    def __call__(
        self,
        state: TriangulationState,
        next_state: TriangulationState,
    ) -> float:
        raise NotImplementedError("Subclasses must implement __call__().")
