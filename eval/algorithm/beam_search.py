"""Layer-wise beam search with a budget check after each complete parent."""

import heapq

from eval.algorithm.base import SearchContext


class BeamSearchAlgorithm:
    name = "beam_search"

    def __init__(self, beam_width: int = 4):
        if type(beam_width) is not int or beam_width <= 0:
            raise ValueError("beam_width must be a positive integer.")
        self.beam_width = beam_width

    def search(self, context: SearchContext) -> None:
        beam = [context.initial]
        while beam and context.remaining_budget > 0:
            def candidates():
                for parent in beam:
                    if context.remaining_budget <= 0:
                        break
                    yield from context.expand(parent)

            # nsmallest fully consumes each parent, retaining only k candidates.
            # The next beam is installed only after processing the current one.
            beam = heapq.nsmallest(self.beam_width, candidates(), key=context.priority)
