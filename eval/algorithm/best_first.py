"""Best first graph search ranked by the state objective."""

import heapq

from eval.algorithm.base import SearchContext


class BestFirstAlgorithm:
    name = "best_first"

    def search(self, context: SearchContext) -> None:
        frontier = [(*context.priority(context.initial), context.initial)]
        while frontier and context.remaining_budget > 0:
            _, _, node = heapq.heappop(frontier)
            for candidate in context.expand(node):
                heapq.heappush(frontier, (*context.priority(candidate), candidate))
