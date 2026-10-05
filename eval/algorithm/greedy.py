from eval.algorithm.base import EvaluatedAction, EvaluationContext


class GreedyAlgorithm:
    name = "greedy"

    def select_action(self, context: EvaluationContext) -> EvaluatedAction:
        candidates = (context.evaluate_action(action) for action in context.actions)
        # min/max retain the first candidate on ties, in canonical action order.
        choose = max if context.goal == "max" else min
        return choose(candidates, key=lambda candidate: candidate.objective)
