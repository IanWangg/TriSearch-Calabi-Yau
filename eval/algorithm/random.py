from eval.algorithm.base import EvaluatedAction, EvaluationContext


class RandomAlgorithm:
    name = "random"

    def select_action(self, context: EvaluationContext) -> EvaluatedAction:
        action = context.actions[int(context.rng.integers(len(context.actions)))]
        return context.evaluate_action(action)
