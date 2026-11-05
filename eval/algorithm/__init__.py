from eval.algorithm.base import (
    Algorithm, EvaluatedAction, EvaluationAlgorithm, EvaluationContext,
    FrontierSearchAlgorithm, SearchContext, SearchNode,
)
from eval.algorithm.beam_search import BeamSearchAlgorithm
from eval.algorithm.best_first import BestFirstAlgorithm
from eval.algorithm.greedy import GreedyAlgorithm
from eval.algorithm.random import RandomAlgorithm
from eval.algorithm.cyopt_ga import CyoptGAAlgorithm, CyoptGANoCrossAlgorithm
from eval.algorithm.face_mutation import FaceMutationBeFS, FaceMutationHillClimb
from eval.algorithm.face_mutation_guided import FaceMutationGuidedBeFS, GUIDED_NAME, RANDOM_NAME
from eval.algorithm.rl import (
    RL_ALGORITHM_NAMES, RLAlgorithm, RLStochasticPolicy, RLPolicyBeamSearch, RLValueBeamSearch, RLValueBestFirst,
    RLMetricValueBeamSearch, RLMetricValueBestFirst,
)


def get_algorithm(name: str, *, beam_width: int = 4, policy_proposal_count: int | None = None,
                  value_discount: float = 0.9, ga_population_size: int = 50,
                  ga_mutation_rate: float = 0.1, ga_elitism: int = 1,
                  ga_max_stalled_generations: int = 20, face_score_aggregation: str = "mean2") -> Algorithm:
    if name == "cyopt_ga":
        return CyoptGAAlgorithm(ga_population_size, ga_mutation_rate, ga_elitism, ga_max_stalled_generations)
    if name == "cyopt_ga_nocross":
        return CyoptGANoCrossAlgorithm(ga_population_size, ga_mutation_rate, ga_elitism, ga_max_stalled_generations)
    if name == "face_mutation_hc":
        return FaceMutationHillClimb()
    if name == "face_mutation_befs":
        return FaceMutationBeFS()
    if name == GUIDED_NAME:
        return FaceMutationGuidedBeFS(selection="score", aggregation=face_score_aggregation)
    if name == RANDOM_NAME:
        return FaceMutationGuidedBeFS(selection="random", aggregation=face_score_aggregation)
    if name == "rl_stochastic_policy":
        return RLStochasticPolicy()
    if name == "rl_policy_beam_search":
        return RLPolicyBeamSearch(beam_width)
    if name == "rl_value_beam_search":
        return RLValueBeamSearch(beam_width, policy_proposal_count, value_discount)
    if name == "rl_value_best_first":
        return RLValueBestFirst(policy_proposal_count, value_discount)
    if name == "rl_metric_value_beam_search":
        return RLMetricValueBeamSearch(beam_width, policy_proposal_count, value_discount)
    if name == "rl_metric_value_best_first":
        return RLMetricValueBestFirst(policy_proposal_count, value_discount)
    if name == "beam_search":
        return BeamSearchAlgorithm(beam_width)
    factories = {"random": RandomAlgorithm, "greedy": GreedyAlgorithm, "best_first": BestFirstAlgorithm}
    if name not in factories:
        raise ValueError(f"Unknown evaluation algorithm {name!r}; available: {(*factories, 'beam_search', 'cyopt_ga', 'cyopt_ga_nocross', 'face_mutation_hc', 'face_mutation_befs', GUIDED_NAME, RANDOM_NAME, *RL_ALGORITHM_NAMES)}.")
    return factories[name]()


__all__ = ["Algorithm", "EvaluationAlgorithm", "EvaluationContext", "EvaluatedAction",
           "FrontierSearchAlgorithm", "SearchContext", "SearchNode", "BestFirstAlgorithm",
           "BeamSearchAlgorithm", "RandomAlgorithm", "GreedyAlgorithm", "get_algorithm",
           "CyoptGAAlgorithm", "CyoptGANoCrossAlgorithm", "FaceMutationHillClimb", "FaceMutationBeFS", "FaceMutationGuidedBeFS",
           "RLAlgorithm", "RLStochasticPolicy", "RLPolicyBeamSearch", "RLValueBeamSearch", "RLValueBestFirst",
           "RLMetricValueBeamSearch", "RLMetricValueBestFirst", "RL_ALGORITHM_NAMES"]
