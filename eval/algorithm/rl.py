"""RL selection rules; geometry, accounting and batching live in the runner."""

from __future__ import annotations

import math

import numpy as np


class RLAlgorithm:
    is_walk = False
    is_best_first = False
    requires_policy = True
    requires_values = False
    supported_objectives = None

    def __init__(self, beam_width: int = 4):
        if type(beam_width) is not int or beam_width <= 0:
            raise ValueError("beam_width must be a positive integer.")
        self.beam_width = beam_width
        self.proposal_count = beam_width

    def propose(self, next_keys, log_probabilities, seen, rng, reserved=()):
        """Top proposals after search-local deduplication, returned canonically."""
        order = (range(len(next_keys)) if not self.requires_policy else
                 sorted(range(len(next_keys)), key=lambda i: (-log_probabilities[i], i)))
        selected, targets = [], set()
        for index in order:
            key = next_keys[index]
            if key in seen or key in reserved or key in targets:
                continue
            selected.append(index)
            targets.add(key)
            if self.proposal_count != -1 and len(selected) >= self.proposal_count:
                break
        return sorted(selected)

    def score(self, parent_score, log_probability, reward, value):
        raise NotImplementedError

    def score_candidate(self, parent_score, log_probability, parent_objective, objective, reward, value):
        """Score recorded objectives; preserve the original four-argument score hook."""
        transition_reward = reward.from_objectives(parent_objective, objective) if self.requires_values else 0.0
        return self.score(parent_score, log_probability, transition_reward, value)


class RLStochasticPolicy(RLAlgorithm):
    name = "rl_stochastic_policy"
    is_walk = True

    def __init__(self):
        super().__init__(beam_width=1)

    def propose(self, next_keys, log_probabilities, seen, rng, reserved=()):
        available = [i for i, key in enumerate(next_keys) if key not in seen and key not in reserved]
        if not available:
            return []
        # Normalize only after masking visited targets. Subtracting the maximum
        # avoids underflow when all remaining actions had tiny original mass.
        logits = np.asarray(log_probabilities, dtype=np.float64)[available]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return [available[int(rng.choice(len(available), p=probabilities))]]

    def score(self, parent_score, log_probability, reward, value):
        return 0.0


class RLPolicyBeamSearch(RLAlgorithm):
    name = "rl_policy_beam_search"

    def score(self, parent_score, log_probability, reward, value):
        return parent_score + log_probability


class RLValueBeamSearch(RLAlgorithm):
    """Rank kcup candidates by absolute log volume plus weighted critic value."""

    name = "rl_value_beam_search"
    requires_values = True
    supported_objectives = ("max_kcup", "min_mori_cap")
    value_score_definition = "ln_objective_plus_discounted_value"
    value_discount_upper_bound = math.inf

    def __init__(self, beam_width: int = 4, policy_proposal_count: int | None = None,
                 value_discount: float = 0.9):
        super().__init__(beam_width)
        count = beam_width if policy_proposal_count is None else policy_proposal_count
        if type(count) is not int or (count != -1 and count <= 0):
            raise ValueError("policy_proposal_count must be -1 or a positive integer.")
        if not math.isfinite(value_discount) or not 0 <= value_discount <= self.value_discount_upper_bound:
            raise ValueError(f"value_discount must be finite and in [0, {self.value_discount_upper_bound}].")
        self.proposal_count = count
        self.value_discount = value_discount
        self.requires_policy = count != -1

    def score_candidate(self, parent_score, log_probability, parent_objective, objective, reward, value):
        metric = -objective if getattr(reward, "reward_name", None) == "min_mori_cap" else math.log(objective)
        return metric + self.value_discount * value


class RLValueBestFirst(RLValueBeamSearch):
    """Absolute log metric plus critic scores on a persistent global frontier."""

    name = "rl_value_best_first"
    is_best_first = True

    def __init__(self, policy_proposal_count: int | None = None, value_discount: float = 0.9):
        # One parent per scheduling round; proposal count is independent of beam width.
        super().__init__(beam_width=1,
                         policy_proposal_count=4 if policy_proposal_count is None else policy_proposal_count,
                         value_discount=value_discount)


class RLTunedBestFirst(RLValueBestFirst):
    """RL BeFS with inference-time knobs for tuning; registered name is unchanged.

    proposal_mode: "actor" (top-k by policy), "random" (k uniform unseen
    neighbors, a control for the actor ranking) or "canonical" (first k).
    score = ln V + value_discount * critic + logp_coef * log pi(edge).
    widen_penalty: if not None, an expanded parent with unproposed unseen
    neighbors returns to the frontier with score - widen_penalty, so its next
    proposals can be queried later (progressive widening).
    """

    def __init__(self, policy_proposal_count: int = 4, value_discount: float = 0.9, *,
                 proposal_mode: str = "actor", logp_coef: float = 0.0, widen_penalty: float | None = None):
        if proposal_mode not in ("actor", "random", "canonical"):
            raise ValueError("proposal_mode must be actor, random or canonical.")
        if not math.isfinite(logp_coef) or (widen_penalty is not None and not math.isfinite(widen_penalty)):
            raise ValueError("logp_coef and widen_penalty must be finite.")
        super().__init__(policy_proposal_count=policy_proposal_count, value_discount=value_discount)
        self.proposal_mode = proposal_mode
        self.logp_coef = logp_coef
        self.widen_penalty = widen_penalty
        # Keep actor logits for logging and logp_coef even when all neighbors are proposed.
        self.requires_policy = True

    def propose(self, next_keys, log_probabilities, seen, rng, reserved=()):
        if self.proposal_mode == "actor":
            order = sorted(range(len(next_keys)), key=lambda i: (-log_probabilities[i], i))
        elif self.proposal_mode == "random":
            order = [int(i) for i in rng.permutation(len(next_keys))]
        else:
            order = range(len(next_keys))
        selected, targets = [], set()
        for index in order:
            key = next_keys[index]
            if key in seen or key in reserved or key in targets:
                continue
            selected.append(index)
            targets.add(key)
            if self.proposal_count != -1 and len(selected) >= self.proposal_count:
                break
        return sorted(selected)

    def score_candidate(self, parent_score, log_probability, parent_objective, objective, reward, value):
        base = super().score_candidate(parent_score, log_probability, parent_objective, objective, reward, value)
        return base + self.logp_coef * float(log_probability)

    def describe(self):
        return dict(policy_proposal_count=self.proposal_count, value_discount=self.value_discount,
                    proposal_mode=self.proposal_mode, logp_coef=self.logp_coef,
                    widen_penalty=self.widen_penalty)


class RLMetricValueBeamSearch(RLValueBeamSearch):
    """Compatibility name for RL value beam search."""

    name = "rl_metric_value_beam_search"


class RLMetricValueBestFirst(RLValueBestFirst):
    """Compatibility name for RL value best-first search."""

    name = "rl_metric_value_best_first"


RL_ALGORITHM_NAMES = (RLStochasticPolicy.name, RLPolicyBeamSearch.name, RLValueBeamSearch.name,
                      RLValueBestFirst.name, RLMetricValueBeamSearch.name, RLMetricValueBestFirst.name)
