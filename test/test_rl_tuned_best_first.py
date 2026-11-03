import math

import numpy as np

from eval.algorithm.rl import RLTunedBestFirst


def test_actor_mode_takes_top_unseen():
    algorithm = RLTunedBestFirst(2, 0.9)
    keys = ["a", "b", "c", "d"]
    logp = np.log([0.1, 0.4, 0.3, 0.2])
    assert algorithm.propose(keys, logp, {"b"}, np.random.default_rng(0)) == [2, 3]


def test_random_mode_ignores_actor_and_is_seeded():
    keys = [str(i) for i in range(10)]
    logp = np.zeros(10)
    a = RLTunedBestFirst(3, 0.0, proposal_mode="random", proposal_seed=5)
    b = RLTunedBestFirst(3, 0.0, proposal_mode="random", proposal_seed=5)
    picked = a.propose(keys, logp, {"0"}, np.random.default_rng(0))
    assert picked == b.propose(keys, logp, {"0"}, np.random.default_rng(1))
    assert len(picked) == 3 and 0 not in picked


def test_score_adds_logp_term():
    algorithm = RLTunedBestFirst(4, 0.5, logp_coef=2.0)
    score = algorithm.score_candidate(0.0, -1.0, 1.0, math.e, None, 3.0)
    assert math.isclose(score, 1.0 + 0.5 * 3.0 - 2.0)
    assert algorithm.name == "rl_value_best_first" and algorithm.requires_policy
