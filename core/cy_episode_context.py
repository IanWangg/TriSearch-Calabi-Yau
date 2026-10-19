"""Per-episode context and reward; never stored in geometry caches."""

from dataclasses import dataclass
import math

STEP_LOG_VOLUME = "step_log_volume"
BEST_SO_FAR_LOG_VOLUME = "best_so_far_log_volume"
TRAINING_REWARD_MODES = (STEP_LOG_VOLUME, BEST_SO_FAR_LOG_VOLUME)
CONTEXT_FIELDS = ("log_current_volume", "log_best_gap", "log_remaining_steps")


def log_volume(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Episode volume must be finite and positive.")
    return math.log(value)


@dataclass(frozen=True)
class EpisodeContext:
    current_volume: float
    best_volume: float
    remaining_steps: int

    def features(self):
        current, best = log_volume(self.current_volume), log_volume(self.best_volume)
        if best < current:
            raise ValueError("Episode best_volume cannot be smaller than current_volume.")
        if type(self.remaining_steps) is not int or self.remaining_steps < 0:
            raise ValueError("remaining_steps must be a non-negative integer.")
        return (current, best - current, math.log1p(self.remaining_steps))


def best_so_far_reward(best_volume, next_volume):
    previous = log_volume(best_volume)
    log_volume(next_volume)
    best = max(float(best_volume), float(next_volume))
    return best, log_volume(best) - previous
