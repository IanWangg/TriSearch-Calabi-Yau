from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.cy_training_config import parse_args
from reward_functions import get_objective, get_reward, infer_goal


def _state(key, count):
    cap = Mock()
    cap.extremal_rays.return_value = [[1, 0]] * count
    cy = Mock()
    cy.mori_cone_cap.return_value = cap
    state = SimpleNamespace(key=key, cy_triangulation=Mock())
    state.cy_triangulation.get_cy.return_value = cy
    return state, cy, cap


@pytest.mark.parametrize("following_count", [0, 3, 5, 7])
def test_min_mori_cap_counts_extremal_rays_and_rewards_decreases(following_count):
    current, current_cy, current_cap = _state("current", 5)
    following, following_cy, following_cap = _state("following", following_count)
    reward = get_reward("min_mori_cap")
    objective = get_objective("min_mori_cap", reward=reward)

    assert infer_goal("min_mori_cap") == "min"
    assert objective(current) == 5.0
    assert objective(following) == float(following_count)
    assert reward(current, following) == 5.0 - following_count
    assert reward.from_objectives(5.0, following_count) == 5.0 - following_count
    # Repeated objectives and reward calls reuse cached geometry, including zero.
    for cy, cap in ((current_cy, current_cap), (following_cy, following_cap)):
        cy.mori_cone_cap.assert_called_once_with(in_basis=True)
        cap.extremal_rays.assert_called_once_with()


def test_min_mori_cap_uses_managed_objective_provider():
    provider = Mock(return_value=4.0)
    state = SimpleNamespace(objective_value=provider)

    assert get_objective("min_mori_cap")(state) == 4.0
    provider.assert_called_once_with("min_mori_cap")


def test_min_mori_cap_requires_geometry_or_provider():
    with pytest.raises(ValueError, match="materialized CYTools triangulation"):
        get_objective("min_mori_cap")(SimpleNamespace(key="missing"))


@pytest.mark.parametrize("flag", ["--reward", "--reward_function"])
def test_min_mori_cap_is_available_in_train_cli(flag):
    assert parse_args([flag, "min_mori_cap"]).reward_function == "min_mori_cap"
