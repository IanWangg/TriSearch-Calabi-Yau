"""two_face_invariant actor: factory registration, checkpoint round trip, and the rollout observation path."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from core.cy_checkpointing import save_policy_checkpoint
from core.cy_policy_inference import build_cy_data_list, policy_build_kind, validate_policy_observations
from core.cy_runtime_utils import load_policy_checkpoint
from core.cy_two_face_data import create_two_face_data_from_state
from core.two_face_invariant_features import (
    InvariantTwoFaceData, add_invariant_fields, create_invariant_two_face_data_from_state,
    invariant_topology_cache_stats)
from eval.config import EvaluationSpec
from eval.policy import EvaluationPolicy
from models.invariant_two_face_agent import InvariantTwoFaceAgent
from models.subcomplex_policy_config import (
    normalize_subcomplex_actor_type, observation_kind_for_subcomplex_actor, value_feature_source_for_subcomplex_actor)
from models.subcomplex_policy_factory import build_subcomplex_agent
from test_cy_two_face_data import real_representatives  # noqa: F401  (fixture)
from mdp.cy_geometry_worker import execute_geometry_request


def build_invariant(seed=0, **overrides):
    torch.manual_seed(seed)
    kwargs = dict(model_type="egnn", subcomplex_actor_type="two_face_invariant", in_channels=4,
                  out_channels=16, hidden_channels=16, num_layers=2, two_face_encoder="deep_sets")
    kwargs.update(overrides)
    return build_subcomplex_agent(**kwargs).eval()


def test_factory_and_config_registration():
    assert normalize_subcomplex_actor_type("Two_Face_Invariant") == "two_face_invariant"
    assert observation_kind_for_subcomplex_actor("two_face_invariant") == "two_face"
    assert value_feature_source_for_subcomplex_actor("two_face_invariant") == "two_face"
    model = build_invariant()
    assert isinstance(model, InvariantTwoFaceAgent)
    assert model.model_config["subcomplex_actor_type"] == "two_face_invariant"
    assert model.model_config["encoder_variant"] == "gl4z_invariant_v1"
    assert policy_build_kind(model) == "two_face_invariant"
    plain = build_subcomplex_agent(model_type="egnn", subcomplex_actor_type="two_face_deep_sets", in_channels=4,
                                   out_channels=16, hidden_channels=16, num_layers=2)
    assert policy_build_kind(plain) == "two_face"  # default two_face path is unchanged


def test_checkpoint_round_trip_and_evaluation_policy_load(tmp_path):
    model = build_invariant(seed=3)
    checkpoint = tmp_path / "checkpoints" / "1.pth"
    save_policy_checkpoint(model, str(checkpoint))
    assert (checkpoint.parent / "model_config.json").is_file()
    restored = build_invariant(seed=11)
    load_policy_checkpoint(restored, str(checkpoint), map_location=torch.device("cpu"))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    # A mismatched architecture must be refused by the saved model_config.
    mismatched = build_subcomplex_agent(model_type="egnn", subcomplex_actor_type="two_face_invariant",
                                        in_channels=4, out_channels=16, hidden_channels=16, num_layers=3,
                                        two_face_encoder="deep_sets")
    with pytest.raises(ValueError, match="model_config mismatch"):
        load_policy_checkpoint(mismatched, str(checkpoint), map_location=torch.device("cpu"))
    spec = EvaluationSpec(1, 12, 1, 1, policy_checkpoint=str(checkpoint.parent), subcomplex_actor_type="two_face_invariant",
                          force_cpu=True, in_channels=4, hidden_channels=16, out_channels=16, num_layers=2)
    policy = EvaluationPolicy.from_spec(spec)
    assert isinstance(policy.model, InvariantTwoFaceAgent)
    assert policy.observation_kind == "two_face"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(policy.model.state_dict()[key].cpu(), value.cpu())


def test_rollout_observation_path_equals_invariant_reference(real_representatives):
    row, state, _ = real_representatives
    expansion = execute_geometry_request(dict(operation="expand", configuration=state.configuration,
                                              state=state.to_payload(), objective_mode=True,
                                              action_order="canonical"))
    actions = expansion.candidate_actions
    assert len(actions) > 0
    model = build_invariant(seed=5, hidden_channels=16)
    rollout_data = build_cy_data_list([state], [actions], observation_kind=policy_build_kind(model),
                                      subcomplex_width=4)
    validate_policy_observations(rollout_data, model)
    assert isinstance(rollout_data[0], InvariantTwoFaceData)
    reference = add_invariant_fields(create_two_face_data_from_state(state, actions))
    assert set(rollout_data[0].keys()) == set(reference.keys())
    for key in ("x", "edge_index", "triangle_vertices", "point_index", "point_edge_index", "point_x_inv",
                "face_x_inv", "subcomplex_vertices", "action_face", "removed_triangle_ids", "added_triangle_vertices"):
        assert torch.equal(rollout_data[0][key], reference[key]), key
    with torch.no_grad():
        value, logits = model.get_value_and_logits(Batch.from_data_list(rollout_data))
        ref_value, ref_logits = model.get_value_and_logits(Batch.from_data_list([reference]))
    torch.testing.assert_close(value, ref_value, atol=1e-6, rtol=0)
    torch.testing.assert_close(logits, ref_logits, atol=1e-6, rtol=0)
    # Second observation of the same topology is served from the invariant cache.
    before = invariant_topology_cache_stats()["hits"]
    create_invariant_two_face_data_from_state(state, actions[:1])
    assert invariant_topology_cache_stats()["hits"] == before + 1
