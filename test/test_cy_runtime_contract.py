from pathlib import Path

from core.cy_runtime_utils import load_policy_checkpoint
from core.train_cy import parse_args as parse_train_args
from models.egnn_subcomplex_predictor import EGNNSubcomplexAgent
from models.gcn_subcomplex_predictor import GCNSubcomplexAgent
from models.subcomplex_policy_config import (
    DEFAULT_SUBCOMPLEX_ACTOR_TYPE,
    normalize_subcomplex_actor_type,
)
from models.subcomplex_policy_factory import build_subcomplex_agent


def _build_agent(actor_type: str | None = None):
    kwargs = {}
    if actor_type is not None:
        kwargs["subcomplex_actor_type"] = actor_type
    return build_subcomplex_agent(
        model_type="egnn",
        in_channels=4,
        out_channels=64,
        hidden_channels=64,
        num_layers=3,
        mlp_hidden_channel_list=[64],
        device="cpu",
        **kwargs,
    )


def test_snn_simplex_is_the_default_across_public_policy_apis():
    assert DEFAULT_SUBCOMPLEX_ACTOR_TYPE == "snn_simplex"
    assert normalize_subcomplex_actor_type("default") == "snn_simplex"
    assert parse_train_args([]).subcomplex_actor_type == "snn_simplex"

    egnn = EGNNSubcomplexAgent(in_channels=4, out_channels=8, hidden_channels=8)
    gcn = GCNSubcomplexAgent(in_channels=4, out_channels=8, hidden_channels=8)
    factory_agent = build_subcomplex_agent(
        model_type="egnn",
        in_channels=4,
        out_channels=8,
        hidden_channels=8,
        num_layers=1,
    )

    for policy in (egnn, gcn, factory_agent):
        assert policy.subcomplex_actor_type == "snn_simplex"
        assert policy.value_feature_source == "snn_simplex"


def test_explicit_gnn_and_snn_bundled_checkpoints_remain_loadable():
    repo_root = Path(__file__).resolve().parents[1]
    snn_policy = _build_agent()
    gnn_policy = _build_agent("gnn")

    load_policy_checkpoint(
        snn_policy,
        str(repo_root / "ckpt/cy_train_h15_snn/latest.pth"),
        map_location="cpu",
    )
    load_policy_checkpoint(
        gnn_policy,
        str(repo_root / "ckpt/cy_train_h15/latest.pth"),
        map_location="cpu",
    )

    assert snn_policy.subcomplex_actor_type == "snn_simplex"
    assert gnn_policy.subcomplex_actor_type == "gnn"


