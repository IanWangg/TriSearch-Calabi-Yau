from __future__ import annotations

from typing import Type

import torch.nn as nn

from .egnn_subcomplex_predictor import EGNNSubcomplexAgent
from .gcn_subcomplex_predictor import GCNSubcomplexAgent


SUPPORTED_SUBCOMPLEX_MODEL_TYPES = ("egnn", "gcn")
SUPPORTED_SUBCOMPLEX_ACTOR_TYPES = EGNNSubcomplexAgent.SUPPORTED_SUBCOMPLEX_ACTOR_TYPES
SUBCOMPLEX_ACTOR_TYPE_ALIASES = {"default": "gnn"}


def normalize_subcomplex_model_type(model_type: str) -> str:
    resolved_model_type = str(model_type).strip().lower()
    if resolved_model_type not in SUPPORTED_SUBCOMPLEX_MODEL_TYPES:
        raise ValueError(
            f"Unsupported subcomplex model_type '{model_type}'. "
            f"Expected one of: {', '.join(SUPPORTED_SUBCOMPLEX_MODEL_TYPES)}."
        )
    return resolved_model_type


def normalize_subcomplex_actor_type(subcomplex_actor_type: str) -> str:
    resolved_actor_type = str(subcomplex_actor_type).strip().lower()
    resolved_actor_type = SUBCOMPLEX_ACTOR_TYPE_ALIASES.get(
        resolved_actor_type,
        resolved_actor_type,
    )
    if resolved_actor_type not in SUPPORTED_SUBCOMPLEX_ACTOR_TYPES:
        raise ValueError(
            f"Unsupported subcomplex_actor_type '{subcomplex_actor_type}'. "
            f"Expected one of: {', '.join(SUPPORTED_SUBCOMPLEX_ACTOR_TYPES)}."
        )
    return resolved_actor_type


def get_subcomplex_agent_class(model_type: str) -> Type[nn.Module]:
    resolved_model_type = normalize_subcomplex_model_type(model_type)
    if resolved_model_type == "egnn":
        return EGNNSubcomplexAgent
    if resolved_model_type == "gcn":
        return GCNSubcomplexAgent
    raise AssertionError(f"Unhandled subcomplex model_type '{resolved_model_type}'.")


def build_subcomplex_agent(
    *,
    model_type: str,
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    num_layers: int,
    subcomplex_decoder_num_layers: int = 2,
    share_encoder: bool = True,
    mlp_hidden_channel_list=None,
    use_projection: bool = True,
    act: str = "silu",
    subcomplex_actor_type: str = "gnn",
    device="cpu",
) -> nn.Module:
    if mlp_hidden_channel_list is None:
        mlp_hidden_channel_list = [64]
    resolved_model_type = normalize_subcomplex_model_type(model_type)
    resolved_actor_type = normalize_subcomplex_actor_type(subcomplex_actor_type)

    agent_cls = get_subcomplex_agent_class(resolved_model_type)
    kwargs = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        share_encoder=share_encoder,
        mlp_hidden_channel_list=mlp_hidden_channel_list,
        use_projection=use_projection,
        act=act,
        device=device,
    )
    if resolved_model_type == "egnn":
        kwargs["subcomplex_decoder_num_layers"] = subcomplex_decoder_num_layers
    kwargs["subcomplex_actor_type"] = resolved_actor_type
    return agent_cls(**kwargs)
