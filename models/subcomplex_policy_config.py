from __future__ import annotations


DEFAULT_SUBCOMPLEX_ACTOR_TYPE = "snn_simplex"
CANONICAL_SUBCOMPLEX_ACTOR_TYPES = (
    "mlp",
    "gnn",
    "circuit_pool",
    "snn_simplex",
    "two_face_deep_sets",
)
SUBCOMPLEX_ACTOR_TYPE_ALIASES = {
    "default": DEFAULT_SUBCOMPLEX_ACTOR_TYPE,
}
SUPPORTED_SUBCOMPLEX_ACTOR_TYPES = (
    *CANONICAL_SUBCOMPLEX_ACTOR_TYPES,
    *SUBCOMPLEX_ACTOR_TYPE_ALIASES,
)


def normalize_subcomplex_actor_type(subcomplex_actor_type: str) -> str:
    resolved_actor_type = str(subcomplex_actor_type).strip().lower()
    resolved_actor_type = SUBCOMPLEX_ACTOR_TYPE_ALIASES.get(
        resolved_actor_type,
        resolved_actor_type,
    )
    if resolved_actor_type not in CANONICAL_SUBCOMPLEX_ACTOR_TYPES:
        raise ValueError(
            f"Unsupported subcomplex_actor_type '{subcomplex_actor_type}'. "
            f"Expected one of: {', '.join(SUPPORTED_SUBCOMPLEX_ACTOR_TYPES)}."
        )
    return resolved_actor_type


def value_feature_source_for_subcomplex_actor(subcomplex_actor_type: str) -> str:
    resolved_actor_type = normalize_subcomplex_actor_type(subcomplex_actor_type)
    if resolved_actor_type == "two_face_deep_sets":
        return "two_face"
    return "snn_simplex" if resolved_actor_type == "snn_simplex" else "egnn"


def observation_kind_for_subcomplex_actor(subcomplex_actor_type: str) -> str:
    return ("two_face" if normalize_subcomplex_actor_type(subcomplex_actor_type) == "two_face_deep_sets"
            else "full_triangulation")
