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
SUPPORTED_TWO_FACE_ENCODERS = ("deep_sets", "triangle_dual")


def validate_two_face_encoder_options(actor_type, encoder, triangle_layers):
    if encoder is not None and encoder not in SUPPORTED_TWO_FACE_ENCODERS:
        raise ValueError(f"Unsupported two_face_encoder: {encoder!r}.")
    if triangle_layers is not None and (type(triangle_layers) is not int or triangle_layers <= 0):
        raise ValueError("two_face_triangle_layers must be a positive integer.")
    if normalize_subcomplex_actor_type(actor_type) != "two_face_deep_sets":
        if encoder is not None or triangle_layers is not None:
            raise ValueError("two_face encoder options require two_face_deep_sets.")
    if encoder == "deep_sets" and triangle_layers is not None:
        raise ValueError("two_face_triangle_layers requires triangle_dual.")


def resolve_two_face_encoder_options(encoder=None, triangle_layers=None, *, saved_config=None):
    """Resolve explicit flags or legacy checkpoint metadata without inspecting weights."""
    validate_two_face_encoder_options("two_face_deep_sets", encoder, triangle_layers)
    saved = saved_config or {}
    saved_encoder = saved.get("two_face_encoder", "triangle_dual" if "triangle_num_layers" in saved else "deep_sets")
    resolved_encoder = encoder if encoder is not None else saved_encoder
    resolved_layers = triangle_layers
    if resolved_encoder == "triangle_dual" and resolved_layers is None:
        resolved_layers = saved.get("triangle_num_layers", 2)
    validate_two_face_encoder_options("two_face_deep_sets", resolved_encoder, resolved_layers)
    if saved and (resolved_encoder != saved_encoder or (
            resolved_encoder == "triangle_dual" and resolved_layers != saved.get("triangle_num_layers"))):
        raise ValueError("Checkpoint model_config mismatch: explicit two_face encoder options.")
    return dict(two_face_encoder=resolved_encoder, two_face_triangle_layers=resolved_layers)


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
