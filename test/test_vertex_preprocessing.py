from __future__ import annotations

from types import SimpleNamespace

import torch

from core.cy_data_utils import create_data_from_cy_state_with_subcomplex
from core.vertex_preprocessing import VertexPreprocessor, fit_vertex_transform


def _covariance(points: torch.Tensor) -> torch.Tensor:
    centered = points - points.mean(dim=0, keepdim=True)
    return centered.t() @ centered / float(points.size(0))


def test_rms_radius_preprocessing_centers_and_scales_vertices():
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 6.0],
        ],
        dtype=torch.float,
    )

    transformed = fit_vertex_transform(vertices, mode="rms_radius").apply(vertices)

    assert torch.allclose(transformed.mean(dim=0), torch.zeros(3), atol=1e-6)
    rms_radius = torch.sqrt(transformed.pow(2).sum(dim=1).mean())
    assert torch.allclose(rms_radius, torch.tensor(1.0), atol=1e-6)


def test_whitening_preprocessing_produces_identity_covariance_for_full_rank_vertices():
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 5.0],
            [2.0, 3.0, 7.0],
        ],
        dtype=torch.float,
    )

    transformed = fit_vertex_transform(vertices, mode="whitening").apply(vertices)

    assert torch.allclose(transformed.mean(dim=0), torch.zeros(3), atol=1e-5)
    assert torch.allclose(_covariance(transformed), torch.eye(3), atol=1e-4, rtol=1e-4)


def test_whitening_preprocessing_stays_finite_for_degenerate_vertices():
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float,
    )

    transformed = fit_vertex_transform(vertices, mode="whitening").apply(vertices)

    assert torch.isfinite(transformed).all()
    assert torch.allclose(transformed.mean(dim=0), torch.zeros(3), atol=1e-6)


def test_cy_data_preprocessing_changes_only_vertex_tensor():
    state = SimpleNamespace(
        point_config_index=19,
        vertices=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        edges=frozenset(
            {
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
            }
        ),
        simplices=frozenset({(0, 1, 2), (0, 1, 3)}),
    )
    subcomplex_actions = ((0, 1, 2, 3),)

    raw_data = create_data_from_cy_state_with_subcomplex(
        state,
        ensure_actions_ready=False,
        subcomplex_actions=subcomplex_actions,
    )
    processed_data = create_data_from_cy_state_with_subcomplex(
        state,
        ensure_actions_ready=False,
        subcomplex_actions=subcomplex_actions,
        vertex_preprocessor=VertexPreprocessor("whitening"),
    )

    assert not torch.allclose(raw_data.x, processed_data.x)
    assert torch.equal(raw_data.edge_index, processed_data.edge_index)
    assert torch.equal(raw_data.subcomplex_vertices, processed_data.subcomplex_vertices)
    assert int(raw_data.num_available_subcomplexes) == int(processed_data.num_available_subcomplexes)
    assert torch.isfinite(processed_data.x).all()
