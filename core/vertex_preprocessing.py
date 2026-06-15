from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


SUPPORTED_PREPROCESSING = ("none", "rms_radius", "whitening")


def normalize_preprocessing_mode(mode: str) -> str:
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in SUPPORTED_PREPROCESSING:
        raise ValueError(
            f"Unsupported preprocessing '{mode}'. "
            f"Expected one of: {', '.join(SUPPORTED_PREPROCESSING)}."
        )
    return resolved_mode


@dataclass(frozen=True)
class _FittedVertexTransform:
    centroid: torch.Tensor
    linear_map: torch.Tensor

    def apply(self, vertices: torch.Tensor) -> torch.Tensor:
        if vertices.dim() != 2:
            raise ValueError(f"Expected vertices to be 2D, got shape {tuple(vertices.shape)}.")

        vertices_fp64 = vertices.to(dtype=torch.float64)
        centroid = self.centroid.to(device=vertices.device)
        linear_map = self.linear_map.to(device=vertices.device)
        transformed = (vertices_fp64 - centroid) @ linear_map
        return transformed.to(dtype=vertices.dtype)


def fit_vertex_transform(
    vertices: torch.Tensor,
    *,
    mode: str,
    eps: float = 1e-8,
    whitening_trace_scale: float = 1e-6,
) -> _FittedVertexTransform:
    if vertices.dim() != 2:
        raise ValueError(f"Expected vertices to be 2D, got shape {tuple(vertices.shape)}.")

    resolved_mode = normalize_preprocessing_mode(mode)
    num_vertices, coordinate_dim = vertices.shape
    vertices_fp64 = vertices.detach().to(device="cpu", dtype=torch.float64)

    if coordinate_dim == 0:
        return _FittedVertexTransform(
            centroid=torch.zeros((1, 0), dtype=torch.float64),
            linear_map=torch.zeros((0, 0), dtype=torch.float64),
        )

    if resolved_mode == "none":
        return _FittedVertexTransform(
            centroid=torch.zeros((1, coordinate_dim), dtype=torch.float64),
            linear_map=torch.eye(coordinate_dim, dtype=torch.float64),
        )

    centroid = vertices_fp64.mean(dim=0, keepdim=True)
    centered = vertices_fp64 - centroid

    if resolved_mode == "rms_radius":
        if num_vertices == 0:
            scale = 1.0
        else:
            scale_tensor = torch.sqrt(centered.pow(2).sum(dim=1).mean()).clamp_min(float(eps))
            scale = float(scale_tensor.item())
        linear_map = torch.eye(coordinate_dim, dtype=torch.float64) / scale
        return _FittedVertexTransform(centroid=centroid, linear_map=linear_map)

    if num_vertices == 0:
        linear_map = torch.eye(coordinate_dim, dtype=torch.float64)
        return _FittedVertexTransform(centroid=centroid, linear_map=linear_map)

    covariance = centered.t() @ centered / float(num_vertices)
    trace_value = float(torch.trace(covariance).item())
    lambda_floor = max(float(eps), float(whitening_trace_scale) * trace_value / float(coordinate_dim))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    clipped_eigenvalues = torch.clamp(eigenvalues, min=lambda_floor)
    inverse_sqrt = torch.diag(torch.rsqrt(clipped_eigenvalues))
    linear_map = eigenvectors @ inverse_sqrt @ eigenvectors.t()
    return _FittedVertexTransform(centroid=centroid, linear_map=linear_map)


class VertexPreprocessor:
    def __init__(
        self,
        mode: str,
        *,
        eps: float = 1e-8,
        whitening_trace_scale: float = 1e-6,
    ) -> None:
        self.mode = normalize_preprocessing_mode(mode)
        self.eps = float(eps)
        self.whitening_trace_scale = float(whitening_trace_scale)
        self._transform_by_point_config: Dict[int, _FittedVertexTransform] = {}

    def transform_vertices(
        self,
        *,
        point_config_index: Optional[int],
        vertices: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "none":
            return vertices

        if point_config_index is None:
            fitted = fit_vertex_transform(
                vertices,
                mode=self.mode,
                eps=self.eps,
                whitening_trace_scale=self.whitening_trace_scale,
            )
            return fitted.apply(vertices)

        cache_key = int(point_config_index)
        fitted = self._transform_by_point_config.get(cache_key)
        if fitted is None:
            fitted = fit_vertex_transform(
                vertices,
                mode=self.mode,
                eps=self.eps,
                whitening_trace_scale=self.whitening_trace_scale,
            )
            self._transform_by_point_config[cache_key] = fitted
        return fitted.apply(vertices)


def maybe_create_vertex_preprocessor(mode: str) -> Optional[VertexPreprocessor]:
    resolved_mode = normalize_preprocessing_mode(mode)
    if resolved_mode == "none":
        return None
    return VertexPreprocessor(resolved_mode)
