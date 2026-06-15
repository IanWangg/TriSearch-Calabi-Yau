from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class SimilarityTransform:
    matrix: torch.Tensor
    bias: torch.Tensor


def _rand_uniform(shape, *, device, dtype, generator: Optional[torch.Generator]):
    if generator is None:
        return torch.rand(shape, device=device, dtype=dtype)
    return torch.rand(shape, generator=generator, device="cpu", dtype=dtype).to(device=device)


def _rand_normal(shape, *, device, dtype, generator: Optional[torch.Generator]):
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    return torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device=device)


def _rand_int(low: int, high: int, *, generator: Optional[torch.Generator]) -> int:
    if generator is None:
        return int(torch.randint(low=low, high=high, size=(1,)).item())
    return int(torch.randint(low=low, high=high, size=(1,), generator=generator, device="cpu").item())


def _identity_similarity_transform(*, dim: int, device, dtype) -> SimilarityTransform:
    return SimilarityTransform(
        matrix=torch.eye(dim, device=device, dtype=dtype),
        bias=torch.zeros((1, dim), device=device, dtype=dtype),
    )


def sample_random_orthogonal(
    dim: int,
    *,
    device,
    dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")

    gaussian_matrix = _rand_normal(
        (dim, dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    q, r = torch.linalg.qr(gaussian_matrix)

    diag = torch.diagonal(r)
    signs = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
    return q * signs


def _sample_log_uniform_scale(
    *,
    scale_min: float,
    scale_max: float,
    device,
    dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if scale_min <= 0.0:
        raise ValueError(f"scale_min must be > 0, got {scale_min}.")
    if scale_max < scale_min:
        raise ValueError(
            f"scale_max must be >= scale_min, got scale_min={scale_min}, scale_max={scale_max}."
        )

    if scale_min == scale_max:
        return torch.tensor(scale_min, device=device, dtype=dtype)

    log_min = math.log(scale_min)
    log_max = math.log(scale_max)
    u = _rand_uniform((), device=device, dtype=dtype, generator=generator)
    return torch.exp(log_min + u * (log_max - log_min))


def _sample_upper_triangular_shear_matrix(
    dim: int,
    *,
    shear_std: float,
    device,
    dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if shear_std < 0.0:
        raise ValueError(f"shear_std must be >= 0, got {shear_std}.")
    shear = torch.eye(dim, device=device, dtype=dtype)
    if dim <= 1 or shear_std == 0.0:
        return shear

    upper_triangle_indices = torch.triu_indices(
        row=dim,
        col=dim,
        offset=1,
        device=device,
    )
    shear_values = _rand_normal(
        (upper_triangle_indices.size(1),),
        device=device,
        dtype=dtype,
        generator=generator,
    ) * float(shear_std)
    shear[upper_triangle_indices[0], upper_triangle_indices[1]] = shear_values
    return shear


def sample_similarity_transform(
    x: torch.Tensor,
    *,
    aug_prob: float = 1.0,
    scale_min: float,
    scale_max: float,
    shift_std: float,
    reflect_prob: float,
    shear_std: float = 0.0,
    eps: float = 1e-6,
    generator: Optional[torch.Generator] = None,
) -> SimilarityTransform:
    if x.dim() != 2:
        raise ValueError(f"Expected x to be 2D, got shape {tuple(x.shape)}.")
    if not (0.0 <= aug_prob <= 1.0):
        raise ValueError(f"aug_prob must be in [0, 1], got {aug_prob}.")
    if shift_std < 0.0:
        raise ValueError(f"shift_std must be >= 0, got {shift_std}.")
    if not (0.0 <= reflect_prob <= 1.0):
        raise ValueError(f"reflect_prob must be in [0, 1], got {reflect_prob}.")
    if shear_std < 0.0:
        raise ValueError(f"shear_std must be >= 0, got {shear_std}.")

    num_nodes, dim = x.shape
    if num_nodes == 0 or aug_prob == 0.0:
        return _identity_similarity_transform(dim=dim, device=x.device, dtype=x.dtype)

    if aug_prob < 1.0:
        apply_sample = float(
            _rand_uniform((), device=x.device, dtype=x.dtype, generator=generator).item()
        )
        if apply_sample >= aug_prob:
            return _identity_similarity_transform(dim=dim, device=x.device, dtype=x.dtype)

    centroid = x.mean(dim=0, keepdim=True)
    centered = x - centroid
    graph_radius = centered.norm(dim=1).mean().clamp_min(eps)

    rotation = sample_random_orthogonal(
        dim=dim,
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )

    if reflect_prob > 0.0:
        reflect_sample = float(
            _rand_uniform((), device=x.device, dtype=x.dtype, generator=generator).item()
        )
        if reflect_sample < reflect_prob:
            axis = _rand_int(low=0, high=dim, generator=generator)
            rotation = rotation.clone()
            rotation[:, axis] = -rotation[:, axis]

    scale = _sample_log_uniform_scale(
        scale_min=scale_min,
        scale_max=scale_max,
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )
    shear = _sample_upper_triangular_shear_matrix(
        dim=dim,
        shear_std=float(shear_std),
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )
    matrix = scale * (rotation @ shear)
    shift = _rand_normal(
        (1, dim),
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    ) * (shift_std * graph_radius)
    bias = centroid - centroid @ matrix.t() + shift
    return SimilarityTransform(matrix=matrix, bias=bias)


def apply_similarity_transform(
    x: torch.Tensor,
    transform: SimilarityTransform,
) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"Expected x to be 2D, got shape {tuple(x.shape)}.")
    if transform.matrix.dim() != 2:
        raise ValueError("transform.matrix must be 2D.")
    if transform.bias.dim() != 2:
        raise ValueError("transform.bias must be 2D.")
    if x.size(1) != transform.matrix.size(0) or transform.matrix.size(0) != transform.matrix.size(1):
        raise ValueError("Transform matrix shape must match coordinate dimension.")
    if transform.bias.size(0) != 1 or transform.bias.size(1) != x.size(1):
        raise ValueError("Transform bias must have shape (1, coord_dim).")
    return x @ transform.matrix.t() + transform.bias


def compose_similarity_transforms(
    first: SimilarityTransform,
    second: SimilarityTransform,
) -> SimilarityTransform:
    if first.matrix.shape != second.matrix.shape:
        raise ValueError("Cannot compose similarity transforms with different matrix shapes.")
    if first.bias.shape != second.bias.shape:
        raise ValueError("Cannot compose similarity transforms with different bias shapes.")
    return SimilarityTransform(
        matrix=second.matrix @ first.matrix,
        bias=first.bias @ second.matrix.t() + second.bias,
    )


def transform_data_list_vertex_coords(
    data_list: Sequence[Data],
    transforms: Sequence[SimilarityTransform],
) -> List[Data]:
    if len(data_list) != len(transforms):
        raise ValueError("data_list and transforms must have the same length.")

    transformed_data_list: List[Data] = []
    for data, transform in zip(data_list, transforms):
        if not hasattr(data, "x"):
            raise AttributeError("Each Data object must contain vertex coordinates in `x`.")
        cloned = data.clone()
        cloned.x = apply_similarity_transform(cloned.x, transform)
        transformed_data_list.append(cloned)
    return transformed_data_list


def augment_vertex_coordinates(
    x: torch.Tensor,
    *,
    scale_min: float,
    scale_max: float,
    shift_std: float,
    reflect_prob: float,
    eps: float = 1e-6,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    transform = sample_similarity_transform(
        x,
        aug_prob=1.0,
        scale_min=scale_min,
        scale_max=scale_max,
        shift_std=shift_std,
        reflect_prob=reflect_prob,
        eps=eps,
        generator=generator,
    )
    return apply_similarity_transform(x, transform)


def augment_data_list_vertex_coords(
    data_list: Iterable[Data],
    *,
    aug_prob: float = 1.0,
    scale_min: float = 0.9,
    scale_max: float = 1.1,
    shift_std: float = 0.05,
    reflect_prob: float = 0.1,
    eps: float = 1e-6,
    generator: Optional[torch.Generator] = None,
) -> List[Data]:
    transforms: List[SimilarityTransform] = []
    normalized_data_list = list(data_list)
    for data in normalized_data_list:
        if not hasattr(data, "x"):
            raise AttributeError("Each Data object must contain vertex coordinates in `x`.")
        transforms.append(
            sample_similarity_transform(
                data.x,
                aug_prob=aug_prob,
                scale_min=scale_min,
                scale_max=scale_max,
                shift_std=shift_std,
                reflect_prob=reflect_prob,
                eps=eps,
                generator=generator,
            )
        )
    return transform_data_list_vertex_coords(normalized_data_list, transforms)
