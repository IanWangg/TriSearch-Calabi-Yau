import torch
from torch_geometric.data import Data

from core.vertex_augmentation import (
    apply_similarity_transform,
    augment_data_list_vertex_coords,
    augment_vertex_coordinates,
    compose_similarity_transforms,
    sample_similarity_transform,
    transform_data_list_vertex_coords,
)


def _build_graph_data() -> Data:
    return Data(
        x=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float,
        ),
        edge_index=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 0],
                [1, 0, 2, 1, 3, 2, 0, 3],
            ],
            dtype=torch.long,
        ),
        subcomplex_vertices=torch.tensor(
            [
                [0, 1, 2, -1],
                [0, 2, 3, -1],
            ],
            dtype=torch.long,
        ),
        num_available_subcomplexes=2,
    )


def test_augment_data_preserves_combinatorial_fields_and_input_tensor():
    torch.manual_seed(0)
    data = _build_graph_data()
    x_before = data.x.clone()
    edge_before = data.edge_index.clone()
    subcomplex_before = data.subcomplex_vertices.clone()

    augmented = augment_data_list_vertex_coords(
        [data],
        aug_prob=1.0,
        scale_min=0.9,
        scale_max=1.1,
        shift_std=0.05,
        reflect_prob=0.1,
    )

    assert len(augmented) == 1
    aug_data = augmented[0]

    assert torch.equal(data.x, x_before)
    assert torch.equal(aug_data.edge_index, edge_before)
    assert torch.equal(aug_data.subcomplex_vertices, subcomplex_before)
    assert int(aug_data.num_available_subcomplexes) == 2



def test_augment_data_prob_zero_keeps_coordinates():
    torch.manual_seed(0)
    data = _build_graph_data()

    augmented = augment_data_list_vertex_coords(
        [data],
        aug_prob=0.0,
        scale_min=0.9,
        scale_max=1.1,
        shift_std=0.05,
        reflect_prob=0.1,
    )

    assert len(augmented) == 1
    assert augmented[0] is not data
    assert torch.allclose(augmented[0].x, data.x)



def test_rotation_only_preserves_pairwise_distances():
    torch.manual_seed(0)
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=torch.float,
    )

    x_aug = augment_vertex_coordinates(
        x,
        scale_min=1.0,
        scale_max=1.0,
        shift_std=0.0,
        reflect_prob=0.0,
    )

    d_orig = torch.cdist(x, x)
    d_aug = torch.cdist(x_aug, x_aug)
    assert torch.allclose(d_orig, d_aug, atol=1e-5)


def test_sampled_similarity_transform_can_be_reapplied_consistently():
    torch.manual_seed(0)
    data = _build_graph_data()
    transform = sample_similarity_transform(
        data.x,
        aug_prob=1.0,
        scale_min=0.9,
        scale_max=1.1,
        shift_std=0.05,
        reflect_prob=0.1,
    )

    transformed = transform_data_list_vertex_coords([data, data], [transform, transform])

    assert torch.allclose(transformed[0].x, transformed[1].x)
    assert torch.equal(transformed[0].edge_index, data.edge_index)
    assert torch.equal(transformed[0].subcomplex_vertices, data.subcomplex_vertices)


def test_composed_similarity_transform_matches_sequential_application():
    torch.manual_seed(0)
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=torch.float,
    )
    first = sample_similarity_transform(
        x,
        aug_prob=1.0,
        scale_min=0.9,
        scale_max=1.1,
        shift_std=0.05,
        reflect_prob=0.1,
    )
    x_first = apply_similarity_transform(x, first)
    second = sample_similarity_transform(
        x_first,
        aug_prob=1.0,
        scale_min=0.9,
        scale_max=1.1,
        shift_std=0.05,
        reflect_prob=0.1,
    )
    sequential = apply_similarity_transform(x_first, second)
    composed = apply_similarity_transform(x, compose_similarity_transforms(first, second))

    assert torch.allclose(sequential, composed, atol=1e-6)
