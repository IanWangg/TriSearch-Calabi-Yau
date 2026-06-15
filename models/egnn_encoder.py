"""
EGNN-based encoder with dual-head projection.

Extracts the encoder and projection logic that was previously embedded in
EGNNLinkPredictor.  All attribute names are kept identical so that existing
checkpoints load without key mismatches.
"""

import torch.nn as nn
import torch_geometric.nn as gnn

from .egnn import EGNN
from .act_resolver import activation_resolver


class EGNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=64,
        num_layers=3,
        share_encoder=True,
        mlp_hidden_channel_list=[64],
        use_projection=True,
        act="silu",
        device="cpu",
    ):
        super(EGNNEncoder, self).__init__()

        self.share_encoder = share_encoder
        self.act = activation_resolver(act)
        self.use_projection = use_projection
        self.device = device

        self.encoder_remove = EGNN(
            in_node_nf=in_channels,
            hidden_nf=hidden_channels,
            out_node_nf=out_channels,
            in_edge_nf=0,
            act_fn=activation_resolver(act),
            n_layers=num_layers,
            device=device,
        )
        if self.share_encoder:
            self.encoder_add = None
        else:
            self.encoder_add = EGNN(
                in_node_nf=in_channels,
                hidden_nf=hidden_channels,
                out_node_nf=out_channels,
                in_edge_nf=0,
                act_fn=activation_resolver(act),
                n_layers=num_layers,
                device=device,
            )

        if use_projection:
            mlp_full_layer_list = [out_channels] + mlp_hidden_channel_list + [out_channels]
            self.projection_remove = gnn.models.MLP(mlp_full_layer_list, act=act)
            self.projection_add = gnn.models.MLP(mlp_full_layer_list, act=act)
        else:
            self.projection_remove = nn.Identity()
            self.projection_add = nn.Identity()

        self.to(device)

    def encode(self, h, x, edges, edge_attr=None):
        z_remove, _ = self.encoder_remove(h, x, edges, edge_attr=edge_attr)
        if self.share_encoder:
            z_add = z_remove
        else:
            z_add, _ = self.encoder_add(h, x, edges, edge_attr=edge_attr)
        return z_remove, z_add

    def encode_projection(self, h, x, edges, edge_attr=None, return_z_before_proj=False):
        z_remove, z_add = self.encode(h, x, edges, edge_attr=edge_attr)
        z_remove_proj = self.projection_remove(z_remove)
        z_add_proj = self.projection_add(z_add)
        if return_z_before_proj:
            return z_remove_proj, z_add_proj, z_remove, z_add
        return z_remove_proj, z_add_proj
