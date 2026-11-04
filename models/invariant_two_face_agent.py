"""GL(4,Z)-invariant TwoFaceAgent: point-merged message passing over invariant features.

Only the encoder differs from TwoFaceAgent. Node embeddings come from an MPNN over merged lattice
points (point_x_inv, point_edge_index), faces are described by face_x_inv, and the global feature
concatenates the sum and mean of phi_face(faces) before a LayerNorm. Triangle MLP, actor and value
heads, get_value_and_logits and forward are inherited unchanged. batch.x is never read.
Inputs must come from core.two_face_invariant_features.add_invariant_fields.
"""

import torch
from torch import nn

from core.cy_episode_context import CONTEXT_FIELDS
from core.two_face_invariant_features import NUM_FACE_FEATURES, NUM_POINT_FEATURES
from .two_face_agent import TwoFaceAgent, _mlp, _sum


class PointMessageLayer(nn.Module):
    """Pre-LayerNorm message passing with sum, mean and max aggregation and a residual update."""

    def __init__(self, channels, act):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.message = _mlp(2 * channels, channels, channels, act)
        self.update = _mlp(4 * channels, channels, channels, act)

    def forward(self, h, edge_index):
        source, target = edge_index
        normalized = self.norm(h)
        messages = self.message(torch.cat((normalized[source], normalized[target]), dim=-1))
        total = _sum(messages, target, h.size(0))
        degree = torch.bincount(target, minlength=h.size(0)).to(h.dtype).unsqueeze(-1)
        mean = total / degree.clamp_min(1)
        maximum = h.new_zeros(h.shape).scatter_reduce(
            0, target.unsqueeze(-1).expand_as(messages), messages, reduce="amax", include_self=False)
        return h + self.update(torch.cat((normalized, total, mean, maximum), dim=-1))


class PointMessagePassing(nn.Module):
    def __init__(self, in_features, channels, num_layers, act):
        super().__init__()
        self.input = _mlp(in_features, channels, channels, act)
        self.layers = nn.ModuleList(PointMessageLayer(channels, act) for _ in range(num_layers))

    def forward(self, point_features, edge_index):
        h = self.input(point_features)
        for layer in self.layers:
            h = layer(h, edge_index)
        return h


class InvariantTwoFaceAgent(TwoFaceAgent):
    def __init__(self, *, mp_layers=4, **kwargs):
        super().__init__(**kwargs)
        if type(mp_layers) is not int or mp_layers <= 0:
            raise ValueError("mp_layers must be a positive integer.")
        out_channels = self.model_config["out_channels"]
        hidden_channels = self.model_config["hidden_channels"]
        act = self.model_config["activation"]
        device = kwargs.get("device", "cpu")
        self.model_config.update(encoder_variant="gl4z_invariant_v1", mp_layers=mp_layers,
                                 invariant_point_features=NUM_POINT_FEATURES,
                                 invariant_face_features=NUM_FACE_FEATURES)
        self.point_egnn = PointMessagePassing(NUM_POINT_FEATURES, out_channels, mp_layers, act)
        self.face_mlp = _mlp(out_channels + NUM_FACE_FEATURES, hidden_channels, out_channels, act)
        self.global_norm = nn.LayerNorm(2 * out_channels)
        self.rho = _mlp(2 * out_channels + 1, hidden_channels, out_channels, act)
        self.to(device)

    def _encode(self, batch):
        if getattr(batch, "observation_kind", None) != self.observation_kind:
            raise ValueError("InvariantTwoFaceAgent requires a TwoFaceData batch.")
        if getattr(batch, "point_index", None) is None or getattr(batch, "face_x_inv", None) is None:
            raise ValueError("InvariantTwoFaceAgent requires add_invariant_fields data.")
        point_h = self.point_egnn(batch.point_x_inv, batch.point_edge_index)
        h = point_h[batch.point_index]
        z = self.triangle_mlp(h[batch.triangle_vertices].sum(dim=1))
        for layer in self.triangle_dual:
            z = layer(z, batch.triangle_edge_index)
        face_counts = batch.num_faces.reshape(-1).long()
        face_graph = torch.repeat_interleave(torch.arange(batch.num_graphs, device=h.device), face_counts)
        num_faces = face_graph.numel()
        face_sum = _sum(self.phi_triangle(z), batch.triangle_face, num_faces)
        faces = self.face_mlp(torch.cat((face_sum, batch.face_x_inv.to(face_sum)), dim=-1))
        face_features = self.phi_face(faces)
        global_sum = _sum(face_features, face_graph, batch.num_graphs)
        global_mean = global_sum / face_counts.to(h.dtype).unsqueeze(-1).clamp_min(1)
        counts = face_counts.to(h.dtype).unsqueeze(-1)
        global_features = self.rho(torch.cat(
            (self.global_norm(torch.cat((global_sum, global_mean), dim=-1)), counts), dim=-1))
        if self.requires_episode_context:
            context = getattr(batch, "episode_context", None)
            if context is None or context.shape != (batch.num_graphs, len(CONTEXT_FIELDS)):
                raise ValueError("Context policy requires one episode_context row per state.")
            if not torch.isfinite(context).all():
                raise ValueError("episode_context must be finite.")
            global_features = self.context_mlp(torch.cat((global_features, context.to(global_features)), dim=-1))
        return h, z, faces, global_features, face_graph
