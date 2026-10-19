"""Shared face-local EGNN, triangle readout, and Deep Sets actor/critic."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from core.cy_two_face_data import TWO_FACE_OBSERVATION_SCHEMA_VERSION
from core.cy_episode_context import BEST_SO_FAR_LOG_VOLUME, STEP_LOG_VOLUME, CONTEXT_FIELDS
from .act_resolver import activation_resolver
from .egnn import EGNN


def _sum(features, indices, size):
    return features.new_zeros((size, features.size(-1))).index_add(0, indices, features)


def _mlp(inputs, hidden, outputs, act):
    return nn.Sequential(nn.Linear(inputs, hidden), activation_resolver(act), nn.Linear(hidden, outputs))


class TwoFaceAgent(nn.Module):
    observation_kind = "two_face"
    observation_schema_version = TWO_FACE_OBSERVATION_SCHEMA_VERSION
    subcomplex_actor_type = "two_face_deep_sets"
    value_feature_source = "two_face"

    def __init__(self, *, in_channels, out_channels=64, hidden_channels=64, num_layers=3,
                 act="silu", device="cpu", training_reward_mode=STEP_LOG_VOLUME, episode_horizon=None):
        super().__init__()
        if training_reward_mode not in (STEP_LOG_VOLUME, BEST_SO_FAR_LOG_VOLUME):
            raise ValueError("Unsupported training_reward_mode.")
        self.training_reward_mode = training_reward_mode
        self.requires_episode_context = training_reward_mode == BEST_SO_FAR_LOG_VOLUME
        if self.requires_episode_context:
            if type(episode_horizon) is not int or episode_horizon <= 0:
                raise ValueError("Context policy requires a positive episode_horizon.")
            self.observation_schema_version = 2
        if in_channels not in (3, 4):
            raise ValueError("two_face_deep_sets requires coordinate dimension 3 or 4.")
        if min(out_channels, hidden_channels, num_layers) <= 0:
            raise ValueError("Model channels and layer counts must be positive.")
        self.model_config = dict(
            model_type="egnn", subcomplex_actor_type=self.subcomplex_actor_type,
            observation_kind=self.observation_kind, observation_schema_version=self.observation_schema_version,
            in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
            num_layers=num_layers, activation=act,
            face_pooling="sum", global_pooling="sum",
            vertex_preprocessing="none", vertex_augmentation=False, actor_global_context=True,
        )
        if self.requires_episode_context:
            self.model_config.update(
                training_reward_mode=training_reward_mode, episode_horizon=episode_horizon,
                episode_context_fields=list(CONTEXT_FIELDS),
                critic_target="expected_remaining_best_log_volume_gain", gamma=1.0,
            )
        self.point_egnn = EGNN(in_node_nf=in_channels, hidden_nf=hidden_channels,
                               out_node_nf=out_channels, n_layers=num_layers, act_fn=act, device=device)
        self.triangle_mlp = _mlp(out_channels, hidden_channels, out_channels, act)
        self.phi_triangle = _mlp(out_channels, hidden_channels, out_channels, act)
        self.face_mlp = _mlp(out_channels + in_channels + 1, hidden_channels, out_channels, act)
        self.phi_face = _mlp(out_channels, hidden_channels, out_channels, act)
        self.rho = _mlp(out_channels + 1, hidden_channels, out_channels, act)
        self.actor_mlp = _mlp(4 * out_channels, hidden_channels, 1, act)
        self.value_mlp = _mlp(out_channels, hidden_channels, 1, act)
        if self.requires_episode_context:
            self.context_mlp = _mlp(out_channels + len(CONTEXT_FIELDS), hidden_channels, out_channels, act)
        self.to(device)

    def _encode(self, batch):
        if getattr(batch, "observation_kind", None) != self.observation_kind:
            raise ValueError("TwoFaceAgent requires a TwoFaceData batch.")
        h, _ = self.point_egnn(h=batch.x, x=batch.x, edges=batch.edge_index)
        z = self.triangle_mlp(h[batch.triangle_vertices].sum(dim=1))
        face_counts = batch.num_faces.reshape(-1).long()
        face_graph = torch.repeat_interleave(torch.arange(batch.num_graphs, device=h.device), face_counts)
        num_faces = face_graph.numel()
        node_counts = torch.bincount(batch.node_face, minlength=num_faces).to(h.dtype).unsqueeze(-1)
        centroids = _sum(batch.x, batch.node_face, num_faces) / node_counts.clamp_min(1)
        face_sum = _sum(self.phi_triangle(z), batch.triangle_face, num_faces)
        faces = self.face_mlp(torch.cat((face_sum, centroids, node_counts), dim=-1))
        global_sum = _sum(self.phi_face(faces), face_graph, batch.num_graphs)
        global_features = self.rho(torch.cat((global_sum, face_counts.to(h.dtype).unsqueeze(-1)), dim=-1))
        if self.requires_episode_context:
            context = getattr(batch, "episode_context", None)
            if context is None or context.shape != (batch.num_graphs, len(CONTEXT_FIELDS)):
                raise ValueError("Context policy requires one episode_context row per state.")
            if not torch.isfinite(context).all():
                raise ValueError("episode_context must be finite.")
            global_features = self.context_mlp(torch.cat((global_features, context.to(global_features)), dim=-1))
        return h, z, faces, global_features, face_graph

    def get_value(self, batch):
        # Do not even inspect action tensors on the critic-only path.
        _, _, _, global_features, _ = self._encode(batch)
        return self.value_mlp(global_features).squeeze(-1)

    def get_value_and_logits(self, batch):
        h, z, faces, global_features, face_graph = self._encode(batch)
        value = self.value_mlp(global_features).squeeze(-1)
        removed = z[batch.removed_triangle_ids].sum(dim=1)
        proposed = self.triangle_mlp(h[batch.added_triangle_vertices].sum(dim=2)).sum(dim=1)
        logits = self.actor_mlp(torch.cat((removed, proposed, faces[batch.action_face],
                                          global_features[face_graph[batch.action_face]]), dim=-1)).squeeze(-1)
        counts = batch.num_available_subcomplexes.reshape(-1).tolist()
        padded = pad_sequence(logits.split(counts), batch_first=True, padding_value=float("-inf"))
        return value, padded

    def get_log_prob(self, logits, action_indices):
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.log_prob(action_indices.to(logits.device).long().reshape(-1)), distribution.entropy()

    def forward(self, batch, deterministic=False):
        value, logits = self.get_value_and_logits(batch)
        counts = batch.num_available_subcomplexes.reshape(-1).long()
        valid = counts > 0
        indices = torch.full_like(counts, -1)
        selected = torch.full((batch.num_graphs, 4), -1, dtype=torch.long, device=value.device)
        log_prob, entropy = torch.zeros_like(value), torch.zeros_like(value)
        if valid.any():
            distribution = torch.distributions.Categorical(logits=logits[valid])
            indices[valid] = logits[valid].argmax(dim=-1) if deterministic else distribution.sample()
            log_prob[valid] = distribution.log_prob(indices[valid])
            entropy[valid] = distribution.entropy()
            offsets = counts.cumsum(dim=0) - counts
            selected[valid] = batch.subcomplex_vertices[offsets[valid] + indices[valid]]
        return selected, indices, value, log_prob, entropy
