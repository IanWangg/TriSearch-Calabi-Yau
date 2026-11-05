"""Learned scorer for single-face replacement moves (GL(4,Z)-invariant two-face actor).

A move replaces the triangulation of one 2-face of the parent FRST. For the parent it runs the shared
encoder once and returns, for each candidate move, the actor logit the trained InvariantTwoFaceAgent
assigns to it. The actor input is the same as for a backend action: removed = parent face triangles
not in the new face FRT, added = new face triangles not in the parent, each aggregated as a set
(the triangle embedding z for removed, triangle_mlp of the summed node embeddings for added).

Aggregation: the actor was trained on flips with exactly two removed and two added triangles, where its
input is a sum over the two. "mean2" uses 2 * mean over each set, which equals the sum for flips and is
size-normalized otherwise; "sum" uses the raw sum.
"""

from __future__ import annotations

from collections import namedtuple
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

DEFAULT_CHECKPOINT = Path(
    "/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/runs/inv_rl_full_20261104/checkpoints/final.pth")
AGGREGATIONS = ("mean2", "sum")


def load_face_move_model(checkpoint):
    """Rebuild the invariant two-face actor from model_config.json beside the checkpoint, then load it."""
    from core.cy_runtime_utils import load_policy_checkpoint
    from models.subcomplex_policy_factory import build_subcomplex_agent

    path = Path(checkpoint).expanduser().resolve()
    saved = json.loads((path.parent / "model_config.json").read_text(encoding="utf-8"))
    if saved.get("subcomplex_actor_type") != "two_face_invariant" or saved.get("observation_kind") != "two_face":
        raise ValueError(f"Checkpoint {path} is not an invariant two-face actor.")
    model = build_subcomplex_agent(
        model_type=saved["model_type"], in_channels=saved["in_channels"], out_channels=saved["out_channels"],
        hidden_channels=saved["hidden_channels"], num_layers=saved["num_layers"], act=saved["activation"],
        subcomplex_actor_type=saved["subcomplex_actor_type"], two_face_encoder=saved["two_face_encoder"],
        device="cpu")
    load_policy_checkpoint(model, str(path), map_location=torch.device("cpu"))
    return model.eval()


MoveIndex = namedtuple("MoveIndex", ("move_face", "removed_ids", "removed_move", "added_nodes", "added_move"))


def move_index(parent_state, moves):
    """Map single-face moves of parent_state to triangle ids of its invariant two-face topology.

    moves: list of (face_labels, new_triangles) as documented on FaceMoveScorer.score. Returns a MoveIndex:
    move_face[m] is the ambient 2-face index of move m; removed_ids / removed_move list the parent
    triangle ids leaving the face and their owner move; added_nodes / added_move list the node triples of
    the entering triangles and their owner move. Shared by FaceMoveScorer and rl_jump; raises ValueError
    as documented there.
    """
    from core.cy_two_face_data import _build_topology
    from mdp.cy_state_record import two_face_restrictions

    configuration = parent_state.configuration
    restrictions = two_face_restrictions(configuration, parent_state.simplices)
    topology = _build_topology(configuration, restrictions)
    face_of_labels = {labels: index for index, (labels, _) in enumerate(restrictions)}
    old_triangles = [set(triangles) for _, triangles in restrictions]

    move_face, removed_ids, removed_move, added_nodes, added_move = [], [], [], [], []
    for m, (labels, new_triangles) in enumerate(moves):
        key = tuple(sorted(int(label) for label in labels))
        face = face_of_labels.get(key)
        if face is None:
            raise ValueError(f"Face {key} is not a 2-face of the parent state.")
        node_map, triangle_map = topology.node_maps[face], topology.triangle_maps[face]
        new = {tuple(sorted(int(v) for v in triangle)) for triangle in new_triangles}
        for triangle in new:
            if len(triangle) != 3 or len(set(triangle)) != 3 or not set(triangle) <= node_map.keys():
                raise ValueError(f"Triangle {triangle} is not a 3-subset of face {key}.")
        removed = sorted(old_triangles[face] - new)
        added = sorted(new - old_triangles[face])
        if not removed and not added:
            raise ValueError(f"Move leaves face {key} unchanged.")
        move_face.append(face)
        for triangle in removed:
            removed_ids.append(triangle_map[triangle])
            removed_move.append(m)
        for triangle in added:
            added_nodes.append([node_map[label] for label in triangle])
            added_move.append(m)
    return MoveIndex(move_face, removed_ids, removed_move, added_nodes, added_move)


class FaceMoveScorer:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, *, aggregation="mean2"):
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}.")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.aggregation = aggregation
        self.model = load_face_move_model(self.checkpoint)
        self.last_seconds = 0.0

    def score(self, parent_state, moves):
        """Actor logits (higher is better) for single-face moves of parent_state, as a float64 array.

        moves: iterable of (face_labels, new_triangles). face_labels is the ambient label set of a 2-face of
        the parent, matched by its sorted label tuple (never by position). new_triangles are the triangles
        of that face's new triangulation in ambient labels. Raises ValueError for an unmatched face, a
        triangle outside the face, or a move that leaves the face unchanged.
        """
        from core.two_face_invariant_features import create_invariant_two_face_data_from_state

        started = time.perf_counter()
        moves = list(moves)
        if not moves:
            self.last_seconds = 0.0
            return np.empty(0, dtype=np.float64)
        index = move_index(parent_state, moves)
        move_face, removed_ids, removed_move = index.move_face, index.removed_ids, index.removed_move
        added_nodes, added_move = index.added_nodes, index.added_move

        data = create_invariant_two_face_data_from_state(parent_state, ())
        batch = Batch.from_data_list([data])
        long = torch.long
        with torch.no_grad():
            h, z, faces, global_features, _ = self.model._encode(batch)
            count = len(moves)
            removed_index = torch.tensor(removed_ids, dtype=long)
            removed_owner = torch.tensor(removed_move, dtype=long)
            added_index = torch.tensor(added_nodes, dtype=long).reshape(-1, 3)
            added_owner = torch.tensor(added_move, dtype=long)
            removed_sum = z.new_zeros((count, z.size(1))).index_add(0, removed_owner, z[removed_index])
            proposed = self.model.triangle_mlp(h[added_index].sum(dim=1))
            added_sum = z.new_zeros((count, z.size(1))).index_add(0, added_owner, proposed)
            removed_count = torch.bincount(removed_owner, minlength=count).to(z.dtype).unsqueeze(-1)
            added_count = torch.bincount(added_owner, minlength=count).to(z.dtype).unsqueeze(-1)
            if self.aggregation == "mean2":
                removed_sum = 2 * removed_sum / removed_count.clamp_min(1)
                added_sum = 2 * added_sum / added_count.clamp_min(1)
            move_face_index = torch.tensor(move_face, dtype=long)
            global_rows = global_features[0].expand(count, -1)
            logits = self.model.actor_mlp(
                torch.cat((removed_sum, added_sum, faces[move_face_index], global_rows), dim=-1)).squeeze(-1)
        values = logits.double().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite face-move logits.")
        self.last_seconds = time.perf_counter() - started
        return values
