"""Smoke policy for JumpEnv: the invariant two-face actor head with mean2 aggregation, randomly initialised.

batched_move_logits is FaceMoveScorer.score generalised to a batch of graphs: each candidate is
owned by its graph's candidate range, so one forward pass serves several envs. Weights are random
(the architecture comes from the checkpoint's model_config.json); this is a smoke policy, not training.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from eval.algorithm.face_move_scorer import DEFAULT_CHECKPOINT


def build_random_policy(checkpoint=DEFAULT_CHECKPOINT, *, seed=0):
    from models.subcomplex_policy_factory import build_subcomplex_agent

    path = Path(checkpoint).expanduser().resolve()
    saved = json.loads((path.parent / "model_config.json").read_text(encoding="utf-8"))
    if saved.get("subcomplex_actor_type") != "two_face_invariant" or saved.get("observation_kind") != "two_face":
        raise ValueError(f"Checkpoint {path} is not an invariant two-face actor.")
    torch.manual_seed(int(seed))
    model = build_subcomplex_agent(
        model_type=saved["model_type"], in_channels=saved["in_channels"], out_channels=saved["out_channels"],
        hidden_channels=saved["hidden_channels"], num_layers=saved["num_layers"], act=saved["activation"],
        subcomplex_actor_type=saved["subcomplex_actor_type"], two_face_encoder=saved["two_face_encoder"],
        device="cpu")
    return model.eval()


def move_logits(model, batch, aggregation="mean2"):
    """Move-head logits for every candidate of a JumpObsData batch (grad-enabled), plus global features.

    Returns (logits, global_features): logits is (total_candidates,) in candidate order, graph by graph;
    global_features is (num_graphs, channels) from the shared encoder.
    """
    if aggregation not in ("mean2", "sum"):
        raise ValueError("aggregation must be 'mean2' or 'sum'.")
    counts = batch.num_candidates.reshape(-1).long()
    total = int(counts.sum())
    h, z, faces, global_features, _ = model._encode(batch)
    if total == 0:
        return global_features.new_zeros(0), global_features
    long = torch.long
    owner_graph = torch.repeat_interleave(torch.arange(batch.num_graphs, device=h.device), counts)
    removed_index = batch.move_removed_ids.to(long)
    removed_owner = batch.move_removed_owner.to(long)
    added_index = batch.move_added_nodes.to(long).reshape(-1, 3)
    added_owner = batch.move_added_owner.to(long)
    removed_sum = z.new_zeros((total, z.size(1))).index_add(0, removed_owner, z[removed_index])
    proposed = model.triangle_mlp(h[added_index].sum(dim=1))
    added_sum = z.new_zeros((total, z.size(1))).index_add(0, added_owner, proposed)
    removed_count = torch.bincount(removed_owner, minlength=total).to(z.dtype).unsqueeze(-1)
    added_count = torch.bincount(added_owner, minlength=total).to(z.dtype).unsqueeze(-1)
    if aggregation == "mean2":
        removed_sum = 2 * removed_sum / removed_count.clamp_min(1)
        added_sum = 2 * added_sum / added_count.clamp_min(1)
    logits = model.actor_mlp(torch.cat(
        (removed_sum, added_sum, faces[batch.move_face.to(long)], global_features[owner_graph]), dim=-1))
    return logits.reshape(-1), global_features


def batched_move_logits(model, batch, aggregation="mean2"):
    """Actor logits for every candidate of a JumpObsData batch (no grad, float64), as in FaceMoveScorer."""
    with torch.no_grad():
        logits, _ = move_logits(model, batch, aggregation)
    values = logits.double()
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite jump logits.")
    return values


def sample_actions(logits, counts, generator):
    """One softmax sample per graph (None for a graph without candidates)."""
    actions = []
    for chunk in torch.split(logits, [int(c) for c in counts]):
        if chunk.numel() == 0:
            actions.append(None)
            continue
        probs = torch.softmax(chunk, dim=0)
        actions.append(int(torch.multinomial(probs, 1, generator=generator)))
    return actions
