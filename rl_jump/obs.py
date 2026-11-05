"""Policy observation for JumpEnv: invariant two-face data of the current FRST plus per-candidate move ids.

The candidate ids come from eval.algorithm.face_move_scorer.move_index, the same mapping the learned
FaceMoveScorer uses, so a candidate here and a scorer move with the same face and new triangles
produce identical actor inputs. Candidate fields are stored with PyG batching keys so several envs
batch into one graph batch: move_face indexes faces, move_removed_ids / move_added_nodes index
triangles / nodes, and the owner keys index candidates (all offset per graph).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import torch

from core.two_face_invariant_features import InvariantTwoFaceData

MOVE_KEYS = ("move_face", "move_removed_ids", "move_removed_owner", "move_added_nodes", "move_added_owner")


class JumpObsData(InvariantTwoFaceData):
    """InvariantTwoFaceData with per-candidate move fields; num_candidates is a scalar per graph."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("move_removed_owner", "move_added_owner"):
            return int(self.num_candidates)
        if key == "move_removed_ids":
            return int(self.num_triangles)
        if key == "move_added_nodes":
            return int(self.num_nodes)
        if key == "move_face":
            return int(self.num_faces)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in MOVE_KEYS:
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


@dataclass
class JumpObservation:
    """What the policy sees for one state. Plain metadata is kept beside the tensors (picklable)."""

    data: JumpObsData
    dna: tuple
    value: float
    t: int
    candidates: list = field(default_factory=list)  # dicts: dna, position, index, face, symdiff
    stats: dict = field(default_factory=dict)  # work spent entering this state (decode, obs build, ...)


def build_observation(state, candidates):
    """JumpObsData for `state` with one move per candidate (candidates: objects with face_labels, new_triangles)."""
    from core.two_face_invariant_features import create_invariant_two_face_data_from_state
    from eval.algorithm.face_move_scorer import move_index

    base = create_invariant_two_face_data_from_state(state, ())
    index = move_index(state, [(c.face_labels, c.new_triangles) for c in candidates])
    data = JumpObsData()
    for key, value in base.items():
        data[key] = value
    data.num_candidates = len(candidates)
    data.move_face = torch.as_tensor(index.move_face, dtype=torch.long)
    data.move_removed_ids = torch.as_tensor(index.removed_ids, dtype=torch.long)
    data.move_removed_owner = torch.as_tensor(index.removed_move, dtype=torch.long)
    data.move_added_nodes = torch.as_tensor(index.added_nodes, dtype=torch.long).reshape(-1, 3)
    data.move_added_owner = torch.as_tensor(index.added_move, dtype=torch.long)
    return data


def timed_build(state, candidates):
    started = time.perf_counter()
    data = build_observation(state, candidates)
    return data, time.perf_counter() - started
