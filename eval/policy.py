"""One inference service shared by the RL evaluation family."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch

from core.cy_checkpointing import find_latest_policy_checkpoint
from core.cy_policy_inference import evaluate_policy_scores
from core.cy_runtime_utils import load_policy_checkpoint, resolve_training_device
from models.subcomplex_policy_factory import build_subcomplex_agent


class PolicyScorer(Protocol):
    def score_actions(self, states, action_lists) -> Sequence[np.ndarray]:
        """Full-action log probabilities, in input state/action order."""
        ...

    def score_values(self, states) -> Sequence[float]:
        """State critic values; no outgoing actions or geometry queries."""
        ...


class EvaluationPolicy:
    def __init__(self, model, *, device, max_graph_size=250000, metadata=None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.max_graph_size = max_graph_size
        self.metadata = dict(metadata or {})
        self.observation_kind = getattr(model, "observation_kind", "full_triangulation")
        self.metadata.update(
            policy_observation_kind=self.observation_kind,
            policy_observation_schema_version=getattr(model, "observation_schema_version", 1),
            model_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        )
        self.reset_stats()

    @classmethod
    def from_spec(cls, spec):
        path = Path(spec.policy_checkpoint).expanduser().resolve()
        if path.is_dir():
            path = find_latest_policy_checkpoint(str(path))
        if path is None or not path.is_file():
            raise FileNotFoundError(f"No policy checkpoint found at {spec.policy_checkpoint!r}.")
        device = resolve_training_device(gpu_index=spec.gpu_index, force_cpu=spec.force_cpu)
        config = dict(model_type="egnn", in_channels=spec.in_channels, out_channels=spec.out_channels,
                      hidden_channels=spec.hidden_channels, num_layers=spec.num_layers,
                      subcomplex_actor_type=spec.subcomplex_actor_type, share_encoder=True,
                      mlp_hidden_channel_list=[64], act="silu")
        model = build_subcomplex_agent(**config, device=str(device))
        load_policy_checkpoint(model, str(path), map_location=device)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = {"checkpoint": str(path), "checkpoint_sha256": digest,
                    "model": config, "device": str(device), "vertex_preprocessing": "none",
                    "vertex_augmentation": False, "policy_max_graph_size": spec.policy_max_graph_size,
                    "profile_cuda_timing": spec.profile_cuda_timing,
                    "resolved_policy_proposal_count": spec.resolved_policy_proposal_count,
                    "value_discount": spec.value_discount}
        return cls(model, device=device, max_graph_size=spec.policy_max_graph_size, metadata=metadata)

    def reset_stats(self):
        self._stats = dict(action_batches=0, value_batches=0, action_states=0, value_states=0,
                           physical_batches=0, max_logical_batch_states=0,
                           data_build_sec=0.0, batch_transfer_sec=0.0, inference_sec=0.0)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def stats(self):
        return {**self._stats, "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0)}

    def _score(self, states, action_lists, *, value_only):
        result = evaluate_policy_scores(states, action_lists, self.model, device=self.device,
                                        value_only=value_only, max_graph_size=self.max_graph_size)
        kind = "value" if value_only else "action"
        self._stats[f"{kind}_batches"] += int(bool(states))
        self._stats[f"{kind}_states"] += len(states)
        self._stats["physical_batches"] += result.physical_batches
        self._stats["max_logical_batch_states"] = max(self._stats["max_logical_batch_states"], len(states))
        for name in ("data_build_sec", "batch_transfer_sec", "inference_sec"):
            self._stats[name] += getattr(result, name)
        return result

    def score_actions(self, states, action_lists):
        if len(states) != len(action_lists):
            raise ValueError("states and action_lists must have the same length.")
        if not states:
            return []
        result = self._score(states, action_lists, value_only=False)
        # One device-to-host transfer. Per-start sampling uses independent NumPy
        # RNGs, so pruning another trajectory cannot perturb this one's draws.
        logits = result.logits_tensor.detach().cpu().double().numpy()
        rows = []
        for index, actions in enumerate(action_lists):
            row = logits[index, :len(actions)]
            if row.size:
                if not np.isfinite(row).all():
                    raise ValueError(f"Nonfinite policy logits for state {states[index].key}.")
                row = row - row.max()
                row = row - np.log(np.exp(row).sum())
            rows.append(row)
        return rows

    def score_values(self, states):
        if not states:
            return np.empty(0)
        result = self._score(states, [()] * len(states), value_only=True)
        values = result.value_tensor.detach().cpu().double().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite critic values in evaluation batch.")
        return values
