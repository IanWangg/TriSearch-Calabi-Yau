"""Shared pieces for the R2 trainers (PPO and expected-SARSA Q): network, episode driver, evaluation, logs.

JumpNetwork = InvariantTwoFaceAgent encoder (fresh init from the architecture in model_config.json,
no checkpoint weights) + move head (agent.actor_mlp, mean2) + value head on global features (PPO only).
Q-values are the move-head outputs. Episode driver: each slot runs one episode at a time through the
EpisodePool; a per-step callback receives (obs, action, reward, done, info, next_obs).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch

from rl_jump.policy import move_logits

ROOT = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau")
MODEL_CONFIG = ROOT / "runs/inv_rl_full_20261104/checkpoints/model_config.json"
TRAIN_ROWS = ROOT / "data/cy/favorable_v1/train/h11_20.samples.jsonl"
DEV_ROWS = ROOT / "data/cy/favorable_v1/dev/h11_20.samples.jsonl"


class JumpNetwork(nn.Module):
    def __init__(self, config_path=MODEL_CONFIG, *, seed=0):
        super().__init__()
        from models.subcomplex_policy_factory import build_subcomplex_agent

        saved = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if saved.get("subcomplex_actor_type") != "two_face_invariant" or saved.get("observation_kind") != "two_face":
            raise ValueError("model config is not an invariant two-face actor.")
        torch.manual_seed(int(seed))
        self.agent = build_subcomplex_agent(
            model_type=saved["model_type"], in_channels=saved["in_channels"], out_channels=saved["out_channels"],
            hidden_channels=saved["hidden_channels"], num_layers=saved["num_layers"], act=saved["activation"],
            subcomplex_actor_type=saved["subcomplex_actor_type"], two_face_encoder=saved["two_face_encoder"],
            device="cpu")
        channels = int(saved["out_channels"])
        self.value_head = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, 1))

    def forward(self, batch):
        """(candidate logits or Q-values, state values): logits (total,), values (num_graphs,)."""
        logits, global_features = move_logits(self.agent, batch, "mean2")
        return logits, self.value_head(global_features).squeeze(-1)


def graph_counts(batch):
    return [int(c) for c in batch.num_candidates.reshape(-1).tolist()]


def segment_log_softmax(values, counts):
    """Per-graph log-softmax over candidates (graphs with no candidates contribute nothing)."""
    chunks = torch.split(values, counts)
    return torch.cat([torch.log_softmax(chunk, dim=0) for chunk in chunks]) if values.numel() else values


def flat_indices(counts, actions):
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]) if len(counts) else np.zeros(0, dtype=np.int64)
    return torch.as_tensor(offsets + np.asarray(actions, dtype=np.int64), dtype=torch.long)


def to_batch(observations, device):
    return Batch.from_data_list([o.data for o in observations]).to(device)


def run_episodes(pool, assignments, act_fn, *, on_step=None):
    """Run one episode per assigned slot. act_fn(list of observations) -> (actions, aux list).

    Returns {slot: dict(start_value, best_value, final_value, final_obs, steps, rewards, done_reason)}.
    """
    obs = pool.begin(assignments)
    state = {}
    for slot, o in obs.items():
        state[slot] = dict(start_value=o.value, best_value=o.value, final_value=o.value, final_obs=o,
                           steps=0, rewards=[], done_reason=None)
    active = {slot for slot, o in obs.items() if o.candidates}
    for slot, o in obs.items():
        if slot not in active:
            state[slot]["done_reason"] = "dead_end"
    while active:
        slots = sorted(active)
        before = [obs[s] for s in slots]
        actions, aux = act_fn(before)
        results = pool.step({s: a for s, a in zip(slots, actions)})
        for s, o_before, a, x in zip(slots, before, actions, aux):
            new_obs, reward, done, info = results[s]
            st = state[s]
            st["steps"] += 1
            st["rewards"].append(float(reward))
            st["best_value"] = info["episode_best"]
            st["final_value"] = info["value"]
            st["final_obs"] = new_obs
            if on_step is not None:
                on_step(s, o_before, a, x, float(reward), bool(done), info, new_obs)
            obs[s] = new_obs
            if done:
                st["done_reason"] = info["done_reason"]
                active.discard(s)
    return state


def choose_random(observations, rng):
    actions = [int(rng.integers(o.data.num_candidates)) for o in observations]
    return actions, [None] * len(observations)


def choose_greedy(model, observations, device):
    with torch.no_grad():
        batch = to_batch(observations, device)
        logits, _ = model(batch)
        counts = graph_counts(batch)
        actions = [int(np.argmax(chunk.cpu().numpy())) for chunk in torch.split(logits, counts)]
    return actions, [None] * len(observations)


def evaluate(pool, dev_lines, act_fn, *, slots, rows_path=DEV_ROWS):
    """Protocol: every dev polytope once from its start 0, H as configured in the pool, deterministic policy.

    Returns the list of per-polytope records (line, start_value, best_dlnV, final_dlnV, steps).
    """
    records = []
    for offset in range(0, len(dev_lines), slots):
        chunk = dev_lines[offset:offset + slots]
        assignments = {slot: (str(rows_path), int(line), 0) for slot, line in enumerate(chunk)}
        state = run_episodes(pool, assignments, act_fn)
        for slot, line in enumerate(chunk):
            st = state[slot]
            records.append(dict(line=int(line), start_value=float(st["start_value"]),
                                best_dlnV=math.log(st["best_value"]) - math.log(st["start_value"]),
                                final_dlnV=math.log(st["final_value"]) - math.log(st["start_value"]),
                                steps=st["steps"], done_reason=st["done_reason"]))
    return records


def summarize_eval(records):
    best = np.array([r["best_dlnV"] for r in records])
    final = np.array([r["final_dlnV"] for r in records])
    return dict(n_polytopes=len(records), mean_best_dlnV=float(best.mean()),
                median_best_dlnV=float(np.median(best)), mean_final_dlnV=float(final.mean()),
                max_best_dlnV=float(best.max()))


class JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=float) + "\n")


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def explained_variance(prediction, target):
    var = np.var(target)
    if var <= 1e-12:
        return float("nan")
    return float(1.0 - np.var(target - prediction) / var)


def now():
    return time.perf_counter()
