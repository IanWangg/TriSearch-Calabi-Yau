"""Collect data for the value-generalization / supervised probe (SPEC.md, 2026-11-03).

Each state is a uniform random walk of L ~ Uniform{0..30} flips from an FRST start of a
favorable v1 polytope. Walks use per-walk seeded RNGs, so stage B replays exactly the
states of stage A (replayed keys are checked against stage A).

Stage A (all sets): 1-hop Kcup labels for every state and its canonical children,
  TwoFaceData of each state with its actions (torch shards), and for S_crit_train and
  S_dev the outputs of each checkpoint: V(s), log pi(.|s), V(child).
Stage B (S_crit_train and S_dev): y2 (best Kcup within 2 flips, excluding s itself) on a
  seeded 25% subset, per-child 2-hop reach y1(child), and M Monte Carlo rollouts of
  MC_STEPS steps under the policy of checkpoint G (MC returns for the critic target).

Examples:
  python tools/probe_collect.py --stage A --out_dir runs/probe_20261103
  python tools/probe_collect.py --stage B --out_dir runs/probe_20261103
  python tools/probe_collect.py --stage A --sets dev --limit_per_set 1 --states 2 --workers 2 \
      --out_dir runs/probe_20261103/smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mdp.cy_rollout import (  # noqa: E402
    CYRandomRolloutEngine, build_cy_rollout_collection, create_transition_pool, load_cy_sample_rows,
)
from mdp.cy_state_record import state_key  # noqa: E402
from core.cy_managed_runtime import managed_rollout_runtime  # noqa: E402
from core.cy_policy_inference import build_cy_data_list  # noqa: E402
from reward_functions import get_reward  # noqa: E402
from tune_rl_search import KcupCache, install_cache  # noqa: E402

ORIGINAL_OBJECTIVE_VALUES = CYRandomRolloutEngine.objective_values  # captured before the cache is installed

SET_CODE = {"sup": 1, "crit": 2, "dev": 3}
SPLIT_CODE = {"train": 0, "dev": 1}
DEFAULT_CKPTS = {
    "G": "runs/fv1_mix_g099_s0_20261103_0604/checkpoints/200.pth",
    "R": "runs/fv1_R1_mix_s0_20261102_2207/checkpoints/200.pth",
    "G50": "runs/fv1_mix_g099_s0_20261103_0604/checkpoints/50.pth",
}
MAX_WALK = 30
GROUP_POLYTOPES = 16
GAMMA = 0.99


def log(message):
    print(message, flush=True)


# ---------------------------------------------------------------- selection

def select_polytopes(data_root, args):
    """Seeded polytope selection. Returns entries with set/split/h11/role/row."""
    train = {h: load_cy_sample_rows(str(data_root / "train" / f"h11_{h}.samples.jsonl")) for h in (15, 20)}
    dev = {h: load_cy_sample_rows(str(data_root / "dev" / f"h11_{h}.samples.jsonl")) for h in (15, 20)}
    rng = np.random.default_rng([args.seed, 11])
    sup_counts = {15: args.sup_h15, 20: args.sup_h20}
    crit_counts = {15: args.crit_h15, 20: args.crit_h20}
    entries = []
    sup_ids = set()
    for h in (15, 20):
        pick = np.sort(rng.choice(len(train[h]), sup_counts[h], replace=False))
        n_val = max(1, round(0.1 * sup_counts[h]))
        val_positions = set(rng.choice(len(pick), n_val, replace=False).tolist())
        for position, row_index in enumerate(pick):
            row = train[h][row_index]
            sup_ids.add((h, int(row["polytope_index"])))
            entries.append(dict(set="sup", split="train", h11=h, role="val" if position in val_positions else "train",
                                row=row))
    crit_rng = np.random.default_rng([args.seed, 12])
    for h in (15, 20):
        candidates = [i for i, row in enumerate(train[h]) if (h, int(row["polytope_index"])) not in sup_ids]
        pick = np.sort(crit_rng.choice(candidates, crit_counts[h], replace=False))
        for row_index in pick:
            entries.append(dict(set="crit", split="train", h11=h, role="crit", row=train[h][row_index]))
    for h in (15, 20):
        for row in dev[h]:
            entries.append(dict(set="dev", split="dev", h11=h, role="dev", row=row))
    if args.limit_per_set:
        kept, counts = [], {}
        for entry in entries:
            key = (entry["set"], entry["h11"])
            if counts.get(key, 0) < args.limit_per_set:
                counts[key] = counts.get(key, 0) + 1
                kept.append(entry)
        entries = kept
    entries = [entry for entry in entries if entry["set"] in args.sets]
    for entry in entries:
        entry["polytope_index"] = int(entry["row"]["polytope_index"])
    return entries


def walk_specs(entries, states_per_set, seed):
    """One spec per state. The RNG of each walk depends only on its identity."""
    specs = []
    for entry in entries:
        n_starts = len(entry["row"]["frst_list"])
        for j in range(states_per_set[entry["set"]]):
            rng = np.random.default_rng([seed, 21, SET_CODE[entry["set"]], SPLIT_CODE[entry["split"]],
                                         entry["h11"], entry["polytope_index"], j])
            specs.append(dict(
                entry=entry, state_index=j, state_id=f'{entry["split"]}:{entry["h11"]}:{entry["polytope_index"]}:{j}',
                start_index=int(rng.integers(n_starts)), walk_target=int(rng.integers(0, MAX_WALK + 1)), rng=rng,
            ))
    return specs


def groups_of(specs):
    """Group consecutive specs by polytope, GROUP_POLYTOPES polytopes per batch."""
    by_polytope = {}
    for spec in specs:
        by_polytope.setdefault((spec["entry"]["set"], spec["entry"]["h11"], spec["entry"]["polytope_index"]), []).append(spec)
    keys = list(by_polytope)
    for start in range(0, len(keys), GROUP_POLYTOPES):
        yield [spec for key in keys[start:start + GROUP_POLYTOPES] for spec in by_polytope[key]]


# ---------------------------------------------------------------- engine wrapper

class Collector:
    def __init__(self, engine, collection, cache, policies):
        self.engine = engine
        self.collection = collection
        self.cache = cache
        self.policies = policies
        self.expansions = 0
        self.objective_calls = 0
        self.done = 0
        self.total = 0
        self.started = time.time()

    def objective(self, states):
        states = list(states)
        self.objective_calls += len(states)
        if not states:
            return np.empty(0)
        values = np.array(list(self.engine.objective_values(states, "max_kcup")), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Invalid Kcup objective in probe collection.")
        return np.log(values)

    def expand(self, states):
        if not states:
            return [], []
        actions, _ = self.engine.candidate_actions_for_states(states)
        transitions = [self.engine.nodes_by_key[str(state.key)].transitions for state in states]
        self.expansions += len(states)
        return [tuple(a) for a in actions], transitions

    def base_state(self, spec):
        entry = spec["entry"]
        simplices = entry["row"]["frst_list"][spec["start_index"]]["simplices"]
        key = state_key(entry["polytope_index"], simplices, "two_neighbors")
        return self.collection.base_states[key]

    def children(self, states):
        """Per state: (actions, child states, child lnV) in canonical action order."""
        actions, transitions = self.expand(states)
        per_state, flat = [], []
        for state, acts, trans in zip(states, actions, transitions):
            children = [self.engine.materialize_transition(state, trans[a]) for a in acts]
            per_state.append((acts, children))
            flat.extend(children)
        values = self.objective(flat)
        out, offset = [], 0
        for acts, children in per_state:
            out.append((acts, children, values[offset:offset + len(children)]))
            offset += len(children)
        return out

    def walk(self, specs):
        current = [self.base_state(spec) for spec in specs]
        steps = [0] * len(specs)
        stuck = [False] * len(specs)
        for t in range(max([spec["walk_target"] for spec in specs], default=0)):
            active = [i for i, spec in enumerate(specs) if not stuck[i] and spec["walk_target"] > t]
            if not active:
                break
            actions, transitions = self.expand([current[i] for i in active])
            for position, i in enumerate(active):
                if not actions[position]:
                    stuck[i] = True
                    continue
                choice = specs[i]["rng"].integers(len(actions[position]))
                current[i] = self.engine.materialize_transition(current[i], transitions[position][actions[position][choice]])
                steps[i] += 1
            self.engine.release_active_states()
        return current, steps, stuck

    def progress(self, label, done, total, extra=""):
        elapsed = time.time() - self.started
        queries = self.cache.hits + self.cache.misses
        hit_rate = self.cache.hits / max(1, queries)
        log(f"[{label}] {done}/{total} states elapsed={elapsed:.0f}s logical_queries={queries} "
            f"queries_per_s={queries / max(elapsed, 1e-9):.1f} cache_hit_rate={hit_rate:.3f} "
            f"expansions={self.expansions}{extra}")


# ---------------------------------------------------------------- stage A

def stage_a_group(collector, specs, sanity_ids, sanity_log, shard_path, records_stream, label, use_policies):
    states, steps, stuck = collector.walk(specs)
    lnv = collector.objective(states)
    children = collector.children(states)
    action_lists = [acts for acts, _, _ in children]
    datas = build_cy_data_list(states, action_lists, observation_kind="two_face")
    flat_children = [child for _, kids, _ in children for child in kids]
    checkpoint_out = {name: {} for name in (collector.policies if use_policies else {})}
    for name, policy in (collector.policies.items() if use_policies else []):
        values = np.asarray(policy.score_values(states), dtype=float)
        logps = policy.score_actions(states, action_lists)
        child_values = np.asarray(policy.score_values(flat_children), dtype=float)
        offset = 0
        for i, (acts, kids, _) in enumerate(children):
            checkpoint_out[name][i] = dict(
                V=float(values[i]), logp=[float(x) for x in logps[i]],
                V_child=[float(x) for x in child_values[offset:offset + len(kids)]])
            offset += len(kids)
    if sanity_log is not None:
        sanity_check(collector, specs, states, lnv, checkpoint_out, sanity_ids, sanity_log)
    torch_save(datas, shard_path)
    for i, spec in enumerate(specs):
        acts, kids, kid_lnv = children[i]
        entry = spec["entry"]
        record = dict(
            state_id=spec["state_id"], set=entry["set"], split=entry["split"], role=entry["role"], h11=entry["h11"],
            polytope_index=entry["polytope_index"], start_index=spec["start_index"], state_index=spec["state_index"],
            walk_target=spec["walk_target"], walk_length=steps[i], stuck=bool(stuck[i]),
            state_key=str(states[i].key), lnV=float(lnv[i]), n_children=len(acts),
            children=[[str(child.key), float(value)] for child, value in zip(kids, kid_lnv)],
            y1=float(max(kid_lnv) - lnv[i]) if len(kids) else None,
            shard=shard_path.name, shard_index=i,
            ckpt={name: checkpoint_out[name][i] for name in checkpoint_out} if checkpoint_out else {},
        )
        records_stream.write(json.dumps(record) + "\n")
    records_stream.flush()
    collector.engine.prune_runtime_caches()
    collector.done += len(specs)
    collector.progress(label, collector.done, collector.total)
    return len(specs)


def torch_save(datas, path):
    import torch

    torch.save(datas, path)


def sanity_check(collector, specs, states, lnv, checkpoint_out, sanity_ids, sanity_log):
    """(a) lnV equals an uncached geometry recomputation and the cache; (b) G's V and log pi
    from single-state calls equal the batched values (the eval/policy.py path)."""
    policy = collector.policies.get("G")
    single_checked = 0
    for i, spec in enumerate(specs):
        if spec["state_id"] not in sanity_ids:
            continue
        state = states[i]
        fresh = math.log(float(list(ORIGINAL_OBJECTIVE_VALUES(collector.engine, [state], "max_kcup"))[0]))
        cached = collector.cache.get_many("max_kcup", [state.key])[0]
        entry = dict(state_id=spec["state_id"], lnV=float(lnv[i]), fresh_diff=abs(float(lnv[i]) - fresh),
                     cache_diff=(abs(float(lnv[i]) - math.log(cached)) if cached is not None else None))
        if policy is not None and single_checked < 10:
            acts, _ = collector.expand([state])
            v1 = float(np.asarray(policy.score_values([state]))[0])
            lp1 = policy.score_actions([state], acts)[0]
            entry["V_diff"] = abs(v1 - checkpoint_out["G"][i]["V"])
            entry["logp_maxdiff"] = float(np.max(np.abs(np.asarray(lp1) - np.asarray(checkpoint_out["G"][i]["logp"])))) \
                if len(lp1) else 0.0
            single_checked += 1
        sanity_log.append(entry)


def run_stage_a(args, collector, entries, out_dir, runtime_seed_rng):
    stage_dir = out_dir / "stageA"
    (stage_dir / "shards").mkdir(parents=True, exist_ok=True)
    sanity_ids = set()
    dev_specs_all = walk_specs([e for e in entries if e["set"] == "dev"], {"dev": args.states_dev or 16}, args.seed)
    if dev_specs_all:
        chosen = runtime_seed_rng.choice(len(dev_specs_all), min(args.sanity_n, len(dev_specs_all)), replace=False)
        sanity_ids = {dev_specs_all[k]["state_id"] for k in chosen}
    sanity_log = []
    summary = {}
    for set_name in ("sup", "crit", "dev"):
        set_entries = [e for e in entries if e["set"] == set_name]
        if not set_entries:
            continue
        states_per_set = {"sup": args.states_sup, "crit": args.states_crit, "dev": args.states_dev}
        specs = walk_specs(set_entries, {set_name: states_per_set[set_name]}, args.seed)
        n_records = 0
        collector.done, collector.total = 0, len(specs)
        with (stage_dir / f"records_{set_name}.jsonl").open("w") as stream:
            for g, group in enumerate(groups_of(specs)):
                shard = stage_dir / "shards" / f"{set_name}_{g:04d}.pt"
                n_records += stage_a_group(collector, group, sanity_ids if set_name == "dev" else set(),
                                           sanity_log if set_name == "dev" else None, shard, stream,
                                           f"A {set_name} group {g}", use_policies=set_name != "sup")
        summary[set_name] = dict(polytopes=len(set_entries), states=n_records)
    if sanity_log:
        (out_dir / "sanity_a_b.json").write_text(json.dumps(sanity_log, indent=1))
        worst_a = max(e["fresh_diff"] for e in sanity_log)
        cache_vals = [e["cache_diff"] for e in sanity_log if e["cache_diff"] is not None]
        worst_b = max([e.get("V_diff", 0.0) for e in sanity_log] + [0.0])
        worst_c = max([e.get("logp_maxdiff", 0.0) for e in sanity_log] + [0.0])
        summary["sanity"] = dict(checked=len(sanity_log), max_lnV_fresh_diff=worst_a,
                                 max_lnV_cache_diff=max(cache_vals) if cache_vals else None,
                                 max_V_single_vs_batch=worst_b, max_logp_single_vs_batch=worst_c,
                                 single_state_checks=sum(1 for e in sanity_log if "V_diff" in e))
    return summary


# ---------------------------------------------------------------- stage B

def load_stage_a_keys(out_dir, set_names):
    keys = {}
    for set_name in set_names:
        path = out_dir / "stageA" / f"records_{set_name}.jsonl"
        if not path.exists():
            continue
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                keys[record["state_id"]] = record["state_key"]
    return keys


def run_stage_b(args, collector, entries, out_dir, policy_g):
    stage_dir = out_dir / "stageB"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_keys = load_stage_a_keys(out_dir, ("crit", "dev"))
    summary = {}
    for set_name in ("crit", "dev"):
        set_entries = [e for e in entries if e["set"] == set_name]
        if not set_entries:
            continue
        states_per_set = {"crit": args.states_crit, "dev": args.states_dev}
        specs = walk_specs(set_entries, {set_name: states_per_set[set_name]}, args.seed)
        n_states = n_y2 = 0
        y2_rng = np.random.default_rng([args.seed, 31, SET_CODE[set_name]])
        y2_mask = set(specs[k]["state_id"] for k in y2_rng.choice(len(specs), round(args.y2_frac * len(specs)),
                                                                   replace=False).tolist())
        with (stage_dir / f"{set_name}.jsonl").open("w") as stream:
            for g, group in enumerate(groups_of(specs)):
                states, _, _ = collector.walk(group)
                for spec, state in zip(group, states):
                    expected = stage_keys.get(spec["state_id"])
                    if expected is not None and expected != str(state.key):
                        raise RuntimeError(f"Replay mismatch for {spec['state_id']}.")
                lnv = collector.objective(states)
                selected = [i for i, spec in enumerate(group) if spec["state_id"] in y2_mask]
                y2_info = {}
                if selected:
                    first = collector.children([states[i] for i in selected])
                    grand = collector.children([child for _, kids, _ in first for child in kids])
                    offset = 0
                    for position, i in enumerate(selected):
                        _, kids, kid_lnv = first[position]
                        reach, child_y1 = list(kid_lnv), []
                        for c, child in enumerate(kids):
                            _, gkids, gvals = grand[offset]
                            offset += 1
                            keep = [v for g_, v in zip(gkids, gvals) if str(g_.key) != str(states[i].key)]
                            child_y1.append(float(max(keep) - kid_lnv[c]) if keep else None)
                            reach.extend(keep)
                        y2_info[i] = dict(y2=float(max(reach) - lnv[i]) if reach else None, child_y1=child_y1)
                mc = monte_carlo(collector, group, states, lnv, policy_g, args.mc_m, args.mc_steps)
                for i, spec in enumerate(group):
                    info = y2_info.get(i, {})
                    stream.write(json.dumps(dict(state_id=spec["state_id"], set=set_name, h11=spec["entry"]["h11"],
                                                 polytope_index=spec["entry"]["polytope_index"],
                                                 y2=info.get("y2"), child_y1=info.get("child_y1"),
                                                 mc=mc[i])) + "\n")
                n_states += len(group)
                n_y2 += len(selected)
                collector.engine.prune_runtime_caches()
                collector.progress(f"B {set_name} group {g}", n_states, len(specs), extra=f" y2_states={n_y2}")
        summary[set_name] = dict(polytopes=len(set_entries), states=n_states, y2_states=n_y2)
    return summary


def monte_carlo(collector, group, states, lnv0, policy, m_count, steps):
    """M stochastic policy rollouts per state; each trajectory has its own RNG."""
    trajectories = [(i, m) for i in range(len(states)) for m in range(m_count)]
    rngs = [np.random.default_rng([41, SET_CODE[group[i]["entry"]["set"]],
                                   SPLIT_CODE[group[i]["entry"]["split"]], group[i]["entry"]["h11"],
                                   group[i]["entry"]["polytope_index"], group[i]["state_index"], m])
            for i, m in trajectories]
    current = [states[i] for i, _ in trajectories]
    logv = np.zeros((len(trajectories), steps + 1))
    logv[:, 0] = [lnv0[i] for i, _ in trajectories]
    returns = np.zeros(len(trajectories))
    stuck = np.zeros(len(trajectories), dtype=bool)
    for t in range(steps):
        active = [k for k in range(len(trajectories)) if not stuck[k]]
        if not active:
            logv[:, t + 1:] = logv[:, [t]]
            break
        actions, transitions = collector.expand([current[k] for k in active])
        logps = policy.score_actions([current[k] for k in active], actions) if active else []
        moved = {}
        for position, k in enumerate(active):
            if not actions[position]:
                stuck[k] = True
                continue
            probs = np.exp(logps[position])
            choice = min(int(np.searchsorted(np.cumsum(probs), rngs[k].random(), side="right")), len(probs) - 1)
            moved[k] = collector.engine.materialize_transition(current[k], transitions[position][actions[position][choice]])
        collector.engine.release_active_states()
        keys = list(moved)
        new_values = collector.objective([moved[k] for k in keys])
        for value, k in zip(new_values, keys):
            current[k] = moved[k]
            returns[k] += GAMMA ** t * (value - logv[k, t])
            logv[k, t + 1] = value
        for k in range(len(trajectories)):
            if k not in moved:
                logv[k, t + 1] = logv[k, t]
    end_values = np.zeros(len(trajectories))
    alive = [k for k in range(len(trajectories)) if not stuck[k]]
    if alive:
        end_values[alive] = np.asarray(policy.score_values([current[k] for k in alive]), dtype=float)
    boot = returns + GAMMA ** steps * np.where(stuck, 0.0, end_values)
    best = logv.max(axis=1) - logv[:, 0]
    final = logv[:, -1] - logv[:, 0]
    out = []
    for i in range(len(states)):
        rows = [k for k, (ii, _) in enumerate(trajectories) if ii == i]
        out.append(dict(G20=[float(returns[k]) for k in rows], boot=[float(boot[k]) for k in rows],
                        best=[float(best[k]) for k in rows], final=[float(final[k]) for k in rows],
                        stuck=[bool(stuck[k]) for k in rows]))
    return out


# ---------------------------------------------------------------- main

def load_policy(path):
    from eval.config import EvaluationSpec
    from eval.policy import EvaluationPolicy

    spec = EvaluationSpec(num_polytopes=1, h11=15, num_starts=1, objective_budget=1,
                          subcomplex_actor_type="two_face_deep_sets", policy_checkpoint=str(path), force_cpu=True)
    return EvaluationPolicy.from_spec(spec)


def parse_ckpts(text):
    out = {}
    for item in text.split(","):
        name, _, path = item.partition("=")
        out[name.strip()] = str(Path(path.strip()) if Path(path.strip()).is_absolute() else MAIN_ROOT / path.strip())
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("A", "B"), required=True)
    parser.add_argument("--out_dir", default=str(MAIN_ROOT / "runs/probe_20261103"))
    parser.add_argument("--data_root", default=str(MAIN_ROOT / "data/cy/favorable_v1"))
    parser.add_argument("--cache", default=str(MAIN_ROOT / "runs/fv1_infer_tune/kcup_cache.sqlite3"))
    parser.add_argument("--sets", default="sup,crit,dev")
    parser.add_argument("--ckpts", default=",".join(f"{k}={v}" for k, v in DEFAULT_CKPTS.items()))
    parser.add_argument("--mc_ckpt", default="G")
    parser.add_argument("--workers", type=int, default=16, help="Geometry workers (at most 16).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sup_h15", type=int, default=300)
    parser.add_argument("--sup_h20", type=int, default=150)
    parser.add_argument("--crit_h15", type=int, default=60)
    parser.add_argument("--crit_h20", type=int, default=40)
    parser.add_argument("--states_sup", type=int, default=12)
    parser.add_argument("--states_crit", type=int, default=8)
    parser.add_argument("--states_dev", type=int, default=16)
    parser.add_argument("--states", type=int, help="Override states per polytope in every set (smoke tests).")
    parser.add_argument("--limit_per_set", type=int, default=0, help="Keep the first N polytopes per (set, h11).")
    parser.add_argument("--mc_m", type=int, default=4)
    parser.add_argument("--mc_steps", type=int, default=20)
    parser.add_argument("--y2_frac", type=float, default=0.25)
    parser.add_argument("--sanity_n", type=int, default=20)
    parser.add_argument("--torch_threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be between 1 and 16 (geometry worker budget).")
    if args.states is not None:
        args.states_sup = args.states_crit = args.states_dev = args.states
    args.sets = [s for s in args.sets.split(",") if s]
    if args.stage == "B":
        args.sets = [s for s in args.sets if s in ("crit", "dev")]

    import torch

    torch.set_num_threads(args.torch_threads)
    os.environ.setdefault("MOSEKLM_LICENSE_FILE", str(Path.home() / "mosek/mosek.lic"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    entries = select_polytopes(data_root, args)
    log(f"selected {len(entries)} polytopes: " + ", ".join(
        f"{s}/h{h}={sum(1 for e in entries if e['set'] == s and e['h11'] == h)}"
        for s in ("sup", "crit", "dev") for h in (15, 20)))

    ckpts = parse_ckpts(args.ckpts)
    if args.stage == "A":
        policies = {name: load_policy(path) for name, path in ckpts.items()}
    else:
        policies = {args.mc_ckpt: load_policy(ckpts[args.mc_ckpt])}
    cache = KcupCache(args.cache)
    install_cache(cache)
    reward = get_reward("max_kcup", mori_cap_backend="cytools")
    rows = [entry["row"] for entry in entries]
    runtime_args = Namespace(
        transition_num_workers=args.workers, use_multiprocessing=args.workers > 1,
        transition_mp_start_method="spawn", action_order="canonical", runtime_cache_gb=1.0,
        memory_budget_gb=64.0, transition_task_timeout_sec=300.0, policy_max_graph_size=250000,
        profile_cuda_timing=False,
    )
    started = time.time()
    with managed_rollout_runtime(runtime_args, create_transition_pool) as (pool, cache_bytes, register_engine):
        collection = build_cy_rollout_collection(
            rows, include_points_interior_to_facets=False, neighbor_mode="two_neighbors", transition_pool=pool,
            two_face_state=False, include_two_face_metadata=True)
        history = out_dir / "runtime" / f"{args.stage}_{uuid.uuid4().hex[:8]}.sqlite3"
        history.parent.mkdir(parents=True, exist_ok=True)
        engine = CYRandomRolloutEngine(
            collection=collection, include_points_interior_to_facets=False, neighbor_mode="two_neighbors",
            reward_function=reward, state_cache_mode="lru", max_hot_states=100000,
            cache_budget_bytes=cache_bytes, history_path=str(history), action_order="canonical",
            two_face_state=False)
        register_engine(engine)
        log(f"collection ready: {len(collection.base_states)} base states, {len(rows)} polytopes, "
            f"{time.time() - started:.0f}s")
        collector = Collector(engine, collection, cache, policies)
        if args.stage == "A":
            summary = run_stage_a(args, collector, entries, out_dir, np.random.default_rng([args.seed, 51]))
        else:
            summary = run_stage_b(args, collector, entries, out_dir, policies[args.mc_ckpt])
    wall = time.time() - started
    queries = cache.hits + cache.misses
    report = dict(stage=args.stage, args=vars(args), counts=summary, wall_sec=wall, logical_queries=queries,
                  cache_hits=cache.hits, cache_misses=cache.misses,
                  cache_hit_rate=cache.hits / max(1, queries), queries_per_sec=queries / max(wall, 1e-9),
                  expansions=collector.expansions)
    (out_dir / f"stage{args.stage}_summary.json").write_text(json.dumps(report, indent=2, default=str))
    log(json.dumps({k: report[k] for k in ("stage", "counts", "wall_sec", "logical_queries", "cache_hit_rate")}))


if __name__ == "__main__":
    main()
