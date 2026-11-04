"""CPU invariance test for InvariantTwoFaceAgent under x -> x @ U.T with U in GL(4, Z).

Adapted from branch claude/project-thread-5a2oxn (eval/sweep/gl4z_invariance_20261103/gl4z_test.py).
The same 200 states (seed 0, same eligibility rule on records_sup.jsonl) and the same U sets
(identity, 20 signed permutations, 20 single shears, 20 double shears) are used. The transform is applied
to data.x before add_invariant_fields. A freshly initialized model (seed 0, eval mode) is used.
Extra checks: batch.x replaced by NaN, and random node/face/triangle permutations of the TwoFaceData.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch_geometric.data import Batch

from core.cy_two_face_data import TwoFaceData
from core.two_face_invariant_features import add_invariant_fields
from models.invariant_two_face_agent import InvariantTwoFaceAgent

R = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau")
DATA_DIR = R / "runs/probe_20261103/stageA"
OUT_DIR = R / "runs/inv_encoder_20261104"
SEED = 0
N_U = 20
N_PERM = 2
MIN_ACTIONS = 5
CONDITIONS = ("A", "B", "C1", "C2")
OLD_RECORDS_SHA = "869afb2629eea9181ad6bd0110ee721084339be6de9a3dac501f01e92e861334"
# The branch run drew its states from a 5184-record file (5164 eligible). The U sets depend only on how
# much of the seeded stream the state draw consumed, so the old draw is replayed with n=5164 for them.
OLD_N_ELIGIBLE = 5164
OLD_RESULTS = R / "runs/gl4z_invariance_20261103/results.json"


def build_model():
    torch.manual_seed(SEED)
    return InvariantTwoFaceAgent(in_channels=4, out_channels=64, hidden_channels=64, num_layers=3, act="silu",
                                 device="cpu", two_face_encoder="deep_sets", mp_layers=4).eval()


def elementary_shear(i, j, s):
    U = np.eye(4, dtype=np.int64)
    U[i, j] += s
    return U


def make_transforms(rng):
    def shear():
        i, j = rng.choice(4, 2, replace=False)
        return elementary_shear(int(i), int(j), int(rng.choice([-1, 1])))

    transforms = {"A": [np.eye(4, dtype=np.int64)], "B": [], "C1": [], "C2": []}
    for _ in range(N_U):
        U = np.zeros((4, 4), dtype=np.int64)
        U[np.arange(4), rng.permutation(4)] = rng.choice([-1, 1], size=4)
        transforms["B"].append(U)
    transforms["C1"] = [shear() for _ in range(N_U)]
    transforms["C2"] = [shear() @ shear() for _ in range(N_U)]
    return transforms


def check_unimodular(transforms):
    dets = {}
    for cond, mats in transforms.items():
        dets[cond] = []
        for U in mats:
            assert np.issubdtype(U.dtype, np.integer)
            det = int(round(float(np.linalg.det(U))))
            assert abs(det) == 1, f"det={det}"
            dets[cond].append(det)
    return dets


def load_states(n_states, rng):
    records_path = DATA_DIR / "records_sup.jsonl"
    rows = [json.loads(line) for line in records_path.open()]
    eligible = [r for r in rows if r["n_children"] >= MIN_ACTIONS]
    picked = rng.choice(len(eligible), size=n_states, replace=False)
    shards, states, skipped = {}, [], []
    for k in picked:
        r = eligible[k]
        if r["shard"] not in shards:
            shards[r["shard"]] = torch.load(DATA_DIR / "shards" / r["shard"], weights_only=False, map_location="cpu")
        data = shards[r["shard"]][r["shard_index"]]
        n = int(data.num_available_subcomplexes)
        if n != r["n_children"] or n != len(r["children"]):
            skipped.append(r["state_id"])
            continue
        states.append(dict(state_id=r["state_id"], data=data, n=n))
    sha = hashlib.sha256(records_path.read_bytes()).hexdigest()
    return dict(n_records=len(rows), n_eligible=len(eligible), sha256=sha,
                picked_state_ids=[eligible[k]["state_id"] for k in picked]), states, skipped


def permute_two_face(data, rng):
    """Consistently relabel nodes, faces and triangles and shuffle edge order. Action order is kept."""
    N, F, T = data.x.size(0), int(data.num_faces), data.triangle_vertices.size(0)
    perm_n, perm_f, perm_t = rng.permutation(N), rng.permutation(F), rng.permutation(T)
    inv_n = torch.as_tensor(np.argsort(perm_n))  # old node -> new node
    inv_f = torch.as_tensor(np.argsort(perm_f))
    inv_t = torch.as_tensor(np.argsort(perm_t))
    perm_n, perm_t = torch.as_tensor(perm_n), torch.as_tensor(perm_t)
    edge_index = inv_n[data.edge_index]
    edge_index = edge_index[:, torch.as_tensor(rng.permutation(edge_index.size(1)), dtype=torch.long)]
    dual = inv_t[data.triangle_edge_index]
    dual = dual[:, torch.as_tensor(rng.permutation(dual.size(1)), dtype=torch.long)]
    out = TwoFaceData()
    for key, value in data.items():
        out[key] = value
    out.x = data.x[perm_n]
    out.node_face = inv_f[data.node_face[perm_n]]
    out.edge_index = edge_index
    out.triangle_vertices = inv_n[data.triangle_vertices[perm_t]]
    out.triangle_face = inv_f[data.triangle_face[perm_t]]
    out.triangle_edge_index = dual
    out.action_face = inv_f[data.action_face]
    out.removed_triangle_ids = inv_t[data.removed_triangle_ids]
    out.added_triangle_vertices = inv_n[data.added_triangle_vertices]
    return out


def run(states, transforms, perm_rng):
    model = build_model()
    cond_list = [(c, u_idx, U) for c in CONDITIONS for u_idx, U in enumerate(transforms[c])]
    stats = {c: dict(logit=[], value=[], top1_changed=0, pairs=0) for c in CONDITIONS}
    perm = dict(logit=[], value=[], top1_changed=0, pairs=0)
    nan = dict(logit=[], value=[])
    max_logit_scale = 0.0
    for s, st in enumerate(states):
        base = add_invariant_fields(st["data"])
        graphs = [base]
        for _, _, U in cond_list:
            g = copy.deepcopy(st["data"])
            g.x = g.x @ torch.as_tensor(U, dtype=g.x.dtype).T
            graphs.append(add_invariant_fields(g))
        permuted = [add_invariant_fields(permute_two_face(st["data"], perm_rng)) for _ in range(N_PERM)]
        with torch.no_grad():
            value, logits = model.get_value_and_logits(Batch.from_data_list(graphs + permuted))
            nan_batch = Batch.from_data_list([base])
            nan_batch.x = torch.full_like(nan_batch.x, float("nan"))
            nan_value, nan_logits = model.get_value_and_logits(nan_batch)
        n = st["n"]
        value = value.double().numpy()
        logits = logits.double().numpy()[:, :n]
        assert np.isfinite(logits).all() and np.isfinite(value).all()
        assert np.isfinite(nan_logits.numpy()[:, :n]).all() and np.isfinite(nan_value.numpy()).all()
        l0, v0 = logits[0], value[0]
        max_logit_scale = max(max_logit_scale, float(np.abs(l0).max()))
        nan["logit"].append(float(np.abs(nan_logits.double().numpy()[0, :n] - l0).max()))
        nan["value"].append(float(abs(nan_value.double().numpy()[0] - v0)))
        top1_0 = int(np.argmax(l0))
        for p, (cond, _, _) in enumerate(cond_list):
            lk, vk = logits[p + 1], value[p + 1]
            stats[cond]["logit"].append(float(np.abs(lk - l0).max()))
            stats[cond]["value"].append(float(abs(vk - v0)))
            stats[cond]["top1_changed"] += int(int(np.argmax(lk)) != top1_0)
            stats[cond]["pairs"] += 1
        for q in range(N_PERM):
            lk, vk = logits[len(graphs) + q], value[len(graphs) + q]
            perm["logit"].append(float(np.abs(lk - l0).max()))
            perm["value"].append(float(abs(vk - v0)))
            perm["top1_changed"] += int(int(np.argmax(lk)) != top1_0)
            perm["pairs"] += 1
        if (s + 1) % 50 == 0:
            print(f"  {s + 1}/{len(states)} states", flush=True)
    result = {}
    for cond in CONDITIONS:
        result[cond] = dict(n_U=len(transforms[cond]), n_pairs=stats[cond]["pairs"],
                            max_abs_logit_diff=max(stats[cond]["logit"]), max_abs_V_diff=max(stats[cond]["value"]),
                            top1_changed=stats[cond]["top1_changed"])
    result["perm"] = dict(n_perm=N_PERM * len(states), max_abs_logit_diff=max(perm["logit"]),
                          max_abs_V_diff=max(perm["value"]), top1_changed=perm["top1_changed"])
    result["nan_batch_x"] = dict(n_states=len(states), max_abs_logit_diff=max(nan["logit"]),
                                 max_abs_V_diff=max(nan["value"]))
    result["max_abs_logit_scale"] = max_logit_scale
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-states", type=int, default=200)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "gl4z_invariance_results.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    sample_info, states, skipped = load_states(args.n_states, rng)
    u_rng = np.random.default_rng(SEED)
    u_rng.choice(OLD_N_ELIGIBLE, size=args.n_states, replace=False)
    transforms = make_transforms(u_rng)
    dets = check_unimodular(transforms)
    old = json.loads(OLD_RESULTS.read_text())["U_matrices"] if OLD_RESULTS.exists() else None
    u_match = old is not None and all(
        np.array_equal(np.asarray(transforms[c]), np.asarray(old[c])) for c in CONDITIONS)
    print(f"U sets identical to branch run: {u_match}", flush=True)
    print(f"records={sample_info['n_records']} eligible={sample_info['n_eligible']} used={len(states)} "
          f"skipped={len(skipped)} sha_matches_branch_run={sample_info['sha256'] == OLD_RECORDS_SHA}", flush=True)
    result = run(states, transforms, np.random.default_rng(SEED + 7))
    out = dict(seed=SEED, records=sample_info, n_states=len(states), skipped_state_ids=skipped,
               U_sets_identical_to_branch_run=u_match, state_sample_identical_to_branch_run=False,
               state_ids=[st["state_id"] for st in states],
               U_per_condition={c: len(transforms[c]) for c in CONDITIONS},
               det_per_condition={c: sorted(set(dets[c])) for c in CONDITIONS},
               U_matrices={c: [U.tolist() for U in transforms[c]] for c in CONDITIONS},
               results=result, runtime_sec=time.perf_counter() - t0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {args.out} runtime={out['runtime_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
