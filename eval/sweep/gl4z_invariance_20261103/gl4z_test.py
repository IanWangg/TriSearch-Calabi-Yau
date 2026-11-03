"""CPU invariance test for TwoFaceAgent under x -> x @ U.T with U in GL(4, Z).

The policy reads raw lattice coordinates, so a unimodular change of basis
should leave the geometry (and ideally the policy) unchanged. Any change in
actor logits or value measured here is pure error.
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

import core.cy_two_face_data  # noqa: F401  (shards unpickle TwoFaceData)
from core.cy_runtime_utils import load_policy_checkpoint
from models.subcomplex_policy_config import resolve_two_face_encoder_options
from models.subcomplex_policy_factory import build_subcomplex_agent

R = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau")
DATA_DIR = R / "runs/probe_20261103/stageA"
CHECKPOINTS = {
    "G": R / "runs/fv1_mix_g099_s0_20261103_0604/checkpoints/200.pth",
    "G50": R / "runs/fv1_mix_g099_s0_20261103_0604/checkpoints/50.pth",
}
OUT_DIR = R / "runs/gl4z_invariance_20261103"
SEED = 0
N_U = 20
N_BOOT = 1000
MIN_ACTIONS = 5
CONDITIONS = ("A", "B", "C1", "C2")


def load_model(ckpt):
    saved = json.loads((ckpt.parent / "model_config.json").read_text())
    encoder = resolve_two_face_encoder_options(None, None, saved_config=saved)
    model = build_subcomplex_agent(
        model_type="egnn", in_channels=4, out_channels=64, hidden_channels=64, num_layers=3,
        share_encoder=True, mlp_hidden_channel_list=[64], act="silu",
        subcomplex_actor_type="two_face_deep_sets", device="cpu", **encoder)
    load_policy_checkpoint(model, str(ckpt), map_location=torch.device("cpu"))
    return model.eval()


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
            det = float(np.linalg.det(U))
            assert np.issubdtype(U.dtype, np.integer), "non-integer U"
            assert abs(det - round(det)) < 1e-9 and abs(abs(round(det)) - 1) == 0, f"det={det}"
            dets[cond].append(int(round(det)))
    return dets


def spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def bootstrap_ci(per_state, rng):
    idx = rng.integers(0, len(per_state), size=(N_BOOT, len(per_state)))
    means = np.nanmean(per_state[idx], axis=1)
    return [float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5))]


def load_states(n_states, rng):
    rows = [json.loads(line) for line in (DATA_DIR / "records_sup.jsonl").open()]
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
        child_lnV = np.array([c[1] for c in r["children"]], dtype=np.float64)
        assert np.isfinite(child_lnV).all() and np.isfinite(r["lnV"])
        states.append(dict(state_id=r["state_id"], data=data, n=n, dlnV=child_lnV - r["lnV"]))
    return len(rows), len(eligible), states, skipped


def run_checkpoint(name, ckpt, states, transforms):
    model = load_model(ckpt)
    cond_list = [(c, u_idx, U) for c in CONDITIONS for u_idx, U in enumerate(transforms[c])]
    n_pairs = len(cond_list)
    shape = (len(states), n_pairs)
    rho_self, top1, top4, dV, rho_true_U, maxabs_U, logit_diff = (np.full(shape, np.nan) for _ in range(7))
    V0, rho_true_0, maxabs_0 = np.zeros(len(states)), np.full(len(states), np.nan), np.zeros(len(states))
    for s, st in enumerate(states):
        graphs = [st["data"]]
        for _, _, U in cond_list:
            g = copy.deepcopy(st["data"])
            g.x = g.x @ torch.as_tensor(U, dtype=g.x.dtype).T
            graphs.append(g)
        batch = Batch.from_data_list(graphs)
        with torch.no_grad():
            value, logits = model.get_value_and_logits(batch)
        value = value.double().numpy()
        logits = logits.double().numpy()[:, :st["n"]]
        assert np.isfinite(logits).all() and np.isfinite(value).all()
        l0, v0 = logits[0], value[0]
        V0[s] = v0
        rho_true_0[s] = spearman(l0, st["dlnV"])
        maxabs_0[s] = float(graphs[0].x.abs().max())
        top4_0 = set(np.argsort(-l0, kind="stable")[:4])
        for p, (_, _, _) in enumerate(cond_list):
            lk, vk, gk = logits[p + 1], value[p + 1], graphs[p + 1]
            rho_self[s, p] = spearman(l0, lk)
            top1[s, p] = float(np.argmax(l0) == np.argmax(lk))
            top4[s, p] = len(top4_0 & set(np.argsort(-lk, kind="stable")[:4])) / 4
            dV[s, p] = abs(vk - v0)
            rho_true_U[s, p] = spearman(lk, st["dlnV"])
            maxabs_U[s, p] = float(gk.x.abs().max())
            logit_diff[s, p] = float(np.abs(lk - l0).max())
        if (s + 1) % 50 == 0:
            print(f"  [{name}] {s + 1}/{len(states)} states", flush=True)

    rng = np.random.default_rng(SEED + 1)
    result = {}
    offset = 0
    for cond in CONDITIONS:
        k = len(transforms[cond])
        cols = slice(offset, offset + k)
        offset += k
        rs = rho_self[:, cols]
        t1 = top1[:, cols]
        rs_state = np.nanmean(rs, axis=1)
        t1_state = np.nanmean(t1, axis=1)
        result[cond] = {
            "n_states": len(states),
            "n_U": k,
            "n_pairs": int(rs.size),
            "rho_self_mean": float(np.nanmean(rs)),
            "rho_self_mean_ci95": bootstrap_ci(rs_state, rng),
            "rho_self_p10": float(np.nanpercentile(rs, 10)),
            "rho_self_nan_pairs": int(np.isnan(rs).sum()),
            "top1_agree": float(np.nanmean(t1)),
            "top1_agree_ci95": bootstrap_ci(t1_state, rng),
            "top4_overlap": float(np.nanmean(top4[:, cols])),
            "dV_mean_abs": float(np.nanmean(dV[:, cols])),
            "V_orig_std_over_states": float(np.std(V0, ddof=1)),
            "rho_true_orig": float(np.nanmean(rho_true_0)),
            "rho_true_U": float(np.nanmean(rho_true_U[:, cols])),
            "rho_true_nan_pairs_U": int(np.isnan(rho_true_U[:, cols]).sum()),
            "max_abs_x_after": float(np.nanmean(maxabs_U[:, cols])),
            "max_abs_x_orig": float(np.mean(maxabs_0)),
            "max_abs_logit_diff": float(np.nanmax(logit_diff[:, cols])),
            "max_abs_V_diff": float(np.nanmax(dV[:, cols])),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-states", type=int, default=200)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "results.json")
    args = parser.parse_args()
    torch.set_num_threads(8)
    t0 = time.perf_counter()

    rng = np.random.default_rng(SEED)
    n_records, n_eligible, states, skipped = load_states(args.n_states, rng)
    transforms = make_transforms(rng)
    dets = check_unimodular(transforms)
    print(f"records={n_records} eligible={n_eligible} used={len(states)} skipped={len(skipped)}", flush=True)

    checkpoints = {}
    for name, ckpt in CHECKPOINTS.items():
        print(f"checkpoint {name}: {ckpt}", flush=True)
        checkpoints[name] = dict(path=str(ckpt), conditions=run_checkpoint(name, ckpt, states, transforms))

    out = dict(
        seed=SEED, records_sha256=hashlib.sha256((DATA_DIR / "records_sup.jsonl").read_bytes()).hexdigest(),
        n_records=n_records, n_eligible=n_eligible, n_states=len(states),
        skipped_count=len(skipped), skipped_state_ids=skipped,
        U_per_condition={c: len(transforms[c]) for c in CONDITIONS},
        det_per_condition={c: sorted(set(dets[c])) for c in CONDITIONS},
        U_matrices={c: [U.tolist() for U in transforms[c]] for c in CONDITIONS},
        checkpoints=checkpoints, runtime_sec=time.perf_counter() - t0,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} runtime={out['runtime_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
