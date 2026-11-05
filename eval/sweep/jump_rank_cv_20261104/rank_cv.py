"""Stage 2: leave-two-polytopes-out CV for ranking single-face jump candidates.

Methods (all on the same groups): random, heuristics (smaller / larger face_symdiff first),
zero-shot FaceMoveScorer (mean2 and sum), and fine-tuned scorers: "rank" (listwise loss only) and
"rank+feas" (listwise loss plus a feasibility head trained with BCE on all children; the ranking
score is still the actor logit).

Usage:
  python rank_cv.py --verify              # my batched logits vs FaceMoveScorer.score on sampled parents
  python rank_cv.py --zero-shot --heuristics   # baselines only
  python rank_cv.py --train --folds 0 1 2 3 4 --variants rank rank+feas
  python rank_cv.py --report              # metrics table from saved scores
"""
import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

W = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau/.claude/worktrees/bridge-cse_01W7whPSySKiYGCEwuxiRkNf")
R = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau")
OUT = R / "runs/jump_rank_cv_20261104"
CHECKPOINT = R / "runs/inv_rl_full_20261104/checkpoints/final.pth"
POLYS = list(range(10))
FOLDS = [(2 * i, 2 * i + 1) for i in range(5)]
TAU = 0.5
BATCH_GROUPS = 16
LR = 1e-4
MAX_EPOCHS = 30
PATIENCE = 5
VAL_FRACTION = 0.10
TIE_DRAWS = 200
BOOT = 2000
K_CUP = 16
os.environ.setdefault("MOSEKLM_LICENSE_FILE", str(Path.home() / "mosek/mosek.lic"))
sys.path.insert(0, str(W))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(4)


def log(message):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def load_records():
    import torch as _torch
    records = []
    for poly in POLYS:
        records += _torch.load(OUT / "dataset" / f"poly{poly}.pt", weights_only=False)
    return records


def child_arrays(record):
    children = record["children"]
    feas = np.array([c["feasible"] for c in children], dtype=bool)
    dlnv = np.array([c["dlnV"] if (c["feasible"] and c["dlnV"] is not None) else np.nan for c in children], float)
    symdiff = np.array([c["face_symdiff"] for c in children], float)
    return feas, dlnv, symdiff


def pick_device():
    best, best_free = None, -1
    for index in (5, 6):
        free, _ = torch.cuda.mem_get_info(index)
        log(f"cuda:{index} free {free / 2**30:.2f} GiB")
        if free > best_free:
            best, best_free = index, free
    if best_free < 2 * 2**30:
        raise RuntimeError("Neither GPU 5 nor 6 has 2 GiB free.")
    return torch.device(f"cuda:{best}")


def make_model(device, feas_head):
    from eval.algorithm.face_move_scorer import load_face_move_model
    model = load_face_move_model(CHECKPOINT).to(device)
    head = None
    if feas_head:
        in_dim = model.actor_mlp[0].in_features
        hidden = model.actor_mlp[0].out_features
        head = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(device)
    return model, head


def forward(model, head, record, aggregation, device):
    """Actor logits (and feasibility logits if head) for every child of one parent; mirrors FaceMoveScorer.score."""
    batch = record["batch"].to(device)
    index = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in record["index"].items()}
    h, z, faces, global_features, _ = model._encode(batch)
    count = index["count"]
    removed_sum = z.new_zeros((count, z.size(1))).index_add(0, index["removed_move"], z[index["removed_ids"]])
    proposed = model.triangle_mlp(h[index["added_nodes"]].sum(dim=1))
    added_sum = z.new_zeros((count, z.size(1))).index_add(0, index["added_move"], proposed)
    if aggregation == "mean2":
        removed_count = torch.bincount(index["removed_move"], minlength=count).to(z.dtype).unsqueeze(-1)
        added_count = torch.bincount(index["added_move"], minlength=count).to(z.dtype).unsqueeze(-1)
        removed_sum = 2 * removed_sum / removed_count.clamp_min(1)
        added_sum = 2 * added_sum / added_count.clamp_min(1)
    x = torch.cat((removed_sum, added_sum, faces[index["move_face"]],
                   global_features[0].expand(count, -1)), dim=-1)
    logit = model.actor_mlp(x).squeeze(-1)
    feas_logit = head(x).squeeze(-1) if head is not None else None
    return logit, feas_logit


def group_loss(model, head, record, aggregation, device):
    feas, dlnv, _ = child_arrays(record)
    logit, feas_logit = forward(model, head, record, aggregation, device)
    mask = torch.tensor(feas, device=device)
    target = torch.softmax(torch.tensor(dlnv[feas] / TAU, dtype=torch.float32, device=device), dim=0)
    loss = -(target * torch.log_softmax(logit[mask], dim=0)).sum()
    if feas_logit is not None:
        labels = torch.tensor(feas, dtype=torch.float32, device=device)
        loss = loss + F.binary_cross_entropy_with_logits(feas_logit, labels)
    return loss


def evaluate_loss(model, head, records, aggregation, device):
    model.eval()
    if head is not None:
        head.eval()
    total = 0.0
    with torch.no_grad():
        for record in records:
            total += float(group_loss(model, head, record, aggregation, device))
    return total / max(len(records), 1)


def train_variant(records_train, records_val, device, variant, seed, log_prefix):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    use_head = variant == "rank+feas"
    model, head = make_model(device, use_head)
    params = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    optimizer = torch.optim.Adam(params, lr=LR)
    best_loss, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        if head is not None:
            head.train()
        order = rng.permutation(len(records_train))
        for start in range(0, len(order), BATCH_GROUPS):
            batch_ids = order[start:start + BATCH_GROUPS]
            optimizer.zero_grad()
            for i in batch_ids:
                loss = group_loss(model, head, records_train[i], "mean2", device)
                (loss / len(batch_ids)).backward()
            optimizer.step()
        val_loss = evaluate_loss(model, head, records_val, "mean2", device)
        history.append(dict(epoch=epoch, val_loss=val_loss, sec=time.perf_counter() - started))
        log(f"{log_prefix} epoch {epoch} val_loss {val_loss:.4f} ({history[-1]['sec']:.1f}s)")
        if val_loss < best_loss - 1e-5:
            best_loss, bad = val_loss, 0
            best_state = (deepcopy(model.state_dict()), deepcopy(head.state_dict()) if head is not None else None,
                          epoch)
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state[0])
    if head is not None:
        head.load_state_dict(best_state[1])
    return model, head, dict(best_epoch=best_state[2], best_val_loss=best_loss, history=history)


def expected_max_random(sorted_values, k):
    """E[max] of k values drawn without replacement from sorted_values (ascending)."""
    n = len(sorted_values)
    total = math.comb(n, k)
    return sum(sorted_values[j] * math.comb(j, k - 1) for j in range(n)) / total


def auc(scores, labels):
    from scipy.stats import rankdata
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return np.nan
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def group_metrics(score_list, feas, dlnv, rng, feas_score=None):
    """Metrics for one group. score_list: score vectors (one per random draw; normally one)."""
    from scipy.stats import spearmanr
    f_idx = np.where(feas)[0]
    n_all, n_f = len(feas), len(f_idx)
    values = dlnv[f_idx]
    k = min(n_f, max(1, int(math.floor(K_CUP * n_f / n_all + 0.5))))
    erand = expected_max_random(np.sort(values), k)
    best = values.max()
    out = dict(F=n_f, N=n_all, k=k, erand=erand, best=best, spearman=[], recall=[], topk=[], gain=[], auc=[])
    for scores in score_list:
        sf = scores[f_idx]
        rho = spearmanr(sf, values).correlation if np.ptp(sf) > 0 else np.nan
        has_ties = len(np.unique(sf)) < n_f
        reps = TIE_DRAWS if has_ties else 1
        recalls, topks = [], []
        for _ in range(reps):
            order = np.lexsort((rng.random(n_f), -sf))
            recalls.append(float(values[order[0]] == best))
            topks.append(values[order[:k]].max())
        out["spearman"].append(rho)
        out["recall"].append(float(np.mean(recalls)))
        out["topk"].append(float(np.mean(topks)))
        out["gain"].append(float(np.mean(topks)) - erand)
        out["auc"].append(auc(scores, feas))
    if feas_score is not None:
        out["auc_feas_head"] = auc(feas_score, feas)
    return out


def summarize_method(per_group, polys, rng):
    """Mean over groups with cluster bootstrap over polytopes (95% CI)."""
    metrics = ["spearman", "recall", "topk", "gain", "auc"]
    extra = ["auc_feas_head"] if any("auc_feas_head" in g for g in per_group) else []
    poly_ids = np.array(polys)
    unique_polys = sorted(set(polys))
    table = {}
    for name in metrics + extra:
        values = np.array([g[name] for g in per_group], float)
        point = float(np.nanmean(values))
        by_poly = {p: np.where(poly_ids == p)[0] for p in unique_polys}
        boots = []
        for _ in range(BOOT):
            chosen = rng.choice(unique_polys, size=len(unique_polys), replace=True)
            idx = np.concatenate([by_poly[p] for p in chosen])
            boots.append(np.nanmean(values[idx]))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        table[name] = dict(mean=point, lo=float(lo), hi=float(hi), n=int(np.sum(~np.isnan(values))))
    table["groups"] = len(per_group)
    table["mean_k"] = float(np.mean([g["k"] for g in per_group]))
    table["mean_F"] = float(np.mean([g["F"] for g in per_group]))
    table["mean_N"] = float(np.mean([g["N"] for g in per_group]))
    table["mean_erand"] = float(np.mean([g["erand"] for g in per_group]))
    return table


def scores_to_group_metrics(scores_per_record, records, rng, feas_scores=None):
    out = []
    for i, (record, scores) in enumerate(zip(records, scores_per_record)):
        feas, dlnv, _ = child_arrays(record)
        fs = feas_scores[i] if feas_scores is not None else None
        out.append(group_metrics([scores], feas, dlnv, rng, feas_score=fs))
    return out


def flatten(metrics_list):
    """Collapse per-group lists (tie-break draws) to one scalar per group and metric."""
    flat = []
    for g in metrics_list:
        item = dict(g)
        for name in ("spearman", "recall", "topk", "gain", "auc"):
            item[name] = float(np.mean(g[name]))
        flat.append(item)
    return flat


def run_zero_shot_and_heuristics(records, device):
    results = {}
    rng = np.random.default_rng(0)
    polys = [r["poly"] for r in records]
    for aggregation in ("mean2", "sum"):
        model, _ = make_model(device, False)
        model.eval()
        scores = []
        with torch.no_grad():
            for record in records:
                logit, _ = forward(model, None, record, aggregation, device)
                scores.append(logit.double().cpu().numpy())
        results[f"zeroshot_{aggregation}"] = scores
        del model
    results["heur_smaller_symdiff"] = [-child_arrays(r)[2] for r in records]
    results["heur_larger_symdiff"] = [child_arrays(r)[2] for r in records]
    results["random"] = [rng.random(len(r["children"])) for r in records]
    return results, polys


def metrics_for_method(scores, records, rng, feas_scores=None):
    per = []
    for i, record in enumerate(records):
        feas, dlnv, _ = child_arrays(record)
        fs = feas_scores[i] if feas_scores is not None else None
        per.append(group_metrics([scores[i]], feas, dlnv, rng, feas_score=fs))
    return flatten(per)


def verify_scorer(records, device):
    """Batched logits must equal FaceMoveScorer.score on the same parent (mean2 and sum)."""
    from eval.algorithm.face_move_scorer import FaceMoveScorer
    from mdp.cy_rollout import build_cy_rollout_collection, create_transition_pool
    from mdp.cy_state_record import CyStateRecord
    from build_dataset import parent_cache_path, SETUPS, RESULTS
    import hashlib
    checked = 0
    worst = 0.0
    scorers = {agg: FaceMoveScorer(CHECKPOINT, aggregation=agg) for agg in ("mean2", "sum")}
    model, _ = make_model(device, False)
    model.eval()
    pool = create_transition_pool(num_workers=1)
    try:
        for poly in POLYS:
            mine = [r for r in records if r["poly"] == poly][:2]
            if not mine:
                continue
            from eval.setup import load_eval_setup
            setup = load_eval_setup(SETUPS / f"seed0_poly{poly}_s3")
            collection = build_cy_rollout_collection(
                setup.rows, include_points_interior_to_facets=False, neighbor_mode="two_neighbors",
                transition_pool=pool, two_face_state=False, include_two_face_metadata=True)
            configuration = next(iter(collection.base_states.values())).configuration
            reference = json.loads((RESULTS / f"face_mutation_befs_budget1000_seed0_poly{poly}" /
                                    "cyopt_encoding" / f"polytope_{poly}.json").read_text())
            codebook = reference["face_triangulations"]
            for record in mine:
                simplices = [tuple(s) for s in json.loads(parent_cache_path(
                    poly, record["start_hash"], record["parent_dna"]).read_text())["simplices"]]
                state = CyStateRecord(configuration, frozenset(simplices), "two_neighbors", True, True)
                moves = []
                for child in record["children"]:
                    f = child["face"]
                    labels = tuple(sorted({int(v) for entry in codebook[f] for t in entry for v in t}))
                    moves.append((labels, codebook[f][child["child_dna"][child["dna_position"]]]))
                for aggregation in ("mean2", "sum"):
                    expected = scorers[aggregation].score(state, moves)
                    with torch.no_grad():
                        got = forward(model, None, record, aggregation, device)[0].double().cpu().numpy()
                    worst = max(worst, float(np.max(np.abs(got - expected))))
                    checked += 1
    finally:
        pool.close()
    log(f"verify: {checked} comparisons, max |diff| = {worst:.3e}")
    if worst > 1e-4:
        raise RuntimeError("Batched logits differ from FaceMoveScorer.")
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--variants", nargs="+", default=["rank", "rank+feas"])
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--rescore", action="store_true",
                        help="rebuild held-out scores from saved fold checkpoints")
    args = parser.parse_args()
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    records = load_records()
    log(f"loaded {len(records)} groups")
    device = pick_device()
    if args.verify:
        verify_scorer(records, device)
        return
    scores_path = OUT / "results" / "scores.npz"
    if args.zero_shot:
        started = time.perf_counter()
        scores, polys = run_zero_shot_and_heuristics(records, device)
        np.savez(scores_path, **{k: np.array(v, dtype=object) for k, v in scores.items()})
        log(f"zero-shot and heuristic scores done in {time.perf_counter() - started:.1f}s")
    if args.train:
        cache = {}
        for fold, held in enumerate(FOLDS):
            if fold not in args.folds:
                continue
            test_idx = [i for i, r in enumerate(records) if r["poly"] in held]
            train_all = [i for i, r in enumerate(records) if r["poly"] not in held]
            rng = np.random.default_rng(1000 + fold)
            val = set(rng.choice(train_all, size=int(math.ceil(VAL_FRACTION * len(train_all))), replace=False))
            train_idx = [i for i in train_all if i not in val]
            val_idx = sorted(val)
            for variant in args.variants:
                started = time.perf_counter()
                prefix = f"fold{fold} {variant}"
                model, head, info = train_variant([records[i] for i in train_idx],
                                                  [records[i] for i in val_idx], device, variant,
                                                  seed=fold, log_prefix=prefix)
                torch.save(dict(model=model.state_dict(), head=head.state_dict() if head is not None else None,
                                info=info, held_out=list(held), variant=variant),
                           OUT / "checkpoints" / f"{variant.replace('+', '_')}_fold{fold}.pt")
                model.eval()
                with torch.no_grad():
                    for i in test_idx:
                        logit, feas_logit = forward(model, head, records[i], "mean2", device)
                        cache.setdefault(variant, {})[i] = (logit.double().cpu().numpy(),
                                                            None if feas_logit is None
                                                            else feas_logit.double().cpu().numpy())
                log(f"{prefix} done: best epoch {info['best_epoch']}, wall {time.perf_counter() - started:.1f}s")
                with (OUT / "results" / "train_log.jsonl").open("a") as handle:
                    handle.write(json.dumps(dict(fold=fold, variant=variant, held_out=list(held),
                                                 train_groups=len(train_idx), val_groups=len(val_idx),
                                                 best_epoch=info["best_epoch"],
                                                 best_val_loss=info["best_val_loss"],
                                                 wall_sec=time.perf_counter() - started,
                                                 history=info["history"])) + "\n")
                del model, head
        torch.save(cache, OUT / "results" / f"heldout_scores_folds{'_'.join(map(str, args.folds))}.pt")
    if args.rescore:
        for fold, held in enumerate(FOLDS):
            if fold not in args.folds:
                continue
            test_idx = [i for i, r in enumerate(records) if r["poly"] in held]
            for variant in args.variants:
                path = OUT / "checkpoints" / f"{variant.replace('+', '_')}_fold{fold}.pt"
                if not path.exists():
                    log(f"rescore: missing {path.name}, skipped")
                    continue
                checkpoint = torch.load(path, weights_only=False)
                model, head = make_model(device, checkpoint["head"] is not None)
                model.load_state_dict(checkpoint["model"])
                if head is not None:
                    head.load_state_dict(checkpoint["head"])
                model.eval()
                cache = {}
                with torch.no_grad():
                    for i in test_idx:
                        logit, feas_logit = forward(model, head, records[i], "mean2", device)
                        cache[i] = (logit.double().cpu().numpy(),
                                    None if feas_logit is None else feas_logit.double().cpu().numpy())
                torch.save({variant: cache}, OUT / "results" / f"heldout_scores_rescore_fold{fold}_{variant.replace('+', '_')}.pt")
                log(f"rescore: fold {fold} {variant} {len(cache)} groups")
    if args.report:
        report(records)


def report(records):
    """Build the results table. Learned scores come from the held-out fold cache files."""
    rng = np.random.default_rng(7)
    polys = [r["poly"] for r in records]
    raw = dict(np.load(OUT / "results" / "scores.npz", allow_pickle=True))
    learned = {}
    for path in sorted((OUT / "results").glob("heldout_scores*.pt")):
        cache = torch.load(path, weights_only=False)
        for variant, per in cache.items():
            learned.setdefault(variant, {}).update(per)
    methods = {}
    order = ["random", "heur_smaller_symdiff", "heur_larger_symdiff", "zeroshot_mean2", "zeroshot_sum"]
    for name in order:
        methods[name] = (raw[name], None)
    for variant, per in learned.items():
        if len(per) != len(records):
            log(f"warning: {variant} has {len(per)} held-out groups, expected {len(records)}")
        scores = [per[i][0] for i in range(len(records))]
        feas_scores = [per[i][1] for i in range(len(records))] if per[0][1] is not None else None
        methods[variant] = (scores, feas_scores)
    table = {}
    per_group_store = {}
    for name, (scores, feas_scores) in methods.items():
        per = metrics_for_method(scores, records, rng, feas_scores=feas_scores)
        table[name] = summarize_method(per, polys, rng)
        per_group_store[name] = per
    with (OUT / "results" / "summary.json").open("w") as handle:
        json.dump(table, handle, indent=1)
    cols = ["spearman", "recall", "topk", "gain", "auc", "auc_feas_head"]
    lines = ["| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, stats in table.items():
        cells = []
        for c in cols:
            if c in stats:
                s = stats[c]
                cells.append(f"{s['mean']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]")
            else:
                cells.append("-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    (OUT / "results" / "table.md").write_text(text + "\n")
    log("\n" + text)
    log(f"groups {len(records)}, mean F {table['random']['mean_F']:.2f}, mean N {table['random']['mean_N']:.2f}, "
        f"mean k {table['random']['mean_k']:.2f}, mean random E[max] {table['random']['mean_erand']:.4f}")


if __name__ == "__main__":
    main()
