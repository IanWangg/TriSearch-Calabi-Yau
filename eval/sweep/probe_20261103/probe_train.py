"""Supervised upper-bound probe (SPEC Q3): can the two_face network rank children given full 1-hop labels?

Trains a fresh TwoFaceAgent (same architecture and config as checkpoint G) on the S_sup
states, using the stage A records and shards written by tools/probe_collect.py.
  actor: listwise cross entropy, target softmax(dlnV / tau), tau = 0.1
  value: MSE on y1 = max_child dlnV
variant "ctx": standardized lnV(s) and h11/20 are concatenated to the global features and
  projected back to the global width (Linear(66 -> 64)) before the actor and value MLPs.
Early stopping on S_sup_val (in-distribution validation polytopes).

Writes runs/probe_20261103/probe/<variant>_s<seed>_pred.jsonl with V(s), log pi(.|s) and
dlnV for S_sup train/val states and S_dev states, for analyze.py.

Example:
  python eval/sweep/probe_20261103/probe_train.py --variant plain --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[3]
MAIN_ROOT = Path("/home/yiranwang/combinartorics/TriSearch-Calabi-Yau")
sys.path.insert(0, str(REPO))

from torch_geometric.data import Batch  # noqa: E402

from core.cy_two_face_data import TwoFaceData  # noqa: E402,F401  (needed to unpickle shards)
from models.two_face_agent import TwoFaceAgent  # noqa: E402


class ContextTwoFaceAgent(TwoFaceAgent):
    """Adds [standardized lnV(s), h11/20] to the global features before actor and value heads."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        width = self.model_config["out_channels"]
        self.ctx_proj = nn.Linear(width + 2, width).to(kwargs["device"])

    def _encode(self, batch):
        h, z, faces, global_features, face_graph = super()._encode(batch)
        context = batch.probe_ctx.to(global_features)
        return h, z, faces, self.ctx_proj(torch.cat((global_features, context), dim=-1)), face_graph


POOLING_MODES = ("sum", "mean", "max", "summax")


def _sum(features, indices, size):
    return features.new_zeros((size, features.size(-1))).index_add(0, indices, features)


def _reduce(values, index, size, mode):
    """Segment reduction over `index` (sum, mean or max) for a [N, C] tensor."""
    width = values.size(-1)
    if mode == "sum":
        return values.new_zeros((size, width)).index_add(0, index, values)
    if mode == "mean":
        total = values.new_zeros((size, width)).index_add(0, index, values)
        count = torch.bincount(index, minlength=size).to(values.dtype).unsqueeze(-1)
        return total / count.clamp_min(1)
    if mode == "max":
        out = values.new_zeros((size, width))
        return out.scatter_reduce(0, index.unsqueeze(-1).expand_as(values), values, reduce="amax", include_self=False)
    raise ValueError(mode)


class PoolingTwoFaceAgent(TwoFaceAgent):
    """TwoFaceAgent with the same pooling choice for triangles -> face and faces -> graph.

    sum reproduces TwoFaceAgent._encode exactly. summax concatenates sum and max and
    projects back to the channel width with a Linear layer (one per pooling site).
    """

    def __init__(self, *, pooling="sum", **kwargs):
        if pooling not in POOLING_MODES:
            raise ValueError(pooling)
        super().__init__(**kwargs)
        self.pooling = pooling
        width = self.model_config["out_channels"]
        device = kwargs["device"]
        if pooling == "summax":
            self.face_proj = nn.Linear(2 * width, width).to(device)
            self.global_proj = nn.Linear(2 * width, width).to(device)

    def _pool_sites(self, values, index, size, projection):
        if self.pooling != "summax":
            return _reduce(values, index, size, self.pooling)
        both = torch.cat((_reduce(values, index, size, "sum"), _reduce(values, index, size, "max")), dim=-1)
        return projection(both)

    def _encode(self, batch):
        h, _ = self.point_egnn(h=batch.x, x=batch.x, edges=batch.edge_index)
        z = self.triangle_mlp(h[batch.triangle_vertices].sum(dim=1))
        for layer in self.triangle_dual:
            z = layer(z, batch.triangle_edge_index)
        face_counts = batch.num_faces.reshape(-1).long()
        face_graph = torch.repeat_interleave(torch.arange(batch.num_graphs, device=h.device), face_counts)
        num_faces = face_graph.numel()
        node_counts = torch.bincount(batch.node_face, minlength=num_faces).to(h.dtype).unsqueeze(-1)
        centroids = _sum(batch.x, batch.node_face, num_faces) / node_counts.clamp_min(1)
        face_pool = self._pool_sites(self.phi_triangle(z), batch.triangle_face, num_faces,
                                     getattr(self, "face_proj", None))
        faces = self.face_mlp(torch.cat((face_pool, centroids, node_counts), dim=-1))
        graph_pool = self._pool_sites(self.phi_face(faces), face_graph, batch.num_graphs,
                                      getattr(self, "global_proj", None))
        global_features = self.rho(torch.cat((graph_pool, face_counts.to(h.dtype).unsqueeze(-1)), dim=-1))
        return h, z, faces, global_features, face_graph


def load_records(run_dir, set_names):
    shard_cache, items = {}, []
    for set_name in set_names:
        path = run_dir / "stageA" / f"records_{set_name}.jsonl"
        if not path.exists():
            continue
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["shard"] not in shard_cache:
                    shard_cache[record["shard"]] = torch.load(
                        run_dir / "stageA" / "shards" / record["shard"], weights_only=False)
                if record["n_children"] == 0:
                    continue
                lnv = record["lnV"]
                dlnv = np.array([value - lnv for _, value in record["children"]], dtype=np.float32)
                data = shard_cache[record["shard"]][record["shard_index"]]
                items.append(dict(record=record, data=data, dlnv=dlnv, y1=float(dlnv.max())))
    return items


def pick_device(force_cpu):
    """Choose a CUDA device by free memory, never GPU 0 or 3."""
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    candidates = [i for i in range(torch.cuda.device_count()) if i not in (0, 3)]
    if not candidates:
        return torch.device("cpu")
    best = max(candidates, key=lambda i: torch.cuda.mem_get_info(i)[0])
    return torch.device(f"cuda:{best}")


def make_batch(items, device, ctx_mean, ctx_std, variant):
    datas = []
    for item in items:
        data = item["data"]
        if variant == "ctx":
            lnv = (item["record"]["lnV"] - ctx_mean) / ctx_std
            data.probe_ctx = torch.tensor([[lnv, item["record"]["h11"] / 20.0]], dtype=torch.float32)
        datas.append(data)
    batch = Batch.from_data_list(datas).to(device)
    counts = [len(item["dlnv"]) for item in items]
    width = max(counts)
    dlnv = torch.zeros(len(items), width)
    mask = torch.zeros(len(items), width, dtype=torch.bool)
    for row, item in enumerate(items):
        dlnv[row, :counts[row]] = torch.from_numpy(item["dlnv"])
        mask[row, :counts[row]] = True
    y1 = torch.tensor([item["y1"] for item in items], dtype=torch.float32)
    return batch, dlnv.to(device), mask.to(device), y1.to(device)


def losses(model, batch, dlnv, mask, y1, tau, value_coef):
    value, logits = model.get_value_and_logits(batch)
    logp = torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    target = torch.softmax((dlnv / tau).masked_fill(~mask, float("-inf")), dim=-1)
    cross_entropy = -(target * torch.where(mask, logp, torch.zeros_like(logp))).sum(dim=-1).mean()
    value_loss = ((value - y1) ** 2).mean()
    return cross_entropy + value_coef * value_loss, cross_entropy.item(), value_loss.item()


@torch.no_grad()
def predict(model, items, device, ctx_mean, ctx_std, variant, batch_size=128):
    model.eval()
    out = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch, _, _, _ = make_batch(chunk, device, ctx_mean, ctx_std, variant)
        value, logits = model.get_value_and_logits(batch)
        for row, item in enumerate(chunk):
            count = len(item["dlnv"])
            row_logits = logits[row, :count].double()
            out.append(dict(V=float(value[row]), logp=torch.log_softmax(row_logits, dim=-1).cpu().numpy()))
    return out


def spearman(a, b):
    from scipy.stats import spearmanr

    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def quick_metrics(items, preds):
    """Log-only summary; the authoritative metrics come from analyze.py."""
    if not items:
        return {}
    recall1, recall4, rho, v_pred, y1 = [], [], [], [], []
    for item, pred in zip(items, preds):
        n = len(item["dlnv"])
        v_pred.append(pred["V"])
        y1.append(item["y1"])
        if n < 2:
            continue
        best = int(np.argmax(item["dlnv"]))
        order = np.argsort(-pred["logp"], kind="stable")
        rank = int(np.where(order == best)[0][0])
        recall1.append(rank < 1)
        recall4.append(rank < 4)
        rho.append(spearman(pred["logp"], item["dlnv"]))
    y1, v_pred = np.array(y1), np.array(v_pred)
    ev = 1 - np.var(y1 - v_pred) / max(np.var(y1), 1e-12)
    return dict(n_states=len(items), recall1=float(np.mean(recall1)), recall4=float(np.mean(recall4)),
                spearman_actor=float(np.nanmean(rho)), spearman_value=spearman(v_pred, y1), ev_value=float(ev))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=("plain", "ctx"), default="plain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_dir", default=str(MAIN_ROOT / "runs/probe_20261103"))
    parser.add_argument("--out_dir", default=None, help="Default: <run_dir>/probe")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--max_states", type=int, default=0, help="Smoke tests only: truncate the train set.")
    parser.add_argument("--n_polytopes", type=int, default=0,
                        help="Memorization check: keep only this many seeded S_sup train polytopes (0 = all).")
    parser.add_argument("--width", type=int, default=64, help="hidden and out channels of the EGNN and heads.")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Memorization check: train all epochs and keep the final weights.")
    parser.add_argument("--tag", default="", help="Suffix for output names.")
    parser.add_argument("--pooling", choices=POOLING_MODES, default="sum",
                        help="Pooling for triangles -> face and faces -> graph (plain variant).")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng([args.seed, 61])
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.force_cpu)
    print(f"device={device} variant={args.variant} seed={args.seed}", flush=True)

    sup_items = load_records(run_dir, ["sup"])
    dev_items = load_records(run_dir, ["dev"])
    train_items = [it for it in sup_items if it["record"]["role"] == "train"]
    val_items = [it for it in sup_items if it["record"]["role"] == "val"]
    if args.max_states:
        train_items, val_items = train_items[:args.max_states], val_items[:max(1, args.max_states // 4)]
    if args.n_polytopes:
        pids = sorted({it["record"]["polytope_index"] for it in train_items})
        chosen = set(np.random.default_rng([args.seed, 62]).choice(pids, args.n_polytopes, replace=False).tolist())
        train_items = [it for it in train_items if it["record"]["polytope_index"] in chosen]
        val_items = []
    print(f"S_sup train states={len(train_items)} val states={len(val_items)} dev states={len(dev_items)}",
          flush=True)
    lnv_train = np.array([it["record"]["lnV"] for it in train_items])
    ctx_mean, ctx_std = float(lnv_train.mean()), float(lnv_train.std() + 1e-8)

    config = dict(in_channels=4, out_channels=args.width, hidden_channels=args.width, num_layers=3, act="silu",
                  two_face_encoder="deep_sets")
    # Entropy of the actor targets: the floor of the listwise cross entropy for these states.
    entropies = []
    for item in train_items:
        target = np.exp((item["dlnv"] / args.tau) - (item["dlnv"] / args.tau).max())
        target = target / target.sum()
        entropies.append(float(-(target * np.log(target)).sum()))
    entropy_floor = float(np.mean(entropies)) if entropies else float("nan")
    print(f"n_train_states={len(train_items)} n_polytopes={len({it['record']['polytope_index'] for it in train_items})} "
          f"entropy_floor_ce={entropy_floor:.4f}", flush=True)
    if args.variant == "ctx":
        model = ContextTwoFaceAgent(device=device, **config).to(device)
    elif args.pooling != "sum":
        model = PoolingTwoFaceAgent(pooling=args.pooling, device=device, **config).to(device)
    else:
        model = TwoFaceAgent(device=device, **config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state, best_epoch, stale = float("inf"), None, -1, 0
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_items))
        epoch_loss, epoch_ce, batches = 0.0, 0.0, 0
        for start in range(0, len(order), args.batch_size):
            chunk = [train_items[k] for k in order[start:start + args.batch_size]]
            batch, dlnv, mask, y1 = make_batch(chunk, device, ctx_mean, ctx_std, args.variant)
            loss, ce, _ = losses(model, batch, dlnv, mask, y1, args.tau, args.value_coef)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_ce += ce
            batches += 1
        model.eval()
        val_total, val_ce, val_mse, val_n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_items), 128):
                chunk = val_items[start:start + 128]
                batch, dlnv, mask, y1 = make_batch(chunk, device, ctx_mean, ctx_std, args.variant)
                loss, ce, mse = losses(model, batch, dlnv, mask, y1, args.tau, args.value_coef)
                val_total += loss.item() * len(chunk)
                val_ce += ce * len(chunk)
                val_mse += mse * len(chunk)
                val_n += len(chunk)
        val_loss = val_total / max(val_n, 1) if val_n else epoch_loss / max(batches, 1)
        row = dict(epoch=epoch, train_loss=epoch_loss / max(batches, 1), train_ce=epoch_ce / max(batches, 1),
                   val_loss=val_loss,
                   val_ce=val_ce / max(val_n, 1), val_mse=val_mse / max(val_n, 1),
                   elapsed_sec=time.time() - started)
        history.append(row)
        print(json.dumps(row), flush=True)
        if args.no_early_stop:
            continue
        if val_loss < best_val - 1e-6:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break
    if args.no_early_stop:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = len(history) - 1
    else:
        model.load_state_dict(best_state)
    tag = f"{args.variant}_s{args.seed}{args.tag}"
    torch.save(best_state, out_dir / f"{tag}.pth")
    (out_dir / f"{tag}_train_log.json").write_text(json.dumps(dict(
        variant=args.variant, seed=args.seed, config=config, history=history, best_epoch=best_epoch,
        best_val_loss=best_val, ctx_mean=ctx_mean, ctx_std=ctx_std, tau=args.tau, lr=args.lr,
        value_coef=args.value_coef, batch_size=args.batch_size, device=str(device),
        n_train=len(train_items), n_val=len(val_items), n_dev=len(dev_items), entropy_floor_ce=entropy_floor,
        n_polytopes=args.n_polytopes, no_early_stop=args.no_early_stop, pooling=args.pooling,
        wall_sec=time.time() - started), indent=2))

    predictions = []
    for name, items, role in (("sup", train_items, "train"), ("sup", val_items, "val"), ("dev", dev_items, "dev")):
        preds = predict(model, items, device, ctx_mean, ctx_std, args.variant)
        for item, pred in zip(items, preds):
            predictions.append(dict(state_id=item["record"]["state_id"], set=name, role=role,
                                    h11=item["record"]["h11"], polytope_index=item["record"]["polytope_index"],
                                    V=pred["V"], logp=[float(x) for x in pred["logp"]],
                                    dlnV=[float(x) for x in item["dlnv"]], y1=item["y1"]))
        print(f"{role} ({name}) {json.dumps(quick_metrics(items, preds))}", flush=True)
    with (out_dir / f"{tag}_pred.jsonl").open("w") as stream:
        for row in predictions:
            stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
