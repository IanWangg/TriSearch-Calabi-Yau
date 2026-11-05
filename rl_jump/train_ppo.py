"""R2 arm A: PPO over the flat candidate list (softmax of move-head logits over feasible candidates).

Per iteration: n envs run one episode each (random train polytope, random start, H steps); GAE(gamma,
lambda) with value bootstrap on horizon truncation and 0 at dead ends; PPO clipped objective with
the value head, entropy bonus; epochs x minibatches over the rollout.

Example:
  python -m rl_jump.train_ppo --out_dir R/runs/rl_jump_r2_20261105/ppo --device cuda:4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from rl_jump import train_common as tc
from rl_jump.pool import EpisodePool


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--total_env_steps", type=int, default=100_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=10_000)
    parser.add_argument("--eval_limit", type=int, default=30, help="dev polytopes evaluated (30 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--near_fraction", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max_points", type=int, default=12)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--model_config", default=str(tc.MODEL_CONFIG))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    train_rows = tc.read_rows(tc.TRAIN_ROWS)
    train_starts = [len(row["frst_list"]) for row in train_rows]
    dev_lines = list(range(min(args.eval_limit, len(tc.read_rows(tc.DEV_ROWS)))))
    eval_writer = tc.JsonlWriter(out / "eval.jsonl")
    train_writer = tc.JsonlWriter(out / "train.jsonl")
    env_kwargs = dict(K=args.K, near_fraction=args.near_fraction, horizon=args.horizon, samples=args.samples,
                      max_points=args.max_points, codebook_seed=args.seed, seed=args.seed)
    pool = EpisodePool(env_kwargs, args.num_envs, inline=True)
    net = tc.JumpNetwork(args.model_config, seed=args.seed).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    sample_gen = torch.Generator().manual_seed(args.seed)
    try:
        baseline_rng = np.random.default_rng(args.seed)
        started = time.perf_counter()
        records = tc.evaluate(pool, dev_lines, lambda obs: tc.choose_random(obs, baseline_rng),
                              slots=args.num_envs)
        eval_writer.write(dict(policy="random_uniform", env_steps=0, summary=tc.summarize_eval(records),
                               per_polytope=records, eval_seconds=time.perf_counter() - started))
        print("random baseline:", tc.summarize_eval(records), flush=True)

        def greedy(obs):
            return tc.choose_greedy(net, obs, device)

        env_steps, next_eval, iteration = 0, args.eval_every, 0
        while env_steps < args.total_env_steps:
            iteration += 1
            iter_started = time.perf_counter()
            lines = rng.integers(len(train_rows), size=args.num_envs)
            assignments = {slot: (str(tc.TRAIN_ROWS), int(lines[slot]), int(rng.integers(train_starts[lines[slot]])))
                           for slot in range(args.num_envs)}
            trajectories = {slot: [] for slot in range(args.num_envs)}

            def act(observations):
                net.eval()
                with torch.no_grad():
                    batch = tc.to_batch(observations, device)
                    logits, values = net(batch)
                    counts = tc.graph_counts(batch)
                    logits, values = logits.float().cpu(), values.float().cpu()
                actions, aux = [], []
                for g, chunk in enumerate(torch.split(logits, counts)):
                    logp = torch.log_softmax(chunk, dim=0)
                    action = int(torch.multinomial(logp.exp(), 1, generator=sample_gen))
                    actions.append(action)
                    aux.append(dict(logp=float(logp[action]), value=float(values[g])))
                return actions, aux

            def on_step(slot, obs_before, action, x, reward, done, info, new_obs):
                trajectories[slot].append((obs_before, action, x, reward))

            state = tc.run_episodes(pool, assignments, act, on_step=on_step)
            # Bootstrap: value of the final state on horizon truncation; 0 at a dead end.
            boot_slots = [s for s in state if state[s]["done_reason"] == "horizon"]
            boot = {}
            if boot_slots:
                with torch.no_grad():
                    batch = tc.to_batch([state[s]["final_obs"] for s in boot_slots], device)
                    _, values = net(batch)
                boot = {s: float(v) for s, v in zip(boot_slots, values.cpu().tolist())}

            observations, actions, old_logp, advantages, returns, old_values = [], [], [], [], [], []
            episode_returns, best_dlnv = [], []
            for slot in range(args.num_envs):
                traj = trajectories[slot]
                st = state[slot]
                episode_returns.append(sum(r for *_, r in traj))
                best_dlnv.append(math.log(st["best_value"]) - math.log(st["start_value"]))
                if not traj:
                    continue
                rewards = np.array([item[3] for item in traj])
                vals = np.array([item[2]["value"] for item in traj])
                adv = np.zeros(len(traj))
                last = 0.0
                for t in reversed(range(len(traj))):
                    next_value = vals[t + 1] if t + 1 < len(traj) else boot.get(slot, 0.0)
                    delta = rewards[t] + args.gamma * next_value - vals[t]
                    last = delta + args.gamma * args.lam * last
                    adv[t] = last
                ret = adv + vals
                for t, (obs_before, action, x, _) in enumerate(traj):
                    observations.append(obs_before)
                    actions.append(action)
                    old_logp.append(x["logp"])
                    advantages.append(adv[t])
                    returns.append(ret[t])
                    old_values.append(vals[t])
            n = len(observations)
            if n == 0:
                raise RuntimeError("No transitions collected in an iteration.")
            adv_arr = np.array(advantages)
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
            ret_arr, old_val_arr = np.array(returns), np.array(old_values)
            ev = tc.explained_variance(old_val_arr, ret_arr)
            old_logp_t = torch.tensor(old_logp, dtype=torch.float32, device=device)
            adv_t = torch.tensor(adv_arr, dtype=torch.float32, device=device)
            ret_t = torch.tensor(ret_arr, dtype=torch.float32, device=device)
            actions_np = np.array(actions)

            net.train()
            losses = dict(policy=[], value=[], entropy=[], approx_kl=[], clipfrac=[])
            for _ in range(args.epochs):
                perm = rng.permutation(n)
                for start in range(0, n, args.minibatch):
                    idx = perm[start:start + args.minibatch]
                    idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
                    batch = tc.to_batch([observations[i] for i in idx], device)
                    logits, values = net(batch)
                    counts = tc.graph_counts(batch)
                    logp_all = tc.segment_log_softmax(logits, counts)
                    flat = tc.flat_indices(counts, actions_np[idx]).to(device)
                    new_logp = logp_all[flat]
                    ratio = torch.exp(new_logp - old_logp_t[idx_t])
                    a = adv_t[idx_t]
                    policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                    value_loss = ((values - ret_t[idx_t]) ** 2).mean()
                    probs = logp_all.exp()
                    graph = torch.repeat_interleave(torch.arange(len(counts), device=device),
                                                    torch.tensor(counts, device=device))
                    per_graph_entropy = torch.zeros(len(counts), device=device).index_add(
                        0, graph, -(probs * logp_all))
                    entropy = per_graph_entropy.mean()
                    loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                    optimizer.step()
                    losses["policy"].append(float(policy_loss))
                    losses["value"].append(float(value_loss))
                    losses["entropy"].append(float(entropy))
                    losses["approx_kl"].append(float((old_logp_t[idx_t] - new_logp).mean()))
                    losses["clipfrac"].append(float(((ratio - 1).abs() > args.clip).float().mean()))
            env_steps += n
            stats = pool.stats()
            iter_wall = time.perf_counter() - iter_started
            train_writer.write(dict(
                iteration=iteration, env_steps=env_steps, transitions_this_iter=n,
                objective_queries=stats.get("queries", 0), mean_episode_return=float(np.mean(episode_returns)),
                mean_best_dlnV=float(np.mean(best_dlnv)), max_best_dlnV=float(np.max(best_dlnv)),
                policy_loss=float(np.mean(losses["policy"])), value_loss=float(np.mean(losses["value"])),
                entropy=float(np.mean(losses["entropy"])), approx_kl=float(np.mean(losses["approx_kl"])),
                clipfrac=float(np.mean(losses["clipfrac"])), explained_variance=ev,
                steps_per_s=n / iter_wall, iter_seconds=iter_wall, gradient_steps=len(losses["policy"]),
                finite=bool(all(math.isfinite(v) for k in losses for v in losses[k]))))
            print(f"iter {iteration} env_steps {env_steps} queries {stats.get('queries', 0)} "
                  f"return {np.mean(episode_returns):.3f} ev {ev:.3f} sps {n / iter_wall:.2f}", flush=True)
            if env_steps >= next_eval or env_steps >= args.total_env_steps:
                started = time.perf_counter()
                records = tc.evaluate(pool, dev_lines, greedy, slots=args.num_envs)
                eval_writer.write(dict(policy="ppo_greedy", env_steps=env_steps, summary=tc.summarize_eval(records),
                                       per_polytope=records, eval_seconds=time.perf_counter() - started))
                torch.save(dict(model=net.state_dict(), env_steps=env_steps), out / f"ckpt_{env_steps}.pt")
                print("eval:", tc.summarize_eval(records), flush=True)
                while next_eval <= env_steps:
                    next_eval += args.eval_every
        (out / "done.json").write_text(json.dumps(dict(env_steps=env_steps, iterations=iteration,
                                                       objective_queries=pool.stats().get("queries", 0)),
                                                  indent=2), encoding="utf-8")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
