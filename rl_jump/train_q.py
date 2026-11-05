"""R2 arm B: expected-SARSA Q-learning with replay over the same move-head network.

Q(s, a) is the move-head output for candidate a. Behaviour policy: softmax(Q/T), with T annealed from
1.0 to 0.1 over the first 50% of env steps. Target: y = r + gamma * sum_a' pi(a'|s') Q_target(s', a'),
pi = softmax(Q_target/T) at the current behaviour temperature, y = r at episode end. Huber loss,
hard target update every 1000 gradient steps, one gradient step per collected env step (after a
warm-up of learn_start stored transitions). Replay stores each step's observation (graph data and
candidate index tensors), so no geometry is recomputed during learning.

Example:
  python -m rl_jump.train_q --out_dir R/runs/rl_jump_r2_20261105/q --device cuda:5
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from rl_jump import train_common as tc
from rl_jump.pool import EpisodePool


class Replay:
    """Ring buffer of single-step records; each episode's block is contiguous, so entry i+1 is the next state."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.obs = [None] * self.capacity
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float64)
        self.done = np.zeros(self.capacity, dtype=bool)
        self.pos = 0
        self.size = 0

    def add_block(self, items):
        for observation, action, reward, done in items:
            self.obs[self.pos] = observation
            self.action[self.pos] = action
            self.reward[self.pos] = reward
            self.done[self.pos] = done
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--total_env_steps", type=int, default=100_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=10_000)
    parser.add_argument("--eval_limit", type=int, default=30, help="dev polytopes evaluated (30 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--replay", type=int, default=50_000)
    parser.add_argument("--learn_start", type=int, default=1000, help="transitions stored before learning starts")
    parser.add_argument("--target_every", type=int, default=1000, help="gradient steps between hard target copies")
    parser.add_argument("--t_start", type=float, default=1.0)
    parser.add_argument("--t_end", type=float, default=0.1)
    parser.add_argument("--t_anneal_fraction", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--near_fraction", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max_points", type=int, default=12)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--model_config", default=str(tc.MODEL_CONFIG))
    return parser.parse_args(argv)


def temperature(args, env_steps):
    fraction = min(1.0, env_steps / max(1.0, args.t_anneal_fraction * args.total_env_steps))
    return args.t_start + (args.t_end - args.t_start) * fraction


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
    target = copy.deepcopy(net).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    # Only the move head (agent) is trained by this arm; the value head receives no gradient.
    optimizer = torch.optim.Adam(net.agent.parameters(), lr=args.lr)
    sample_gen = torch.Generator().manual_seed(args.seed)
    replay = Replay(args.replay)
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

        def soft_values(q_logits, counts, temp):
            """Per-graph sum_a pi(a) Q(a) with pi = softmax(Q/temp)."""
            pieces = []
            for chunk in torch.split(q_logits, counts):
                pi = torch.softmax(chunk / temp, dim=0)
                pieces.append((pi * chunk).sum())
            return torch.stack(pieces)

        env_steps, next_eval, iteration, grad_steps = 0, args.eval_every, 0, 0
        while env_steps < args.total_env_steps:
            iteration += 1
            iter_started = time.perf_counter()
            temp = temperature(args, env_steps)
            lines = rng.integers(len(train_rows), size=args.num_envs)
            assignments = {slot: (str(tc.TRAIN_ROWS), int(lines[slot]), int(rng.integers(train_starts[lines[slot]])))
                           for slot in range(args.num_envs)}
            blocks = {slot: [] for slot in range(args.num_envs)}

            def act(observations):
                net.eval()
                with torch.no_grad():
                    batch = tc.to_batch(observations, device)
                    q, _ = net(batch)
                    counts = tc.graph_counts(batch)
                    q = q.float().cpu()
                actions = []
                for chunk in torch.split(q, counts):
                    pi = torch.softmax(chunk / temp, dim=0)
                    actions.append(int(torch.multinomial(pi, 1, generator=sample_gen)))
                return actions, [None] * len(observations)

            def on_step(slot, obs_before, action, x, reward, done, info, new_obs):
                blocks[slot].append((obs_before, action, reward, done))

            state = tc.run_episodes(pool, assignments, act, on_step=on_step)
            transitions = 0
            episode_returns, best_dlnv = [], []
            for slot in range(args.num_envs):
                st = state[slot]
                episode_returns.append(sum(st["rewards"]))
                best_dlnv.append(math.log(st["best_value"]) - math.log(st["start_value"]))
                replay.add_block(blocks[slot])
                transitions += len(blocks[slot])
            env_steps += transitions
            losses, td_abs, mean_q, count_updates = [], [], [], 0
            if replay.size >= args.learn_start:
                net.train()
                for _ in range(transitions):
                    idx = rng.integers(replay.size, size=args.batch)
                    obs_s = [replay.obs[i] for i in idx]
                    actions = replay.action[idx]
                    rewards = replay.reward[idx]
                    dones = replay.done[idx]
                    live = np.nonzero(~dones)[0]
                    next_idx = (idx[live] + 1) % replay.capacity
                    batch_s = tc.to_batch(obs_s, device)
                    q_all, _ = net(batch_s)
                    counts_s = tc.graph_counts(batch_s)
                    q_sa = q_all[tc.flat_indices(counts_s, actions).to(device)]
                    targets = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        if len(live):
                            batch_n = tc.to_batch([replay.obs[i] for i in next_idx], device)
                            q_next, _ = target(batch_n)
                            v_next = soft_values(q_next, tc.graph_counts(batch_n), temp)
                            live_t = torch.as_tensor(live, dtype=torch.long, device=device)
                            targets = targets.index_add(0, live_t, args.gamma * v_next)
                    loss = F.smooth_l1_loss(q_sa, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    grad_steps += 1
                    count_updates += 1
                    losses.append(float(loss))
                    td_abs.append(float((q_sa.detach() - targets).abs().mean()))
                    mean_q.append(float(q_sa.detach().mean()))
                    if grad_steps % args.target_every == 0:
                        target.load_state_dict(net.state_dict())
                        target.eval()
            stats = pool.stats()
            iter_wall = time.perf_counter() - iter_started
            finite = all(math.isfinite(v) for v in losses + td_abs + mean_q)
            train_writer.write(dict(
                iteration=iteration, env_steps=env_steps, transitions_this_iter=transitions,
                objective_queries=stats.get("queries", 0), mean_episode_return=float(np.mean(episode_returns)),
                mean_best_dlnV=float(np.mean(best_dlnv)), max_best_dlnV=float(np.max(best_dlnv)),
                huber_loss=float(np.mean(losses)) if losses else None,
                td_abs_error=float(np.mean(td_abs)) if td_abs else None,
                mean_q=float(np.mean(mean_q)) if mean_q else None, temperature=temp,
                replay_size=replay.size, gradient_steps=grad_steps, steps_per_s=transitions / iter_wall,
                iter_seconds=iter_wall, finite=bool(finite)))
            print(f"iter {iteration} env_steps {env_steps} queries {stats.get('queries', 0)} "
                  f"return {np.mean(episode_returns):.3f} T {temp:.2f} grad {grad_steps} "
                  f"loss {np.mean(losses) if losses else float('nan'):.4f} sps {transitions / iter_wall:.2f}",
                  flush=True)
            if env_steps >= next_eval or env_steps >= args.total_env_steps:
                started = time.perf_counter()
                records = tc.evaluate(pool, dev_lines, greedy, slots=args.num_envs)
                eval_writer.write(dict(policy="q_greedy", env_steps=env_steps, summary=tc.summarize_eval(records),
                                       per_polytope=records, eval_seconds=time.perf_counter() - started))
                torch.save(dict(model=net.state_dict(), env_steps=env_steps, grad_steps=grad_steps),
                           out / f"ckpt_{env_steps}.pt")
                print("eval:", tc.summarize_eval(records), flush=True)
                while next_eval <= env_steps:
                    next_eval += args.eval_every
        (out / "done.json").write_text(json.dumps(dict(env_steps=env_steps, iterations=iteration,
                                                       grad_steps=grad_steps,
                                                       objective_queries=pool.stats().get("queries", 0)),
                                                  indent=2), encoding="utf-8")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
