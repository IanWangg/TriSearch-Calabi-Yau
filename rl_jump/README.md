# rl_jump: single-face jump environment

A standalone RL environment whose actions are single-face jumps: one action replaces the FRT of one
2-face of the current FRST (a flip is the symmetric difference 4 case). It reuses the GA evaluation's
codebook (`eval/algorithm/cyopt_ga.py`, `CyoptEncoding`), the geometry worker's decode/objective
paths, the reward registry, and the invariant two-face observation and move mapping. Nothing in
`core/`, `models/`, or the existing `eval/` behaviour changes.

## Files

| file | role |
| --- | --- |
| `env.py` | `JumpEnv`: reset/step, candidate proposals, decode, objective cache, failure isolation |
| `obs.py` | `JumpObsData` (batchable invariant two-face data + per-candidate move fields), `build_observation` |
| `policy.py` | `build_random_policy` (invariant actor head, mean2, random init), `batched_move_logits`, `sample_actions` |
| `vec.py` | `VecJumpEnv`: envs in spawned processes, optionally one env inline in the caller |
| `smoke.py` | random-policy vectorized smoke runner (timings, infeasible fraction, cache hits, best dlnV) |
| `../eval/algorithm/face_move_scorer.py` | `move_index` factored out of `FaceMoveScorer.score` (behaviour unchanged) |
| `../test/test_rl_jump_env.py` | tests (reward, candidate mapping, flip path, caching, failures, batching) |

## Design

**State.** A DNA (one codebook index per interesting 2-face, `encoding.bounds`) and its decoded FRST
(`CyStateRecord`). Starts are the row's `frst_list` FRSTs, built with `neighbor_mode="two_neighbors"`,
`include_points_interior_to_facets=False`, exactly as the GA evaluation builds them, so the codebook
and DNA indices match the GA logs.

**Candidates (K per state).** Each proposal picks a DNA position uniformly among the interesting
faces, then a new codebook index on that face. The first `round(near_fraction*K)` proposals are
"near": the index is drawn uniformly among the `near_k=8` entries with the smallest triangle symmetric
difference to the current entry (excluding it; ties broken randomly). The rest are uniform over the
face's other entries. Proposals are distinct and differ from the current DNA. Proposals are a pure
function of `(seed, DNA)`, so a revisited state gets the same candidates and its cached objective is
reused. Each proposal is decoded (LP tip / non-solid-cone check); infeasible ones are dropped and
counted. If the neighbourhood is smaller than K, the full enumeration fills it.

**Objective and reward.** The objective is `reward_functions.get_objective(name)` (volume for
`max_kcup`), computed in-process on the decoded triangulation. Reward is
`get_reward(name).from_objectives(V_current, V_next)`, which for `max_kcup` is
`log V_next - log V_current`. Objective values are cached by DNA (failures too), so repeated states
cost no query. The start objective is also one query. `counters["queries"]` counts real computations.

**Failures.** Decode failures and objective/reward failures (tip solver, nonfinite volume, a
`SIGALRM` time limit of `geometry_timeout=300 s`) never escape `step`. A failing candidate is
removed from the current state and the step returns reward 0 and `info["failure"]`. The state stays,
and the step still counts toward the horizon.

**Observation.** `build_observation` uses `create_invariant_two_face_data_from_state` for the current
FRST and adds, per candidate, its face (`move_face`), removed triangle ids (`move_removed_ids`), added
node triples (`move_added_nodes`), and owner indices. The owner and id keys are offset per graph, so
several envs batch with `torch_geometric.data.Batch`. `move_index` is the scorer's mapping, so a
candidate has the same actor inputs as the same move under `FaceMoveScorer`.

**Policy (smoke only).** `InvariantTwoFaceAgent` actor head, mean2 aggregation, with the architecture
from `runs/inv_rl_full_20261104/checkpoints/model_config.json` and randomly initialised weights. Scores
go through a softmax per env; actions are sampled. The forward pass runs in the caller's process.

**Vectorization.** `VecJumpEnv` starts each env in a `spawn` process (`geometry` on CPU; threads set to
1 by `configure_cytools(worker=True)`). With `inline_last=True`, the last env runs in the caller's
process, so N envs use N-1 workers plus the caller, which keeps the total at N processes.

## Commands

From the worktree root, with the sage environment:

```bash
export PYTHONPATH=$PWD MOSEKLM_LICENSE_FILE=$HOME/mosek/mosek.lic
PY=/home/yiranwang/anaconda3/envs/sage/bin/python
R=/home/yiranwang/combinartorics/TriSearch-Calabi-Yau

$PY -m pytest test/test_rl_jump_env.py test/test_face_move_scorer.py -q

$PY -m rl_jump.smoke --rows $R/data/cy/favorable_v1/train/h11_15.samples.jsonl --lines 0,1,2,3 \
    --envs 8 --inline 1 --episodes 2 --horizon 20 --out_dir $R/runs/rl_jump_smoke_20261104/h11_15
$PY -m rl_jump.smoke --rows $R/data/cy/favorable_v1/train/h11_20.samples.jsonl --lines 0,1,2,3 \
    --envs 8 --inline 1 --episodes 2 --horizon 20 --out_dir $R/runs/rl_jump_smoke_20261104/h11_20
$PY -m rl_jump.smoke --rows $R/data/cy/favorable_large_v1/train/h11_35.samples.jsonl --lines 0 \
    --envs 1 --inline 1 --episodes 1 --horizon 10 --out_dir $R/runs/rl_jump_smoke_20261104/h11_35
```

Each run writes `summary.json` and `steps.jsonl` (per-step info and entry stats) under `--out_dir`.
Logs are in `R/runs/rl_jump_smoke_20261104/logs/`. Smoke outputs are in `R/runs/rl_jump_smoke_20261104/`.

## Smoke results (random-init policy, CPU, 2026-11-04)

Per-step columns are totals divided by env steps. Reset-state entries are included in the totals.

| | h11=15 | h11=20 | h11=35 |
|---|---|---|---|
| polytopes / envs / episodes / H | 4 / 8 / 2 / 20 | 4 / 8 / 2 / 20 | 1 / 1 / 1 / 10 |
| env steps | 320 | 320 | 10 |
| setup per env (codebook + build), s | 1.24 (max 1.46) | 1.35 (max 1.48) | 14.3 |
| candidate decode (LP) ms/step | 103 (4.8 ms/call, 21.3 calls/step) | 300 (8.9 ms/call, 33.6 calls/step) | 503 (14.3 ms/call, 35.2 calls/step) |
| infeasible fraction of proposals | 2.8% | 21.3% | 18.5% |
| feasible candidates per state (of K=32) | 19.7 | 25.2 | 26.1 |
| objective query ms/query | 178 | 262 | 758 |
| objective queries/step | 0.92 | 1.00 | 1.10 |
| objective ms/step | 164 | 263 | 833 |
| objective cache hit rate | 12.5% | 4.5% | 0% |
| decode cache hit rate | 48% | 14% | 4% |
| observation build ms/step | 7.4 | 9.0 | 10.6 |
| policy forward ms/call (batch) | 7.7 (8 envs) | 9.2 (8 envs) | 3.5 (1 env) |
| policy ms/env-step | 0.96 | 1.15 | 3.5 |
| steps/s overall | 17.7 | 11.1 | 0.74 |
| step failures | 0 | 0 | 0 |
| episode best dlnV (random policy): mean / max | 2.16 / 4.81 (16 episodes) | 2.13 / 3.93 (16 episodes) | 0.27 / 0.27 (1 episode) |

Note on h11=15: the neighbourhood holds only about 22 distinct proposals, so K=32 is not reached.
Objective cache hits are low on h11=20 (4.5%) and h11=35 (0%); I did not measure the cause. The best-dlnV numbers come from the random-init head, so they describe a random policy, not
a learned one. Both episodes of an env share its caches, which is why the h11=15 cache hit rate is
above zero.

## Tests

`test/test_rl_jump_env.py` (7 tests, h11=15 favorable_v1 row 0, real geometry):

- reset/step reward equals `log V_next - log V_current`, with the volumes computed through the worker's
  `objective` request (an independent code path);
- each candidate's face triangles, applied as a move, give exactly its decoded FRT; every 2-face of the
  decoded child equals its codebook entry at the candidate DNA; the observation's removed/added ids equal
  the parent topology's ids for that move;
- a flip-sized candidate (symdiff 4) yields the same removed ids and added node triples as
  `core.cy_two_face_data.build_two_face_data` for that flip circuit;
- stepping into a known DNA (and resetting to a cached start) costs no query, and the cache hit is reported;
- a repeated objective request for one DNA is one query;
- an injected tip-solver failure on one candidate gives reward 0, leaves the state unchanged, and drops
  only that candidate;
- the batched actor logits (two envs in one graph batch) equal `FaceMoveScorer.score` (mean2) on the same moves.

`test/test_face_move_scorer.py` (4 existing tests) passes after the `move_index` refactor.

Result (2026-11-04): `11 passed, 1 warning in 10.24s` for both files together.

## Deviations and choices

- **Near set.** "Smallest symmetric difference" is implemented as the `near_k=8` nearest entries with
  random tie-breaking. Taking only the exact minimum would often be the whole flip level, which can be
  hundreds of entries.
- **Proposals are per state.** Candidate draws are seeded by `(seed, DNA)`, not by the episode RNG.
  This makes objective caching correct across paths. It also means candidates are fixed per state.
- **Feasible candidates only.** Infeasible proposals are dropped before observation, so the action
  count varies per state (as the spec asked).
- **Start counted as a query.** The start objective goes through the same counter and cache as steps.
- **Process count.** Smoke runs use 8 processes in total: 7 spawned workers plus the caller, which
  hosts the 8th env inline. The policy also runs in the caller.
- **No shared codebook across processes.** Each env builds its own codebook. Envs sharing a polytope
  repeat the setup (1–1.5 s on h11=15/20, 14 s on h11=35).
- **Scorer refactor.** The mapping in `FaceMoveScorer.score` moved into `move_index` (same code,
  returns the same lists). Its tests pass unchanged.
- **Timeout mechanism.** The env is in-process, so there is no worker timeout. A `SIGALRM` limit
  (default 300 s, main thread only) stands in for it. It was not exercised in the smoke runs.
- **Leftover probe output.** `R/runs/rl_jump_smoke_20261104/dev_h11_15_probe/` is a 2-env, 3-step
  dry run from before the real runs. It is not part of the results. I could not delete it because
  the shell refused a command containing `rm -rf`.
