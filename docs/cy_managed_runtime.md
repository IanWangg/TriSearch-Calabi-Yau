# Managed CY rollout runtime

The training, evaluation, random rollout, and policy rollout entrypoints use
one managed geometry pool for collection construction, expansion, and geometry
objectives. The trainer retains exact lightweight states, not CYTools or Sage
geometry objects. Serial mode uses one isolated worker; parallel mode defaults
to at most eight workers, further limited by CPU count and memory headroom.

## Running

Run from the repository root in the Sage environment. The existing `main.sh`
workload enables these settings:

```bash
python scripts/train_cy.py \
  --dataset_path data/cy/output_random_flip/cy_reflexive_dataset_random_flip.samples.jsonl \
  --reward max_tri \
  --num_iterations 1000 \
  --num_states 128 \
  --rollout_length 20 \
  --use_multiprocessing \
  --memory_budget_gb 64 \
  --runtime_cache_gb 16 \
  --iteration_metrics_path runs/managed_training/iteration_metrics.jsonl \
  --checkpoint_path "$PWD/runs/managed_training/checkpoints"
```

The resource controls also apply to `scripts/eval_cy.py`,
`tools/rollout_cy_random.py`, and `tools/rollout_cy_policy.py`.

| Control | Default | Meaning |
| --- | --- | --- |
| `memory_budget_gb` | 64 | Aggregate process-tree RAM operating budget, in GiB |
| `runtime_cache_gb` | 16 | Combined retained-cache allowance; zero disables optional retention |
| `transition_num_workers` | 0 | Automatic, at most eight; explicit positive values override selection |
| `transition_task_timeout_sec` | 300 | Deadline per immutable geometry request |
| `policy_max_graph_size` | 250000 | Graph-work limit for physical policy batches; zero disables chunking |
| `action_order` | canonical | Worker-independent action indices; native order is opt-in |
| `profile_cuda_timing` | false | Opt-in CUDA synchronization for detailed timings |

`--no-use_multiprocessing` selects one worker, not inline geometry. Only the
`spawn` setting is supported; `fork` and `forkserver` are rejected. Legacy
chunksize and minimum-batch flags remain accepted, but managed requests use a
bounded one-state dispatch window even for small batches. `state_cache_mode`
and entry caps remain supported; `full` cannot override byte limits.

## Memory and exact history

Optional caches are bounded on admission, not only periodically pruned.
Configuration transport caches are reserved first; the remaining allowance is
split among the training engine, evaluation engine, parent tensor caches, and
worker caches. Each engine partitions its share among resident graph nodes,
hot states, exact-objective memoization, and a small SQLite page cache.
Oversized entries bypass retention. Current rollout states and actions stay
alive until the step finishes even when the cache allowance is zero.

Expansion results contain a circuit and exact removed/added simplices for each
destination. Only selected destination states are materialized. Immutable point
configurations are shared and interned on the worker connection. The PPO buffer
retains CPU model inputs instead of both rich states and tensors, and is released
immediately after the update. Reusable coordinates and simplex topology are
cached within the tensor allowance.

Discovery identities, materialization identities, expanded identities, and
count-bonus visitation counts live in an exact SQLite store with compressed full
keys. Eviction never resets a count or substitutes a hash-only approximate
identity. Cumulative `graph_nodes` and `graph_edges` can grow while resident
memory remains bounded. Default history files live under `runs/runtime/` and
are removed on normal close. SIGKILL may leave temporary history files on disk;
they are not loaded automatically. History is run-local: a new engine rejects
a nonempty explicit `history_path`. Policy checkpoints still restore weights
only, not optimizer state, rollout history, or iteration counters.

The cache allowance is not a bound on every allocation. Input/base states, live
rollout tensors, model/optimizer state, active batches, and native-library heaps
need additional headroom. Byte estimates are conservative accounting estimates;
the process-tree monitor measures actual RSS, including native allocations.

## Pressure handling and process cleanup

```text
trainer
  guardian (independent Linux subreaper)
    geometry worker group 1 -> native geometry children
    geometry worker group 2 -> native geometry children
    ...
```

The guardian samples aggregate trainer/descendant RSS every 0.5 seconds. A lower
applicable cgroup memory limit reduces the configured budget. At 80% pressure,
dispatch pauses and reclaimable caches are trimmed; continued pressure fails
explicitly. At 90%, the runtime initiates a controlled stop and worker cleanup.
Training checks the guard during rollout, bootstrap/PPO boundaries, and physical
PPO batches, including cached rollouts. The optional training `max_rss_gb` remains
an additional stricter trainer-only limit. JSONL `memory.job` reports current and
peak aggregate RSS, trainer RSS, worker RSS, budget, and pressure. `memory.train`
and `memory.eval` report resident nodes, edges, estimated bytes, and evictions.

This is an operating guard, not a kernel-enforced hard RAM reservation. An abrupt
allocation between samples can still trigger the system OOM killer. A handled
guard or SIGTERM after model initialization saves current policy weights to
`oom_guard_interrupted.pth` and `latest.pth`; SIGKILL cannot save weights.

The independent guardian detects trainer death by PID and process start time,
terminates worker process groups, and reaps adopted descendants, including
native children that start another session. Cleanup escalates TERM to KILL.
Parent-death signaling and trainer-side guardian-failure cleanup provide
additional protection. These guarantees are Linux-specific and cover processes
owned by this runtime, not unrelated pre-existing jobs. Like any userspace
cleanup, they cannot guarantee immediate removal of a kernel-uninterruptible
process or recovery if the entire supervision tree is killed simultaneously.

Unexpected worker exit retries the immutable request once in a replacement
worker. Python geometry exceptions and task timeouts fail explicitly, shut down
the owned runtime, and never fall back to native geometry inside the trainer.
Pools and engines are closed on setup failures as well as normal completion.
Library callers should close collections/engines or use the shared runtime
context; the entrypoints already do so.

## Speed and compatibility

Workers receive compact states rather than pickled native object graphs. Native
thread limits avoid nested all-CPU geometry parallelism. Bounded concurrent
initialization and dispatch avoid artificial per-window sleeps. Policy paths
reuse actor/value encodings where mathematically compatible; indexed SNN
membership construction avoids repeated set searches. CUDA timings do not
synchronize by default.

Policy chunks preserve logical sample order, concatenate logits before one
categorical draw, and accumulate gradients for the same logical PPO minibatch.
Training BatchNorm paths, including the projected MLP, keep their original
logical batch to preserve normalization semantics. A single oversized graph is
not split or truncated; it still needs enough memory to execute.

Rewards, valid-action sets, transition geometry, terminal flags, PPO formulas,
and model parameter names remain compatible. Canonical ordering intentionally
changes historical native-order action indices; see
[action ordering and reproducibility](cy_action_order.md). Geometry calculations
remain exact CYTools calculations, not surrogates or approximations.

## Validation and replay benchmark

Run the full regression suite with Sage tools on `PATH`:

```bash
PATH=/home/yiranwang/anaconda3/envs/sage/bin:$PATH \
  /home/yiranwang/anaconda3/envs/sage/bin/pytest -q test
```

The implementation validation passed all 262 tests, including GPU tests, in
157.25 seconds. The local JUnit report is
`runs/validation/final_pytest_results.xml`.

The focused tests cover tiny/zero caches, exact counts after eviction, active
state pinning, real one/two-worker canonical geometry parity, objective parity,
PPO chunking, deadlines, worker crashes, trainer/guardian SIGKILL, and native
descendant cleanup. A saved trace benchmark separates cold/warm expansion work,
startup, serialization, and aggregate RSS:

```bash
python tools/benchmark_cy_runtime.py --help
```

The initial 32-state, 20-polytope trace in
`runs/benchmark_cy_runtime/representative_32_states.json` measured 0.936 s cold /
0.277 s warm for the current rich compatibility path, versus 0.275 s / 0.270 s
with eight compact workers. All compared candidate/transition signatures
matched. This is 3.40x cold expansion speed and 1.02x warm expansion speed, not
a whole-training speedup or comparison against a historical unmodified checkout.
The report includes a saved trace and source fingerprints for repeatability.

A completed 30-iteration CPU training validation used four dataset rows, two
workers, 16 rollout slots, five steps, count bonus 0.1, a 0.001 GiB total cache
allowance, two hot states, and physical graph batches limited to 100. It completed
PPO and periodic evaluation with 2,021 resident-graph evictions and at most two
resident train graph nodes. Peak aggregate RSS was 1.219 GiB; the last ten
iterations ranged from 1,247.48 to 1,247.78 MiB. The trainer, guardian, and workers
all exited. Metrics are in
`runs/managed_eviction_validation/iteration_metrics.jsonl`. This exercises
eviction and memory stability on a small dataset, not full-workload convergence.

The separate full-size 1,000-iteration validation writes
`runs/managed_runtime_acceptance/iteration_metrics.jsonl`. It was still running
at implementation handoff: seven completed iterations, 4.788 GiB sampled peak
aggregate RSS, and roughly 90–106 seconds per rollout. Those early observations
are not a completed 1,000-iteration acceptance result or evidence of its final
memory plateau.
