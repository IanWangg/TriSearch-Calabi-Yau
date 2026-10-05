# CY evaluation

Development rules and evaluation semantics are maintained in [AGENTS.md](AGENTS.md).

## Directory organization

Use these locations for new evaluation work; paths below are relative to the
repository root. Choose a unique run name using `_` between words.

| Purpose | Location |
| --- | --- |
| Disposable smoke results | `eval/smoke_tests/results/<run_name>/` |
| Smoke stdout / stderr logs | `eval/smoke_tests/logs/<run_name>.log` |
| Smoke temporary files, dedicated setups and caches | `eval/smoke_tests/tmp/<run_name>/`, `eval/smoke_tests/setups/`, `eval/smoke_tests/cache/` |
| Shared persistent setups and download cache | `eval/data/setups/`, `eval/data/cache/` |
| Regular full evaluation results | `eval/results/runs/<run_name>/` |
| Regular launcher / console logs | `eval/logs/<run_name>.log` |
| Series of research experiments | `eval/sweep/<study_name>/` |

**All evaluation smoke artifacts belong under the fixed `eval/smoke_tests/`
root**, including preflight checks for sweeps. The user periodically deletes this
directory. Keep reusable code, configurations, checkpoints and lasting research
results outside it; test implementations remain in `test/`. Smoke runs may read
persistent shared setups, but full evaluations and sweeps must not depend on
files inside the disposable directory.

Persist stdout / stderr, launcher output and `.log` files in a **`logs/`
subfolder**. Parallel benchmarks already keep algorithm subprocess logs in
`<run_dir>/logs/`. The structured result files (`queries.jsonl`, `expansions.jsonl`,
`transitions.jsonl`, `rollouts.jsonl`) retain their documented locations within
each run and remain inputs to the existing readers.

**Future sweeps live in `eval/sweep/<study_name>/`.** This includes research on
combining RL policy with other graph search algorithms, ablations of proposal or
scoring rules, and hyperparameter tuning such as beam width, policy proposal count
and value discount. Each study uses `configs/` for study configurations,
`scripts/` for launch and analysis scripts, `results/<run_name>/` for full trials,
`plots/` for aggregate figures, and `logs/` for runtime logs. Its `README.md`
records the research question, search space, fixed settings, setup, checkpoint,
budgets, seeds, commands and the mapping from configurations to results. Reuse the
evaluation engine and result format; shared algorithm implementations belong in
the existing `eval/algorithm/` and related modules. Keep generated artifacts
ignored by Git and reusable code/configurations under version control.

The CLI does not automatically route smoke or sweep runs: pass `--output_dir`
explicitly. For pytest smoke, use an independent `--basetemp` under
`eval/smoke_tests/tmp/`. Create the relevant `logs/` directory before redirecting
output; examples below show both output paths and console log capture.

## Reusable full benchmark and plots

Run these commands from the repository root in `sage`. JSON configurations contain
`EvaluationSpec` fields only: explicit CLI arguments override the file, and missing
optional fields retain the usual defaults. Unknown fields, invalid types/choices,
and missing required parameters fail before evaluation. Relative paths are relative
to the working directory (the repository root). Existing CLI-only calls still work.

The checked-in `configs/benchmark_h11_21.json` compares all eight algorithms on
five polytopes, ten shared distinct starts each, and 1000 objective queries per
rollout. It pins HF revision `60c0e119a03608418df538191f65da3f43b5b819` and enables
`skip_insufficient_starts`: take the first five candidates whose bounded sampler
supplies ten distinct FRSTs, recording every rejected candidate. Prepare the shared setup once:

```bash
python scripts/eval_cy.py --config eval/configs/benchmark_h11_21.json \
  --setup_only --output_dir eval/data/setups/h11_21_5_polytopes_10_starts_seed_0
```

Run a small-budget smoke using the **same** setup, then the full benchmark:

```bash
eval_run_id=$(date +%Y%m%d_%H%M%S)
mkdir -p eval/smoke_tests/logs eval/logs
python scripts/eval_cy.py --config eval/configs/benchmark_h11_21.json \
  --setup_path eval/data/setups/h11_21_5_polytopes_10_starts_seed_0 \
  --objective_budget 5 --parallel_resources eval/configs/benchmark_resources.json \
  --output_dir "eval/smoke_tests/results/benchmark_h11_21_${eval_run_id}" \
  --plot_results > "eval/smoke_tests/logs/benchmark_h11_21_${eval_run_id}.log" 2>&1

python scripts/eval_cy.py --config eval/configs/benchmark_h11_21.json \
  --setup_path eval/data/setups/h11_21_5_polytopes_10_starts_seed_0 \
  --parallel_resources eval/configs/benchmark_resources.json \
  --output_dir "eval/results/runs/benchmark_h11_21_${eval_run_id}" \
  --plot_results > "eval/logs/benchmark_h11_21_${eval_run_id}.log" 2>&1
```

`--parallel_resources` launches one ordinary evaluation CLI process per algorithm;
all read the same saved setup. Without it, evaluation remains sequential. Each
resource entry specifies `cpu_count`, `transition_num_workers`, `memory_budget_gb`,
and an optional `gpu_index`. CPU affinity sets are allocated without overlap from
the caller's allowed CPUs, including library threads created during imports;
BLAS/OpenMP threads are capped at one per process.

The `benchmark_h11_30.json` configuration uses five polytopes, five shared starts
per polytope, and 2000 queries per start for value beam with policy top 4,
Beam, BeFS, and GA. It selects the first five candidates whose existing bounded
sampler supplies five distinct FRSTs and records skipped candidates; the first
strict h11=30 attempt produced only two starts for candidate 0.
`benchmark_h11_30_value_beam_no_policy_filter.json` evaluates
the same setup with `policy_proposal_count=-1`, skipping actor inference entirely.
Both retain beam width 4, discount 0.9, and the earlier h11=15 checkpoint.
The h11=30 resource file reserves GPU 3 for the main value beam; the separate
no-filter run uses GPU 4 and must receive a disjoint CPU affinity via `--cpu_ids`.
Check current machine availability before reusing these device assignments.

RL jobs require distinct available CUDA devices unless `force_cpu=true` (in that
case omit their GPU assignments). Baselines do not see CUDA devices. CPU and GPU
overcommit is rejected before launching children. Memory budgets cover each
algorithm's process tree using the existing managed runtime, not a cache limit.

The provided resource file reserves 2 CPUs / 1 geometry worker / 8 GiB for each
baseline and 8 CPUs / 4 workers / 16 GiB for each RL algorithm, on GPUs 1, 2, 3:
34 CPUs and 88 GiB in total, including `cyopt_ga`. Adapt this file to the machine. Each algorithm retains
the configured cache allowance (1 GiB by default). Each RL process loads its own
model once; completed jobs must agree on the checkpoint SHA256.

A parallel output directory contains `benchmark.json` with resolved resources,
commands, PIDs and completion states, `specs/`, `logs/`, and `algorithms/<name>/`
with the standard evaluation outputs. Failures preserve outputs and terminate
remaining owned jobs; incomplete comparisons are never plotted as complete.
Output directories are new for each run. Use `--output_dir` to choose one explicitly.

Replot either a sequential run or a parallel benchmark without loading geometry:

```bash
python scripts/plot_eval_cy.py --run_dir eval/results/runs/<run_name>
```

Plots currently cover `max_kcup` and read both result format versions 1 and 2.
Derived files default to `<run_dir>/plots/`; `--output_dir` selects another folder.
Replotting replaces only named derived artifacts, never source logs.

| Artifact | Meaning |
| --- | --- |
| `best_volume_by_polytope.pdf/png` | Raw best CY volume vs queries, median and IQR across starts, base 10 log volume axis |
| `best_volume_overall.pdf/png` | Raw best CY volume, median and IQR across all paired starts; each polytope has the same number of starts |
| `budget_endpoint_distribution.pdf/png` | Per-polytope endpoint boxplots with all individual starts |
| `benchmark_summary.csv` | Per-rollout budget endpoint and complete-search best, actual query count, overshoot and termination |
| `polytope_curves.csv`, `overall_curves.csv` | Exact plotted statistics at every integer query index |
| `algorithm_summary.csv` | Endpoint best volume median/IQR, actual queries, overshoot and early-stop counts |
| `plot_config.json` | Source, statistical definitions and plotting package versions |

All new comparisons use absolute best CY volume, with no normalization to the
initial volume and no relative improvement plots. Volume axes use base 10 log
scales with raw volume tick values. Endpoint rankings use median best volume.
Derived plot metadata uses format version 2; CSVs contain raw volume statistics
instead of log gains. Raw evaluation formats and historical artifacts are unchanged.

When an optimal log10 volume is supplied for a specific polytope, its comparison
can additionally plot `gap(q) = reference_log10_kcup - log10(best_kcup(q))`.
Compute each start's gap before taking its mean and sample standard deviation
(`ddof=1`) across starts at each query budget. Plot mean ± SD on a linear gap axis,
without clipping gaps or bands; SD is not a confidence interval. Save the reference,
formula, per-start gaps and summary curves with the comparison. A rounded reference
stays rounded; do not replace it silently with the best observed value or reuse it
for other polytopes. This gap is relative to the supplied optimum, not the initial state.

Budget endpoint values use only queries with `query_index <= objective_budget`.
Completing the final parent can improve the complete-search best beyond that
endpoint; the CSV retains both values. Normal early terminations carry the last
best value forward. Failed/missing trajectories, nonconsecutive queries,
unpaired starts, inconsistent counts and nonpositive volumes raise an error.
Readers stream query JSONL and retain numeric curves, not full state traces.

## Genetic algorithm (cyopt)

`cyopt_ga` calls the installed upstream `cyopt.GA`. The optional dependency was
installed in `sage` from `/home/yiranwang/combinartorics/cyopt`, version 1.1.0,
commit `5a9ba057180c8e5b47f72680c85269b0264e2aea`. Installation uses `pip install
--no-deps /home/yiranwang/combinartorics/cyopt` to preserve the project's custom
CYTools installation. Other algorithms do not require cyopt. Run metadata records
the installed Python source hash, version, installation origin and available local
source revision/status; the installed source hash identifies the actual code used.

The representation is a tuple of 2-face triangulation choices. Codebooks are
sorted by canonical simplices, shared across this polytope's starts, and written
to `cyopt_encoding/polytope_<index>.json` with a checksum and all starting DNAs.
Faces with at most `ga_face_max_points=12` points are fully enumerated; larger
faces use up to `ga_face_samples=1000` seeded upstream samples. Restrictions of
all supplied starts are included before evaluating objectives. For the h11=21
five-polytope setup all faces have at most 10 points, so enumeration is complete.
Preparation uses no objective queries; its wall time is recorded separately.

Each independent GA population begins with that rollout's exact shared FRST;
its known initial objective is reused for free. The remaining initial members
are uniformly sampled DNA and their objective queries consume budget. A repeated
initial DNA resolves to the exact supplied FRST. Distinct full FRST starts can
share DNA: in particular all ten starts of polytope 0 have the same empty DNA.
Such singleton spaces end with `dna_space_singleton`, retaining all paired starts.

Defaults are population 50, tournament selection (`k=3`), one-point crossover,
mutation probability `ga_mutation_rate=0.1`, one mutated gene, and `ga_elitism=1`.
`ga_population_size`, `ga_mutation_rate`, `ga_elitism`,
`ga_max_stalled_generations`, `ga_face_max_points` and `ga_face_samples` are shared
JSON/CLI settings. Elitism is capped at DNA space size minus one for tiny spaces.
Fitness is negative raw `max_kcup` volume (cyopt minimizes); all reported
objectives remain positive raw volumes. Upstream selection, crossover, mutation,
elitism and within-generation uniqueness are used unchanged. This is the cyopt
GA with these declared settings, not a reproduction of every paper hyperparameter.

cyopt's fitness cache is disabled (`cache_size=0`): each valid fitness request
passes through the common rollout counter, including repeated DNA/cache hits.
Elite carry-over reuses known fitness without another query. A DNA for which
upstream reconstruction returns `None` receives internal infinite fitness and
is recorded under `rejected_candidates` in the generation event; it is a free
geometry feasibility check, not a volume query. Unexpected decoder/solver errors
fail the run. After `ga_max_stalled_generations=20` consecutive generations with
no valid queries, the rollout ends with `no_feasible_offspring`.

Decoded DNA aliases use the engine's existing bounded hot-state cache, sharing
its byte/entry limits and pressure/close lifecycle; no separate cache allocation
is added. Keys include polytope, codebook checksum and DNA. Turning caches off
also disables these aliases. Every repeated valid fitness request is still counted.

GA stops before the next fitness request once the exact objective budget is
reached, including partway through initialization/a generation. It has no budget
overshoot. Neighbor methods retain their complete-parent-expansion boundary;
comparison curves for every algorithm use only queries at or below the budget.
GA has zero physical transitions. Its expansion events use `kind=population` and
count initialization/generations; `generation_index=0` is initialization.
Candidate queries carry DNA and generation, with null action/source/depth rather
than fabricated flip edges. The v2 result fields are retained with this explicitly
recorded population event kind. Expansion counts across families measure different
operations and must not be interpreted as a common compute budget.

Add GA to an existing completed benchmark without repeating the other methods:

```bash
python scripts/eval_cy.py --config eval/configs/benchmark_h11_21.json \
  --algorithms cyopt_ga \
  --setup_path eval/data/setups/h11_21_5_polytopes_10_starts_seed_0 \
  --parallel_resources eval/configs/benchmark_resources.json \
  --output_dir eval/results/runs/<new_ga_run> --plot_results

python scripts/plot_eval_cy.py --run_dir eval/results/runs/<completed_baselines> \
  --additional_run_dirs eval/results/runs/<new_ga_run> \
  --output_dir eval/results/runs/<combined_comparison>/plots
```

Combination requires complete runs with disjoint algorithm names, identical setup,
paired initial states/objectives and matching scientific settings. Resource
settings may differ. RL sources must also agree on checkpoint SHA256 and model
architecture. `plot_config.json` records all source directories and the checkpoint hash. Existing
raw logs and previous comparison plots are preserved; a combined plot requires
an explicit output directory. GA uses 2 CPUs, 1 geometry worker, 8 GiB RAM and no GPU
in the supplied resource plan.

Run from the repository root in the `sage` conda environment. Evaluation uses
FRST starts, `two_neighbors` transitions and the existing training geometry,
reward and cache implementations. Available baselines are Random, Greedy,
best first search (BeFS) and layer-wise Beam Search.

```bash
python scripts/eval_cy.py \
  --num_polytopes 1 --h11 12 --num_starts 1 --objective_budget 5
```

All four size parameters are required. The budget applies independently to every
algorithm and every start. Defaults are `--algorithms random greedy`,
`--reward_function max_kcup`, `--seed 0`, and enabled runtime caching with a
combined allowance of `1 GiB`. `--num_vertices`, `--favorable` and
`--non_favorable` are optional selection filters; by default neither favorability
class is excluded. Use `--no_cache_states` to disable runtime caching.

To compare all four baselines, with a beam width of 4:

```bash
python scripts/eval_cy.py \
  --num_polytopes 1 --h11 12 --num_starts 1 --objective_budget 20 \
  --algorithms random greedy best_first beam_search --beam_width 4
```

## RL policy family

```bash
python scripts/eval_cy.py \
  --num_polytopes 1 --h11 12 --num_starts 2 --objective_budget 20 \
  --algorithms rl_stochastic_policy rl_policy_beam_search rl_value_beam_search \
  --beam_width 4 --value_discount 0.9
```

All three algorithms share one loaded SNN actor/critic. The default checkpoint
directory is `runs/cy_snn_kcup_h11_15_20260921_123000_1584461/checkpoints/`.
`--policy_checkpoint` accepts a file or directory; directories prefer `latest.pth`,
then the latest numbered checkpoint. Missing or incompatible weights fail.
The default architecture matches that run: EGNN with `snn_simplex`, coordinate
dimension 4, hidden/output width 64 and 3 layers. Model dimensions and
`--subcomplex_actor_type` can be overridden for compatible alternative checkpoints.
Evaluation uses the original coordinates without augmentation.

| Algorithm | Selection rule |
| --- | --- |
| `rl_stochastic_policy` | Mask actions leading to visited states, renormalize the remaining probabilities, sample one action and move there. |
| `rl_policy_beam_search` | Each parent proposes up to k unseen targets by policy probability; query all proposals, then retain top k by cumulative log probability. |
| `rl_value_beam_search` | Each parent proposes up to m unseen targets by policy probability; query all proposals, then retain top k by natural log of the child kcup volume + discount × child critic value. |
| `rl_value_best_first` | Pop the highest-scoring state from a global frontier, propose up to m unseen targets by policy, and add all queried children with natural log of the child kcup volume + discount × child critic value. |
| `rl_metric_value_beam_search` | Compatibility alias for `rl_value_beam_search`. |
| `rl_metric_value_best_first` | Compatibility alias for `rl_value_best_first`. |

Here k is `beam_width`. `--policy_proposal_count` controls **value beam and value BeFS**:
value beam's default follows k; BeFS defaults to **4**, independently of k.
A positive integer sets m explicitly, and `-1` includes all unseen neighbors.
With `-1`, both value searches skip actor logits and use
a critic-only SNN path. Critic inference needs no child-neighbor enumeration.
`--value_discount` defaults to **0.9**, independently of the default checkpoint's
training gamma of 0.95.

**RL value beam / BeFS means metric + value search.** Both algorithms currently
support only `max_kcup` and score candidates as
`ln(volume_child) + value_discount * V(child)`. The metric is the natural log of
the absolute child volume; it is computed from the recorded objective without
extra queries. Parent volume and cumulative path reward do not enter the score.
The single-step reward + value search implementations have been removed.
The `rl_metric_value_*` names remain compatibility aliases with identical behavior.
For runs containing only these value searches (including aliases), `value_discount`
accepts any finite nonnegative coefficient, including values above 1.
Other runs retain the `[0, 1]` restriction; use separate runs for larger-coefficient
sweeps against objective-ranked baselines.

Select BeFS with `--algorithms rl_value_best_first`. For example,
`--policy_proposal_count 8` queries up to eight unseen neighbors per parent;
`--policy_proposal_count -1` disables policy pre-selection. BeFS retains every
queried, unexpanded candidate in a global priority queue, without a frontier cap.
Only one parent per start is expanded in each scheduling round, with batches
shared across starts. Scores use the absolute child metric plus critic value;
rediscovered states are skipped without updating their first score. Equal scores retain first discovery order, and dead ends do not
discard other queued states. `beam_width` has no effect on this algorithm.

The h11=21 ablation in `configs/benchmark_h11_21_value_beam_no_pre_selection.json`
uses the same five polytopes, ten starts, checkpoint, query budget 1000, beam width
4 and discount 0.9 as the benchmark, with `policy_proposal_count=-1`. It keeps the
existing search deduplication and beam pruning, and disables only policy proposal
selection. Run it against the saved shared setup, first with a small budget:

```bash
eval_run_id=$(date +%Y%m%d_%H%M%S)
mkdir -p eval/smoke_tests/logs eval/logs
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/eval_cy.py \
  --config eval/configs/benchmark_h11_21_value_beam_no_pre_selection.json \
  --setup_path eval/data/setups/h11_21_5_polytopes_10_starts_seed_0 \
  --objective_budget 5 \
  --output_dir "eval/smoke_tests/results/value_beam_no_pre_selection_${eval_run_id}" \
  --plot_results > "eval/smoke_tests/logs/value_beam_no_pre_selection_${eval_run_id}.log" 2>&1

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/eval_cy.py \
  --config eval/configs/benchmark_h11_21_value_beam_no_pre_selection.json \
  --setup_path eval/data/setups/h11_21_5_polytopes_10_starts_seed_0 \
  --output_dir "eval/results/runs/value_beam_no_pre_selection_${eval_run_id}" \
  --plot_results > "eval/logs/value_beam_no_pre_selection_${eval_run_id}.log" 2>&1
```

Each command creates a separate run. Verify that its checkpoint SHA256 matches
the baseline and that `summary.json` reports zero `action_batches` and
`action_states`. Compare budget endpoints from query logs; all-neighbor expansion
can overshoot the budget by more than the baseline's three-query maximum. The
existing `--additional_run_dirs` merger accepts disjoint algorithms with matching
settings, so these two value-beam variants require a separate labeled comparison.

Stochastic evaluation keeps a separate visited set for every start, initialized
with the start itself. It stops with `no_neighbors` if no legal actions exist,
or `no_unvisited_neighbors` if every target was already visited. The latter
records a zero-query expansion without sampling or moving. Cache settings do
not change visited history. Random/Greedy retain their original revisit behavior.

RL beams and value BeFS deduplicate targets before proposing/querying them. Unqueried
actions excluded by a proposal limit are not marked discovered; queried states
pruned from the next beam remain discovered permanently. Policy scores use the
original full-action distribution, without renormalization after deduplication.
Policy ties follow canonical action order; selected proposals are queried in
canonical order. Parent and next-layer score ties follow first discovery order.

All proposals count, including those later pruned. Budgets remain independent
per start and finish one parent at a time. With finite m, value beam and BeFS
query at most m candidates per parent and overshoot by at most m−1; value beam
queries at most k×m per complete layer. Policy beam uses m=k. With `-1`, the number
of unseen neighbors determines the cost. RL beams and BeFS record zero physical transitions.

Every RL algorithm advances **all polytopes and starts together**. Policy states
and value candidates are submitted as logical batches, split into physical GPU
batches by `--policy_max_graph_size` (default 250000). `--gpu_index` defaults to
0; `--force_cpu` forces CPU, and CPU is used when CUDA is unavailable. Per-start
RNGs make random draws independent of other trajectories' progress or batching.
Free neighbor/policy prefetch can include later beam parents; objective queries
and expansion events include only the budget-admitted prefix. BeFS enumerates
only its selected parent, not the entire queued frontier.

Geometry requests use the existing managed pool; `--transition_num_workers`
controls CPU geometry parallelism (default 1). Each logical query still counts
on a cache hit or when physical work is shared. RL settings do not invalidate
saved setups. Runtime caches remain separate between algorithms.

`config.json` records the resolved checkpoint path/SHA256, architecture, device,
proposal count and discount. `policy.resolved_policy_proposal_counts` records the
actual count for each RL algorithm; the legacy singular field retains its
value-beam resolution rule for compatibility. `policy.value_score_definitions`
records `ln_objective_plus_discounted_value` for every value-search algorithm.
Historical results are not rewritten: older `rl_value_*` runs may record
`step_reward_plus_discounted_value` and must be interpreted using their original
scoring metadata, rather than the current algorithm name alone. `summary.json` adds `policy_stats` with logical and
physical batch counts, states scored, wall time and CUDA peak allocated bytes.
Fine-grained inference timings measure host-side work by default; use
`--profile_cuda_timing` for synchronized CUDA timings.

## Data and shared starts

To evaluate explicit vertices, pass `--polytope_file <path>` (also available in
JSON configs). The shared training loader accepts JSON lists, objects containing
`polytopes` / `polytope_specs`, or JSONL; each record contains `vertices` as a list
of four-coordinate N-lattice points. Matrix columns must be transposed to these
point rows. Local loading skips HF, takes the requested number of records in file
order, and validates dimension, reflexivity, actual CY h11 and any vertex/favorability
constraints. An incorrect h11 in either the record or evaluation config fails.
Local inputs cannot use `skip_insufficient_starts`. Setup metadata preserves the
source path, file SHA256 and computed Hodge numbers. Reuse `--polytope_file` with
`--setup_path` to match setup parameters; loading saved starts does not require the
source file to remain available. HF revision/cache settings are unused for local
input, and the default HF path remains unchanged.

The default source is [calabi_yau_data on Hugging Face](https://huggingface.co/datasets/calabi-yau-data/polytopes-4d).
The loader reuses `load_hugging_face_4d_n_lattice_polytope_specs`, streaming the
vertex-count partitions and taking the first N matching records. This is an
ordered selection, not a uniform sample of the full dataset. Source vertices
are interpreted in the **N lattice**: the requested CY `h11` is the source
table's **`h12`**, and the training loader verifies this with CYTools.

Setup reuses the training FRST sampler with `fast_only=True`, `make_star=True`,
and `include_points_interior_to_facets=False`, including its bounded retry and
fallback behavior. Starts must be distinct by their canonical full simplices.
By default, insufficient starts or polytopes cause an error; no replacements or
duplicate padding are performed. `--skip_insufficient_starts` explicitly enables
source-order eligibility filtering: a candidate returning fewer than the required
number of distinct FRSTs is recorded in `setup.json` under `skipped_polytopes`
(full source record/vertices, seed, requested/obtained counts, reason and sampler
diagnostics), then selection continues until the exact polytope count is met.
Unexpected sampler errors, invalid FRSTs and duplicate states still fail. Sampling
never uses search objectives to select candidates. Candidate indices retain gaps
after skips, and seeds derive from those stable indices. This policy participates
in setup validation; it cannot silently reuse a setup with the opposite policy.
Strict v1 setups remain compatible. Distinct FRSTs need not represent distinct CY geometries.

```bash
python scripts/eval_cy.py \
  --num_polytopes 1 --h11 12 --num_starts 1 --objective_budget 5 \
  --setup_only --output_dir eval/data/setups/example_setup

python scripts/eval_cy.py \
  --num_polytopes 1 --h11 12 --num_starts 1 --objective_budget 20 \
  --setup_path eval/data/setups/example_setup --no_cache_states
```

Saved setups contain `setup.json` (source records, resolved HF commit, parameters,
sampling diagnostics and checksum) and training-compatible `samples.jsonl`.
Loading a setup is offline and checks its checksum and selection parameters.
Algorithms, budgets, `beam_width`, rewards and cache settings may change without
regenerating starts. The setup seed must still match. `--hf_revision` can pin a dataset commit;
when loading a saved setup, retain its originally requested revision argument.
The data source is CC BY SA 4.0; see its dataset card for attribution.

## Budget and algorithms

- The initial objective is logged at query index 0 and is free.
- Each call to `context.evaluate_action` costs one logical objective query,
  including cache hits and repeated states. Neighbor enumeration alone is free.
- BeFS and Beam skip previously discovered states before querying their objective.
  Comparing already stored scores is free. Discovery is local to each start,
  includes the initial state and pruned Beam candidates, and is independent of
  runtime caches or other starts.
- Check for positive remaining budget before expanding each parent. Once started,
  finish that parent's queries, even if the budget is crossed. For Beam, stop
  before the next parent, including parents in the same layer. Record the actual
  query count and `budget_overshoot`; never truncate the reported count.
- With remaining budget `r` and `m` unseen neighbors of the last BeFS/Beam parent,
  the overshoot is `max(0, m - r)`. Beam width `k` does not bound `m`, so `k²` is
  **not** a bound on the overshoot. `two_neighbors` names a CYTools neighbor mode;
  it does not mean that each state has two neighbors.
- Budget exhaustion ends every algorithm. Random/Greedy also stop at a state
  without neighbors; BeFS/Beam stop when their frontier is empty. A dead end in
  one branch does not discard other frontier states. FRSTs do not trigger
  termination or reset. Zero budget records only the initial value.
- Best objective includes every evaluated state, including unselected candidates
  and the initial state. Results for `max_kcup` contain raw CY volume; ordering
  neighbors by volume is equivalent to ordering their log volume difference
  reward from the same current state. Nonpositive/nonfinite kcup values fail.

| Algorithm | Search rule | Queries per parent |
| --- | --- | --- |
| `random` | Uniformly select one valid neighbor and move there | 1 |
| `greedy` | Evaluate all valid neighbors and move to the best, even if worse than the current state | Number of valid neighbors |
| `best_first` | Pop the best state from the global frontier and add its unseen children | Number of unseen neighbors |
| `beam_search` | Expand current-layer parents in score order; keep the best `beam_width` new children for the next layer | Number of unseen neighbors |

Best tracking and objective baselines follow the registered min/max direction.
RL search scores are maximized; RL value searches support only `max_kcup`.
Baseline BeFS ranks states by their objective; RL value BeFS ranks by natural log
of child volume plus weighted critic value. Neither has a frontier cap.
Beam starts with one state, never carries parents into the next layer, and does
not revisit pruned states. `beam_width` defaults to 4 and must be a positive integer.
Greedy ties follow canonical action order; BeFS/Beam ties follow first discovery
order. Random/Greedy may revisit states, paying for each query again.

`transition_count` records actual walk moves for Random/Greedy/stochastic RL and is zero for
BeFS and all beam algorithms. `expansion_count` records logical parent expansions for every algorithm,
including cache hits and expansions with zero queries. A frontier switch is not
logged as a physical transition. Query `depth` is the discovered child's depth;
expansion `depth` is its parent's depth (not necessarily shortest-path distance).

Search seeds are derived from the global seed, algorithm name, polytope index
and start index. Algorithm ordering does not change trajectories. Caches are
shared across starts of the same algorithm; each algorithm has its own managed
runtime. Disabling caching sets engine and worker cache budgets to zero and
releases transition graphs between rounds. Required input/current state objects
and SQLite identity history remain. Frontier algorithms also retain the state
records needed for pending parents and their per-start discovery set; these are
algorithm state, not optional caches. Frontier entries do not retain CYTools
geometry objects. Disabling caches does not remove saved datasets.

## Layout, results and extension points

| Location | Responsibility |
| --- | --- |
| `config.py`, `setup.py` | Specifications and persisted FRST starts |
| `data/loader.py`, `data/cache/`, `data/setups/` | HF loading and generated inputs |
| `algorithm/` | Walk/search protocols and the four baselines |
| `rollout.py`, `pipeline.py` | Budgeted single-start search and full runs |
| `batched_rollout.py`, `policy.py` | Batched RL search and shared inference service |
| `results/writer.py`, `results/runs/` | Streaming logs and generated run outputs |
| `smoke_tests/` | Disposable smoke artifacts, including their logs and temporary files |
| `logs/` | Regular evaluation launcher / console logs |
| `sweep/<study_name>/` | RL/search combination studies and hyperparameter sweeps |

`run_evaluation(spec, setup=None, output_dir=None)` returns an `EvaluationResult`.
Results go to a new directory and never overwrite an existing run. They contain:

- `config.json`: complete specification, setup reference, package versions and
  repository commit/dirty status.
- `queries.jsonl`: initial and candidate objectives with cumulative best values.
  Query `round_index` and `expansion_index` both identify the parent expansion;
  they are 0 for the initial objective.
- `expansions.jsonl`: parent key, score and depth, candidate count, query counts
  before/after expansion and completion/failure status.
- `transitions.jsonl`: actual moves and objective-query counts for each round.
- `rollouts.jsonl`: one summary per algorithm/polytope/start, including initial,
  best values/state keys, query/expansion/transition counts and termination reason.
- `summary.json`: successful completion counts and runtime cache statistics.
- `runtime/`: per-algorithm SQLite identity history.

New result `config.json` and `summary.json` use `format_version=2`.
`final_state_key` and `final_objective` are omitted for every algorithm. Initial
and best fields remain; best includes all queried candidates, including those
discarded by Beam. Saved setup format stays at version 1 and existing result
directories are never rewritten. Consumers of version 1 results must account
for these removed fields and the added expansion logs when reading version 2.

Failures preserve completed JSONL records and write `failure.json` with context
and a traceback; worker and history resources are closed. A failed run has no
successful `summary.json`.

Walk algorithms implement `select_action(context)` and return an `EvaluatedAction`
obtained from `context.evaluate_action`. The context also supplies the current
state/value, actions, objective direction, RNG and remaining budget. Frontier
algorithms implement `search(context)` using `context.initial`, `priority(node)`,
`remaining_budget` and `expand(node)`. Fully consume each expansion iterator before
checking the budget or expanding another parent. This interface centralizes
deduplication, objective validation, accounting and best tracking. Algorithms
must not call objective providers or geometry workers directly. Register
custom zero-argument factories through `run_evaluation(..., algorithm_factories=...)`;
a fresh algorithm instance is constructed for each start. `run_rollout` exposes
query/expansion/transition callbacks for additional consumers without retaining full traces
in memory. RL algorithms implement `RLAlgorithm.propose` and `score` and run
through `run_batched_rollouts`; `run_rollout(..., policy=...)` also supports one
RL start through the same runner. `RLValueBestFirst` reuses value-beam proposal
and scoring rules with `is_best_first=True` to retain a global frontier.
The compatible `score_candidate` hook receives recorded parent/child objectives;
by default it delegates to the original four-argument `score` after computing
transition reward. Both RL value searches override this hook for absolute metric + value scoring.
`run_evaluation(..., policy=...)` can inject
a scorer implementing `score_actions` and `score_values`. Geometry, objective
functions and serialization remain shared with training.

## Validation

```bash
mkdir -p eval/smoke_tests/tmp
python -m pytest test/test_eval_*.py -q \
  --basetemp "eval/smoke_tests/tmp/regression_$(date +%Y%m%d_%H%M%S)"

eval_smoke_id=hf_smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p eval/smoke_tests/logs eval/smoke_tests/tmp
CY_EVAL_HF_SMOKE=1 python -m pytest test/test_eval_pipeline.py -k hugging_face -q \
  --basetemp "eval/smoke_tests/tmp/${eval_smoke_id}" \
  > "eval/smoke_tests/logs/${eval_smoke_id}.log" 2>&1
```

The offline tests include real kcup transitions with two shared starts and both
cache settings. The opt-in online smoke downloads one h11=12 polytope, generates
one FRST start, and evaluates all four algorithms with objective budget 5 under
both cache settings. Synthetic graphs separately check frontier order, pruning,
cycle deduplication and the per-parent budget boundary.
