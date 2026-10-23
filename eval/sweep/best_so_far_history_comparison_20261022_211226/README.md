# Current and historical RL comparison

Compare the frozen best_so_far checkpoint (598 total iterations) with ordinary
BeFS, historical GA and previous triangle_dual checkpoints 400/600/800. Use the
same five h11=30 full FRST starts, one per polytope, and only the first 128
logical queries of each saved run. Checkpoint 600 is the duration-matched
historical reference; other completed settings remain visible, including the
stronger checkpoint_400 BeFS. No per_polytope checkpoint selection is applied.

This is a comparison of trained methods. Architecture, reward, context and
critic weight differ, so it does not isolate the effect of the reward change.
The five pairs are a subset of the current model's 25_start benchmark. The
historical group name `non_favorable` is a source label, not a new validation
of favorability.

`configs/comparison.json` lists all inputs. Paths prefixed by `historical/`
refer to the earlier main checkout's data; `current/` refers to the newer
best_so_far archive. They may share one root in a consolidated archive.
Raw results and setups are required locally and are not published with code.
Recorded absolute setup paths are resolved in memory using the supplied roots;
original evidence is not rewritten.

From the repository root with `sage`:

```bash
study=eval/sweep/best_so_far_history_comparison_20261022_211226
python "$study/scripts/compare_history.py" \
  --historical_root /path/to/historical_archive \
  --current_root /path/to/current_archive \
  --output_dir "$study/results/reproduction_01"
```

The new output directory contains generated `tables/`, `plots/` and
`summary.json`. The script records raw source hashes and validates native result counts,
identical vertices/simplices/initial keys and objectives, checkpoint pairing,
and historical BeFS query prefixes. It runs no geometry or policy inference.
