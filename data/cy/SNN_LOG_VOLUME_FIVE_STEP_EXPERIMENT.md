# SNN + Log-Volume Five-Step Experiment

## Setup

- Dataset: `data/cy/two_neighbors_h11_12.samples.jsonl`
- Split: 8 training polytopes and 4 held-out polytopes (`28, 22, 23, 26`)
- Seed: 0
- Policy: EGNN state encoder with the `snn_simplex` action head
- Navigation: four-vertex `two_neighbors` circuit actions
- Reward: `log(V_next) - log(V_current)` for `max_cy_volume`
- Training: 25 PPO iterations, 32 stochastic environments, 5 steps
- Per-iteration evaluation: 32 deterministic held-out environments, 5 steps
- Checkpoints: `latest.pth` every 5 iterations and `25.pth` at iteration 25

The 25 iterations completed normally. Their combined iteration time was
1,907.27 seconds (31.79 minutes), including 21.32 seconds of held-out
evaluation and excluding dataset construction and initial-state filtering.
`25.pth`, `final.pth`, and `latest.pth` contain identical model tensors.

`Train sec` is the complete iteration time minus held-out evaluation time.
All volume columns are raw Kcup volumes; only the PPO return is log-transformed.
Per-iteration held-out rows sample 32 slots with replacement from the four
held-out initial FRSTs.

## Learning Table

| Iter | Train init | Train final | Train best | Train gain | Train improved | Train log return | Train sec | Eval init | Eval final | Eval best | Eval gain | Eval improved | Eval log return | Eval sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 478.03 | 1793.42 | 2654.57 | 2176.54 | 100.0% | 1.2646 | 120.31 | 1095.48 | 1701.43 | 1838.39 | 742.91 | 78.1% | 0.3437 | 8.51 |
| 2 | 504.13 | 1664.26 | 2248.90 | 1744.77 | 96.9% | 1.1435 | 100.75 | 882.99 | 1221.44 | 1377.96 | 494.97 | 75.0% | 0.2317 | 0.08 |
| 3 | 461.19 | 1538.05 | 2346.90 | 1885.70 | 100.0% | 1.2295 | 110.68 | 842.24 | 1358.44 | 1475.83 | 633.59 | 81.2% | 0.3962 | 0.08 |
| 4 | 405.67 | 1316.23 | 2155.52 | 1749.85 | 96.9% | 1.1819 | 98.72 | 945.84 | 1312.63 | 1527.85 | 582.02 | 65.6% | 0.2167 | 0.08 |
| 5 | 666.60 | 1558.31 | 2200.34 | 1533.74 | 93.8% | 0.8969 | 75.80 | 1028.27 | 1739.90 | 1739.90 | 711.62 | 100.0% | 0.5150 | 1.64 |
| 6 | 511.17 | 1222.09 | 1963.92 | 1452.75 | 100.0% | 0.9338 | 88.61 | 801.02 | 1346.21 | 1515.73 | 714.71 | 100.0% | 0.5445 | 1.14 |
| 7 | 545.36 | 1671.08 | 2579.21 | 2033.84 | 100.0% | 1.1591 | 84.89 | 1152.80 | 1509.26 | 2080.83 | 928.03 | 100.0% | 0.5160 | 2.82 |
| 8 | 535.75 | 1641.56 | 2082.44 | 1546.69 | 96.9% | 1.1419 | 73.46 | 945.44 | 1566.77 | 2011.19 | 1065.75 | 100.0% | 0.7525 | 0.08 |
| 9 | 460.32 | 1545.39 | 2297.84 | 1837.52 | 100.0% | 1.2311 | 78.42 | 1009.32 | 1570.69 | 2007.24 | 997.92 | 100.0% | 0.6684 | 0.08 |
| 10 | 615.21 | 1592.54 | 2696.48 | 2081.27 | 100.0% | 0.9966 | 73.47 | 1019.79 | 1524.51 | 1631.71 | 611.91 | 65.6% | 0.6789 | 0.08 |
| 11 | 535.61 | 2055.52 | 2862.24 | 2326.63 | 100.0% | 1.2509 | 67.89 | 1109.67 | 1508.88 | 1691.22 | 581.56 | 65.6% | 0.3885 | 2.39 |
| 12 | 479.56 | 1241.36 | 1720.88 | 1241.32 | 100.0% | 1.0274 | 75.30 | 930.78 | 1281.94 | 1600.40 | 669.62 | 75.0% | 0.3903 | 0.08 |
| 13 | 494.12 | 1242.37 | 1897.49 | 1403.37 | 100.0% | 0.9447 | 65.05 | 987.06 | 1407.00 | 1634.55 | 647.49 | 78.1% | 0.4155 | 0.08 |
| 14 | 622.41 | 1586.84 | 2240.60 | 1618.19 | 100.0% | 0.9077 | 69.11 | 972.00 | 1233.41 | 1587.68 | 615.68 | 68.8% | 0.3260 | 0.08 |
| 15 | 426.99 | 1458.05 | 1857.75 | 1430.76 | 100.0% | 1.1484 | 78.71 | 962.00 | 1191.38 | 1651.96 | 689.95 | 71.9% | 0.2003 | 3.33 |
| 16 | 600.94 | 1551.42 | 1983.82 | 1382.88 | 93.8% | 0.9991 | 75.20 | 1002.75 | 997.62 | 1598.80 | 596.04 | 68.8% | -0.0146 | 0.08 |
| 17 | 486.84 | 1345.95 | 1782.83 | 1295.99 | 100.0% | 0.8422 | 52.32 | 989.91 | 1102.40 | 1648.81 | 658.90 | 75.0% | 0.0411 | 0.08 |
| 18 | 421.93 | 1394.87 | 1777.64 | 1355.71 | 100.0% | 1.1553 | 61.87 | 799.68 | 917.46 | 1553.60 | 753.92 | 81.2% | 0.1313 | 0.08 |
| 19 | 541.51 | 1253.90 | 1821.04 | 1279.53 | 100.0% | 0.8679 | 67.79 | 1126.24 | 1185.79 | 1683.69 | 557.46 | 65.6% | -0.0201 | 0.08 |
| 20 | 563.49 | 1625.00 | 2299.99 | 1736.49 | 100.0% | 1.0019 | 61.77 | 956.77 | 1196.36 | 1663.87 | 707.10 | 75.0% | 0.1829 | 0.08 |
| 21 | 521.79 | 1526.46 | 1910.21 | 1388.42 | 100.0% | 1.0399 | 59.26 | 752.36 | 963.57 | 1550.02 | 797.65 | 81.2% | 0.2532 | 0.08 |
| 22 | 581.80 | 2191.44 | 2764.51 | 2182.71 | 100.0% | 1.3139 | 62.69 | 1009.32 | 1145.28 | 1655.54 | 646.21 | 71.9% | 0.0784 | 0.08 |
| 23 | 515.73 | 1332.60 | 1841.64 | 1325.91 | 93.8% | 0.8652 | 70.45 | 892.41 | 1104.76 | 1627.38 | 734.97 | 78.1% | 0.1769 | 0.08 |
| 24 | 326.53 | 1379.29 | 1818.87 | 1492.34 | 100.0% | 1.2352 | 57.11 | 1025.02 | 1130.34 | 1619.79 | 594.77 | 62.5% | 0.1308 | 0.08 |
| 25 | 408.98 | 1750.44 | 2252.79 | 1843.81 | 100.0% | 1.2952 | 56.32 | 930.78 | 1005.83 | 1600.40 | 669.62 | 75.0% | 0.0526 | 0.08 |

## Final Held-Out Evaluation

`final.pth` was evaluated deterministically for five steps on each held-out
initial FRST exactly once. The aggregate raw initial, final, and best volume
means were 975.72, 1,054.55, and 1,630.16. Mean best-volume improvement was
654.44, three of four FRSTs improved, mean log return was 0.0213, and rollout
runtime was 13.22 seconds.

| Polytope | Initial volume | Final volume | Best volume | Best gain |
| ---: | ---: | ---: | ---: | ---: |
| 22 | 1820.33 | 1714.67 | 1820.33 | 0.00 |
| 23 | 214.83 | 314.79 | 1249.33 | 1034.50 |
| 26 | 1198.94 | 342.75 | 1604.98 | 406.04 |
| 28 | 668.78 | 1846.00 | 1846.00 | 1177.22 |

No untrained baseline was run or included in this experiment.

## Artifacts

- Per-iteration JSONL: `runs/snn_log_volume_five_step_seed0/iteration_metrics.jsonl`
- Training log: `runs/snn_log_volume_five_step_seed0/train_performance.log`
- Final held-out summary: `runs/snn_log_volume_five_step_seed0/final_held_out_eval.json`
- Final held-out log: `runs/snn_log_volume_five_step_seed0/final_held_out_eval.log`
- Checkpoints: `ckpt/runs/snn_log_volume_five_step_seed0/checkpoints/`
