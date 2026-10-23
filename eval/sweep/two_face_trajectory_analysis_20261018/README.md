# Two_face trajectory 与已观测状态图分析

当前记录支持的判断是：**two_face 模型学到了有用的局部动作筛选信号；配合 BeFS 时能在部分 polytope 上显著改善查询效率，但长程搜索不稳定，不能宣称整体优于普通 BeFS 或 GA。** Beam、提案数量、checkpoint 和 polytope 都会明显影响结果。搜索更深不保证找到更好的区域。

分析于 2026_10_18 完成，未修改模型、训练、evaluation 核心逻辑或原始结果，未运行新训练、几何展开、KCUP 求值或 policy inference。全部分析使用 `sage` 环境。

## 数据与适用范围

| 数据 | 完整 source runs | 完整 trajectories | polytope 数 | 状态 |
| --- | ---: | ---: | ---: | --- |
| h11=30 | 61 | 265（245 RL + 20 baseline） | 10 | 仅纳入完成的配置；中断和取消的 runs 排除 |
| h11=50 | 96 | 440（420 RL + 20 baseline） | 10 | 原 sweep 全部完成 |
| 合计 | 157 | 705 | 20 | 每个 polytope 一个共享起点 |

原始研究：

- [h11_30 sweep](../../results/runs/h11_30_two_face_checkpoint_sweep_20261011_202310/README.md)
- [h11_50 sweep](../../results/runs/h11_50_two_face_checkpoint_sweep_20261012_000945/README.md)
- checkpoint：`runs/cy_two_face_kcup_h11_15_20261008_112756_477353/checkpoints/{400,600,800}.pth`。

训练来自 h11=15，评估使用 h11=30/50。每组各有五个 favorable 和五个 non_favorable polytope，按原研究的来源顺序选择，不能当作整个数据集的均匀随机样本。每个配置只有一个 search seed（0）和每 polytope 一个起点。

**版本边界：这些 evaluation 使用 commit `1866f7c` 时的网络，checkpoint 确实包含 `triangle_dual.0.*` 与 `triangle_dual.1.*` 权重。当前分支 `90231dd` 已简化 two_face encoder，去掉该部分。这里的结果属于保存的旧 checkpoint，不能直接当作当前简化模型的性能，也未用不兼容的新结构重新加载旧权重。**

所有端点统一使用 `query_index <= 1000` 的原始 best CY volume；起点免费，超预算尾部排除。历史 favorable baseline 的 native 预算可以大于 1000，仍保留原 metadata。比较先在同一 polytope 上算 ratio，再取中位数；不是两个 cohort 中位数的比值。差异在 `1e-6` 相对容差内视为平局。未完成配置不填零，不参与统计。

## 模型表现

下表是有代表性的已完成配置，**属于本次 sweep 的后验描述，不是独立验证集选出的通用最优配置**。`stochastic_8` 表示 actor 按概率无放回选八个不同目标；`all` 跳过 actor，只保留 `ln(volume) + 0.9 * critic` 排序。Beam width 固定为 4。

| 数据 / 配置 | 配对数 | 胜普通 BeFS | 对 BeFS 的 volume ratio 中位数 | 对 GA 的 ratio 中位数 |
| --- | ---: | ---: | ---: | ---: |
| h11=30 / 400 / Beam / stochastic_4 | 10 | 6 | 1.180 | 1.011 |
| h11=30 / 400 / BeFS / stochastic_8 | 10 | 1 | 0.855 | 0.645 |
| h11=30 / 400 / BeFS / all | 10 | 4（另有 3 平） | 1.000 | 0.891 |
| h11=30 / 600 / BeFS / stochastic_8，只有 non_favorable 完成 | 5 | 4 | 3.170 | 5.408 |
| h11=50 / 400 / BeFS / stochastic_8 | 10 | 6 | 2.354 | 0.094 |
| h11=50 / 600 / BeFS / all | 10 | 5 | 1.916 | 0.141 |

h11=30 的 checkpoint 600 favorable BeFS 未完成，不能把五个 non_favorable 上的优势推广成十个 polytope 的结论。相同的 400 / BeFS / stochastic_8 在 h11=30 只有 1/10 胜，在 h11=50 是 6/10 胜，说明超参数效果具有明显的实例依赖。

h11=50 上，400 / BeFS / stochastic_8 对普通 BeFS 的逐 polytope ratio 范围是 **0.00809 到 253.48**；中位数有优势仍可能包含严重失败。它只在 3/10 polytope 胜 GA。GA 使用 DNA 交叉/变异，不受相同局部 flip 邻接限制；这里比较的是逻辑 objective 查询效率，不是等同的搜索动作，也没有新增 wall_time benchmark。

完整证据：[h11_50 端点热图](plots/analysis_20261018_02/h11_50_paired_endpoint_heatmap.png)、[h11_30 端点热图](plots/analysis_20261018_02/h11_30_paired_endpoint_heatmap.png)、[逐配置配对汇总](plots/analysis_20261018_02/paired_summary_all_polytopes.csv)、[分组配对汇总](plots/analysis_20261018_02/paired_summary_by_group.csv)。所有图另存 PDF。

## 图上的搜索行为

节点使用日志中的 canonical `evaluation_state_key`，即 ambient 2-face triangulation 等价类。边只使用已查询的 `source_evaluation_state_key -> evaluation_state_key`，按无向 flip 边合并；不同 h11、favorability 和 polytope 的身份分开。GA 的状态可以参与 volume 对照，但交叉/变异不加入 flip 边。

| 数据 | 已观测 flip 节点 | 已观测 flip 边 | 包括 GA 后的已观测状态 |
| --- | ---: | ---: | ---: |
| h11=30 | 161,561 | 170,395 | 166,526 |
| h11=50 | 300,480 | 310,413 | 305,312 |

这是多个搜索的**已观测子图**。大部分查询节点从未展开，因此它的中位度数为 1 是观测方式造成的，不能据此说真实图稀疏。原研究独立的 32 状态 / polytope 邻域采样中，h11=50 distinct two_face degree 中位数分别为 non_favorable 27、favorable 35，最大值 119；该采样也不是全图均匀采样。20 个初始点的完整邻居数量已和轨迹第一轮的并集独立核对一致。

图中距离是**观测子图内**的最短距离，对完整图距离只是上界；等价类的不同 FRST 代表还可能暴露不同邻域。日志 `depth` 则是首次发现路径长度，二者分开记录。不声称完整覆盖率、全局 optimum、必要能垒或真实死路。

### 搜索宽度与深度解释了 Beam 的主要弱点

以下是 h11=50 各 checkpoint / mode / polytope 的描述性中位数；这些行不是独立重复实验。

| 每父节点提案数 | Beam 展开次数 | Beam 最大 depth | BeFS 展开次数 | BeFS 最大 depth |
| --- | ---: | ---: | ---: | ---: |
| all | 47.5 | 13 | 45.5 | 39 |
| 12 | 84 | 22 | 84 | 48 |
| 8 | 125 | 32 | 125 | 55 |
| 4 | 250 | 64 | 250 | 72 |

全部邻居查询时，一个父节点会消耗较多预算。Beam 又把后续预算分给同层四个父节点，推进层数更少；BeFS 可以集中推进高分分支。相同 checkpoint 的 all-neighbor BeFS 对 Beam 在 **26/30** 配对中获胜，volume ratio 中位数为 **26.69**。这比把差距笼统归因为网络本身更符合轨迹证据。

减小提案数能加深搜索，但也会丢掉分支。在 h11=50 中，k=4 的最佳祖先路径通常包含更多下降步骤：Beam 中位数 17，BeFS 19.5；k=12 分别为 1 和 7。这证明实际路径可以非单调，不证明这些下降都是到达最优区域所必需的。

### 一次成功和一次失败

固定使用 400 / BeFS / stochastic_8，避免用不同配置解释不同案例：

| h11=50 案例 | RL best volume | 普通 BeFS best volume | RL / BeFS | RL 最佳路径 depth | RL 最大搜索 depth |
| --- | ---: | ---: | ---: | ---: | ---: |
| favorable P0 | 1.32944e11 | 2.10889e9 | 63.04 | 39 | 43 |
| non_favorable P2 | 1.56777e10 | 1.93798e12 | 0.00809 | 41 | 51 |

favorable P0 的 RL 最佳路径包含 12 次显著下降，最大相对历史路径峰值回撤约 0.302 dex（约减半）；普通 BeFS 的最佳路径只有 15 层。这里模型引导与较深路径同时出现，并找到了更高值区域。

non_favorable P2 则相反：RL 最深到 51 层，普通 BeFS 到 42 层；RL 在第 500 至 1000 次查询期间 best 仅增加 0.00570 dex（约 1.3%），普通 BeFS 却在第 553 次查询找到 1.94e12。RL 并非没有走深，而是把预算用在了较差区域。仅凭日志无法进一步唯一分解为 actor 漏选、critic 错排或代表邻域差异。

![已观测图与最佳祖先路径](plots/analysis_20261018_02/observed_graph_case_studies.png)

灰色密度是已查询状态数，不是图的均匀密度。彩线是最佳节点的真实父子祖先链，不是按时间把 BeFS 不同 frontier 节点连成一条物理轨迹。逐节点证据见 [case_paths](plots/analysis_20261018_02/graph_case_paths.csv)。

## Actor 与 critic 分别提供了什么

**Actor 有即时目标的局部筛选信号，但 top-4 经常漏掉最佳邻居。** 20 个公共初始状态都有完整邻居对照，因此这里不需要重新推理或假设未查询候选值。h11=50 的 BeFS 首轮结果如下：

| checkpoint | greedy_4 命中最佳邻居 | greedy_8 | greedy_12 |
| --- | ---: | ---: | ---: |
| 400 | 2/10 | 6/10 | 6/10 |
| 600 | 3/10 | 7/10 | 7/10 |
| 800 | 4/10 | 6/10 | 8/10 |
| 均匀无放回选目标的解析期望 | 1.34/10 | 2.68/10 | 4.02/10 |

均匀参考用各初始节点真实 degree 和最佳目标数计算超几何命中概率，没有运行随机基线，也不是显著性检验。这只检查即时 volume，不把即时最佳动作当作长程最优动作。

有限 k 下的 greedy 与 stochastic 是更接近的配对对照。h11=50、checkpoint 400、BeFS、k=8 时，stochastic 在 7/10 polytope 胜 greedy，ratio 中位数为 2.002。但 h11=30 同配置只有 1 胜、5 平、4 负，不能宣称 stochastic 普遍更好。有限 k 对 all-neighbor 的提升还混合了查询分配与搜索深度效应，不能全部记为 actor 的功劳。

**Critic 确实改变了前沿排序，但现有日志不足以验证 value 校准。** 根据已发现、未展开状态重放 BeFS frontier，h11=50 all-neighbor RL BeFS 的非最大-volume 前沿选择比例中位数约 50%。不过，各轨迹中最大 `log10(frontier_max_volume / expanded_parent_volume)` 的中位数仅 0.075 dex，很多改序发生在接近的值之间。普通 BeFS 的同一重放检查全部得到零改序。

最干净的现成 critic 对照是 all-neighbor RL BeFS 与普通 BeFS：actor 都不筛选，差异来自 metric + critic 优先级及其后续路径。h11=50 的 ratio 中位数从 checkpoint 400 的 **0.591**，到 600 的 **1.916**，再到 800 的 **0.778**。critic 的效果不随训练迭代单调改善。

这批 evaluation 未保存每个候选的 actor logits、critic value 或完整未选动作集合；不能直接计算 value calibration、预测 return 相关性或所有深层动作的 top-k recall。图和 ancestry 只能提供行为证据。

## 训练迭代没有带来稳定改善

h11=50 的 14 个 RL 配置 × 10 个 polytope 可完整配对：checkpoint 800 对 400 只赢 **51/140**，ratio 中位数 **0.678**。favorable 子组为 20/70 胜、中位数 0.500；non_favorable 为 31/70 胜、中位数 0.975。配对内起点和搜索设置相同，但不同设置共享实例，不能把 140 当作独立随机样本。

与此同时，800 的首轮 greedy_4 / greedy_12 命中数比 400 更高。因此这批数据不支持把后期整体退步简单解释成“actor 连局部好动作都不会选”；后续状态分布、长程 priority 和搜索分支更值得检查。它也不足以单独证明训练过拟合或 critic 偏置，需要匹配的 h11=15 验证轨迹与 value 数据才能区分。

## 数值检查与结论边界

- 157 个 source run 全部通过已有 `read_comparison`；验证保存起点、checkpoint SHA256、查询计数、预算、cumulative best 和成功状态。
- h11=50 的 440 个端点与原研究表逐条重新计算一致；20 个初始 degree 与独立 geometry diagnostics 一致。
- 所有 graph search 在预算前缀内都没有重复查询 two_face 节点；父节点先于子节点发现，记录 depth 与父链一致。GA 保留原重复计费语义。
- 同状态跨 run volume 存在 **5 个**超出 `1e-6` 容差的状态：h11=50 原有 4 个，加上本次发现的 h11=30 favorable P0 一个（相对差 1.625e-6）。原值未修正，容差未放宽，不能将跨源 metric consistency 标为通过。
- 这 5 个状态共出现 37 次，均不是记录中的 running best 或预算端点 best。h11=50 原有几何诊断将四例归因于求解数值顺序；h11=30 新例没有重算几何，原因不作同样断言。“未成为 best”也不保证换求解器后搜索优先级完全不变。
- 观测池最佳值仅用于描述性相对 gap，不是已知全局最优值。这里没有独立 SNN 对照，不能把跨架构性能优势归因于 two_face 表示。

对应证据：[validation](results/analysis_20261018_01/validation.json)、[独立检查及数值例外](results/analysis_20261018_01/independent_validation.json)、[source hashes](results/analysis_20261018_01/provenance.json)、[逐轨迹诊断](results/analysis_20261018_01/trajectory_metrics.csv)、[图统计](results/analysis_20261018_01/graph_summary.csv)。

## 下一步实验建议

现有结果最值得优先验证的是 **BeFS + stochastic_8 与 BeFS + all-neighbor**，同时保留普通 BeFS / GA。它们分别检查有限提案的搜索效率与不含 actor 的 critic 作用。不能凭这次后验结果宣布一个固定 checkpoint 对所有 polytope 最优。

为区分模型与搜索器的影响，可在同一 checkpoint、起点和 k 下增加 uniform preselection；这是已有数据里缺少的等查询深度 actor 对照。记录已有 `search_observer` 支持的候选分数、actor 概率、visited mask 和 frontier，并用已观测后续结果核查排序。先用多 seed、匹配的 h11=15/30/50 起点验证，再考虑改变网络或训练目标。本次未执行这些新增实验，也没有继续被用户停止的 h11=30 队列。

## 复现

脚本只写本研究的派生产物；新的 ingest / render 目录必须不存在。不同运行使用新的 output 名称，不覆盖原结果。`--reuse_derived` 只重算保存的 compact trajectory 数据。

从仓库根目录，在 `sage` 环境运行：

```bash
study_dir=eval/sweep/two_face_trajectory_analysis_20261018
analysis_id=analysis_reproduction_01
mkdir -p "$study_dir/logs"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python -B "$study_dir/scripts/analyze_trajectories.py" \
  --output_dir "$study_dir/results/$analysis_id" \
  > "$study_dir/logs/${analysis_id}.log" 2>&1
python -B "$study_dir/scripts/validate_analysis.py" \
  --source_dir "$study_dir/results/$analysis_id" \
  > "$study_dir/logs/${analysis_id}_validation.log" 2>&1
python -B "$study_dir/scripts/render_analysis.py" \
  --source_dir "$study_dir/results/$analysis_id" \
  --output_dir "$study_dir/plots/$analysis_id" \
  > "$study_dir/logs/${analysis_id}_plots.log" 2>&1
```

`analyze_trajectories.py` 按两份 `run_plan.json` 选择 completed jobs 与获准复用的 baselines；`configuration_availability.csv` 明确保留未完成项。原始 two_face key 的 SHA256 是派生图节点身份，查询的 local node id 可通过对应 `*_graph.json` 查回该 hash，再追溯原日志。完整网络权重不参与离线计算。

最终图表位于 `plots/analysis_20261018_02/`；`plots/analysis_20261018_01/` 为加入图案例前的派生版本。CSV 和 `plot_config.json` 记录统计定义。主要图包括两组端点热图、逐 polytope 原始 volume 曲线、搜索诊断、actor 首轮筛选与 graph case studies。
