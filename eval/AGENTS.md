# eval 开发与 evaluation 规则

本文件适用于 `eval/` 及其子目录。修改关联的 CLI 或测试时，也必须保持这里规定的 evaluation 语义。
继续遵守仓库根目录的 [AGENTS.md](../AGENTS.md)；运行示例与结果文件说明见 [README.md](README.md)。

## 1. 开发原则

- 只在当前分支工作，只修改本仓库文件；不 commit、不 push、不执行 `rm -rf`。
- 使用已激活的 `sage` conda 环境，或通过 `conda run -n sage` 运行命令。未经明确批准不安装依赖。
- 文件名、函数名、配置字段、CLI 参数和算法标识使用 `_` 分词，保持与现有代码一致。
- 优先保证正确性、可读性和可复现性。使用最小充分修改，避免无关重构、过度抽象和重依赖。
- 在实现任何新函数之前，先用 `rg` 搜索 `eval/`、`core/`、`mdp/`、`data/cy/`、`reward_functions/` 和相关测试，阅读已有实现及调用方式。
- 已有功能直接复用；接口不够时，优先对共享实现做兼容现有调用的最小扩展。不要复制一份 training 的功能到 `eval/`。
- 性能优化必须保持数据选择、状态身份、objective、查询计数和算法选择规则。需要改变这些语义时，应明确记录变更，并同步更新本文件、README 和相关测试。

### 优先复用的实现

| 能力 | 现有实现 |
| --- | --- |
| HF 下载、分片筛选、N lattice / Hodge 验证 | `data/cy/generate_4d_dataset_hugging_face.py` 中的 `load_hugging_face_4d_n_lattice_polytope_specs` |
| FRST 采样、triangulation 序列化、sample row 的点索引转换 | `data/cy/pipeline.py` 中现有 sampler、`serialize_triangulation` 和 row builder |
| 状态与 simplex 的规范化、状态 key | `mdp/cy_state_record.py` |
| collection、邻居展开、transition materialization、有界缓存 | `mdp/cy_rollout.py` |
| worker、内存控制、资源释放 | `core/cy_managed_runtime.py` 与其使用的 managed process runtime |
| reward、objective 与优化方向 | `reward_functions` 的 `get_reward`、`get_objective`、`infer_goal` |
| evaluation seed 派生 | `eval/config.py` 的 `derive_seed` |

不要重新实现邻居几何、KCUP 求解、状态 key、点标签映射或另一套独立的运行时缓存。

## 2. 模块职责与扩展接口

| 模块 | 职责 |
| --- | --- |
| `config.py` | 定义、验证 evaluation 规格，区分 setup 参数和搜索参数 |
| `data/loader.py` | 封装共享 HF loader，要求严格的 polytope 数量与唯一性 |
| `setup.py` | 生成、验证、保存和离线读取共享起点 |
| `algorithm/` | 根据 context 选择行动或管理搜索前沿 |
| `rollout.py` | 单个起点的查询计费、搜索去重、best tracking、终止条件和事件输出 |
| `batched_rollout.py`、`policy.py` | RL 多起点调度，以及共享 actor / critic 批量推理 |
| `pipeline.py` | 遍历算法、polytope 和起点，管理 runtime 与结果写入 |
| `results/writer.py` | 写入查询、展开、转移、rollout 汇总及失败信息 |
| `results/plotting.py` | 离线读取、配对验证、预算内统计及比较图；不重新运行几何 |
| `parallel.py` | 按资源配置并行调用同一个 CLI，分配互不重叠的 CPU 与独立 GPU，保存运行清单 |
| `../scripts/eval_cy.py` | CLI 参数解析与入口调用 |
| `../test/test_eval_*.py` | evaluation 合约与真实几何验证 |

- 单路径算法实现 `EvaluationAlgorithm.select_action(context)`，返回本轮通过 `context.evaluate_action(action)` 获得的 `EvaluatedAction`。
- 前沿算法实现 `FrontierSearchAlgorithm.search(context)`，通过 `initial`、`priority(node)`、`remaining_budget` 和 `expand(node)` 管理搜索；必须完整消费每个 expansion iterator，再检查预算或开始下一个父节点。
- RL 算法实现 `RLAlgorithm` 的提案与评分规则，由 `run_batched_rollouts` 同步推进每个算法的全部 polytope / 起点。共享 `PolicyScorer.score_actions` / `score_values`；每个起点仍创建新的算法实例。
- 候选 objective 查询统一经过 rollout 的计数逻辑：单路径使用 `evaluate_action`，前沿搜索使用内部调用它的 `expand`。不能直接调用 reward、objective provider 或 worker 来绕过计数。
- 已知的 objective 使用 `context.current_objective` 或 `SearchNode.objective`；排序、移动到已评估候选或将其加入前沿时直接复用数值。
- 通过 `algorithm_factories` 注册扩展算法；每个起点创建新的算法实例，避免算法内部搜索状态泄漏到下一条轨迹。
- 日志通过 rollout 的 query / expansion / transition callbacks 输出。不要为了输出完整轨迹而在算法里无限累积状态。
- 修改保存格式时，明确格式版本和旧数据兼容行为，并补充读取验证；不得静默把不兼容数据当作当前 setup 使用。

## 3. 数据与 setup 规则

### Evaluation 规格

每次 evaluation 必须明确 `num_polytopes`、`h11`、`num_starts` 和 `objective_budget`。
前者是严格的 polytope 数量，`num_starts` 是每个 polytope 的严格起点数。
`objective_budget` 独立应用于每个 `(algorithm, polytope_index, start_index)`；不同轨迹之间不共享剩余预算。

当前默认值为 `seed=0`、`algorithms=("random", "greedy")`、`reward_function="max_kcup"` 和开启运行时缓存。
可选算法为 `random`、`greedy`、`best_first`、`beam_search`、`cyopt_ga`、`rl_stochastic_policy`、`rl_policy_beam_search`、`rl_value_beam_search` 和 `rl_value_best_first`。`beam_width=4` 为默认值，必须为正整数；它只影响 beam 搜索，不参与 setup 参数或校验标识。

RL value beam / BeFS 统一代表 metric + value 搜索，目前仅支持 `max_kcup`。`rl_metric_value_beam_search` 与 `rl_metric_value_best_first` 仅保留为对应算法的兼容别名；单步 reward + value 搜索实现已删除。
本版本使用 `two_neighbors` 的 FRST evaluation。新增其他 transition 模式时，必须明确合法状态、起点和 objective 的适用范围。

### 来源与筛选

- 默认来源为 Hugging Face 的 `calabi-yau-data/polytopes-4d`。指定 `polytope_file` 时，复用 training 的 JSON/JSONL loader 读取本地 N 格顶点，不访问 HF；按文件顺序取严格的前 N 项，验证 4D reflexive、实际 h11 及可选 vertex / favorability 条件，并保存来源路径、SHA256 和 CYTools 的 Hodge 验证结果。
- 本地输入不允许 `skip_insufficient_starts`，不能替换用户提供的 polytope。文件中声明的 h11 与计算值不符时明确报错，不能静默改标。`polytope_file` 路径进入 setup 参数；已保存 setup 可在源文件不再可用时离线读取。旧 HF setup 的参数与校验标识保持兼容。
- 顶点解释为 **N lattice**。目标 CY 的 `h11` 对应源表的 **`h12`**，继续通过共享 loader 做 CYTools 验证。
- 按现有分片和源记录顺序选取前 N 个匹配 polytope，不把这种选择描述为全数据集的均匀随机抽样。
- 显式设置 `skip_insufficient_starts=True` 时，改为按同一来源顺序选取前 N 个能由现有有限重试 sampler 获得足够互异 FRST 的候选。该选项默认关闭；本次 h11=21 的 5×10 benchmark 已获用户授权开启。
- 可选筛选为 `num_vertices` 和 `favorable`；默认不排除任何 favorability 类别。
- 必须保存来源记录、筛选参数、选择方式和解析后的 HF commit。保留原始 source Hodge 字段，避免混淆源表与 CY 的约定。
- 数量不足或出现重复 polytope 时明确报错；不静默缩小样本、重复填充或替换数据。
- 开启上述筛选时，只跳过 sampler 正常返回但起点数不足的候选；不吞掉异常、重复状态或非法 FRST。每个跳过项在 setup 的 `skipped_polytopes` 保存完整源记录、顶点、采样 seed、请求/实际数量、原因和完整 sampler diagnostics。选择方式明确记为 `first_n_with_enough_distinct_frst_in_source_order`。
- 候选 polytope index 保持来源顺序序号，跳过后不重新编号；FRST seed 仍从该序号派生。该策略进入 setup 参数和校验标识；默认严格模式的旧 v1 setup 保持兼容。最终仍须获得严格的 N 个 polytope × S 个互异起点。

### 起点与复用

- 复用 training 的 FRST sampler，参数为 `fast_only=True`、`make_star=True`、`include_points_interior_to_facets=False`，沿用其有限重试与 fallback。
- 每个 polytope 必须获得规定数量的互异 FRST。互异性按 canonical full simplices 判断，不等同于互异 CY 几何。
- 所有待比较算法使用完全相同的起点。起点准备与算法搜索分开，不根据某个算法的搜索结果更换起点。
- 序列化和重建必须保持点标签及 simplex 索引对应关系；使用现有转换函数，不能假定局部 triangulation 索引就是 polytope 索引。
- 保存源数据、起点 simplices、setup 参数、采样 seed、diagnostics 和校验值。保存的 setup 可以离线读取。
- 读取时检查格式版本、校验值和 setup 参数。算法、搜索预算、`beam_width`、reward 和 cache 设置可改变；起点 seed、筛选条件及其余 setup 参数仍须匹配。

## 4. Evaluation 逻辑规则

### 邻居与 objective

- 邻居来自共享引擎的有效候选 actions，使用 canonical 顺序。
- `two_neighbors` 是 CYTools 的邻居定义，**不表示一个状态恰好有两个邻居**；不能硬编码邻居数量为 2。
- 使用 registry 中的 objective 和优化方向；通用算法遵循 `goal="min"` 或 `goal="max"`。
- `max_kcup` 的 objective 是原始 CY volume；其 transition reward 是 `log(V_next) - log(V_current)`。固定当前状态时，两种方式对邻居的排序一致。
- `max_kcup` 的 c=1 MoSEK QP 数值失败时，可重试较小的正 stretch c，并将所得 tip 除以 c 后计算 volume；最小范数 QP 的齐次性保持原 c=1 objective，约束检查 tolerance 必须同步乘 c。原 c=1 成功结果保持不变；缩放重试仍失败后沿用 OSQP/CVXOPT fallback。
- 结果中的 `max_kcup` objective 记录原始 volume。所有 objective 必须有限；`max_kcup` 还必须严格为正。求解失败沿用共享实现的处理，失败后不能伪造数值继续搜索。

### 查询预算

| 事件 | 预算消耗 |
| --- | --- |
| 起点 objective | 0；记录为 `query_index=0` |
| 展开邻居、枚举 actions | 0 |
| 通过 `evaluate_action` 查询一个候选 objective | 1 |
| 查询命中缓存或重复访问同一状态 | 每次仍为 1 |
| 移动到本轮已经评估过的候选 | 不额外计费 |
| 对已保存的 objective 排序、比较，或在查询前去重跳过候选 | 0 |

- 计费单位是**逻辑 objective 查询次数**，不是实际求解次数、唯一状态数或物理转移次数。
- `objective_queries`、`transition_count` 和 `expansion_count` 分别记录，不能用同一个计数器代替。前沿算法的 `transition_count=0`，不能把前沿切换伪造为物理转移。
- 每个父节点展开前检查预算。若尚未耗尽，完成该算法对这个父节点的全部查询；跨过预算后不展开下一个父节点。Beam 在当前层中也遵守此边界，不继续完成整层。
- `expansion_count` 计入每次逻辑父节点展开，包括缓存命中、无邻居或所有邻居均已发现的零查询展开。初始 objective 不算展开。
- `budget_overshoot = max(0, objective_queries - objective_budget)`，必须保留真实查询数，不能截断或改写结果掩盖超预算。
- 例如预算为 5，已经查询 4 次，而下一轮 Greedy 有 3 个邻居：完整评估后查询数为 7，本轮转移 1 次，超预算量为 2。
- BeFS/Beam 的最后一个父节点若有 `m` 个未发现邻居，展开前剩余预算为 `r > 0`，超出量为 `max(0, m-r)`。`beam_width=k` 不限制邻居数，超出量没有 `k²` 上界。

### Baseline 行为

| 算法 | 选择规则 | 一个父节点的查询与转移 |
| --- | --- | --- |
| Random | 在有效邻居中均匀随机选一个，只评估该邻居 | 1 次查询，1 次转移 |
| Greedy | 评估所有有效邻居，按 objective 的优化方向选择最优者 | 邻居数次查询，1 次转移 |
| BeFS (`best_first`) | 从全局前沿取 objective 最优节点，展开并加入未发现的子节点 | 未发现邻居数次查询，0 次转移 |
| Beam (`beam_search`) | 按 objective 顺序展开当前层父节点，从新子节点中保留最优 `beam_width` 个作为下一层 | 未发现邻居数次查询，0 次转移 |

- Greedy 并列时选择 canonical action 顺序中的第一个。
- 即使所有邻居都不优于当前状态，Greedy 也移动到最好的邻居；允许回退和重复访问，不在局部最优处自动停止。
- Greedy 的最后一轮继续使用全部有效邻居，不改成部分邻域搜索，也不因剩余预算不足而跳过整轮。
- BeFS 按节点 objective 排序，不累计路径 reward，不设置任意前沿大小上限；死路不影响继续处理其他前沿节点。
- Beam 初始层只有起点。每层仅保留新子节点，父节点不作为 elite 带入下一层；当前层按 objective 和首次发现顺序展开。
- BeFS/Beam 遵循 objective 的 min/max 方向，objective 并列时按首次发现顺序决定优先级，不依赖集合或堆的偶然顺序。
- 普通 BeFS/Beam 可通过现有 `engine.objective_values` 并行计算同一父节点的未发现邻居；按 canonical action 顺序消费结果，每项仍走 rollout 查询计数，保持完整父节点预算边界并及时关闭批量 iterator。

### 搜索去重

- `two_face_state=False` 为默认值，保持 full-FRST 身份与现有行为。开启时目前只支持 `max_kcup`；所有搜索 visited/discovered/reserved 身份与 objective 缓存使用各 ambient 2-face 的 canonical triangulation key。点标签与面集合由共享 geometry worker 提供，规范化复用 `mdp/cy_state_record.py`；不能将完整 simplices 的所有三点子集当作二维面表示。
- 完整 FRST 与原始 `state.key` 仍用于几何定位、邻居展开和 actor/critic 输入；同类候选按现有提案/查询顺序保留首次成功查询代表与分数。Random/Greedy 与 GA 保留原重复查询计费。开关不参与 setup 校验，互异完整 FRST 起点允许属于同一二维面等价类，搜索历史仍逐起点独立。
- 开启时，v2 查询/转移事件追加 `evaluation_state_key`、`source_evaluation_state_key`、`best_evaluation_state_key`，展开事件追加父节点 `evaluation_state_key`，rollout 追加 `initial_evaluation_state_key` / `best_evaluation_state_key`。Population query 的 source 仍为 null。关闭时不追加这些结果字段。离线 metric 验证按二维面 key 检查同类 volume 的一致性，数值求解相对容差为 `1e-6`；保留完整 key 的几何追溯与起点配对检查。旧结果缺少开关视为 False；并行/合并结果必须采用相同设置。
- BeFS/Beam 在单个 `(algorithm, polytope_index, start_index)` 内按 canonical state key 全局去重，在 objective 查询之前跳过已发现状态。
- 起点立即标为已发现；候选首次成功查询后标为已发现。Beam 丢弃的候选仍属于已发现状态，不能在后续层重新加入。
- 去重排除自环、反向边、重复 action 和多父节点共享子节点；不能把重复邻居再次查询后再去重。
- 去重集合属于算法语义，必须在每个起点重新建立，不能使用引擎跨起点的 history/discovered 集合或缓存代替。缓存命中不等于本次搜索已发现。
- Random/Greedy 不使用此搜索去重规则，保留允许重复访问并逐次计费的行为。

### RL family

- 默认使用 EGNN + `snn_simplex` actor/critic（输入 4 维、hidden/out channels 64、3 层）。checkpoint 默认来自 `runs/cy_snn_kcup_h11_15_20260921_123000_1584461/checkpoints/`，复用 latest 解析与严格加载；一个 evaluation 只加载一次模型。模型与设备参数均记录，不影响 setup。
- `rl_stochastic_policy` 禁止重复访问。起点立即进入本起点的 `visited_state_keys`；屏蔽所有指向已访问状态的动作后，重新归一化剩余 policy 概率并采样。成功查询并移动后加入 visited。没有邻居时为 `no_neighbors`，邻居全部已访问时为 `no_unvisited_neighbors`；后一种情况记录零查询 expansion，不抽样、不移动。
- `rl_policy_beam_search`：每父节点取最多 `beam_width=k` 个 policy 概率最高的未发现目标，全部查询 objective；按累计 `log π` 保留下一层 top k，不作长度归一化。
- `rl_value_beam_search`：每父节点取最多 `policy_proposal_count=m` 个 policy 提案，全部查询 objective；按 `ln(volume_child) + value_discount * V(child)` 保留下一层 top k。
- `rl_value_best_first`：从全局 frontier 取上述 metric + value 分数最大的节点，每次只展开一个父节点，所有已查询的未展开子节点保留在全局优先队列中，不设置 frontier 上限。分数并列按首次发现顺序；初始节点直接展开，死路后继续处理其余 frontier。
- RL value beam / BeFS 的 metric 明确为绝对 volume 的自然对数，不减父节点 metric，不累计路径 reward。直接复用已经查询的 child objective，不能为 metric 再查询。非 `max_kcup` 输入必须明确报错。`rl_metric_value_*` 仅是对应算法的兼容别名，不代表另一种评分规则。
- `RLAlgorithm.score_candidate` 接收已记录的父/子 objective；默认实现保留四参数 `score` 扩展兼容。两个 RL value 搜索均覆盖该入口使用绝对 metric，不调用 transition reward。
- 仅包含 RL value 搜索（含兼容别名）的运行允许 `value_discount` 为任意非负有限系数（包括大于 1）；默认值为 0.9。其他运行保留 `[0,1]` 限制。大于 1 的 sweep 与已有 baseline 使用独立 run，通过共享 setup 配对比较。
- `policy_proposal_count` 影响 value beam 和 value BeFS，允许正整数或 `-1`。省略时 value beam 随 `beam_width`，value BeFS 默认 **4**；BeFS 可显式传入其他正整数，且不受 `beam_width` 影响。`-1` 使用全部未发现邻居，完全跳过 actor logits。critic 使用 SNN simplex 特征，不能因为跳过 actor 而改用另一种 value 特征。
- value BeFS 在提案前排除本起点已发现状态和重复目标；未通过 policy 筛选的目标不标为已发现。首次成功查询后永久标为已发现，再次遇到时不查询、不更新首次分数。提案并列和查询顺序与 RL beam 一致；不得把绝对 metric + value 分数改为累计路径 reward。
- Beam 提案先排除已发现目标与重复目标，概率并列按 canonical action 顺序；选择完提案后按 canonical 顺序查询。累计概率使用完整合法动作集合的分布，不能在搜索去重后重新归一化。父节点与下一层按 beam 分数降序、首次发现顺序排序；丢弃的已查询候选仍永久属于已发现状态。
- stochastic 每次移动查询 1 次。policy beam 每父节点最多 k 次，最后一个父节点最多超出 k−1 次；value beam 在有限 m 时每父节点最多 m 次、完整一层最多 k×m 次，最后一个父节点最多超出 m−1 次。`m=-1` 没有这个固定上界。两个 RL beam 的 `transition_count=0`。
- value beam / BeFS 从已记录的 child objective 计算自然对数 metric，不计算 transition reward，禁止为 metric 再查询 objective。网络推理不消耗 objective 预算；best 包含所有已查询候选。
- value BeFS 每个活动起点每轮只选一个父节点，跨起点合并推理和 objective 批次，不免费预展开整个 frontier。完整处理已开始的父节点后检查预算，有限 m 最多超出 m−1 次，`m=-1` 无固定上界；`transition_count=0`，frontier 为空时正常以 `frontier_exhausted` 结束。
- RL 每轮合并全部活动起点，beam 合并全部活动前沿；物理 GPU 批次按 `policy_max_graph_size`（默认 250000）分块。允许免费预取邻居与 policy 分数，但只有预算允许的父节点才形成逻辑 expansion；超出预算的父节点不得查询 objective。
- 批量 objective 复用 managed engine / worker，有序消费结果，通过同一个 rollout session 逐次计费与输出。先完全消费或关闭一个 geometry iterator，再提交下一个；批次统一释放活动状态，不能被一个起点提前释放。
- visited / discovered 与 RNG 属于每个起点，不属于 runtime cache；cache 关闭不清空搜索历史。随机抽样使用派生的起点 RNG，不使用全局 Torch RNG。

### cyopt genetic algorithm

- `cyopt_ga` 是 population family，经 `PopulationAlgorithm.run_population` 和共享 `_PopulationContext` 计费；直接调用已安装的上游 `cyopt.GA`，不另写选择、交叉、变异或代际去重实现。目前仅支持 `max_kcup`。
- 二维面 DNA 由上游 `cyopt.frst` 编解码。每个 polytope 的 codebook 按 canonical simplices 固定排序，最多 12 点的面完整枚举，更大的面以派生 seed 采样最多 1000 个；参数 `ga_face_max_points` / `ga_face_samples` 可配置。目标求值前加入所有共享起点的面限制，保证输入起点可表示。codebook、哈希、各起点 DNA、参数和准备耗时写入 `cyopt_encoding/`，准备不消耗 objective 查询。
- 每个 rollout 的种群第一项必须是该起点 DNA，复用其免费初始值；其余随机种群成员的有效 objective 查询计费。重复起点 DNA 返回原始完整 FRST，不能替换 q=0 的状态。多个互异完整 FRST 可有同一 DNA；空 DNA 正常以 `dna_space_singleton` 结束，保留全部配对起点。
- 大面的上游 `grow_frt` 可能返回重新编号的局部二维 polytope。适配器必须按点坐标映射回该 face 的 ambient 标签，再规范化、排序和合并起点限制；点集合不匹配时明确失败，不能把局部编号当作全局标签。
- 默认种群 50、tournament k=3、单点 crossover、mutation rate=0.1、mutation k=1、elitism=1。优化 fitness 是负的原始 volume；记录的 objective 仍是正原始 volume。极小 DNA 空间的有效 elitism 截到空间大小减一并写入 generation 日志。
- 上游 fitness cache 必须为零；每次有效 fitness 请求走共享 rollout 查询计数，即使重复 DNA 或命中 engine cache 也计一次。保留 elite 的已知 fitness 不重新查询。上游重建明确返回 None 的 DNA 是免费几何拒绝，内部 fitness 为 inf，但不得把 inf 写成 volume；拒绝 DNA 及原因写入当前 expansion。其他异常正常失败，不吞掉。
- DNA 解码别名可存入 engine 既有 bounded hot-state cache，以独立前缀、polytope index、codebook 哈希、DNA 为 key，共用原有容量和 pressure/close 生命周期，不另建缓存或增加额度。cache 关闭也必须关闭此别名缓存，起点 DNA 始终先返回该 rollout 的原始 FRST。缓存前后轨迹及计费须一致。
- GA 在每次 fitness 请求前检查预算，允许在初始化/代际中途精确停下，无 overshoot；邻居算法仍完整处理已开始的父节点。所有比较取 `query_index <= budget`。连续 `ga_max_stalled_generations=20` 代没有有效查询时以 `no_feasible_offspring` 正常结束，不能无限重试。
- GA 没有物理 transition。`expansion_count` 对应种群初始化及各代，事件 `kind=population`，初始化 generation index 为 0；query 的 DNA/generation 显式记录，action/source/depth 为 null。保留 v2 公共结果字段，读者按事件 kind 区分计数含义，不能跨 family 将 expansion 当作统一计算预算。
- config 记录 cyopt 版本、安装源、安装代码哈希及可用的本地源码 commit/status；安装只在获准后进行。GA 参数不影响 setup 校验值。测试覆盖真实上游代际操作、缓存一致性、重复查询、初始化预算中断、小空间、拒绝/异常及真实 FRST 编码往返。

### 通用终止、best 与复现

- 预算耗尽时结束搜索；Random/Greedy 在没有有效邻居时以 `no_neighbors` 结束，BeFS/Beam 在其前沿为空时以 `frontier_exhausted` 结束。FRST 是正常搜索状态，不因为它是 training 的目标状态就结束或 reset。
- `objective_budget=0` 时只计算并记录起点 objective，不展开邻居、不转移。
- `best_objective` 覆盖起点和所有已查询候选，包括没有选中的邻居和被 Beam 丢弃的候选。
- objective 并列时保留先前的 best。所有算法只保存 initial / best 的值与状态 key，不输出 `final_state_key` 或 `final_objective`。
- `SearchNode.depth` 是首次发现时的父节点深度加一，不保证是图上的最短路径距离。查询日志的 depth 属于候选，展开日志的 depth 属于父节点；query 的 `round_index` 与 `expansion_index` 都表示父节点展开序号，初始查询为 0。
- 通过 `derive_seed` 按全局 seed、算法名、polytope index 和 start index 派生搜索 seed。算法使用 context 的 RNG，不使用全局随机状态或 Python 的进程随机 hash。
- 相同 setup、配置和运行环境下，调整算法执行顺序或 cache 开关不应改变轨迹、objective 和逻辑查询数。

## 5. Cache 与资源生命周期

- `cache_states=True` 为默认值，`runtime_cache_gb=1.0` 是默认的合计保留缓存额度；它不是整个进程的内存上限。
- 复用 training 的有界 state、transition graph 和 objective 缓存。同一算法的起点之间可共享缓存，不同算法使用独立 managed runtime。
- `cache_states=False` 必须同时关闭 engine 与 worker 的运行时缓存，包括 objective 缓存，并在轮次结束后释放 transition graph。
- 只设置 `state_cache_mode="none"` 不足以实现上述关闭语义，必须同时设置相关缓存额度为零。
- 关闭缓存仍保留必要的输入、当前计算对象和 SQLite 状态身份历史；HF 下载缓存与持久化 setup 独立存在。
- 前沿状态描述及每个起点的去重集合是搜索必需状态，关闭缓存时仍须保留。前沿节点不持有 CYTools geometry 对象，不为保存日志而保留完整父链；引擎淘汰状态后仍能从状态描述正确展开。
- cache 命中、淘汰、重建只能改变实际计算开销，不能改变查询计费或搜索语义。
- 使用 managed runtime 的生命周期接口；正常结束和异常退出都必须释放 worker、活动状态与 history 连接。

## 6. 结果与验证要求

- 每次运行创建新的结果目录，不覆盖已有 run。记录完整规格、setup 引用和校验标识、代码版本及依赖版本。
- RL 额外记录解析后的 checkpoint 路径与 SHA256、模型与设备配置、提案数与 discount；`policy.resolved_policy_proposal_counts` 按算法记录实际提案数，旧单数字段保留 value beam 的解析规则以兼容旧结果。summary 记录逻辑 / 物理 inference batch、状态数、耗时和 CUDA 峰值分配。默认异步 CUDA 的细分时间是主机侧时间，精确计时使用 `profile_cuda_timing`。
- `policy.value_score_definitions` 对所有 value 搜索记录 `ln_objective_plus_discounted_value`。历史结果不改写：旧 `rl_value_*` 可能记录 `step_reward_plus_discounted_value`，须按历史评分元数据解释，不能仅按当前算法名称解释。不改变结果格式版本或 setup 校验标识。
- 分开输出 objective 查询、逻辑展开、实际转移和每条 rollout 的汇总；保留真实预算消耗、超预算量和终止原因。`expansions.jsonl` 记录父节点 key/objective/depth、候选数、展开前后查询数和成功/失败状态。
- 新结果的 `config.json` 和 `summary.json` 使用 `format_version=2`，所有算法移除 final 字段并增加 expansion 信息；setup 格式仍为 1。旧结果不重写，读取结果的后续工具须按版本处理字段差异。
- 进入结果写入阶段后的异常保留已完成记录，并写入 `failure.json`。只有全部算法和起点成功完成后才写成功的 `summary.json`。
- 共享下载缓存、长期复用的 setup 和常规正式结果分别放在 `data/cache/`、`data/setups/`、`results/runs/`；smoke 与 sweep 按下面的目录规则存放。生成产物应被 Git 忽略；代码、配置、规则与测试应可被版本管理。
- 改动核心逻辑后运行相关测试；改动共享 training 实现时，额外运行对应的 training 回归测试。
- 测试应验证实际合约：严格数量、起点配对、索引往返、离线加载、缓存命中计费、Greedy 整轮超预算、局部最优处继续移动、并列处理、零预算、无邻居、best 覆盖未选候选，以及异常清理。
- 搜索测试还须覆盖 BeFS 全局优先级与死路恢复、Beam 逐层顺序与永久剪枝、min/max 方向、循环和共享子节点去重、去重集合跨起点隔离、父节点预算边界及结果版本。
- cache 变更必须验证开/关时轨迹与查询数一致，并验证实际缓存行为；不能只比较最终分数。
- 普通测试不依赖网络。小型图可用于精确计数测试；真实 kcup smoke 使用 CYTools 和注册的 `max_kcup` objective，不能用假 reward 代替。

### 目录组织：smoke、logs 与 sweep

以下路径均相对于仓库根目录。这些规则适用于后续 evaluation 工作，运行时显式指定输出位置。

- **Smoke test 固定放在 `eval/smoke_tests/`。** 该目录由用户定期清理，只存可重新生成的验证产物：运行结果放 `results/<run_name>/`，运行日志放 `logs/`，pytest 临时文件放 `tmp/<run_name>/`，smoke 专用 setup / cache 也放在此目录内。包括 sweep 的预检查在内，所有 eval smoke 都使用这一固定根目录，不散落在正式结果或研究目录中。
- Smoke 可以引用 `eval/data/setups/` 中的长期共享起点；正式 evaluation 和 sweep 不得依赖 `eval/smoke_tests/` 中的文件。可复用代码、配置、checkpoint 和需要长期保留的研究结论也不得存入该清理目录。测试代码仍放在 `test/`。
- **持久化运行日志必须放在 `logs/` 子目录。** 常规运行的 stdout / stderr、launcher 输出和 `.log` 文件放在 `eval/logs/<run_name>.log`；smoke 放在 `eval/smoke_tests/logs/`；sweep 放在 `eval/sweep/<study_name>/logs/`。并行 benchmark 已有的 `<run_dir>/logs/` 继续用于各算法子进程日志。不要将 `.log` 文件直接放在 `eval/`、`results/runs/` 或研究目录顶层。
- `queries.jsonl`、`expansions.jsonl`、`transitions.jsonl`、`rollouts.jsonl` 属于结构化评估结果，继续遵守现有结果格式，保存在对应 run 内；不要仅为整理运行日志改变 writer / reader 的文件协议。
- **后续系列研究统一放在 `eval/sweep/<study_name>/`。** 包括 RL policy 与其他图搜索算法的组合、评分或提案策略的消融，以及 hyperparameter sweep / tuning（如 beam width、policy proposal count、value discount）。研究专用配置放 `configs/`，调度和分析脚本放 `scripts/`，正式试验结果放 `results/<run_name>/`，汇总图表放 `plots/`，运行日志放 `logs/`。
- 每项 sweep 用 `README.md` 记录研究问题、搜索空间、固定参数、setup / checkpoint、预算、seeds、运行命令和配置到结果的对应关系。继续复用本 evaluation 的算法接口、计费、去重、共享起点和结果读取逻辑；可复用算法实现仍放在 `eval/algorithm/` 等现有模块中。
- Run 名称使用 `_` 分词并带唯一标识，避免覆盖。CLI 目前不会根据实验目的自动区分 smoke / sweep，必须显式传入 `--output_dir`；pytest smoke 使用该固定目录下独立的 `--basetemp`，日志重定向前先创建相应的 `logs/` 目录。

### 可复用配置、并行运行与绘图

- `--config` 只读取 `EvaluationSpec` 字段；显式 CLI 参数覆盖配置，配置覆盖默认值。类型、choices 和合并后的必需参数必须验证。控制输出、setup 和并行资源的参数保留在 CLI。
- `--parallel_resources` 使用独立进程运行各算法，继续调用同一 `scripts/eval_cy.py` 与 `run_evaluation`。同一份 setup 只准备一次，所有算法使用相同校验标识和起点。
- 资源文件按算法给出 CPU 数、geometry worker 数、进程树内存预算及 RL GPU index。CPU affinity 不重叠，并覆盖导入期间创建的后台线程及后续 worker；RL GPU 不共用，BLAS/OpenMP 每进程一个线程。每个 RL 子进程加载一次模型，并校验所有 RL 结果的 checkpoint SHA256 一致。
- 每个子 run 保持结果格式 v2；父目录使用独立的 `benchmark.json`（格式 v1）记录资源、命令和状态，不复制或合并大体积原始日志。父清单只有所有子 run 成功后才标为 complete；失败保留日志并清理未完成进程。
- 绘图支持完整单 run 的 v1/v2 与完整并行 benchmark，当前目标为 `max_kcup`。未知版本、不完整运行、错误计数、未配对起点和失败查询必须报错。
- 主图以逻辑查询次数为横轴，预算内最佳值仅使用 `query_index <= objective_budget`；完整展开造成的超预算最佳值保存在汇总 CSV，不提前算入曲线。正常提前结束可延续最后 best，失败不能按此方式填充。
- Comparison 图统一展示原始 **best CY volume**，使用以 10 为底的对数坐标轴，刻度保留原始 volume 数值；不再展示相对起点的提升或 log gain。各 polytope 展示起点间中位数和 IQR；总览汇总全部配对起点的中位数和 IQR（每个 polytope 的起点数相同）。预算末端排名使用最佳 volume 中位数。
- 用户提供最优参考值时，可额外添加 optimality gap comparison。默认对每个起点、每个查询预算先计算 `gap = reference_log10_kcup - log10(best_kcup)`，再计算起点间均值与样本标准差（`ddof=1`）；不得先平均 volume 再取 log。Gap 图使用线性纵轴与 mean ± SD 阴影，不截断 gap 或阴影，不将 SD 标为置信区间。保存参考值、来源、公式、逐起点 gap 和统计曲线；参考值仅用于对应的 polytope，不能自动套用到其他对象。该统计相对于最优参考值，与起点归一化提升不同。
- 新派生图表的 `plot_config.json` 使用格式版本 2，记录绝对 volume 的统计定义；CSV 汇总绝对 volume，不再输出起点归一化的 gain 字段。原始 evaluation / setup 格式保持不变，历史结果不批量重写。
- 自动出图和独立重画共享 `plot_evaluation`；只写派生图表和 CSV，不修改原始结果。
- `--additional_run_dirs` 可合并已完成且算法不重叠的 runs；必须验证 setup、科学参数与 initial key/objective 配对，RL 来源还需一致的 checkpoint SHA256 和模型结构。要求显式独立 output_dir，plot_config 保存所有来源目录与 checkpoint 哈希，不复制/改写原始日志。支持 GA 两种正常提前终止原因。

从仓库根目录执行：

```bash
mkdir -p eval/smoke_tests/tmp
conda run -n sage python -m pytest \
  test/test_eval_*.py -q \
  --basetemp "eval/smoke_tests/tmp/regression_$(date +%Y%m%d_%H%M%S)"
```

需要验证 HF 下载到 rollout 的完整流程时执行在线 smoke：

```bash
eval_smoke_id=hf_smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p eval/smoke_tests/logs eval/smoke_tests/tmp
CY_EVAL_HF_SMOKE=1 conda run -n sage python -m pytest \
  test/test_eval_pipeline.py -k hugging_face -q \
  --basetemp "eval/smoke_tests/tmp/${eval_smoke_id}" \
  > "eval/smoke_tests/logs/${eval_smoke_id}.log" 2>&1
```

当前在线 smoke 规格为 1 个 polytope、`h11=12`、1 个起点、查询预算 5，运行 Random / Greedy / BeFS / Beam 并比较 cache 开/关的结果。
