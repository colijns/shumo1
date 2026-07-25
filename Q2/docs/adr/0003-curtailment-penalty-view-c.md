# 弃电惩罚目标函数扩展（View C）-- 作为 Q2(2) 平行结果集隔离实现

## 背景

ADR 0002 在 Q2(2) 共享储能定价口径抉择中考虑了三种 View，最终采纳 **View A**（储能充电按 0.4/0.5 计、弃电不计成本），得出 $0/0$ 最优、年综合 $5514296.083$ 元。**View C**（加弃电惩罚项 $\lambda$ 元/kWh）当时被列为备选但未采纳，理由是"改变全题目标函数，须同步回 Q1/Q2(1) 重算"。

现需求：在 **不回改 Q1/Q2(1)** 的前提下，把 View C 作为 Q2(2) 的一套**平行敏感性结果**实现，$\lambda\in\{0,0.3,0.6\}$ 元/kWh，代码与输出与无惩罚版完全隔离。目的：考察"对弃电计价"这一政策性约束如何改变最优储能配置——$\lambda=0$ 锚定 View A 结论，$\lambda>0$ 展示惩罚强度把最优解从 $0/0$ 推向正配置的临界点。

## 决策

1. **目标函数扩展**：在 View A 基础上加弃电惩罚项

   $$
   C^{\mathrm{day}}(\lambda)=\sum_t\big[
   0.4(P_t^{\mathrm{pv,L}}+P_t^{\mathrm{pv,ch}})
   +0.5(P_t^{\mathrm{w,L}}+P_t^{\mathrm{w,ch}})
   +P_t^{\mathrm{grid}}
   +\lambda(P_t^{\mathrm{curt,pv}}+P_t^{\mathrm{curt,w}})
   \big]\Delta t.
   $$

   年综合成本 $C^{\mathrm{ann}}(\lambda)=365\,C^{\mathrm{day}}(\lambda)+C^{\mathrm{inv,ann}}$。$\lambda=0$ 退化为 View A。

2. **同口径比较**：方案对比表中"联合无储能"基准行也按 $\lambda$ 计弃电惩罚。无储能时日弃电 $1237.18$ kWh 固定，基准年成本 $=5514296.083+365\cdot\lambda\cdot1237.18$（$\lambda=0.3$ 约 $565$ 万、$\lambda=0.6$ 约 $578$ 万）。储能方案与基准在同一 $\lambda$ 下比较，避免"两个世界"的不自洽。

3. **隔离实现**：新增 `Q2/q2_storage_penalty.py` + `Q2/test_q2_2_penalty_verify.py`，输出到 `Q2/output_penalty/`（与 `Q2/output/` 分离）。不改动 `q2_storage.py` 及其输出。

4. **共享 MILP 向后兼容扩展**：给 `Q1/milp_model.py::solve_park_optimization` 与 `Q1/grid_search.py::grid_search_capacity` 新增可选参数 `curt_penalty=0.0`（默认 $0$ 时目标函数不变，Q1/Q2(1)/Q2(2) 无惩罚版行为完全一致）。新文件通过该参数传入 $\lambda$。遵循 ADR 0001 的统一 MILP 复用原则，不复制代码。

5. **$\lambda=0$ 一致性校验**：$\lambda=0$ 必须复现无惩罚版 $0/0=5514296.083$（测试断言），证明新代码路径与 View A 同构。

## Considered Options

共享 MILP 改动方式：

- **A：共享 MILP 加向后兼容 `curt_penalty` 参数 —— 采用**。改动最小（两处函数各加一参数 + 目标函数加一项），无重复代码，Q1/Q2.1 零影响。代价：共享文件被修改（但默认行为不变，既有测试守护）。
- **B：在 milp_model.py 新增并行函数 `solve_park_optimization_with_penalty`**。不动现有签名，隔离性好；但 MILP 构建逻辑基本复制一份，违背 ADR 0001 复用原则，后续维护两份。
- **C：把 MILP 构建整体复制到新文件**。完全隔离；但全仓库复用架构被打破，最差。

基准口径：

- **基准也计惩罚（同口径）—— 采用**。唯一自洽选项：所有方案在同一 $\lambda$ 下比较。
- 基准不计惩罚：储能受罚而基准不受，对比无意义。

敏感性范围：

- **保留 c/s 敏感性 + 新增 $\lambda$ 敏感性 —— 采用**。每个 $\lambda$ 各跑充电成本 $c$ 扫描与投资价 $s$ 扫描，外加"最优容量/成本/弃电 vs $\lambda$"主线总表。
- 仅 $\lambda$ 敏感性：聚焦新维度但丢失 $c/s$ 稳健性证据。
- c/s 仅在参考 $\lambda=0.3$ 跑：折中，但不同 $\lambda$ 下 $c/s$ 边界移动本身有信息量，不省。

网格范围：

- **每个 $\lambda$ 各跑完整 $41\times61=2501$ 点网格 —— 采用**。3 张热力图完整展示惩罚强度增加时 landscape 从"原点最优"翻转为"储能最优"的演变（约 4-5 分钟）。
- 仅参考 $\lambda$ 出一张：省时但丢失演变信息。
- 网格收稀：粒度变粗，临界点定位不准。

## Consequences

- 共享层 `solve_park_optimization` / `grid_search_capacity` 获得 `curt_penalty` 参数；`daily_cost` 在 `curt_penalty>0` 时包含惩罚项（保持 `年综合 = 365×日运行 + 年均投资` 恒等），另导出 `daily_curt_penalty` 与 `daily_cost_pure`（不含惩罚）供报表分解展示。
- 新文件复刻无惩罚版的求解栈（连续 MILP / 工程整数 / 2D 网格 / $c$ 敏感性 / $s$ 敏感性），每个 $\lambda$ 各跑一遍 2501 点网格出热力图；新增 `sensitivity_lambda` 给出"最优容量·成本·弃电 vs $\lambda$"主线总表与曲线。
- 验收分两层：$\lambda=0$ 跑无惩罚版全部 15 项检查（含 $0/0$ 全局最优、$c^\ast\in(0.35,0.40]$、$s=1\to0$）；$\lambda>0$ 改用惩罚适用检查（最优年综合 $\le$ 基准、最优弃电 $\le$ 基准弃电、消纳率 $\ge$ 基准、网格最优 $=$ 连续最优、低 $c$ 下容量为正）。
- **预期发现**：$\lambda$ 足够大时最优配置从 $0/0$ 翻转为正（惩罚使弃电消纳的边际收益增加 $\lambda$/kWh），具体临界 $\lambda$ 由 `sensitivity_lambda` 定位。此结论不覆盖 ADR 0002 的 $0/0$——两者在不同目标函数下各自成立。
- Q2(2) 主结论（View A，$0/0$ 最优）不变；本 ADR 产出的是政策敏感性伴随分析。
- 不回改 Q1/Q2(1)：惩罚仅在 Q2(2) 容量优化范围内生效，$\lambda=0$ 与既有结果锚定，故 ADR 0002 当时"须同步回 Q1/Q2(1)"的顾虑在本隔离方案下不成立。
