# 问题2 联合运营语境

三园区（A/B/C）作为整体统一功率平衡，研究园区间风光互济与共享储能的经济性。本表固定论文写作与代码中使用的术语。

## Language

**联合运营 (Joint Operation)**：
把三个园区每小时的负荷与风光出力先相加，再统一做功率平衡的运行方式。
_Avoid_: 集中运行、统一调度（含义偏窄）

**独立运营 (Independent Operation)**：
每个园区各自做功率平衡，互不交换电力，最后仅汇总指标。问题1即此口径。
_Avoid_: 单独运行、分散运营

**园区间互济 (Inter-park Mutual Aid)**：
同一时段内，某园区富余、本将被弃掉的风光电，直接供给存在电力缺口的另一园区。在不计损耗时，每 1 kWh 互济同时减少 1 kWh 弃电与 1 kWh 电网购电。
_Avoid_: 电力交换、功率转移（未点明"消纳弃电"的本质）

**风光消纳 (RE Accommodation)**：
风光电被联合系统实际抽取使用，包括直接供给负荷（`P^pv,L`/`P^w,L`）与给共享储能充电（`P^pv,ch`/`P^w,ch`）两种去向。消纳的风光按 0.4/0.5 元/kWh 计成本。**储能充电属于消纳、不是弃电拯救**，见 `docs/adr/0002-storage-charge-costed-as-accommodation.md`。
_Avoid_: 利用（未区分是否计成本）、消纳电量（口语化）

**弃风弃光 (Curtailment)**：
联合负荷与储能充电都未能消纳、被迫丢弃的风电与光伏电量，分弃风量与弃光量。弃电不计成本，也不得上网出售（售价=0）。与 [[风光消纳]] 互斥：同一 kWh 风光要么被消纳（计成本）要么被弃（免费）。
_Avoid_: 弃电（单独用易与"弃负荷"混淆）、浪费电量

**风光消纳率 (RE Accommodation Rate)**：
`1 − 弃风弃光总量 / 风光发电总量`，衡量风光被负荷消纳的比例。
_Avoid_: 新能源利用率、消纳比例

**单位供电成本 (Unit Supply Cost)**：
`日供电成本 / 联合负荷电量`。分母必须是实际负荷电量，不能用风光发电量或电网购电量。
_Avoid_: 度电成本、平均购电成本（口径不一）

**联合负荷 / 联合光伏 / 联合风电 (Joint Load / PV / Wind)**：
三园区逐时相加得到的 `L^J / G^{pv,J} / G^{w,J}`，上标 J 表示联合园区。园区B无光伏、园区A无风电，对应项为0。

## Pareto 主模型（ε-约束法）

**年实际成本 (Annual Actual Cost)**：
$C^{\mathrm{ann}}=365\,C^{\mathrm{day}}+C^{\mathrm{inv,ann}}$，其中 $C^{\mathrm{day}}$ 为**不含弃电惩罚**的日实际运行成本。问题2（2）主模型与问题2（3）三层比较均用此口径。与 λ 法的"年综合成本"（含 λ·弃电）区分。
_Avoid_: 年综合成本（λ 法专用，含弃电惩罚）、年总成本

**ε-约束法 (Epsilon-Constraint Method)**：
多目标优化转单目标的方法：目标保持 $\min C^{\mathrm{ann}}$，把第二目标（风光消纳率）转为硬约束 $R_{\mathrm{re}}\geq R_0$。逐档设定 $R_0$ 求解，得到 Pareto 前沿。实现为 `solve_park_optimization` 的 `min_accom_rate` 参数（默认 None 时不加约束，向后兼容）。
_Avoid_: 罚函数法（那是 λ 法）、加权法

**最低消纳率 R0 (Minimum Accommodation Rate)**：
ε-约束的右端项，工程上可解释的消纳率目标。取 $R_0\in\{95\%,97\%,98\%,99\%,100\%\}$ 五档。$R_0=\,$None 时不加约束，等价于纯经济最优（0/0 基准）。
_Avoid_: 消纳率目标值

**Pareto 前沿 (Pareto Front)**：
年实际成本（越小越好）与风光消纳率（越大越好）的非劣解集。每个 $R_0$ 求连续 MILP 与工程整数 MILP 各一点，构成前沿。单调性：$R_0$ 越严，成本不降、弃电不增。
_Avoid_: 帕累托前沿、非劣解集（口语化）

**理想点距离 (Ideal Point Distance)**：
前沿折中选型准则。候选集 = {纯经济基准 0/0} ∪ {各 $R_0$ 连续最优}；等权归一化 $\widehat C=(C-C_{\min})/(C_{\max}-C_{\min})$、$\widehat G=(R_{\max}-R)/(R_{\max}-R_{\min})$，距离 $d=\sqrt{\widehat C^2+\widehat G^2}$。取 $d$ 最小者为折中目标 $R_0^\ast$，再取该 $R_0^\ast$ 的工程整数方案为推荐配置。$R_0^\ast=97\%$，推荐 175 kW/890 kWh。
_Avoid_: 加权理想点（本题为等权）

## 弃电惩罚扩展（View C，补充敏感性分析）

> **定位**：λ 法是问题2（2）的**补充敏感性分析**，不决定主结论。主结论见 [[Pareto 主模型]] 的 175 kW/890 kWh（ADR 0004）。λ 法用于考察弃电等效价值变化对配置的影响，代码 `q2_storage_penalty.py` 与输出独立、不与主模型耦合。

**弃电惩罚 (Curtailment Penalty)**：
把原本免费的弃风弃光按 λ 元/kWh 计入目标函数的政策性成本项，迫使优化器提高消纳。λ=0 退化为 [[弃风弃光]]不计成本的原口径（纯经济基准，0/0）。λ∈{0,0.3,0.6}，见 `docs/adr/0003-curtailment-penalty-view-c.md`。
_Avoid_: 弃电成本（未点明是惩罚性计价）、弃电罚款

**惩罚目标函数 (Penalized Objective)**：
$C^{\mathrm{day}}(\lambda)$ = View A 日运行成本 + $\lambda\sum_t(P_t^{\mathrm{curt,pv}}+P_t^{\mathrm{curt,w}})\Delta t$。年综合 = $365\,C^{\mathrm{day}}(\lambda)+C^{\mathrm{inv,ann}}$。`daily_cost` 在 λ>0 时含惩罚项，`daily_cost_pure` 为不含惩罚的纯运行成本。
_Avoid_: 带罚项目标（口语化）

**λ 敏感性 (Lambda Sensitivity)**：
固定其它参数扫惩罚因子 λ，观察最优储能功率/容量、年综合成本、弃电量、消纳率随 λ 的变化；定位使最优配置从 0/0 翻转为正的临界 λ。
_Avoid_: 惩罚扫描
