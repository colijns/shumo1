# 问题2 运行说明

三园区（A/B/C）联合运营的三个小问：2（1）联合无储能、2（2）联合共享储能（Pareto 主模型）、2（3）联合 vs 独立三层对比。

## 2（1）联合运营、无储能

先把三个园区每小时的负荷与风光出力相加，再统一做功率平衡，研究园区间风光互济的收益。详见 `docs/问题2（1）报告.md`。

```bash
conda activate math
python Q2/q2_main.py                       # 求解 + 输出表格与图片
python -m pytest Q2/test_q2_verify.py -v   # 验收测试
```

结果保存在 `Q2/output/`。联合无储能年供电成本 5,514,296.08 元，风光消纳率 92.437%。

## 2（2）联合共享储能 -- Pareto ε-约束法（主模型）

纯经济最优为 0/0（储能投资收益不足），故采用 ε-约束法构造年实际成本-风光消纳率 Pareto 前沿，等权理想点距离选折中点。详见 `docs/问题2（2）报告.md`。

```bash
python Q2/q2_pareto.py                       # Pareto 前沿 + 理想点选型 + 9 项验证
python -m pytest Q2/test_q2_pareto.py -v     # 24 项验收测试
```

结果保存在 `Q2/output_pareto/`。推荐配置 **175 kW / 890 kWh**（$R_0^\ast=97\%$），实际消纳率 97.019%，年实际成本 5,575,300.76 元。

### λ 敏感性分析（补充）

弃电惩罚因子 λ 的敏感性分析作为补充见 `docs/问题2（2）lambda敏感性.md`，不决定主结论。代码与输出独立：

```bash
python Q2/q2_storage_penalty.py              # λ∈{0,0.3,0.6} 敏感性
python -m pytest Q2/test_q2_2_penalty_verify.py -v
```

结果在 `Q2/output_penalty_revised/`。

## 2（3）联合 vs 独立三层对比

三层比较（最终方案 / 纯经济 / 同 97% 标准）量化联合运营年收益。详见 `docs/问题2（3）报告.md`。

```bash
python Q2/q2_3_compare.py                       # 三层对比 + doc 预测值校验
python -m pytest Q2/test_q2_3_compare.py -v     # 15 项验收测试
```

结果保存在 `Q2/output_q2_3/`。三层年收益：最终方案 30.13 万元（5.13%）、纯经济 36.23 万元（6.17%）、同 97% 标准 44.01 万元（7.32%）。

## 实现要点

- **复用 Q1 统一 MILP 模型**（见 `Q2/docs/adr/0001-reuse-q1-unified-milp.md`）：跨目录 import `Q1/data_loader` 与 `Q1/milp_model`。储能功率/容量设为 0 时自动退化为联合无储能模型。
- **联合 profile**：$L^J=L_A+L_B+L_C$、$G^{\mathrm{pv},J}=\Sigma G^{\mathrm{pv}}$、$G^{\mathrm{w},J}=\Sigma G^{\mathrm{w}}$（园区 B 无光伏、园区 A 无风电）。
- **ε-约束扩展**（ADR 0004）：`solve_park_optimization` 加可选 `min_accom_rate=None`，默认 None 时不加约束（向后兼容），设定时施加 $R_{\mathrm{re}}\geq R_0$。
- **充电计价口径**（ADR 0002）：储能充电按 0.4/0.5 元/kWh 计（与消纳同价），弃电不计成本。该口径在 Pareto 主模型与 λ 补充中一致。

## 模型口径

- 典型日重复 365 天估算全年成本；
- 光伏/风电/电网购电成本分别为 0.4 / 0.5 / 1.0 元/kWh；
- 基准模型只对实际消纳的风光电计成本，弃风弃光不计成本；
- 联合系统整体不向外部电网售电；
- 不考虑园区间输电容量、传输损耗与内部结算费用。

## 相关文档

- `Q2/CONTEXT.md` -- 术语表（含 Pareto/ε-约束/理想点距离/λ 补充）
- `Q2/docs/adr/0001-reuse-q1-unified-milp.md` -- 复用 Q1 统一 MILP
- `Q2/docs/adr/0002-storage-charge-costed-as-accommodation.md` -- 充电计价口径（主结论已被 0004 取代，口径仍有效）
- `Q2/docs/adr/0003-curtailment-penalty-view-c.md` -- λ 法（主结论已被 0004 取代，降级为补充）
- `Q2/docs/adr/0004-pareto-as-main-model.md` -- Pareto ε-约束法作为主模型（当前主结论）
