# -*- coding: utf-8 -*-
"""
模块：integer_programming
功能：整数规划 / 0-1 规划求解，基于 PuLP（调用 CBC 等求解器）
适用题型：通用最高频（0-1 背包、选址、指派、生产计划取整等）
依赖：pulp（可选依赖，未安装时 demo 优雅跳过，退出码 0）
用法：直接运行 `python integer_programming.py` 查看 demo；或 import 后参考 demo 中的建模套路
"""

try:
    import pulp
    _HAS_PULP = True
except ImportError:
    _HAS_PULP = False


def solve_knapsack(values, weights, capacity):
    """
    求解经典 0-1 背包问题：max sum(values[i]*x[i])，满足 sum(weights[i]*x[i]) <= capacity。

    参数：
        values   : 各物品价值，shape (n,)
        weights  : 各物品重量，shape (n,)
        capacity : 背包容量（标量）

    返回：
        dict，键为：
            'chosen'  : 被选中的物品下标列表
            'x'       : 0-1 决策向量（numpy 数组）
            'fun'     : 最大总价值
            'status'  : 求解状态字符串
    """
    n = len(values)
    # 建立最大化问题
    prob = pulp.LpProblem("背包问题", pulp.LpMaximize)
    # 0-1 决策变量
    x = [pulp.LpVariable(f"x{i}", cat=pulp.LpBinary) for i in range(n)]
    # 目标函数：总价值最大
    prob += pulp.lpSum(values[i] * x[i] for i in range(n)), "总价值"
    # 容量约束
    prob += pulp.lpSum(weights[i] * x[i] for i in range(n)) <= capacity, "容量约束"
    # 静默求解（不打印求解器日志）
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    x_val = [int(round(pulp.value(x[i]))) for i in range(n)]
    return {
        'chosen': [i for i in range(n) if x_val[i] == 1],
        'x': x_val,
        'fun': pulp.value(prob.objective),
        'status': pulp.LpStatus[prob.status],
    }


def solve_integer_production(profits, resource_use, resource_avail):
    """
    求解整数生产计划问题：max 利润，产量为非负整数，资源消耗不超过可用量。

    参数：
        profits        : 各产品单位利润，shape (n,)
        resource_use   : 资源消耗矩阵，shape (m, n)，第 j 行第 i 列为产品 i 消耗资源 j 的量
        resource_avail : 各资源可用量，shape (m,)

    返回：
        dict，键为：
            'x'      : 最优整数产量向量
            'fun'    : 最大利润
            'status' : 求解状态字符串
    """
    n = len(profits)
    m = len(resource_avail)
    prob = pulp.LpProblem("整数生产计划", pulp.LpMaximize)
    # 非负整数决策变量
    x = [pulp.LpVariable(f"x{i}", lowBound=0, cat=pulp.LpInteger) for i in range(n)]
    prob += pulp.lpSum(profits[i] * x[i] for i in range(n)), "总利润"
    for j in range(m):
        prob += (pulp.lpSum(resource_use[j][i] * x[i] for i in range(n))
                 <= resource_avail[j]), f"资源{j + 1}约束"
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    return {
        'x': [int(round(pulp.value(x[i]))) for i in range(n)],
        'fun': pulp.value(prob.objective),
        'status': pulp.LpStatus[prob.status],
    }


if __name__ == "__main__":
    if not _HAS_PULP:
        # 可选依赖缺失：按 SPEC 约定打印提示并正常退出（退出码 0）
        print("未安装 pulp，可 pip install pulp，本 demo 跳过")
    else:
        print("=" * 60)
        print("整数规划 demo1：0-1 背包问题")
        print("=" * 60)
        # 10 件物品的价值与重量，背包容量 30
        values = [10, 8, 5, 4, 7, 12, 9, 6, 3, 11]
        weights = [6, 5, 3, 2, 4, 7, 5, 4, 2, 6]
        capacity = 30
        res1 = solve_knapsack(values, weights, capacity)
        print(f"求解状态：{res1['status']}")
        print(f"背包容量：{capacity}，物品数：{len(values)}")
        print(f"选中物品下标（从0计）：{res1['chosen']}")
        print(f"决策向量 x：{res1['x']}")
        print(f"最大总价值：{res1['fun']:.4f}")
        total_w = sum(weights[i] for i in res1['chosen'])
        print(f"总重量校验：{total_w:.4f} <= {capacity}")

        print("-" * 60)
        print("整数规划 demo2：整数生产计划")
        print("-" * 60)
        # 3 种产品，2 种资源；产品产量必须取整数
        profits = [5, 4, 6]
        resource_use = [[2, 3, 4],      # 资源1：工时
                        [3, 2, 5]]      # 资源2：原材料
        resource_avail = [20, 18]
        res2 = solve_integer_production(profits, resource_use, resource_avail)
        print(f"求解状态：{res2['status']}")
        print(f"最优整数产量：产品A {res2['x'][0]} 件，产品B {res2['x'][1]} 件，"
              f"产品C {res2['x'][2]} 件")
        print(f"最大利润：{res2['fun']:.4f} 元")
        # 资源消耗校验
        for j in range(len(resource_avail)):
            used = sum(resource_use[j][i] * res2['x'][i] for i in range(len(profits)))
            print(f"资源{j + 1}消耗：{used:.4f}/{resource_avail[j]}")
