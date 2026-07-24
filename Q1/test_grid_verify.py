# -*- coding: utf-8 -*-
"""
测试性检验：工程取整粒度一致性验证

背景：
    队友反馈主程序工程取整粒度 (5kW/10kWh) 与网格搜索粒度 (10kW/20kWh) 不一致，
    按 5/10 在连续最优附近补测发现：
      - A 仍不配置储能
      - B 附近最优 35kW/120kWh，年综合成本约 1846084.92 元，优于当前 30/110、30/100
      - C 仍 60kW/70kWh

目的：
    在统一的 5kW/10kWh 粒度下，对 A/B/C 三个园区对比三条口径：
      1) 连续理论最优 (run_optimized_storage)
      2) MILP 工程整数最优 (run_engineering_storage, 5/10 粒度)
      3) 局部网格搜索最优 (grid_search_capacity, 5/10 粒度, center=连续最优)
    验证队友结论，并定位 MILP 整数规划是否漏解。

口径一致性：
    年综合成本 = 365 * 日运行成本 + (C_P_ESS*P + C_E_ESS*E) / Y
    与 grid_search.py / milp_model.py 完全相同。
"""
import sys
import os
import time

# Windows 控制台中文输出
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.append(os.path.dirname(__file__))
import numpy as np
from data_loader import (
    load_load_data, load_solar_wind_data,
    C_P_ESS, C_E_ESS, Y,
)
from milp_model import (
    run_no_storage, run_optimized_storage, run_engineering_storage, run_fixed_capacity,
)
from grid_search import grid_search_capacity

# 统一粒度（与 q1_main.py 当前定义一致）
P_STEP = 5    # kW
E_STEP = 10   # kWh
P_MAX = 200   # kW
E_MAX = 600   # kWh

# 局部网格搜索半宽（以连续最优为中心）
P_HALF = 20   # kW
E_HALF = 40   # kWh


def annual_cost(p, e, daily_cost):
    """年综合成本，口径同 grid_search / milp_model。"""
    return 365.0 * daily_cost + (C_P_ESS * p + C_E_ESS * e) / Y


def eval_point(L, Pv, W, p, e):
    """求解指定 (P,E) 下的最优运行，返回年综合成本；失败返回 None。"""
    r = run_fixed_capacity(L, Pv, W, P_ess=p, E_ess=e)
    if not r['success']:
        return None
    return annual_cost(p, e, r['daily_cost'])


def align_to_step(x, step):
    return int(round(x / step) * step)


def main():
    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()

    lines = []

    def log(msg=''):
        print(msg)
        lines.append(msg)

    summary = []

    for park in ['A', 'B', 'C']:
        L = load_data[park]
        Pv = G_pv_data[park]
        W = G_w_data[park]

        log(f'\n{"=" * 72}')
        log(f'园区 {park}')
        log(f'{"=" * 72}')

        # 1) 无储能
        r0 = run_no_storage(L, Pv, W)
        ac0 = annual_cost(0, 0, r0['daily_cost'])
        log(f'[无储能]             P=     0 kW, E=      0 kWh, 年成本= {ac0:.2f}')

        # 2) 连续理论最优
        t0 = time.time()
        r2 = run_optimized_storage(L, Pv, W)
        t1 = time.time()
        log(f'[连续最优]           P={r2["P_ess"]:7.2f} kW, E={r2["E_ess"]:8.2f} kWh, '
            f'年成本= {r2["annual_cost"]:.2f}   ({t1 - t0:.1f}s)')

        # 3) MILP 工程整数最优 (5/10 粒度)
        t0 = time.time()
        r3 = run_engineering_storage(L, Pv, W,
                                     P_step=P_STEP, E_step=E_STEP,
                                     P_max=P_MAX, E_max=E_MAX)
        t1 = time.time()
        log(f'[MILP工程整数 5/10]  P={r3["P_ess"]:7.0f} kW, E={r3["E_ess"]:8.0f} kWh, '
            f'年成本= {r3["annual_cost"]:.2f}   ({t1 - t0:.1f}s)')

        # 4) 局部网格搜索 (5/10 粒度, center=连续最优对齐)
        p_c = align_to_step(r2['P_ess'], P_STEP)
        e_c = align_to_step(r2['E_ess'], E_STEP)
        p_lo, p_hi = max(0, p_c - P_HALF), min(P_MAX, p_c + P_HALF)
        e_lo, e_hi = max(0, e_c - E_HALF), min(E_MAX, e_c + E_HALF)
        t0 = time.time()
        gs = grid_search_capacity(
            L, Pv, W,
            P_range=(p_lo, p_hi, P_STEP),
            E_range=(e_lo, e_hi, E_STEP),
            verbose=False,
        )
        t1 = time.time()
        gb = gs['best']
        log(f'[网格搜索 5/10]      P={gb["P_ess"]:7.0f} kW, E={gb["E_ess"]:8.0f} kWh, '
            f'年成本= {gb["annual_cost"]:.2f}   (评估 {len(gs["grid"])} 点, {t1 - t0:.1f}s)')
        log(f'   网格中心(连续最优对齐)= {p_c}/{e_c}, '
            f'范围 P[{p_lo},{p_hi}] E[{e_lo},{e_hi}]')

        # 5) 一致性判定
        match = (abs(r3['P_ess'] - gb['P_ess']) < 1e-6
                 and abs(r3['E_ess'] - gb['E_ess']) < 1e-6)
        if match:
            log('[一致性] MILP工程最优 == 网格搜索最优  ✓')
        else:
            gap = r3['annual_cost'] - gb['annual_cost']
            log(f'[一致性] MILP工程最优 != 网格搜索最优  ✗  差额= {gap:.2f} 元/年')
            log(f'   -> MILP 漏掉更优解 {gb["P_ess"]:.0f}/{gb["E_ess"]:.0f}，'
                f'需排查 timeLimit / 整数变量建模 / 求解器收敛')

        summary.append({
            'park': park,
            'no_storage': ac0,
            'cont': (r2['P_ess'], r2['E_ess'], r2['annual_cost']),
            'milp': (r3['P_ess'], r3['E_ess'], r3['annual_cost']),
            'grid': (gb['P_ess'], gb['E_ess'], gb['annual_cost']),
            'match': match,
        })

        # 6) B 园区专项：逐一验证队友提到的候选点
        if park == 'B':
            log('\n[B 专项] 队友候选点逐一验证 (5/10 粒度下均合法):')
            candidates = [(30, 100), (30, 110), (35, 120), (35, 110),
                          (40, 120), (30, 120), (35, 130)]
            for (p, e) in candidates:
                ac = eval_point(L, Pv, W, p, e)
                if ac is None:
                    log(f'   {p:3d}kW/{e:3d}kWh -> 求解失败')
                    continue
                tag = ''
                if abs(ac - 1846084.92) < 0.5:
                    tag = '   <-- 命中队友报的 1846084.92'
                log(f'   {p:3d}kW/{e:3d}kWh -> 年成本 {ac:.2f}{tag}')

    # ---- 总览 ----
    log(f'\n{"=" * 72}')
    log('总览')
    log(f'{"=" * 72}')
    log(f'{"园区":<4}{"无储能年成本":>16}  {"连续最优 P/E=年成本":>26}  '
        f'{"MILP工程 P/E=年成本":>26}  {"网格5/10 P/E=年成本":>26}  一致')
    for s in summary:
        cP, cE, cC = s['cont']
        mP, mE, mC = s['milp']
        gP, gE, gC = s['grid']
        log(f'{s["park"]:<4}{s["no_storage"]:>16.2f}  '
            f'{cP:>5.1f}/{cE:>6.1f}={cC:>11.2f}  '
            f'{mP:>5.0f}/{mE:>6.0f}={mC:>11.2f}  '
            f'{gP:>5.0f}/{gE:>6.0f}={gC:>11.2f}  '
            f'{"是" if s["match"] else "否 ✗"}')

    # ---- 结论 ----
    log(f'\n{"=" * 72}')
    log('结论')
    log(f'{"=" * 72}')
    b = next(s for s in summary if s['park'] == 'B')
    b_grid_cost = b['grid'][2]
    b_milp_cost = b['milp'][2]
    if abs(b_grid_cost - 1846084.92) < 1.0:
        log(f'• B 园区 5/10 网格搜索最优年成本 = {b_grid_cost:.2f}，'
            f'与队友报的 1846084.92 一致 ✓')
    else:
        log(f'• B 园区 5/10 网格搜索最优年成本 = {b_grid_cost:.2f}，'
            f'与队友报的 1846084.92 不符 ✗')
    if b['match']:
        log(f'• B 园区 MILP工程整数最优 == 网格搜索最优，'
            f'当前代码 5/10 粒度下口径一致，无漏解 ✓')
    else:
        log(f'• B 园区 MILP工程整数最优 ({b_milp_cost:.2f}) != 网格搜索最优 ({b_grid_cost:.2f})，'
            f'MILP 漏解 ✗')
        log(f'  建议：增大 solve_park_optimization 的 timeLimit，或改用网格搜索结果作为工程最优。')

    # 写报告
    report_path = os.path.join(os.path.dirname(__file__), 'test_grid_verify_result.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# 工程取整粒度一致性测试报告\n\n')
        f.write(f'**统一粒度**: P_STEP={P_STEP} kW, E_STEP={E_STEP} kWh, '
                f'P_MAX={P_MAX}, E_MAX={E_MAX}\n\n')
        f.write('**对比口径**: 连续最优 / MILP工程整数最优 / 局部网格搜索最优\n\n')
        f.write('```\n')
        f.write('\n'.join(lines))
        f.write('\n```\n')
    print(f'\n报告已写入: {report_path}')


if __name__ == '__main__':
    main()
