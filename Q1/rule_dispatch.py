# -*- coding: utf-8 -*-
"""
规则调度对照模型
功能：基于启发式规则的储能调度策略，用于交叉验证 MILP 结果
规则：
  1. 风光优先直接供负荷
  2. 风光剩余时给储能充电（不超过储能功率上限和SOC上限）
  3. 风光不足时优先由储能放电（不超过SOC下限）
  4. 储能仍不足时从电网购电
"""

import numpy as np

from data_loader import ETA_C, ETA_D, S_MIN, S_MAX, S_0, DT


def rule_based_dispatch(load, G_pv, G_w, P_ess, E_ess):
    """
    规则调度：给定储能容量，模拟逐时运行。

    参数：
        load  : array[24], 负荷 kW
        G_pv  : array[24], 光伏出力 kW
        G_w   : array[24], 风电出力 kW
        P_ess : 储能额定功率 kW
        E_ess : 储能额定容量 kWh

    返回：
        dict, 与 MILP 模型同样格式的结果
    """
    T = 24

    # 初始化
    E = np.zeros(T + 1)
    E[0] = S_0 * E_ess

    P_load_pv = np.zeros(T)
    P_load_w = np.zeros(T)
    P_ch_pv = np.zeros(T)
    P_ch_w = np.zeros(T)
    P_curt_pv = np.zeros(T)
    P_curt_w = np.zeros(T)
    P_ch = np.zeros(T)
    P_dis = np.zeros(T)
    P_grid = np.zeros(T)
    z = np.zeros(T)  # 1=充电, 0=放电

    for t in range(T):
        remaining_load = load[t]

        # === 步骤1: 风光直供负荷 ===
        # 优先用便宜的 PV (0.4元), 再用 W (0.5元)
        pv_avail = G_pv[t]
        w_avail = G_w[t]

        pv_to_load = min(pv_avail, remaining_load)
        remaining_load -= pv_to_load

        w_to_load = min(w_avail, remaining_load)
        remaining_load -= w_to_load

        pv_after_load = pv_avail - pv_to_load
        w_after_load = w_avail - w_to_load

        P_load_pv[t] = pv_to_load
        P_load_w[t] = w_to_load

        # === 步骤2: 剩余风光给储能充电 ===
        if P_ess > 0 and E_ess > 0:
            surplus = pv_after_load + w_after_load
            # 储能还能充多少？
            soc_current = E[t] / E_ess
            soc_capacity = S_MAX - soc_current
            energy_capacity = soc_capacity * E_ess / ETA_C  # 考虑充电效率

            max_ch_energy = min(P_ess * DT, energy_capacity)  # kWh
            max_ch_power = max_ch_energy / DT  # kW

            if surplus > 0 and max_ch_power > 0:
                ch_power = min(surplus, max_ch_power)
                if ch_power > 0:
                    z[t] = 1

                # 风光按剩余比例分配充电功率
                total_surplus = pv_after_load + w_after_load
                if total_surplus > 0:
                    pv_ch = ch_power * pv_after_load / total_surplus
                    w_ch = ch_power * w_after_load / total_surplus
                else:
                    pv_ch = 0
                    w_ch = 0

                P_ch_pv[t] = pv_ch
                P_ch_w[t] = w_ch
                P_ch[t] = ch_power
                # 更新剩余
                pv_after_load -= pv_ch
                w_after_load -= w_ch

        # === 弃电 ===
        P_curt_pv[t] = max(0, pv_after_load)
        P_curt_w[t] = max(0, w_after_load)

        # === 步骤3: 负荷不足时，储能放电 ===
        if remaining_load > 0 and P_ess > 0 and E_ess > 0:
            soc_current = E[t] / E_ess
            # 最多放出到 S_MIN
            dischargeable_energy = (soc_current - S_MIN) * E_ess * ETA_D  # kWh (考虑放电效率，实际输出)
            max_dis_power = min(P_ess, dischargeable_energy / DT)

            if max_dis_power > 0:
                dis_power = min(remaining_load, max_dis_power)
                P_dis[t] = dis_power
                remaining_load -= dis_power

        # === 步骤4: 电网购电 ===
        P_grid[t] = max(0, remaining_load)

        # === 更新储能电量 ===
        if P_ess > 0 and E_ess > 0:
            energy_in = ETA_C * P_ch[t] * DT
            energy_out = P_dis[t] * DT / ETA_D
            E[t+1] = max(S_MIN * E_ess, min(S_MAX * E_ess,
                                             E[t] + energy_in - energy_out))
        else:
            E[t+1] = E[t]

    # === 结果组装 ===
    result = {
        'success': True,
        'status': 'RuleBased',
        'P_ess': P_ess,
        'E_ess': E_ess,
        't': list(range(T)),
        'P_load_pv': P_load_pv.tolist(),
        'P_load_w': P_load_w.tolist(),
        'P_ch_pv': P_ch_pv.tolist(),
        'P_ch_w': P_ch_w.tolist(),
        'P_curt_pv': P_curt_pv.tolist(),
        'P_curt_w': P_curt_w.tolist(),
        'P_ch': P_ch.tolist(),
        'P_dis': P_dis.tolist(),
        'E': E.tolist(),
        'z': z.tolist(),
        'P_grid': P_grid.tolist(),
    }

    # 指标
    result['load_total'] = float(np.sum(load) * DT)
    result['grid_total'] = float(np.sum(P_grid) * DT)
    result['pv_use'] = float(np.sum(P_load_pv) + np.sum(P_ch_pv)) * DT
    result['w_use'] = float(np.sum(P_load_w) + np.sum(P_ch_w)) * DT
    result['pv_curt'] = float(np.sum(P_curt_pv)) * DT
    result['w_curt'] = float(np.sum(P_curt_w)) * DT
    result['curt_total'] = result['pv_curt'] + result['w_curt']
    result['pv_gen'] = float(np.sum(G_pv) * DT)
    result['w_gen'] = float(np.sum(G_w) * DT)
    total_re_gen = result['pv_gen'] + result['w_gen']
    total_re_use = result['pv_use'] + result['w_use']
    result['re_ratio'] = total_re_use / total_re_gen if total_re_gen > 0 else 0.0

    # 成本
    from data_loader import C_PV, C_W, C_G, C_P_ESS, C_E_ESS, Y
    daily_cost = (C_PV * (np.sum(P_load_pv) + np.sum(P_ch_pv)) * DT +
                  C_W * (np.sum(P_load_w) + np.sum(P_ch_w)) * DT +
                  C_G * np.sum(P_grid) * DT)
    result['daily_cost'] = float(daily_cost)
    result['inv_cost'] = C_P_ESS * P_ess + C_E_ESS * E_ess
    result['inv_annual'] = result['inv_cost'] / Y
    result['annual_cost'] = 365 * daily_cost + result['inv_annual']
    result['unit_daily_cost'] = result['daily_cost'] / result['load_total'] if result['load_total'] > 0 else 0
    result['unit_annual_cost'] = result['annual_cost'] / (365 * result['load_total']) if result['load_total'] > 0 else 0

    return result


if __name__ == '__main__':
    from data_loader import load_load_data, load_solar_wind_data

    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()

    park = 'A'
    L = load_data[park]
    P = G_pv_data[park]
    W = G_w_data[park]

    # 无储能
    r0 = rule_based_dispatch(L, P, W, 0, 0)
    print(f'\n园区{park} 无储能 (规则调度):')
    print(f'日运行成本: {r0["daily_cost"]:.2f} 元')
    print(f'购电量: {r0["grid_total"]:.2f} kWh')
    print(f'弃电量: {r0["curt_total"]:.2f} kWh')
    print(f'消纳率: {r0["re_ratio"]*100:.1f}%')

    # 固定储能
    r1 = rule_based_dispatch(L, P, W, 50, 100)
    print(f'\n园区{park} 固定 50kW/100kWh (规则调度):')
    print(f'日运行成本: {r1["daily_cost"]:.2f} 元')
    print(f'购电量: {r1["grid_total"]:.2f} kWh')
    print(f'弃电量: {r1["curt_total"]:.2f} kWh')
    print(f'消纳率: {r1["re_ratio"]*100:.1f}%')

    print('\nrule_dispatch demo 运行完毕')
