# -*- coding: utf-8 -*-
"""快速重算问题2（2）的核心方案，不执行耗时的完整二维网格。

输出 JSON 供结果表生成与复核使用；连续 MILP 和工程整数 MILP 本身均为全局优化，
完整网格仅保留为慢速交叉验证和热力图用途。
"""

import json
import os

from q2_storage_penalty import (
    E_MAX,
    E_STEP,
    LAMBDA_VALUES,
    P_MAX,
    P_STEP,
    T,
    TOL,
    build_joint_profile,
    load_load_data,
    load_solar_wind_data,
    run_fixed_storage,
    run_no_storage,
    solve_continuous,
    solve_integer,
)


def _check_result(result, load, pv, wind, require_granularity=False):
    checks = {
        "求解成功": bool(result.get("success")),
        "功率平衡": result["errors"]["max_balance_error"] < TOL,
        "风光分配": result["errors"]["max_re_error"] < TOL,
        "不同时充放电": all(
            not (result["P_ch"][t] > TOL and result["P_dis"][t] > TOL)
            for t in range(T)
        ),
        "只用风光余电充电": all(
            result["P_ch"][t]
            <= max(pv[t] + wind[t] - load[t], 0.0) + TOL
            for t in range(T)
        ),
        "年综合成本重构": abs(
            result["annual_cost"]
            - (365 * result["daily_cost"] + result["inv_annual"])
        ) < 1e-2,
    }
    if require_granularity:
        checks["功率工程粒度"] = (
            abs(result["P_ess"] / P_STEP - round(result["P_ess"] / P_STEP))
            < TOL
        )
        checks["容量工程粒度"] = (
            abs(result["E_ess"] / E_STEP - round(result["E_ess"] / E_STEP))
            < TOL
        )
    return {name: bool(passed) for name, passed in checks.items()}


def _summary_row(lam, scheme, result, checks):
    return {
        "lambda": lam,
        "scheme": scheme,
        "P_ess_kW": result["P_ess"],
        "E_ess_kWh": result["E_ess"],
        "daily_cost_pure_yuan": result["daily_cost_pure"],
        "daily_curt_penalty_yuan": result["daily_curt_penalty"],
        "annualized_investment_yuan": result["inv_annual"],
        "annual_actual_cost_yuan": (
            365 * result["daily_cost_pure"] + result["inv_annual"]
        ),
        "annual_comprehensive_cost_yuan": result["annual_cost"],
        "daily_curtailment_kWh": result["curt_total"],
        "renewable_consumption_rate": result["re_ratio"],
        "hits_capacity_upper": bool(result.get("hits_capacity_upper", False)),
        "checks_passed": all(checks.values()),
        "checks": checks,
    }


def run(output_path=None):
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "output_penalty_revised",
            "core_results.json",
        )

    load_data = load_load_data()
    pv_data, wind_data = load_solar_wind_data()
    load, pv, wind = build_joint_profile(load_data, pv_data, wind_data)

    summary = []
    hourly = {}

    for lam in LAMBDA_VALUES:
        schemes = [
            ("联合无储能", run_no_storage(load, pv, wind, curt_penalty=lam), False),
            (
                "固定50kW/100kWh",
                run_fixed_storage(
                    load, pv, wind,
                    P_ess=50, E_ess=100, curt_penalty=lam,
                ),
                False,
            ),
            ("连续优化", solve_continuous(load, pv, wind, lam=lam), False),
            ("工程整数", solve_integer(load, pv, wind, lam=lam), True),
        ]

        for scheme, result, engineering in schemes:
            checks = _check_result(
                result, load, pv, wind,
                require_granularity=engineering,
            )
            if scheme == "连续优化":
                checks["连续解未触碰物理上界"] = not result["hits_capacity_upper"]
            summary.append(_summary_row(lam, scheme, result, checks))

        continuous = schemes[2][1]
        hourly[str(lam)] = [
            {
                "hour": t + 1,
                "load_kW": load[t],
                "pv_kW": pv[t],
                "wind_kW": wind[t],
                "renewable_surplus_kW": max(pv[t] + wind[t] - load[t], 0.0),
                "charge_kW": continuous["P_ch"][t],
                "discharge_kW": continuous["P_dis"][t],
                "grid_kW": continuous["P_grid"][t],
                "pv_curtailment_kW": continuous["P_curt_pv"][t],
                "wind_curtailment_kW": continuous["P_curt_w"][t],
                "soc_kWh": continuous["E"][t],
            }
            for t in range(T)
        ]

    payload = {
        "model": {
            "P_max_kW": P_MAX,
            "E_max_kWh": E_MAX,
            "P_step_kW": P_STEP,
            "E_step_kWh": E_STEP,
            "surplus_only_charge": True,
            "lambda_values": LAMBDA_VALUES,
        },
        "summary": summary,
        "hourly_continuous": hourly,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    for row in summary:
        if row["scheme"] in ("连续优化", "工程整数"):
            print(
                f"λ={row['lambda']:.1f} {row['scheme']}: "
                f"P={row['P_ess_kW']:.4f} kW, "
                f"E={row['E_ess_kWh']:.4f} kWh, "
                f"年实际={row['annual_actual_cost_yuan']:.3f} 元, "
                f"年综合={row['annual_comprehensive_cost_yuan']:.3f} 元, "
                f"弃电={row['daily_curtailment_kWh']:.3f} kWh, "
                f"校验={'通过' if row['checks_passed'] else '失败'}"
            )

    if not all(row["checks_passed"] for row in summary):
        raise RuntimeError("存在未通过的核心校验，请检查 core_results.json")

    print(f"核心结果已写入：{output_path}")
    return payload


if __name__ == "__main__":
    run()
