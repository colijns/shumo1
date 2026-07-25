# -*- coding: utf-8 -*-
"""临时验证脚本：核对叠合图逐时数据与报告标注口径，用完即删"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))
import numpy as np
from data_loader import load_load_data, load_solar_wind_data
from milp_model import run_no_storage, run_fixed_storage

load_data = load_load_data()
G_pv, G_w = load_solar_wind_data()
for park in ['A', 'B', 'C']:
    L = load_data[park]; P = G_pv[park]; W = G_w[park]
    r_no = run_no_storage(L, P, W)
    r_fix = run_fixed_storage(L, P, W, P_ess=50, E_ess=100)
    curt = np.array(r_fix['P_curt_pv']) + np.array(r_fix['P_curt_w'])
    pch = np.array(r_fix['P_ch'])
    pdis = np.array(r_fix['P_dis'])
    print(f'\n===== 园区{park} =====')
    print(f'  弃电sum={curt.sum():.1f} (报告 {867.0 if park=="A" else 723.4 if park=="B" else 993.8})')
    print(f'  充电sum={pch.sum():.1f} (报告 {84.2 if park=="A" else 174.1 if park=="B" else 134.2})')
    print(f'  放电sum={pdis.sum():.1f}')
    print(f'  成本降幅={r_no["daily_cost"]-r_fix["daily_cost"]:.2f} (报告 {42.32 if park=="A" else 70.06 if park=="B" else 62.44})')
    print(f'  curt={np.round(curt,1).tolist()}')
    print(f'  pch ={np.round(pch,1).tolist()}')
    print(f'  pdis={np.round(pdis,1).tolist()}')
    # 硬性约束检查
    ch_nocurt = (pch > 1e-3) & (curt < 1e-3)
    both = (pch > 1e-3) & (pdis > 1e-3)
    print(f'  充电但无弃电的时段 idx={np.where(ch_nocurt)[0].tolist()} 数量={ch_nocurt.sum()}')
    print(f'  同时充放电时段数={both.sum()}  pch峰值={pch.max():.1f} pdis峰值={pdis.max():.1f}')
    # 弃电是否仅在 G>L 时
    G = P + W
    curt_when_gl_le = (curt > 1e-3) & (G <= L + 1e-3)
    print(f'  弃电但G<=L的时段数={curt_when_gl_le.sum()}')
