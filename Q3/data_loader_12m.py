# -*- coding: utf-8 -*-
"""
附件3 数据加载器：12个月典型日风光发电数据

附件3结构：
  - Sheet1, 28行×49列
  - Row 0: 说明文字
  - Row 1: 月份标签（1月~12月，每月占4列合并单元格）
  - Row 2: 园区标签（园区A/园区B/园区C，第4列为合并残留NaN）
  - Row 3: 类型标签（光伏出力/风电出力/风电出力/光伏出力）
  - Rows 4-27: 24小时数据，Col 0=时间，Cols 1-48=12月×4列

每月4列顺序：A光伏、B风电、C风电、C光伏

返回归一化标幺值 φ（无量纲），实际发电量 = (K0 + ΔK) × φ
"""

import os
import sys

import numpy as np
import pandas as pd

# 项目根目录
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 沿用 Q3(1) 的现有容量 K₀
K0_PV = {'A': 750, 'B': 0, 'C': 600}
K0_W  = {'A': 0, 'B': 1000, 'C': 500}

# 各月天数（非闰年）
DAYS_PER_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# 分时电价
def grid_price(t):
    """返回时段 t (0..23) 的电网购电价格（元/kWh）。
    峰段 7:00-21:00 (t=7..21) → 1.0 元/kWh
    谷段 0:00-6:00, 22:00-23:00 (t=0..6,22,23) → 0.4 元/kWh
    """
    if 7 <= t <= 21:
        return 1.0
    else:
        return 0.4

# 成本参数 (同 Q3(1))
C_PV = 0.4
C_W = 0.5
C_P_ESS = 800
C_E_ESS = 1800
Y_ESS = 10
ETA_C = 0.95
ETA_D = 0.95
S_MIN = 0.10
S_MAX = 0.90
S_0 = 0.50
DT = 1.0


def load_12m_data(path=None):
    """读取附件3，返回12个月归一化风光出力。

    Args:
        path: 附件3路径，默认 attachment/附件3：12个月各园区典型日风光发电数据.xlsx

    Returns:
        dict:
            'phi_pv': dict[str, np.ndarray]  # {park: array[12, 24]}
            'phi_w':  dict[str, np.ndarray]  # {park: array[12, 24]}
            'months': int = 12
            'hours': int = 24
            'days_per_month': np.ndarray[12]
    """
    if path is None:
        path = os.path.join(_PROJECT_ROOT, 'attachment',
                            '附件3：12个月各园区典型日风光发电数据.xlsx')

    raw = pd.read_excel(path, sheet_name='Sheet1', header=None)

    # 数据从行4开始（0-indexed），列1开始（列0=时间）
    data_block = raw.iloc[4:28, 1:].values.astype(float)  # shape (24, 48)

    if data_block.shape != (24, 48):
        raise ValueError(f"期望数据块形状 (24,48)，实际 {data_block.shape}")

    NM = 12  # 月份数
    # 每月4列：A光伏, B风电, C风电, C光伏
    phi_pv_A = np.zeros((NM, 24))
    phi_w_B  = np.zeros((NM, 24))
    phi_w_C  = np.zeros((NM, 24))
    phi_pv_C = np.zeros((NM, 24))

    for m in range(NM):
        base_col = m * 4
        phi_pv_A[m, :] = data_block[:, base_col + 0]  # A光伏
        phi_w_B[m, :]  = data_block[:, base_col + 1]  # B风电
        phi_w_C[m, :]  = data_block[:, base_col + 2]  # C风电
        phi_pv_C[m, :] = data_block[:, base_col + 3]  # C光伏

    # 合理性检查
    for arr, label in [
        (phi_pv_A, 'A光伏'), (phi_w_B, 'B风电'),
        (phi_w_C, 'C风电'), (phi_pv_C, 'C光伏')
    ]:
        if np.any(arr < 0) or np.any(arr > 1.5):
            raise ValueError(f"{label} 标幺值越界: min={arr.min():.4f}, max={arr.max():.4f}")
        # 光伏列夜间应为零
        if '光伏' in label:
            night_max = arr[:, 0:6].max()  # t=0..5
            if night_max > 0.01:
                print(f"  [WARN] {label} 夜间存在非零值: max={night_max:.4f}")

    result = {
        'phi_pv': {'A': phi_pv_A, 'C': phi_pv_C},  # 仅A和C有光伏
        'phi_w':  {'B': phi_w_B, 'C': phi_w_C},     # 仅B和C有风电
        'months': NM,
        'hours': 24,
        'days_per_month': DAYS_PER_MONTH.copy(),
    }

    print(f"[data_loader_12m] 已加载 12 个月 × 24h 风光标幺数据")
    print(f"  光伏 A: range [{phi_pv_A.min():.4f}, {phi_pv_A.max():.4f}]")
    print(f"  风电 B: range [{phi_w_B.min():.4f}, {phi_w_B.max():.4f}]")
    print(f"  风电 C: range [{phi_w_C.min():.4f}, {phi_w_C.max():.4f}]")
    print(f"  光伏 C: range [{phi_pv_C.min():.4f}, {phi_pv_C.max():.4f}]")

    return result


def load_load_12m(path=None):
    """读取附件1负荷（沿用 Q3(1) 逻辑），放大1.5倍，返回12个月共用。

    Returns:
        dict[str, np.ndarray]: {park: array[12, 24]} — 每月相同的负荷
    """
    if path is None:
        path = os.path.join(_PROJECT_ROOT, 'attachment',
                            '附件1：各园区典型日负荷数据.xlsx')

    raw = pd.read_excel(path, header=None)

    # 从00:00行开始取数据
    data_start = None
    for i in range(len(raw)):
        val = str(raw.iloc[i, 0]).strip()
        if val == '00:00' or '00:00:00' in val:
            data_start = i
            break
    if data_start is None:
        raise ValueError("附件1中未找到数据起始行")

    load_data = raw.iloc[data_start:data_start + 24, 1:4].values.astype(float) * 1.5

    # 附件1列序：园区A负荷, 园区B负荷, 园区C负荷
    load_dict = {}
    for idx, park in enumerate(['A', 'B', 'C']):
        base = load_data[:, idx]  # shape (24,)
        load_dict[park] = np.tile(base, (12, 1))  # shape (12, 24)

    print(f"[data_loader_12m] 负荷 1.5× 已加载，园区A峰值={load_dict['A'].max():.0f}kW")
    return load_dict


# =====================================================================
# 自检
# =====================================================================
if __name__ == '__main__':
    d12 = load_12m_data()
    ld = load_load_12m()
    print("\n数据形状检查:")
    for p in ['A', 'B', 'C']:
        print(f"  负荷 {p}: {ld[p].shape}")
    print(f"  光伏 A: {d12['phi_pv']['A'].shape}")
    print(f"  风电 B: {d12['phi_w']['B'].shape}")
    print(f"  风电 C: {d12['phi_w']['C'].shape}")
    print(f"  光伏 C: {d12['phi_pv']['C'].shape}")
    print(f"  各月天数: {d12['days_per_month']}, sum={d12['days_per_month'].sum()}")
    print("测试通过 ✓")
