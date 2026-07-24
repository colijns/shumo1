# -*- coding: utf-8 -*-
"""
数据加载与预处理模块
功能：读取附件1负荷数据和附件2风光标幺数据，转换为模型所需格式
"""

import sys
import os

import pandas as pd
import numpy as np

# 项目根目录
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 将 templates 加入路径，复用 common 工具
sys.path.append(os.path.join(_PROJECT_ROOT, 'templates'))
from common.io_utils import load_table

# ----- 园区配置 -----
PARKS = {
    'A': {'pv': 750, 'w': 0},
    'B': {'pv': 0, 'w': 1000},
    'C': {'pv': 600, 'w': 500},
}

# 常量参数
C_PV = 0.4       # 光伏购电成本 元/kWh
C_W = 0.5        # 风电购电成本 元/kWh
C_G = 1.0        # 电网购电价格 元/kWh
C_P_ESS = 800    # 储能功率单价 元/kW
C_E_ESS = 1800   # 储能容量单价 元/kWh
Y = 10           # 储能寿命 年
ETA_C = 0.95     # 充电效率
ETA_D = 0.95     # 放电效率
S_MIN = 0.10     # SOC 下限
S_MAX = 0.90     # SOC 上限
S_0 = 0.50       # 初始 SOC
DT = 1.0         # 时段长度 h


def load_load_data(path=None):
    if path is None:
        path = os.path.join(_PROJECT_ROOT, 'attachment', '附件1：各园区典型日负荷数据.xlsx')
    """读取负荷数据，返回 dict[园区] -> array[24] (kW)"""
    raw = load_table(path)
    # 列名处理：可能有空格/BOM
    raw.columns = [c.strip() for c in raw.columns]
    # 提取时间列
    time_col = [c for c in raw.columns if '时间' in c][0]
    times = raw[time_col].values

    load = {}
    for park in ['A', 'B', 'C']:
        col = [c for c in raw.columns if f'{park}' in c and '负荷' in c]
        if col:
            load[park] = raw[col[0]].values.astype(float)
        else:
            # 如果没有精确匹配，尝试按列顺序
            pass

    # 如果按列名没找到，按固定列顺序
    if not load:
        load['A'] = raw.iloc[:, 1].values.astype(float)
        load['B'] = raw.iloc[:, 2].values.astype(float)
        load['C'] = raw.iloc[:, 3].values.astype(float)

    # 保证每个园区都有
    for p in ['A', 'B', 'C']:
        if p not in load:
            raise KeyError(f'园区 {p} 的负荷列未找到')

    print(f'负荷数据：{len(times)} 个时段，{list(load.keys())}')
    return load


def load_solar_wind_data(path=None):
    if path is None:
        path = os.path.join(_PROJECT_ROOT, 'attachment', '附件2：各园区典型日风光发电数据.xlsx')
    """读取风光标幺出力数据，乘装机容量后返回
    返回: (G_pv, G_w) 两个 dict[园区] -> array[24] (kW)
    """
    raw = load_table(path, header=None)

    # 找到数据起始行（00:00:00 所在行）
    data_start = None
    for i in range(raw.shape[0]):
        val = str(raw.iloc[i, 0])
        if '00:00' in val:
            data_start = i
            break
    if data_start is None:
        raise ValueError('未找到数据起始行（00:00:00）')

    # 读取 columns info 行
    header_row = None
    for i in range(min(data_start, 5)):
        vals = [str(v) for v in raw.iloc[i].values]
        if '园区A' in vals[0] or '园区A' in vals[1]:
            header_row = i
            break

    # 列映射
    # 时间(0), 园区A光伏(1), 园区B风电(2), 园区C光伏(3), 园区C风电(4)
    n_cols = raw.shape[1]
    data = raw.iloc[data_start:data_start+24, :5].copy()
    data.columns = ['time', 'A_pv', 'B_w', 'C_pv', 'C_w']

    # 数值转换
    for col in ['A_pv', 'B_w', 'C_pv', 'C_w']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # 乘装机容量
    G_pv = {}
    G_w = {}
    G_pv['A'] = data['A_pv'].values * PARKS['A']['pv']
    G_w['A'] = np.zeros(24)
    G_pv['B'] = np.zeros(24)
    G_w['B'] = data['B_w'].values * PARKS['B']['w']
    G_pv['C'] = data['C_pv'].values * PARKS['C']['pv']
    G_w['C'] = data['C_w'].values * PARKS['C']['w']

    print(f'风光数据：{len(data)} 个时段')
    return G_pv, G_w


def convert_excel_time(x):
    """将 Excel 时间小数转换为 0~23 整数时段
    x: float 如 0, 0.041667 (1/24), ...
    """
    return int(round(24 * float(x)))


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    load = load_load_data()
    for p in ['A', 'B', 'C']:
        print(f'园区{p} 负荷(kW): {load[p]}')

    G_pv, G_w = load_solar_wind_data()
    for p in ['A', 'B', 'C']:
        print(f'园区{p} 光伏(kW): {G_pv[p]}')
        print(f'园区{p} 风电(kW): {G_w[p]}')
    print('data_loader demo 运行完毕')
