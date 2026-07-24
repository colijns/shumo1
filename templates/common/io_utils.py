"""
模块：io_utils
功能：数据 IO 工具--按扩展名自动读取 csv/xlsx，导出 DataFrame 或 dict 为 csv/xlsx
适用题型：通用（比赛数据通常为 Excel/CSV，适配各模块前的数据加载与结果导出）
依赖：pandas
用法：直接运行 `python io_utils.py` 查看 demo；或在比赛代码中
      from common.io_utils import load_table, export_result
注意：本模块供使用者在比赛代码中调用，不被各算法模块 import（保持单文件独立原则）。
"""
import os

import pandas as pd


def load_table(path, sheet_name=0, **kwargs):
    """按扩展名自动读取表格文件为 DataFrame。

    参数：
        path       : str，文件路径，支持 .csv / .xlsx / .xls
        sheet_name : Excel 工作表（csv 忽略），默认第 0 张
        **kwargs   : 透传给 pd.read_csv / pd.read_excel
    返回：
        pandas.DataFrame
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path, **kwargs)
    elif ext in ('.xlsx', '.xls'):
        return pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    else:
        raise ValueError(f'不支持的文件扩展名: {ext}（仅支持 .csv/.xlsx/.xls）')


def export_result(obj, path, **kwargs):
    """把 DataFrame 或 dict-of-lists 导出为 csv/xlsx。

    参数：
        obj  : pandas.DataFrame 或可构造 DataFrame 的 dict
        path : str，输出路径，扩展名决定格式
        **kwargs : 透传给 to_csv / to_excel
    返回：
        str，输出文件的绝对路径
    """
    df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        df.to_csv(path, index=False, **kwargs)
    elif ext in ('.xlsx', '.xls'):
        df.to_excel(path, index=False, **kwargs)
    else:
        raise ValueError(f'不支持的文件扩展名: {ext}（仅支持 .csv/.xlsx/.xls）')
    return os.path.abspath(path)


if __name__ == '__main__':
    import numpy as np
    df = pd.DataFrame({'方案': ['A', 'B', 'C'], '得分': [0.8, 0.6, 0.9]})
    p = export_result(df, 'io_utils_demo.csv')
    print(f'已导出：{p}')
    print('读回：')
    print(load_table(p))
    os.remove(p)
    print('\nio_utils demo 运行完毕')
