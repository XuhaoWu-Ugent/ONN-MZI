import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def create_phase_to_voltage_lookup():
    """
    创建相位到电压的查找函数
    """

    # 读取您生成的对应关系文件
    try:
        # Make path relative to this script file
        import os
        script_dir = os.path.dirname(__file__)
        csv_path = os.path.join(script_dir, 'Voltage_Phase_Correspondence.csv')
        df = pd.read_csv(csv_path)
        voltage = df['Voltage_V'].values
        phase_rad = df['Phase_Difference_rad'].values

        # 创建插值函数 (从相位到电压)
        # 注意：需要确保相位值是单调递增的
        phase_to_volt_func = interp1d(phase_rad, voltage,
                                      kind='linear',
                                      bounds_error=False,
                                      fill_value='extrapolate')

        return phase_to_volt_func, np.min(phase_rad), np.max(phase_rad)

    except FileNotFoundError:
        print("未找到 Voltage_Phase_Correspondence.csv 文件")
        return None, None, None


def find_voltage_for_phase(target_phase_rad):
    """
    使用查找表方法找到对应电压
    """

    lookup_func, min_phase, max_phase = create_phase_to_voltage_lookup()

    if lookup_func is None:
        return None

    # 检查相位范围
    if target_phase_rad < min_phase or target_phase_rad > max_phase:
        print(f"警告：目标相位 {target_phase_rad:.4f} rad 超出数据范围 [{min_phase:.4f}, {max_phase:.4f}]")
        print("将使用外推方法，结果可能不准确")

    # 计算对应电压
    voltage = lookup_func(target_phase_rad)

    return voltage
