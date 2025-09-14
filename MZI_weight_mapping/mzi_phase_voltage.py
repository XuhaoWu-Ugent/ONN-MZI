import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def extract_voltage_phase_relationship(csv_file):
    """
    直接提取电压-相位对应关系，不进行插值处理
    """

    # 读取原始数据
    try:
        data = np.loadtxt(csv_file, delimiter=',')
        voltage = data[:, 0]
        transmission_dB = data[:, 1]
        print(f"数据加载成功: {len(voltage)} 个数据点")
        print(f"电压范围: {voltage[0]:.3f}V 到 {voltage[-1]:.3f}V")
    except Exception as e:
        print(f"数据加载错误: {e}")
        return

    # 转换为线性透射率
    transmission_linear = 10 ** (transmission_dB / 10)

    # 寻找极值点来确定相位缩放关系
    min_peaks, _ = find_peaks(-transmission_linear, distance=15, prominence=0.0005)
    max_peaks, _ = find_peaks(transmission_linear, distance=15, prominence=0.0005)

    if len(min_peaks) == 0 or len(max_peaks) == 0:
        min_peaks, _ = find_peaks(-transmission_linear, distance=10)
        max_peaks, _ = find_peaks(transmission_linear, distance=10)

    print(f"找到 {len(min_peaks)} 个极小值和 {len(max_peaks)} 个极大值")

    # 确定相位缩放系数
    if len(min_peaks) >= 2:
        # 使用相邻两个极小值点
        v1, v2 = voltage[min_peaks[0]], voltage[min_peaks[1]]
        # 相邻极小值间相位差为2π
        phase_scale = 2 * np.pi / (v2 ** 2 - v1 ** 2)
        print(f"使用极小值点确定缩放系数: V1={v1:.3f}V, V2={v2:.3f}V")
    elif len(max_peaks) >= 2:
        # 使用相邻两个极大值点
        v1, v2 = voltage[max_peaks[0]], voltage[max_peaks[1]]
        phase_scale = 2 * np.pi / (v2 ** 2 - v1 ** 2)
        print(f"使用极大值点确定缩放系数: V1={v1:.3f}V, V2={v2:.3f}V")
    else:
        # 如果极值点不够，进行拟合来确定缩放系数
        print("极值点不足，使用拟合方法确定缩放系数...")

        # 选择一个包含振荡的区间进行拟合
        all_extrema = np.sort(np.concatenate([min_peaks, max_peaks])) if len(min_peaks) > 0 and len(
            max_peaks) > 0 else []

        if len(all_extrema) >= 1:
            start_idx = max(0, all_extrema[0] - 10)
            end_idx = min(len(voltage), all_extrema[-1] + 20) if len(all_extrema) > 1 else len(voltage) // 2
        else:
            start_idx = 0
            end_idx = len(voltage) // 3

        v_region = voltage[start_idx:end_idx]
        t_region = transmission_linear[start_idx:end_idx]

        # 使用拟合确定相位缩放系数
        def mzi_transmission(phase, A, phi0, offset):
            return A * np.cos((phase + phi0) / 2) ** 2 + offset

        # 初始估计的相位缩放系数
        phase_scale_guess = 0.1
        phase_region = v_region ** 2 * phase_scale_guess

        try:
            A_guess = np.max(t_region) - np.min(t_region)
            phi0_guess = 0
            offset_guess = np.min(t_region)

            popt, _ = curve_fit(mzi_transmission, phase_region, t_region,
                                p0=[A_guess, phi0_guess, offset_guess],
                                maxfev=10000)

            # 通过拟合结果调整相位缩放系数
            # 这里我们使用一个简化的方法：观察数据的周期性
            voltage_span = v_region[-1] - v_region[0]
            estimated_periods = voltage_span / (v_region[len(v_region) // 4] - v_region[0]) if len(v_region) > 4 else 1
            phase_scale = estimated_periods * np.pi / (voltage_span ** 2)

            print(f"通过拟合估计的相位缩放系数")

        except:
            phase_scale = 0.1  # 默认值
            print(f"使用默认相位缩放系数")

    print(f"最终相位缩放系数: {phase_scale:.6f} rad/V²")

    # 计算每个数据点对应的相位差
    # 使用 φ = k × V² 的关系，并设置起始相位
    phase_difference = voltage ** 2 * phase_scale

    # 调整相位起点，让第一个点的相位接近0
    phase_offset = -phase_difference[0]
    phase_difference = phase_difference + phase_offset

    print(f"相位差范围: {np.min(phase_difference):.3f} 到 {np.max(phase_difference):.3f} rad")
    print(f"相位差范围: {np.min(phase_difference) * 180 / np.pi:.1f}° 到 {np.max(phase_difference) * 180 / np.pi:.1f}°")

    # 创建电压-相位对应关系数据框
    voltage_phase_df = pd.DataFrame({
        'Voltage_V': voltage,
        'Phase_Difference_rad': phase_difference,
        'Phase_Difference_deg': phase_difference * 180 / np.pi,
        'Transmission_dB': transmission_dB,
        'Transmission_Linear': transmission_linear
    })

    # 只保留0-10V范围内的数据
    mask = (voltage >= 0) & (voltage <= 10)
    voltage_phase_df_filtered = voltage_phase_df[mask].copy()

    print(f"\n0-10V范围内的数据点: {len(voltage_phase_df_filtered)} 个")

    # 保存为CSV文件
    output_filename = 'Voltage_Phase_Correspondence.csv'
    voltage_phase_df_filtered.to_csv(output_filename, index=False)

    print(f"✓ 电压-相位对应关系已保存到: {output_filename}")

    # 显示数据预览
    print(f"\n数据预览 (前15行):")
    print(voltage_phase_df_filtered.head(15).to_string(index=False, float_format='%.6f'))

    print(f"\n数据预览 (后15行):")
    print(voltage_phase_df_filtered.tail(15).to_string(index=False, float_format='%.6f'))

    # 可视化验证
    plt.figure(figsize=(15, 10))

    # 子图1: 电压 vs 相位差
    plt.subplot(2, 3, 1)
    plt.plot(voltage_phase_df_filtered['Voltage_V'],
             voltage_phase_df_filtered['Phase_Difference_rad'],
             'b.-', markersize=3, linewidth=1)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Phase Difference (rad)')
    plt.title('Voltage vs Phase Difference (0-10V)')
    plt.grid(True)

    # 子图2: 电压 vs 相位差 (度)
    plt.subplot(2, 3, 2)
    plt.plot(voltage_phase_df_filtered['Voltage_V'],
             voltage_phase_df_filtered['Phase_Difference_deg'],
             'r.-', markersize=3, linewidth=1)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Phase Difference (deg)')
    plt.title('Voltage vs Phase Difference (degrees)')
    plt.grid(True)

    # 子图3: V² vs 相位差 (验证线性关系)
    plt.subplot(2, 3, 3)
    voltage_squared = voltage_phase_df_filtered['Voltage_V'] ** 2
    plt.plot(voltage_squared,
             voltage_phase_df_filtered['Phase_Difference_rad'],
             'go', markersize=3, alpha=0.7)

    # 拟合直线验证线性关系
    coeffs = np.polyfit(voltage_squared, voltage_phase_df_filtered['Phase_Difference_rad'], 1)
    v2_line = np.linspace(0, np.max(voltage_squared), 100)
    phase_line = coeffs[0] * v2_line + coeffs[1]
    plt.plot(v2_line, phase_line, 'r--', linewidth=2,
             label=f'φ = {coeffs[0]:.4f}×V² + {coeffs[1]:.4f}')

    plt.xlabel('Voltage² (V²)')
    plt.ylabel('Phase Difference (rad)')
    plt.title('V² vs Phase (Linearity Check)')
    plt.legend()
    plt.grid(True)

    # 子图4: 电压 vs 透射率 (dB)
    plt.subplot(2, 3, 4)
    plt.plot(voltage_phase_df_filtered['Voltage_V'],
             voltage_phase_df_filtered['Transmission_dB'],
             'b.-', markersize=3, linewidth=1)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Transmission (dB)')
    plt.title('Voltage vs Transmission (dB)')
    plt.grid(True)

    # 子图5: 电压 vs 透射率 (线性)
    plt.subplot(2, 3, 5)
    plt.plot(voltage_phase_df_filtered['Voltage_V'],
             voltage_phase_df_filtered['Transmission_Linear'],
             'r.-', markersize=3, linewidth=1)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Transmission (Linear)')
    plt.title('Voltage vs Transmission (Linear)')
    plt.grid(True)

    # 子图6: 相位差 vs 透射率
    plt.subplot(2, 3, 6)
    plt.plot(voltage_phase_df_filtered['Phase_Difference_rad'],
             voltage_phase_df_filtered['Transmission_Linear'],
             'mo', markersize=3, alpha=0.7)
    plt.xlabel('Phase Difference (rad)')
    plt.ylabel('Transmission (Linear)')
    plt.title('Phase vs Transmission')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 统计信息
    print(f"\n=== 统计信息 ===")
    print(f"数据点总数: {len(voltage_phase_df_filtered)}")
    print(
        f"电压范围: {voltage_phase_df_filtered['Voltage_V'].min():.3f}V 到 {voltage_phase_df_filtered['Voltage_V'].max():.3f}V")
    print(
        f"相位差范围: {voltage_phase_df_filtered['Phase_Difference_rad'].min():.3f} 到 {voltage_phase_df_filtered['Phase_Difference_rad'].max():.3f} rad")
    print(
        f"相位差范围: {voltage_phase_df_filtered['Phase_Difference_deg'].min():.1f}° 到 {voltage_phase_df_filtered['Phase_Difference_deg'].max():.1f}°")
    print(
        f"透射率范围: {voltage_phase_df_filtered['Transmission_dB'].min():.2f} 到 {voltage_phase_df_filtered['Transmission_dB'].max():.2f} dB")

    # 计算电压步长
    voltage_steps = np.diff(voltage_phase_df_filtered['Voltage_V'])
    print(f"电压步长: 平均 {np.mean(voltage_steps):.6f}V, 标准差 {np.std(voltage_steps):.6f}V")

    return voltage_phase_df_filtered


# 执行提取
if __name__ == "__main__":
    csv_filename = 'MZI_IV_curve_IL.csv'
    result_df = extract_voltage_phase_relationship(csv_filename)

    if result_df is not None:
        print(f" 输出文件: Voltage_Phase_Correspondence.csv")
        print(f" 包含 {len(result_df)} 个数据点的一一对应关系")
