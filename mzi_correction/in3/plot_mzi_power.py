
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Read the data from the text file
file_path = r'C:\Users\17958\AppData\Roaming\JetBrains\PyCharm2025.2\scratches\mzi_correction\in3\MZI_array_output_in3.txt'
data = pd.read_csv(file_path, sep='\s+', comment='#', header=None)
data.columns = ['input_port', 'input_power', 'input_mzi_number', 'input_mzi_voltage', 'output_port', 'output_power']

# MZI numbers to analyze
mzi_numbers = [3, 8, 13, 22, 26, 30, 34, 37, 38, 42, 46]

# Filter for output port o5
data_o1 = data[data['output_port'] == 7]
# data_o1 = data[data['output_port'] == 4]

print("Individual MZI power ranges:")

def mzi_power_model(voltage, a, b, p_max, p_min):
    """
    MZI power model: P = P_min + (P_max - P_min) * cos^2((a * V^2 + b) / 2)
    where a, b are phase-voltage relationship parameters
    """
    phase = a * voltage**2 + b
    power = p_min + (p_max - p_min) * np.cos(phase / 2)**2
    return power

def fit_mzi_parameters(voltage_data, power_data):
    """
    Fit MZI parameters based on phi = a * V^2 + b relationship
    Using individual MZI power max/min values as fixed parameters
    """
    # Calculate individual MZI power extrema
    individual_p_max = np.max(power_data)
    individual_p_min = np.min(power_data)

    # Simplified model, only fit phase parameters a and b
    def simplified_mzi_model(voltage, a, b):
        phase = a * voltage**2 + b
        power = individual_p_min + (individual_p_max - individual_p_min) * np.cos(phase / 2)**2
        return power

    try:
        # Based on single MZI test results: a = 0.1801
        # This is the reference value from your testing
        a_guess = 0.1801  # Reference value from single MZI testing
        print(f"Expected phase scaling coefficient a ~= {a_guess:.6f}")

        # Parameter b can have deviation, but not too large
        b_guess = 0

        # Use more reasonable parameter bounds
        # a value should be near reference value, allow +/-30% deviation
        a_min = a_guess * 0.7
        a_max = a_guess * 1.3

        # Fit simplified model containing only phase parameters
        popt, _ = curve_fit(
            simplified_mzi_model,
            voltage_data,
            power_data,
            p0=[a_guess, b_guess],
            maxfev=20000,
            bounds=([a_min, -2*np.pi], [a_max, 2*np.pi])
        )

        # Return complete parameters [a, b, p_max, p_min]
        return [popt[0], popt[1], individual_p_max, individual_p_min]

    except Exception as e:
        print(f"Fitting failed: {e}")
        # If fitting fails, use theoretical estimate
        return [a_guess, 0, individual_p_max, individual_p_min]

# Create a plot
plt.figure(figsize=(15, 10))

# Create subplots
plt.subplot(2, 1, 1)

# Define colors for each MZI to maintain consistency
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Plot data for each MZI with interpolation
for i, mzi_num in enumerate(mzi_numbers):
    mzi_data = data_o1[data_o1['input_mzi_number'] == mzi_num].sort_values(by='input_mzi_voltage')
    if not mzi_data.empty:
        voltage_points = mzi_data['input_mzi_voltage'].values
        power_points = mzi_data['output_power'].values

        # Print individual MZI power range
        individual_p_max = np.max(power_points)
        individual_p_min = np.min(power_points)
        print(f"MZI {mzi_num}: P_max = {individual_p_max:.6f} W, P_min = {individual_p_min:.6f} W")

        # Fit parameters
        fitted_params = fit_mzi_parameters(voltage_points, power_points)
        a, b, p_max, p_min = fitted_params

        # Generate interpolation voltage sequence
        v_min, v_max = np.min(voltage_points), np.max(voltage_points)
        voltage_interp = np.linspace(v_min, v_max, 200)

        # Calculate interpolated power using fitted parameters
        power_interp = mzi_power_model(voltage_interp, a, b, p_max, p_min)

        # Use same color for data points and fitted curve
        color = colors[i % len(colors)]
        plt.plot(voltage_points, power_points, 'o', markersize=6, color=color,
                label=f'MZI {mzi_num} (data)', zorder=3)
        plt.plot(voltage_interp, power_interp, '-', linewidth=2, alpha=0.8, color=color,
                label=f'MZI {mzi_num} (fitted: a={a:.4f})', zorder=2)

# Second subplot: Display fitting parameters
plt.subplot(2, 1, 2)
fitted_params_list = []

for mzi_num in mzi_numbers:
    mzi_data = data_o1[data_o1['input_mzi_number'] == mzi_num].sort_values(by='input_mzi_voltage')
    if not mzi_data.empty:
        voltage_points = mzi_data['input_mzi_voltage'].values
        power_points = mzi_data['output_power'].values
        fitted_params = fit_mzi_parameters(voltage_points, power_points)
        fitted_params_list.append((mzi_num, fitted_params))

# Extract parameters for visualization
mzi_nums = [x[0] for x in fitted_params_list]
a_values = [x[1][0] for x in fitted_params_list]
b_values = [x[1][1] for x in fitted_params_list]

# Create dual y-axis plot for both a and b parameters
plt.subplot(2, 1, 2)
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot parameter a on left y-axis
line1 = ax1.plot(mzi_nums, a_values, 'ro-', linewidth=2, markersize=6, label='Parameter a (phase scaling)')
ax1.set_xlabel('MZI Number')
ax1.set_ylabel('Parameter a', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.grid(True, alpha=0.3)

# Plot parameter b on right y-axis
line2 = ax2.plot(mzi_nums, b_values, 'bs-', linewidth=2, markersize=6, label='Parameter b (phase offset)')
ax2.set_ylabel('Parameter b (rad)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Add reference line for a = 0.1801
ax1.axhline(y=0.1801, color='red', linestyle='--', alpha=0.7, label='Reference a = 0.1801')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('MZI Parameters vs MZI Number')

plt.subplot(2, 1, 1)

# Add labels and title
plt.xlabel('Voltage (V)')
plt.ylabel('Output Power (W)')
plt.title('MZI Output Power (o1) vs. Voltage')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the plot
plt.savefig(r'C:\Users\17958\AppData\Roaming\JetBrains\PyCharm2025.2\scratches\mzi_correction\in3\mzi_power_analysis_fitted.png', dpi=300, bbox_inches='tight')

print("Plot saved to mzi_power_analysis_fitted.png")
print("\n=== Fitting Parameters Summary ===")
for i, (mzi_num, params) in enumerate(fitted_params_list):
    a, b, p_max, p_min = params
    print(f"MZI {mzi_num}: a={a:.6f}, b={b:.3f}, P_max={p_max:.6f}, P_min={p_min:.6f}")

# Calculate parameter statistics
a_mean = np.mean(a_values)
a_std = np.std(a_values)
a_cv = a_std / a_mean if a_mean != 0 else 0  # Coefficient of variation

print(f"\nParameter a statistics: mean={a_mean:.6f}, std={a_std:.6f}, CV={a_cv:.3f}")
print(f"Phase-voltage relationship: phi = {a_mean:.6f} * V^2 + b")

# Check consistency of parameter a
reference_a = 0.1801  # Reference value from single MZI testing
print(f"Reference a = {reference_a:.6f}")
print(f"Actual average a = {a_mean:.6f} (deviation: {abs(a_mean - reference_a)/reference_a*100:.1f}%)")

# Calculate parameter b statistics
b_mean = np.mean(b_values)
b_std = np.std(b_values)
print(f"Parameter b statistics: mean={b_mean:.6f}, std={b_std:.6f}")
print(f"Phase offset range: {np.min(b_values):.3f} to {np.max(b_values):.3f} rad")
print(f"Phase offset range: {np.min(b_values)*180/np.pi:.1f} to {np.max(b_values)*180/np.pi:.1f} degrees")

# Display fitting quality check
if a_cv < 0.2:  # Coefficient of variation less than 20%
    print("Good consistency between MZIs")
else:
    print("Warning: Large differences between MZIs, may need to check data or fitting")
