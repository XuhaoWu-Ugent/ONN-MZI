import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Configure target MZI number for analysis
TARGET_MZI = 48  # Change this value to select MZI to analyze (0, 5, 10, 15, 20, 25, 30, 35, 40, 49)

# Read the data from the text file
file_path = r'C:\Users\17958\AppData\Roaming\JetBrains\PyCharm2025.2\scratches\mzi_correction\in7\MZI_array_output.txt'
data = pd.read_csv(file_path, sep='\s+', comment='#', header=None)
data.columns = ['input_port', 'input_power', 'input_mzi_number', 'input_mzi_voltage', 'output_port', 'output_power']

# All MZI numbers for analysis
mzi_numbers = [1, 6, 11, 16, 21, 26, 31, 40, 44, 48]

# Filter for output port o3
data_o1 = data[data['output_port'] == 3]

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
        popt, pcov = curve_fit(
            simplified_mzi_model,
            voltage_data,
            power_data,
            p0=[a_guess, b_guess],
            maxfev=20000,
            bounds=([a_min, -2*np.pi], [a_max, 2*np.pi])
        )

        # Calculate fitting quality metrics
        fitted_power = simplified_mzi_model(voltage_data, popt[0], popt[1])
        r_squared = 1 - np.sum((power_data - fitted_power)**2) / np.sum((power_data - np.mean(power_data))**2)
        rmse = np.sqrt(np.mean((power_data - fitted_power)**2))

        # Return complete parameters [a, b, p_max, p_min] and fitting quality
        return [popt[0], popt[1], individual_p_max, individual_p_min], r_squared, rmse, pcov

    except Exception as e:
        print(f"Fitting failed: {e}")
        # If fitting fails, use theoretical estimate
        return [a_guess, 0, individual_p_max, individual_p_min], 0, float('inf'), None

# Get target MZI data
target_mzi_data = data_o1[data_o1['input_mzi_number'] == TARGET_MZI].sort_values(by='input_mzi_voltage')

if target_mzi_data.empty:
    print(f"Error: Cannot find data for MZI {TARGET_MZI}")
    print(f"Available MZI numbers: {mzi_numbers}")
    exit()

voltage_points = target_mzi_data['input_mzi_voltage'].values
power_points = target_mzi_data['output_power'].values

# Calculate individual MZI power range
individual_p_max = np.max(power_points)
individual_p_min = np.min(power_points)

print(f"\n=== Analyzing MZI {TARGET_MZI} ===")
print(f"Number of data points: {len(voltage_points)}")
print(f"Voltage range: {np.min(voltage_points):.3f}V to {np.max(voltage_points):.3f}V")
print(f"Power range: {individual_p_min:.6f}W to {individual_p_max:.6f}W")

# Fit parameters
fitted_params, r_squared, rmse, pcov = fit_mzi_parameters(voltage_points, power_points)
a, b, p_max, p_min = fitted_params

print(f"\n=== Fitting Results ===")
print(f"Phase scaling coefficient a = {a:.6f}")
print(f"Phase offset b = {b:.3f} rad = {b*180/np.pi:.1f} deg")
print(f"Used P_max = {p_max:.6f}W")
print(f"Used P_min = {p_min:.6f}W")
print(f"R-squared = {r_squared:.4f}")
print(f"Root Mean Square Error RMSE = {rmse:.2e}W")

# Parameter uncertainty analysis
if pcov is not None:
    param_errors = np.sqrt(np.diag(pcov))
    print(f"Parameter uncertainties: a +/- {param_errors[0]:.6f}, b +/- {param_errors[1]:.3f}")

# Generate detailed interpolation data
v_min, v_max = np.min(voltage_points), np.max(voltage_points)
voltage_interp = np.linspace(v_min, v_max, 500)  # High-density interpolation
power_interp = mzi_power_model(voltage_interp, a, b, p_max, p_min)

# Extended interpolation to 0-10V range (if data doesn't cover full range)
voltage_extended = np.linspace(0, 10, 1000)
power_extended = mzi_power_model(voltage_extended, a, b, p_max, p_min)

# Create detailed visualization
plt.figure(figsize=(16, 12))

# Subplot 1: Main interpolation results
plt.subplot(2, 3, 1)
plt.plot(voltage_points, power_points, 'ro', markersize=8, label=f'MZI {TARGET_MZI} Raw Data', zorder=3)
plt.plot(voltage_interp, power_interp, 'b-', linewidth=2, label=f'Fitted Curve (R²={r_squared:.3f})', zorder=2)
plt.fill_between(voltage_interp, power_interp, alpha=0.2)
plt.xlabel('Voltage (V)')
plt.ylabel('Output Power (W)')
plt.title(f'MZI {TARGET_MZI} Power Interpolation Results')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Extended to full 0-10V range
plt.subplot(2, 3, 2)
plt.plot(voltage_points, power_points, 'ro', markersize=8, label='Raw Data', zorder=3)
plt.plot(voltage_extended, power_extended, 'b-', linewidth=1.5, label='0-10V Prediction', alpha=0.8, zorder=1)
plt.plot(voltage_interp, power_interp, 'g-', linewidth=2, label='Data Range Fit', zorder=2)
plt.xlabel('Voltage (V)')
plt.ylabel('Output Power (W)')
plt.title('Extended 0-10V Prediction')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Residual analysis
plt.subplot(2, 3, 3)
fitted_power_at_data = mzi_power_model(voltage_points, a, b, p_max, p_min)
residuals = power_points - fitted_power_at_data
plt.plot(voltage_points, residuals, 'ro-', markersize=6)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Voltage (V)')
plt.ylabel('Residuals (W)')
plt.title(f'Fitting Residuals (RMSE={rmse:.2e})')
plt.grid(True, alpha=0.3)

# Subplot 4: Phase vs voltage relationship
plt.subplot(2, 3, 4)
phase_at_points = a * voltage_points**2 + b
phase_extended = a * voltage_extended**2 + b
plt.plot(voltage_points, phase_at_points, 'ro', markersize=8, label='Data Point Phases')
plt.plot(voltage_extended, phase_extended, 'b-', linewidth=1.5, label='Phase Curve')
plt.xlabel('Voltage (V)')
plt.ylabel('Phase (rad)')
plt.title(f'Phase-Voltage Relationship (phi = {a:.4f}*V^2 + {b:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 5: Normalized power vs phase
plt.subplot(2, 3, 5)
power_normalized = (power_points - p_min) / (p_max - p_min)
theoretical_normalized = np.cos(phase_at_points / 2)**2
plt.plot(phase_at_points, power_normalized, 'ro', markersize=8, label='Actual Data')
phase_theory = np.linspace(np.min(phase_at_points), np.max(phase_at_points), 100)
power_theory = np.cos(phase_theory / 2)**2
plt.plot(phase_theory, power_theory, 'b-', linewidth=2, label='Theoretical cos^2(phi/2)')
plt.xlabel('Phase (rad)')
plt.ylabel('Normalized Power')
plt.title('Power vs Phase (Verify cos^2 relationship)')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 6: Data distribution and statistics
plt.subplot(2, 3, 6)
plt.text(0.1, 0.9, f'MZI Number: {TARGET_MZI}', transform=plt.gca().transAxes, fontsize=12, weight='bold')
plt.text(0.1, 0.8, f'Data Points: {len(voltage_points)}', transform=plt.gca().transAxes)
plt.text(0.1, 0.7, f'Voltage Range: {np.min(voltage_points):.2f}-{np.max(voltage_points):.2f}V', transform=plt.gca().transAxes)
plt.text(0.1, 0.6, f'Power Range: {np.min(power_points):.4f}-{np.max(power_points):.4f}W', transform=plt.gca().transAxes)
plt.text(0.1, 0.5, f'Fit Parameters:', transform=plt.gca().transAxes, weight='bold')
plt.text(0.1, 0.4, f'  a = {a:.6f}', transform=plt.gca().transAxes)
plt.text(0.1, 0.3, f'  b = {b:.3f} rad', transform=plt.gca().transAxes)
plt.text(0.1, 0.2, f'Fit Quality:', transform=plt.gca().transAxes, weight='bold')
plt.text(0.1, 0.1, f'  R^2 = {r_squared:.4f}', transform=plt.gca().transAxes)
plt.text(0.1, 0.0, f'  RMSE = {rmse:.2e}W', transform=plt.gca().transAxes)

# Calculate extrema positions in 0-10V range
if a > 0:
    # Maximum positions: cos^2(phase/2) = 1, i.e. phase = 2n*pi
    # a*V^2 + b = 2n*pi, V = sqrt((2n*pi - b)/a)
    max_phases = []
    for n in range(-5, 10):  # Check multiple periods
        phase_target = 2 * n * np.pi
        if (phase_target - b) / a >= 0:
            v_max_candidate = np.sqrt((phase_target - b) / a)
            if 0 <= v_max_candidate <= 10:
                max_phases.append(v_max_candidate)

    plt.text(0.6, 0.9, f'Predicted Maxima Positions:', transform=plt.gca().transAxes, weight='bold')
    for i, v_pos in enumerate(sorted(max_phases)[:5]):  # Show first 5
        plt.text(0.6, 0.8-i*0.1, f'  {v_pos:.2f}V', transform=plt.gca().transAxes)

plt.axis('off')

plt.tight_layout()

# Save figure
output_filename = f'mzi_{TARGET_MZI}_detailed_analysis.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"\n=== Analysis Complete ===")
print(f"Detailed analysis figure saved as: {output_filename}")

# Phase-voltage relationship summary
reference_a = 0.1801
print(f"\n=== Comparison with Reference ===")
print(f"Reference a = {reference_a:.6f}")
print(f"Actual fitted a = {a:.6f}")
print(f"Deviation: {abs(a - reference_a)/reference_a*100:.1f}%")

plt.show()