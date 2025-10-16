import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_optical_power_data(csv_file_path):
    """
    Analyze optical power data and calculate arithmetic average

    Parameters:
    csv_file_path (str): Path to the CSV file containing time and power data

    Returns:
    dict: Dictionary containing analysis results
    """

    # Read the CSV file
    print("Reading data from CSV file...")
    df = pd.read_csv(csv_file_path)

    # Extract time and power data
    time_ms = df['time in ms'].values
    power_dBm = df['power in dBm'].values

    print(f"Data loaded successfully: {len(time_ms)} data points")
    print(f"Time range: {time_ms[0]:.1f} - {time_ms[-1]:.1f} ms")
    print(f"Power range: {power_dBm.min():.2e} - {power_dBm.max():.2e} W")

    # Calculate arithmetic average
    arithmetic_average = np.mean(power_dBm)

    # Calculate statistics
    power_std = np.std(power_dBm)
    power_min = np.min(power_dBm)
    power_max = np.max(power_dBm)
    cv = (power_std / arithmetic_average) * 100  # Coefficient of variation

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Subplot 1: Power vs Time
    plt.subplot(2, 1, 1)
    plt.plot(time_ms / 1000, power_dBm, 'b-', linewidth=1, alpha=0.7, label='Measured Data')
    plt.axhline(y=arithmetic_average, color='r', linestyle='--', linewidth=2,
                label=f'Arithmetic Average: {arithmetic_average:.3f} dBm')
    plt.xlabel('Time (s)')
    plt.ylabel('Optical Power (dBm)')
    plt.title('Optical Power vs Time with Arithmetic Average')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Power Distribution Histogram
    plt.subplot(2, 1, 2)
    plt.hist(power_dBm, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=arithmetic_average, color='r', linestyle='--', linewidth=2,
                label=f'Arithmetic Average: {arithmetic_average:.3f} dBm')
    plt.xlabel('Optical Power (dBm)')
    plt.ylabel('Frequency')
    plt.title('Optical Power Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Prepare results dictionary
    results = {
        'arithmetic_average_dBm': arithmetic_average,
        'standard_deviation_dBm': power_std,
        'minimum_power_dBm': power_min,
        'maximum_power_dBm': power_max,
        'coefficient_of_variation_percent': cv,
        'number_of_measurements': len(power_dBm),
        'total_measurement_time_s': (time_ms[-1] - time_ms[0]) / 1000,
        'average_sampling_interval_ms': np.mean(np.diff(time_ms))
    }

    # Print results
    print("\n" + "=" * 50)
    print("OPTICAL POWER ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Arithmetic Average:      {arithmetic_average:.3f} dBm")
    print(f"Standard Deviation:      {power_std:.3f} dBm")
    print(f"Coefficient of Variation: {cv:.2f}%")
    print(f"Power Range:             {power_min:.2f} - {power_max:.2f} dBm")
    print(f"Number of Measurements:  {len(power_dBm)}")
    print(f"Total Measurement Time:  {(time_ms[-1] - time_ms[0]) / 1000:.2f} s")
    print(f"Average Sampling Rate:   {1000 / np.mean(np.diff(time_ms)):.2f} Hz")

    return results


# Example usage:
if __name__ == "__main__":
    # Analyze the data
    results = analyze_optical_power_data('in0_mzi5_v_10.0_o8_dbm.csv')

    # Save results to CSV file
    results_df = pd.DataFrame([results])
    results_df.to_csv('optical_power_analysis_results_in0_mzi5_v_10.0_o8_dbm.csv', index=False)
    print(f"\nResults saved to: optical_power_analysis_results.csv")