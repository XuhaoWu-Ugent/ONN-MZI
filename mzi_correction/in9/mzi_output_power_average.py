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
    power_W = df['power in W'].values

    print(f"Data loaded successfully: {len(time_ms)} data points")
    print(f"Time range: {time_ms[0]:.1f} - {time_ms[-1]:.1f} ms")
    print(f"Power range: {power_W.min():.2e} - {power_W.max():.2e} W")

    # Calculate arithmetic average
    arithmetic_average = np.mean(power_W)

    # Calculate statistics
    power_std = np.std(power_W)
    power_min = np.min(power_W)
    power_max = np.max(power_W)
    cv = (power_std / arithmetic_average) * 100  # Coefficient of variation

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Subplot 1: Power vs Time
    plt.subplot(2, 1, 1)
    plt.plot(time_ms / 1000, power_W * 1e6, 'b-', linewidth=1, alpha=0.7, label='Measured Data')
    plt.axhline(y=arithmetic_average * 1e6, color='r', linestyle='--', linewidth=2,
                label=f'Arithmetic Average: {arithmetic_average * 1e6:.3f} uW')
    plt.xlabel('Time (s)')
    plt.ylabel('Optical Power (uW)')
    plt.title('Optical Power vs Time with Arithmetic Average')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Power Distribution Histogram
    plt.subplot(2, 1, 2)
    plt.hist(power_W * 1e6, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=arithmetic_average * 1e6, color='r', linestyle='--', linewidth=2,
                label=f'Arithmetic Average: {arithmetic_average * 1e6:.3f} uW')
    plt.xlabel('Optical Power (uW)')
    plt.ylabel('Frequency')
    plt.title('Optical Power Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Prepare results dictionary
    results = {
        'arithmetic_average_W': arithmetic_average,
        'standard_deviation_W': power_std,
        'minimum_power_W': power_min,
        'maximum_power_W': power_max,
        'coefficient_of_variation_percent': cv,
        'number_of_measurements': len(power_W),
        'total_measurement_time_s': (time_ms[-1] - time_ms[0]) / 1000,
        'average_sampling_interval_ms': np.mean(np.diff(time_ms))
    }

    # Print results
    print("\n" + "=" * 50)
    print("OPTICAL POWER ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Arithmetic Average:      {arithmetic_average:.6e} W ({arithmetic_average * 1e6:.3f} uW)")
    print(f"Standard Deviation:      {power_std:.6e} W ({power_std * 1e6:.3f} uW)")
    print(f"Coefficient of Variation: {cv:.2f}%")
    print(f"Power Range:             {power_min * 1e6:.2f} - {power_max * 1e6:.2f} uW")
    print(f"Number of Measurements:  {len(power_W)}")
    print(f"Total Measurement Time:  {(time_ms[-1] - time_ms[0]) / 1000:.2f} s")
    print(f"Average Sampling Rate:   {1000 / np.mean(np.diff(time_ms)):.2f} Hz")

    return results


# Example usage:
if __name__ == "__main__":
    # Analyze the data
    results = analyze_optical_power_data('./mzi6/in0_mzi6_v_0.0_o8.csv')

    # Save results to CSV file
    results_df = pd.DataFrame([results])
    results_df.to_csv('optical_power_analysis_results_in0_mzi6_v_0.0_o8.csv', index=False)
    print(f"\nResults saved to: optical_power_analysis_results.csv")