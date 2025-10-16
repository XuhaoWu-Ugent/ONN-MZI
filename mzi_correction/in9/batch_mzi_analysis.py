
'''
This script batch processes MZI test data.

It recursively finds all CSV files in the specified directory, runs an analysis
on each to calculate optical power statistics, generates plots, and saves
both individual and summary results.
'''
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_optical_power_data(csv_file_path, plots_dir, results_csv_dir):
    """
    Analyzes optical power data from a single CSV file, saves analysis results
    to a new CSV file, and generates a plot saved as a PNG file.

    Parameters:
    csv_file_path (str): Path to the input CSV file.
    plots_dir (str): Directory to save the output plot.
    results_csv_dir (str): Directory to save the output CSV results.

    Returns:
    dict: A dictionary containing the analysis results, or None if processing fails.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Check for required columns
        if 'time in ms' not in df.columns or 'power in W' not in df.columns:
            print(f"  - Skipping {csv_file_path}: missing required columns.")
            return None

        # Extract time and power data
        time_ms = df['time in ms'].values
        power_W = df['power in W'].values

        if len(time_ms) < 2:
            print(f"  - Skipping {csv_file_path}: not enough data points.")
            return None

        # --- Calculations ---
        arithmetic_average = np.mean(power_W)
        power_std = np.std(power_W)
        power_min = np.min(power_W)
        power_max = np.max(power_W)
        cv = (power_std / arithmetic_average) * 100 if arithmetic_average != 0 else 0

        # --- Visualization ---
        plt.figure(figsize=(12, 8))

        # Subplot 1: Power vs Time
        plt.subplot(2, 1, 1)
        plt.plot(time_ms / 1000, power_W * 1e6, 'b-', linewidth=1, alpha=0.7, label='Measured Data')
        plt.axhline(y=arithmetic_average * 1e6, color='r', linestyle='--', linewidth=2,
                    label=f'Arithmetic Average: {arithmetic_average * 1e6:.3f} uW')
        plt.xlabel('Time (s)')
        plt.ylabel('Optical Power (uW)')
        plt.title(f'Optical Power vs Time: {os.path.basename(csv_file_path)}')
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

        # Save plot to file
        plot_filename = os.path.basename(csv_file_path).replace('.csv', '.png')
        plot_save_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_save_path)
        plt.close()  # Close the figure to free up memory

        # --- Prepare results ---
        results = {
            'source_file': csv_file_path,
            'arithmetic_average_W': arithmetic_average,
            'standard_deviation_W': power_std,
            'minimum_power_W': power_min,
            'maximum_power_W': power_max,
            'coefficient_of_variation_percent': cv,
            'number_of_measurements': len(power_W),
            'total_measurement_time_s': (time_ms[-1] - time_ms[0]) / 1000,
            'average_sampling_interval_ms': np.mean(np.diff(time_ms))
        }

        # Save individual result to CSV
        result_filename = 'analysis_' + os.path.basename(csv_file_path)
        result_save_path = os.path.join(results_csv_dir, result_filename)
        pd.DataFrame([results]).to_csv(result_save_path, index=False)

        return results

    except Exception as e:
        print(f"  - Error processing {csv_file_path}: {e}")
        return None

def batch_process_mzi_data():
    """
    Finds all MZI data files, processes them, and saves a summary of the results.
    """
    # Setup output directories
    plots_dir = 'analysis_plots'
    results_csv_dir = 'analysis_csv_results'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_csv_dir, exist_ok=True)

    # Find all CSV files recursively, excluding previous analysis results
    print("Searching for data files...")
    all_files = glob.glob('**/*.csv', recursive=True)
    data_files = [f for f in all_files if not os.path.basename(f).startswith(('analysis_', 'summary_', 'optical_power_analysis_results'))]

    if not data_files:
        print("No data files found to process. Please check file paths and naming conventions.")
        return

    print(f"Found {len(data_files)} data files to process.")

    all_results = []
    for i, file_path in enumerate(data_files):
        print(f"Processing file {i + 1}/{len(data_files)}: {file_path}")
        results = analyze_optical_power_data(file_path, plots_dir, results_csv_dir)
        if results:
            all_results.append(results)

    if not all_results:
        print("No data could be analyzed.")
        return

    # Create and save a summary DataFrame
    summary_df = pd.DataFrame(all_results)
    # Reorder columns for better readability
    cols = ['source_file'] + [col for col in summary_df.columns if col != 'source_file']
    summary_df = summary_df[cols]

    summary_path = 'summary_analysis_results.csv'
    summary_df.to_csv(summary_path, index=False)

    print(f"\nBatch processing complete.")
    print(f"Individual plots saved in: '{plots_dir}'")
    print(f"Individual CSV results saved in: '{results_csv_dir}'")
    print(f"Summary of all results saved to: '{summary_path}'")


if __name__ == "__main__":
    batch_process_mzi_data()
