import pandas as pd
import numpy as np
import os
import glob

# Define the directory containing the CSV files
csv_dir = "C:\\Users\\17958\\ONN-MZI\\mzi_raw_sin_thetas_output"

print(f"Searching for CSV files in: {csv_dir}")

# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

if not csv_files:
    print("No CSV files found to process.")
else:
    print(f"Found {len(csv_files)} CSV files. Starting conversion...")

    # Define the conversion function
    def convert_to_phase(raw_theta_value):
        # 1. Apply sigmoid function
        sigmoid_val = 1 / (1 + np.exp(-raw_theta_value))
        # 2. Scale to [-1, 1] to get the sin_theta value
        sin_theta = 2 * sigmoid_val - 1
        # 3. Apply arcsin to get the phase in radians
        # Clip the value to be strictly within [-1.0, 1.0] to avoid floating point inaccuracies with arcsin
        clipped_sin_theta = np.clip(sin_theta, -1.0, 1.0)
        phase_radians = np.arcsin(clipped_sin_theta)
        return phase_radians

    # Process each file
    for file_path in csv_files:
        try:
            # Read the CSV
            df = pd.read_csv(file_path)

            # Check if the required column exists
            if 'raw_sin_theta' in df.columns:
                # Apply the conversion to create the new columns
                df['phase_difference_radians'] = df['raw_sin_theta'].apply(convert_to_phase)
                df['phase_difference_degrees'] = np.rad2deg(df['phase_difference_radians'])
                
                # Save the updated DataFrame back to the same file
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f"Updated file with phase difference columns: {os.path.basename(file_path)}")
            else:
                print(f"Warning: 'raw_sin_theta' column not found in {os.path.basename(file_path)}. Skipping.")

        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")

    print("\nConversion complete.")
