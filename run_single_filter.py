import torch
import pandas as pd
import numpy as np
import argparse
import os
import json

# --- Import necessary modules from your project ---
from module.channel import SingleChannelFilter



def load_parameters_from_csv(model, csv_path):
    """
    Loads MZI parameters from a CSV file into the model.
    This corrected version reads the 'raw_sin_theta' column directly.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Cannot find filter CSV file at {csv_path}")
        return False

    # Check if the required column exists
    if 'raw_sin_theta' not in df.columns:
        print(f"Error: CSV file {csv_path} is missing the required 'raw_sin_theta' column.")
        return False

    df_sorted = df.sort_values(by=['mzi_array_layer_in_filter', 'mzi_index_in_layer']).reset_index()

    # --- Load MZI parameters ---
    mzi_params = [p for layer in model.layers for mzi in layer.MZI for p in mzi.parameters()]
    if len(df_sorted) != len(mzi_params):
        print(f"Error: Mismatch between MZI parameters in CSV ({len(df_sorted)}) and model ({len(mzi_params)}).")
        return False

    with torch.no_grad():
        for idx, row in df_sorted.iterrows():
            # Directly read the raw_sin_theta value from the CSV
            raw_sin_theta_val = row['raw_sin_theta']
            mzi_params[idx].fill_(raw_sin_theta_val)
            
    print(f"Successfully loaded {len(mzi_params)} MZI parameters from '{os.path.basename(csv_path)}' using the 'raw_sin_theta' column.")
    
    # Set diagonal_matrix to all ones for pure MZI analysis
    with torch.no_grad():
        model.diagonal_matrix.fill_(1.0)
    # print("Set diagonal_matrix to all ones for direct MZI output analysis.")

    return True

def run_simulation(filter_csv, input_vector_str):
    """
    Main simulation function.
    """
    model = SingleChannelFilter(mzi_row_num=5, mzi_column_num=4, repeat_num=5, kernel_size=3)
    model.eval()

    if not load_parameters_from_csv(model, filter_csv):
        return

    try:
        input_list_of_lists = json.loads(input_vector_str)
        if len(input_list_of_lists) != 10:
            raise ValueError("Input vector must have 10 elements.")
        input_vector = np.array([c[0] + 1j * c[1] for c in input_list_of_lists], dtype=np.complex128)
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        print(f"Error: Invalid format for input vector. {e}")
        print("Please provide a JSON array of 10 [real, imag] pairs, e.g., '[[1,0], [0,0], ...]'")
        return
    
    kernel_size = model.kernel_size
    num_patch_elements = kernel_size * kernel_size

    input_patch_real = np.real(input_vector[:num_patch_elements])
    dummy_image = torch.tensor(input_patch_real, dtype=torch.float32).view(1, kernel_size, kernel_size)

    total_input_power = np.sum(np.abs(input_vector)**2)

    print(f"\nSimulating with input vector: {input_vector_str}...")
    # print(f"Constructed a ({dummy_image.shape[1]}x{dummy_image.shape[2]}) dummy image for forward pass.")

    with torch.no_grad():
        final_output, returned_input_patch, processed_patch = model(dummy_image, return_intermediate=True)

    output_vector = processed_patch.numpy().flatten()
    output_powers = np.abs(output_vector)**2
    total_output_power = np.sum(output_powers)

    print("--- Simulation Results (from intermediate hook) ---")
    for i in range(10):
        print(f"Output Port {i} Power: {output_powers[i]:.8f}, Complex Amp: {output_vector[i]}")
    print("--------------------------")
    print(f"Total Input Power:  {total_input_power:.8f}")
    print(f"Total Output Power: {total_output_power:.8f}")
    
    if not np.isclose(total_input_power, total_output_power):
        print("Warning: Power is not conserved!")
    else:
        print("Power successfully conserved.")

if __name__ == '__main__':
    csv_file_to_test = "C:\\Users\\17958\\ONN-MZI\\mzi_raw_sin_thetas_output\\layer_3_filter_0_raw_sin_thetas.csv"
    input_vector_to_test = "[[0.001653542509302497, 0], [0.0018996569560840726, 0], [0.001801564241759479, 0], [0.0027509424835443497, 0], [0.0031814996618777514, 0], [0.003051634645089507, 0], [0.0029122873675078154, 0], [0.0032277426216751337, 0], [0.003886190941557288, 0], [0.0, 0]]"
    
    print("--- Running with hard-coded parameters found in the script ---")
    run_simulation(csv_file_to_test, input_vector_to_test)