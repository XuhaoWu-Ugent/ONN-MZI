import torch
import pandas as pd
import numpy as np
import json
import os
import glob

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

    if 'raw_sin_theta' not in df.columns:
        print(f"Error: CSV file {csv_path} is missing the required 'raw_sin_theta' column.")
        return False

    df_sorted = df.sort_values(by=['mzi_array_layer_in_filter', 'mzi_index_in_layer']).reset_index()

    mzi_params = [p for layer in model.layers for mzi in layer.MZI for p in mzi.parameters()]
    if len(df_sorted) != len(mzi_params):
        print(f"[Warning] Mismatch for {os.path.basename(csv_path)}: CSV has {len(df_sorted)} params, model has {len(mzi_params)}. Skipping file.")
        return False

    with torch.no_grad():
        for idx, row in df_sorted.iterrows():
            raw_sin_theta_val = row['raw_sin_theta']
            mzi_params[idx].fill_(raw_sin_theta_val)
            
    with torch.no_grad():
        model.diagonal_matrix.fill_(1.0)

    return True

def run_single_simulation(model, active_port, power=1.0):
    """
    Runs a single simulation for a given model and active port using the correct forward pass method.
    Returns the complex output vector and the output power vector.
    """
    # Create the base 10-element complex vector for the given active port
    input_vector = np.zeros(10, dtype=np.complex128)
    if 0 <= active_port < 10:
        input_vector[active_port] = np.sqrt(power)
    else:
        return None, None

    # Construct the minimal 3x3 dummy image for the model's forward pass
    kernel_size = model.kernel_size
    num_patch_elements = kernel_size * kernel_size
    input_patch_real = np.real(input_vector[:num_patch_elements])
    dummy_image = torch.tensor(input_patch_real, dtype=torch.float32).view(1, kernel_size, kernel_size)

    # Get the intermediate processed patch from the model's forward pass
    with torch.no_grad():
        _, _, processed_patch = model(dummy_image, return_intermediate=True)
    
    output_vector = processed_patch.numpy().flatten()
    output_powers = np.abs(output_vector)**2
    
    return output_vector, output_powers

if __name__ == '__main__':
    base_path = r'C:\Users\17958\ONN-MZI'
    file_pattern = os.path.join(base_path, 'mzi_raw_sin_thetas_output', 'layer_[0-3]_*.csv')
    filter_files = glob.glob(file_pattern)

    if not filter_files:
        print("Error: No filter files found for layers 0-3.")
        exit()

    print(f"Found {len(filter_files)} filter files to process.")

    model = SingleChannelFilter(mzi_row_num=5, mzi_column_num=4, repeat_num=5, kernel_size=3)
    model.eval()

    all_results = {}
    total_simulations = len(filter_files) * 10
    current_sim = 0

    for f_path in sorted(filter_files):
        f_name = os.path.basename(f_path).replace("_raw_sin_thetas.csv", "")
        print(f"\nProcessing filter: {f_name}...")
        
        if not load_parameters_from_csv(model, f_path):
            continue

        filter_results = {}
        for port in range(10):
            current_sim += 1
            print(f"  - Simulating input port {port} ({current_sim}/{total_simulations})", end='\r')
            
            output_vector, output_powers = run_single_simulation(model, active_port=port)
            
            if output_powers is not None:
                complex_amps_str = [str(c) for c in output_vector]
                filter_results[f'input_{port}'] = {
                    'output_powers': output_powers.tolist(),
                    'output_complex_amplitudes': complex_amps_str
                }
        
        all_results[f_name] = filter_results
        print(f"\nCompleted simulations for {f_name}")

    output_filename = os.path.join(base_path, 'batch_simulation_results.json')
    with open(output_filename, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n\nBatch simulation complete. All results saved to:\n{output_filename}")