
'''
This script generates a summary file from MZI test data based on a specific format.

It finds all CSV files in mzi* subdirectories, parses parameters from the filenames,
calculates the average power for each, and writes the formatted output to a single text file.
'''
import os
import glob
import pandas as pd
import numpy as np
import re

def parse_filename(file_path):
    """
    Parses the filename and path to extract MZI parameters.
    """
    path_for_regex = file_path.replace('\\', '/')
    filename = os.path.basename(path_for_regex)
    
    params = {
        'input_port': None,
        'mzi_number': None,
        'voltage': None,
        'output_port': None
    }

    # Extract MZI number from path or filename
    mzi_match = re.search(r'mzi(\d+)', path_for_regex, re.IGNORECASE)
    if mzi_match:
        params['mzi_number'] = int(mzi_match.group(1))

    # Extract input port
    in_match = re.search(r'in(\d+)', filename, re.IGNORECASE)
    if in_match:
        params['input_port'] = int(in_match.group(1))

    # Extract voltage
    v_match = re.search(r'v_(\d+\.?\d*)', filename, re.IGNORECASE)
    if v_match:
        params['voltage'] = float(v_match.group(1))

    # Extract output port
    o_match = re.search(r'o(\d+)', filename, re.IGNORECASE)
    if o_match:
        params['output_port'] = int(o_match.group(1))

    return params

def get_average_power(csv_file_path):
    """
    Calculates the arithmetic average power from a CSV file.
    Returns None if the file cannot be processed.
    """
    try:
        df = pd.read_csv(csv_file_path)
        if 'power in W' in df.columns and not df['power in W'].empty:
            power_series = pd.to_numeric(df['power in W'], errors='coerce')
            power_series.dropna(inplace=True)
            if not power_series.empty:
                return np.mean(power_series.values)
    except Exception as e:
        print(f"  - Could not read or process {csv_file_path}: {e}")
        return None
    return None

def generate_output_file(input_power, output_filename="MZI_array_output.txt"):
    """
    Generates the final output file in the specified format.
    """
    # Find all data files in mzi* directories
    data_files = glob.glob('mzi*/**/*.csv', recursive=True)
    
    # Also include CSVs in the root directory that seem to be data files
    root_files = glob.glob('in5_mzi*.csv') + glob.glob('mzi_in5*.csv')
    data_files.extend(root_files)
    data_files = sorted(list(set(data_files)))

    if not data_files:
        print("No data files found.")
        return

    output_lines = ["## input_port input_power input_mzi_number input_mzi_voltage output_port output_power\n"]

    print(f"Found {len(data_files)} files to process.")
    skipped_count = 0
    for file_path in data_files:
        params = parse_filename(file_path)
        avg_power = get_average_power(file_path)

        # Check if all parameters were found and power was calculated
        if all(params.get(k) is not None for k in ['input_port', 'mzi_number', 'voltage', 'output_port']) and avg_power is not None:
            line = (
                f"{params['input_port']} {input_power:.1e} {params['mzi_number']} "
                f"{params['voltage']} {params['output_port']} {avg_power:.2e}\n"
            )
            output_lines.append(line)
        else:
            skipped_count += 1
            # print(f"Skipping file (could not parse all parameters or calculate power): {file_path}")

    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} files due to parsing errors or lack of data.")

    # Write to output file
    with open(output_filename, 'w') as f:
        f.writelines(output_lines)

    print(f"\nProcessing complete. Output saved to '{output_filename}'")

if __name__ == "__main__":
    # This value is based on the user's example.
    INPUT_POWER = 5e-3 
    generate_output_file(INPUT_POWER)
