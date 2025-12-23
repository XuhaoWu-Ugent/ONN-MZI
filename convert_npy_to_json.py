import numpy as np
import json
import os

class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy types.
    Converts numpy arrays to lists, and complex numbers to strings.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (complex, np.complex_, np.complex64, np.complex128)):
            return str(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        return super(NumpyJSONEncoder, self).default(obj)

def convert_npy_to_json(npy_path, json_path):
    """
    Loads data from a .npy file and saves it as a human-readable .json file.
    """
    print(f"Loading data from: {npy_path}")
    try:
        # allow_pickle=True is required for loading object arrays
        data = np.load(npy_path, allow_pickle=True).item()
        print("Successfully loaded .npy file.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {npy_path}")
        return
    except Exception as e:
        print(f"An error occurred while loading the .npy file: {e}")
        return

    print(f"Converting and saving data to: {json_path}")
    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, cls=NumpyJSONEncoder, indent=4)
        print("Successfully created .json file.")
    except Exception as e:
        print(f"An error occurred while saving the .json file: {e}")
        return

if __name__ == '__main__':
    # Define the input and output file paths relative to this script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # input_npy_file = os.path.join(base_dir, 'filter_data', 'hook_data_verified.npy')
    # output_json_file = os.path.join(base_dir, 'filter_data', 'hook_data_verified.json')
    
    input_npy_file = os.path.join(base_dir, 'results', 'mzi_hardware_data_K15.npy')
    output_json_file = os.path.join(base_dir, 'results', 'mzi_hardware_data_K15.json')
    convert_npy_to_json(input_npy_file, output_json_file)
