import numpy as np
import os

# Define the filename to load
# Assume 'hook_data_verified.npy' is in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'hook_data_verified.npy')

# Set print options to display the full array (if needed, but be careful with large arrays)
# np.set_printoptions(threshold=np.inf) # Uncomment this line to try printing a complete large array

try:
    # Load the .npy file
    data_loaded = np.load(file_path, allow_pickle=True)

    print(f"Successfully loaded file: {file_path}")
    print(f"Top-level data type after loading: {type(data_loaded)}")

    if isinstance(data_loaded, np.ndarray) and data_loaded.ndim == 0 and data_loaded.dtype == 'object':
        actual_data = data_loaded.item()
        print(f"Actual data type after extraction: {type(actual_data)}")
    else:
        actual_data = data_loaded

    if isinstance(actual_data, dict):
        print("\nFile content is a dictionary. Structure and content are as follows:")

        for key, value in actual_data.items():
            print(f"\nKey: '{key}'")
            if isinstance(value, np.ndarray):
                print(f"  Type: NumPy array (numpy.ndarray)")
                print(f"  Shape: {value.shape}")
                print(f"  Data type (dtype): {value.dtype}")
                print(f"  Value:\n{value}")  # Print the full content of the array
            elif isinstance(value, dict):
                print(f"  Type: Dictionary (dict)")
                print(f"  Number of keys: {len(value.keys())}")
                print(f"  Internal keys (first 5 as example): {list(value.keys())[:5]}")

                if key == 'filter_internals':
                    print(f"    Detailed view of each filter data inside '{key}':")
                    for filter_key, filter_data in value.items():
                        print(f"\n      Filter key: '{filter_key}'")
                        if isinstance(filter_data, dict):
                            for sub_key, sub_value in filter_data.items():
                                print(f"        Sub-key: '{sub_key}'")
                                if isinstance(sub_value, np.ndarray):
                                    print(f"          Type: NumPy array")
                                    print(f"          Shape: {sub_value.shape}")
                                    print(f"          Data type: {sub_value.dtype}")
                                    print(f"          Value:\n{sub_value}")  # Print the full content of the sub-array
                                elif isinstance(sub_value, list):
                                    print(f"          Type: List")
                                    print(f"          Length: {len(sub_value)}")
                                    if len(sub_value) > 0:
                                        print(f"          Type of first element in list: {type(sub_value[0])}")
                                    print(f"          List content:\n{sub_value}")  # Print the full content of the list
                                else:
                                    print(f"          Type: {type(sub_value)}")
                                    print(f"          Value: {sub_value}")
                        else:
                            print(f"      The value of '{filter_key}' is not in the expected dictionary format, type is: {type(filter_data)}")
            elif isinstance(value, list):
                print(f"  Type: List (list)")
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    print(f"  Type of first element in list: {type(value[0])}")
                print(f"  List content:\n{value}")  # Print the full content of the list
            else:
                print(f"  Type: {type(value)}")
                print(f"  Value: {value}")
    else:
        print("\nLoaded data is not in the expected dictionary format. Raw loaded data is as follows:")
        print(actual_data)
        if isinstance(actual_data, np.ndarray):
            print(f"  Shape: {actual_data.shape}")
            print(f"  Data type (dtype): {actual_data.dtype}")
            print(f"  Value:\n{actual_data}")


except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please ensure the file path is correct.")
except Exception as e:
    print(f"An error occurred while loading or processing the file: {e}")