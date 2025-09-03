import numpy as np

# 定义要加载的文件名
# 假设 'collected_filter_simulation_data.npy' 文件与此脚本在同一目录下
file_path = 'C:\\Users\\17958\\ONN-MZI\\filter_data\\hook_data_verified.npy'

# 设置打印选项，以便完整显示数组（如果需要，但要注意大型数组的输出）
# np.set_printoptions(threshold=np.inf) # 取消注释这行可以尝试打印完整的大型数组

try:
    # 加载 .npy 文件
    data_loaded = np.load(file_path, allow_pickle=True)

    print(f"成功加载文件: {file_path}")
    print(f"加载后顶层数据类型: {type(data_loaded)}")

    if isinstance(data_loaded, np.ndarray) and data_loaded.ndim == 0 and data_loaded.dtype == 'object':
        actual_data = data_loaded.item()
        print(f"提取后的实际数据类型: {type(actual_data)}")
    else:
        actual_data = data_loaded

    if isinstance(actual_data, dict):
        print("\n文件内容是一个字典。结构及具体内容如下：")

        for key, value in actual_data.items():
            print(f"\n键: '{key}'")
            if isinstance(value, np.ndarray):
                print(f"  类型: NumPy 数组 (numpy.ndarray)")
                print(f"  形状 (shape): {value.shape}")
                print(f"  数据类型 (dtype): {value.dtype}")
                print(f"  值:\n{value}")  # 打印数组的完整内容
            elif isinstance(value, dict):
                print(f"  类型: 字典 (dict)")
                print(f"  包含的键数量: {len(value.keys())}")
                print(f"  其内部的键 (示例前5个): {list(value.keys())[:5]}")

                if key == 'filter_internals':
                    print(f"    详细查看 '{key}' 内部的每个滤波器数据:")
                    for filter_key, filter_data in value.items():
                        print(f"\n      滤波器键: '{filter_key}'")
                        if isinstance(filter_data, dict):
                            for sub_key, sub_value in filter_data.items():
                                print(f"        子键: '{sub_key}'")
                                if isinstance(sub_value, np.ndarray):
                                    print(f"          类型: NumPy 数组")
                                    print(f"          形状: {sub_value.shape}")
                                    print(f"          数据类型: {sub_value.dtype}")
                                    print(f"          值:\n{sub_value}")  # 打印子数组的完整内容
                                elif isinstance(sub_value, list):
                                    print(f"          类型: 列表")
                                    print(f"          长度: {len(sub_value)}")
                                    if len(sub_value) > 0:
                                        print(f"          列表第一个元素类型: {type(sub_value[0])}")
                                    print(f"          列表内容:\n{sub_value}")  # 打印列表的完整内容
                                else:
                                    print(f"          类型: {type(sub_value)}")
                                    print(f"          值: {sub_value}")
                        else:
                            print(f"      '{filter_key}' 的值不是预期的字典格式，类型为: {type(filter_data)}")
            elif isinstance(value, list):
                print(f"  类型: 列表 (list)")
                print(f"  长度: {len(value)}")
                if len(value) > 0:
                    print(f"  列表第一个元素的类型: {type(value[0])}")
                print(f"  列表内容:\n{value}")  # 打印列表的完整内容
            else:
                print(f"  类型: {type(value)}")
                print(f"  值: {value}")
    else:
        print("\n加载的数据不是预期的字典格式。原始加载数据如下：")
        print(actual_data)
        if isinstance(actual_data, np.ndarray):
            print(f"  形状 (shape): {actual_data.shape}")
            print(f"  数据类型 (dtype): {actual_data.dtype}")
            print(f"  值:\n{actual_data}")


except FileNotFoundError:
    print(f"错误：文件 '{file_path}' 未找到。请确保文件路径正确，或者文件与脚本在同一目录下。")
except Exception as e:
    print(f"加载或处理文件时发生错误: {e}")
