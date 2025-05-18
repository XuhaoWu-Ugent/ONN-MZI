# Required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.MZI_array.mzi import MZI

class MZIlayer_row(nn.Module):
    def __init__(self, num):
        """Initialize a row of MZI (Mach-Zehnder Interferometer) modules"""
        super().__init__()
        # Create a list of 5 MZI modules
        self.MZI = nn.ModuleList([MZI() for _ in range(num)])  # n MZIs per row(default 5)
        # Flag for debugging/tracking purposes
        self.flag = 0
        # Dictionary to store timing statistics
        self.timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}
        
    def get_all_time(self):
        """
        Recursively collect timing statistics from all MZI modules
        Returns:
            Dictionary containing accumulated timing statistics
        """
        total_stats = {'allocation': 0, 'computation': 0, 'total': 0}
        
        # Add current layer's timing stats
        for key in total_stats:
            if hasattr(self, 'timing_stats'):
                total_stats[key] += self.timing_stats[key]
        
        # Recursively add timing stats from all child MZI modules
        if hasattr(self, 'MZI'):
            for layer in self.MZI:
                # If child has get_all_time method, call it
                if hasattr(layer, 'get_all_time'):
                    layer_stats = layer.get_all_time()
                # Otherwise try to get timing_stats directly
                elif hasattr(layer, 'timing_stats'):
                    layer_stats = layer.timing_stats
                else:
                    continue
                    
                # Accumulate child layer timing stats
                for key in total_stats:
                    total_stats[key] += layer_stats[key]
        
        return total_stats
        
    def forward(self, input):
        """
        Forward pass through the MZI row
        Args:
            input: Input tensor of shape (batch_size, height, width, 10)
        Returns:
            Transformed tensor of shape (batch_size, height, width, 10)
        """
        # Extract input dimensions
        batch_size, height, width, _ = input.shape
        MZI_num = len(self.MZI)
        
        # Reshape input to match MZI structure (batch_size, height, width, MZI_num, 2)
        input_reshaped = input.view(batch_size, height, width, MZI_num, 2)
        
        # Prepare full 4-element input for each MZI by padding with zeros
        full_input = torch.zeros(batch_size, height, width, MZI_num, 4, 
                                 dtype=torch.complex64, device=input.device)
        full_input[..., 2:] = input_reshaped  # 填充最后两个位置为 reshaped 的输入
        
        # 初始化输出列表
        outputs = []
        for i, mzi in enumerate(self.MZI):
            # 选择当前 MZI 的输入 (batch_size, height, width, 4)
            mzi_input = full_input[:, :, :, i, :]
            
            # 展平输入以便传递给 MZI (batch_size * height * width, 4)
            mzi_input_flat = mzi_input.view(-1, 4)
            
            # 通过 MZI 处理并恢复形状 (batch_size, height, width, 4)
            mzi_output = mzi(mzi_input_flat).view(batch_size, height, width, 4)
            
            # 只保留前两个元素 (batch_size, height, width, 2)
            outputs.append(mzi_output[..., :2])
        
        # 堆叠所有 MZI 的输出 (batch_size, height, width, MZI_num, 2)
        output = torch.stack(outputs, dim=3)
        
        # 最终调整输出形状为 (batch_size, height, width, 10)
        return output.view(batch_size, height, width, -1)

