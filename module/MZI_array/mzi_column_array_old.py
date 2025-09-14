# Required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.MZI_array.mzi import MZI


class MZIlayer_column(nn.Module):
    def __init__(self,num):
        """Initialize a column of MZI (Mach-Zehnder Interferometer) modules"""
        super(MZIlayer_column, self).__init__()
        # Create a list of n-1 MZI modules
        self.MZI = nn.ModuleList([MZI() for _ in range(num)])  # n-1 MZIs per column(default n=5)
        # Flag for debugging/tracking purposes
        self.flag = 0
        self.num=num
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
        Forward pass through the MZI column
        Args:
            input: Input tensor of shape (batch_size, num_patches, 10)
        Returns:
            Transformed tensor of shape (batch_size, num_patches, 10)
        """
        # Extract input dimensions
        batch_size, num_patches, _ = input.shape
        last_input_num=self.num*2+1
        # Initialize output tensor, preserving first and last elements
        output = torch.zeros_like(input)
        output[:, :, 0] = input[:, :, 0]   # Preserve first element
        output[:, :, last_input_num] = input[:, :, last_input_num]   # Preserve last element

        # Process middle n*2 elements
        mid_input = input[:, :, 1:last_input_num]  # Shape: (batch_size, num_patches, last_input_num-1)
        # Reshape for MZI processing
        mid_input_reshaped = mid_input.view(batch_size, num_patches, self.num, 2)  
        
        # Process through n MZIs
        mzi_outputs = []
        for i, mzi in enumerate(self.MZI):
            # Get input pair for current MZI
            mzi_input = mid_input_reshaped[:, :, i, :]  # Shape: (batch_size, num_patches, 2)
            
            # Prepare 4-port input for MZI
            full_input = torch.zeros((batch_size, num_patches, 4), 
                                   dtype=torch.complex64, device=input.device)
            full_input[:, :, 0] = mzi_input[:, :, 0]  # First port
            full_input[:, :, 2] = mzi_input[:, :, 1]  # Third port
            
            # Process through MZI
            mzi_output = mzi(full_input.view(-1, 4)).view(batch_size, num_patches, 4)

            # Extract and swap second and fourth port outputs
            y2 = mzi_output[:, :, 1]  # Second port output
            y4 = mzi_output[:, :, 3]  # Fourth port output
            swapped = torch.stack([y4, y2], dim=-1)  # Swap outputs
            mzi_outputs.append(swapped)
            
            # Debugging gradient print (commented out)
            # if self.flag==1:
            #     print(mzi.raw_sin_theta.grad)
            
        # Concatenate all MZI outputs
        mzi_outputs_tensor = torch.cat(mzi_outputs, dim=2)  # Shape: (batch_size, num_patches, num*2)

        # Place processed middle section back into output tensor
        output[:, :, 1:last_input_num] = mzi_outputs_tensor

        return output  # Shape: (batch_size, num_patches, num*2+2)
