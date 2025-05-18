import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.MZI_array.mzi_column_array import *
from module.MZI_array.mzi_row_array import *


class SingleChannelFilter(nn.Module):
    def __init__(self, mzi_row_num=5 ,mzi_column_num=4, repeat_num=5, kernel_size=3):
        """
        Initialize single channel filter module
        
        Args:
            repeat_num (int): Number of MZI layer repetitions
            kernel_size (int): Size of the convolution kernel
        """
        super(SingleChannelFilter, self).__init__()
        
        # Create alternating row and column MZI layers
        self.layers = nn.ModuleList()
        for _ in range(repeat_num):
            self.layers.append(MZIlayer_row(num=mzi_row_num))
            self.layers.append(MZIlayer_column(num=mzi_column_num))
        self.layers.append(MZIlayer_row(num=mzi_row_num))
        
        # Trainable diagonal weight matrix
        self.diagonal_matrix = nn.Parameter(torch.randn(mzi_row_num*2))
        
        # Other parameters
        self.kernel_size = kernel_size
        self.flag = 0
        self.timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}

    def get_all_time(self):
        """
        Recursively collect timing statistics from all layers
        
        Returns:
            dict: Dictionary containing accumulated timing statistics
        """
        total_stats = {'allocation': 0, 'computation': 0, 'total': 0}
        
        # Add current layer timing stats
        for key in total_stats:
            if hasattr(self, 'timing_stats'):
                total_stats[key] += self.timing_stats[key]
        
        # Recursively add all sublayer timing stats
        if hasattr(self, 'layers'):
            for layer in self.layers:
                if hasattr(layer, 'get_all_time'):
                    layer_stats = layer.get_all_time()
                elif hasattr(layer, 'timing_stats'):
                    layer_stats = layer.timing_stats
                else:
                    continue
                    
                for key in total_stats:
                    total_stats[key] += layer_stats[key]
        
        return total_stats

    def forward(self, x):
        """
        Forward propagation function
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, height, width)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, output_height, output_width)
        """
        # Get input dimensions
        batch_size, height, width = x.shape
        
        # Calculate output dimensions
        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1
        
        # Extract image patches
        # Add channel dimension and unfold into patches
        patches = F.unfold(x.unsqueeze(1), 
                          kernel_size=self.kernel_size, 
                          stride=1)  # Shape: (batch_size, kernel_size*kernel_size, num_patches)
        
        # Adjust dimension order
        patches = patches.permute(0, 2, 1)  # Shape: (batch_size, num_patches, kernel_size*kernel_size)
        
        # Pad to 10 dimensions to match MZI layers
        patches = F.pad(patches, (0, 1))  # Shape: (batch_size, num_patches, 10)
        
        # Convert to complex type for complex operations
        patches = patches.to(torch.complex64)
        
        # Pass through sequence of MZI layers
        for layer in self.layers:
            if self.flag == 1:
                layer.flag = 1
            patches = layer(patches)
        
        # Apply diagonal weight matrix and sum
        weighted_output = patches * self.diagonal_matrix.view(1, 1, -1)
        
        # Reshape output to desired spatial dimensions
        output = weighted_output.sum(dim=2).view(batch_size, output_height, output_width)
        
        return output
