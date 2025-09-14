import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from module.channel import SingleChannelFilter

class CNN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mzi_repeat_num, mzi_row_num, mzi_column_num):
        """
        Initialize CNN layer with multiple single channel filters
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of convolution kernel
        """
        super(CNN_layer, self).__init__()
        
        # Layer parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters_per_input = out_channels // in_channels
        self.flag = 0
        
        # Create filters for each input channel
        self.filters = nn.ModuleList([
            SingleChannelFilter(kernel_size=kernel_size,
                                repeat_num=mzi_repeat_num,
                                mzi_row_num =mzi_row_num,
                                mzi_column_num=mzi_column_num) 
            for _ in range(self.filters_per_input)
        ])
        
        # Initialize timing statistics
        self.timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}

    def get_all_time(self):
        """
        Recursively collect timing statistics from all filters
        
        Returns:
            dict: Dictionary containing accumulated timing statistics
        """
        total_stats = {'allocation': 0, 'computation': 0, 'total': 0}
        
        # Add current layer timing stats
        for key in total_stats:
            if hasattr(self, 'timing_stats'):
                total_stats[key] += self.timing_stats[key]
        
        # Recursively add all filter timing stats
        if hasattr(self, 'filters'):
            for layer in self.filters:
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
        Forward propagation through CNN layer
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor: Output tensor with transformed channels
        """
        # Get input dimensions
        batch_size, channels, height, width = x.shape
        
        # List to store outputs from each filter
        outputs = []
        
        # Process each input channel through all filters
        for i in range(self.in_channels):
            # Extract single channel input
            channel_input = x[:, i]  # Shape: (batch_size, height, width)
            
            # Apply each filter to the channel
            for j in range(self.filters_per_input):
                # Update filter flag if needed
                if self.flag == 1:
                    self.filters[j].flag = 1
                    
                # Apply filter and collect output
                outputs.append(self.filters[j](channel_input))
        
        # Stack outputs along channel dimension
        return torch.stack(outputs, dim=1)  # Shape: (batch_size, out_channels, new_height, new_width)
