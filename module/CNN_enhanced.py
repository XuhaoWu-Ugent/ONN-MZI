import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from module.channel import SingleChannelFilter
from module.channel_power import SingleChannelFilterPower


class CNN_layer_enhanced(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mzi_repeat_num, mzi_row_num, mzi_column_num,
                 filter_type='coherent'):
        """
        Initialize enhanced CNN layer with multiple single channel filters supporting different detection modes

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of convolution kernel
            mzi_repeat_num (int): Number of MZI repetitions
            mzi_row_num (int): Number of MZIs per row
            mzi_column_num (int): Number of MZIs per column
            filter_type (str): Type of filter - 'coherent' (amplitude interference) or 'power' (power superposition)
        """
        super(CNN_layer_enhanced, self).__init__()

        # Layer parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters_per_input = out_channels // in_channels
        self.filter_type = filter_type
        self.flag = 0

        # Select filter class based on type
        if filter_type == 'coherent':
            filter_class = SingleChannelFilter
        elif filter_type == 'power':
            filter_class = SingleChannelFilterPower
        else:
            raise ValueError(f"Unknown filter type: {filter_type}. Choose from 'coherent' or 'power'")

        # Create common filter arguments
        filter_args = {
            'kernel_size': kernel_size,
            'repeat_num': mzi_repeat_num,
            'mzi_row_num': mzi_row_num,
            'mzi_column_num': mzi_column_num
        }

        # Create filters for each input channel
        self.filters = nn.ModuleList([
            filter_class(**filter_args)
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
        Forward propagation through enhanced CNN layer

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