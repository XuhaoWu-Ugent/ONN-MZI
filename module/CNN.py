import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from module.channel import SingleChannelFilter


class CNN_layer(nn.Module):
    """
    Unified CNN layer for optical neural networks

    This layer implements a balanced depthwise-style architecture where each input channel
    uses a fixed number of optical filters. The total number of filters equals the number
    of output channels, ensuring symmetric architecture across all layers.

    Architecture principle:
        - filters_per_input = out_channels // in_channels
        - total_filters = out_channels
        - Each input channel is processed by filters_per_input dedicated filters

    Examples:
        Layer 0: 1 -> 4 channels: 4 filters total (4 filters for 1 input channel)
        Layer 1: 4 -> 4 channels: 4 filters total (1 filter per input channel)
        Layer 2: 4 -> 4 channels: 4 filters total (1 filter per input channel)

        This gives balanced 4+4+4+4 architecture instead of unbalanced 16+1+1+1

    Detection Modes:
        - 'coherent': Complex amplitude interference through MZI array
        - 'power': Power-domain linear superposition (incoherent)
    """

    def __init__(self, in_channels, out_channels, kernel_size, mzi_repeat_num,
                 mzi_row_num, mzi_column_num, detection_mode='coherent',
                 input_phase_noise_sigma=0.0):
        """
        Initialize CNN layer with optical filters

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of convolution kernel
            mzi_repeat_num (int): Number of MZI layer repetitions
            mzi_row_num (int): Number of MZIs per row
            mzi_column_num (int): Number of MZIs per column
            detection_mode (str): 'coherent' or 'power', default 'coherent'
            input_phase_noise_sigma (float): Std of input phase noise in radians, default 0.0

        Note:
            out_channels must be divisible by in_channels for proper filter distribution
        """
        super(CNN_layer, self).__init__()

        # Validate parameters
        if out_channels % in_channels != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by in_channels ({in_channels})"
            )

        # Layer parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.detection_mode = detection_mode
        self.filters_per_input = out_channels // in_channels
        self.flag = 0

        # Create out_channels filters (one filter per output channel)
        # Layer 0: 1->4, creates 4 filters
        # Layer 1: 4->4, creates 4 filters (1 per input channel, NOT shared)
        self.filters = nn.ModuleList([
            SingleChannelFilter(
                kernel_size=kernel_size,
                repeat_num=mzi_repeat_num,
                mzi_row_num=mzi_row_num,
                mzi_column_num=mzi_column_num,
                detection_mode=detection_mode,
                input_phase_noise_sigma=input_phase_noise_sigma
            )
            for _ in range(out_channels)
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

        Process each input channel through its assigned filters:
        - Input channel i uses filters [i * filters_per_input : (i+1) * filters_per_input]
        - Total outputs = out_channels

        Example for 4->4 layer (filters_per_input = 1):
            Channel 0 -> Filter 0 -> Output channel 0
            Channel 1 -> Filter 1 -> Output channel 1
            Channel 2 -> Filter 2 -> Output channel 2
            Channel 3 -> Filter 3 -> Output channel 3

        Example for 1->4 layer (filters_per_input = 4):
            Channel 0 -> Filters [0,1,2,3] -> Output channels [0,1,2,3]

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, new_height, new_width)
        """
        # Get input dimensions
        batch_size, channels, height, width = x.shape

        # Validate input channels
        assert channels == self.in_channels, \
            f"Input has {channels} channels, but layer expects {self.in_channels}"

        # List to store outputs from each filter
        outputs = []

        # Process each input channel with dedicated filters (NOT shared)
        # Each input channel uses filters_per_input consecutive filters
        for i in range(self.in_channels):
            # Extract single channel input
            channel_input = x[:, i]  # Shape: (batch_size, height, width)

            # Apply filters_per_input filters to this input channel
            for j in range(self.filters_per_input):
                # Calculate the dedicated filter index for this input-output pair
                filter_idx = i * self.filters_per_input + j

                # Update filter flag if needed
                if self.flag == 1:
                    self.filters[filter_idx].flag = 1

                # Apply the DEDICATED filter (not shared across input channels)
                outputs.append(self.filters[filter_idx](channel_input))

        # Stack outputs along channel dimension
        return torch.stack(outputs, dim=1)  # Shape: (batch_size, out_channels, new_height, new_width)
