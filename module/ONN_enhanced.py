import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.CNN_enhanced import CNN_layer_enhanced


class OpticalNetworkEnhanced(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_size, mzi_repeat_num, mzi_row_num, mzi_column_num,
                 num_layers=3, input_size=14, kernel_size=3, filter_type='coherent'):
        """
        Initialize the Enhanced Optical Neural Network with support for different detection modes

        Args:
            input_channels (int): Number of input channels
            hidden_channels (int): Number of channels in hidden layers
            output_size (int): Size of network output
            mzi_repeat_num (int): Number of MZI repetitions
            mzi_row_num (int): Number of MZIs per row
            mzi_column_num (int): Number of MZIs per column
            num_layers (int): Number of CNN layers
            input_size (int): Size of input images
            kernel_size (int): Size of convolution kernels
            filter_type (str): Type of filter - 'coherent' (amplitude interference) or 'power' (power superposition)
        """
        super(OpticalNetworkEnhanced, self).__init__()

        # Store filter configuration
        self.filter_type = filter_type

        # Initialize network components
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.flag = 0

        # Track current channel count and feature size
        in_channels = input_channels
        current_size = input_size

        # Build CNN layers with batch normalization
        for i in range(num_layers):
            out_channels = hidden_channels
            self.layers.append(
                CNN_layer_enhanced(
                    in_channels, out_channels, kernel_size,
                    mzi_repeat_num, mzi_row_num, mzi_column_num,
                    filter_type=filter_type
                )
            )
            self.bns.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
            current_size = current_size - kernel_size + 1

        # 1x1 convolution layer to reduce channel dimension
        self.conv1x1 = nn.Conv2d(hidden_channels, 4, kernel_size=1)

        # Activation and flatten layers
        self.activation = nn.ELU()
        self.flatten = nn.Flatten()

        # Calculate final feature dimensions
        self.feature_size = 4 * current_size * current_size

        # Final fully connected layer
        self.fc = nn.Linear(self.feature_size, output_size)

        # Initialize timing statistics
        self.timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}

    def get_all_time(self):
        """
        Recursively collect timing statistics from all network components

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

    def get_filter_info(self):
        """
        Get information about the filter configuration

        Returns:
            dict: Dictionary containing filter configuration information
        """
        info = {
            'filter_type': self.filter_type,
            'detection_principle': self._get_detection_principle()
        }
        return info

    def _get_detection_principle(self):
        """Get human-readable description of detection principle"""
        if self.filter_type == 'coherent':
            return "Coherent detection - complex amplitude interference"
        elif self.filter_type == 'power':
            return "Power-domain superposition - independent sources, no coherent interference"
        else:
            return "Unknown detection principle"

    def forward(self, x):
        """
        Forward propagation through the enhanced network

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_channels, height, width)

        Returns:
            Tensor: Output predictions
        """
        # Process through CNN layers
        for layer_idx, (layer, bn) in enumerate(zip(self.layers, self.bns), start=1):
            # Propagate debug flag
            if self.flag == 1:
                layer.flag = 1

            # Apply CNN layer
            x = layer(x)

            # x = bn(x)  # Batch normalization (currently disabled)

            # Apply activation based on detection mode
            if self.filter_type == 'coherent':
                x = torch.abs(x)  # Convert complex amplitude to magnitude
            elif self.filter_type == 'power':
                # x is already in power domain (real and positive)
                # Convert power to amplitude-like scale for consistent processing
                x = torch.sqrt(x + 1e-8)  # Add small epsilon for numerical stability

            x = self.activation(x)

            # Extract convolution feature maps as grayscale images
            if hasattr(self, 'save_feature_maps') and self.save_feature_maps:
                batch_idx = 0  # Save first sample in batch
                num_channels = x.shape[1]
                for ch_idx in range(num_channels):
                    feature_map = x[batch_idx, ch_idx].detach().cpu().numpy()
                    if not hasattr(self, 'feature_maps'):
                        self.feature_maps = []
                    self.feature_maps.append({
                        'layer_idx': layer_idx,
                        'channel_idx': ch_idx,
                        'data': feature_map,
                        'filter_type': self.filter_type  # Record which mode was used
                    })

        # Reduce channel dimension
        x = self.conv1x1(x)

        # Flatten and apply final linear layer
        x = self.flatten(x)
        x = self.fc(x)

        return x