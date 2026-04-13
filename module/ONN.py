import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.CNN import CNN_layer
from module.optical_linear import OpticalLinear


class OpticalNetwork(nn.Module):
    """
    Unified Optical Neural Network with Balanced Architecture

    Architecture Example (hidden_channels=4, num_layers=4):
        Layer 0: 1 -> 4 channels (4 filters, 4 per input channel)
        Layer 1: 4 -> 4 channels (4 filters, 1 per input channel, NOT shared)
        Layer 2: 4 -> 4 channels (4 filters, 1 per input channel, NOT shared)
        Layer 3: 4 -> 4 channels (4 filters, 1 per input channel, NOT shared)
        Total: 4+4+4+4 = 16 filters

    Detection Modes:
        - 'coherent': Complex amplitude interference (phase information preserved)
        - 'power': Power-domain linear superposition (incoherent detection)

    The network uses the same underlying MZI array architecture for both modes,
    with the difference being in the detection/measurement process at each filter.
    """

    def __init__(self, input_channels, hidden_channels, output_size, mzi_repeat_num,
                 mzi_row_num, mzi_column_num, num_layers=4, input_size=14, kernel_size=3,
                 detection_mode='coherent', use_optical_fc=True, fc_activation_mode='linear'):
        """
        Initialize the Optical Neural Network

        Args:
            input_channels (int): Number of input channels (typically 1 for grayscale)
            hidden_channels (int): Number of channels in all layers
            output_size (int): Size of network output (number of classes)
            mzi_repeat_num (int): Number of MZI layer repetitions in each filter
            mzi_row_num (int): Number of MZIs per row in MZI array
            mzi_column_num (int): Number of MZIs per column in MZI array
            num_layers (int): Number of CNN layers, default 4
            input_size (int): Size of input images, default 14
            kernel_size (int): Size of convolution kernels, default 3
            detection_mode (str): 'coherent' or 'power', default 'coherent'
            use_optical_fc (bool): Use OpticalLinear instead of nn.Linear, default True
            fc_activation_mode (str): 'linear' or 'nonlinear' for optical FC, default 'linear'
        """
        super(OpticalNetwork, self).__init__()

        # Store configuration
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.detection_mode = detection_mode
        self.num_layers = num_layers
        self.use_optical_fc = use_optical_fc
        self.fc_activation_mode = fc_activation_mode
        self.mzi_repeat_num = mzi_repeat_num
        self.mzi_row_num = mzi_row_num
        self.mzi_column_num = mzi_column_num

        # Initialize network components
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.flag = 0

        # Track current channel count and feature size
        in_channels = input_channels
        current_size = input_size

        # Build CNN layers with batch normalization
        # All layers after the first will have balanced architecture
        for i in range(num_layers):
            out_channels = hidden_channels

            self.layers.append(CNN_layer(
                in_channels,
                out_channels,
                kernel_size,
                mzi_repeat_num,
                mzi_row_num,
                mzi_column_num,
                detection_mode=detection_mode
            ))
            self.bns.append(nn.BatchNorm2d(out_channels))

            in_channels = out_channels
            current_size = current_size - kernel_size + 1

        # Activation and flatten layers
        self.activation = nn.ELU()
        self.flatten = nn.Flatten()

        # Calculate final feature dimensions
        # No 1x1 conv - directly use hidden_channels
        self.feature_size = hidden_channels * current_size * current_size

        # Final fully connected layer
        if self.use_optical_fc:
            # Calculate MZI start index (after all CNN layers)
            fc_start_index = self._calculate_total_mzis()

            self.fc = OpticalLinear(
                in_features=self.feature_size,
                out_features=output_size,
                r=10,  # Hardware constraint
                activation_mode=self.fc_activation_mode,
                detection_mode='power',  # Always use power mode
                mzi_row_num=self.mzi_row_num,
                mzi_column_num=self.mzi_column_num,
                repeat_num=self.mzi_repeat_num,
                start_index=fc_start_index
            )
        else:
            # Use electronic linear layer
            self.fc = nn.Linear(self.feature_size, output_size)

        # Initialize timing statistics
        self.timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}

        # Print architecture summary
        self._print_architecture_summary(current_size)

    def _calculate_total_mzis(self):
        """
        Calculate total number of MZIs used by all CNN layers

        Returns:
            int: Total MZI count
        """
        total = 0
        for layer in self.layers:
            if isinstance(layer, CNN_layer):
                for filter_module in layer.filters:
                    if hasattr(filter_module, 'total_mzis'):
                        total += filter_module.total_mzis
        return total

    def _print_architecture_summary(self, final_size):
        """
        Print detailed architecture summary

        Args:
            final_size (int): Final spatial dimension before FC layer
        """
        print(f"\n{'='*70}")
        print(f"Optical Neural Network Initialized")
        print(f"{'='*70}")
        print(f"Detection Mode: {self.detection_mode.upper()}")
        print(f"Architecture: {self.input_channels} -> " +
              f"{' -> '.join([str(self.hidden_channels)]*self.num_layers)}")
        print(f"\nLayer Details:")

        total_filters = 0
        for i, layer in enumerate(self.layers):
            num_filters = len(layer.filters)
            filters_per_input = layer.filters_per_input
            total_filters += num_filters

            print(f"  Layer {i}: {layer.in_channels}->{layer.out_channels} channels")
            print(f"           Filters: {num_filters} ({filters_per_input} per input channel)")

        print(f"\nTotal Filters: {total_filters}")
        print(f"Final Feature Map: {self.hidden_channels} x {final_size} x {final_size}")
        print(f"FC Input Size: {self.feature_size}")
        print(f"{'='*70}\n")

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

    def forward(self, x):
        """
        Forward propagation through the network

        Processing flow:
        1. Input -> CNN Layers (with batch norm and activation)
        2. Flatten
        3. Fully connected layer -> Output

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_channels, height, width)

        Returns:
            Tensor: Output predictions of shape (batch_size, output_size)
        """
        # Process through CNN layers
        for layer_idx, (layer, bn) in enumerate(zip(self.layers, self.bns)):
            # Propagate debug flag
            if self.flag == 1:
                layer.flag = 1

            # Apply CNN layer
            x = layer(x)

            # Batch normalization (optional, currently disabled)
            # x = bn(x)

            # Convert to real values and apply activation
            # Different processing based on detection_mode:
            # - coherent: x is complex amplitude, use abs() to get magnitude
            # - power: x is real power values, use sqrt() to get amplitude-like scale
            if self.detection_mode == 'coherent':
                x = torch.abs(x)  # Convert complex amplitude to magnitude
            else:  # 'power'
                # x is already real power (positive), convert to amplitude scale
                # Add small epsilon for numerical stability
                x = torch.sqrt(x + 1e-8)

            x = self.activation(x)

            # Feature map extraction for visualization (if enabled)
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
                        'data': feature_map
                    })

        # Flatten and apply final linear layer
        x = self.flatten(x)

        # Power domain handling for optical FC
        if self.use_optical_fc:
            # Optical layer needs power domain input (non-negative)
            # x is currently in amplitude scale after activation
            x = x ** 2  # amplitude -> power domain

            x = self.fc(x)  # OpticalLinear (operates in power domain)

            # Output includes electrical bias and can be negative (Logits).
            # Do NOT take sqrt here.
            # x = torch.sqrt(x + 1e-8)
        else:
            # Electronic linear layer (no domain conversion needed)
            x = self.fc(x)

        return x
