import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.CNN import CNN_layer
from module.optical_linear_shared import OpticalLoRALinear # Modified import


class OpticalNetwork(nn.Module):
    """
    Unified Optical Neural Network with Balanced Architecture (Shared Weights Version)
    """

    def __init__(self, input_channels, hidden_channels, output_size, mzi_repeat_num,
                 mzi_row_num, mzi_column_num, num_layers=4, input_size=14, kernel_size=3,
                 detection_mode='coherent', use_optical_fc=True, fc_activation_mode='linear',
                 num_shared_weights=None, fc_pos_only: bool = False,
                 input_phase_noise_sigma: float = 0.0):
        """
        Initialize the Optical Neural Network (Shared Weights)

        Args:
            ...
            num_shared_weights (int): Number of shared MZI processors for the FC layer.
                                      If None, uses full independent processors.
            input_phase_noise_sigma (float): Std of input phase noise in radians for CNN filters.
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
        self.num_shared_weights = num_shared_weights
        self.fc_pos_only = fc_pos_only
        self.input_phase_noise_sigma = input_phase_noise_sigma

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

            self.layers.append(CNN_layer(
                in_channels,
                out_channels,
                kernel_size,
                mzi_repeat_num,
                mzi_row_num,
                mzi_column_num,
                detection_mode=detection_mode,
                input_phase_noise_sigma=input_phase_noise_sigma
            ))
            self.bns.append(nn.BatchNorm2d(out_channels))

            in_channels = out_channels
            current_size = current_size - kernel_size + 1

        # Activation and flatten layers
        self.activation = nn.ELU()
        self.flatten = nn.Flatten()

        # Calculate final feature dimensions
        self.feature_size = hidden_channels * current_size * current_size

        # Final fully connected layer
        if self.use_optical_fc:
            # Calculate MZI start index (after all CNN layers)
            fc_start_index = self._calculate_total_mzis()

            use_clean_fc = os.environ.get('OPTICAL_FC_CLEAN', '0') == '1'
            self.fc = OpticalLoRALinear(
                in_features=self.feature_size,
                out_features=output_size,
                r=10,  # Hardware constraint
                activation_mode=self.fc_activation_mode,
                detection_mode='power',  # Always use power mode
                mzi_row_num=self.mzi_row_num,
                mzi_column_num=self.mzi_column_num,
                repeat_num=self.mzi_repeat_num,
                start_index=fc_start_index,
                num_shared_weights=self.num_shared_weights, # Pass shared weights param
                pos_only=self.fc_pos_only,
                use_clean_fc=use_clean_fc,
            )
        else:
            # Use electronic linear layer
            self.fc = nn.Linear(self.feature_size, output_size)

        # Initialize timing statistics
        self.timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}

        # Print architecture summary
        self._print_architecture_summary(current_size)

    def _calculate_total_mzis(self):
        total = 0
        for layer in self.layers:
            if isinstance(layer, CNN_layer):
                for filter_module in layer.filters:
                    if hasattr(filter_module, 'total_mzis'):
                        total += filter_module.total_mzis
        return total

    def _print_architecture_summary(self, final_size):
        print(f"\n{'='*70}")
        print(f"Optical Neural Network Initialized (Shared Weights Version)")
        print(f"{'='*70}")
        print(f"Detection Mode: {self.detection_mode.upper()}")
        print(f"Architecture: {self.input_channels} -> " +
              f"{' -> '.join([str(self.hidden_channels)]*self.num_layers)}")
        print(f"Shared Weights (K): {self.num_shared_weights if self.num_shared_weights else 'None (Full)'}")
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
        total_stats = {'allocation': 0, 'computation': 0, 'total': 0}
        for key in total_stats:
            if hasattr(self, 'timing_stats'):
                total_stats[key] += self.timing_stats[key]
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

    def forward(self, x, return_features=False):
        # When feeding an Optical FC in power mode, the CNN output P (≥0
        # after the filter's ReLU) is sent through sqrt → ELU → square
        # before the FC. ELU is identity on non-negative inputs and
        # sqrt·square cancel, so the chain is a mathematical no-op on P
        # — but sqrt's derivative blows up near 0, polluting Adam's 2nd
        # moment. We bypass the round-trip and feed P directly to the FC.
        # Gated by env var so old behavior remains the default.
        bypass_amp_roundtrip = (
            os.environ.get('OPTICAL_FC_POWER_BYPASS', '0') == '1'
            and self.use_optical_fc
            and self.detection_mode == 'power'
        )

        for layer_idx, (layer, bn) in enumerate(zip(self.layers, self.bns)):
            if self.flag == 1:
                layer.flag = 1
            x = layer(x)
            if not bypass_amp_roundtrip:
                if self.detection_mode == 'coherent':
                    x = torch.abs(x)
                else:
                    x = torch.sqrt(x + 1e-8)
                x = self.activation(x)
            if hasattr(self, 'save_feature_maps') and self.save_feature_maps:
                batch_idx = 0
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

        # Capture features before flattening
        features = x

        x = self.flatten(x)

        if self.use_optical_fc:
            if not bypass_amp_roundtrip:
                x = x ** 2
            x = self.fc(x)
            # No sqrt here, using raw logits
        else:
            x = self.fc(x)

        if return_features:
            return x, features
        return x
