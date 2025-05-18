import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.CNN import CNN_layer

class OpticalNetwork(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_size, mzi_repeat_num ,mzi_row_num,mzi_column_num, num_layers=3, input_size=14, kernel_size=3):
        """
        Initialize the Optical Neural Network
        
        Args:
            input_channels (int): Number of input channels
            hidden_channels (int): Number of channels in hidden layers
            output_size (int): Size of network output
            num_layers (int): Number of CNN layers
            input_size (int): Size of input images
            kernel_size (int): Size of convolution kernels
        """
        super(OpticalNetwork, self).__init__()
        
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
            self.layers.append(CNN_layer(in_channels, out_channels, kernel_size, mzi_repeat_num ,mzi_row_num,mzi_column_num))
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
        
    def forward(self, x):
        """
        Forward propagation through the network
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_channels, height, width)
            
        Returns:
            Tensor: Output predictions
        """
        # Process through CNN layers
        for layer, bn in zip(self.layers, self.bns):
            # Propagate debug flag
            if self.flag == 1:
                layer.flag = 1
                
            # Apply CNN layer
            x = layer(x)
            
            #x = bn(x)  # Batch normalization (currently disabled)
            
            # Apply absolute value and activation
            x = torch.abs(x)
            x = self.activation(x)
            
        # Reduce channel dimension
        x = self.conv1x1(x)
        
        # Flatten and apply final linear layer
        x = self.flatten(x)
        x = self.fc(x)
        
        # import pdb; pdb.set_trace()  # Debug breakpoint (currently disabled)
        
        return x
