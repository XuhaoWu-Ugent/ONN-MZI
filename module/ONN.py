import torch
import torch.nn as nn
import torch.nn.functional as F
from module.CNN import CNN_layer

class OpticalNetwork(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_size, mzi_repeat_num, mzi_row_num, mzi_column_num,
                 input_size=32, kernel_size=3, fc_hidden_size=256, pool_kernel_size=2, pool_stride=2,padding=1):
        """
        Initialize the Optical Neural Network
        
        Args:
            input_channels (int): Number of input channels
            hidden_channels (list): List of channels for each hidden layer
            output_size (int): Size of network output
            mzi_repeat_num (int): Number of MZI repetitions
            mzi_row_num (int): Number of MZIs per row
            mzi_column_num (int): Number of MZIs per column
            input_size (int): Size of input images
            kernel_size (int): Size of convolution kernels
            fc_hidden_size (int): Number of neurons in the fully connected hidden layer
            pool_kernel_size (int): Size of the pooling kernel (default: 2)
            pool_stride (int): Stride for the pooling operation (default: 2)
        """
        super(OpticalNetwork, self).__init__()
        
        # Initialize network components
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.pools = nn.ModuleList()  # 添加池化层列表
        
        # Track current channel count and feature size
        in_channels = input_channels
        current_size = input_size
        
        # Build CNN layers dynamically based on `hidden_channels` list
        for out_channels in hidden_channels:
            self.layers.append(CNN_layer(in_channels, out_channels, kernel_size, mzi_repeat_num, mzi_row_num, mzi_column_num))
            self.bns.append(nn.BatchNorm2d(out_channels))
            self.pools.append(nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride))  # 添加最大池化层
            in_channels = out_channels
            current_size = (current_size - kernel_size + padding*2 + 1) // pool_stride  # 更新特征图尺寸

        # 1x1 convolution layer to reduce channel dimension
        self.conv1x1 = nn.Conv2d(hidden_channels[-1], 4, kernel_size=1)
        
        # Activation and flatten layers
        self.activation = nn.ELU()
        self.flatten = nn.Flatten()
        
        # Calculate final feature dimensions
        self.feature_size = 4 * current_size * current_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, fc_hidden_size)  # 第一层全连接层
        self.fc2 = nn.Linear(fc_hidden_size, output_size)  # 输出层
        
        # Initialize timing statistics
        self.timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}

    def forward(self, x):
        """
        Forward pass through the network
        """
        for layer, bn, pool in zip(self.layers, self.bns, self.pools):
            x = pool(self.activation(bn(torch.abs(layer(x)))))  # 卷积 + 批归一化 + 激活 + 池化
        
        # 1x1 convolution
        x = self.activation(self.conv1x1(x))
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))  # 第一层全连接层 + ReLU 激活
        x = self.fc2(x)  # 输出层
        return x

        
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
        

