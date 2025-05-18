import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from module.MZI_array.mzi_column_array import *
from module.MZI_array.mzi_row_array import *


class CNN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, repeat_num, mzi_row_num, mzi_column_num,padding=1):
        super(CNN_layer, self).__init__()
        # 保存输入输出通道数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # 使用 ModuleList 存储每个输出通道的卷积核
        self.layers = nn.ModuleList()
        self.padding=padding
        # 存储每个卷积核的对角矩阵
        self.diag_matrices = nn.ParameterList()
        
        # 为每个输出通道构造对应的卷积核结构
        for out_channel in range(out_channels):
            channel_layers = nn.ModuleList()
            for in_channel in range(in_channels):
                # 每个输入通道到输出通道的映射由 5 层 row 和 4 层 column 交替构成
                single_kernel_layers = nn.ModuleList()
                
                # 添加 5 层 row 和 4 层 column 的交替结构
                for i in range(5):  # 总共 5 层 row
                    single_kernel_layers.append(MZIlayer_row(num=mzi_row_num))
                    if i < 4:  # 前 4 层 row 后面接 column
                        single_kernel_layers.append(MZIlayer_column(num=mzi_column_num))
                
                # 将该输入通道到输出通道的卷积核结构添加到列表中
                channel_layers.append(single_kernel_layers)
                
                # 为该卷积核添加一个对角矩阵
                self.diag_matrices.append(nn.Parameter(torch.randn(mzi_row_num * 2)))  # 对角矩阵
            
            # 将所有输入通道到该输出通道的卷积核结构存储
            self.layers.append(channel_layers)

    def forward(self, x):
        """
        x 的形状为 (batch_size, in_channels, height, width)
        """
        batch_size, in_channels, height, width = x.shape
        assert in_channels == self.in_channels, "输入通道数与定义的输入通道数不匹配！"
        
        # 初始化输出特征图
        output = torch.zeros(batch_size, self.out_channels, height, width, device=x.device, dtype=torch.complex64)
        
        diag_matrix_index = 0  # 用于索引 self.diag_matrices 中的对角矩阵
        
        # 遍历每个输出通道
        for out_channel, channel_layers in enumerate(self.layers):
            # 初始化当前输出通道的特征图
            out_feature_map = torch.zeros(batch_size, height, width, device=x.device, dtype=torch.complex64)
            
            # 遍历每个输入通道
            for in_channel, single_kernel_layers in enumerate(channel_layers):
                # 对当前输入通道进行处理
                input_feature_map = x[:, in_channel, :, :]  # 取出当前输入通道的特征图，形状为 (batch_size, height, width)
                
                # 使用 unfold 提取滑动窗口
                kernel_size = 3  # 假设卷积核大小为 3x3
                padding = self.padding  # 假设使用 1 像素的零填充
                stride = 1  # 假设步幅为 1
                
                # 对输入特征图进行零填充
                input_feature_map = torch.nn.functional.pad(input_feature_map, (padding, padding, padding, padding))
                
                # 提取滑动窗口，形状为 (batch_size, num_patches, kernel_size * kernel_size)
                unfolded = input_feature_map.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
                unfolded = unfolded.contiguous().view(batch_size, height, width, -1)  # 展平滑动窗口，形状为 (batch_size, height, width, kernel_size * kernel_size)
                
                # 为滑动窗口增加一个额外的维度，形状为 (batch_size, height, width, kernel_size * kernel_size + 1)
                unfolded = torch.cat([unfolded, torch.ones(batch_size, height, width, 1, device=x.device)], dim=-1)
                
                # 依次通过 5 层 row 和 4 层 column
                for layer in single_kernel_layers:
                    unfolded = layer(unfolded)
                
                # 最后一层为对角矩阵，逐元素相乘
                diag_matrix = self.diag_matrices[diag_matrix_index]  # 获取对角矩阵
                diag_matrix_index += 1  # 更新索引
                
                # 将对角矩阵应用到滑动窗口的最后一维
                unfolded = unfolded * diag_matrix.view(1, 1, 1, -1)
                
                # 将滑动窗口的结果累加到输出特征图
                out_feature_map += unfolded.sum(dim=-1)  # 对最后一维求和，形状变为 (batch_size, height, width)
            
            # 将结果存入输出特征图
            output[:, out_channel, :, :] = out_feature_map
        
        return output
