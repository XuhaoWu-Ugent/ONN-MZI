# Required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MZIlayer_column(nn.Module):
    def __init__(self):
        """
        Initialize a simplified column layer that performs fixed permutations
        No MZI modules needed as this just does direct mapping
        """
        super(MZIlayer_column, self).__init__()

    def forward(self, input):
        """
        Forward pass that performs fixed permutation of elements
        
        Args:
            input: Input tensor of shape (batch_size, num_patches, 10)
                  Where 10 represents the number of channels to be permuted
                  
        Returns:
            output: Permuted tensor of shape (batch_size, num_patches, 10)
                   With elements mapped according to specified pattern
        """
        # Extract input dimensions
        batch_size, num_patches, _ = input.shape
        
        # Initialize output tensor with same shape and type as input
        output = torch.zeros_like(input)

        # Map elements according to fixed permutation pattern:
        output[:, :, 0] = input[:, :, 0]   # Position 1 -> 1 (unchanged)
        output[:, :, 1] = input[:, :, 2]   # Position 3 -> 2
        output[:, :, 2] = input[:, :, 1]   # Position 2 -> 3
        output[:, :, 3] = input[:, :, 4]   # Position 5 -> 4
        output[:, :, 4] = input[:, :, 3]   # Position 4 -> 5
        output[:, :, 5] = input[:, :, 6]   # Position 7 -> 6
        output[:, :, 6] = input[:, :, 5]   # Position 6 -> 7
        output[:, :, 7] = input[:, :, 8]   # Position 9 -> 8
        output[:, :, 8] = input[:, :, 7]   # Position 8 -> 9
        output[:, :, 9] = input[:, :, 9]   # Position 10 -> 10 (unchanged)

        return output  # Return permuted tensor
