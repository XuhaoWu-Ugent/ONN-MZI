import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.MZI_array.mzi_column_array import MZIlayer_column
from module.MZI_array.mzi_row_array import MZIlayer_row


class SingleChannelFilter(nn.Module):
    def __init__(self, mzi_row_num=5, mzi_column_num=4, repeat_num=5, kernel_size=3):
        """
        Initialize single channel filter module using new MZI array implementation
        
        Args:
            mzi_row_num (int): Number of MZIs in row arrays, default 5
            mzi_column_num (int): Number of MZIs in column arrays, default 4
            repeat_num (int): Number of MZI layer repetitions, default 5
            kernel_size (int): Size of the convolution kernel, default 3
        """
        super(SingleChannelFilter, self).__init__()
        
        # Store parameters
        self.mzi_row_num = mzi_row_num
        self.mzi_column_num = mzi_column_num
        self.repeat_num = repeat_num
        self.kernel_size = kernel_size
        self.num_ports = mzi_row_num * 2  # 10 ports for compatibility
        
        # Create alternating row and column MZI layers (same structure as original)
        self.layers = nn.ModuleList()
        for _ in range(repeat_num):
            self.layers.append(MZIlayer_row(num=mzi_row_num))
            self.layers.append(MZIlayer_column(num=mzi_column_num))
        self.layers.append(MZIlayer_row(num=mzi_row_num))
        
        # Trainable diagonal weight matrix (same as original)
        self.diagonal_matrix = nn.Parameter(torch.randn(self.num_ports))
        
        # Other parameters (same as original)
        self.flag = 0
        self.timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}
        
        # Store matrix size for dynamic computation
        self.matrix_size = 2 * self.num_ports  # 20x20

        # Pre-allocate combined transfer matrix (cached only in eval mode)
        self._combined_matrix = None
        # Remove gradient accumulation counter to prevent training issues

    def _get_combined_matrix(self, device):
        """
        Get combined transfer matrix with training-safe caching
        CRITICAL: Always rebuild in training mode to prevent gradient graph issues
        """
        # TRAINING MODE: Always rebuild matrix to avoid gradient conflicts
        if self.training:
            # Always create fresh matrix during training
            return self._build_combined_transfer_matrix(device)

        # EVAL MODE: Use caching for performance
        should_update = (self._combined_matrix is None or
                        self._combined_matrix.device != device)

        if should_update:
            self._combined_matrix = self._build_combined_transfer_matrix(device)
            # Detach to prevent gradient issues when switching between modes
            self._combined_matrix = self._combined_matrix.detach().clone()

        return self._combined_matrix

    def _build_combined_transfer_matrix(self, device):
        """
        Build combined transfer matrix by multiplying all layer matrices
        Following the sequence: Row → Column → Row → Column → ... → Row (11 layers total)
        """
        # Start with identity matrix
        combined_matrix = torch.eye(self.matrix_size, dtype=torch.complex64, device=device)

        # Multiply matrices in reverse order (right to left multiplication)
        # This ensures proper composition: output = M_n * M_{n-1} * ... * M_1 * input
        for layer in reversed(self.layers):
            # Get the transfer matrix from each layer dynamically
            layer_matrix = layer._build_transfer_matrix(device)
            # Matrix multiplication: combined = current_layer * previous_combined
            combined_matrix = torch.matmul(layer_matrix, combined_matrix)

        return combined_matrix

    def get_all_time(self):
        """
        Recursively collect timing statistics from all layers (same as original)
        
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

    def train(self, mode=True):
        """Override train() to clear cache when switching to training mode"""
        super().train(mode)
        if mode:  # Entering training mode
            self._combined_matrix = None  # Clear cached matrix to prevent gradient issues
        return self

    def forward(self, x, return_intermediate=False):
        """
        Forward propagation function using combined transfer matrix
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, height, width)
            return_intermediate (bool): Whether to return intermediate values for the first patch
            
        Returns:
            Tensor or tuple: Output tensor, or tuple of (output, input_patch, processed_patch) if return_intermediate=True
        """
        # Get input dimensions (same as original)
        batch_size, height, width = x.shape

        # Calculate output dimensions (same as original)
        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1

        # Extract image patches (same as original)
        # Add channel dimension and unfold into patches
        patches = F.unfold(x.unsqueeze(1),
                          kernel_size=self.kernel_size,
                          stride=1)  # Shape: (batch_size, kernel_size*kernel_size, num_patches)

        # Adjust dimension order (same as original)
        patches = patches.permute(0, 2, 1)  # Shape: (batch_size, num_patches, kernel_size*kernel_size)

        # Pad to 10 dimensions to match MZI layers (same as original)
        patches = F.pad(patches, (0, 1))  # Shape: (batch_size, num_patches, 10)

        # Save first patch input if needed (same as original)
        if return_intermediate:
            input_patch = patches[:, 0, :].clone().detach()

        # Convert to complex type for complex operations (same as original)
        patches = patches.to(torch.complex64)

        # NEW: Process through combined transfer matrix with optimized batch operations
        batch_size, num_patches, num_ports = patches.shape

        # Get or update combined transfer matrix with gradient accumulation
        combined_matrix = self._get_combined_matrix(patches.device)

        # Batch processing without loops - pre-allocate all tensors
        batch_total = batch_size * num_patches
        patches_flat = patches.view(batch_total, num_ports)

        # Create state vectors for entire batch at once
        state_vectors = torch.zeros(batch_total, self.matrix_size,
                                   dtype=torch.complex64, device=patches.device)
        state_vectors[:, :num_ports] = patches_flat

        # Batch matrix multiplication - process entire batch at once
        new_states = torch.matmul(state_vectors, combined_matrix.T)

        # Extract outputs for entire batch
        output_flat = new_states[:, self.num_ports:self.num_ports + num_ports]
        output_patches = output_flat.view(batch_size, num_patches, num_ports)

        # Save MZI processed first patch if needed (same as original)
        if return_intermediate:
            processed_patch = output_patches[:, 0, :].clone().detach()

        # Apply diagonal weight matrix and sum (same as original)
        weighted_output = output_patches * self.diagonal_matrix.view(1, 1, -1)

        # Reshape output to desired spatial dimensions (same as original)
        output = weighted_output.sum(dim=2).view(batch_size, output_height, output_width)

        # Return based on parameters (same as original)
        if return_intermediate:
            return output, input_patch, processed_patch
        else:
            return output