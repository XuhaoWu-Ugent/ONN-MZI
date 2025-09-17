# Required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.MZI_array.mzi import MZI


class MZIlayer_column(nn.Module):
    """
    Fixed ultra-fast MZI Column Array implementation
    Solves gradient graph issues while maintaining performance
    """

    def __init__(self, num=4):
        """
        Args:
            num (int): Number of MZIs, default 4 (ports 0,9 direct bypass)
        """
        super(MZIlayer_column, self).__init__()
        self.num = num
        self.num_ports = 2 * (num + 1)

        # Create MZI instances (maintain consistent naming with original interface)
        self.MZI = nn.ModuleList([MZI() for _ in range(num)])
        self.mzis = self.MZI  # Backward compatibility alias

        # Add attributes compatible with original implementation
        self.flag = 0
        self.timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}

        # Store matrix size for dynamic computation
        self.matrix_size = 2 * self.num_ports

        # Smart caching with parameter tracking
        self._cached_matrix = None
        self._cached_device = None
        self._parameter_hash = None
        self._cache_hits = 0
        self._cache_misses = 0

        # Pre-allocate tensors for batch operations
        self._max_batch_size = 1024  # Reasonable default
        self._preallocated_state_vectors = None
        self._preallocated_outputs = None

    def _compute_parameter_hash(self):
        """Compute a hash of all parameters for efficient change detection"""
        params = []
        for mzi in self.MZI:
            params.append(mzi.raw_sin_theta.data.item())
        return hash(tuple(params))

    def _is_cache_valid(self, device):
        """Fast cache validation - NEVER cache in training mode to avoid gradient issues"""
        # CRITICAL: Never cache during training to prevent gradient graph issues
        if (self._cached_matrix is None or
            self._cached_device != device or
            self.training):
            return False

        # Fast parameter change detection
        current_hash = self._compute_parameter_hash()
        return current_hash == self._parameter_hash

    def _build_transfer_matrix(self, device):
        """Optimized matrix building for column array"""
        # Check cache validity first
        if self._is_cache_valid(device):
            self._cache_hits += 1
            return self._cached_matrix

        self._cache_misses += 1

        # Build matrix efficiently
        matrix = torch.zeros((self.matrix_size, self.matrix_size),
                           dtype=torch.complex64, device=device)

        # Direct connections for first and last ports
        matrix[self.num_ports + 0, 0] = 1.0
        matrix[0, self.num_ports + 0] = 1.0
        matrix[self.num_ports + (self.num_ports - 1), self.num_ports - 1] = 1.0
        matrix[self.num_ports - 1, self.num_ports + (self.num_ports - 1)] = 1.0

        # True batch processing for all MZI matrices (column array specific)
        if self.num > 0:
            # Compute all MZI matrices using proper 4-port MZI transform
            # This preserves the original MZI.forward() behavior for bidirectional support
            identity_4x4 = torch.eye(4, dtype=torch.complex64, device=device)
            batch_identity = identity_4x4.unsqueeze(0).repeat(self.num, 1, 1)  # (num_mzi, 4, 4)

            # Stack all MZI parameters for batch processing
            batch_sin_theta = []
            batch_cos_theta = []
            for mzi in self.MZI:
                sin_val, cos_val = mzi.get_sin_cos()
                # Ensure scalar values by squeezing
                batch_sin_theta.append(sin_val.squeeze().to(torch.complex64))
                batch_cos_theta.append(cos_val.squeeze().to(torch.complex64))

            sin_theta_batch = torch.stack(batch_sin_theta)  # (num_mzi,)
            cos_theta_batch = torch.stack(batch_cos_theta)  # (num_mzi,)

            # Batch compute MZI transformations using the actual MZI forward logic
            mzi_matrices = torch.zeros(self.num, 4, 4, dtype=torch.complex64, device=device)

            # Apply vectorized MZI transformation for all MZIs at once
            for j in range(4):
                input_vec = batch_identity[:, :, j]  # (num_mzi, 4)

                # Vectorized cosine and sine components
                x_c = input_vec * cos_theta_batch.unsqueeze(-1)  # (num_mzi, 4)
                x_s = input_vec * (-1j * sin_theta_batch.unsqueeze(-1))  # (num_mzi, 4)

                # Apply MZI transformation logic (vectorized version of MZI.forward)
                output_port_0 = x_c[..., 2] + x_s[..., 3]  # (num_mzi,)
                output_port_1 = x_c[..., 3] + x_s[..., 2]  # (num_mzi,)
                output_port_2 = x_c[..., 0] + x_s[..., 1]  # (num_mzi,)
                output_port_3 = x_c[..., 1] + x_s[..., 0]  # (num_mzi,)

                # Assign each output port to the corresponding row of column j
                mzi_matrices[:, 0, j] = output_port_0
                mzi_matrices[:, 1, j] = output_port_1
                mzi_matrices[:, 2, j] = output_port_2
                mzi_matrices[:, 3, j] = output_port_3

            # Column array specific port mapping - vectorized assembly preserving bidirectionality
            port_indices = torch.arange(self.num, device=device)
            port1_indices = port_indices * 2 + 1  # Ports 1,3,5,7
            port2_indices = port_indices * 2 + 2  # Ports 2,4,6,8

            # Forward direction (input → output) - vectorized assignment
            matrix[self.num_ports + port1_indices, port1_indices] = mzi_matrices[:, 2, 0]
            matrix[self.num_ports + port1_indices, port2_indices] = mzi_matrices[:, 2, 1]
            matrix[self.num_ports + port2_indices, port1_indices] = mzi_matrices[:, 3, 0]
            matrix[self.num_ports + port2_indices, port2_indices] = mzi_matrices[:, 3, 1]

            # Reverse direction (output → input) - vectorized assignment
            matrix[port1_indices, self.num_ports + port1_indices] = mzi_matrices[:, 0, 2]
            matrix[port1_indices, self.num_ports + port2_indices] = mzi_matrices[:, 0, 3]
            matrix[port2_indices, self.num_ports + port1_indices] = mzi_matrices[:, 1, 2]
            matrix[port2_indices, self.num_ports + port2_indices] = mzi_matrices[:, 1, 3]

            # Self-coupling terms (input-input and output-output) - vectorized assignment
            matrix[port1_indices, port1_indices] = mzi_matrices[:, 0, 0]
            matrix[port1_indices, port2_indices] = mzi_matrices[:, 0, 1]
            matrix[port2_indices, port1_indices] = mzi_matrices[:, 1, 0]
            matrix[port2_indices, port2_indices] = mzi_matrices[:, 1, 1]

            matrix[self.num_ports + port1_indices, self.num_ports + port1_indices] = mzi_matrices[:, 2, 2]
            matrix[self.num_ports + port1_indices, self.num_ports + port2_indices] = mzi_matrices[:, 2, 3]
            matrix[self.num_ports + port2_indices, self.num_ports + port1_indices] = mzi_matrices[:, 3, 2]
            matrix[self.num_ports + port2_indices, self.num_ports + port2_indices] = mzi_matrices[:, 3, 3]

        # Update cache - ONLY in eval mode
        if not self.training:
            # Use detach() and clone() to break gradient connection for cached matrices
            self._cached_matrix = matrix.detach().clone()  # Clone to ensure no sharing
            self._cached_device = device
            self._parameter_hash = self._compute_parameter_hash()

        return matrix

    def get_cache_stats(self):
        """Get cache performance statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }

    def _preallocate_tensors(self, batch_size, device):
        """Pre-allocate tensors for batch operations"""
        self._preallocated_state_vectors = torch.zeros(
            batch_size, self.matrix_size, dtype=torch.complex64, device=device)
        self._preallocated_outputs = torch.zeros(
            batch_size, self.num_ports, dtype=torch.complex64, device=device)

    def clear_cache(self):
        """Clear all caches and pre-allocated tensors"""
        self._cached_matrix = None
        self._cached_device = None
        self._parameter_hash = None
        self._preallocated_state_vectors = None
        self._preallocated_outputs = None

    def train(self, mode=True):
        """Override train() to clear cache when switching to training mode"""
        super().train(mode)
        if mode:  # Entering training mode
            self.clear_cache()  # Clear all cached tensors to prevent gradient issues
        return self

    def get_all_time(self):
        """Recursively collect timing statistics from all MZI modules"""
        total_stats = {'allocation': 0, 'computation': 0, 'total': 0}

        for key in total_stats:
            if hasattr(self, 'timing_stats'):
                total_stats[key] += self.timing_stats[key]

        if hasattr(self, 'MZI'):
            for layer in self.MZI:
                if hasattr(layer, 'get_all_time'):
                    layer_stats = layer.get_all_time()
                elif hasattr(layer, 'timing_stats'):
                    layer_stats = layer.timing_stats
                else:
                    continue

                for key in total_stats:
                    total_stats[key] += layer_stats[key]

        return total_stats

    def forward(self, input):
        """
        Fixed ultra-fast forward propagation for column array
        Avoids tensor reuse issues that cause gradient graph problems

        Args:
            input (torch.Tensor): Input tensor (batch, height, width, num_ports)

        Returns:
            torch.Tensor: Output tensor (batch, height, width, num_ports)
        """
        batch_size, height, width, num_ports = input.shape
        num_patches = height * width
        batch_total = batch_size * num_patches

        # Ensure input is complex type
        if not input.dtype.is_complex:
            input = input.to(dtype=torch.complex64)

        # Get transfer matrix (with caching)
        transfer_matrix = self._build_transfer_matrix(input.device)

        # Reshape 4D input to 3D for processing: (batch, height, width, ports) -> (batch, num_patches, ports)
        input_3d = input.view(batch_size, num_patches, num_ports)

        # Optimized vectorized processing with pre-allocation
        input_flat = input_3d.view(batch_total, num_ports)

        # TRAINING MODE: Always create fresh tensors to avoid gradient graph issues
        if self.training:
            # Always create new tensors during training to prevent gradient conflicts
            state_vectors = torch.zeros(batch_total, self.matrix_size,
                                      dtype=torch.complex64, device=input.device)
            state_vectors[:, :num_ports] = input_flat
        else:
            # EVAL MODE: Use pre-allocated tensors for performance
            if batch_total <= self._max_batch_size:
                # Ensure pre-allocated tensors exist and have correct size
                if (self._preallocated_state_vectors is None or
                    self._preallocated_state_vectors.size(0) < batch_total or
                    self._preallocated_state_vectors.device != input.device):
                    self._preallocate_tensors(batch_total, input.device)

                # Use pre-allocated tensors (safe in eval mode)
                state_vectors = self._preallocated_state_vectors[:batch_total]
                state_vectors.zero_()  # Clear previous values
                state_vectors[:, :num_ports] = input_flat
            else:
                # Fall back to dynamic allocation for very large batches
                state_vectors = torch.zeros(batch_total, self.matrix_size,
                                          dtype=torch.complex64, device=input.device)
                state_vectors[:, :num_ports] = input_flat

        # Batch matrix multiplication
        new_states = torch.matmul(state_vectors, transfer_matrix.T)

        # Extract output
        output_flat = new_states[:, self.num_ports:self.num_ports + num_ports]

        # Reshape back to 3D format first
        output_3d = output_flat.view(batch_size, num_patches, num_ports)

        # Reshape back to original 4D format: (batch, num_patches, ports) -> (batch, height, width, ports)
        output = output_3d.view(batch_size, height, width, num_ports)

        return output