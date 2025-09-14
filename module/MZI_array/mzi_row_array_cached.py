# Required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.MZI_array.mzi import MZI


class MZIlayer_row(nn.Module):
    """
    Cached MZI Row Array implementation - FOR TESTING PURPOSES ONLY
    This version caches the transfer matrix to test if parameter updates work properly
    WARNING: This approach may break gradient flow and parameter updates
    """

    def __init__(self, num=5):
        """
        Args:
            num (int): Number of MZIs, default 5 (corresponding to 10-port system)
        """
        super(MZIlayer_row, self).__init__()
        self.num = num
        self.num_ports = num * 2

        # Create MZI instances (maintain consistent naming with original interface)
        self.MZI = nn.ModuleList([MZI() for _ in range(num)])
        self.mzis = self.MZI  # Backward compatibility alias

        # Add attributes compatible with original implementation
        self.flag = 0
        self.timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}

        # Store matrix size for dynamic computation
        self.matrix_size = 2 * self.num_ports

        # Cache-related attributes
        self._cached_matrix = None
        self._cached_device = None
        self._cached_parameters = None
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_current_parameters(self):
        """Get current parameter values as a hashable tuple for cache validation"""
        params = []
        for mzi in self.MZI:
            params.append(mzi.raw_sin_theta.data.item())
        return tuple(params)

    def _is_cache_valid(self, device):
        """Check if cached matrix is still valid"""
        if (self._cached_matrix is None or
            self._cached_device != device or
            self.training):  # Never cache in training mode for safety
            return False

        current_params = self._get_current_parameters()
        return current_params == self._cached_parameters

    def _build_transfer_matrix(self, device):
        """Build transfer matrix with caching"""
        # Check cache validity first
        if self._is_cache_valid(device):
            self._cache_hits += 1
            return self._cached_matrix

        self._cache_misses += 1

        matrix = torch.zeros((self.matrix_size, self.matrix_size), dtype=torch.complex64, device=device)

        # Get 4x4 transfer matrix for each MZI
        mzi_matrices = []
        for mzi in self.MZI:
            identity_4x4 = torch.eye(4, dtype=torch.complex64, device=device)
            mzi_matrix = torch.zeros((4, 4), dtype=torch.complex64, device=device)

            for i in range(4):
                input_vec = identity_4x4[i:i + 1, :]
                output_vec = mzi(input_vec)
                mzi_matrix[:, i] = output_vec.squeeze()

            mzi_matrices.append(mzi_matrix)

        # Row Array connections: each MZI processes consecutive port pairs [0,1], [2,3], [4,5], [6,7], [8,9]
        # Support full bidirectional propagation
        for mzi_idx in range(self.num):
            port1 = mzi_idx * 2
            port2 = mzi_idx * 2 + 1
            mzi_matrix = mzi_matrices[mzi_idx]

            # Complete 4x4 MZI matrix mapping for bidirectional propagation
            # Forward direction: input side -> output side
            matrix[self.num_ports + port1, port1] = mzi_matrix[2, 0]
            matrix[self.num_ports + port1, port2] = mzi_matrix[2, 1]
            matrix[self.num_ports + port2, port1] = mzi_matrix[3, 0]
            matrix[self.num_ports + port2, port2] = mzi_matrix[3, 1]

            # Reverse direction: output side -> input side
            matrix[port1, self.num_ports + port1] = mzi_matrix[0, 2]
            matrix[port1, self.num_ports + port2] = mzi_matrix[0, 3]
            matrix[port2, self.num_ports + port1] = mzi_matrix[1, 2]
            matrix[port2, self.num_ports + port2] = mzi_matrix[1, 3]

            # Self-coupling terms (within same side)
            matrix[port1, port1] = mzi_matrix[0, 0]
            matrix[port1, port2] = mzi_matrix[0, 1]
            matrix[port2, port1] = mzi_matrix[1, 0]
            matrix[port2, port2] = mzi_matrix[1, 1]

            matrix[self.num_ports + port1, self.num_ports + port1] = mzi_matrix[2, 2]
            matrix[self.num_ports + port1, self.num_ports + port2] = mzi_matrix[2, 3]
            matrix[self.num_ports + port2, self.num_ports + port1] = mzi_matrix[3, 2]
            matrix[self.num_ports + port2, self.num_ports + port2] = mzi_matrix[3, 3]

        # Update cache (only in eval mode)
        if not self.training:
            self._cached_matrix = matrix.detach()  # Detach from computation graph!
            self._cached_device = device
            self._cached_parameters = self._get_current_parameters()

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

    def clear_cache(self):
        """Manually clear the cache"""
        self._cached_matrix = None
        self._cached_device = None
        self._cached_parameters = None

    def get_all_time(self):
        """
        Recursively collect timing statistics from all MZI modules
        Returns:
            Dictionary containing accumulated timing statistics
        """
        total_stats = {'allocation': 0, 'computation': 0, 'total': 0}

        # Add current layer's timing stats
        for key in total_stats:
            if hasattr(self, 'timing_stats'):
                total_stats[key] += self.timing_stats[key]

        # Recursively add timing stats from all child MZI modules
        if hasattr(self, 'MZI'):
            for layer in self.MZI:
                # If child has get_all_time method, call it
                if hasattr(layer, 'get_all_time'):
                    layer_stats = layer.get_all_time()
                # Otherwise try to get timing_stats directly
                elif hasattr(layer, 'timing_stats'):
                    layer_stats = layer.timing_stats
                else:
                    continue

                # Accumulate child layer timing stats
                for key in total_stats:
                    total_stats[key] += layer_stats[key]

        return total_stats

    def forward(self, input):
        """
        Forward propagation with matrix caching - FOR TESTING ONLY

        Args:
            input (torch.Tensor): Input tensor (batch, num_patches, num_ports)

        Returns:
            torch.Tensor: Output tensor (batch, num_patches, num_ports)
        """
        batch_size, num_patches, num_ports = input.shape

        # Ensure input is complex type for optical processing
        if not input.dtype.is_complex:
            input = input.to(dtype=torch.complex64)

        # Convert input to dynamic-dimensional state vector
        output = torch.zeros_like(input, dtype=torch.complex64)

        # Build transfer matrix with caching
        transfer_matrix = self._build_transfer_matrix(input.device)

        for b in range(batch_size):
            for p in range(num_patches):
                # Build state vector: [input state, zero output state]
                state_vector = torch.zeros(self.matrix_size, dtype=torch.complex64, device=input.device)
                state_vector[:num_ports] = input[b, p, :]

                # Matrix transformation
                new_state = torch.matmul(transfer_matrix, state_vector)

                # Extract output state
                output[b, p, :] = new_state[self.num_ports:self.num_ports + num_ports]

        return output