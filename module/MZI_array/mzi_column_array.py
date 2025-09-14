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

    def _compute_parameter_hash(self):
        """Compute a hash of all parameters for efficient change detection"""
        params = []
        for mzi in self.MZI:
            params.append(mzi.raw_sin_theta.data.item())
        return hash(tuple(params))

    def _is_cache_valid(self, device):
        """Fast cache validation"""
        if (self._cached_matrix is None or
            self._cached_device != device or
            self.training):  # Never cache in training mode
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

        # Get MZI matrices
        identity_4x4 = torch.eye(4, dtype=torch.complex64, device=device)
        mzi_matrices = []

        for mzi in self.MZI:
            mzi_matrix = torch.zeros((4, 4), dtype=torch.complex64, device=device)
            for j in range(4):
                mzi_matrix[:, j] = mzi(identity_4x4[j:j+1, :]).squeeze()
            mzi_matrices.append(mzi_matrix)

        # Assemble transfer matrix for column array
        for mzi_idx in range(self.num):
            port1 = mzi_idx * 2 + 1  # Ports 1,3,5,7
            port2 = mzi_idx * 2 + 2  # Ports 2,4,6,8
            mzi_matrix = mzi_matrices[mzi_idx]

            # Forward direction
            matrix[self.num_ports + port1, port1] = mzi_matrix[2, 0]
            matrix[self.num_ports + port1, port2] = mzi_matrix[2, 1]
            matrix[self.num_ports + port2, port1] = mzi_matrix[3, 0]
            matrix[self.num_ports + port2, port2] = mzi_matrix[3, 1]

            # Reverse direction
            matrix[port1, self.num_ports + port1] = mzi_matrix[0, 2]
            matrix[port1, self.num_ports + port2] = mzi_matrix[0, 3]
            matrix[port2, self.num_ports + port1] = mzi_matrix[1, 2]
            matrix[port2, self.num_ports + port2] = mzi_matrix[1, 3]

            # Self-coupling terms
            matrix[port1, port1] = mzi_matrix[0, 0]
            matrix[port1, port2] = mzi_matrix[0, 1]
            matrix[port2, port1] = mzi_matrix[1, 0]
            matrix[port2, port2] = mzi_matrix[1, 1]

            matrix[self.num_ports + port1, self.num_ports + port1] = mzi_matrix[2, 2]
            matrix[self.num_ports + port1, self.num_ports + port2] = mzi_matrix[2, 3]
            matrix[self.num_ports + port2, self.num_ports + port1] = mzi_matrix[3, 2]
            matrix[self.num_ports + port2, self.num_ports + port2] = mzi_matrix[3, 3]

        # Update cache
        if not self.training:
            self._cached_matrix = matrix.detach()
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

    def clear_cache(self):
        """Clear all caches"""
        self._cached_matrix = None
        self._cached_device = None
        self._parameter_hash = None

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
            input (torch.Tensor): Input tensor (batch, num_patches, num_ports)

        Returns:
            torch.Tensor: Output tensor (batch, num_patches, num_ports)
        """
        batch_size, num_patches, num_ports = input.shape
        batch_total = batch_size * num_patches

        # Ensure input is complex type
        if not input.dtype.is_complex:
            input = input.to(dtype=torch.complex64)

        # Get transfer matrix (with caching)
        transfer_matrix = self._build_transfer_matrix(input.device)

        # Vectorized processing without tensor reuse
        input_flat = input.view(batch_total, num_ports)

        # Create state vectors - NO REUSE to avoid gradient issues
        state_vectors = torch.zeros(batch_total, self.matrix_size,
                                  dtype=torch.complex64, device=input.device)
        state_vectors[:, :num_ports] = input_flat

        # Batch matrix multiplication
        new_states = torch.matmul(state_vectors, transfer_matrix.T)

        # Extract output
        output_flat = new_states[:, self.num_ports:self.num_ports + num_ports]

        # Reshape back to original format
        output = output_flat.view(batch_size, num_patches, num_ports)

        return output