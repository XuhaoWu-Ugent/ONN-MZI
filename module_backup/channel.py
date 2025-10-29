import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.MZI_array.mzi_column_array import MZIlayer_column
from module.MZI_array.mzi_row_array import MZIlayer_row


class SingleChannelFilter(nn.Module):
    """
    Unified single channel optical filter supporting both detection modes

    Detection Modes:
        - 'coherent': Complex amplitude interference - processes all inputs coherently
                     through the MZI array with phase information preserved
        - 'power': Power-domain linear superposition - processes each input port
                   independently and sums output powers (no coherent interference)

    Architecture:
        Input -> MZI Array (Row-Column layers) -> Detection -> Diagonal Weights -> Output
    """
    
    def __init__(self, mzi_row_num=5, mzi_column_num=4, repeat_num=5, kernel_size=3,
                 detection_mode='coherent'):
        
        """
        Initialize single channel filter with configurable detection mode

        Args:
            mzi_row_num (int): Number of MZIs in row arrays, default 5
            mzi_column_num (int): Number of MZIs in column arrays, default 4
            repeat_num (int): Number of MZI layer repetitions, default 5
            kernel_size (int): Size of the convolution kernel, default 3
            detection_mode (str): 'coherent' or 'power', default 'coherent'
        """
        super(SingleChannelFilter, self).__init__()

        # Validate detection mode
        if detection_mode not in ['coherent', 'power']:
            raise ValueError(f"detection_mode must be 'coherent' or 'power', got {detection_mode}")

        # Store parameters
        self.mzi_row_num = mzi_row_num
        self.mzi_column_num = mzi_column_num
        self.repeat_num = repeat_num
        self.kernel_size = kernel_size
        self.detection_mode = detection_mode
        self.num_ports = mzi_row_num * 2  # 10 ports for compatibility

        # Create alternating row and column MZI layers
        # Structure: Row -> Column -> Row -> Column -> ... -> Row (2*repeat_num + 1 layers)
        self.layers = nn.ModuleList()
        for _ in range(repeat_num):
            self.layers.append(MZIlayer_row(num=mzi_row_num))
            self.layers.append(MZIlayer_column(num=mzi_column_num))
        self.layers.append(MZIlayer_row(num=mzi_row_num))

        # Trainable diagonal weight matrix
        self.diagonal_matrix = nn.Parameter(torch.randn(self.num_ports))

        # Other parameters
        self.flag = 0
        self.timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}

        # Store matrix size for dynamic computation
        self.matrix_size = 2 * self.num_ports  # 20x20

        # Pre-allocate combined transfer matrix (cached only in eval mode)
        self._combined_matrix = None

    def _get_combined_matrix(self, device):
        
        """
        Get combined transfer matrix with training-safe caching

        CRITICAL: Always rebuild in training mode to prevent gradient graph issues
        In eval mode, cache the matrix for performance optimization

        Args:
            device: Target device for the matrix

        Returns:
            torch.Tensor: Combined transfer matrix of shape (matrix_size, matrix_size)
        """
        # TRAINING MODE: Always rebuild matrix to avoid gradient conflicts
        
        if self.training:
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
        Build combined transfer matrix by multiplying all MZI layer matrices

        Follows the sequence: Row -> Column -> Row -> Column -> ... -> Row
        Matrix multiplication is performed in reverse order for proper composition

        Args:
            device: Target device for the matrix

        Returns:
            torch.Tensor: Combined transfer matrix
        """
        # Start with identity matrix
        combined_matrix = torch.eye(self.matrix_size, dtype=torch.complex64, device=device)

        # Multiply matrices in reverse order (right to left multiplication)
        # This ensures proper composition: output = M_n * M_{n-1} * ... * M_1 * input
        for layer in reversed(self.layers):
            layer_matrix = layer._build_transfer_matrix(device)
            combined_matrix = torch.matmul(layer_matrix, combined_matrix)

        return combined_matrix

    def _coherent_forward(self, patches, combined_matrix, return_intermediate=False):
        """
        Coherent detection: Process all inputs together through MZI array

        Complex amplitudes interfere coherently, preserving phase information.
        This is the traditional optical computing approach.

        Args:
            patches (torch.Tensor): Input patches, shape (batch_size, num_patches, num_ports)
            combined_matrix (torch.Tensor): Combined MZI transfer matrix
            return_intermediate (bool): Whether to save intermediate values

        Returns:
            tuple: (output_patches, processed_patch) if return_intermediate else output_patches
        """
        batch_size, num_patches, num_ports = patches.shape

        # Batch processing - flatten batch and patch dimensions
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

        # Save processed first patch if needed
        processed_patch = None
        if return_intermediate:
            processed_patch = output_patches[:, 0, :].clone().detach()

        return output_patches, processed_patch

    def _power_forward(self, patches, combined_matrix, return_intermediate=False):
        """
        Power detection: Process each input port independently, sum output powers

        Simulates multiple independent optical sources (e.g., frequency comb teeth,
        incoherent sources) where outputs add in power domain, not amplitude.
        No coherent interference between different input ports.

        Args:
            patches (torch.Tensor): Input patches, shape (batch_size, num_patches, num_ports)
            combined_matrix (torch.Tensor): Combined MZI transfer matrix
            return_intermediate (bool): Whether to save intermediate values

        Returns:
            tuple: (output_powers, processed_patch) if return_intermediate else output_powers
        """
        batch_size, num_patches, num_ports = patches.shape
        device = patches.device

        # Initialize power accumulator
        output_powers = torch.zeros(batch_size, num_patches, num_ports,
                                   dtype=torch.float32, device=device)

        # Process each input port independently
        batch_total = batch_size * num_patches

        for port_idx in range(num_ports):
            # Extract input for this port across all patches
            port_input = patches[:, :, port_idx].reshape(batch_total)

            # Create state vectors (input only at this specific port)
            state_vectors = torch.zeros(batch_total, self.matrix_size,
                                       dtype=torch.complex64, device=device)
            state_vectors[:, port_idx] = port_input

            # Process through MZI array
            new_states = torch.matmul(state_vectors, combined_matrix.T)

            # Extract output ports
            port_outputs = new_states[:, self.num_ports:self.num_ports + num_ports]
            port_outputs = port_outputs.view(batch_size, num_patches, num_ports)

            # Convert to power and accumulate
            # CORE PRINCIPLE: Power-domain linear superposition (no coherent interference)
            port_power = torch.abs(port_outputs) ** 2
            output_powers += port_power

        # Save processed first patch if needed
        processed_patch = None
        if return_intermediate:
            processed_patch = output_powers[:, 0, :].clone().detach()

        return output_powers, processed_patch

    def get_all_time(self):
        """
        Recursively collect timing statistics from all layers

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
        """
        Override train() to clear cache when switching to training mode

        This prevents gradient graph issues from cached matrices
        """
        super().train(mode)
        if mode:  # Entering training mode
            self._combined_matrix = None  # Clear cached matrix
        return self

    def forward(self, x, return_intermediate=False):
        """
        Forward propagation function using combined transfer matrix

        The forward process depends on the detection_mode:
        - coherent: All inputs processed together, complex amplitude summation
        - power: Each input port processed independently, power summation

        Args:
            x (Tensor): Input tensor of shape (batch_size, height, width)
            return_intermediate (bool): Whether to return intermediate values for the first patch

        Returns:
            Tensor or tuple: Output tensor, or tuple of (output, input_patch, processed_patch)
                           if return_intermediate=True
        """
        # Get input dimensions
        batch_size, height, width = x.shape

        # Calculate output dimensions
        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1

        # Extract image patches
        # Add channel dimension and unfold into patches
        patches = F.unfold(x.unsqueeze(1),
                          kernel_size=self.kernel_size,
                          stride=1)  # Shape: (batch_size, kernel_size^2, num_patches)

        # Adjust dimension order
        patches = patches.permute(0, 2, 1)  # Shape: (batch_size, num_patches, kernel_size^2)

        # Pad to match MZI port count (9 -> 10)
        patches = F.pad(patches, (0, 1))  # Shape: (batch_size, num_patches, 10)

        # Save first patch input if needed
        input_patch = None
        if return_intermediate:
            input_patch = patches[:, 0, :].clone().detach()

        # Convert to complex type for MZI processing
        patches = patches.to(torch.complex64)

        # Get combined transfer matrix
        combined_matrix = self._get_combined_matrix(patches.device)

        # Process through MZI array based on detection mode
        if self.detection_mode == 'coherent':
            output_patches, processed_patch = self._coherent_forward(
                patches, combined_matrix, return_intermediate
            )
            # Apply trainable diagonal weights to complex amplitudes
            weighted_output = output_patches * self.diagonal_matrix.view(1, 1, -1)

        else:  # 'power'
            output_powers, processed_patch = self._power_forward(
                patches, combined_matrix, return_intermediate
            )
            # Apply trainable diagonal weights to powers
            # For power domain: weight^2 is applied to power values
            weighted_output = output_powers * (self.diagonal_matrix.abs() ** 2).view(1, 1, -1)

        # Sum across all output ports and reshape to spatial dimensions
        output = weighted_output.sum(dim=2).view(batch_size, output_height, output_width)

        # Return based on parameters
        if return_intermediate:
            return output, input_patch, processed_patch
        else:
            return output
