import torch
import torch.nn as nn
import torch.nn.functional as F
from module.MZI_array.mzi_column_array import MZIlayer_column
from module.MZI_array.mzi_row_array import MZIlayer_row


class SingleChannelFilterPower(nn.Module):
    """
    Single channel filter with power-domain linear superposition

    Core principle: Process each input port independently through MZI array,
    then perform power-domain summation to avoid coherent interference.
    This simulates multiple independent optical sources (e.g., frequency comb teeth,
    incoherent sources) where outputs add in power, not amplitude.
    """
    def __init__(self, mzi_row_num=5, mzi_column_num=4, repeat_num=5, kernel_size=3):
        """
        Initialize single channel filter with power-domain processing

        Args:
            mzi_row_num (int): Number of MZIs in row arrays, default 5
            mzi_column_num (int): Number of MZIs in column arrays, default 4
            repeat_num (int): Number of MZI layer repetitions, default 5
            kernel_size (int): Size of the convolution kernel, default 3
        """
        super(SingleChannelFilterPower, self).__init__()

        # Store parameters
        self.mzi_row_num = mzi_row_num
        self.mzi_column_num = mzi_column_num
        self.repeat_num = repeat_num
        self.kernel_size = kernel_size
        self.num_ports = mzi_row_num * 2  # 10 ports for compatibility

        # Create alternating row and column MZI layers
        self.layers = nn.ModuleList()
        for _ in range(repeat_num):
            self.layers.append(MZIlayer_row(num=mzi_row_num))
            self.layers.append(MZIlayer_column(num=mzi_column_num))
        self.layers.append(MZIlayer_row(num=mzi_row_num))

        # Trainable diagonal weight matrix
        self.diagonal_matrix = nn.Parameter(torch.randn(self.num_ports))

        # Store matrix size for dynamic computation
        self.matrix_size = 2 * self.num_ports  # 20x20

        # Cached combined transfer matrix (only in eval mode)
        self._combined_matrix = None

    def _get_combined_matrix(self, device):
        """
        Get combined transfer matrix with training-safe caching
        Always rebuild in training mode to maintain gradient graph
        """
        # TRAINING MODE: Always rebuild matrix
        if self.training:
            return self._build_combined_transfer_matrix(device)

        # EVAL MODE: Use caching for performance
        if self._combined_matrix is None or self._combined_matrix.device != device:
            self._combined_matrix = self._build_combined_transfer_matrix(device)
            self._combined_matrix = self._combined_matrix.detach().clone()

        return self._combined_matrix

    def _build_combined_transfer_matrix(self, device):
        """
        Build combined transfer matrix by multiplying all layer matrices
        """
        # Start with identity matrix
        combined_matrix = torch.eye(self.matrix_size, dtype=torch.complex64, device=device)

        # Multiply matrices in reverse order (right to left multiplication)
        for layer in reversed(self.layers):
            layer_matrix = layer._build_transfer_matrix(device)
            combined_matrix = torch.matmul(layer_matrix, combined_matrix)

        return combined_matrix

    def train(self, mode=True):
        """Override train() to clear cache when switching to training mode"""
        super().train(mode)
        if mode:
            self._combined_matrix = None
        return self

    def forward(self, x, return_intermediate=False):
        """
        Forward propagation using power-domain linear superposition

        Key principle:
        1. Process each input port independently through MZI array
        2. Convert each output to power (|amplitude|^2)
        3. Sum powers across all ports (linear superposition in power domain)
        4. Apply trainable weights and spatial summation

        Args:
            x (Tensor): Input tensor of shape (batch_size, height, width)
            return_intermediate (bool): Whether to return intermediate values

        Returns:
            Tensor or tuple: Output tensor, or (output, input_patch, processed_patch)
        """
        # Get input dimensions
        batch_size, height, width = x.shape

        # Calculate output dimensions
        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1

        # Extract image patches
        patches = F.unfold(x.unsqueeze(1),
                          kernel_size=self.kernel_size,
                          stride=1)  # Shape: (batch_size, kernel_size^2, num_patches)

        patches = patches.permute(0, 2, 1)  # Shape: (batch_size, num_patches, kernel_size^2)

        # Pad to match MZI port count (9 -> 10)
        patches = F.pad(patches, (0, 1))  # Shape: (batch_size, num_patches, 10)

        # Save first patch input if needed
        if return_intermediate:
            input_patch = patches[:, 0, :].clone().detach()

        # Convert to complex type
        patches = patches.to(torch.complex64)

        # Get dimensions
        batch_size, num_patches, num_ports = patches.shape
        device = patches.device

        # Get combined transfer matrix
        combined_matrix = self._get_combined_matrix(device)

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
            # CORE: Power-domain linear superposition (no coherent interference)
            port_power = torch.abs(port_outputs) ** 2
            output_powers += port_power

        # Save processed first patch if needed
        if return_intermediate:
            processed_patch = output_powers[:, 0, :].clone().detach()

        # Apply trainable diagonal weights
        # Note: We apply weights to power (not amplitude), so use diagonal_matrix directly
        # For single input: |w*A|^2 = |w|^2 * |A|^2
        # For multiple inputs with power superposition: sum(|w*A_i|^2) = sum(|w|^2 * |A_i|^2) = |w|^2 * sum(|A_i|^2)
        weighted_output = output_powers * (self.diagonal_matrix.abs() ** 2).view(1, 1, -1)

        # Sum across all output ports and reshape to spatial dimensions
        output = weighted_output.sum(dim=2).view(batch_size, output_height, output_width)

        # Return results
        if return_intermediate:
            return output, input_patch, processed_patch
        else:
            return output
