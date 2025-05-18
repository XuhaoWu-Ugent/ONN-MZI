# Required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MZI(nn.Module):
    def __init__(self):
        """Initialize the MZI (Mach-Zehnder Interferometer) module"""
        super().__init__()
        # Initialize trainable parameter for phase shift
        self.raw_sin_theta = nn.Parameter(torch.randn(1))
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Store previous sin_theta value to check if matrix needs updating
        self.sin_theta_prev = torch.inf
        # Initialize the transfer matrix (4x4 complex matrix with gradients)
        self.matrix = torch.zeros([4,4], dtype=torch.complex64, device=self.device, requires_grad=True)

    def get_sin_cos(self):
        """
        Calculate sine and cosine values from the raw parameter
        Returns:
            sin_theta: Sine value constrained between -1 and 1
            cos_theta: Corresponding cosine value
        """
        # Transform raw parameter to sine value using sigmoid and scaling
        sin_theta = 2 * torch.sigmoid(self.raw_sin_theta) - 1
        # Calculate cosine using trigonometric identity
        cos_theta = torch.sqrt(1 - sin_theta**2)
        return sin_theta, cos_theta
        
    def get_matrix(self, input_device):
        """
        Update the MZI transfer matrix with current parameters
        Args:
            input_device: Device where the matrix should be created
        """
        # Get sine and cosine values and convert to complex
        sin_theta, cos_theta = self.get_sin_cos()
        sin_theta = sin_theta.to(torch.complex64)
        cos_theta = cos_theta.to(torch.complex64)
        
        # Fill the transfer matrix with cosine terms
        self.matrix[0, 2] = cos_theta
        self.matrix[1, 3] = cos_theta
        self.matrix[2, 0] = cos_theta
        self.matrix[3, 1] = cos_theta
        
        # Fill the transfer matrix with sine terms (with -i factor)
        self.matrix[0, 3] = -1.0j * sin_theta
        self.matrix[1, 2] = -1.0j * sin_theta
        self.matrix[2, 1] = -1.0j * sin_theta
        self.matrix[3, 0] = -1.0j * sin_theta
        
    def forward(self, input):
        """
        Forward pass through the MZI
        Args:
            input: Input tensor representing the four ports
        Returns:
            output: Transformed tensor after MZI operation
        """
        # Update matrix only if parameter has changed
        if self.raw_sin_theta.item() != self.sin_theta_prev:
            self.get_matrix(input.device)
            self.sin_theta_prev = self.raw_sin_theta.item()

        # Handle different input dimensions
        if input.dim() == 1:
            # For 1D input, multiply matrix directly
            output = torch.matmul(self.matrix, input)
        else:
            # For batch inputs, multiply with transposed matrix
            output = torch.matmul(input, self.matrix.T)
            
        return output
