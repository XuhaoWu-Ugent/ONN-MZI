# Required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class MZI(nn.Module):
    def __init__(self):
        """Initialize the MZI (Mach-Zehnder Interferometer) module"""
        super().__init__()
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize trainable parameter for the phase shift angle with small random values
        bound = 0.3 ** 0.5
        self.raw_sin_theta = nn.Parameter(torch.empty(1).uniform_(-bound, bound))

        #self.matrix=torch.zeros([4, 4], dtype=torch.complex64,requires_grad=False,device=self.device)
        #self.timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}

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
        Create the MZI transfer matrix (Currently not used in forward pass)
        Args:
            input_device: Device where the matrix should be created
        """
        # Get sine and cosine values and convert to complex
        sin_theta, cos_theta = self.get_sin_cos()
        sin_theta = sin_theta.to(torch.complex64)
        cos_theta = cos_theta.to(torch.complex64)
        
        # Initialize 2x2 complex matrix
        matrix = torch.zeros([2, 2], dtype=torch.complex64, device=input_device)
        # Define connection indices for the matrix
        indices = [(0,2), (1,3), (2,0), (3,1)]
        
        # Fill matrix with corresponding values
        for i, j in indices:
            self.matrix[i,j] = cos_theta
            self.matrix[i,3-i] = -1.0j * sin_theta
        #import pdb; pdb.set_trace()
        #return matrix
        
    def forward(self, input):
        """
        Forward pass through the MZI
        Args:
            input: Input tensor with shape (..., 4) representing the four ports
        Returns:
            Output tensor with transformed values
        """
        # Create CUDA events for timing (if needed)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # Calculate sine and cosine values and convert to complex
        sin_theta, cos_theta = self.get_sin_cos()
        sin_theta = sin_theta.to(torch.complex64)
        cos_theta = cos_theta.to(torch.complex64)

        # Apply cosine and sine transformations
        x_c = input * cos_theta  # Cosine component
        x_s = input * -1j * sin_theta  # Sine component

        # Combine transformed components according to MZI operation
        # Stack the results in specific order for the four output ports
          #alloc_end.record()
        '''if input.dim() == 1:
            
            result=torch.cat(cos_theta*input[2]+sin_theta*input[2],sin_theta*input[2],,)
            torch.matmul(torch.tensor(
                [[0,0,cos_theta,sin_theta],
                 [0,0,sin_theta,cos_theta],
                 [cos_theta,sin_theta,0,0],
                 [sin_theta,cos_theta,0,0]],
                device=self.device), input)'''
            
        #else:
        #print("input shape:", input.size())
        #print("result shape:", result.size())
        #end.record()
        #torch.cuda.synchronize()

        #print(f"Matmul time: {alloc_end.elapsed_time(compute_end)} ms")
        #self.timing_stats['allocation'] += start.elapsed_time(alloc_end)
        #self.timing_stats['computation'] += alloc_end.elapsed_time(end)
        #self.timing_stats['total'] += start.elapsed_time(end)
        
        return torch.stack([

            x_c[...,2] + x_s[...,3],  # Output port 1
            x_c[...,3] + x_s[...,2],  # Output port 2
            x_c[...,0] + x_s[...,1],  # Output port 3
            x_c[...,1] + x_s[...,0]   # Output port 4
        ], dim=-1)
