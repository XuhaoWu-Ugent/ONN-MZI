import torch
import numpy as np
import os

class ChaoticSource:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChaoticSource, cls).__new__(cls)
            cls._instance.data = None
            cls._instance.enabled = False
            cls._instance.scale = 1.0
            cls._instance.mode = 'replace' # 'replace' (source = noise) or 'add' (source = 1 + noise)
            cls._instance.device = None
    
        return cls._instance

    def load_data(self, path, scale=1.0, mode='replace'):
        """
        Load chaotic signal data from CSV.
        
        Args:
            path (str): Path to the CSV file.
            scale (float): Scaling factor for the noise.
            mode (str): 'replace' to use the signal as the light source directly,
                        'add' to add the signal to a unit DC source (1 + signal).
        """
        if not os.path.exists(path):
            print(f"Warning: Chaotic data file not found at {path}. Noise disabled.")
            self.enabled = False
            return

        try:
            # Load CSV using numpy (assuming no header, 2 columns: index, value)
            # Based on file snippet: -5000,-0.02545484
            data = np.loadtxt(path, delimiter=',')
            # We only want the second column (values)
            if data.ndim > 1 and data.shape[1] >= 2:
                signal = data[:, 1]
            else:
                signal = data # Fallback if 1D
            
            # Normalize signal to be zero-mean, unit variance if it's not already?
            # The user didn't ask for normalization, but raw values might be small/large.
            # For now, we use raw values * scale.
            
            self.data = torch.tensor(signal, dtype=torch.float32)
            self.scale = scale
            self.mode = mode
            self.enabled = True
            print(f"Chaotic data loaded: {len(self.data)} samples. Mode: {mode}, Scale: {scale}")
            
        except Exception as e:
            print(f"Error loading chaotic data: {e}")
            self.enabled = False

    def get_noise(self, shape, device):
        """
        Get a tensor of noise samples with the specified shape.
        
        Args:
            shape (tuple): Desired shape of the noise tensor.
            device (torch.device): Device to put the tensor on.
            
        Returns:
            torch.Tensor: Noise tensor.
        """
        if not self.enabled or self.data is None:
            # If not enabled, return 1.0 (ideal source) if multiplicative, or 0.0 if additive?
            # But the caller expects a noise factor.
            # If we are multiplying: Input * Noise. Default should be 1.0.
            return torch.ones(shape, device=device, dtype=torch.float32)
        
        # Random sampling
        num_samples = np.prod(shape)
        # Ensure we don't sample out of bounds
        indices = torch.randint(0, len(self.data), (num_samples,))
        
        # Move data to device if needed (lazy loading to device)
        if self.data.device != device:
             self.data = self.data.to(device)
             
        noise = self.data[indices].view(shape)
        
        noise = noise * self.scale
        
        if self.mode == 'add':
            # Source = 1 + Noise
            noise = 1.0 + noise
        elif self.mode == 'replace':
            # Source = Noise
            # If noise is negative, this might be physical nonsense for intensity,
            # but for AC signal it's fine.
            # Assuming the user knows what they are doing with the signal.
            pass
            
        return noise

# Global instance
chaotic_source = ChaoticSource()
