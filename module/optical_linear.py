"""
Optical Linear Layer (non-shared, Dual-Path Differential)

Refactored based on CNN.py design pattern:
- Pre-create all slice processors in __init__ (similar to multiple filters in CNN_layer)
- Each processor creates its own MZI array and manages its transfer matrix cache
- Use batch matrix operations in forward, iterating only through processors

Previously named `OpticalLoRALinear`; renamed because the architecture
is not related to LoRA. A backward-compat alias `OpticalLoRALinear` is
kept at module bottom.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from module.MZI_array.mzi import redheffer_star_product
from module.MZI_array.mzi_row_array import MZIlayer_row
from module.MZI_array.mzi_column_array import MZIlayer_column


class OpticalSliceProcessor(nn.Module):
    """
    Single slice processor for vector processing.

    Processes [Batch, 10] vector input through a 10x10 MZI array (50 MZIs).
    Uses Clements architecture: alternating Row-Column MZI layers.

    Key optimizations (referencing SingleChannelFilter):
    - Pre-create all MZI layers in __init__
    - Use batch matrix operations to process the entire batch in forward
    - Cache transfer matrix (in eval mode)
    """
    def __init__(self, mzi_row_num=5, mzi_column_num=4, repeat_num=5, start_index=0):
        super(OpticalSliceProcessor, self).__init__()

        # Pre-create MZI layers in __init__ (Clements architecture)
        self.layers = nn.ModuleList()
        next_index = start_index

        # Repeat (row + column) layers, add one final row layer
        for rep in range(repeat_num):
            self.layers.append(MZIlayer_row(num=mzi_row_num, start_index=next_index))
            next_index += mzi_row_num
            self.layers.append(MZIlayer_column(num=mzi_column_num, start_index=next_index))
            next_index += mzi_column_num

        # Final row layer
        self.layers.append(MZIlayer_row(num=mzi_row_num, start_index=next_index))
        next_index += mzi_row_num

        self.total_mzis = next_index - start_index  # 5*(5+4) + 5 = 50

        # Configuration
        self.num_ports = mzi_row_num * 2  # 10
        self.matrix_size = 2 * self.num_ports  # 20

        # Cache for transfer matrix (similar to SingleChannelFilter)
        self._combined_matrix = None

        # Hook data storage (for recording optical signals and voltages)
        self.hook_data = {
            'optical_input': None,      # Optical input [Batch, 10]
            'optical_output': None,     # Optical output [Batch, 10]
            'mzi_voltages': None,       # MZI voltages [total_mzis]
        }
        self.enable_hook = False  # Default: hooks disabled

    def _get_combined_matrix(self, device):
        """
        Get combined transfer matrix (modified: cached even during training).

        Training mode: Use cache (performance optimization), but clear at start of each forward.
        Eval mode: Use cache and detach.
        """
        # Check if rebuild is needed
        should_update = (self._combined_matrix is None or
                        self._combined_matrix.device != device)

        if should_update:
            self._combined_matrix = self._build_combined_matrix(device)
            # Eval mode: detach to save memory
            if not self.training:
                self._combined_matrix = self._combined_matrix.detach().clone()

        return self._combined_matrix

    def clear_cache(self):
        """Clear cache (called at start of forward to avoid cross-batch gradient graph conflicts)."""
        self._combined_matrix = None

    def _build_combined_matrix(self, device):
        """
        Build combined transfer matrix using Redheffer star product.

        Returns:
            [matrix_size, matrix_size] complex transfer matrix
        """
        N = self.num_ports  # single-side port count

        # Redheffer identity: zero reflection, perfect transmission
        combined = torch.zeros(
            self.matrix_size, self.matrix_size,
            dtype=torch.complex64, device=device,
        )
        combined[:N, N:] = torch.eye(N, dtype=torch.complex64, device=device)
        combined[N:, :N] = torch.eye(N, dtype=torch.complex64, device=device)

        for layer in self.layers:
            layer_matrix = layer._build_transfer_matrix(device)
            combined = redheffer_star_product(combined, layer_matrix, N)

        return combined

    def forward(self, x):
        """
        Process vector through MZI array (Power detection mode, batch processing).

        Referencing batch processing logic from SingleChannelFilter._power_forward.

        Args:
            x: [Batch, 10] vector input (Power domain, non-negative)
        Returns:
            [Batch, 10] vector output (Power domain)
        """
        batch_size = x.size(0)
        device = x.device

        # Hook: Record input optical signal
        if self.enable_hook:
            self.hook_data['optical_input'] = x.detach().cpu().clone()

        # Get transfer matrix (with caching)
        combined_matrix = self._get_combined_matrix(device)

        # Hook: Record MZI voltages
        if self.enable_hook:
            voltages = []
            for layer in self.layers:
                if hasattr(layer, 'MZI'):
                    for mzi in layer.MZI:
                        voltages.append(mzi.get_voltage().item())
            self.hook_data['mzi_voltages'] = torch.tensor(voltages)

        # Input x is Optical Power P.
        # MZI processes Electric Field Amplitude E.
        # Relation: P = |E|^2  =>  E = sqrt(P)
        # We must take the square root to convert to amplitude before matrix multiplication,
        # then square the result to convert back to power.
        # This ensures Power In -> Power Out is linear (P_out = M * P_in).
        # Otherwise P_out proportional to P_in^2, causing exponential explosion in deep networks.
        x_amplitude = torch.sqrt(x + 1e-8)  # epsilon prevent NaN gradient

        # Convert to complex
        x_complex = x_amplitude.to(torch.complex64)

        # Power mode: Process port by port (physically required).
        # But use batch matrix multiplication to process the whole batch at once.
        output_powers = torch.zeros(batch_size, self.num_ports,
                                    dtype=torch.float32, device=device)

        for port_idx in range(self.num_ports):
            # Create state vectors (process entire batch)
            state_vectors = torch.zeros(batch_size, self.matrix_size,
                                       dtype=torch.complex64, device=device)
            state_vectors[:, port_idx] = x_complex[:, port_idx]

            # Batch matrix multiplication
            new_states = torch.matmul(state_vectors, combined_matrix.T)

            # Extract output ports
            port_outputs = new_states[:, self.num_ports:self.num_ports + self.num_ports]

            # Convert to power and accumulate
            output_powers += torch.abs(port_outputs) ** 2

        # Hook: Record output optical signal
        if self.enable_hook:
            self.hook_data['optical_output'] = output_powers.detach().cpu().clone()

        return output_powers


class OpticalLinear(nn.Module):
    """
    Optical Linear Layer (Dual-Path Differential Architecture, non-shared)

    Implements true signed weights using differential detection:
    Y = (Y_pos - Y_neg) + Bias
    Where Y_pos and Y_neg are calculated by independent MZI arrays.

    Mathematical Model:
    - Encoder: Processes input slices using pos/neg MZI arrays.
    - Decoder: Processes encoded features using pos/neg MZI arrays.
    - Differential Output: Subtraction in electrical domain.

    This architecture solves the limitation of non-negative weights in incoherent optical systems.
    """

    def __init__(self, 
                 in_features,
                 out_features,
                 r=10,
                 activation_mode='linear',
                 detection_mode='power',
                 mzi_row_num=5,
                 mzi_column_num=4,
                 repeat_num=5,
                 start_index=0):
        super(OpticalLinear, self).__init__()

        # Validate parameters
        if r != 10:
            raise ValueError(f"Rank r must be 10 (hardware constraint), got {r}")
        if detection_mode != 'power':
            raise ValueError(f"Only 'power' detection mode is supported, got {detection_mode}")
        if activation_mode not in ['linear', 'nonlinear']:
            raise ValueError(f"activation_mode must be 'linear' or 'nonlinear', got {activation_mode}")

        # Store configuration
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.activation_mode = activation_mode
        self.detection_mode = detection_mode

        # Calculate slices and padding
        self.n_slices = math.ceil(in_features / r)
        self.padded_in = self.n_slices * r

        # === POSITIVE PATH ===
        # Encoder
        self.pos_encoder_slices = nn.ModuleList()
        next_index = start_index
        for slice_idx in range(self.n_slices):
            processor = OpticalSliceProcessor(
                mzi_row_num=mzi_row_num,
                mzi_column_num=mzi_column_num,
                repeat_num=repeat_num,
                start_index=next_index
            )
            self.pos_encoder_slices.append(processor)
            next_index += processor.total_mzis
        self.pos_encoder_mzis = next_index - start_index
        
        # Decoder
        pos_decoder_start = next_index
        self.pos_decoder_slice = OpticalSliceProcessor(
            mzi_row_num=mzi_row_num,
            mzi_column_num=mzi_column_num,
            repeat_num=repeat_num,
            start_index=next_index
        )
        next_index += self.pos_decoder_slice.total_mzis

        # === NEGATIVE PATH ===
        # Encoder
        neg_encoder_start = next_index
        self.neg_encoder_slices = nn.ModuleList()
        for slice_idx in range(self.n_slices):
            processor = OpticalSliceProcessor(
                mzi_row_num=mzi_row_num,
                mzi_column_num=mzi_column_num,
                repeat_num=repeat_num,
                start_index=next_index
            )
            self.neg_encoder_slices.append(processor)
            next_index += processor.total_mzis
        self.neg_encoder_mzis = next_index - neg_encoder_start

        # Decoder
        neg_decoder_start = next_index
        self.neg_decoder_slice = OpticalSliceProcessor(
            mzi_row_num=mzi_row_num,
            mzi_column_num=mzi_column_num,
            repeat_num=repeat_num,
            start_index=next_index
        )
        next_index += self.neg_decoder_slice.total_mzis

        self.total_mzis = next_index - start_index

        # Output projection (if out_features != r)
        if out_features != r:
            self.output_proj = nn.Linear(r, out_features, bias=False)
        else:
            self.output_proj = None

        # Trainable Bias (Applied in electrical domain after differential detection)
        # Initialize with small random values to break symmetry, especially important
        # when hardware errors might dampen the initial differential signal.
        self.bias = nn.Parameter(torch.randn(r) * 0.1)

        # Print architecture info
        self._print_architecture()

    def _print_architecture(self):
        """Print architecture summary."""
        print(f"\n{'='*70}")
        print(f"OpticalLinear Initialized (Dual-Path Differential)")
        print(f"{'='*70}")
        print(f"Input Features: {self.in_features}")
        print(f"Output Features: {self.out_features}")
        print(f"Rank: {self.r}")
        print(f"Detection Mode: {self.detection_mode}")
        print(f"Architecture: Positive Path - Negative Path + Bias")
        print(f"\nEncoder:")
        print(f"  - Slice processors: {self.n_slices} x 2 (Pos/Neg)")
        print(f"  - Input padding: {self.in_features} -> {self.padded_in}")
        print(f"\nDecoder:")
        print(f"  - Slice processors: 1 x 2 (Pos/Neg)")
        print(f"\nTotal:")
        print(f"  - Total MZIs: {self.total_mzis}")
        print(f"  - Total slice processors: {(self.n_slices + 1) * 2}")
        if self.output_proj is not None:
            print(f"  - Output projection: {self.r} -> {self.out_features}")
        print(f"{'='*70}\n")

    def forward(self, x):
        """
        Forward propagation (Dual-Path Differential).

        Args:
            x (Tensor): Input of shape (batch_size, in_features)

        Returns:
            Tensor: Output of shape (batch_size, out_features)
        """
        # Training mode: Clear all caches
        if self.training:
            for p in self.pos_encoder_slices: p.clear_cache()
            for p in self.neg_encoder_slices: p.clear_cache()
            self.pos_decoder_slice.clear_cache()
            self.neg_decoder_slice.clear_cache()

        # === POSITIVE PATH ===
        pos_hidden = self.forward_encoder(x, self.pos_encoder_slices)
        pos_output = self.forward_decoder(pos_hidden, self.pos_decoder_slice)

        # === NEGATIVE PATH ===
        neg_hidden = self.forward_encoder(x, self.neg_encoder_slices)
        neg_output = self.forward_decoder(neg_hidden, self.neg_decoder_slice)

        # === DIFFERENTIAL OUTPUT ===
        # Output = (Pos - Neg) + Bias
        output = (pos_output - neg_output) + self.bias

        # 4. Output projection (if needed)
        if self.output_proj is not None:
            output = self.output_proj(output)

        return output

    def forward_encoder(self, x, slice_processors):
        """
        Generic Encoder Forward Pass (Vectorized Version).
        Executes all slice processors in parallel using batched matrix multiplication.
        """
        batch_size = x.size(0)
        device = x.device
        num_slices = self.n_slices
        num_ports = slice_processors[0].num_ports      # 10
        matrix_size = slice_processors[0].matrix_size  # 20

        # Step 1: Padding
        if self.padded_in > self.in_features:
            x = F.pad(x, (0, self.padded_in - self.in_features))

        # Step 2: Reshape to slices [Batch, n_slices, 10]
        # This x represents Optical Power (P)
        x_sliced = x.view(batch_size, num_slices, self.r)

        # Step 3: Collect Transfer Matrices [n_slices, 20, 20]
        # We still need to loop to build/get matrices, but this avoids the data processing overhead in the loop
        matrices_list = [p._get_combined_matrix(device) for p in slice_processors]
        combined_matrices = torch.stack(matrices_list)  # Shape: (S, 20, 20)

        # Step 4: Prepare Input Vector (Convert Power to Amplitude)
        # Relation: E = sqrt(P)
        x_amplitude = torch.sqrt(torch.abs(x_sliced) + 1e-8)
        x_complex = x_amplitude.to(torch.complex64)

        # Step 5: Prepare State Vectors [Batch, Slices, 20]
        # Pad the 10-port input to 20-port state vector
        # (The MZI matrix is 20x20, input enters first 10 ports)
        state_vectors = torch.zeros(batch_size, num_slices, matrix_size,
                                  dtype=torch.complex64, device=device)
        state_vectors[:, :, :num_ports] = x_complex

        # Step 6: Vectorized Matrix Multiplication
        # Input: (Batch, Slices, 20)
        # Matrix: (Slices, 20, 20) -> Transpose to (Slices, 20, 20) for y = M*x
        # But our MZI formulation y = x * M.T is standard in Linear layers
        # Code check: previously "matmul(state_vectors, combined_matrix.T)"
        # So effective op: state_vec (1x20) * M.T (20x20)
        # Einstein Summation:
        # b: batch, s: slice, i: input_dim (20), j: output_dim (20)
        # state_vectors: b s i
        # combined_matrices (T): s j i (because we want M.T, so we use M directly with correct indices)
        # Actually: M is (20,20). We want x @ M.T.
        # Element-wise: out[b,s,j] = sum_i ( x[b,s,i] * M[s,j,i] )
        # Wait, if M is transfer matrix, usually y = M x.
        # But Pytorch Linear is x A.T.
        # Previous code: combined = matmul(layer, combined). (Right multiplication accumulation).
        # Previous forward: matmul(state, combined.T).
        # So we want to multiply by combined.T.
        # Let's use einsum for clarity:
        # x: [batch, slice, in_port]
        # M: [slice, out_port, in_port] (This represents M.T if M is [out, in])
        # Let's trust combined_matrices is M.
        # We want x @ M.T.
        # x: (b, s, i)
        # M: (s, j, i)  <- normally (s, i, j), but we want transpose, so we interact with dim 1?
        # Let's standard: M is (s, a, b). Transpose is (s, b, a).
        # x (b, s, a) @ M.T (s, b, a) -> (b, s, b)? No.
        # Target: (b,s,20)
        # Matmul: (b,s,20) x (s,20,20) = (b,s,20)
        # We want to multiply each (b,s,:) vector by the TRANSPOSE of matrices[s].
        # matrices.transpose(1, 2) gives (s, 20, 20) where each matrix is transposed.
        # But torch.matmul doesn't support broadcasting (Batch, Slice) against (Slice).
        # We use einsum.
        # x: b s i
        # M: s j i (This is M[s] with j,i. We want to sum over i. So this is M[s] where i is cols).
        # Note: M is (Row, Col). Transpose means we sum over Rows of M?
        # Previous: x @ M.T.
        # x[k] * M.T[k, j] = x[k] * M[j, k].
        # So we sum over the second dimension of M (columns).
        # Einsum: 'bsi, sji -> bsj'
        # b: batch
        # s: slice
        # i: input index (20)
        # j: output index (20)
        # x[b,s,i]
        # M[s,j,i] -> This treats M as (s, output, input).
        # If combined_matrices is (s, out, in), then 'bsi, sji -> bsj' works.
        # If combined_matrices is (s, in, out), we need 'bsi, sij -> bsj'.
        # Previous code: matmul(vec, M.T).
        # M is (20,20). M.T is (20,20).
        # vec (1,20) @ M.T (20,20) -> (1,20).
        # This implies standard row-vector * matrix multiplication.
        # So we want 'bsi, sji -> bsj'. (Summing over i).
        
        new_states = torch.einsum('bsi, sji -> bsj', state_vectors, combined_matrices)

        # Step 7: Extract Output Ports [Batch, Slices, 10]
        # Output ports are [num_ports : 2*num_ports]
        output_complex = new_states[:, :, num_ports : 2*num_ports]

        # Step 8: Convert to Power [Batch, Slices, 10]
        output_power = torch.abs(output_complex) ** 2

        # Step 9: Accumulate over Slices [Batch, 10]
        hidden = output_power.sum(dim=1)

        # Normalization: Use sqrt(n_slices) to maintain variance
        hidden = hidden / math.sqrt(self.n_slices)

        return hidden

    def forward_decoder(self, hidden, decoder_processor):
        """
        Generic Decoder Forward Pass (used for both Pos and Neg paths).
        """
        hidden = torch.abs(hidden)
        output = decoder_processor(hidden)
        return output

    def enable_hooks(self):
        """Enable hooks for all slice processors."""
        for p in self.pos_encoder_slices: p.enable_hook = True
        for p in self.neg_encoder_slices: p.enable_hook = True
        self.pos_decoder_slice.enable_hook = True
        self.neg_decoder_slice.enable_hook = True

    def disable_hooks(self):
        """Disable hooks for all slice processors."""
        for p in self.pos_encoder_slices: p.enable_hook = False
        for p in self.neg_encoder_slices: p.enable_hook = False
        self.pos_decoder_slice.enable_hook = False
        self.neg_decoder_slice.enable_hook = False

    def get_hook_data(self):
        """
        Collect hook data from all slice processors (Pos and Neg).
        """
        hook_data = {
            'pos_encoder': [], 'neg_encoder': [],
            'pos_decoder': None, 'neg_decoder': None,
            'bias': self.bias.detach().cpu()
        }

        # Helper to collect list data
        def collect_list(processors, target_list):
            for idx, p in enumerate(processors):
                if p.hook_data['optical_input'] is not None:
                    target_list.append({
                        'slice_idx': idx,
                        'optical_input': p.hook_data['optical_input'],
                        'optical_output': p.hook_data['optical_output'],
                        'mzi_voltages': p.hook_data['mzi_voltages']
                    })
        
        # Helper to collect single data
        def collect_single(processor):
            if processor.hook_data['optical_input'] is not None:
                return {
                    'optical_input': processor.hook_data['optical_input'],
                    'optical_output': processor.hook_data['optical_output'],
                    'mzi_voltages': processor.hook_data['mzi_voltages']
                }
            return None

        collect_list(self.pos_encoder_slices, hook_data['pos_encoder'])
        collect_list(self.neg_encoder_slices, hook_data['neg_encoder'])
        hook_data['pos_decoder'] = collect_single(self.pos_decoder_slice)
        hook_data['neg_decoder'] = collect_single(self.neg_decoder_slice)

        return hook_data

    def save_hook_data(self, filepath):
        """Save hook data to file."""
        hook_data = self.get_hook_data()
        torch.save(hook_data, filepath)
        print(f"Hook data saved to {filepath}")


# Backward-compat alias. Prefer `OpticalLinear` in new code.
OpticalLoRALinear = OpticalLinear
