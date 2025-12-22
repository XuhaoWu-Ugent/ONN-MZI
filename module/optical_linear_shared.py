"""
Optical LoRA Linear Layer (Shared Weights Version)

Based on the Dual-Path Differential Architecture, but supports Weight Sharing.
Key feature: Allows 'num_shared_weights' (K) < 'n_slices' (N).
The N slices reuse the K MZI arrays in a cyclic manner.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from module.MZI_array.mzi_row_array import MZIlayer_row
from module.MZI_array.mzi_column_array import MZIlayer_column


class OpticalSliceProcessor(nn.Module):
    """
    Single slice processor for vector processing.
    (Same as standard version)
    """
    def __init__(self, mzi_row_num=5, mzi_column_num=4, repeat_num=5, start_index=0):
        super(OpticalSliceProcessor, self).__init__()

        # Pre-create MZI layers in __init__ (Clements architecture)
        self.layers = nn.ModuleList()
        next_index = start_index

        for rep in range(repeat_num):
            self.layers.append(MZIlayer_row(num=mzi_row_num, start_index=next_index))
            next_index += mzi_row_num
            self.layers.append(MZIlayer_column(num=mzi_column_num, start_index=next_index))
            next_index += mzi_column_num

        self.layers.append(MZIlayer_row(num=mzi_row_num, start_index=next_index))
        next_index += mzi_row_num

        self.total_mzis = next_index - start_index
        self.num_ports = mzi_row_num * 2
        self.matrix_size = 2 * self.num_ports
        self._combined_matrix = None
        self.hook_data = {
            'optical_input': None, 'optical_output': None, 'mzi_voltages': None
        }
        self.enable_hook = False

    def _get_combined_matrix(self, device):
        should_update = (self._combined_matrix is None or
                        self._combined_matrix.device != device)
        if should_update:
            self._combined_matrix = self._build_combined_matrix(device)
            if not self.training:
                self._combined_matrix = self._combined_matrix.detach().clone()
        return self._combined_matrix

    def clear_cache(self):
        self._combined_matrix = None

    def _build_combined_matrix(self, device):
        combined = torch.eye(self.matrix_size, dtype=torch.complex64, device=device)
        for layer in reversed(self.layers):
            layer_matrix = layer._build_transfer_matrix(device)
            combined = torch.matmul(layer_matrix, combined)
        return combined

    def forward(self, x):
        # Used for decoder inference
        batch_size = x.size(0)
        device = x.device
        if self.enable_hook: self.hook_data['optical_input'] = x.detach().cpu().clone()
        
        combined_matrix = self._get_combined_matrix(device)
        
        if self.enable_hook:
            voltages = []
            for layer in self.layers:
                if hasattr(layer, 'MZI'):
                    for mzi in layer.MZI: voltages.append(mzi.get_voltage().item())
            self.hook_data['mzi_voltages'] = torch.tensor(voltages)

        x_amplitude = torch.sqrt(x + 1e-8)
        x_complex = x_amplitude.to(torch.complex64)
        
        output_powers = torch.zeros(batch_size, self.num_ports, dtype=torch.float32, device=device)
        for port_idx in range(self.num_ports):
            state_vectors = torch.zeros(batch_size, self.matrix_size, dtype=torch.complex64, device=device)
            state_vectors[:, port_idx] = x_complex[:, port_idx]
            new_states = torch.matmul(state_vectors, combined_matrix.T)
            port_outputs = new_states[:, self.num_ports:self.num_ports + self.num_ports]
            output_powers += torch.abs(port_outputs) ** 2
            
        if self.enable_hook: self.hook_data['optical_output'] = output_powers.detach().cpu().clone()
        return output_powers


class OpticalLoRALinear(nn.Module):
    """
    Optical LoRA Linear Layer (Shared Weights & Dual-Path)
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
                 start_index=0,
                 num_shared_weights=None,  # New Parameter
                 pos_only: bool = False):  # Debug: only use positive path
        super(OpticalLoRALinear, self).__init__()

        # Validate parameters
        if r != 10: raise ValueError(f"Rank r must be 10, got {r}")
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.activation_mode = activation_mode
        self.detection_mode = detection_mode
        self.num_shared_weights = num_shared_weights # Store K
        self.pos_only = pos_only  # If True, skip negative path (debug/diagnosis)

        self.n_slices = math.ceil(in_features / r)
        self.padded_in = self.n_slices * r
        
        # Determine how many PHYSICAL processors to create
        # If shared: K. If not shared: n_slices.
        self.num_processors = self.n_slices if num_shared_weights is None else num_shared_weights
        
        if self.num_processors > self.n_slices:
             self.num_processors = self.n_slices # Cap at N

        # === POSITIVE PATH ===
        # Encoder
        self.pos_encoder_slices = nn.ModuleList()
        next_index = start_index
        for _ in range(self.num_processors):
            processor = OpticalSliceProcessor(
                mzi_row_num=mzi_row_num, mzi_column_num=mzi_column_num,
                repeat_num=repeat_num, start_index=next_index
            )
            self.pos_encoder_slices.append(processor)
            next_index += processor.total_mzis
        self.pos_encoder_mzis = next_index - start_index
        
        # Decoder (Always 1)
        self.pos_decoder_slice = OpticalSliceProcessor(
            mzi_row_num=mzi_row_num, mzi_column_num=mzi_column_num,
            repeat_num=repeat_num, start_index=next_index
        )
        next_index += self.pos_decoder_slice.total_mzis

        # === NEGATIVE PATH ===
        # Encoder
        neg_encoder_start = next_index
        self.neg_encoder_slices = nn.ModuleList()
        for _ in range(self.num_processors):
            processor = OpticalSliceProcessor(
                mzi_row_num=mzi_row_num, mzi_column_num=mzi_column_num,
                repeat_num=repeat_num, start_index=next_index
            )
            self.neg_encoder_slices.append(processor)
            next_index += processor.total_mzis
        self.neg_encoder_mzis = next_index - neg_encoder_start

        # Decoder
        self.neg_decoder_slice = OpticalSliceProcessor(
            mzi_row_num=mzi_row_num, mzi_column_num=mzi_column_num,
            repeat_num=repeat_num, start_index=next_index
        )
        next_index += self.neg_decoder_slice.total_mzis

        self.total_mzis = next_index - start_index

        if out_features != r:
            self.output_proj = nn.Linear(r, out_features, bias=False)
        else:
            self.output_proj = None

        # Trainable Bias (Initialized randomly to break symmetry)
        self.bias = nn.Parameter(torch.randn(r) * 0.1)
        self._print_architecture()

    def _print_architecture(self):
        print(f"\n{'='*70}")
        print(f"OpticalLoRALinear Initialized (Shared Weights + Dual-Path)")
        print(f"{'='*70}")
        print(f"Input Features: {self.in_features}")
        print(f"Slices (N): {self.n_slices}")
        print(f"Shared Weights (K): {self.num_processors} ({'Independent' if self.num_shared_weights is None else 'Shared'})")
        print(f"Total MZI Resources: {self.total_mzis}")
        print(f"{'='*70}\n")

    def forward(self, x):
        if self.training:
            for p in self.pos_encoder_slices: p.clear_cache()
            for p in self.neg_encoder_slices: p.clear_cache()
            self.pos_decoder_slice.clear_cache()
            self.neg_decoder_slice.clear_cache()

        # === POSITIVE PATH ===
        pos_hidden = self.forward_encoder(x, self.pos_encoder_slices)
        pos_output = self.forward_decoder(pos_hidden, self.pos_decoder_slice)

        if self.pos_only:
            output = pos_output + self.bias
        else:
            # === NEGATIVE PATH ===
            neg_hidden = self.forward_encoder(x, self.neg_encoder_slices)
            neg_output = self.forward_decoder(neg_hidden, self.neg_decoder_slice)

            # === DIFFERENTIAL OUTPUT ===
            output = (pos_output - neg_output) + self.bias

        if self.output_proj is not None:
            output = self.output_proj(output)

        return output
    
    def forward(self, x):
        if self.training:
            for p in self.pos_encoder_slices: p.clear_cache()
            for p in self.neg_encoder_slices: p.clear_cache()
            self.pos_decoder_slice.clear_cache()
            self.neg_decoder_slice.clear_cache()

        # === POSITIVE PATH ===
        pos_hidden = self.forward_encoder(x, self.pos_encoder_slices)
        pos_output = self.forward_decoder(pos_hidden, self.pos_decoder_slice)

        if self.pos_only:
            output = pos_output + self.bias
        else:
            # === NEGATIVE PATH ===
            neg_hidden = self.forward_encoder(x, self.neg_encoder_slices)
            neg_output = self.forward_decoder(neg_hidden, self.neg_decoder_slice)

            # === DIFFERENTIAL OUTPUT ===
            output = (pos_output - neg_output) + self.bias

        if self.output_proj is not None:
            output = self.output_proj(output)

        return output

    def forward_encoder(self, x, slice_processors):
        batch_size = x.size(0)
        device = x.device
        num_slices = self.n_slices
        num_physical_processors = len(slice_processors)
        
        # Step 1: Padding
        if self.padded_in > self.in_features:
            x = F.pad(x, (0, self.padded_in - self.in_features))

        # Step 2: Reshape
        x_sliced = x.view(batch_size, num_slices, self.r)

        # Step 3: Collect Physical Matrices [K, 20, 20]
        matrices_list = [p._get_combined_matrix(device) for p in slice_processors]
        k_matrices = torch.stack(matrices_list)

        # === Hook: Record Physical Processor Data ===
        # Since we use vectorized einsum, we manually trigger hooks for physical processors
        first_p = slice_processors[0]
        if first_p.enable_hook:
            for i, p in enumerate(slice_processors):
                # Record Voltages
                v_list = []
                for layer in p.layers:
                    if hasattr(layer, 'MZI'):
                        for mzi in layer.MZI: v_list.append(mzi.get_voltage().item())
                p.hook_data['mzi_voltages'] = torch.tensor(v_list)
                
                # Record Input (for physical processor i, we take the corresponding slice from batch)
                # In shared mode, processor i handles multiple slices. We record the first occurrence.
                p.hook_data['optical_input'] = x_sliced[:, i, :].detach().cpu().clone()

        # Step 3.5: Expand to Logical Matrices [N, 20, 20]
        if num_physical_processors < num_slices:
            # Weight Sharing: Map N slices to K processors
            indices = torch.arange(num_slices, device=device) % num_physical_processors
            combined_matrices = k_matrices.index_select(0, indices)
        else:
            combined_matrices = k_matrices

        # Step 4: Input Vector
        x_amplitude = torch.sqrt(torch.abs(x_sliced) + 1e-8)
        x_complex = x_amplitude.to(torch.complex64)

        # Step 5: State Vectors
        matrix_size = first_p.matrix_size
        num_ports = first_p.num_ports
        state_vectors = torch.zeros(batch_size, num_slices, matrix_size,
                                  dtype=torch.complex64, device=device)
        state_vectors[:, :, :num_ports] = x_complex

        # Step 6: Vectorized Matmul
        new_states = torch.einsum('bsi, sji -> bsj', state_vectors, combined_matrices)

        # Step 7: Output
        output_complex = new_states[:, :, num_ports : 2*num_ports]
        output_power = torch.abs(output_complex) ** 2
        
        # === Hook: Record Physical Output Power ===
        if first_p.enable_hook:
            for i, p in enumerate(slice_processors):
                # Again, record the output of the first slice handled by this processor
                p.hook_data['optical_output'] = output_power[:, i, :].detach().cpu().clone()

        hidden = output_power.sum(dim=1)

        # Normalization
        hidden = hidden / math.sqrt(self.n_slices)

        return hidden

    def forward_decoder(self, hidden, decoder_processor):
        hidden = torch.abs(hidden)
        # decoder_processor.forward already handles hooks
        output = decoder_processor(hidden)
        return output

    # Hooks (Simplified for brevity, records physical processors)
    def enable_hooks(self):
        for p in self.pos_encoder_slices: p.enable_hook = True
        for p in self.neg_encoder_slices: p.enable_hook = True
        self.pos_decoder_slice.enable_hook = True
        self.neg_decoder_slice.enable_hook = True

    def disable_hooks(self):
        for p in self.pos_encoder_slices: p.enable_hook = False
        for p in self.neg_encoder_slices: p.enable_hook = False
        self.pos_decoder_slice.enable_hook = False
        self.neg_decoder_slice.enable_hook = False

    def get_hook_data(self):
        # ... Implementation similar to benchmark but iterates over physical processors ...
        return {} # Placeholder, assume user won't run collecting mode on shared test immediately
    
    def save_hook_data(self, filepath):
        pass
