from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # Create alternating row and column MZI layers with global indexing
        # Structure: Row -> Column -> Row -> Column -> ... -> Row (2*repeat_num + 1 layers)
        self.layers = nn.ModuleList()
        next_index = 0
        mzi_kwargs = {}
        for rep in range(repeat_num):
            row_layer = MZIlayer_row(
                num=mzi_row_num,
                start_index=next_index,
                mzi_kwargs=mzi_kwargs,
            )
            self.layers.append(row_layer)
            next_index += mzi_row_num

            column_layer = MZIlayer_column(
                num=mzi_column_num,
                start_index=next_index,
                mzi_kwargs=mzi_kwargs,
            )
            self.layers.append(column_layer)
            next_index += mzi_column_num

        final_row = MZIlayer_row(
            num=mzi_row_num,
            start_index=next_index,
            mzi_kwargs=mzi_kwargs,
        )
        self.layers.append(final_row)
        next_index += mzi_row_num
        self.total_mzis = next_index

        # Trainable diagonal weight matrix
        self.diagonal_matrix = nn.Parameter(torch.randn(self.num_ports))

        # Trainable Bias (Applied in electrical domain, used as thresholding)
        self.bias = nn.Parameter(torch.zeros(1))

        # Other parameters
        self.flag = 0
        self.timing_stats = {'allocation': 0, 'computation': 0, 'total': 0}

        # Store matrix size for dynamic computation
        self.matrix_size = 2 * self.num_ports  # 20x20

        # Pre-allocate combined transfer matrix (cached only in eval mode)
        self._combined_matrix = None
        
        # === HOOK MECHANISM ===
        self.enable_hook = False
        self.hook_data = {
            'optical_input': None, 
            'optical_output': None, 
            'mzi_voltages': None
        }

    # === Hook Methods ===
    def enable_hooks(self):
        self.enable_hook = True
        
    def disable_hooks(self):
        self.enable_hook = False
        self.hook_data = {'optical_input': None, 'optical_output': None, 'mzi_voltages': None}

    def _prepare_voltages(
        self,
        voltages: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if voltages is None:
            return None

        if not torch.is_tensor(voltages):
            voltage_tensor = torch.tensor(voltages, dtype=torch.float32, device=device)
        else:
            voltage_tensor = voltages.to(device=device, dtype=torch.float32)

        if voltage_tensor.dim() == 1:
            if voltage_tensor.numel() != self.total_mzis:
                raise ValueError(
                    f"Expected {self.total_mzis} voltages, got {voltage_tensor.numel()}."
                )
            return voltage_tensor.unsqueeze(0).expand(batch_size, -1)

        if voltage_tensor.dim() == 2:
            if voltage_tensor.size(1) != self.total_mzis:
                raise ValueError(
                    f"Expected voltage vectors of length {self.total_mzis}, "
                    f"got {voltage_tensor.size(1)}."
                )
            if voltage_tensor.size(0) == batch_size:
                return voltage_tensor
            if voltage_tensor.size(0) == 1:
                return voltage_tensor.expand(batch_size, -1)
            raise ValueError(
                f"Voltage batch dimension mismatch: expected {batch_size}, "
                f"got {voltage_tensor.size(0)}."
            )

        raise ValueError("voltages must be None, 1D or 2D tensor.")

    def _get_combined_matrix(self, device, voltages: Optional[torch.Tensor] = None):
        if voltages is not None:
            return self._build_combined_transfer_matrix(device, voltages=voltages)

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

    def _build_combined_transfer_matrix(
        self, device, voltages: Optional[torch.Tensor] = None
    ):
        if voltages is not None:
            voltages = voltages.to(device=device)
            if voltages.numel() != self.total_mzis:
                raise ValueError(
                    f"Expected {self.total_mzis} voltages, got {voltages.numel()}."
                )

        # Start with identity matrix
        combined_matrix = torch.eye(self.matrix_size, dtype=torch.complex64, device=device)

        for layer in reversed(self.layers):
            layer_voltage = None
            if voltages is not None and hasattr(layer, "mzi_indices"):
                layer_voltage = voltages[layer.mzi_indices]
            layer_matrix = layer._build_transfer_matrix(
                device, voltage_overrides=layer_voltage
            )
            combined_matrix = torch.matmul(layer_matrix, combined_matrix)

        return combined_matrix

    def _coherent_forward(self, patches, combined_matrix, return_intermediate=False):
        batch_size, num_patches, num_ports = patches.shape

        # === Hook: Record Inputs ===
        if self.enable_hook:
            self.hook_data['optical_input'] = patches.detach().cpu().clone()

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

        # === Hook: Record Outputs ===
        if self.enable_hook:
            # Save as power for consistency with measurement
            self.hook_data['optical_output'] = (torch.abs(output_patches)**2).detach().cpu().clone()

        # Save processed first patch if needed
        processed_patch = None
        if return_intermediate:
            processed_patch = output_patches[:, 0, :].clone().detach()

        return output_patches, processed_patch

    def _power_forward(self, patches, combined_matrix, return_intermediate=False):
        batch_size, num_patches, num_ports = patches.shape
        device = patches.device

        # === Hook: Record Inputs ===
        if self.enable_hook:
            self.hook_data['optical_input'] = patches.detach().cpu().clone()

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
            port_power = torch.abs(port_outputs) ** 2
            output_powers += port_power

        # === Hook: Record Outputs ===
        if self.enable_hook:
            self.hook_data['optical_output'] = output_powers.detach().cpu().clone()

        # Save processed first patch if needed
        processed_patch = None
        if return_intermediate:
            processed_patch = output_powers[:, 0, :].clone().detach()

        return output_powers, processed_patch

    def get_all_time(self):
        total_stats = {'allocation': 0, 'computation': 0, 'total': 0}
        for key in total_stats:
            if hasattr(self, 'timing_stats'):
                total_stats[key] += self.timing_stats[key]
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
        super().train(mode)
        if mode:
            self._combined_matrix = None
        return self

    def forward(
        self,
        x,
        voltages: Optional[torch.Tensor] = None,
        return_intermediate=False,
    ):
        # === Hook: Record Voltages ===
        if self.enable_hook:
            # Collect voltages from all MZI layers
            v_list = []
            for layer in self.layers:
                if hasattr(layer, 'MZI'):
                    for mzi in layer.MZI:
                        v_list.append(mzi.get_voltage().item())
            self.hook_data['mzi_voltages'] = torch.tensor(v_list)

        # Get input dimensions
        batch_size, height, width = x.shape

        # Calculate output dimensions
        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1

        # Extract image patches
        patches = F.unfold(x.unsqueeze(1),
                          kernel_size=self.kernel_size,
                          stride=1) 
        patches = patches.permute(0, 2, 1) 
        patches = F.pad(patches, (0, 1)) 

        # Save first patch input if needed
        input_patch = None
        if return_intermediate:
            input_patch = patches[:, 0, :].clone().detach()

        # Convert to complex type for MZI processing
        patches = patches.to(torch.complex64)

        # Resolve per-batch voltages
        voltage_batch = self._prepare_voltages(voltages, batch_size, patches.device)

        # Process through MZI array based on detection mode
        if self.detection_mode == 'coherent':
            if voltage_batch is None:
                combined_matrix = self._get_combined_matrix(patches.device)
                output_patches, processed_patch = self._coherent_forward(
                    patches, combined_matrix, return_intermediate
                )
            else:
                outputs = []
                processed_patch = None
                for sample_idx in range(batch_size):
                    combined_matrix = self._build_combined_transfer_matrix(
                        patches.device, voltages=voltage_batch[sample_idx]
                    )
                    sample_output, sample_processed = self._coherent_forward(
                        patches[sample_idx : sample_idx + 1],
                        combined_matrix,
                        return_intermediate,
                    )
                    outputs.append(sample_output)
                    if return_intermediate and processed_patch is None:
                        processed_patch = sample_processed
                output_patches = torch.cat(outputs, dim=0)
            # Apply trainable diagonal weights to complex amplitudes
            weighted_output = output_patches * self.diagonal_matrix.view(1, 1, -1)

        else:  # 'power'
            if voltage_batch is None:
                combined_matrix = self._get_combined_matrix(patches.device)
                output_powers, processed_patch = self._power_forward(
                    patches, combined_matrix, return_intermediate
                )
            else:
                outputs = []
                processed_patch = None
                for sample_idx in range(batch_size):
                    combined_matrix = self._build_combined_transfer_matrix(
                        patches.device, voltages=voltage_batch[sample_idx]
                    )
                    sample_output, sample_processed = self._power_forward(
                        patches[sample_idx : sample_idx + 1],
                        combined_matrix,
                        return_intermediate,
                    )
                    outputs.append(sample_output)
                    if return_intermediate and processed_patch is None:
                        processed_patch = sample_processed
                output_powers = torch.cat(outputs, dim=0)
            # Apply trainable diagonal weights to powers
            # For power domain: weight^2 is applied to power values
            weighted_output = output_powers * (self.diagonal_matrix.abs() ** 2).view(1, 1, -1)

        # Sum across all output ports and reshape to spatial dimensions
        if self.detection_mode == 'coherent':
            # For coherent mode, output_patches contains complex amplitudes.
            # We take the intensity (abs squared) before summing.
            output = (torch.abs(weighted_output)**2).sum(dim=2).view(batch_size, output_height, output_width)
        else:
            # For power mode, weighted_output already contains power values.
            output = weighted_output.sum(dim=2).view(batch_size, output_height, output_width)

        # Add Bias and apply ReLU (Thresholding logic)
        output = F.relu(output + self.bias)

        if return_intermediate:
            return output, input_patch, processed_patch
        else:
            return output