"""
Optical Shared Linear Layer

Dual-Path Differential architecture with K-way weight sharing across
N input slices. The N slices reuse K physical MZI processors in a
cyclic manner (slice i uses processor i % K).

Previously named `OpticalLoRALinear`; renamed because the architecture
is not related to LoRA (no frozen base weight + low-rank correction).
A backward-compat alias `OpticalLoRALinear` is kept at module bottom.
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
    (Same as standard version)
    """
    def __init__(self, mzi_row_num=5, mzi_column_num=4, repeat_num=5, start_index=0):
        super(OpticalSliceProcessor, self).__init__()

        # Pre-create MZI layers in __init__ (Clements architecture)
        self.layers = nn.ModuleList()
        # Per-channel local indices (start at 0) so the same crosstalk matrix
        # built from physical-MZI indices 0..49 can be sliced uniformly across
        # every processor instance.
        next_index = 0

        for rep in range(repeat_num):
            self.layers.append(MZIlayer_row(num=mzi_row_num, start_index=next_index))
            next_index += mzi_row_num
            self.layers.append(MZIlayer_column(num=mzi_column_num, start_index=next_index))
            next_index += mzi_column_num

        self.layers.append(MZIlayer_row(num=mzi_row_num, start_index=next_index))
        next_index += mzi_row_num

        self.total_mzis = next_index
        self.num_ports = mzi_row_num * 2
        self.matrix_size = 2 * self.num_ports
        self._combined_matrix = None
        self.hook_data = {
            'optical_input': None, 'optical_output': None, 'mzi_voltages': None
        }
        self.enable_hook = False

        # === THERMAL CROSSTALK ===
        # crosstalk_matrix[dst, src] = (C_dst_src - R_dst_src), units rad / V^2.
        # extra_phase[dst] = sum_src crosstalk_matrix[dst, src] * V_src^2,
        # added to the self-heating delta_phi inside batch_mzi_transfer_matrices.
        # Initialised to zero (no crosstalk); populated by
        # `module.calibration_loader.load_mzi_calibration` when the
        # multi-MZI fit JSON contains a `_crosstalk` block.
        self.register_buffer(
            "crosstalk_matrix",
            torch.zeros(self.total_mzis, self.total_mzis, dtype=torch.float32),
            persistent=False,
        )
        self._has_crosstalk = False

    # === Thermal Crosstalk ===
    def set_crosstalk_matrix(self, K: torch.Tensor) -> None:
        """
        Install a thermal crosstalk coupling matrix for this processor.

        Args:
            K: (total_mzis, total_mzis) tensor where
               `K[dst, src]` is the heater-to-heater coupling
               coefficient (rad / V^2) such that
                   extra_phase[dst] = sum_src K[dst, src] * V_src^2.
        """
        if K.shape != (self.total_mzis, self.total_mzis):
            raise ValueError(
                f"Expected crosstalk matrix shape "
                f"({self.total_mzis}, {self.total_mzis}), got {tuple(K.shape)}."
            )
        with torch.no_grad():
            self.crosstalk_matrix.copy_(
                K.to(
                    dtype=self.crosstalk_matrix.dtype,
                    device=self.crosstalk_matrix.device,
                )
            )
        self._has_crosstalk = bool(torch.any(self.crosstalk_matrix != 0).item())
        self._combined_matrix = None  # invalidate cache built without crosstalk

    def disable_crosstalk(self) -> None:
        with torch.no_grad():
            self.crosstalk_matrix.zero_()
        self._has_crosstalk = False
        self._combined_matrix = None

    def _gather_voltages(self, device: torch.device) -> torch.Tensor:
        """
        Collect every MZI's _voltage parameter in this processor as a
        single (total_mzis,) tensor in the same order as `layer.mzi_indices`.
        """
        v_list = []
        for layer in self.layers:
            if hasattr(layer, "MZI"):
                for mzi in layer.MZI:
                    v_list.append(mzi._voltage)
        return torch.cat(v_list).to(device=device)

    def _get_combined_matrix(self, device):
        # When crosstalk is active the combined matrix depends on every
        # per-MZI voltage (via the K @ V^2 term), so the eval-mode cache
        # would have to refresh on every voltage change. Simpler and
        # safer: rebuild every call when crosstalk is on.
        if self._has_crosstalk:
            return self._build_combined_matrix(device)

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
        N = self.num_ports  # single-side port count

        # === Compute thermal-crosstalk extra phases for the whole channel ===
        # extra_phases_all[dst] = sum_src crosstalk_matrix[dst, src] * V_src^2,
        # then sliced per-layer via layer.mzi_indices below.
        if self._has_crosstalk:
            v_all = self._gather_voltages(device)
            C = self.crosstalk_matrix.to(device=device, dtype=v_all.dtype)
            extra_phases_all = C @ (v_all ** 2)
        else:
            extra_phases_all = None

        # Redheffer identity: zero reflection, perfect transmission
        combined = torch.zeros(
            self.matrix_size, self.matrix_size,
            dtype=torch.complex64, device=device,
        )
        combined[:N, N:] = torch.eye(N, dtype=torch.complex64, device=device)
        combined[N:, :N] = torch.eye(N, dtype=torch.complex64, device=device)

        for layer in self.layers:
            layer_extra = None
            if extra_phases_all is not None and hasattr(layer, "mzi_indices"):
                layer_extra = extra_phases_all[layer.mzi_indices]
            layer_matrix = layer._build_transfer_matrix(
                device, extra_phases=layer_extra
            )
            combined = redheffer_star_product(combined, layer_matrix, N)
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


class OpticalSharedLinear(nn.Module):
    """
    Optical Shared Linear Layer (Dual-Path + K-way weight sharing).
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
                 pos_only: bool = False,   # Debug: only use positive path
                 use_clean_fc: bool = False):  # Drop decoder + replace bias/DC-cancel with diagonal affine
        super(OpticalSharedLinear, self).__init__()

        # Validate parameters
        if r != 10: raise ValueError(f"Rank r must be 10, got {r}")

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.activation_mode = activation_mode  # Unused (kept for backward compat)
        self.detection_mode = detection_mode
        self.num_shared_weights = num_shared_weights # Store K
        self.pos_only = pos_only  # If True, skip negative path (debug/diagnosis)
        self.use_clean_fc = use_clean_fc

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

        # Decoder (Legacy only — clean mode skips it: in power mode the
        # composition of two non-negative linear maps is still rank-10
        # non-negative, so the decoder provides no expressive gain and
        # costs 50 MZIs per path.)
        if not use_clean_fc:
            self.pos_decoder_slice = OpticalSliceProcessor(
                mzi_row_num=mzi_row_num, mzi_column_num=mzi_column_num,
                repeat_num=repeat_num, start_index=next_index
            )
            next_index += self.pos_decoder_slice.total_mzis
        else:
            self.pos_decoder_slice = None

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

        # Decoder (Legacy only)
        if not use_clean_fc:
            self.neg_decoder_slice = OpticalSliceProcessor(
                mzi_row_num=mzi_row_num, mzi_column_num=mzi_column_num,
                repeat_num=repeat_num, start_index=next_index
            )
            next_index += self.neg_decoder_slice.total_mzis
        else:
            self.neg_decoder_slice = None

        self.total_mzis = next_index - start_index

        if out_features != r:
            self.output_proj = nn.Linear(r, out_features, bias=False)
        else:
            self.output_proj = None

        if use_clean_fc:
            # Clean mode: diagonal affine on (pos − neg), applied in
            # r-space (before output_proj) to match the legacy bias
            # placement. Replaces the dead bias + running_logit_mean
            # hack with proper trainable per-port scale + shift. No
            # cross-port mixing, so the optical mesh remains the source
            # of feature combination. 2r electronic params.
            self.bias = None
            self.output_scale = nn.Parameter(torch.ones(r))
            self.output_shift = nn.Parameter(torch.zeros(r))
        else:
            # Legacy: bias + running mean subtraction (bias is effectively
            # killed by the mean subtraction; kept for backward compat).
            self.bias = nn.Parameter(torch.randn(r) * 0.1)
            # === DC offset cancellation (running mean subtraction) ===
            # Under multi-MZI calibration with frozen fabrication parameters
            # the differential dual-path output develops a per-channel DC
            # offset (random batch-mean mismatch between the two independently
            # initialised pos/neg meshes + the trainable fc.bias) that
            # dominates the much smaller input-dependent signal at init time
            # and locks every sample to the same argmax class. We remove this
            # DC offset with a per-channel running mean subtraction in the
            # electronic domain — the digital analogue of an AC-coupling /
            # DC-blocking primitive standard in differential photonic
            # detection front-ends. See HARDWARE_AWARE_FC_DESIGN_NOTES.md.
            self.register_buffer(
                "running_logit_mean",
                torch.zeros(out_features),
                persistent=True,
            )
            self.bn_momentum = 0.1

        # Hook data for FC input (needed for hardware extraction)
        self._fc_input_hook = None
        self._enable_fc_input_hook = False

        self._print_architecture()

    def _print_architecture(self):
        mode = "Clean (no decoder, diagonal affine)" if self.use_clean_fc else "Legacy (decoder + DC-cancel)"
        print(f"\n{'='*70}")
        print(f"OpticalSharedLinear Initialized ({mode})")
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
            if self.pos_decoder_slice is not None:
                self.pos_decoder_slice.clear_cache()
            if self.neg_decoder_slice is not None:
                self.neg_decoder_slice.clear_cache()

        # Record FC input for hardware extraction
        if self._enable_fc_input_hook:
            self._fc_input_hook = x.detach().cpu().clone()

        # === POSITIVE PATH ===
        pos_hidden = self.forward_encoder(x, self.pos_encoder_slices)
        if self.use_clean_fc:
            pos_output = pos_hidden  # Skip decoder
        else:
            pos_output = self.forward_decoder(pos_hidden, self.pos_decoder_slice)

        if self.pos_only:
            if self.use_clean_fc:
                output = self.output_scale * pos_output + self.output_shift
            else:
                output = pos_output + self.bias
        else:
            # === NEGATIVE PATH ===
            neg_hidden = self.forward_encoder(x, self.neg_encoder_slices)
            if self.use_clean_fc:
                neg_output = neg_hidden  # Skip decoder
            else:
                neg_output = self.forward_decoder(neg_hidden, self.neg_decoder_slice)

            # === DIFFERENTIAL OUTPUT ===
            diff = pos_output - neg_output
            if self.use_clean_fc:
                output = self.output_scale * diff + self.output_shift
            else:
                output = diff + self.bias

        if self.output_proj is not None:
            output = self.output_proj(output)

        if not self.use_clean_fc:
            # === DC offset cancellation ===
            # Subtract the running per-channel batch mean. Training: use the
            # current batch mean and update the running EMA. Inference: use
            # the frozen running EMA. This zeros out the input-independent
            # DC offset on the differential output without touching the
            # input-dependent component.
            if self.training:
                batch_mean = output.mean(dim=0)
                output = output - batch_mean.unsqueeze(0)
                with torch.no_grad():
                    self.running_logit_mean.mul_(1.0 - self.bn_momentum).add_(
                        self.bn_momentum * batch_mean.detach()
                    )
            else:
                output = output - self.running_logit_mean.unsqueeze(0)

        return output

    def forward_encoder(self, x, slice_processors):
        batch_size = x.size(0)
        device = x.device
        num_slices = self.n_slices
        num_physical_processors = len(slice_processors)

        # Step 1: Padding  (576 -> 580, so N=58 slices of r=10)
        if self.padded_in > self.in_features:
            x = F.pad(x, (0, self.padded_in - self.in_features))

        # Step 2: Reshape into N slices  (B, 580) -> (B, 58, 10)
        x_sliced = x.view(batch_size, num_slices, self.r)

        # Step 3: Collect K Physical Matrices [K, 20, 20]
        matrices_list = [p._get_combined_matrix(device) for p in slice_processors]
        k_matrices = torch.stack(matrices_list)

        # === Hook: Record Physical Processor Data ===
        first_p = slice_processors[0]
        if first_p.enable_hook:
            for i, p in enumerate(slice_processors):
                v_list = []
                for layer in p.layers:
                    if hasattr(layer, 'MZI'):
                        for mzi in layer.MZI: v_list.append(mzi.get_voltage().item())
                p.hook_data['mzi_voltages'] = torch.tensor(v_list)
                p.hook_data['optical_input'] = x_sliced[:, i, :].detach().cpu().clone()

        # Step 3.5: Weight Sharing — expand K physical to N logical [N, 20, 20]
        #   Slice i uses processor (i % K)
        if num_physical_processors < num_slices:
            indices = torch.arange(num_slices, device=device) % num_physical_processors
            combined_matrices = k_matrices.index_select(0, indices)
        else:
            combined_matrices = k_matrices

        num_ports = first_p.num_ports      # 10
        matrix_size = first_p.matrix_size  # 20

        # Step 4-7: Detection-mode-dependent propagation through N slices
        if self.detection_mode == 'power':
            # ---- Power mode (incoherent): each port processed independently ----
            # Extract T[s,j,i] = |M[s, 10+j, i]|^2  for all N slices
            T_matrices = torch.abs(
                combined_matrices[:, num_ports:2*num_ports, :num_ports]
            ) ** 2  # (N, 10, 10)

            # Input power per slice: x_power[b,s,i] = |x_sliced[b,s,i]| + eps
            x_power = torch.abs(x_sliced) + 1e-8  # (B, N, 10)

            # Power-mode output per slice: output[b,s,j] = Σ_i T[s,j,i] * x_power[b,s,i]
            output_power = torch.einsum('bsi, sji -> bsj', x_power, T_matrices)  # (B, N, 10)
        else:
            # ---- Coherent mode: all inputs simultaneously ----
            x_amplitude = torch.sqrt(torch.abs(x_sliced) + 1e-8)
            x_complex = x_amplitude.to(torch.complex64)

            state_vectors = torch.zeros(batch_size, num_slices, matrix_size,
                                      dtype=torch.complex64, device=device)
            state_vectors[:, :, :num_ports] = x_complex  # (B, N, 20)

            new_states = torch.einsum('bsi, sji -> bsj', state_vectors, combined_matrices)
            output_complex = new_states[:, :, num_ports:2*num_ports]  # (B, N, 10)
            output_power = torch.abs(output_complex) ** 2

        # === Hook: Record Physical Output Power ===
        if first_p.enable_hook:
            for i, p in enumerate(slice_processors):
                p.hook_data['optical_output'] = output_power[:, i, :].detach().cpu().clone()

        # Step 8: Sum over all N slices -> hidden (B, 10)
        hidden = output_power.sum(dim=1)

        # Step 9: Normalization
        hidden = hidden / math.sqrt(self.n_slices)

        return hidden

    def forward_decoder(self, hidden, decoder_processor):
        hidden = torch.abs(hidden)
        # decoder_processor.forward already handles hooks
        output = decoder_processor(hidden)
        return output

    # Hooks (Simplified for brevity, records physical processors)
    def enable_hooks(self):
        self._enable_fc_input_hook = True
        for p in self.pos_encoder_slices: p.enable_hook = True
        for p in self.neg_encoder_slices: p.enable_hook = True
        if self.pos_decoder_slice is not None:
            self.pos_decoder_slice.enable_hook = True
        if self.neg_decoder_slice is not None:
            self.neg_decoder_slice.enable_hook = True

    def disable_hooks(self):
        self._enable_fc_input_hook = False
        for p in self.pos_encoder_slices: p.enable_hook = False
        for p in self.neg_encoder_slices: p.enable_hook = False
        if self.pos_decoder_slice is not None:
            self.pos_decoder_slice.enable_hook = False
        if self.neg_decoder_slice is not None:
            self.neg_decoder_slice.enable_hook = False

    def get_fc_input(self):
        """Get the recorded FC input for hardware extraction."""
        return self._fc_input_hook

    def get_hook_data(self):
        # ... Implementation similar to benchmark but iterates over physical processors ...
        return {} # Placeholder, assume user won't run collecting mode on shared test immediately
    
    def save_hook_data(self, filepath):
        pass


# Backward-compat alias. External scripts (extract_*.py, etc.) that
# still import by the old name continue to work. Prefer
# `OpticalSharedLinear` in new code.
OpticalLoRALinear = OpticalSharedLinear
