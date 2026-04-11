from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .mzi import MZI, batch_mzi_transfer_matrices


class MZIlayer_row(nn.Module):
    """
    Horizontal array of Mach–Zehnder interferometers for CNN training.

    This version uses MZIs with calibrated hardware parameters (fixed) and
    trainable internal voltage parameters. When no external voltage is provided,
    each MZI uses its own trainable _voltage parameter.
    """

    def __init__(
        self,
        num: int = 5,
        *,
        start_index: int = 0,
        mzi_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if num <= 0:
            raise ValueError("Row array must contain at least one MZI.")

        mzi_kwargs = mzi_kwargs or {}

        self.num = num
        self.num_ports = num * 2
        self.matrix_size = 2 * self.num_ports


        self.mzi_indices: List[int] = [start_index + i for i in range(num)]
        self.MZI = nn.ModuleList(
            [MZI(index=self.mzi_indices[i], **mzi_kwargs) for i in range(num)]
        )

        self.flag = 0
        self.timing_stats = {"allocation": 0.0, "computation": 0.0, "total": 0.0}

        self._cached_matrix: Optional[torch.Tensor] = None
        self._cached_device: Optional[torch.device] = None
        self._cached_signature: Optional[Tuple[Tuple[float, ...], ...]] = None

        # Pre-compute scatter indices for vectorized matrix assembly.
        self._precompute_scatter_indices()

    @property
    def mzis(self):
        """Backward-compatible alias for self.MZI (read-only property to
        avoid nn.Module double-registration in state_dict)."""
        return self.MZI

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _parameter_signature(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(mzi.hash_key() for mzi in self.MZI)

    def clear_cache(self) -> None:
        self._cached_matrix = None
        self._cached_device = None
        self._cached_signature = None

    def train(self, mode: bool = True) -> "MZIlayer_row":
        super().train(mode)
        if mode:
            self.clear_cache()
        return self

    # ------------------------------------------------------------------
    # Voltage handling
    # ------------------------------------------------------------------
    def _resolve_voltages(
        self,
        voltages: Optional[Union[torch.Tensor, Sequence[float]]],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Resolve voltage inputs into a standardized format.

        Returns:
            None if voltages should use internal parameters (shared across batch)
            1D tensor (num,) if all samples share the same voltage (shared mode)
            2D tensor (batch, num) if each sample has different voltages (batched mode)
        """
        if voltages is None:
            return None

        if isinstance(voltages, torch.Tensor):
            voltage_tensor = voltages.to(device=device, dtype=dtype)
        else:
            voltage_tensor = torch.as_tensor(voltages, dtype=dtype, device=device)

        if voltage_tensor.dim() == 1:
            # Case 1: Compact voltage vector (length == num MZIs in this layer)
            if voltage_tensor.numel() == self.num:
                # Return 1D - indicates shared voltage across all samples
                return voltage_tensor

            # Case 2: Global voltage vector (must cover the max index)
            if voltage_tensor.numel() < max(self.mzi_indices) + 1:
                raise ValueError(
                    f"Voltage vector too short. Expected length {self.num} (compact) or >= {max(self.mzi_indices) + 1} (global), got {voltage_tensor.numel()}."
                )
            selected = voltage_tensor[self.mzi_indices]
            # Return 1D - indicates shared voltage across all samples
            return selected

        if voltage_tensor.dim() == 2:
            # Case 1: Compact voltage matrix (cols == num MZIs)
            if voltage_tensor.size(1) == self.num:
                 if voltage_tensor.size(0) == batch_size:
                    return voltage_tensor
                 if voltage_tensor.size(0) == 1:
                    # Single row - same voltage for all samples, return 1D
                    return voltage_tensor.squeeze(0)
                 raise ValueError(f"Voltage batch size mismatch. Expected {batch_size}, got {voltage_tensor.size(0)}.")

            # Case 2: Global voltage matrix
            if voltage_tensor.size(1) < max(self.mzi_indices) + 1:
                raise ValueError(
                    "Voltage matrix does not have enough columns for slicing."
                )
            if voltage_tensor.size(0) == batch_size:
                return voltage_tensor[:, self.mzi_indices]
            if voltage_tensor.size(0) == 1:
                # Single row - same voltage for all samples, return 1D
                return voltage_tensor[:, self.mzi_indices].squeeze(0)
            raise ValueError("Voltage batch dimension mismatch.")

        raise ValueError("Voltages must be a 1D or 2D tensor.")

    # ------------------------------------------------------------------
    # Vectorized scatter indices (computed once in __init__)
    # ------------------------------------------------------------------
    def _precompute_scatter_indices(self) -> None:
        """Pre-compute index tensors for one-shot matrix assembly."""
        mat_r, mat_c, s_r, s_c, mzi_id = [], [], [], [], []
        for idx in range(self.num):
            upper = 2 * idx
            lower = upper + 1
            N = self.num_ports
            for r, c, sr, sc in [
                (N + upper, upper, 2, 0), (N + upper, lower, 2, 1),
                (N + lower, upper, 3, 0), (N + lower, lower, 3, 1),
                (upper, N + upper, 0, 2), (upper, N + lower, 0, 3),
                (lower, N + upper, 1, 2), (lower, N + lower, 1, 3),
                (upper, upper, 0, 0), (upper, lower, 0, 1),
                (lower, upper, 1, 0), (lower, lower, 1, 1),
                (N + upper, N + upper, 2, 2), (N + upper, N + lower, 2, 3),
                (N + lower, N + upper, 3, 2), (N + lower, N + lower, 3, 3),
            ]:
                mat_r.append(r); mat_c.append(c)
                s_r.append(sr); s_c.append(sc)
                mzi_id.append(idx)
        self.register_buffer("_sc_mr", torch.tensor(mat_r, dtype=torch.long))
        self.register_buffer("_sc_mc", torch.tensor(mat_c, dtype=torch.long))
        self.register_buffer("_sc_sr", torch.tensor(s_r, dtype=torch.long))
        self.register_buffer("_sc_sc", torch.tensor(s_c, dtype=torch.long))
        self.register_buffer("_sc_mi", torch.tensor(mzi_id, dtype=torch.long))

    # ------------------------------------------------------------------
    # Transfer matrix construction (vectorized)
    # ------------------------------------------------------------------
    def _build_transfer_matrix(
        self,
        device: torch.device,
        voltage_overrides: Optional[torch.Tensor] = None,
        extra_phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the 2N×2N scattering matrix for the entire row.

        Uses vectorized batch computation — all MZI S-matrices are computed
        in a single fused pass and scattered into the array matrix via
        pre-computed index tensors (no Python per-MZI loop).

        Args:
            device: Target device for the matrix.
            voltage_overrides: Optional tensor of shape (num,) or (Batch, num).
            extra_phases: Optional (num,) extra phase from thermal crosstalk.

        Returns:
            (matrix_size, matrix_size) or (Batch, matrix_size, matrix_size).
        """
        # Cache check (non-batched, no voltage override, eval mode)
        use_cache = (
            not self.training
            and voltage_overrides is None
            and extra_phases is None
            and self._cached_matrix is not None
            and self._cached_device == device
            and self._cached_signature == self._parameter_signature()
        )
        if use_cache:
            return self._cached_matrix

        # --- Collect parameters from all MZIs into (K,) tensors ---
        raw_a = torch.cat([m._raw_a for m in self.MZI]).to(device)
        raw_b = torch.cat([m._raw_b for m in self.MZI]).to(device)
        raw_dr = torch.cat([m._raw_delta_r for m in self.MZI]).to(device)
        raw_p0 = torch.cat([m._raw_phi0 for m in self.MZI]).to(device)
        raw_ppi = torch.cat([m._raw_p_pi for m in self.MZI]).to(device)
        raw_alpha = torch.cat([m._raw_alpha for m in self.MZI]).to(device)

        is_batched = False
        if voltage_overrides is not None:
            v = voltage_overrides.to(device=device, dtype=raw_a.dtype)
            if v.dim() == 2:
                is_batched = True
                if v.size(1) != self.num:
                    raise ValueError(
                        f"Expected {self.num} voltages per sample, got {v.size(1)}."
                    )
            elif v.dim() == 1:
                if v.numel() != self.num:
                    raise ValueError(
                        f"Expected {self.num} voltages, got {v.numel()}."
                    )
            else:
                raise ValueError("voltage_overrides must be 1D or 2D tensor.")
        else:
            v = torch.cat([m._voltage for m in self.MZI]).to(device)

        # --- Vectorized S-matrix computation ---
        mzi0 = self.MZI[0]
        nom_r = torch.tensor([m.nominal_resistance for m in self.MZI],
                             dtype=raw_a.dtype, device=device)
        s_all = batch_mzi_transfer_matrices(
            raw_a, raw_b, raw_dr, raw_p0, raw_ppi, raw_alpha, v,
            nominal_resistance=nom_r,
            epsilon=mzi0.epsilon,
            orientation=mzi0.orientation,
            extra_phases=extra_phases,
        )  # (K, 4, 4) or (B, K, 4, 4)

        # --- Scatter into array matrix ---
        mr = self._sc_mr.to(device)
        mc = self._sc_mc.to(device)
        vals_idx = (self._sc_mi.to(device), self._sc_sr.to(device), self._sc_sc.to(device))

        if is_batched:
            B = v.size(0)
            matrix = torch.zeros(
                B, self.matrix_size, self.matrix_size,
                dtype=torch.complex64, device=device,
            )
            # s_all: (B, K, 4, 4) → values: (B, 16*K)
            values = s_all[:, vals_idx[0], vals_idx[1], vals_idx[2]]
            matrix[:, mr, mc] = values
        else:
            matrix = torch.zeros(
                self.matrix_size, self.matrix_size,
                dtype=torch.complex64, device=device,
            )
            values = s_all[vals_idx[0], vals_idx[1], vals_idx[2]]
            matrix[mr, mc] = values

        # Cache if applicable
        if voltage_overrides is None and not self.training:
            self._cached_matrix = matrix.detach().clone()
            self._cached_device = device
            self._cached_signature = self._parameter_signature()

        return matrix

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------
    def forward(
        self,
        input_tensor: torch.Tensor,
        voltages: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    ) -> torch.Tensor:
        """
        Propagate complex amplitudes through the row array.

        Args:
            input_tensor: Tensor with shape (batch, ports) or
                          (batch, patches, ports).
            voltages: Optional per-MZI voltage overrides. If None, each MZI
                     uses its internal trainable _voltage parameter.
        """
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(1)
            squeeze_output = True
        elif input_tensor.dim() == 3:
            squeeze_output = False
        else:
            raise ValueError("input_tensor must have 2 or 3 dimensions.")

        batch_size, num_patches, num_ports = input_tensor.shape
        if num_ports != self.num_ports:
            raise ValueError(
                f"Expected {self.num_ports} ports but received {num_ports}."
            )

        if not input_tensor.dtype.is_complex:
            input_tensor = input_tensor.to(torch.complex64)

        device = input_tensor.device
        dtype = input_tensor.real.dtype

        voltage_resolved = self._resolve_voltages(
            voltages, batch_size, dtype=dtype, device=device
        )

        input_flat = input_tensor.reshape(-1, num_ports)
        batch_total = input_flat.shape[0]

        if voltage_resolved is None or voltage_resolved.dim() == 1:
            # Shared voltage mode: all samples use the same transfer matrix
            # voltage_resolved is None (use internal params) or 1D (shared external voltage)
            transfer = self._build_transfer_matrix(device, voltage_overrides=voltage_resolved)

            state_vectors = torch.zeros(
                batch_total, self.matrix_size, dtype=torch.complex64, device=device
            )
            state_vectors[:, :num_ports] = input_flat

            new_states = torch.matmul(state_vectors, transfer.T)
            output_flat = new_states[:, self.num_ports : self.num_ports + num_ports]
        else:
            # Batched voltage mode: each sample has different voltage
            # voltage_resolved is 2D (batch, num)
            voltage_flat = voltage_resolved.unsqueeze(1).repeat(1, num_patches, 1)
            voltage_flat = voltage_flat.reshape(batch_total, self.num)

            # Build batched transfer matrices (Batch_Total, matrix_size, matrix_size)
            transfer_batched = self._build_transfer_matrix(
                device, voltage_overrides=voltage_flat
            )

            # Prepare state vectors (Batch_Total, matrix_size)
            state_vectors = torch.zeros(
                batch_total, self.matrix_size, dtype=torch.complex64, device=device
            )
            state_vectors[:, :num_ports] = input_flat

            # Vectorized batch matrix multiplication
            # state_vectors: (Batch_Total, matrix_size)
            # transfer_batched.transpose(-2, -1): (Batch_Total, matrix_size, matrix_size)
            # Result: (Batch_Total, matrix_size)
            new_states = torch.bmm(
                state_vectors.unsqueeze(1), transfer_batched.transpose(-2, -1)
            ).squeeze(1)

            output_flat = new_states[:, self.num_ports : self.num_ports + num_ports]

        output = output_flat.view(batch_size, num_patches, num_ports)

        if squeeze_output:
            return output.squeeze(1)
        return output

    def get_all_time(self) -> dict:
        """Compatibility helper mirroring the legacy API."""
        total_stats = {"allocation": 0.0, "computation": 0.0, "total": 0.0}
        for mzi in self.MZI:
            if hasattr(mzi, "get_all_time"):
                mzi_stats = mzi.get_all_time()
                for key, value in mzi_stats.items():
                    total_stats[key] = total_stats.get(key, 0.0) + value
        return total_stats


# Backward compatible alias.
MZILayerRow = MZIlayer_row
