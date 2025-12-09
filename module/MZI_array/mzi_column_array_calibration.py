from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .mzi import MZI


class MZIlayer_column(nn.Module):
    """
    Vertical array of Mach–Zehnder interferometers with fixed bypass ports.
    """

    def __init__(
        self,
        num: int = 4,
        *,
        start_index: int = 0,
        mzi_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if num <= 0:
            raise ValueError("Column array must contain at least one MZI.")

        mzi_kwargs = mzi_kwargs or {}

        self.num = num
        self.num_ports = 2 * (num + 1)
        self.matrix_size = 2 * self.num_ports

        self.mzi_indices: List[int] = [start_index + i for i in range(num)]
        self.MZI = nn.ModuleList(
            [MZI(index=self.mzi_indices[i], **mzi_kwargs) for i in range(num)]
        )
        self.mzis = self.MZI
        self.flag = 0
        self.timing_stats = {"allocation": 0.0, "computation": 0.0, "total": 0.0}

        self._cached_matrix: Optional[torch.Tensor] = None
        self._cached_device: Optional[torch.device] = None
        self._cached_signature: Optional[Tuple[Tuple[float, ...], ...]] = None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _parameter_signature(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(mzi.hash_key() for mzi in self.MZI)

    def clear_cache(self) -> None:
        self._cached_matrix = None
        self._cached_device = None
        self._cached_signature = None

    def train(self, mode: bool = True) -> "MZIlayer_column":
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
        if voltages is None:
            return None

        if isinstance(voltages, torch.Tensor):
            voltage_tensor = voltages.to(device=device, dtype=dtype)
        else:
            voltage_tensor = torch.as_tensor(voltages, dtype=dtype, device=device)

        if voltage_tensor.dim() == 1:
            if voltage_tensor.numel() < max(self.mzi_indices) + 1:
                raise ValueError(
                    "Voltage vector is too short for the configured MZI indices."
                )
            selected = voltage_tensor[self.mzi_indices]
            return selected.unsqueeze(0).expand(batch_size, -1)

        if voltage_tensor.dim() == 2:
            if voltage_tensor.size(1) < max(self.mzi_indices) + 1:
                raise ValueError(
                    "Voltage matrix does not have enough columns for slicing."
                )
            if voltage_tensor.size(0) == batch_size:
                return voltage_tensor[:, self.mzi_indices]
            if voltage_tensor.size(0) == 1:
                return voltage_tensor[:, self.mzi_indices].expand(batch_size, -1)
            raise ValueError("Voltage batch dimension mismatch.")

        raise ValueError("Voltages must be a 1D or 2D tensor.")

    # ------------------------------------------------------------------
    # Transfer matrix construction
    # ------------------------------------------------------------------
    def _build_transfer_matrix(
        self,
        device: torch.device,
        voltage_overrides: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_cache = ( voltage_overrides is None
            and self._cached_matrix is not None
            and self._cached_signature == self._parameter_signature()
        )
        if use_cache:
            return self._cached_matrix

        matrix=torch.zeros(
            (self.matrix_size, self.matrix_size),
            dtype=torch.complex64,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        )

        param_dtype = self.MZI[0]._raw_a.dtype if self.MZI else torch.float32
        if voltage_overrides is None:
            voltage_vector = torch.zeros(
                self.num, dtype=param_dtype, device=device
            )
        else:
            voltage_vector = voltage_overrides.to(
                device=device, dtype=param_dtype
            ).view(-1)
            if voltage_vector.numel() != self.num:
                raise ValueError(
                    f"Expected {self.num} voltages, got {voltage_vector.numel()}."
                )

        # Top and bottom bypass ports propagate directly.
        matrix[self.num_ports + 0, 0] = 1.0
        matrix[0, self.num_ports + 0] = 1.0
        matrix[self.num_ports + (self.num_ports - 1), self.num_ports - 1] = 1.0
        matrix[self.num_ports - 1, self.num_ports + (self.num_ports - 1)] = 1.0

        for idx, mzi in enumerate(self.MZI):
            s_matrix = mzi.transfer_matrix(voltage=voltage_vector[idx]).to(
                device=device
            )

            upper = 2 * idx + 1
            lower = upper + 1

            matrix[self.num_ports + upper, upper] = s_matrix[2, 0]
            matrix[self.num_ports + upper, lower] = s_matrix[2, 1]
            matrix[self.num_ports + lower, upper] = s_matrix[3, 0]
            matrix[self.num_ports + lower, lower] = s_matrix[3, 1]

            matrix[upper, self.num_ports + upper] = s_matrix[0, 2]
            matrix[upper, self.num_ports + lower] = s_matrix[0, 3]
            matrix[lower, self.num_ports + upper] = s_matrix[1, 2]
            matrix[lower, self.num_ports + lower] = s_matrix[1, 3]

            matrix[upper, upper] = s_matrix[0, 0]
            matrix[upper, lower] = s_matrix[0, 1]
            matrix[lower, upper] = s_matrix[1, 0]
            matrix[lower, lower] = s_matrix[1, 1]

            matrix[self.num_ports + upper, self.num_ports + upper] = s_matrix[2, 2]
            matrix[self.num_ports + upper, self.num_ports + lower] = s_matrix[2, 3]
            matrix[self.num_ports + lower, self.num_ports + upper] = s_matrix[3, 2]
            matrix[self.num_ports + lower, self.num_ports + lower] = s_matrix[3, 3]

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

        voltage_matrix = self._resolve_voltages(
            voltages, batch_size, dtype=dtype, device=device
        )

        input_flat = input_tensor.reshape(-1, num_ports)
        batch_total = input_flat.shape[0]

        if voltage_matrix is None:
            transfer = self._build_transfer_matrix(device)

            state_vectors = torch.zeros(
                batch_total, self.matrix_size, dtype=torch.complex64, device=device
            )
            state_vectors[:, :num_ports] = input_flat

            new_states = torch.matmul(state_vectors, transfer.T)
            output_flat = new_states[:, self.num_ports : self.num_ports + num_ports]
        else:
            voltage_flat = voltage_matrix.unsqueeze(1).repeat(1, num_patches, 1)
            voltage_flat = voltage_flat.reshape(batch_total, self.num)

            output_chunks = []
            for sample_idx in range(batch_total):
                transfer = self._build_transfer_matrix(
                    device, voltage_overrides=voltage_flat[sample_idx]
                )
                state_vector = torch.zeros(
                    self.matrix_size, dtype=torch.complex64, device=device
                )
                state_vector[:num_ports] = input_flat[sample_idx]
                new_state = torch.matmul(state_vector, transfer.T)
                output_chunks.append(new_state[self.num_ports : self.num_ports + num_ports])

            output_flat = torch.stack(output_chunks, dim=0)

        output = output_flat.view(batch_size, num_patches, num_ports)

        if squeeze_output:
            return output.squeeze(1)
        return output

    def get_all_time(self) -> dict:
        total_stats = {"allocation": 0.0, "computation": 0.0, "total": 0.0}
        for mzi in self.MZI:
            if hasattr(mzi, "get_all_time"):
                mzi_stats = mzi.get_all_time()
                for key, value in mzi_stats.items():
                    total_stats[key] = total_stats.get(key, 0.0) + value
        return total_stats


MZILayerColumn = MZIlayer_column
