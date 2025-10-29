import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


def _exp_neg_j(phi: torch.Tensor) -> torch.Tensor:
    """Compute e^{-j phi} using real-valued trigonometric functions."""
    real = torch.cos(phi)
    imag = -torch.sin(phi)
    return torch.complex(real, imag)


class MZI(nn.Module):
    """
    Four-port Mach–Zehnder interferometer that preserves complex fields.

    The parameterisation follows the definitions in AGENTS.md. All fabrication
    variations (coupling ratios, intrinsic phase, heater resistance error) remain
    trainable, while the heater voltage is supplied at runtime.
    """

    def __init__(
        self,
        index: Optional[int] = None,
        *,
        nominal_resistance: float = 1200.0,
        p_pi: float = 12e-3,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()

        self.index = index
        self.nominal_resistance = float(nominal_resistance)
        self.p_pi = float(p_pi)
        self.epsilon = float(epsilon)

        self.register_buffer(
            "alpha_amplitude", torch.tensor(math.sqrt(0.94), dtype=torch.float32)
        )
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        # Trainable fabrication parameters.
        self._raw_a = nn.Parameter(torch.zeros(1))
        self._raw_b = nn.Parameter(torch.zeros(1))
        self._raw_delta_r = nn.Parameter(torch.zeros(1))
        self._raw_phi0 = nn.Parameter(torch.zeros(1))
        self.matrix = torch.zeros((4, 4), dtype=torch.complex64, device=self.device)
        
    def extra_repr(self) -> str:
        return f"index={self.index}" if self.index is not None else ""

    # ------------------------------------------------------------------
    # Parameter mappings
    # ------------------------------------------------------------------
    def _bounded_split_ratio(self, raw: torch.Tensor) -> torch.Tensor:
        guard = self.epsilon
        # Map sigmoid output [0,1] to [0.45, 0.55]
        sigmoid_out = torch.sigmoid(raw)
        # Scale to range 0.1 and shift to start at 0.45
        return 0.45 + 0.1 * torch.clamp(sigmoid_out, guard, 1.0 - guard)

    def _bounded_delta_r(self, raw: torch.Tensor) -> torch.Tensor:
        return 50.0 * torch.tanh(raw)

    def _bounded_phi0(self, raw: torch.Tensor) -> torch.Tensor:
        return math.pi * torch.tanh(raw)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def physical_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Return the current physical parameters of the interferometer.
        """
        dtype = self._raw_a.dtype
        device = self._raw_a.device

        a = self._bounded_split_ratio(self._raw_a)
        b = self._bounded_split_ratio(self._raw_b)
        delta_r = self._bounded_delta_r(self._raw_delta_r)
        phi0 = self._bounded_phi0(self._raw_phi0)

        return {
            "a": a,
            "b": b,
            "delta_r": delta_r,
            "phi0": phi0,
        }

    def load_physical_parameters(
        self,
        *,
        a: Optional[Union[float, torch.Tensor]] = None,
        b: Optional[Union[float, torch.Tensor]] = None,
        delta_r: Optional[Union[float, torch.Tensor]] = None,
        phi0: Optional[Union[float, torch.Tensor]] = None,
    ) -> None:
        """
        Overwrite the internal trainable parameters with physical values.

        The provided values must respect the valid ranges described in AGENTS.md.
        """
        dtype = self._raw_a.dtype
        device = self._raw_a.device
        eps = torch.finfo(dtype).eps

        with torch.no_grad():
            if a is not None:
                a_tensor = torch.as_tensor(a, dtype=dtype, device=device)
                a_tensor = torch.clamp(a_tensor, 0.45, 0.55)
                self._raw_a.copy_(torch.logit(a_tensor))

            if b is not None:
                b_tensor = torch.as_tensor(b, dtype=dtype, device=device)
                b_tensor = torch.clamp(b_tensor, 0.45, 0.55)
                self._raw_b.copy_(torch.logit(b_tensor))

            if delta_r is not None:
                delta_tensor = torch.as_tensor(delta_r, dtype=dtype, device=device)
                ratio = torch.clamp(delta_tensor / 50.0, -1.0 + eps, 1.0 - eps)
                self._raw_delta_r.copy_(torch.atanh(ratio))

            if phi0 is not None:
                phi_tensor = torch.as_tensor(phi0, dtype=dtype, device=device)
                ratio = torch.clamp(phi_tensor / math.pi, -1.0 + eps, 1.0 - eps)
                self._raw_phi0.copy_(torch.atanh(ratio))

    def transfer_matrix(
        self, voltage: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:

        """
        Compute the 4×4 scattering matrix defined in AGENTS.md.
        """

        params = self.physical_parameters()

        a = params["a"]
        b = params["b"]
        delta_r = params["delta_r"]
        phi0 = params["phi0"]

        dtype = a.dtype
        device = a.device

        if voltage is None:
            voltage_tensor = torch.zeros(1, dtype=dtype, device=device)
        else:
            if not torch.is_tensor(voltage):
                voltage_tensor = torch.tensor(voltage, dtype=dtype, device=device)
            else:
                voltage_tensor = voltage.to(device=device, dtype=dtype)
        voltage_tensor = voltage_tensor.view(1)

        resistance = torch.clamp(
            torch.tensor(self.nominal_resistance, dtype=dtype, device=device) + delta_r,
            min=1.0,
        )
        delta_phi = math.pi * (voltage_tensor**2) / (resistance * self.p_pi)

        phi1 = 0.5 * phi0 + delta_phi
        phi2 = -0.5 * phi0

        eps = self.epsilon
        sqrt = torch.sqrt
        clamp = torch.clamp

        sqrt_ab = sqrt(clamp(a * b, min=eps))
        sqrt_a_1b = sqrt(clamp(a * (1.0 - b), min=eps))
        sqrt_b_1a = sqrt(clamp(b * (1.0 - a), min=eps))
        sqrt_1a_1b = sqrt(clamp((1.0 - a) * (1.0 - b), min=eps))

        exp_phi1 = _exp_neg_j(phi1)
        exp_phi2 = _exp_neg_j(phi2)

        alpha_real = self.alpha_amplitude.to(device=device, dtype=dtype)
        alpha = torch.complex(alpha_real, torch.zeros_like(alpha_real))
        j_const = torch.complex(
            torch.zeros_like(alpha_real), torch.ones_like(alpha_real)
        )

        S31 = alpha * (sqrt_1a_1b * exp_phi1 - sqrt_ab * exp_phi2)
        S32 = -j_const * alpha * (sqrt_a_1b * exp_phi1 + sqrt_b_1a * exp_phi2)
        S41 = -j_const * alpha * (sqrt_b_1a * exp_phi1 + sqrt_a_1b * exp_phi2)
        S42 = alpha * (-sqrt_ab * exp_phi1 + sqrt_1a_1b * exp_phi2)
        zero = torch.zeros_like(S31)
        matrix = torch.stack([
            torch.stack([zero, zero, S31, S41]),
            torch.stack([zero, zero, S32, S42]),
            torch.stack([S31, S32, zero, zero]),
            torch.stack([S41, S42, zero, zero])
        ]).squeeze(-1)
        
        return matrix

    def forward(
        self,
        field: torch.Tensor,
        voltage: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Propagate complex optical fields through the MZI.
        """
        if not field.dtype.is_complex:
            field = field.to(torch.complex64)

        matrix = self.transfer_matrix(voltage=voltage).to(field.device)
        return torch.matmul(field, matrix.T)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def hash_key(self) -> Tuple[float, ...]:
        """Compact signature used by array-level caching."""
        return tuple(
            param.detach().item()
            for param in (
                self._raw_a,
                self._raw_b,
                self._raw_delta_r,
                self._raw_phi0,
            )
        )

    def get_all_time(self) -> Dict[str, float]:
        """Compatibility stub for legacy profiling hooks."""
        return {"allocation": 0.0, "computation": 0.0, "total": 0.0}
