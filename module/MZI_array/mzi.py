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
    Four-port Mach–Zehnder interferometer for CNN training with calibrated hardware.

    This version is designed for CNN training where:
    - Hardware fabrication parameters (a, b, delta_r, phi0) are FIXED after loading
      from calibration results (mzi_parameters.json)
    - Voltage is a TRAINABLE parameter that controls the phase shift to implement
      the desired CNN weights

    The parameterisation follows the definitions in AGENTS.md.
    """

    def __init__(
        self,
        index: Optional[int] = None,
        *,
        nominal_resistance: float = 1200.0,
        p_pi: float = 12e-3,
        epsilon: float = 1e-8,
        trainable_fabrication: bool = False,
        trainable_voltage: bool = True,
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

        # Hardware fabrication parameters (should be loaded from calibration and frozen)
        # These represent the physical imperfections of the manufactured MZI
        self._raw_a = nn.Parameter(torch.zeros(1), requires_grad=trainable_fabrication)
        self._raw_b = nn.Parameter(torch.zeros(1), requires_grad=trainable_fabrication)
        self._raw_delta_r = nn.Parameter(torch.zeros(1), requires_grad=trainable_fabrication)
        self._raw_phi0 = nn.Parameter(torch.zeros(1), requires_grad=trainable_fabrication)

        # Voltage parameter (trainable - this is what we control to implement CNN weights)
        # Initialize with small random values to break symmetry
        self._voltage = nn.Parameter(
            torch.randn(1) * 0.1, requires_grad=trainable_voltage
        )

        self.matrix = torch.zeros((4, 4), dtype=torch.complex64, device=self.device)

    def extra_repr(self) -> str:
        parts = []
        if self.index is not None:
            parts.append(f"index={self.index}")
        parts.append(f"trainable_voltage={self._voltage.requires_grad}")
        parts.append(f"trainable_fabrication={self._raw_a.requires_grad}")
        return ", ".join(parts)

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
        Overwrite the internal fabrication parameters with calibrated physical values.

        The provided values must respect the valid ranges described in AGENTS.md.
        After loading, you should call freeze_fabrication_parameters() to prevent
        these from being trained.
        """
        dtype = self._raw_a.dtype
        device = self._raw_a.device
        eps = torch.finfo(dtype).eps

        with torch.no_grad():
            if a is not None:
                a_tensor = torch.as_tensor(a, dtype=dtype, device=device)
                a_tensor = torch.clamp(a_tensor, 0.45, 0.55)
                # Inverse of sigmoid mapping: raw = logit((a - 0.45) / 0.1)
                normalized = (a_tensor - 0.45) / 0.1
                self._raw_a.copy_(torch.logit(normalized))

            if b is not None:
                b_tensor = torch.as_tensor(b, dtype=dtype, device=device)
                b_tensor = torch.clamp(b_tensor, 0.45, 0.55)
                normalized = (b_tensor - 0.45) / 0.1
                self._raw_b.copy_(torch.logit(normalized))

            if delta_r is not None:
                delta_tensor = torch.as_tensor(delta_r, dtype=dtype, device=device)
                ratio = torch.clamp(delta_tensor / 50.0, -1.0 + eps, 1.0 - eps)
                self._raw_delta_r.copy_(torch.atanh(ratio))

            if phi0 is not None:
                phi_tensor = torch.as_tensor(phi0, dtype=dtype, device=device)
                ratio = torch.clamp(phi_tensor / math.pi, -1.0 + eps, 1.0 - eps)
                self._raw_phi0.copy_(torch.atanh(ratio))

    def freeze_fabrication_parameters(self) -> None:
        """
        Freeze the hardware fabrication parameters so they won't be updated during training.
        Call this after loading calibrated parameters from JSON.
        """
        self._raw_a.requires_grad = False
        self._raw_b.requires_grad = False
        self._raw_delta_r.requires_grad = False
        self._raw_phi0.requires_grad = False

    def unfreeze_fabrication_parameters(self) -> None:
        """
        Unfreeze the hardware fabrication parameters (only for calibration training).
        """
        self._raw_a.requires_grad = True
        self._raw_b.requires_grad = True
        self._raw_delta_r.requires_grad = True
        self._raw_phi0.requires_grad = True

    def get_voltage(self) -> torch.Tensor:
        """
        Get the current voltage parameter value.
        """
        return self._voltage

    def set_voltage(self, voltage: Union[float, torch.Tensor]) -> None:
        """
        Set the voltage parameter to a specific value.
        """
        with torch.no_grad():
            if not torch.is_tensor(voltage):
                voltage = torch.tensor(voltage, dtype=self._voltage.dtype)
            self._voltage.copy_(voltage.to(self._voltage.device))

    def transfer_matrix(
        self, voltage: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:

        """
        Compute the 4×4 scattering matrix defined in AGENTS.md.

        Args:
            voltage: Optional external voltage override. 
                     If None, uses the internal trainable _voltage parameter (scalar).
                     If provided, can be scalar or tensor of shape (Batch,).
        
        Returns:
            Matrix of shape (4, 4) if voltage is scalar, or (Batch, 4, 4) if batched.
        """

        params = self.physical_parameters()

        a = params["a"]
        b = params["b"]
        delta_r = params["delta_r"]
        phi0 = params["phi0"]

        dtype = a.dtype
        device = a.device

        # Handle voltage input
        is_batched = False
        if voltage is None:
            voltage_tensor = self._voltage.to(dtype=dtype, device=device).view(1)
        else:
            if not torch.is_tensor(voltage):
                voltage_tensor = torch.tensor(voltage, dtype=dtype, device=device)
            else:
                voltage_tensor = voltage.to(device=device, dtype=dtype)
            
            if voltage_tensor.numel() > 1:
                is_batched = True
                # Ensure it's 1D for calculation (Batch,)
                voltage_tensor = voltage_tensor.reshape(-1)
            else:
                voltage_tensor = voltage_tensor.view(1)

        resistance = torch.clamp(
            torch.tensor(self.nominal_resistance, dtype=dtype, device=device) + delta_r,
            min=1.0,
        )
        # delta_phi shape: (Batch,) or (1,)
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

        exp_phi1 = _exp_neg_j(phi1) # (Batch,)
        exp_phi2 = _exp_neg_j(phi2) # Scalar

        alpha_real = self.alpha_amplitude.to(device=device, dtype=dtype)
        alpha = torch.complex(alpha_real, torch.zeros_like(alpha_real))
        j_const = torch.complex(
            torch.zeros_like(alpha_real), torch.ones_like(alpha_real)
        )

        # Calculate S-parameters. These will broadcast to (Batch,) if exp_phi1 is batched.
        S31 = alpha * (sqrt_1a_1b * exp_phi1 - sqrt_ab * exp_phi2)
        S32 = -j_const * alpha * (sqrt_a_1b * exp_phi1 + sqrt_b_1a * exp_phi2)
        S41 = -j_const * alpha * (sqrt_b_1a * exp_phi1 + sqrt_a_1b * exp_phi2)
        S42 = alpha * (-sqrt_ab * exp_phi1 + sqrt_1a_1b * exp_phi2)
        
        zero = torch.zeros_like(S31)
        
        # Stack into matrix
        # If batched, Sxx are (Batch,). We want (Batch, 4, 4).
        # If scalar, Sxx are scalar. We want (4, 4).
        
        # Row 0
        r0 = torch.stack([zero, zero, S31, S41], dim=-1)
        r1 = torch.stack([zero, zero, S32, S42], dim=-1)
        r2 = torch.stack([S31, S32, zero, zero], dim=-1)
        r3 = torch.stack([S41, S42, zero, zero], dim=-1)
        
        # (Batch, 4, 4) or (4, 4)
        matrix = torch.stack([r0, r1, r2, r3], dim=-2)

        return matrix

    def forward(
        self,
        field: torch.Tensor,
        voltage: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Propagate complex optical fields through the MZI.

        Args:
            field: Input optical field
            voltage: Optional external voltage override. If None, uses the internal
                    trainable _voltage parameter.
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
                self._voltage,
            )
        )

    def get_all_time(self) -> Dict[str, float]:
        """Compatibility stub for legacy profiling hooks."""
        return {"allocation": 0.0, "computation": 0.0, "total": 0.0}
