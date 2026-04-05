import math
import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


def _exp_neg_j(phi: torch.Tensor) -> torch.Tensor:
    """Compute e^{-j phi} using real-valued trigonometric functions."""
    real = torch.cos(phi)
    imag = -torch.sin(phi)
    return torch.complex(real, imag)


def batch_mzi_transfer_matrices(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    raw_delta_r: torch.Tensor,
    raw_phi0: torch.Tensor,
    voltage: torch.Tensor,
    nominal_resistance: float,
    p_pi: float,
    alpha_amplitude: torch.Tensor,
    epsilon: float = 1e-8,
    orientation: str = "horizontal",
) -> torch.Tensor:
    """
    Compute 4x4 scattering matrices for K MZIs in one vectorized pass.

    Args:
        raw_a, raw_b, raw_delta_r, raw_phi0: (K,) raw parameter tensors.
        voltage: (K,) shared or (B, K) per-sample voltage tensor.
        nominal_resistance, p_pi: scalar physical constants.
        alpha_amplitude: scalar tensor (insertion-loss amplitude).
        epsilon: numerical guard.
        orientation: 'horizontal' or 'vertical'.

    Returns:
        (K, 4, 4) complex tensor if voltage is 1-D,
        (B, K, 4, 4) complex tensor if voltage is 2-D.
    """
    guard = epsilon
    # --- bounded parameter mappings (all shape (K,)) ---
    a = 0.45 + 0.1 * torch.clamp(torch.sigmoid(raw_a), guard, 1.0 - guard)
    b = 0.45 + 0.1 * torch.clamp(torch.sigmoid(raw_b), guard, 1.0 - guard)
    delta_r = 50.0 * torch.tanh(raw_delta_r)
    phi0 = math.pi * torch.tanh(raw_phi0)

    dtype = a.dtype
    device = a.device

    resistance = torch.clamp(
        torch.tensor(nominal_resistance, dtype=dtype, device=device) + delta_r,
        min=1.0,
    )  # (K,)

    # delta_phi: (K,) or (B, K) depending on voltage shape
    delta_phi = math.pi * (voltage ** 2) / (resistance * p_pi)

    phi1 = 0.5 * phi0 + delta_phi     # (K,) or (B, K)
    phi2 = -0.5 * phi0                 # (K,)

    sqrt_ab = torch.sqrt(torch.clamp(a * b, min=epsilon))
    sqrt_a_1b = torch.sqrt(torch.clamp(a * (1.0 - b), min=epsilon))
    sqrt_b_1a = torch.sqrt(torch.clamp(b * (1.0 - a), min=epsilon))
    sqrt_1a_1b = torch.sqrt(torch.clamp((1.0 - a) * (1.0 - b), min=epsilon))

    exp_phi1 = _exp_neg_j(phi1)   # (K,) or (B, K)
    exp_phi2 = _exp_neg_j(phi2)   # (K,)

    alpha_c = torch.complex(
        alpha_amplitude.to(dtype=dtype, device=device),
        torch.zeros(1, dtype=dtype, device=device),
    )
    j_c = torch.complex(
        torch.zeros(1, dtype=dtype, device=device),
        torch.ones(1, dtype=dtype, device=device),
    )

    # S-parameters — all (K,) or (B, K) complex
    S31 = alpha_c * (sqrt_1a_1b * exp_phi1 - sqrt_ab * exp_phi2)
    S32 = -j_c * alpha_c * (sqrt_a_1b * exp_phi1 + sqrt_b_1a * exp_phi2)
    S41 = -j_c * alpha_c * (sqrt_b_1a * exp_phi1 + sqrt_a_1b * exp_phi2)
    S42 = alpha_c * (-sqrt_ab * exp_phi1 + sqrt_1a_1b * exp_phi2)

    zero = torch.zeros_like(S31)

    if orientation == "horizontal":
        r0 = torch.stack([zero, zero, S31, S41], dim=-1)
        r1 = torch.stack([zero, zero, S32, S42], dim=-1)
        r2 = torch.stack([S31, S32, zero, zero], dim=-1)
        r3 = torch.stack([S41, S42, zero, zero], dim=-1)
    else:  # vertical
        r0 = torch.stack([zero, S31, zero, S41], dim=-1)
        r1 = torch.stack([S31, zero, S32, zero], dim=-1)
        r2 = torch.stack([zero, S32, zero, S42], dim=-1)
        r3 = torch.stack([S41, zero, S42, zero], dim=-1)

    # (K, 4, 4) or (B, K, 4, 4)
    return torch.stack([r0, r1, r2, r3], dim=-2)


def redheffer_star_product(
    S_A: torch.Tensor, S_B: torch.Tensor, N: int
) -> torch.Tensor:
    """
    Cascade two 2N x 2N scattering matrices via the Redheffer star product.

    Given:
        S_A -- 2N x 2N S-matrix of upstream element (light hits A first)
        S_B -- 2N x 2N S-matrix of downstream element
        N   -- single-side port count (block size)

    Each S-matrix is partitioned as::

        [[S11, S12],   S11: N x N left-to-left   (back-reflection)
         [S21, S22]]   S21: N x N left-to-right   (forward transmission)
                       S12: N x N right-to-left   (reverse transmission)
                       S22: N x N right-to-right  (back-reflection)

    Returns:
        2N x 2N cascaded S-matrix that accounts for all multiple reflections
        between A's right interface and B's left interface (steady-state).

    Fully differentiable (uses ``torch.linalg.solve``).
    """
    dtype = S_A.dtype
    device = S_A.device

    A11, A12 = S_A[:N, :N], S_A[:N, N:]
    A21, A22 = S_A[N:, :N], S_A[N:, N:]
    B11, B12 = S_B[:N, :N], S_B[:N, N:]
    B21, B22 = S_B[N:, :N], S_B[N:, N:]

    I = torch.eye(N, dtype=dtype, device=device)

    lhs_F = I - A22 @ B11
    lhs_G = I - B11 @ A22

    F_A21 = torch.linalg.solve(lhs_F, A21)
    F_A22_B12 = torch.linalg.solve(lhs_F, A22 @ B12)
    G_B11_A21 = torch.linalg.solve(lhs_G, B11 @ A21)
    G_B12 = torch.linalg.solve(lhs_G, B12)

    S11 = A11 + A12 @ G_B11_A21
    S12 = A12 @ G_B12
    S21 = B21 @ F_A21
    S22 = B22 + B21 @ F_A22_B12

    return torch.cat(
        [torch.cat([S11, S12], dim=-1),
         torch.cat([S21, S22], dim=-1)],
        dim=-2,
    )


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
        lossless: bool = False,
        orientation: str = "horizontal",
    ) -> None:
        super().__init__()

        if orientation not in ("horizontal", "vertical"):
            raise ValueError(
                f"orientation must be 'horizontal' or 'vertical', got {orientation!r}"
            )

        self.index = index
        self.orientation = orientation
        self.nominal_resistance = float(nominal_resistance)
        self.p_pi = float(p_pi)
        self.epsilon = float(epsilon)
        # 训练阶段可通过 lossless 参数或环境变量关闭插入损耗
        self.lossless = bool(lossless) or bool(int(os.getenv("MZI_LOSSLESS", "0")))

        alpha_value = 1.0 if self.lossless else math.sqrt(0.94)
        self.register_buffer(
            "alpha_amplitude", torch.tensor(alpha_value, dtype=torch.float32)
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
        parts.append(f"orientation={self.orientation}")
        parts.append(f"trainable_voltage={self._voltage.requires_grad}")
        parts.append(f"trainable_fabrication={self._raw_a.requires_grad}")
        parts.append(f"lossless={self.lossless}")
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
        #
        # Port convention: {0,1} = left ports, {2,3} = right ports.
        #
        # Horizontal MZI (standard): light flows left→right, no back-reflection.
        #   S₁₁ = S₂₂ = 0
        #
        # Vertical MZI (rotated 90°): port permutation swaps local ports 1↔2,
        #   creating non-zero S₁₁ and S₂₂ (back-reflection between adjacent
        #   waveguides on the same side of the mesh).
        #   Physical routing:
        #     LU→LL (S₁₁[1,0]=S31, back-reflection) and LU→RL (S₂₁[1,0]=S41)
        #     LL→LU (S₁₁[0,1]=S31, back-reflection) and LL→RU (S₂₁[0,1]=S32)

        if self.orientation == "horizontal":
            r0 = torch.stack([zero, zero, S31, S41], dim=-1)
            r1 = torch.stack([zero, zero, S32, S42], dim=-1)
            r2 = torch.stack([S31, S32, zero, zero], dim=-1)
            r3 = torch.stack([S41, S42, zero, zero], dim=-1)
        else:  # vertical
            r0 = torch.stack([zero, S31, zero, S41], dim=-1)
            r1 = torch.stack([S31, zero, S32, zero], dim=-1)
            r2 = torch.stack([zero, S32, zero, S42], dim=-1)
            r3 = torch.stack([S41, zero, S42, zero], dim=-1)

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
