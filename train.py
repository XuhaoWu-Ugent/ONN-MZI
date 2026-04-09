import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from module.MZI_array.mzi import redheffer_star_product
from module.MZI_array.mzi_column_array import MZIlayer_column
from module.MZI_array.mzi_row_array import MZIlayer_row


class FabricationDataset(Dataset):
    """
    Dataset wrapping the on-chip measurement logs.

    Each entry describes a single experiment where one input port is excited and
    exactly one MZI receives a non-zero heater voltage. The measurement records
    the optical power observed on a specific output port.
    """

    def __init__(self, data_dir: str) -> None:
        self.samples: List[dict] = []
        data_path = Path(data_dir)
        files = sorted(data_path.glob("MZI_array_output_in*.txt"))
        if not files:
            raise FileNotFoundError(
                f"No measurement files found in {data_path.resolve()}."
            )

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) != 6:
                        raise ValueError(
                            f"Unexpected line in {file_path.name}: {line.strip()}"
                        )
                    input_port, input_power, mzi_index, voltage, output_port, output_power = parts
                    self.samples.append(
                        {
                            "input_port": int(input_port),
                            "input_power": float(input_power),
                            "mzi_index": int(mzi_index),
                            "voltage": float(voltage),
                            "output_port": int(output_port),
                            "output_power": float(output_power),
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["input_port"], dtype=torch.long),
            torch.tensor(sample["input_power"], dtype=torch.float32),
            torch.tensor(sample["mzi_index"], dtype=torch.long),
            torch.tensor(sample["voltage"], dtype=torch.float32),
            torch.tensor(sample["output_port"], dtype=torch.long),
            torch.tensor(sample["output_power"], dtype=torch.float32),
        )


class MZIArray(nn.Module):
    """
    Full six-row by five-column MZI array operating on complex amplitudes.
    """

    def __init__(
        self,
        *,
        row_mzis: int = 5,
        column_mzis: int = 4,
        row_layers: int = 6,
        enable_crosstalk: bool = False,
        enable_temp_drift: bool = False,
        trainable_fabrication: bool = False,
    ) -> None:
        super().__init__()

        self.port_count = row_mzis * 2
        self.layers = nn.ModuleList()
        next_index = 0
        mzi_kwargs = {"trainable_fabrication": trainable_fabrication}
        for row_id in range(row_layers):
            row_layer = MZIlayer_row(
                num=row_mzis,
                start_index=next_index,
                mzi_kwargs=mzi_kwargs,
            )
            self.layers.append(row_layer)
            next_index += row_mzis

            # Insert a column layer between consecutive row layers.
            if row_id < row_layers - 1:
                column_layer = MZIlayer_column(
                    num=column_mzis,
                    start_index=next_index,
                    mzi_kwargs=mzi_kwargs,
                )
                self.layers.append(column_layer)
                next_index += column_mzis

        self.total_mzis = next_index

        # Thermal crosstalk: sparse matrix of phase coupling between MZI pairs
        self.enable_crosstalk = enable_crosstalk
        if enable_crosstalk:
            self._init_crosstalk()

        # --- Frozen parameters (kept for analysis/paper, not trained) ---
        # Global substrate temperature rise (α_R)
        self._raw_alpha_R = nn.Parameter(torch.tensor(-5.0), requires_grad=False)
        # Per-port detector model (gain + noise floor)
        self._raw_det_gain = nn.Parameter(torch.zeros(self.port_count), requires_grad=False)
        self._raw_noise_floor = nn.Parameter(torch.full((self.port_count,), -5.0), requires_grad=False)

        # Global temperature drift: R_eff = R_meas × (1 + γ × json_idx)
        # γ is trainable, initialized near zero. Typical range: ~1e-4 to 1e-2.
        self.enable_temp_drift = enable_temp_drift
        if enable_temp_drift:
            self._raw_gamma = nn.Parameter(torch.zeros(1))

    def temp_drift_factor(self, json_idx: int) -> torch.Tensor:
        """Compute voltage scaling factor for temperature drift.

        R_eff = R × (1 + γ × json_idx), so V_eff = V / sqrt(1 + γ × idx).
        γ is bounded to [-0.05, 0.05] via tanh to prevent divergence.
        Returns a scalar tensor (keeps gradient flow).
        """
        if not self.enable_temp_drift:
            return torch.ones(1)
        gamma = 0.05 * torch.tanh(self._raw_gamma)  # bounded ±5%
        factor = 1.0 / torch.sqrt(1.0 + gamma * json_idx)
        return factor

    @staticmethod
    def _mzi_ports(layer, local_idx: int) -> Tuple[int, int]:
        """Return the (upper, lower) waveguide port pair for a MZI within a layer."""
        if isinstance(layer, MZIlayer_row):
            upper = 2 * local_idx
            lower = upper + 1
        else:  # MZIlayer_column
            upper = 2 * local_idx + 1
            lower = upper + 1
        return (upper, lower)

    def _init_crosstalk(self) -> None:
        """Build physical-neighbor adjacency list and trainable crosstalk coefficients.

        Three types of physical adjacency (max 8 neighbors per MZI):
        1. Same-layer adjacent: consecutive MZIs within the same row/column layer
        2. Adjacent-layer port-sharing: MZIs in layers L and L±1 that share ≥1 port
        3. Skip-layer same-position: MZIs in layers L and L±2 with identical port pairs
        """
        pair_set = set()
        layer_list = list(self.layers)
        n_layers = len(layer_list)

        # Precompute port pairs for each (layer_idx, local_idx) → global_mzi_index
        mzi_port_map = {}  # global_mzi_idx → (upper_port, lower_port)
        for li, layer in enumerate(layer_list):
            for k, gidx in enumerate(layer.mzi_indices):
                mzi_port_map[gidx] = self._mzi_ports(layer, k)

        for li, layer in enumerate(layer_list):
            mzis = layer.mzi_indices
            # Type 1: Same-layer adjacent (consecutive indices)
            for k in range(len(mzis) - 1):
                pair = tuple(sorted((mzis[k], mzis[k + 1])))
                pair_set.add(pair)

            # Type 2: Adjacent-layer port-sharing (L and L+1)
            if li + 1 < n_layers:
                next_layer = layer_list[li + 1]
                for m1 in mzis:
                    p1 = set(mzi_port_map[m1])
                    for m2 in next_layer.mzi_indices:
                        p2 = set(mzi_port_map[m2])
                        if p1 & p2:  # share at least one port
                            pair_set.add(tuple(sorted((m1, m2))))

            # Type 3: Skip-layer same-position (L and L+2, same port pair)
            if li + 2 < n_layers:
                skip_layer = layer_list[li + 2]
                for m1 in mzis:
                    p1 = mzi_port_map[m1]
                    for m2 in skip_layer.mzi_indices:
                        p2 = mzi_port_map[m2]
                        if p1 == p2:  # identical port pair
                            pair_set.add(tuple(sorted((m1, m2))))

        # Convert to directed pairs (both directions for asymmetric coupling)
        pairs = []
        for a, b in sorted(pair_set):
            pairs.append((a, b))
            pairs.append((b, a))

        self.crosstalk_pairs = pairs  # list of (i, j) tuples
        # Indices as tensors for vectorized computation
        self.register_buffer(
            "_xt_src", torch.tensor([p[1] for p in pairs], dtype=torch.long)
        )
        self.register_buffer(
            "_xt_dst", torch.tensor([p[0] for p in pairs], dtype=torch.long)
        )
        # Trainable coefficients, initialized near zero
        self._raw_crosstalk = nn.Parameter(torch.zeros(len(pairs)))
        # Reference-arm crosstalk (frozen at zero — kept for analysis/paper, not trained)
        self._raw_refarm_xt = nn.Parameter(torch.full((len(pairs),), -20.0), requires_grad=False)

    def compute_crosstalk_phases(self, voltages: torch.Tensor) -> torch.Tensor:
        """Compute net extra phase per MZI from thermal crosstalk.

        Two effects on the same neighbor pairs:
        1. Heated-arm crosstalk (additive): neighbor heater → this MZI's heated arm
           extra_phase[i] += C_ij * V_j²
        2. Reference-arm crosstalk (subtractive): neighbor heater → this MZI's ref arm
           extra_phase[i] -= R_ij * V_j²

        Net: extra_phase[i] = Σ_j (C_ij - R_ij) * V_j²

        Args:
            voltages: (50,) voltage vector.

        Returns:
            extra_phases: (50,) net additional phase shift per MZI.
        """
        v_sq = voltages ** 2  # (50,)
        # Heated-arm crosstalk (can be positive or negative after training)
        coeffs_heat = self._raw_crosstalk * 0.01
        # Reference-arm crosstalk (bounded non-negative via softplus)
        coeffs_ref = torch.nn.functional.softplus(self._raw_refarm_xt) * 0.01
        # Net coefficient per pair
        net_coeffs = coeffs_heat - coeffs_ref
        # Gather source V² values and multiply by net coefficients
        contributions = net_coeffs * v_sq[self._xt_src]  # (N_pairs,)
        # Scatter-add to destination MZIs
        extra = torch.zeros(self.total_mzis, dtype=voltages.dtype, device=voltages.device)
        extra.scatter_add_(0, self._xt_dst, contributions)
        return extra

    def forward(
        self, input_powers: torch.Tensor, voltages: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagate powers through the full array using Redheffer star product.

        Builds a combined 2N×2N scattering matrix by cascading all layer
        S-matrices via the Redheffer star product, then applies it to the
        input amplitude vector.  This correctly accounts for back-reflection
        in vertical (column) MZI layers.

        Args:
            input_powers: Tensor with shape (batch, port_count).
            voltages: Tensor with shape (batch, total_mzis).
        """
        if input_powers.dim() != 2:
            raise ValueError("input_powers must have shape (batch, port_count).")
        if voltages.dim() != 2:
            raise ValueError("voltages must have shape (batch, total_mzis).")

        N = self.port_count
        device = input_powers.device
        batch_size = input_powers.shape[0]

        if input_powers.dtype.is_complex:
            amplitudes = input_powers
        else:
            amplitudes = torch.sqrt(torch.clamp(input_powers, min=0.0)).to(
                torch.complex64
            )

        eye_N = torch.eye(N, dtype=torch.complex64, device=device)

        # Each sample may have a different voltage vector, so we build one
        # combined scattering matrix per sample.
        output_list = []
        for b in range(batch_size):
            # Identity element for Redheffer: [[0, I], [I, 0]]
            combined = torch.zeros(
                2 * N, 2 * N, dtype=torch.complex64, device=device
            )
            combined[:N, N:] = eye_N
            combined[N:, :N] = eye_N

            v_sample = voltages[b]
            for layer in self.layers:
                layer_volts = v_sample[layer.mzi_indices]
                layer_matrix = layer._build_transfer_matrix(
                    device, voltage_overrides=layer_volts
                )
                combined = redheffer_star_product(combined, layer_matrix, N)

            # Apply combined S-matrix: input from the left side
            state = torch.zeros(2 * N, dtype=torch.complex64, device=device)
            state[:N] = amplitudes[b]
            result = combined @ state
            output_list.append(result[N : 2 * N])

        return torch.stack(output_list)

    def iter_mzis(self) -> Iterable:
        for layer in self.layers:
            if hasattr(layer, "mzis"):
                for mzi in layer.mzis:
                    yield mzi

    def alpha_R(self) -> torch.Tensor:
        """Substrate thermal coefficient, bounded to [0, 0.05] W⁻¹."""
        return 0.05 * torch.sigmoid(self._raw_alpha_R)

    def substrate_voltage_scale(self, voltage_vec: torch.Tensor) -> torch.Tensor:
        """Compute effective voltage scaling from substrate temperature rise.

        Args:
            voltage_vec: (50,) voltage vector for all MZIs.

        Returns:
            scale: scalar tensor, V_eff = V × scale.

        Physics:
            P_total = Σ_i V_i² / R_nominal  (approximate total heater power)
            R_eff = R_nominal × (1 + α_R × P_total)
            Since Δφ = π V² / (R × P_pi), increasing R reduces phase.
            Equivalent to scaling V_eff = V / √(1 + α_R × P_total).
        """
        aR = self.alpha_R()
        P_total = (voltage_vec ** 2).sum() / 1200.0  # approx total power (W)
        return 1.0 / torch.sqrt(1.0 + aR * P_total)

    def det_gain(self) -> torch.Tensor:
        """Per-output-port multiplicative correction, bounded to [-0.3, +0.3]."""
        return 0.3 * torch.tanh(self._raw_det_gain)

    def noise_floor(self) -> torch.Tensor:
        """Per-output-port additive noise floor in mW, bounded to [0, 0.001] mW (0–1 μW)."""
        return 0.001 * torch.sigmoid(self._raw_noise_floor)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate fabrication errors of the MZI array using measured data."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing MZI_array_output_in*.txt files.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/mzi_parameters.json",
        help="Destination JSON file for the calibrated parameters.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the calibration on (e.g., 'cpu' or 'cuda').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser


def ensure_output_directory(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def train_model(args: argparse.Namespace) -> Tuple[MZIArray, FabricationDataset]:
    torch.manual_seed(args.seed)

    dataset = FabricationDataset(args.data_dir)

    device = torch.device(args.device)
    model = MZIArray(trainable_fabrication=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    mse = nn.MSELoss()

    # --- Pre-group samples by (mzi_index, voltage) ---
    # Each group shares one Redheffer cascade (only 1 MZI has non-zero V).
    from collections import defaultdict
    groups = defaultdict(list)
    for i, sample in enumerate(dataset.samples):
        key = (sample["mzi_index"], sample["voltage"])
        groups[key].append(sample)

    N = model.port_count
    group_keys = list(groups.keys())
    print(f"  {len(dataset)} samples in {len(group_keys)} voltage groups")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        # Shuffle group order each epoch
        perm = torch.randperm(len(group_keys))

        for gi in perm:
            mzi_idx, volt = group_keys[gi]
            samples = groups[(mzi_idx, volt)]

            # Build voltage vector: only one MZI has non-zero voltage
            voltage_vec = torch.zeros(model.total_mzis, device=device)
            voltage_vec[mzi_idx] = volt

            # Build combined S-matrix via Redheffer (once per group)
            combined = torch.zeros(
                2 * N, 2 * N, dtype=torch.complex64, device=device
            )
            eye_N = torch.eye(N, dtype=torch.complex64, device=device)
            combined[:N, N:] = eye_N
            combined[N:, :N] = eye_N

            for layer in model.layers:
                layer_volts = voltage_vec[layer.mzi_indices]
                layer_matrix = layer._build_transfer_matrix(
                    device, voltage_overrides=layer_volts
                )
                combined = redheffer_star_product(combined, layer_matrix, N)

            # Vectorized: process all samples in this group at once
            n = len(samples)
            inp_ports = torch.tensor([s["input_port"] for s in samples], device=device)
            inp_pows = torch.tensor([s["input_power"] for s in samples], dtype=torch.float32, device=device)
            out_ports = torch.tensor([s["output_port"] for s in samples], device=device)
            out_pows = torch.tensor([s["output_power"] for s in samples], dtype=torch.float32, device=device)

            # Build input amplitudes: (n, N)
            amplitudes = torch.zeros(n, N, dtype=torch.complex64, device=device)
            sqrt_pows = torch.sqrt(torch.clamp(inp_pows, min=0.0))
            amplitudes.scatter_(1, inp_ports.unsqueeze(1).long(),
                                sqrt_pows.unsqueeze(1).to(torch.complex64))

            # Apply combined S-matrix: state = [amplitudes, 0] @ S^T
            state = torch.zeros(n, 2 * N, dtype=torch.complex64, device=device)
            state[:, :N] = amplitudes
            result = torch.matmul(state, combined.T)
            output_fields = result[:, N:2*N]

            # Extract predicted power at measured output ports
            port_fields = output_fields.gather(1, out_ports.unsqueeze(1).long()).squeeze(1)
            predicted_power = port_fields.abs() ** 2

            loss = mse(predicted_power, out_pows)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * n
            total_samples += n

        average_loss = total_loss / total_samples
        print(f"Epoch {epoch:03d}: loss={average_loss:.6e}")

    return model, dataset


def export_parameters(model: MZIArray, output_file: str) -> None:
    ensure_output_directory(output_file)
    result = {}
    for mzi in model.iter_mzis():
        if mzi.index is None:
            continue
        params = mzi.physical_parameters()
        entry = {
            "a": float(params["a"].item()),
            "b": float(params["b"].item()),
            "delta_r": float(params["delta_r"].item()),
            "p_pi": float(params["p_pi"].item()),
            "phi0": float(params["phi0"].item()),
            "alpha": float(params["alpha"].item()),
        }
        if hasattr(mzi, "measured_resistance") and mzi.measured_resistance is not None:
            entry["resistance"] = mzi.measured_resistance
        result[mzi.index] = entry

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Calibrated parameters written to {output_file}")

def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    model, _ = train_model(args)
    model.eval()
    export_parameters(model, args.output_file)

if __name__ == "__main__":
    main()
