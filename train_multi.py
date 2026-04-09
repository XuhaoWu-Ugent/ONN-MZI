"""
Train MZI fabrication parameters (a, b, delta_r, phi0, alpha) from full-mesh
measurement data collected during Bayesian voltage optimization rounds.

Unlike train.py which uses single-MZI excitation data, this script uses
measurements where all 50 MZIs are active simultaneously under various voltage
configurations, providing richer gradient signal for parameter estimation.
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module.MZI_array.mzi import redheffer_star_product
from train import MZIArray, ensure_output_directory

MIN_PM_DBM = -45.0


# ---------------------------------------------------------------------------
# Resistance file parser
# ---------------------------------------------------------------------------

def parse_resistance_file(filepath: str) -> Dict[int, float]:
    """Parse MZI_resistance.md → {mzi_index: resistance_ohm}.

    File format: 'MZI{idx} {label} {resistance_kohm}' per line.
    Returns resistance in Ohms (kΩ × 1000).
    """
    result = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0].startswith("MZI"):
                idx = int(parts[0][3:])
                r_kohm = float(parts[2])
                result[idx] = r_kohm * 1000.0  # kΩ → Ω

    return result


def load_measured_resistances(model: MZIArray, resistance_map: Dict[int, float]) -> int:
    """Set measured_resistance on each MZI in the model. Returns count loaded."""
    loaded = 0
    for mzi in model.iter_mzis():
        if mzi.index in resistance_map:
            mzi.measured_resistance = resistance_map[mzi.index]
            loaded += 1
    return loaded


def load_measured_physics(
    model: MZIArray,
    physics_path: str,
    freeze_p_pi: bool = False,
    freeze_phi0: bool = False,
) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """Load measured P_pi, phi0, resistance from mzi_measured_physics.json.

    Sets measured_resistance, loads P_pi and phi0 via load_physical_parameters,
    then optionally freezes them (requires_grad=False).

    Returns (count_loaded, measured_targets) where measured_targets maps
    MZI index → {"p_pi": value, "phi0": value} for soft regularization.
    """
    with open(physics_path, "r", encoding="utf-8") as f:
        physics = json.load(f)

    loaded = 0
    measured_targets: Dict[int, Dict[str, float]] = {}
    for mzi in model.iter_mzis():
        key = str(mzi.index)
        if key not in physics:
            continue

        entry = physics[key]

        # Set measured resistance as nominal (delta_r starts from 0)
        if "resistance" in entry:
            mzi.measured_resistance = entry["resistance"]
            mzi.nominal_resistance = entry["resistance"]

        # Load P_pi and phi0 via inverse mapping
        kwargs = {}
        targets = {}
        if "p_pi" in entry:
            kwargs["p_pi"] = entry["p_pi"]
            targets["p_pi"] = entry["p_pi"]
        if "phi0" in entry:
            kwargs["phi0"] = entry["phi0"]
            targets["phi0"] = entry["phi0"]
        if kwargs:
            mzi.load_physical_parameters(**kwargs)

        if targets:
            measured_targets[mzi.index] = targets

        # Hard freeze if requested
        if freeze_p_pi:
            mzi._raw_p_pi.requires_grad_(False)
        if freeze_phi0:
            mzi._raw_phi0.requires_grad_(False)

        loaded += 1

    return loaded, measured_targets


def compute_physics_regularization(
    model: MZIArray,
    measured_targets: Dict[int, Dict[str, float]],
    lambda_p_pi: float,
    lambda_phi0: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute soft regularization loss pulling P_pi and phi0 toward measured values.

    L_reg = lambda_p_pi * Σ_i (P_pi_i - P_pi_meas_i)² + lambda_phi0 * Σ_i (phi0_i - phi0_meas_i)²

    Uses normalized units: P_pi in Watts (typical ~0.014), phi0 in radians (typical ~0.5).
    """
    reg = torch.tensor(0.0, device=device)
    for mzi in model.iter_mzis():
        if mzi.index not in measured_targets:
            continue
        targets = measured_targets[mzi.index]
        params = mzi.physical_parameters()

        if lambda_p_pi > 0 and "p_pi" in targets and mzi._raw_p_pi.requires_grad:
            target_val = torch.tensor(targets["p_pi"], dtype=torch.float32, device=device)
            reg = reg + lambda_p_pi * (params["p_pi"].squeeze() - target_val) ** 2

        if lambda_phi0 > 0 and "phi0" in targets and mzi._raw_phi0.requires_grad:
            target_val = torch.tensor(targets["phi0"], dtype=torch.float32, device=device)
            reg = reg + lambda_phi0 * (params["phi0"].squeeze() - target_val) ** 2

    return reg


# ---------------------------------------------------------------------------
# Fast CSV parser (adapted from analyze_round2_multi.py)
# ---------------------------------------------------------------------------

def fast_parse_csv(filepath: Path) -> Optional[dict]:
    """Parse a measurement CSV using only string ops (no pandas per file)."""
    meta = {}
    data_line = None
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('"#') or s.startswith("#"):
                if s.startswith('"#'):
                    s = s[2:].strip()
                if s.startswith("#"):
                    s = s[1:].strip()
                if ":" in s:
                    key, val = s.split(":", 1)
                    key, val = key.strip(), val.strip()
                    if key == "config_idx":
                        try:
                            meta[key] = int(val)
                        except ValueError:
                            meta[key] = val
                    elif key == "input_ch":
                        try:
                            meta[key] = int(val)
                        except ValueError:
                            meta[key] = val
                    elif key == "sim_in_power":
                        try:
                            meta[key] = float(val)
                        except ValueError:
                            meta[key] = val
                    elif key == "json_idx":
                        try:
                            meta[key] = int(val)
                        except ValueError:
                            meta[key] = val
                    elif key in ("block_tag", "system"):
                        meta[key] = val
                continue
            if data_line is None and not s.startswith("p_dbm"):
                data_line = s

    if data_line is None:
        return None

    parts = data_line.split(",")
    if len(parts) < 7:
        return None

    try:
        meta["p_dbm"] = float(parts[0])
        meta["pm_ch1_mean_dbm"] = float(parts[1])
        meta["pm_ch1_mean_mw"] = float(parts[3])
        meta["pm_ch2_mean_dbm"] = float(parts[4])
        meta["pm_ch2_mean_mw"] = float(parts[6])
    except (ValueError, IndexError):
        return None

    return meta


# ---------------------------------------------------------------------------
# Voltage loading
# ---------------------------------------------------------------------------

def load_voltage_table(csv_path: Path) -> Dict[str, np.ndarray]:
    """Load a voltage CSV and return {block_tag: voltage_array[50]}."""
    import csv

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # Determine block_tag columns (everything except step_idx, source_channel, system)
    skip = {"step_idx", "source_channel", "system"}
    tag_cols = [c for c in fieldnames if c not in skip]

    # Sort by step_idx to get voltage vector in MZI index order (0-49)
    rows.sort(key=lambda r: int(r["step_idx"]))

    result = {}
    for col in tag_cols:
        voltages = np.array([float(rows[i][col]) for i in range(len(rows))],
                            dtype=np.float32)
        result[col] = voltages

    return result


def load_all_voltages(
    input_data_dir: Path,
    n_jsons: int = 20,
) -> Dict[Tuple[int, str], np.ndarray]:
    """Load voltages for all (json_idx, block_tag) pairs.

    Returns dict: (json_idx, block_tag) -> voltage_array[50].
    """
    voltage_map = {}

    for json_idx in range(n_jsons):
        json_dir = input_data_dir / f"json_{json_idx}"
        if not json_dir.is_dir():
            continue

        # CNN voltages
        cnn_csv = json_dir / "mzi_voltages_with_mapping_sorted_by_source_channel.csv"
        if cnn_csv.exists():
            table = load_voltage_table(cnn_csv)
            for tag, volts in table.items():
                voltage_map[(json_idx, tag)] = volts

        # FC voltages
        fc_csv = json_dir / "fc_voltages_with_mapping_sorted_by_source_channel.csv"
        if fc_csv.exists():
            table = load_voltage_table(fc_csv)
            for tag, volts in table.items():
                voltage_map[(json_idx, tag)] = volts

    return voltage_map


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MeshMeasurementDataset:
    """
    Grouped dataset of full-mesh MZI measurements.

    Samples are grouped by voltage config (json_idx, block_tag) so that the
    transfer matrix is built once per group rather than once per sample.

    Attributes:
        unique_voltages: (N_configs, 50) unique voltage vectors
        groups: list of dicts, each with keys:
            input_ports: (G,) long
            input_powers: (G,) float
            output_ports: (G,) long
            output_powers: (G,) float
    """

    def __init__(
        self,
        mesh_dir: str,
        cache_path: Optional[str] = None,
    ) -> None:
        mesh_path = Path(mesh_dir)
        results_dir = mesh_path / "results"
        input_data_dir = mesh_path / "Input_Data"

        if cache_path is None:
            cache_path = str(mesh_path / "parsed_dataset_grouped.pt")

        if os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            cached = torch.load(cache_path, weights_only=True)
            self.unique_voltages = cached["unique_voltages"]
            self.config_ids = cached["config_ids"]
            self.json_ids = cached["json_ids"]
            self.input_ports = cached["input_ports"]
            self.input_powers = cached["input_powers"]
            self.output_ports = cached["output_ports"]
            self.output_powers = cached["output_powers"]
            self.config_to_json = cached["config_to_json"]
            self.n_configs = self.unique_voltages.shape[0]
            self.total_samples = len(self.input_ports)
            print(f"  Loaded {self.total_samples} samples, {self.n_configs} unique voltage configs")
            return

        print("Parsing measurement CSVs (first run, will be cached)...")
        t0 = time.time()

        # Load all voltages
        voltage_map = load_all_voltages(input_data_dir)
        print(f"  Loaded voltages for {len(voltage_map)} (json_idx, tag) pairs")

        # Assign integer IDs to unique voltage vectors
        vkey_to_id: Dict[tuple, int] = {}
        unique_volts_list: List[np.ndarray] = []
        config_json_map: Dict[int, int] = {}  # config_id -> json_idx
        for vkey, volts in voltage_map.items():
            vtuple = tuple(volts.tolist())
            if vtuple not in vkey_to_id:
                vkey_to_id[vtuple] = len(unique_volts_list)
                unique_volts_list.append(volts)
        # Also map (json_idx, tag) -> config_id
        jtag_to_config: Dict[Tuple[int, str], int] = {}
        jtag_to_json: Dict[Tuple[int, str], int] = {}
        for (json_idx, tag), volts in voltage_map.items():
            vtuple = tuple(volts.tolist())
            cid = vkey_to_id[vtuple]
            jtag_to_config[(json_idx, tag)] = cid
            jtag_to_json[(json_idx, tag)] = json_idx
            config_json_map[cid] = json_idx

        print(f"  {len(unique_volts_list)} unique voltage configs (from {len(voltage_map)} pairs)")

        # Parse all measurement CSVs
        all_config_ids = []
        all_json_ids = []
        all_input_ports = []
        all_input_powers = []
        all_output_ports = []
        all_output_powers = []

        n_files = 0
        n_skipped_snr = 0
        n_skipped_no_voltage = 0

        for subdir in sorted(results_dir.iterdir()):
            if not subdir.is_dir():
                continue
            dm = re.match(r"^(.+?)_in(\d+)_out(\d+)_(\d+)$", subdir.name)
            if not dm:
                continue
            out_port_a = int(dm.group(3))
            out_port_b = int(dm.group(4))

            for json_subdir in sorted(subdir.iterdir()):
                if not json_subdir.is_dir():
                    continue
                jm = re.match(r"^json_(\d+)$", json_subdir.name)
                if not jm:
                    continue
                json_idx = int(jm.group(1))

                for csv_path in json_subdir.glob("meas_*.csv"):
                    n_files += 1
                    rec = fast_parse_csv(csv_path)
                    if rec is None:
                        continue

                    block_tag = rec.get("block_tag", "")
                    input_ch = rec.get("input_ch", -1)
                    p_dbm = rec.get("p_dbm", float("nan"))
                    pm1_dbm = rec.get("pm_ch1_mean_dbm", -99)
                    pm1_mw = rec.get("pm_ch1_mean_mw", 0.0)
                    pm2_dbm = rec.get("pm_ch2_mean_dbm", -99)
                    pm2_mw = rec.get("pm_ch2_mean_mw", 0.0)

                    if not (0 <= input_ch < 10):
                        continue

                    config_key = (json_idx, block_tag)
                    if config_key not in jtag_to_config:
                        n_skipped_no_voltage += 1
                        continue
                    config_id = jtag_to_config[config_key]

                    input_power_mw = 10.0 ** (p_dbm / 10.0) if not np.isnan(p_dbm) else 0.0

                    if pm1_dbm >= MIN_PM_DBM:
                        all_config_ids.append(config_id)
                        all_json_ids.append(json_idx)
                        all_input_ports.append(input_ch)
                        all_input_powers.append(input_power_mw)
                        all_output_ports.append(out_port_a)
                        all_output_powers.append(pm1_mw)
                    else:
                        n_skipped_snr += 1

                    if pm2_dbm >= MIN_PM_DBM:
                        all_config_ids.append(config_id)
                        all_json_ids.append(json_idx)
                        all_input_ports.append(input_ch)
                        all_input_powers.append(input_power_mw)
                        all_output_ports.append(out_port_b)
                        all_output_powers.append(pm2_mw)
                    else:
                        n_skipped_snr += 1

                if n_files % 10000 == 0 and n_files > 0:
                    print(f"  Parsed {n_files} files, {len(all_config_ids)} samples...")

        elapsed = time.time() - t0
        print(f"  Parsed {n_files} files in {elapsed:.1f}s")
        print(f"  Total samples: {len(all_config_ids)}")
        print(f"  Skipped (SNR): {n_skipped_snr}, (no voltage): {n_skipped_no_voltage}")

        self.unique_voltages = torch.tensor(
            np.array(unique_volts_list), dtype=torch.float32
        )
        self.config_ids = torch.tensor(all_config_ids, dtype=torch.long)
        self.json_ids = torch.tensor(all_json_ids, dtype=torch.long)
        self.input_ports = torch.tensor(all_input_ports, dtype=torch.long)
        self.input_powers = torch.tensor(all_input_powers, dtype=torch.float32)
        self.output_ports = torch.tensor(all_output_ports, dtype=torch.long)
        self.output_powers = torch.tensor(all_output_powers, dtype=torch.float32)
        # Map config_id -> json_idx (for splitting by json)
        self.config_to_json = torch.tensor(
            [config_json_map.get(i, -1) for i in range(len(unique_volts_list))],
            dtype=torch.long,
        )
        self.n_configs = self.unique_voltages.shape[0]
        self.total_samples = len(all_config_ids)

        print(f"Caching dataset to {cache_path}")
        torch.save({
            "unique_voltages": self.unique_voltages,
            "config_ids": self.config_ids,
            "json_ids": self.json_ids,
            "input_ports": self.input_ports,
            "input_powers": self.input_powers,
            "output_ports": self.output_ports,
            "output_powers": self.output_powers,
            "config_to_json": self.config_to_json,
        }, cache_path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_init_parameters(model: MZIArray, json_path: str) -> None:
    """Warm-start from existing mzi_parameters.json.

    Supports both old format (with delta_r) and new format (with p_pi).
    Old delta_r values are ignored since resistance is now fixed from measurement.
    Respects frozen parameters: skips p_pi/phi0 if requires_grad is False.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    loaded = 0
    for mzi in model.iter_mzis():
        key = str(mzi.index)
        if key in params:
            p = params[key]
            kwargs = {
                "a": p["a"],
                "b": p["b"],
            }
            if "delta_r" in p:
                kwargs["delta_r"] = p["delta_r"]
            # Only load phi0 if it's still trainable
            if mzi._raw_phi0.requires_grad:
                kwargs["phi0"] = p["phi0"]
            # Only load p_pi if it's still trainable
            if mzi._raw_p_pi.requires_grad and "p_pi" in p:
                kwargs["p_pi"] = p["p_pi"]
            # Only load alpha if it's still trainable
            if mzi._raw_alpha.requires_grad and "alpha" in p:
                kwargs["alpha"] = p["alpha"]
            mzi.load_physical_parameters(**kwargs)
            loaded += 1

    print(f"Loaded initial parameters for {loaded} MZIs from {json_path}")


def export_parameters(model: MZIArray, output_file: str) -> None:
    """Export calibrated parameters including p_pi, alpha, measured resistance, and crosstalk."""
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

    # Export crosstalk coefficients if enabled
    if model.enable_crosstalk:
        coeffs_heat = (model._raw_crosstalk * 0.01).detach().cpu().tolist()
        coeffs_ref = (torch.nn.functional.softplus(model._raw_refarm_xt) * 0.01).detach().cpu().tolist()
        xt_list = []
        for k, (i, j) in enumerate(model.crosstalk_pairs):
            xt_list.append({
                "src": j, "dst": i,
                "coeff": coeffs_heat[k],
                "refarm": coeffs_ref[k],
            })
        result["_crosstalk"] = xt_list

    # Export temperature drift parameter
    if model.enable_temp_drift:
        gamma = (0.05 * torch.tanh(model._raw_gamma)).item()
        result["_temp_drift_gamma"] = gamma

    # Export substrate thermal coefficient
    result["_alpha_R"] = model.alpha_R().item()

    # Export detector model
    result["_det_gain"] = model.det_gain().detach().cpu().tolist()
    nf_uw = (model.noise_floor().detach().cpu() * 1000).tolist()  # mW → μW
    result["_noise_floor_uw"] = nf_uw

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Calibrated parameters written to {output_file}")


def forward_grouped(
    model: MZIArray,
    input_powers: torch.Tensor,
    output_ports: torch.Tensor,
    voltage_vec: torch.Tensor,
    device: torch.device,
    json_idx: int = 0,
) -> torch.Tensor:
    """Forward pass for a group of samples sharing the same voltage config.

    Uses Redheffer star product to cascade all layer S-matrices into a single
    combined scattering matrix, correctly accounting for back-reflection in
    vertical (column) MZI layers.

    Args:
        model: MZIArray instance.
        input_powers: (G, 10) input power tensor.
        output_ports: (G,) output port indices.
        voltage_vec: (50,) single voltage vector for this group.
        device: target device.
        json_idx: json file index for temperature drift compensation.

    Returns:
        predicted_power: (G,) predicted output power at each sample's output port.
    """
    # Apply temperature drift: scale voltages to account for R increase
    if model.enable_temp_drift:
        drift_factor = model.temp_drift_factor(json_idx).to(device)
        voltage_vec = voltage_vec * drift_factor

    # Apply global substrate temperature rise: V_eff = V × scale
    sub_scale = model.substrate_voltage_scale(voltage_vec)
    voltage_vec = voltage_vec * sub_scale

    # Compute thermal crosstalk phases if enabled
    xt_phases = None
    if model.enable_crosstalk:
        xt_phases = model.compute_crosstalk_phases(voltage_vec)  # (50,)

    N = model.port_count  # 10

    # Build combined 2N×2N scattering matrix (shared across all samples in group)
    combined = torch.zeros(2 * N, 2 * N, dtype=torch.complex64, device=device)
    eye_N = torch.eye(N, dtype=torch.complex64, device=device)
    combined[:N, N:] = eye_N
    combined[N:, :N] = eye_N

    for layer in model.layers:
        layer_volts = voltage_vec[layer.mzi_indices]
        layer_xt = xt_phases[layer.mzi_indices] if xt_phases is not None else None
        layer_matrix = layer._build_transfer_matrix(
            device, voltage_overrides=layer_volts, extra_phases=layer_xt
        )
        combined = redheffer_star_product(combined, layer_matrix, N)

    # Convert to complex amplitude
    amplitudes = torch.sqrt(torch.clamp(input_powers, min=0.0)).to(torch.complex64)

    # Apply combined S-matrix to all samples at once
    state = torch.zeros(
        amplitudes.shape[0], 2 * N, dtype=torch.complex64, device=device
    )
    state[:, :N] = amplitudes
    result = torch.matmul(state, combined.T)
    output_fields = result[:, N : 2 * N]

    # Extract predicted power at measured output ports
    port_fields = output_fields.gather(1, output_ports.unsqueeze(1)).squeeze(1)
    optical_power = port_fields.abs() ** 2

    # Per-port detector model: P_measured = P_optical × (1 + k) + nf
    k = model.det_gain().to(device)    # (10,)
    nf = model.noise_floor().to(device)  # (10,)
    port_k = k[output_ports]    # (G,)
    port_nf = nf[output_ports]  # (G,)
    return optical_power * (1.0 + port_k) + port_nf


def evaluate(
    model: MZIArray,
    config_sample_indices: List[Tuple[int, torch.Tensor, int]],
    all_input_ports: torch.Tensor,
    all_input_powers: torch.Tensor,
    all_output_ports: torch.Tensor,
    all_output_powers: torch.Tensor,
    unique_voltages: torch.Tensor,
    device: torch.device,
) -> float:
    """Compute average MSE loss over a set of configs (no gradient)."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for cid, indices, json_id in config_sample_indices:
            input_ports = all_input_ports[indices]
            input_powers = all_input_powers[indices]
            output_ports = all_output_ports[indices]
            target_powers = all_output_powers[indices]
            batch_size = len(indices)

            inputs = torch.zeros(
                batch_size, model.port_count, dtype=torch.float32, device=device
            )
            inputs.scatter_(1, input_ports.unsqueeze(1), input_powers.unsqueeze(1))

            predicted_power = forward_grouped(
                model, inputs, output_ports, unique_voltages[cid], device,
                json_idx=json_id,
            )
            loss = nn.functional.mse_loss(predicted_power, target_powers, reduction="sum")
            total_loss += loss.item()
            n_samples += batch_size
    return total_loss / n_samples if n_samples > 0 else float("nan")


def train_model(args: argparse.Namespace) -> MZIArray:
    torch.manual_seed(args.seed)

    dataset = MeshMeasurementDataset(
        mesh_dir=args.mesh_dir,
        cache_path=args.cache_path,
    )

    device = torch.device(args.device)
    model = MZIArray(
        enable_crosstalk=args.enable_crosstalk,
        enable_temp_drift=args.enable_temp_drift,
        trainable_fabrication=True,
    ).to(device)

    # Load measured physics (P_pi, phi0, resistance) — must come before warm start
    measured_targets: Dict[int, Dict[str, float]] = {}
    if args.measured_physics and os.path.exists(args.measured_physics):
        n_phys, measured_targets = load_measured_physics(
            model, args.measured_physics,
            freeze_p_pi=args.freeze_p_pi,
            freeze_phi0=args.freeze_phi0,
        )
        frozen = []
        if args.freeze_p_pi:
            frozen.append("P_pi")
        if args.freeze_phi0:
            frozen.append("phi0")
        reg_info = []
        if args.lambda_p_pi > 0 and not args.freeze_p_pi:
            reg_info.append(f"λ_P_pi={args.lambda_p_pi:.0e}")
        if args.lambda_phi0 > 0 and not args.freeze_phi0:
            reg_info.append(f"λ_phi0={args.lambda_phi0:.0e}")
        print(f"Loaded measured physics for {n_phys} MZIs from {args.measured_physics}")
        print(f"  Frozen: {', '.join(frozen) if frozen else 'none'}")
        if reg_info:
            print(f"  Soft regularization: {', '.join(reg_info)}")
    else:
        # Fallback: load resistances only
        if args.resistance_file and os.path.exists(args.resistance_file):
            resistance_map = parse_resistance_file(args.resistance_file)
            n_loaded = load_measured_resistances(model, resistance_map)
            print(f"Loaded measured resistances for {n_loaded} MZIs from {args.resistance_file}")
        else:
            print(f"WARNING: No resistance/physics file, using nominal values")

    # Warm start (a, b, alpha from previous training — does NOT overwrite frozen P_pi/phi0)
    if args.init_params and os.path.exists(args.init_params):
        load_init_parameters(model, args.init_params)

    # Build optimizer only with trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    mse = nn.MSELoss()

    # Move data to device
    all_config_ids = dataset.config_ids.to(device)
    all_input_ports = dataset.input_ports.to(device)
    all_input_powers = dataset.input_powers.to(device)
    all_output_ports = dataset.output_ports.to(device)
    all_output_powers = dataset.output_powers.to(device)
    unique_voltages = dataset.unique_voltages.to(device)

    # Split configs into train/val by json_idx
    val_jsons = set()
    if args.val_jsons:
        val_jsons = set(int(x) for x in args.val_jsons.split(","))

    train_configs = []  # list of (config_id, sample_indices, json_idx)
    val_configs = []
    for cid in range(dataset.n_configs):
        indices = (all_config_ids == cid).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            continue
        json_id = dataset.config_to_json[cid].item()
        if json_id in val_jsons:
            val_configs.append((cid, indices, json_id))
        else:
            train_configs.append((cid, indices, json_id))

    n_train = sum(len(idx) for _, idx, _ in train_configs)
    n_val = sum(len(idx) for _, idx, _ in val_configs)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    print(f"\nTrain/val split by json_idx (val={val_jsons}):")
    print(f"  Train: {n_train} samples, {len(train_configs)} configs")
    print(f"  Val:   {n_val} samples, {len(val_configs)} configs")
    print(f"  lr={args.lr}, epochs={args.epochs}, device={device}")
    print(f"  Parameters: {n_trainable} trainable / {n_total} total")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        t0 = time.time()

        # Shuffle config order each epoch
        perm = torch.randperm(len(train_configs))

        for pi in perm:
            cid, indices, json_id = train_configs[pi.item()]

            input_ports = all_input_ports[indices]
            input_powers = all_input_powers[indices]
            output_ports = all_output_ports[indices]
            target_powers = all_output_powers[indices]

            batch_size = len(indices)

            inputs = torch.zeros(
                batch_size, model.port_count, dtype=torch.float32, device=device
            )
            inputs.scatter_(1, input_ports.unsqueeze(1), input_powers.unsqueeze(1))

            voltage_vec = unique_voltages[cid]

            predicted_power = forward_grouped(
                model, inputs, output_ports, voltage_vec, device, json_idx=json_id
            )

            mse_loss = mse(predicted_power, target_powers)
            loss = mse_loss

            # Soft regularization on P_pi and phi0 (added once per config, not per sample)
            if measured_targets and (args.lambda_p_pi > 0 or args.lambda_phi0 > 0):
                reg_loss = compute_physics_regularization(
                    model, measured_targets, args.lambda_p_pi, args.lambda_phi0, device
                )
                loss = loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += mse_loss.item() * batch_size
            n_samples += batch_size

        train_loss = total_loss / n_samples
        elapsed = time.time() - t0

        # Validation
        val_loss = evaluate(
            model, val_configs, all_input_ports, all_input_powers,
            all_output_ports, all_output_powers, unique_voltages, device,
        ) if val_configs else float("nan")

        scheduler.step(train_loss)

        # Save best model by val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            all_params = [mzi.physical_parameters() for mzi in model.iter_mzis()]
            p_pis_mw = [p["p_pi"].item() * 1000 for p in all_params]  # W → mW
            phi0s = [p["phi0"].item() for p in all_params]
            alphas = [p["alpha"].item() for p in all_params]
            extra_str = f" alpha=[{min(alphas):.4f}, {max(alphas):.4f}]"
            extra_str += f" α_R={model.alpha_R().item():.5f}"
            if model.enable_crosstalk:
                xt_abs = model._raw_crosstalk.abs()
                extra_str += f" xt_max={xt_abs.max().item()*0.01:.5f} xt_mean={xt_abs.mean().item()*0.01:.5f}"
                refarm = torch.nn.functional.softplus(model._raw_refarm_xt) * 0.01
                extra_str += f" ref_max={refarm.max().item():.5f} ref_mean={refarm.mean().item():.5f}"
            if measured_targets and (args.lambda_p_pi > 0 or args.lambda_phi0 > 0):
                with torch.no_grad():
                    reg_val = compute_physics_regularization(
                        model, measured_targets, args.lambda_p_pi, args.lambda_phi0, device
                    ).item()
                extra_str += f" reg={reg_val:.4e}"
            if model.enable_temp_drift:
                gamma = (0.05 * torch.tanh(model._raw_gamma)).item()
                extra_str += f" γ={gamma:.5f}"
            det_k = model.det_gain().detach().cpu()
            extra_str += f" k=[{det_k.min().item():.4f}, {det_k.max().item():.4f}]"
            nf_uw = model.noise_floor().detach().cpu() * 1000  # mW → μW
            extra_str += f" nf=[{nf_uw.min().item():.3f}, {nf_uw.max().item():.3f}]μW"
            print(
                f"Epoch {epoch:03d}: train={train_loss:.6e} val={val_loss:.6e} "
                f"({elapsed:.1f}s) "
                f"p_pi=[{min(p_pis_mw):.2f}, {max(p_pis_mw):.2f}] mW "
                f"phi0=[{min(phi0s):.3f}, {max(phi0s):.3f}]"
                f"{extra_str}"
            )
        else:
            print(f"Epoch {epoch:03d}: train={train_loss:.6e} val={val_loss:.6e} ({elapsed:.1f}s)")

    # Restore best model
    if val_configs and best_val_loss < float("inf"):
        model.load_state_dict(best_state)
        print(f"\nRestored best model (val_loss={best_val_loss:.6e})")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MZI parameters from full-mesh measurement data."
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default="../ONN-MZI-MZIparameters/Mesh_measurements_multi",
        help="Root directory of mesh measurement data.",
    )
    parser.add_argument(
        "--resistance-file",
        type=str,
        default="MZI_resistance.md",
        help="Path to measured MZI resistance file (kΩ per MZI).",
    )
    parser.add_argument(
        "--init-params",
        type=str,
        default="results/mzi_parameters.json",
        help="Path to existing parameters for warm start.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/mzi_parameters_multi.json",
        help="Output JSON file for calibrated parameters.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to cache parsed dataset (default: auto in mesh-dir).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device.",
    )
    parser.add_argument(
        "--val-jsons",
        type=str,
        default="18,19",
        help="Comma-separated json indices to hold out for validation (default: 18,19).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--enable-crosstalk",
        action="store_true",
        help="Enable thermal crosstalk modeling between adjacent MZIs.",
    )
    parser.add_argument(
        "--enable-temp-drift",
        action="store_true",
        help="Enable per-json_idx temperature drift compensation (1 param: γ).",
    )
    parser.add_argument(
        "--measured-physics",
        type=str,
        default=None,
        help="Path to mzi_measured_physics.json (measured P_pi, phi0, resistance).",
    )
    parser.add_argument(
        "--freeze-p-pi",
        action="store_true",
        default=False,
        help="Hard-freeze P_pi at measured values.",
    )
    parser.add_argument(
        "--no-freeze-p-pi",
        action="store_false",
        dest="freeze_p_pi",
        help="Allow P_pi to be trainable (default).",
    )
    parser.add_argument(
        "--freeze-phi0",
        action="store_true",
        default=False,
        help="Hard-freeze phi0 at measured values.",
    )
    parser.add_argument(
        "--no-freeze-phi0",
        action="store_false",
        dest="freeze_phi0",
        help="Allow phi0 to be trainable (default).",
    )
    parser.add_argument(
        "--lambda-p-pi",
        type=float,
        default=0,
        help="Soft regularization weight for P_pi toward measured values (default: 0 = off). "
             "Higher = stronger pull. Scale: P_pi ~0.014 W, so (ΔP_pi)² ~2e-4 for 1 mW error.",
    )
    parser.add_argument(
        "--lambda-phi0",
        type=float,
        default=1e2,
        help="Soft regularization weight for phi0 toward measured values (default: 1e2). "
             "Lower than lambda_p_pi because phi0 measurement has larger uncertainty.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    model = train_model(args)
    model.eval()
    export_parameters(model, args.output_file)


if __name__ == "__main__":
    main()
