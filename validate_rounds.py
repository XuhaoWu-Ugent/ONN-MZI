"""Validate v4b model on independent round_0 and round_1 datasets."""

import csv
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

from train import MZIArray
from train_multi import forward_grouped, fast_parse_csv, MIN_PM_DBM
from analyze_all_versions import load_model


def load_voltages_round(input_data_dir: str) -> dict:
    """Load voltage mapping for round_0/round_1 (no json subdirs).

    Returns: {block_tag: np.array(50,)} voltage vectors
    """
    input_path = Path(input_data_dir)

    # Load block_tag -> voltage_column mapping
    tag_to_col = {}
    tag_map_file = input_path / "block_tag_to_voltage_column.csv"
    with open(tag_map_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_to_col[row["block_tag"]] = row["voltage_column"]

    # Load MZI voltage table
    volt_file = input_path / "mzi_voltages_with_mapping_sorted_by_source_channel.csv"
    with open(volt_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Build voltage vectors per block_tag
    # Each row is one MZI (source_channel), columns are block_tags
    result = {}
    for tag, col in tag_to_col.items():
        volts = np.zeros(50)
        for row in rows:
            step_idx = int(row["step_idx"])
            if col in row:
                volts[step_idx] = float(row[col])
        result[tag] = volts

    return result


def parse_round_dataset(results_dir: str, voltage_map: dict):
    """Parse measurement CSVs for round_0 or round_1 (no json subdirs)."""
    results_path = Path(results_dir)

    # Assign config IDs to unique voltage vectors
    vkey_to_id = {}
    unique_volts = []
    tag_to_config = {}

    for tag, volts in voltage_map.items():
        vtuple = tuple(volts.tolist())
        if vtuple not in vkey_to_id:
            vkey_to_id[vtuple] = len(unique_volts)
            unique_volts.append(volts)
        tag_to_config[tag] = vkey_to_id[vtuple]

    all_config_ids = []
    all_input_ports = []
    all_input_powers = []
    all_output_ports = []
    all_output_powers = []
    n_files = 0
    n_skipped = 0

    for subdir in sorted(results_path.iterdir()):
        if not subdir.is_dir():
            continue
        dm = re.match(r"^(.+?)_in(\d+)_out(\d+)_(\d+)$", subdir.name)
        if not dm:
            continue
        out_port_a = int(dm.group(3))
        out_port_b = int(dm.group(4))

        # No json subdirs — CSVs directly in folder
        for csv_path in subdir.glob("meas_*.csv"):
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

            if block_tag not in tag_to_config:
                n_skipped += 1
                continue
            config_id = tag_to_config[block_tag]

            input_power_mw = 10.0 ** (p_dbm / 10.0) if not np.isnan(p_dbm) else 0.0

            if pm1_dbm >= MIN_PM_DBM:
                all_config_ids.append(config_id)
                all_input_ports.append(input_ch)
                all_input_powers.append(input_power_mw)
                all_output_ports.append(out_port_a)
                all_output_powers.append(pm1_mw)

            if pm2_dbm >= MIN_PM_DBM:
                all_config_ids.append(config_id)
                all_input_ports.append(input_ch)
                all_input_powers.append(input_power_mw)
                all_output_ports.append(out_port_b)
                all_output_powers.append(pm2_mw)

    print(f"  Parsed {n_files} files, {len(all_config_ids)} samples, "
          f"{len(unique_volts)} configs, {n_skipped} skipped (no voltage)")

    return {
        "unique_voltages": torch.tensor(np.array(unique_volts), dtype=torch.float32),
        "config_ids": torch.tensor(all_config_ids, dtype=torch.long),
        "input_ports": torch.tensor(all_input_ports, dtype=torch.long),
        "input_powers": torch.tensor(all_input_powers, dtype=torch.float32),
        "output_ports": torch.tensor(all_output_ports, dtype=torch.long),
        "output_powers": torch.tensor(all_output_powers, dtype=torch.float32),
        "n_configs": len(unique_volts),
    }


def evaluate_model_on_dataset(model, ds, device):
    """Run inference and compute error metrics."""
    N = model.port_count
    n_configs = ds["n_configs"]
    uv = ds["unique_voltages"].to(device)
    config_ids = ds["config_ids"].to(device)
    input_ports = ds["input_ports"].to(device)
    input_powers = ds["input_powers"].to(device)
    output_ports = ds["output_ports"].to(device)
    output_powers = ds["output_powers"].to(device)

    all_preds = torch.zeros(len(config_ids), device=device)

    with torch.no_grad():
        for cid in range(n_configs):
            indices = (config_ids == cid).nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                continue
            inp_ports = input_ports[indices]
            inp_pows = input_powers[indices]
            out_ports = output_ports[indices]
            bs = len(indices)
            inputs = torch.zeros(bs, N, dtype=torch.float32, device=device)
            inputs.scatter_(1, inp_ports.unsqueeze(1), inp_pows.unsqueeze(1))
            pred = forward_grouped(model, inputs, out_ports, uv[cid], device)
            all_preds[indices] = pred

    preds = all_preds.cpu().numpy()
    targets = output_powers.cpu().numpy()

    mse = np.mean((preds - targets) ** 2)
    rmse_uw = np.sqrt(mse) * 1000

    mask = targets > 1e-4
    p_f, t_f = preds[mask], targets[mask]
    abs_err_uw = np.abs(p_f - t_f) * 1000
    t_uw = t_f * 1000
    rel_err = abs_err_uw / t_uw * 100

    print(f"    MSE: {mse:.6e}  RMSE: {rmse_uw:.2f} uW")
    print(f"    Median relative error: {np.median(rel_err):.1f}%")

    # By output power
    out_bins = [0.1, 1, 5, 10, 50, 100, 500, 10000]
    out_labels = ["0.1-1", "1-5", "5-10", "10-50", "50-100", "100-500"]
    print(f"\n    {'Output(uW)':>12s} | {'MedAE(uW)':>10s} | {'MeanAE(uW)':>10s} | {'P99AE(uW)':>10s} | {'MedRel%':>8s} | {'Count':>8s}")
    print(f"    {'-'*75}")
    for i, label in enumerate(out_labels):
        bmask = (t_uw >= out_bins[i]) & (t_uw < out_bins[i + 1])
        if bmask.sum() == 0:
            continue
        med_ae = np.median(abs_err_uw[bmask])
        mean_ae = np.mean(abs_err_uw[bmask])
        p99_ae = np.percentile(abs_err_uw[bmask], 99)
        med_rel = np.median(rel_err[bmask])
        print(f"    {label:>12s} | {med_ae:>10.2f} | {mean_ae:>10.2f} | {p99_ae:>10.2f} | {med_rel:>7.1f}% | {bmask.sum():>8d}")

    return mse, rmse_uw


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = "C:/Users/17958/ONN-MZI-MZIparameters/Mesh_Measurements"

    # Load v4b model
    print("Loading v4b model...")
    model, params = load_model("results/mzi_parameters_multi_v4b.json", device)

    rounds = [
        ("round_0", f"{base}/results_round_0", f"{base}/Input_Data_round0"),
        ("round_1", f"{base}/results_round_1", f"{base}/Input_Data_round0"),  # same input data?
    ]

    # Check if round_1 has its own input data
    if os.path.isdir(f"{base}/Input_Data_round1"):
        rounds[1] = ("round_1", f"{base}/results_round_1", f"{base}/Input_Data_round1")

    for name, results_dir, input_dir in rounds:
        print(f"\n{'='*70}")
        print(f"Evaluating v4b on {name}")
        print(f"{'='*70}")

        print(f"  Loading voltages from {input_dir}")
        voltage_map = load_voltages_round(input_dir)
        print(f"  {len(voltage_map)} block_tags with voltages")

        print(f"  Parsing measurements from {results_dir}")
        ds = parse_round_dataset(results_dir, voltage_map)

        print(f"\n  Results:")
        evaluate_model_on_dataset(model, ds, device)


if __name__ == "__main__":
    main()
