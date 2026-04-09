"""Compare error metrics across all v4 model versions."""

import json
import math
import numpy as np
import torch
from train import MZIArray
from train_multi import MeshMeasurementDataset, forward_grouped


def load_model(param_path, device):
    """Load model from JSON, supporting all parameter versions."""
    with open(param_path) as f:
        params = json.load(f)

    has_xt = "_crosstalk" in params
    model = MZIArray(
        enable_crosstalk=has_xt, trainable_fabrication=True
    ).to(device)

    for mzi in model.iter_mzis():
        key = str(mzi.index)
        if key not in params:
            continue
        p = params[key]
        kwargs = {}
        for k in ("a", "b", "delta_r", "phi0", "p_pi", "alpha"):
            if k in p:
                kwargs[k] = p[k]
        if kwargs:
            mzi.load_physical_parameters(**kwargs)

    if has_xt:
        has_refarm = len(params["_crosstalk"]) > 0 and "refarm" in params["_crosstalk"][0]
        with torch.no_grad():
            # Disable refarm for versions that don't have it
            if not has_refarm:
                model._raw_refarm_xt.fill_(-20.0)  # softplus(-20) ≈ 0
            for k, entry in enumerate(params["_crosstalk"]):
                model._raw_crosstalk[k] = entry["coeff"] / 0.01
                if has_refarm:
                    rv = entry["refarm"] / 0.01
                    if rv > 0:
                        model._raw_refarm_xt[k] = math.log(math.exp(rv) - 1) if rv < 20 else rv
                    else:
                        model._raw_refarm_xt[k] = -20.0

    # Load substrate thermal coefficient
    if "_alpha_R" in params:
        aR = params["_alpha_R"]
        aR_clamped = max(1e-6, min(aR / 0.05, 1 - 1e-6))
        with torch.no_grad():
            model._raw_alpha_R.copy_(torch.tensor(math.log(aR_clamped / (1 - aR_clamped))))

    # Load detector model
    if "_det_gain" in params:
        det_k = torch.tensor(params["_det_gain"])
        det_k_clamped = torch.clamp(det_k / 0.3, -1 + 1e-6, 1 - 1e-6)
        with torch.no_grad():
            model._raw_det_gain.copy_(torch.atanh(det_k_clamped))

    if "_noise_floor_uw" in params:
        nf_uw = params["_noise_floor_uw"]
        nf_mw = torch.tensor([v / 1000 for v in nf_uw])
        nf_clamped = torch.clamp(nf_mw / 0.001, 1e-6, 1 - 1e-6)
        with torch.no_grad():
            model._raw_noise_floor.copy_(torch.log(nf_clamped / (1 - nf_clamped)))

    model.eval()
    return model, params


def run_inference(model, dataset, device):
    """Run inference on all samples, return predictions and targets."""
    n_samples = dataset.config_ids.shape[0]
    all_config_ids = dataset.config_ids.to(device)
    all_input_ports = dataset.input_ports.to(device)
    all_input_powers = dataset.input_powers.to(device)
    all_output_ports = dataset.output_ports.to(device)
    all_output_powers = dataset.output_powers.to(device)
    unique_voltages = dataset.unique_voltages.to(device)

    N = model.port_count
    all_preds = torch.zeros(n_samples, device=device)

    with torch.no_grad():
        for cid in range(dataset.n_configs):
            indices = (all_config_ids == cid).nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                continue
            inp_ports = all_input_ports[indices]
            inp_pows = all_input_powers[indices]
            out_ports = all_output_ports[indices]
            bs = len(indices)
            inputs = torch.zeros(bs, N, dtype=torch.float32, device=device)
            inputs.scatter_(1, inp_ports.unsqueeze(1), inp_pows.unsqueeze(1))
            pred = forward_grouped(model, inputs, out_ports, unique_voltages[cid], device)
            all_preds[indices] = pred

    return (
        all_preds.cpu().numpy(),
        all_output_powers.cpu().numpy(),
        all_input_powers.cpu().numpy(),
    )


def compute_metrics(preds, targets, inp_pows):
    """Compute all error metrics, return as dict."""
    mse = np.mean((preds - targets) ** 2)
    rmse_uw = np.sqrt(mse) * 1000

    # Filter above noise floor
    mask = targets > 1e-4  # 0.1 uW
    p_f, t_f, inp_f = preds[mask], targets[mask], inp_pows[mask]
    rel_err = np.abs(p_f - t_f) / t_f * 100
    signed_rel = (p_f - t_f) / t_f * 100
    abs_err_uw = np.abs(p_f - t_f) * 1000
    signed_abs_uw = (p_f - t_f) * 1000
    t_uw = t_f * 1000

    result = {
        "mse": mse,
        "rmse_uw": rmse_uw,
        "n_valid": int(mask.sum()),
        "median_rel": float(np.median(rel_err)),
        "mean_rel": float(np.mean(rel_err)),
        "bias_rel": float(np.mean(signed_rel)),
    }

    # By output power
    out_bins = [0.1, 1, 5, 10, 50, 100, 500, 10000]
    out_labels = ["0.1-1", "1-5", "5-10", "10-50", "50-100", "100-500", ">500"]
    result["by_output"] = {}
    for i, label in enumerate(out_labels):
        bmask = (t_uw >= out_bins[i]) & (t_uw < out_bins[i + 1])
        if bmask.sum() == 0:
            continue
        result["by_output"][label] = {
            "count": int(bmask.sum()),
            "median_rel": float(np.median(rel_err[bmask])),
            "median_abs_uw": float(np.median(abs_err_uw[bmask])),
            "mean_abs_uw": float(np.mean(abs_err_uw[bmask])),
            "p99_abs_uw": float(np.percentile(abs_err_uw[bmask], 99)),
            "bias_abs_uw": float(np.mean(signed_abs_uw[bmask])),
        }

    # By input power
    inp_bins = [0, 0.5, 1.0, 2.0, 3.0, 5.0, 100.0]
    inp_labels = ["<0.5", "0.5-1", "1-2", "2-3", "3-5", ">5"]
    result["by_input"] = {}
    for i, label in enumerate(inp_labels):
        bmask = (inp_f >= inp_bins[i]) & (inp_f < inp_bins[i + 1])
        if bmask.sum() == 0:
            continue
        result["by_input"][label] = {
            "count": int(bmask.sum()),
            "median_abs_uw": float(np.median(abs_err_uw[bmask])),
            "mean_abs_uw": float(np.mean(abs_err_uw[bmask])),
            "p99_abs_uw": float(np.percentile(abs_err_uw[bmask], 99)),
        }

    # Model parameters summary
    return result


def get_param_summary(params):
    """Extract parameter ranges from JSON."""
    mzi_keys = [k for k in params if k.isdigit()]
    p_pis = [params[k].get("p_pi", None) for k in mzi_keys]
    p_pis = [v * 1000 for v in p_pis if v is not None]  # W → mW
    alphas = [params[k].get("alpha", None) for k in mzi_keys]
    alphas = [v for v in alphas if v is not None]

    summary = {}
    if p_pis:
        summary["p_pi_mw"] = (min(p_pis), max(p_pis))
    if alphas:
        summary["alpha"] = (min(alphas), max(alphas))

    if "_alpha_R" in params:
        summary["alpha_R"] = params["_alpha_R"]
    if "_det_gain" in params:
        k_vals = params["_det_gain"]
        summary["det_k"] = (min(k_vals), max(k_vals))
    if "_noise_floor_uw" in params:
        nf = params["_noise_floor_uw"]
        summary["nf_uw"] = (min(nf), max(nf))

    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MeshMeasurementDataset(
        mesh_dir="../ONN-MZI-MZIparameters/Mesh_measurements_multi"
    )

    versions = {
        "v4":  "results/mzi_parameters_multi_v4.json",
        "v4b": "results/mzi_parameters_multi_v4b.json",
        "v4c": "results/mzi_parameters_multi_v4c.json",
        "v4d": "results/mzi_parameters_multi_v4d.json",
        "v4e": "results/mzi_parameters_multi_v4e.json",
        "v4f": "results/mzi_parameters_multi_v4f.json",
        "v4g": "results/mzi_parameters_multi_v4g.json",
        "v4h": "results/mzi_parameters_multi_v4h.json",
    }

    all_results = {}
    all_param_summaries = {}

    for name, path in versions.items():
        print(f"--- Evaluating {name} ({path}) ---")
        model, params = load_model(path, device)
        preds, targets, inp_pows = run_inference(model, dataset, device)
        metrics = compute_metrics(preds, targets, inp_pows)
        all_results[name] = metrics
        all_param_summaries[name] = get_param_summary(params)

    # ===== Print comparison tables =====
    names = list(versions.keys())

    print("\n" + "=" * 90)
    print("OVERALL COMPARISON")
    print("=" * 90)
    header = f"{'Metric':>20s}" + "".join(f" | {n:>10s}" for n in names)
    print(header)
    print("-" * len(header))

    print(f"{'MSE (×1e-5)':>20s}" + "".join(
        f" | {all_results[n]['mse']*1e5:>10.3f}" for n in names))
    print(f"{'RMSE (μW)':>20s}" + "".join(
        f" | {all_results[n]['rmse_uw']:>10.2f}" for n in names))
    print(f"{'Median rel %':>20s}" + "".join(
        f" | {all_results[n]['median_rel']:>9.1f}%" for n in names))
    print(f"{'Bias rel %':>20s}" + "".join(
        f" | {all_results[n]['bias_rel']:>+9.1f}%" for n in names))

    # Parameter summary
    print("\n" + "=" * 90)
    print("MODEL PARAMETERS")
    print("=" * 90)
    for param_name, fmt_func in [
        ("p_pi_mw", lambda v: f"[{v[0]:.1f}, {v[1]:.1f}]"),
        ("alpha", lambda v: f"[{v[0]:.4f}, {v[1]:.4f}]"),
        ("alpha_R", lambda v: f"{v:.5f}"),
        ("det_k", lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]"),
        ("nf_uw", lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]"),
    ]:
        row = f"{param_name:>20s}"
        for n in names:
            s = all_param_summaries[n]
            if param_name in s:
                row += f" | {fmt_func(s[param_name]):>18s}"
            else:
                row += f" | {'—':>18s}"
        print(row)

    # By output power - relative error
    print("\n" + "=" * 90)
    print("MEDIAN RELATIVE ERROR (%) by OUTPUT POWER")
    print("=" * 90)
    out_labels = ["0.1-1", "1-5", "5-10", "10-50", "50-100", "100-500"]
    header = f"{'Output(μW)':>12s}" + "".join(f" | {n:>10s}" for n in names)
    print(header)
    print("-" * len(header))
    for label in out_labels:
        row = f"{label:>12s}"
        for n in names:
            d = all_results[n]["by_output"].get(label)
            if d:
                row += f" | {d['median_rel']:>9.1f}%"
            else:
                row += f" | {'—':>10s}"
        print(row)

    # By output power - absolute error
    print("\n" + "=" * 90)
    print("MEDIAN ABSOLUTE ERROR (μW) by OUTPUT POWER")
    print("=" * 90)
    header = f"{'Output(μW)':>12s}" + "".join(f" | {n:>10s}" for n in names)
    print(header)
    print("-" * len(header))
    for label in out_labels:
        row = f"{label:>12s}"
        for n in names:
            d = all_results[n]["by_output"].get(label)
            if d:
                row += f" | {d['median_abs_uw']:>10.2f}"
            else:
                row += f" | {'—':>10s}"
        print(row)

    # By output power - mean absolute error
    print("\n" + "=" * 90)
    print("MEAN ABSOLUTE ERROR (μW) by OUTPUT POWER")
    print("=" * 90)
    header = f"{'Output(μW)':>12s}" + "".join(f" | {n:>10s}" for n in names)
    print(header)
    print("-" * len(header))
    for label in out_labels:
        row = f"{label:>12s}"
        for n in names:
            d = all_results[n]["by_output"].get(label)
            if d:
                row += f" | {d['mean_abs_uw']:>10.2f}"
            else:
                row += f" | {'—':>10s}"
        print(row)

    # By output power - P90 absolute error
    print("\n" + "=" * 90)
    print("P99 ABSOLUTE ERROR (μW) by OUTPUT POWER")
    print("=" * 90)
    header = f"{'Output(μW)':>12s}" + "".join(f" | {n:>10s}" for n in names)
    print(header)
    print("-" * len(header))
    for label in out_labels:
        row = f"{label:>12s}"
        for n in names:
            d = all_results[n]["by_output"].get(label)
            if d:
                row += f" | {d['p99_abs_uw']:>10.2f}"
            else:
                row += f" | {'—':>10s}"
        print(row)

    # By output power - bias
    print("\n" + "=" * 90)
    print("ABSOLUTE BIAS (μW) by OUTPUT POWER  (+high / -low)")
    print("=" * 90)
    header = f"{'Output(μW)':>12s}" + "".join(f" | {n:>10s}" for n in names)
    print(header)
    print("-" * len(header))
    for label in out_labels:
        row = f"{label:>12s}"
        for n in names:
            d = all_results[n]["by_output"].get(label)
            if d:
                row += f" | {d['bias_abs_uw']:>+10.2f}"
            else:
                row += f" | {'—':>10s}"
        print(row)

    # By input power - absolute error
    print("\n" + "=" * 90)
    print("MEDIAN ABSOLUTE ERROR (μW) by INPUT POWER")
    print("=" * 90)
    inp_labels = ["1-2", "2-3", "3-5", ">5"]
    header = f"{'Input(mW)':>12s}" + "".join(f" | {n:>10s}" for n in names)
    print(header)
    print("-" * len(header))
    for label in inp_labels:
        row = f"{label:>12s}"
        for n in names:
            d = all_results[n]["by_input"].get(label)
            if d:
                row += f" | {d['median_abs_uw']:>10.2f}"
            else:
                row += f" | {'—':>10s}"
        print(row)

    # P90 by input
    print("\n" + "=" * 90)
    print("P99 ABSOLUTE ERROR (μW) by INPUT POWER")
    print("=" * 90)
    header = f"{'Input(mW)':>12s}" + "".join(f" | {n:>10s}" for n in names)
    print(header)
    print("-" * len(header))
    for label in inp_labels:
        row = f"{label:>12s}"
        for n in names:
            d = all_results[n]["by_input"].get(label)
            if d:
                row += f" | {d['p99_abs_uw']:>10.2f}"
            else:
                row += f" | {'—':>10s}"
        print(row)


if __name__ == "__main__":
    main()
