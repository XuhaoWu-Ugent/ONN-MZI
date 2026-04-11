"""
Unified MZI hardware calibration loader.

Loads physical fabrication parameters from a JSON calibration file and
freezes them in every MZI instance found in the model graph.

Two-file calibration model
--------------------------
The hardware calibration is split across two files that play different
physical roles:

    results/mzi_measured_physics.json
        Per-MZI quantities measured under SINGLE-MZI operating conditions
        (one heater on at a time, near room temperature, no thermal
        cross-talk). The only field used at runtime is `resistance` --
        the directly probed nominal heater resistance R_meas[i] for each
        MZI. This is the physical baseline for that device. Other fields
        in the file (`p_pi`, `phi0`, `v_min`, `v_max`) are kept as a
        starting point for downstream calibration training and are NOT
        applied to the model here.

    results/mzi_parameters_multi.json
        Per-MZI parameters jointly fitted under MULTI-MZI operating
        conditions (all heaters driven simultaneously, real chip
        temperature, with thermal cross-talk active). The fields
        a, b, delta_r, phi0, p_pi, alpha form ONE coupled set: each
        delta_r[i] is the equivalent resistance offset RELATIVE TO
        R_meas[i] needed to reproduce the observed steady-state physics,
        and the same is true for the equivalent p_pi, phi0, etc. They
        must be loaded together and they must NOT be cross-mixed with
        the single-MZI measurement values.

Effective per-MZI resistance used by the forward model
------------------------------------------------------
        R_eff[i] = R_meas[i] + delta_r_multi[i]

This is implemented by setting `mzi.nominal_resistance = R_meas[i]`
(from `mzi_measured_physics.json`) and then loading `delta_r_multi[i]`
into the MZI's frozen `_raw_delta_r` parameter. At runtime
`mzi.py::batch_mzi_transfer_matrices` computes
`resistance = nominal_resistance + delta_r`, which is exactly R_eff.

The fitted file's own `resistance` field is metadata only and is
intentionally ignored.

Per-MZI fields read from the multi-MZI fit file
-----------------------------------------------
    a, b           - directional coupler split ratios (range [0.3, 0.7])
    delta_r        - equivalent heater resistance offset relative to
                     R_meas (Ohm, range [-200, 200])
    phi0           - intrinsic phase offset (rad, range [-pi, pi])
    p_pi           - equivalent heating power for pi phase shift
                     (W, range [5e-3, 75e-3])
    alpha          - per-MZI insertion loss amplitude (range [0.3, 1.0])
    resistance     - IGNORED (metadata; the runtime nominal resistance
                     comes from mzi_measured_physics.json)
    v_min, v_max   - IGNORED (informational voltage references)

Global hardware metadata (optional)
-----------------------------------
Top-level keys starting with an underscore are global hardware
properties attached to `model._mzi_calibration_meta` (a plain dict) so
the forward path can read them later without re-parsing the JSON.
Currently produced by the calibration pipeline:
    _crosstalk        : list of {src, dst, coeff, refarm} dicts
                        describing thermal crosstalk between MZI heaters.
    _alpha_R          : global temperature-coefficient-of-resistance scalar.
    _det_gain         : per-output-port detector gain vector.
    _noise_floor_uw   : per-output-port noise floor (uW) vector.

The MZI index inside the model is mapped to a physical key via
`index % N`, where N is the number of *numeric* entries in the JSON file.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

DEFAULT_CALIBRATION_PATH = os.path.join("results", "mzi_parameters_multi.json")
DEFAULT_MEASURED_PATH = os.path.join("results", "mzi_measured_physics.json")

# Per-MZI fields read from the multi-MZI fit file and forwarded to
# MZI.load_physical_parameters(). The fitted file's own `resistance`
# field is intentionally absent here -- the runtime baseline resistance
# comes from mzi_measured_physics.json.
_PHYSICAL_FIELDS = ("a", "b", "delta_r", "phi0", "p_pi", "alpha")


def _iter_mzi_modules(model: nn.Module) -> Iterable[nn.Module]:
    """Yield every MZI instance in the model graph."""
    # Local import to avoid a circular dependency at module load time.
    from module.MZI_array.mzi import MZI

    for module in model.modules():
        if isinstance(module, MZI):
            yield module


def _iter_channel_modules(model: nn.Module) -> Iterable[nn.Module]:
    """
    Yield every channel-shaped module that wraps a 50-MZI photonic chip
    instance and exposes a `set_crosstalk_matrix(K)` method:

      - SingleChannelFilter (used by the CNN convolution filters)
      - OpticalSliceProcessor (used by the Optical FC layer's encoder /
        decoder slices, both pos and neg paths)

    Both classes share the same Clements row/column MZI layout and
    therefore the same physical 50-MZI mapping; the calibration loader
    treats them uniformly.
    """
    from module.channel import SingleChannelFilter
    from module.optical_linear_shared import OpticalSliceProcessor

    channel_types = (SingleChannelFilter, OpticalSliceProcessor)
    for module in model.modules():
        if isinstance(module, channel_types):
            yield module


def _build_crosstalk_matrix(
    crosstalk_entries: List[Dict[str, Any]],
    num_physical: int,
) -> torch.Tensor:
    """
    Convert the sparse `_crosstalk` list into a dense
    (num_physical, num_physical) matrix `C` where

        C[dst, src] = coeff(src->dst) - refarm(src->dst)

    so that `extra_phase[dst] = sum_src C[dst, src] * V_src^2` reproduces
    the v5d crosstalk model:
        Δφ_i^xtalk = sum_j (C_ij - R_ij) * V_j^2.

    In the v5d fits `refarm` is frozen at softplus(-20) ≈ 0, but we
    still subtract it for forward-compatibility with future fits where
    the reference-arm channel is unfrozen.
    """
    C = torch.zeros(num_physical, num_physical, dtype=torch.float32)
    for entry in crosstalk_entries:
        src = int(entry["src"])
        dst = int(entry["dst"])
        if not (0 <= src < num_physical and 0 <= dst < num_physical):
            continue
        coeff = float(entry.get("coeff", 0.0))
        refarm = float(entry.get("refarm", 0.0))
        C[dst, src] = coeff - refarm
    return C


def _split_calibration(
    raw: Dict[str, Any],
) -> tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """
    Partition a raw calibration JSON into per-MZI entries (keyed by int)
    and global metadata (keys starting with underscore or otherwise
    non-numeric).
    """
    per_mzi: Dict[int, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            meta[key] = value
            continue
        try:
            idx = int(key)
        except (TypeError, ValueError):
            meta[key] = value
            continue
        per_mzi[idx] = value
    return per_mzi, meta


def _load_measured_resistances(
    measured_path: Optional[str],
    verbose: bool,
) -> Optional[Dict[int, float]]:
    """
    Read per-MZI measured nominal resistance from
    mzi_measured_physics.json and return {physical_index: R_meas}.

    Returns None if `measured_path` is None or the file does not exist
    or contains no usable resistance entries.
    """
    if measured_path is None:
        return None
    if not os.path.exists(measured_path):
        if verbose:
            print(
                f"[Calibration] Measured-physics file not found: "
                f"{measured_path}. Falling back to default nominal_resistance."
            )
        return None

    with open(measured_path, "r") as fh:
        raw = json.load(fh)

    per_mzi, _ = _split_calibration(raw)
    resistances: Dict[int, float] = {}
    for idx, entry in per_mzi.items():
        r = entry.get("resistance") if isinstance(entry, dict) else None
        if r is not None:
            resistances[idx] = float(r)

    if not resistances:
        if verbose:
            print(
                f"[Calibration] {measured_path} contains no `resistance` "
                f"entries. Falling back to default nominal_resistance."
            )
        return None

    return resistances


def load_mzi_calibration(
    model: nn.Module,
    json_path: str = DEFAULT_CALIBRATION_PATH,
    *,
    measured_resistance_path: Optional[str] = DEFAULT_MEASURED_PATH,
    voltage_init_range: Optional[tuple] = None,
    verbose: bool = True,
    freeze: bool = True,
) -> int:
    """
    Apply hardware calibration to every MZI in `model`.

    The calibration is the union of two files (see module docstring):

        - `json_path`              : multi-MZI fitted parameters
                                     (a, b, delta_r, phi0, p_pi, alpha).
        - `measured_resistance_path`: single-MZI measured nominal
                                     resistance R_meas[i], used as the
                                     per-instance baseline so that the
                                     runtime effective resistance is
                                         R_eff[i] = R_meas[i] + delta_r[i].

    Args:
        model: Top-level network containing MZI sub-modules.
        json_path: Path to the multi-MZI fit JSON.
        measured_resistance_path: Path to the single-MZI measured-physics
            JSON used to source per-MZI baseline `nominal_resistance`.
            Pass `None` to disable measured-resistance loading and keep
            the default `nominal_resistance=1200` for every MZI
            (backwards-compatible behaviour).
        voltage_init_range: (lo, hi) tuple. After loading the fabrication
            parameters, every MZI's trainable `_voltage` parameter is
            re-initialised by sampling uniformly from this interval.
            Under the multi-MZI fit's working point (R*P_pi ~= 78.5),
            the default randn*0.1 init lives in a regime where the
            per-MZI phase modulation is ~10^-4 rad and the voltage
            gradient is too small to escape collapse during training.
            Re-initialising in (-6, 6) gives E[V^2] = 12, i.e. an average
            per-MZI phase shift of ~0.5 rad — comparable to what an
            11-layer cascade needs to implement a full pi rotation.
            Pass `None` to keep the existing `_voltage` values
            (backwards-compatible behaviour).
        verbose: Print summary line(s) when True.
        freeze: After loading, freeze fabrication parameters so the
            optimiser only updates the per-MZI voltage parameter.

    Returns:
        Number of MZIs that received fitted-parameter calibration.
    """
    if not os.path.exists(json_path):
        if verbose:
            print(
                f"[Calibration] File not found: {json_path}. "
                "MZIs keep default initialisation."
            )
        return 0

    with open(json_path, "r") as fh:
        raw = json.load(fh)

    if not raw:
        if verbose:
            print(f"[Calibration] {json_path} is empty.")
        return 0

    per_mzi, meta = _split_calibration(raw)
    if not per_mzi:
        if verbose:
            print(f"[Calibration] {json_path} has no per-MZI entries.")
        return 0

    # Stash global metadata on the model so the forward path can use it
    # later (thermal crosstalk, detector gain, noise floor, ...).
    if meta:
        model._mzi_calibration_meta = meta  # type: ignore[attr-defined]

    measured_resistances = _load_measured_resistances(
        measured_resistance_path, verbose
    )

    num_physical = len(per_mzi)
    loaded_fields: set[str] = set()
    n_loaded = 0
    n_total = 0
    n_resistance_set = 0
    n_voltage_reinit = 0

    # === Thermal crosstalk: build a (num_physical, num_physical) dense
    # matrix from the `_crosstalk` list and install it on every
    # SingleChannelFilter whose total_mzis matches num_physical (i.e.
    # one channel == one physical chip instance). ===
    crosstalk_entries = meta.get("_crosstalk") if meta else None
    crosstalk_matrix: Optional[torch.Tensor] = None
    if crosstalk_entries:
        crosstalk_matrix = _build_crosstalk_matrix(crosstalk_entries, num_physical)

    n_channels_with_crosstalk = 0
    n_channels_total = 0
    if crosstalk_matrix is not None:
        for channel in _iter_channel_modules(model):
            n_channels_total += 1
            if channel.total_mzis == num_physical:
                channel.set_crosstalk_matrix(crosstalk_matrix)
                n_channels_with_crosstalk += 1

    for mzi in _iter_mzi_modules(model):
        n_total += 1
        if getattr(mzi, "index", None) is None:
            continue

        physical_index = mzi.index % num_physical
        params = per_mzi.get(physical_index)
        if params is None:
            continue

        # 1) Per-instance baseline resistance from single-MZI measurement.
        #    The fitted `delta_r` from the multi-MZI fit will be added on
        #    top of this baseline by the runtime forward model
        #    (resistance = nominal_resistance + delta_r).
        if measured_resistances is not None:
            r_meas = measured_resistances.get(physical_index)
            if r_meas is not None:
                mzi.nominal_resistance = float(r_meas)
                n_resistance_set += 1

        # 2) Multi-MZI fitted fabrication parameters. The fitted file's
        #    own `resistance` field is intentionally NOT read here -- it
        #    is metadata only.
        kwargs = {
            field: params[field]
            for field in _PHYSICAL_FIELDS
            if field in params and params[field] is not None
        }
        if not kwargs:
            continue

        mzi.load_physical_parameters(**kwargs)
        if freeze:
            mzi.freeze_fabrication_parameters()

        # Re-initialise voltage to the multi-MZI working range. The
        # default randn*0.1 init produces phase shifts of ~10^-4 rad
        # under R*P_pi ~= 78.5, which is a flat-gradient region with
        # no escape under standard optimisers. Sampling uniformly from
        # `voltage_init_range` puts every MZI directly in the regime
        # where each device contributes a meaningful (~0.1-1 rad)
        # phase modulation, so the cascade can implement weight
        # matrices from step 1.
        if voltage_init_range is not None:
            lo, hi = float(voltage_init_range[0]), float(voltage_init_range[1])
            with torch.no_grad():
                mzi._voltage.data.uniform_(lo, hi)
            n_voltage_reinit += 1

        loaded_fields.update(kwargs.keys())
        n_loaded += 1

    if verbose:
        fields_str = ", ".join(sorted(loaded_fields)) if loaded_fields else "(none)"
        meta_str = ", ".join(sorted(meta.keys())) if meta else "(none)"
        if measured_resistances is None:
            r_msg = "nominal_resistance: default"
        else:
            r_msg = (
                f"nominal_resistance: {n_resistance_set}/{n_loaded} from "
                f"{measured_resistance_path}"
            )
        if crosstalk_matrix is None:
            xtalk_msg = "crosstalk: none"
        else:
            xtalk_msg = (
                f"crosstalk: {len(crosstalk_entries)} pairs -> "
                f"{n_channels_with_crosstalk}/{n_channels_total} channels"
            )
        if voltage_init_range is None:
            v_msg = "voltage init: kept"
        else:
            v_msg = (
                f"voltage init: uniform({voltage_init_range[0]:.1f}, "
                f"{voltage_init_range[1]:.1f}) -> {n_voltage_reinit}/{n_loaded} MZIs"
            )
        print(
            f"[Calibration] {json_path}: loaded {n_loaded}/{n_total} MZIs "
            f"(fields: {fields_str}; {num_physical} physical entries; "
            f"global metadata: {meta_str}; {r_msg}; {xtalk_msg}; {v_msg})"
        )

    return n_loaded


# Backward-compatible alias used by older scripts.
def load_mzi_parameters_from_json(
    model: nn.Module,
    json_path: str = DEFAULT_CALIBRATION_PATH,
) -> None:
    """Legacy entry point preserved for older call sites."""
    load_mzi_calibration(model, json_path)
