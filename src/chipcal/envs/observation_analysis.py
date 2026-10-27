"""Canonical measurement analysis for policy prompts and training data.

The environment returns finite-shot count datasets.  Humans can inspect plots, but
the training surface needs a stable, JSON-safe summary that does not depend on a
vision model reading pixels.  This module is that narrow layer: it turns each
measurement node into compact structured estimates derived from measured counts
and action-visible metadata only.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import xarray as xr

from chipcal.sim.belief import GridBelief

# Spectroscopy models in which the coupler is probed INDIRECTLY through a qubit: the drive
# and the readout may sit on different qubits, so the probed variable follows drive_qubit
# rather than always being the control.  Every consumer -- structured analysis, the plot the
# policy sees, and the text summary -- must agree on this, or the policy is shown one qubit
# and told about another.
INDIRECT_SPECTROSCOPY_MODELS = ("multimode_lindblad_v1", "multimode_dressed_ladder_v1")


def spectroscopy_probe_var(ds, control_var: str, target_var: Optional[str]) -> tuple[str, bool]:
    """Return (probed variable, indirect?) for a coupler_dc_spectroscopy dataset."""
    indirect = ds.attrs.get("spectroscopy_model") in INDIRECT_SPECTROSCOPY_MODELS
    if indirect and ds.attrs.get("drive_qubit") == 1 and target_var:
        return target_var, True
    return control_var, indirect


DEG = np.pi / 180.0
BLOB_THRESHOLD = 0.12
RB_IDENTIFIABLE_SIGMA_MAX = 0.05
RB_FLOOR = 0.25
RB_EARLY_FLOOR_MARGIN = 0.06


def json_safe(value: Any) -> Any:
    """Convert numpy/xarray-adjacent values into JSON-serialisable objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value


def annotate_rb_identifiability(
    analysis: dict[str, Any], benchmark_info: dict[str, Any],
) -> dict[str, Any]:
    """Attach a conservative, policy-visible quality label to an IRB fit.

    This uses only the measured survival curves and the ordinary fit statistics already
    returned to the policy.  It does not change ``F_hat``, ``F_sigma``, ``F_lcb``, pass/fail,
    or reward.  Its purpose is to prevent a floor-collapsed reference curve or a very uncertain
    ratio fit from being narrated as a high-fidelity gate merely because ``F_hat`` is large.
    """
    out = dict(analysis)
    if out.get("node") != "cz_rb":
        return out

    reasons: list[str] = []
    sigma = benchmark_info.get("F_sigma")
    if not isinstance(sigma, (int, float)) or not np.isfinite(float(sigma)):
        reasons.append("F_sigma is missing or non-finite")
    elif float(sigma) > RB_IDENTIFIABLE_SIGMA_MAX:
        reasons.append(
            f"F_sigma={float(sigma):.6g} exceeds {RB_IDENTIFIABLE_SIGMA_MAX:.3g}"
        )

    reference = (out.get("survival") or {}).get("reference") or []
    early = [
        row for row in reference
        if isinstance(row, dict)
        and isinstance(row.get("cliffords"), (int, float))
        and 0 < float(row["cliffords"]) <= 4
        and isinstance(row.get("mean_survival"), (int, float))
    ]
    floor_limit = RB_FLOOR + RB_EARLY_FLOOR_MARGIN
    if early and min(float(row["mean_survival"]) for row in early) <= floor_limit:
        reasons.append(
            f"reference survival reaches the {RB_FLOOR:.2f} floor by m<=4"
        )

    identifiable = not reasons
    out["status"] = "BENCHMARK_OBSERVED" if identifiable else "BENCHMARK_UNIDENTIFIABLE"
    out["fit_diagnostics"] = {
        "identifiable": identifiable,
        "reasons": reasons,
        "F_hat": benchmark_info.get("F_hat"),
        "F_sigma": benchmark_info.get("F_sigma"),
        "F_lcb": benchmark_info.get("F_lcb"),
        "sigma_threshold": RB_IDENTIFIABLE_SIGMA_MAX,
        "early_reference_floor_threshold": floor_limit,
    }
    return json_safe(out)


def _data_vars(ds: xr.Dataset, context: dict[str, Any]) -> tuple[str, str]:
    q_c, q_t = context.get("qubits", ("q0", "q1"))
    preferred = (f"y_{q_c}", f"y_{q_t}")
    names = [name for name in ds.data_vars if name.startswith("y_")]
    control = preferred[0] if preferred[0] in ds.data_vars else names[0]
    if preferred[1] in ds.data_vars:
        target = preferred[1]
    elif len(names) > 1:
        target = names[1]
    else:
        target = control
    return control, target


def _prob(ds: xr.Dataset, var: str) -> np.ndarray:
    return ds[var].values.astype(float) / max(float(ds.attrs.get("shots", 1.0)), 1.0)


def _axis_info(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, coord in ds.coords.items():
        if name in ("state", "quadrature"):
            continue
        vals = np.asarray(coord.values, dtype=float)
        if vals.size == 0:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": int(vals.size),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


def _smooth_trend(y: np.ndarray, window: int) -> np.ndarray:
    w = max(3, int(window) | 1)
    if y.size < w:
        return y.astype(float).copy()
    pad = w // 2
    ypad = np.pad(y.astype(float), pad, mode="edge")
    kern = np.ones(w) / w
    return np.convolve(ypad, kern, mode="valid")


def detect_tls_notches(
    currents: np.ndarray,
    p1: np.ndarray,
    *,
    frac_thr: float = 0.25,
    min_clean: int = 2,
) -> dict[str, Any]:
    """Detect suppressed spectroscopy-line weight from measured probabilities."""
    currents = np.asarray(currents, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    floor = np.median(p1, axis=1, keepdims=True)
    weight = np.clip(p1 - floor, 0.0, None).sum(axis=1)
    n = weight.size
    out: dict[str, Any] = {"contaminated": [], "clean_biases_A": []}
    if n < 5 or float(np.nanmax(weight)) <= 0.0:
        return out

    trend = _smooth_trend(weight, max(5, n // 5))
    supp = np.clip((trend - weight) / np.maximum(trend, 1e-9), 0.0, 1.0)
    mad = float(np.median(np.abs(supp - np.median(supp)))) * 1.4826 + 1e-9
    threshold = max(frac_thr, 6.0 * mad)
    mask = supp > threshold

    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            c = i + int(np.argmax(supp[i : j + 1]))
            out["contaminated"].append({
                "current_A": float(currents[c]),
                "suppression_fraction": float(supp[c]),
            })
            i = j + 1
        else:
            i += 1

    clean_pool = [k for k in range(n) if not mask[k]]
    clean_pool.sort(key=lambda k: weight[k], reverse=True)
    out["clean_biases_A"] = [float(currents[k]) for k in clean_pool[:min_clean]]
    return out


def _fringe_phase(phases_deg: np.ndarray, p1: np.ndarray) -> float:
    x = phases_deg * DEG
    design = np.column_stack([np.cos(x), np.sin(x), np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, p1, rcond=None)
    return float(np.arctan2(coef[1], coef[0]))


def _wrap_angle(x: float) -> float:
    return float((float(x) + np.pi) % (2.0 * np.pi) - np.pi)


def _ranked_parametrization_points(ds: xr.Dataset, control_var: str, n: int = 6) -> list[dict[str, Any]]:
    p2 = _prob(ds, control_var)[2]
    currents = np.asarray(ds["dc_currents"].values, dtype=float)
    freqs = np.asarray(ds["cz_pulse_frequencies"].values, dtype=float)
    amps = np.asarray(ds["cz_pulse_amplitudes"].values, dtype=float)
    flat = p2.reshape(-1)
    idxs = np.argsort(flat)[-min(n, flat.size) :][::-1]
    rows: list[dict[str, Any]] = []
    for idx in idxs:
        i_i, i_f, i_a = np.unravel_index(int(idx), p2.shape)
        rows.append({
            "current_A": float(currents[i_i]),
            "frequency_Hz": float(freqs[i_f]),
            "amplitude": float(amps[i_a]),
            "p2": float(flat[idx]),
        })
    return rows


def _chevron_candidates_from_signal(
    signal: np.ndarray,
    freqs: np.ndarray,
    durations: np.ndarray,
    n: int = 8,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i_f, freq in enumerate(freqs):
        row = signal[i_f]
        peak_idx = int(np.argmax(row))
        peak = float(row[peak_idx])
        tail = np.arange(peak_idx + 1, row.size) if peak_idx + 1 < row.size else np.arange(row.size)
        ret_idx = int(tail[np.argmin(row[tail])])
        ret = float(row[ret_idx])
        rows.append({
            "frequency_Hz": float(freq),
            "peak_probability": peak,
            "peak_duration_s": float(durations[peak_idx]),
            "return_probability": ret,
            "return_duration_s": float(durations[ret_idx]),
            "score": float(peak - ret),
        })
    rows.sort(reverse=True, key=lambda row: row["score"])
    return rows[:n]


def _chevron_candidates(ds: xr.Dataset, control_var: str, n: int = 8) -> list[dict[str, Any]]:
    p2 = _prob(ds, control_var)[2]
    freqs = np.asarray(ds["cz_pulse_frequencies"].values, dtype=float)
    durations = np.asarray(ds["cz_pulse_durations"].values, dtype=float)
    return _chevron_candidates_from_signal(p2, freqs, durations, n=n)


def _fit_decay_time(delays: np.ndarray, p1: np.ndarray) -> Optional[float]:
    """Rough count-only exponential time estimate for prompt features."""
    delays = np.asarray(delays, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    if delays.size < 4 or np.nanmax(p1) <= np.nanmin(p1):
        return None
    floor = float(np.nanmin(p1))
    y = np.clip(p1 - floor, 1e-6, None)
    mask = y > max(0.03, 0.08 * float(np.nanmax(y)))
    if mask.sum() < 3:
        mask = np.ones_like(y, dtype=bool)
    slope, _ = np.polyfit(delays[mask], np.log(y[mask]), 1)
    if slope >= 0.0:
        return None
    return float(-1.0 / slope)


def _qubit_rows(ds: xr.Dataset, context: dict[str, Any]) -> list[tuple[str, str]]:
    q_c, q_t = context.get("qubits", ("q0", "q1"))
    rows = []
    for q in (q_c, q_t):
        var = f"y_{q}"
        if var in ds.data_vars:
            rows.append((q, var))
    if not rows:
        rows = [(name.removeprefix("y_"), name) for name in ds.data_vars if name.startswith("y_")]
    return rows


def analyze_dataset(ds: xr.Dataset, context: dict[str, Any]) -> dict[str, Any]:
    """Return a compact structured analysis of a measurement dataset."""
    if ds.attrs.get("discovery"):
        from .one_qubit_discovery import analyze
        return analyze(ds)
    node = str(ds.attrs.get("node", "dataset"))
    control_var, target_var = _data_vars(ds, context)
    out: dict[str, Any] = {
        "node": node,
        "shots": int(ds.attrs.get("shots", 0) or 0),
        "control_var": control_var,
        "target_var": target_var,
        "axes": _axis_info(ds),
    }

    if node == "coupler_dc_spectroscopy":
        probe_var, indirect = spectroscopy_probe_var(ds, control_var, target_var)
        p1 = _prob(ds, probe_var)[1]
        currents = np.asarray(ds["dc_currents"].values, dtype=float)
        freqs = np.asarray(ds["qubit_frequencies"].values, dtype=float)
        ridge_idx = np.argmax(p1, axis=1)
        ridge_prob = p1[np.arange(p1.shape[0]), ridge_idx]
        top = np.argsort(ridge_prob)[-min(6, ridge_prob.size) :][::-1]
        contrast = float(np.nanmax(ridge_prob) - np.nanmedian(p1))
        notches = detect_tls_notches(currents, p1)
        out.update({
            "kind": "spectroscopy_ridge",
            "status": "TLS_NOTCH_DETECTED" if notches["contaminated"] else (
                "RIDGE_VISIBLE" if contrast > 0.04 else "NO_SIGNAL"
            ),
            "ridge_frequency_span_Hz": float(freqs[ridge_idx].max() - freqs[ridge_idx].min()),
            "ridge_contrast": contrast,
            "top_ridge_rows": [
                {
                    "current_A": float(currents[i]),
                    "ridge_frequency_Hz": float(freqs[ridge_idx[i]]),
                    "p1": float(ridge_prob[i]),
                }
                for i in top
            ],
            "tls": notches,
        })
        if indirect:
            # A ridge disappearing from a narrow window is not a TLS diagnosis.
            out["status"] = "RIDGE_VISIBLE" if contrast > 0.04 else "NO_SIGNAL"
            out["probe_var"] = probe_var
            out["tls"] = {**notches, "cause_identified": False,
                          "interpretation": "Contrast dips may be window clipping, hybridization, loss, or defects; compare both qubits and neighbouring windows."}
            out["sampling"] = {
                "frequency_points": int(freqs.size), "bias_points": int(currents.size),
                "max_frequency_step_Hz": float(np.max(np.abs(np.diff(freqs)))) if freqs.size > 1 else None,
                "max_bias_step_A": float(np.max(np.abs(np.diff(currents)))) if currents.size > 1 else None,
            }

    elif node == "cz_parametrization":
        p2 = _prob(ds, control_var)[2]
        contrast = float(np.nanmax(p2) - np.nanmedian(p2))
        out.update({
            "kind": "transfer_blob",
            "status": "TRANSFER_BLOB" if contrast >= BLOB_THRESHOLD else "FLAT_MAP",
            "max_p2": float(np.nanmax(p2)),
            "median_p2": float(np.nanmedian(p2)),
            "contrast": contrast,
            "top_transfer_points": _ranked_parametrization_points(ds, control_var),
        })

    elif node == "cz_chevron":
        p2 = _prob(ds, control_var)[2]
        contrast = float(np.nanmax(p2) - np.nanmedian(p2))
        freqs = np.asarray(ds["cz_pulse_frequencies"].values, dtype=float)
        belief_candidate = None
        if freqs.size:
            # A deterministic calculation from visible data only. Identity confusion is the
            # fallback when no readout characterization is in the policy-visible context.
            q_control = str(context.get("qubits", ("q0", "q1"))[0])
            confusion = np.asarray(
                (context.get("confusion") or {}).get(q_control, np.eye(3)),
                dtype=float,
            )
            belief = GridBelief.from_context(
                delta_center=float(0.5 * (freqs.min() + freqs.max())),
                delta_span=16e6,
                g_max=10e6,
            )
            belief.update_from_chevron(ds, control_var, confusion)
            belief_candidate = belief.commit_candidates()
        out.update({
            "kind": "transfer_return_chevron",
            "status": "RETURN_CANDIDATES" if contrast >= 0.08 else "WEAK_OR_FLAT_CHEVRON",
            "max_p2": float(np.nanmax(p2)),
            "median_p2": float(np.nanmedian(p2)),
            "contrast": contrast,
            "belief_map_candidate": belief_candidate,
            "transfer_return_candidates": _chevron_candidates(ds, control_var),
        })

    elif node == "swap_chevron":
        p_control = _prob(ds, control_var)[1]
        p_target = _prob(ds, target_var)[0]
        signal = np.maximum(p_control, p_target)
        freqs = np.asarray(ds["cz_pulse_frequencies"].values, dtype=float)
        durations = np.asarray(ds["cz_pulse_durations"].values, dtype=float)
        contrast = float(np.nanmax(signal) - np.nanmedian(signal))
        out.update({
            "kind": "single_excitation_swap_chevron",
            "status": "SWAP_RETURN_CANDIDATES" if contrast >= 0.08 else "WEAK_OR_FLAT_CHEVRON",
            "max_swap_probability": float(np.nanmax(signal)),
            "median_swap_probability": float(np.nanmedian(signal)),
            "contrast": contrast,
            "transfer_return_candidates": _chevron_candidates_from_signal(signal, freqs, durations),
        })

    elif node == "cz_calibration":
        phases = np.asarray(ds["ramsey_phases"].values, dtype=float)
        shots = max(float(ds.attrs.get("shots", 1.0)), 1.0)
        shifts: list[dict[str, Any]] = []
        for w in range(ds.sizes["working_points"]):
            p1 = ds[target_var].values[1, w].astype(float) / shots
            phi_off = _fringe_phase(phases, p1[0])
            phi_on = _fringe_phase(phases, p1[1])
            shift = _wrap_angle(phi_on - phi_off)
            shifts.append({
                "frequency_Hz": float(ds.attrs["wp_frequencies"][w]),
                "duration_s": float(ds.attrs["wp_durations"][w]),
                "conditional_phase_rad": shift,
                "abs_pi_error_rad": float(abs(abs(shift) - np.pi)),
            })
        shifts.sort(key=lambda row: row["abs_pi_error_rad"])
        out.update({
            "kind": "conditional_phase_ramsey",
            "status": "PHASE_NEAR_PI" if shifts and shifts[0]["abs_pi_error_rad"] < 0.25 else "NEEDS_REFINEMENT",
            "ranked_working_points": shifts[:9],
        })

    elif node == "cz_local_phases":
        phases = np.asarray(ds["local_phases"].values, dtype=float)
        shots = max(float(ds.attrs.get("shots", 1.0)), 1.0)
        estimates: dict[str, float] = {}
        for swap, var, label in ((1, control_var, "control"), (0, target_var, "target")):
            p1 = ds[var].values[1, :, swap].astype(float) / shots
            estimates[label] = _wrap_angle(_fringe_phase(phases, p1[1]) - _fringe_phase(phases, p1[0])) % (2 * np.pi)
        out.update({
            "kind": "local_phase_ramsey",
            "status": "LOCAL_PHASES_ESTIMATED",
            "phase_corrections_rad": estimates,
        })

    elif node == "cz_rb":
        counts = ds["y_survival"].values.astype(float)
        shots = max(float(ds.attrs.get("shots", 1.0)), 1.0)
        lengths = np.asarray(ds["number_of_cliffords"].values, dtype=int)
        mean_survival = counts.mean(axis=1) / shots
        out.update({
            "kind": "benchmark_survival",
            "status": "BENCHMARK_OBSERVED",
            "survival": {
                label: [
                    {"cliffords": int(m), "mean_survival": float(mean_survival[mode_idx, i])}
                    for i, m in enumerate(lengths)
                ]
                for mode_idx, label in enumerate(("reference", "interleaved"))
            },
        })

    elif node == "gate_class_estimate":
        out.update({
            "kind": "native_gate_class",
            "status": "GATE_CLASS_ESTIMATED",
            "estimate": {
                "swap_angle_rad": float(ds.attrs.get("estimate_swap_angle", 0.0)),
                "conditional_phase_rad": float(ds.attrs.get("estimate_cond_phase", 0.0)),
                "leakage": float(ds.attrs.get("estimate_leakage", 0.0)),
                "weyl": [float(x) for x in ds.attrs.get("estimate_weyl", [])],
                "nearest_gate": ds.attrs.get("estimate_nearest_gate"),
                "weyl_distance": float(ds.attrs.get("estimate_weyl_distance", 0.0)),
                "entangling_power": float(ds.attrs.get("estimate_entangling_power", 0.0)),
                "perfect_entangler": bool(ds.attrs.get("estimate_perfect_entangler", False)),
            },
        })

    elif node == "qubit_spectroscopy":
        freqs = np.asarray(ds["qubit_frequencies"].values, dtype=float)
        transition = int(ds.attrs.get("transition", 1))
        state_idx = 1 if transition == 1 else 2
        peaks = {}
        for q, var in _qubit_rows(ds, context):
            p = _prob(ds, var)[state_idx]
            idx = int(np.argmax(p))
            peaks[q] = {
                "transition": transition,
                "frequency_Hz": float(freqs[idx]),
                "peak_probability": float(p[idx]),
                "contrast": float(np.nanmax(p) - np.nanmedian(p)),
            }
        out.update({
            "kind": "qubit_spectroscopy",
            "status": "QUBIT_LINES_ESTIMATED",
            "peaks": peaks,
        })
        if ds.attrs.get("spectroscopy_model") in INDIRECT_SPECTROSCOPY_MODELS:
            # Only one qubit is driven. Do not turn an idle qubit's noise maximum
            # into an f01 estimate. This is a conservative visibility heuristic,
            # not a confidence interval for the transition frequency.
            unresolved = []
            shots = max(float(ds.attrs.get("shots", 1)), 1)
            for q, var in _qubit_rows(ds, context):
                p = _prob(ds, var)[state_idx]
                baseline = float(np.median(p))
                threshold = max(0.04, 5 * np.sqrt(max(baseline * (1-baseline), 1/shots)/shots))
                if peaks[q]["contrast"] < threshold:
                    unresolved.append(q)
            out["peaks"] = {q: p for q, p in peaks.items() if q not in unresolved}
            out["unresolved_qubits"] = unresolved
            out["status"] = "QUBIT_LINES_ESTIMATED" if out["peaks"] else "NO_SIGNAL"

    elif node == "qubit_t1":
        delays = np.asarray(ds["delays"].values, dtype=float)
        estimates = {}
        for q, var in _qubit_rows(ds, context):
            p1 = _prob(ds, var)[1]
            estimates[q] = {
                "t1_estimate_s": _fit_decay_time(delays, p1),
                "p1_start": float(p1[0]),
                "p1_end": float(p1[-1]),
            }
        out.update({"kind": "qubit_t1", "status": "T1_ESTIMATED", "estimates": estimates})

    elif node == "qubit_ramsey":
        delays = np.asarray(ds["delays"].values, dtype=float)
        estimates = {}
        for q, var in _qubit_rows(ds, context):
            p1 = _prob(ds, var)[1]
            estimates[q] = {
                "contrast": float(np.nanmax(p1) - np.nanmin(p1)),
                "t2_envelope_estimate_s": _fit_decay_time(delays, np.abs(p1 - np.nanmean(p1))),
                "ramsey_detuning_Hz": float(ds.attrs.get("ramsey_detuning", 0.0)),
            }
        out.update({"kind": "qubit_ramsey", "status": "RAMSEY_OBSERVED", "estimates": estimates})

    elif node == "readout_calibration":
        matrices = {}
        for q, var in _qubit_rows(ds, context):
            matrix = _prob(ds, var)
            matrices[q] = {
                "confusion_matrix": matrix,
                "assignment_diagonal": np.diag(matrix),
            }
        out.update({"kind": "readout_calibration", "status": "READOUT_ESTIMATED", "qubits": matrices})

    elif node == "parametric_resonance_scan":
        freqs = np.asarray(ds["cz_pulse_frequencies"].values, dtype=float)
        p_target = _prob(ds, target_var)
        p_control = _prob(ds, control_var)
        swap_curve = p_target[0, 0] if p_target.ndim >= 3 else p_target.reshape(-1)
        leak_curve = p_control[2, 1] if p_control.ndim >= 3 and p_control.shape[1] > 1 else p_control.reshape(-1)
        out.update({
            "kind": "parametric_resonance_scan",
            "status": "RESONANCES_ESTIMATED",
            "iswap_peak": {
                "frequency_Hz": float(freqs[int(np.argmax(swap_curve))]),
                "probability": float(np.nanmax(swap_curve)),
            },
            "cz_like_peak": {
                "frequency_Hz": float(freqs[int(np.argmax(leak_curve))]),
                "probability": float(np.nanmax(leak_curve)),
            },
        })

    else:
        out.update({"kind": "unclassified_dataset", "status": "UNCLASSIFIED"})

    return json_safe(out)


def analysis_prompt_block(analysis: dict[str, Any]) -> str:
    """Compact prompt rendering of an analysis block."""
    import json

    return json.dumps(json_safe(analysis), sort_keys=True, separators=(",", ":"))
