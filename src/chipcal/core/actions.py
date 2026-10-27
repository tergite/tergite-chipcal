"""Action grammar.

An action is a measurement node with its sweep axes, a commit of one control parameter, a
benchmark, or stop. Node names and sweep keys follow the vocabulary of the open-source
tergite-autocalibration tool. Axes are (center, span, n) triples or explicit value lists;
``validate`` returns every violation of the grammar and of the instrument limits, the same
mask the JSON schema expresses for guided decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chipcal.core.config import COMMITTABLE

MEASUREMENT_NODES = ('coupler_dc_spectroscopy', 'cz_parametrization', 'cz_chevron', 'cz_calibration', 'cz_local_phases', 'qubit_spectroscopy', 'qubit_t1', 'qubit_ramsey', 'readout_calibration')

# Measurement nodes runnable with no committed bias (the scouting / 1Q-characterize
# protocols the agent runs first; everything else needs a parking_current or its
# own dc_currents axis).
BIAS_FREE_NODES = frozenset({
    "coupler_dc_spectroscopy",
    "qubit_spectroscopy", "qubit_t1", "qubit_ramsey", "readout_calibration",
})

# Instrument limits of the simulated control electronics (generic, conservative bounds).
LIMITS = {
    "dc_current": (-3.1e-3, 3.1e-3),       # A
    "drive_frequency": (1e7, 1.5e9),       # Hz (parametric coupler drive)
    "qubit_frequency": (2.0e9, 9.0e9),     # Hz (spectroscopy drive)
    "amplitude": (0.0, 1.0),               # normalized
    "duration": (8e-9, 2e-6),              # s
    "relax_delay": (0.0, 500e-6),          # s (T1/Ramsey idle delays)
    "ramsey_detuning": (-20e6, 20e6),      # Hz (artificial Ramsey detuning)
    "phase_deg": (0.0, 720.0),
    "shots": (16, 4096),
    "max_grid_points": 5000,     # bounds simulation time per measurement
    "max_working_points": 25,
}

# Required / optional sweep axes per node (samplespace keys).
NODE_AXES = {
    "coupler_dc_spectroscopy": {
        "required": ("dc_currents", "qubit_frequencies"),
        "optional": ("drive_qubit", "spec_pulse_duration", "spec_pulse_amplitude"),
    },
    "cz_parametrization": {
        "required": ("cz_pulse_frequencies", "cz_pulse_amplitudes"),
        "optional": ("dc_currents", "cz_pulse_duration"),
    },
    "cz_chevron": {
        "required": ("cz_pulse_frequencies", "cz_pulse_durations"),
        "optional": ("cz_pulse_amplitude",),
    },
    "cz_calibration": {
        "required": ("working_points",),
        "optional": ("ramsey_phases",),
    },
    "cz_local_phases": {
        "required": (),
        "optional": ("local_phases",),
    },
    "qubit_spectroscopy": {
        "required": ("qubit_frequencies",),
        "optional": ("transition", "drive_qubit", "spec_pulse_duration", "spec_pulse_amplitude"),
    },
    "qubit_t1": {
        "required": ("delays",),
        "optional": (),
    },
    "qubit_ramsey": {
        "required": ("delays",),
        "optional": ("ramsey_detuning",),
    },
    "readout_calibration": {
        "required": (),
        "optional": (),
    },
}

_AXIS_LIMIT = {
    "dc_currents": "dc_current",
    "qubit_frequencies": "qubit_frequency",
    "cz_pulse_frequencies": "drive_frequency",
    "cz_pulse_amplitudes": "amplitude",
    "cz_pulse_durations": "duration",
    "cz_pulse_duration": "duration",
    "cz_pulse_amplitude": "amplitude",
    "delays": "relax_delay",
    "ramsey_phases": "phase_deg",
    "local_phases": "phase_deg",
}


# Scalar (non-swept) sweep entries, shared by validate() and the guardrails.
_SCALAR_AXIS_LIMITS = {
    "drive_qubit": (0, 1),
    "spec_pulse_duration": (4e-9, 100e-6),
    "spec_pulse_amplitude": (0.0, 0.1),
}

# Axes carrying a pulse duration, which the schedule timing grid applies to.
_DURATION_AXES = ("cz_pulse_durations", "cz_pulse_duration", "spec_pulse_duration", "delays")

# Axes in SI units, where a decimal slip (ns written as 2.2e8, MHz written as 407) is a
# plausible transcription error. Normalized amplitudes and degree phases are excluded: there
# is no unit to confuse, so an out-of-range value there is a genuine range error.
_UNIT_SLIP_HINT = {
    "cz_pulse_durations": "durations are in seconds (220 ns is 2.2e-07)",
    "cz_pulse_duration": "durations are in seconds (50 ns is 5e-08)",
    "spec_pulse_duration": "durations are in seconds (6 us is 6e-06)",
    "delays": "delays are in seconds (10 us is 1e-05)",
    "cz_pulse_frequencies": "frequencies are in Hz (407 MHz is 4.07e+08)",
    "qubit_frequencies": "frequencies are in Hz (4.5 GHz is 4.5e+09)",
    "dc_currents": "currents are in amperes (-1.55 mA is -1.55e-03)",
}


def axis_values(spec) -> np.ndarray:
    """Resolve a sweep axis: (center, span, n) triple or explicit values."""
    if isinstance(spec, (tuple, list)) and len(spec) == 3 and all(np.isscalar(v) for v in spec):
        center, span, n = spec
        return np.linspace(center - span / 2.0, center + span / 2.0, int(n))
    return np.atleast_1d(np.asarray(spec, dtype=float))


@dataclass
class MeasureAction:
    node: str
    sweeps: dict[str, Any] = field(default_factory=dict)
    shots: int = 128


@dataclass
class CommitAction:
    param: str
    value: float


@dataclass
class BenchmarkAction:
    pass


@dataclass
class StopAction:
    pass


Action = MeasureAction | CommitAction | BenchmarkAction | StopAction


def parse_action(a: dict | Action) -> Action:
    """Accept the dict form of an action or an already-constructed Action."""
    if isinstance(a, (MeasureAction, CommitAction, BenchmarkAction, StopAction)):
        return a
    if not isinstance(a, dict):
        raise TypeError(f"action must be a dict or Action, got {type(a)}")
    if a.get("stop"):
        return StopAction()
    if a.get("benchmark"):
        return BenchmarkAction()
    if "commit" in a:
        c = a["commit"]
        return CommitAction(param=c["param"], value=float(c["value"]))
    if "node" in a:
        return MeasureAction(node=a["node"], sweeps=dict(a.get("sweeps", {})), shots=int(a.get("shots", 128)))
    raise ValueError(f"unparseable action: {a!r}")


def _check_axis(name: str, values: np.ndarray, errors: list[str]) -> None:
    lim_key = _AXIS_LIMIT.get(name)
    if lim_key is None:
        return
    lo, hi = LIMITS[lim_key]
    if values.size == 0:
        errors.append(f"{name}: empty axis")
    elif not np.isfinite(values).all():
        errors.append(f"{name}: axis values must be finite")
    elif values.min() < lo - 8*np.finfo(float).eps*max(abs(lo), abs(hi)) or values.max() > hi + 8*np.finfo(float).eps*max(abs(lo), abs(hi)):
        errors.append(f"{name}: values [{values.min():.4g}, {values.max():.4g}] outside [{lo:.4g}, {hi:.4g}]")


def validate(action: Action, k=None) -> list[str]:
    """Grammar mask: return a list of violations (empty = valid).

    `k` (CommittedConfig) is needed for nodes that fall back to committed
    values (chevron amplitude, calibration/local-phase configs).
    """
    errors: list[str] = []
    if isinstance(action, (BenchmarkAction, StopAction)):
        return errors

    if isinstance(action, CommitAction):
        if action.param not in COMMITTABLE:
            errors.append(f"commit: unknown param {action.param!r}")
            return errors
        lim = {
            "parking_current": "dc_current",
            "cz_pulse_frequency": "drive_frequency",
            "cz_pulse_amplitude": "amplitude",
            "cz_pulse_duration": "duration",
        }.get(action.param)
        if lim is not None:
            lo, hi = LIMITS[lim]
            if not (lo <= action.value <= hi):
                errors.append(f"commit {action.param}: {action.value:.4g} outside [{lo:.4g}, {hi:.4g}]")
        return errors

    if action.node not in MEASUREMENT_NODES:
        errors.append(f"unknown node {action.node!r}")
        return errors

    lo, hi = LIMITS["shots"]
    if not (lo <= action.shots <= hi):
        errors.append(f"shots {action.shots} outside [{lo}, {hi}]")

    spec = NODE_AXES[action.node]
    for ax in spec["required"]:
        if ax not in action.sweeps:
            errors.append(f"{action.node}: missing required axis {ax!r}")
    for ax in action.sweeps:
        if ax not in spec["required"] + spec["optional"]:
            errors.append(f"{action.node}: unexpected axis {ax!r}")

    n_points = 1
    for ax, raw in action.sweeps.items():
        if ax in _SCALAR_AXIS_LIMITS:
            lo, hi = _SCALAR_AXIS_LIMITS[ax]
            if not np.isscalar(raw) or isinstance(raw, (str, bool)) or not np.isfinite(raw) or not lo <= float(raw) <= hi:
                errors.append(f"{ax}: finite scalar in [{lo:g}, {hi:g}] required")
            elif ax == "drive_qubit" and float(raw) not in (0, 1):
                errors.append("drive_qubit: 0 (control) or 1 (target) required")
        elif ax == "working_points":
            wps = list(raw)
            if not (1 <= len(wps) <= LIMITS["max_working_points"]):
                errors.append(f"working_points: {len(wps)} outside [1, {LIMITS['max_working_points']}]")
            for f_d, t_g in wps:
                flo, fhi = LIMITS["drive_frequency"]
                tlo, thi = LIMITS["duration"]
                if not (flo <= f_d <= fhi):
                    errors.append(f"working point frequency {f_d:.4g} outside [{flo:.4g}, {fhi:.4g}]")
                if not (tlo <= t_g <= thi):
                    errors.append(f"working point duration {t_g:.4g} outside [{tlo:.4g}, {thi:.4g}]")
            n_points *= max(len(wps), 1)
        elif ax in ("cz_pulse_duration", "cz_pulse_amplitude"):
            # fixed (non-swept) parameters of the schedule: scalars only
            if np.isscalar(raw):
                _check_axis(ax + "s", np.atleast_1d(float(raw)), errors)
            else:
                errors.append(f"{ax}: must be a scalar, got {type(raw).__name__}")
        elif ax == "transition":
            if raw not in (1, 2, 1.0, 2.0):
                errors.append("transition: must be 1 (|0>-|1>, f01) or 2 (|1>-|2>, f12)")
        elif ax == "ramsey_detuning":
            if not np.isscalar(raw):
                errors.append("ramsey_detuning: must be a scalar")
            else:
                lo, hi = LIMITS["ramsey_detuning"]
                if not (lo <= float(raw) <= hi):
                    errors.append(f"ramsey_detuning: {float(raw):.4g} outside [{lo:.4g}, {hi:.4g}]")
        else:
            values = axis_values(raw)
            _check_axis(ax, values, errors)
            n_points *= values.size

    if n_points > LIMITS["max_grid_points"]:
        errors.append(f"grid has {n_points} points > {LIMITS['max_grid_points']}")

    # Fallbacks to committed configuration
    if k is not None:
        if action.node == "cz_chevron" and "cz_pulse_amplitude" not in action.sweeps and k.cz_pulse_amplitude is None:
            errors.append("cz_chevron: no amplitude in sweeps and none committed")
        if action.node == "cz_calibration" and k.cz_pulse_amplitude is None:
            errors.append("cz_calibration: requires committed cz_pulse_amplitude")
        if action.node == "cz_local_phases" and not k.is_complete():
            errors.append("cz_local_phases: requires complete committed CZ configuration")
        needs_bias = action.node not in BIAS_FREE_NODES and "dc_currents" not in action.sweeps
        if needs_bias and k.parking_current is None:
            errors.append(f"{action.node}: no dc_currents axis and no committed parking_current")
    return errors


# --------------------------------------------------------------------------- guardrails
#
# `unit_slips` names values written in the wrong SI prefix (the action is then rejected, so the
# policy corrects its own units), `coerce_action` repairs the remaining out-of-range NUMERIC values
# into LIMITS (timing grid snap, clamp) and `augment_prereq_hints` states which commit a rejected
# node needs.
# Both are enabled by the env's `guardrails` flag and are no-ops for legal actions. Only range
# violations are coerced: structural problems (unknown node/param, unexpected/missing axis,
# wrong transition, missing prerequisites) are still rejected by validate().

_COMMIT_LIMIT = {
    "parking_current": "dc_current",
    "cz_pulse_frequency": "drive_frequency",
    "cz_pulse_amplitude": "amplitude",
    "cz_pulse_duration": "duration",
}

_PREREQ_HINTS = {
    "no dc_currents axis and no committed parking_current":
        " | HINT: this node needs a committed parking_current or a dc_currents sweep axis.",
    "requires committed cz_pulse_amplitude":
        " | HINT: this node needs a committed cz_pulse_amplitude.",
    "requires complete committed CZ configuration":
        " | HINT: this node needs committed cz_pulse_frequency, cz_pulse_amplitude and cz_pulse_duration.",
}


def _clip_scalar(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


def _scale_repair_factor(values: np.ndarray, lo: float, hi: float) -> float | None:
    """Find the 10**k that carries an entirely out-of-range axis into [lo, hi], else None.

    A policy that reasons in nanoseconds ("center ~220 ns, span ~360 ns") and then writes
    2.2e8 instead of 2.2e-7 has made a transcription error, not a physics error: clamping would
    collapse the sweep onto a range edge, and guessing the unit would act on the harness's
    reading of the intent. `unit_slips` uses this to reject such actions with the unit stated.
    Only fires when *nothing* is in range, so a sweep that merely overhangs the limit is a
    range error and is clamped."""
    if values.size == 0 or not np.isfinite(values).all():
        return None
    if np.any((values >= lo) & (values <= hi)):
        return None
    nonzero = values[values != 0.0]
    if nonzero.size == 0:
        return None
    for k in sorted(range(-18, 19), key=abs):
        # SI prefixes step by 1000, so only whole prefix jumps count as a unit slip: a factor
        # of 10 or 100 is a range error for the clamp, and 1e5 would misread an MHz slip that
        # 1e6 explains exactly.
        if k == 0 or k % 3:
            continue
        scaled = values * (10.0 ** k)
        if np.all((scaled >= lo) & (scaled <= hi)):
            return 10.0 ** k
    return None


def _snap_values_to_grid(values: np.ndarray, grid: float, lo: float, hi: float) -> np.ndarray:
    """Round durations onto the schedule timing grid, staying inside [lo, hi]."""
    snapped = np.rint(values / grid) * grid
    return np.clip(snapped, np.ceil(lo / grid) * grid, np.floor(hi / grid) * grid)


def _coerce_axis(name: str, raw, grid: float | None = None):
    """Repair a sweep-axis spec into its instrument range. Returns (coerced, note|None).

    Two repairs: a snap onto the schedule timing grid for duration axes, and a clamp into the
    range. Unit slips never reach here: `unit_slips` rejects them first. Handles the (center, span, n) triple form,
    explicit arrays/lists, and scalars. Axes with no instrument limit (unknown axis /
    structural) are returned untouched for validate()."""
    if name == "drive_qubit":
        return raw, None  # structural: a wrong qubit index is a decision, not a range slip
    if name in _SCALAR_AXIS_LIMITS:
        lo, hi = _SCALAR_AXIS_LIMITS[name]
    else:
        lim_key = _AXIS_LIMIT.get(name)
        if lim_key is None:
            return raw, None
        lo, hi = LIMITS[lim_key]
    snap = grid if (grid and name in _DURATION_AXES) else None

    # (center, span, n) triple -> rescale, or clamp the centre and shrink the span to fit.
    if isinstance(raw, (tuple, list)) and len(raw) == 3 and all(np.isscalar(v) for v in raw):
        center, span, n = float(raw[0]), float(raw[1]), raw[2]
        note = None
        c = min(max(center, lo), hi)
        sp = min(abs(span), 2.0 * max(0.0, min(c - lo, hi - c)))
        if c != center or sp != span:
            note = f"{name}: sweep clamped into [{lo:.4g}, {hi:.4g}]"
        center, span = c, sp
        if snap is not None:
            values = axis_values((center, span, n))
            snapped = _snap_values_to_grid(values, snap, lo, hi)
            if not np.allclose(values, snapped, rtol=0, atol=snap * 1e-6):
                grid_note = f"{name}: snapped onto the {snap:g} s schedule grid"
                return snapped.tolist(), f"{note}; {grid_note}" if note else grid_note
        if note:
            return (center, span, n), note
        return raw, None

    # scalar
    if np.isscalar(raw):
        value = float(raw)
        notes = []
        clipped = _clip_scalar(value, lo, hi)
        if clipped != value:
            notes.append(f"{name}: {value:.4g}->{clipped:.4g} (clamped)")
        value = clipped
        if snap is not None:
            snapped = float(_snap_values_to_grid(np.atleast_1d(value), snap, lo, hi)[0])
            if abs(snapped - value) > snap * 1e-6:
                notes.append(f"{name}: {value:.4g}->{snapped:.4g} "
                             f"(snapped onto the {snap:g} s schedule grid)")
            value = snapped
        return (value, "; ".join(notes)) if notes else (raw, None)

    # explicit array of values
    arr = np.atleast_1d(np.asarray(raw, dtype=float))
    notes = []
    clipped = np.clip(arr, lo, hi)
    if not np.array_equal(arr, clipped):
        notes.append(f"{name}: values clamped into [{lo:.4g}, {hi:.4g}]")
    arr = clipped
    if snap is not None:
        snapped = _snap_values_to_grid(arr, snap, lo, hi)
        if not np.allclose(arr, snapped, rtol=0, atol=snap * 1e-6):
            notes.append(f"{name}: snapped onto the {snap:g} s schedule grid")
        arr = snapped
    return (arr.tolist(), "; ".join(notes)) if notes else (raw, None)


_COMMIT_UNIT_HINT = {
    "parking_current": "currents are in amperes (-1.55 mA is -1.55e-03)",
    "cz_pulse_frequency": "frequencies are in Hz (407 MHz is 4.07e+08)",
    "cz_pulse_duration": "durations are in seconds (220 ns is 2.2e-07)",
}


def _slip(name: str, values, lo: float, hi: float, hint: str) -> str | None:
    values = np.atleast_1d(np.asarray(values, dtype=float))
    factor = _scale_repair_factor(values, lo, hi)
    if factor is None:
        return None
    return (f"{name}: wrong units? every value lies outside [{lo:.4g}, {hi:.4g}] and would fit only "
            f"after a factor of {factor:g}; {hint}. Resend the action in SI units.")


def unit_slips(action: Action) -> list[str]:
    """Values that look like an SI-prefix slip (all of an axis out of range, in range after a factor of
    1000**n), as rejection messages. Empty for legal actions and for plain range errors, which
    `coerce_action` clamps."""
    errors: list[str] = []
    if isinstance(action, CommitAction):
        lim, hint = _COMMIT_LIMIT.get(action.param), _COMMIT_UNIT_HINT.get(action.param)
        if lim is not None and hint is not None:
            error = _slip(f"commit {action.param}", action.value, *LIMITS[lim], hint)
            if error:
                errors.append(error)
        return errors
    if isinstance(action, MeasureAction):
        for ax, raw in action.sweeps.items():
            if ax == "working_points":
                try:
                    points = np.asarray([[float(p[0]), float(p[1])] for p in raw])
                except Exception:
                    continue
                if points.size:
                    for column, lim, hint in ((0, "drive_frequency", _UNIT_SLIP_HINT["cz_pulse_frequencies"]),
                                              (1, "duration", _UNIT_SLIP_HINT["cz_pulse_durations"])):
                        error = _slip(f"working_points[{'frequency' if column == 0 else 'duration'}]",
                                      points[:, column], *LIMITS[lim], hint)
                        if error:
                            errors.append(error)
                continue
            hint = _UNIT_SLIP_HINT.get(ax)
            if hint is None:
                continue
            if ax in _SCALAR_AXIS_LIMITS:
                lo, hi = _SCALAR_AXIS_LIMITS[ax]
            elif ax in _AXIS_LIMIT:
                lo, hi = LIMITS[_AXIS_LIMIT[ax]]
            else:
                continue
            try:
                values = axis_values(raw) if not np.isscalar(raw) else np.atleast_1d(float(raw))
            except Exception:
                continue
            error = _slip(ax, values, lo, hi, hint)
            if error:
                errors.append(error)
    return errors


def coerce_action(action: Action, k=None, timing_grid_s: float | None = None) -> tuple[Action, list[str]]:
    """Guardrail: repair out-of-range numeric values into the instrument LIMITS.

    Returns (coerced_action, notes). Durations are snapped onto the `timing_grid_s` schedule
    grid when one is enforced, and out-of-range values are clamped (unit slips are rejected
    beforehand by `unit_slips`), so the
    policy's turns land on calibration decisions rather than legality; structural issues are
    left for validate() to reject. No-op (returns the same action, []) for already-legal
    actions."""
    notes: list[str] = []
    if isinstance(action, CommitAction):
        lim = _COMMIT_LIMIT.get(action.param)
        if lim is not None:
            lo, hi = LIMITS[lim]
            c = _clip_scalar(action.value, lo, hi)
            if c != float(action.value):
                notes.append(f"commit {action.param}: {float(action.value):.4g}->{c:.4g} "
                             f"(clamped into [{lo:.4g}, {hi:.4g}])")
            if action.param == "cz_pulse_duration" and timing_grid_s:
                snapped = float(_snap_values_to_grid(np.atleast_1d(c), timing_grid_s, lo, hi)[0])
                if abs(snapped - c) > timing_grid_s * 1e-6:
                    notes.append(f"commit cz_pulse_duration: {c:.4g}->{snapped:.4g} "
                                 f"(snapped onto the {timing_grid_s:g} s schedule grid)")
                c = snapped
            if c != float(action.value):
                action = CommitAction(param=action.param, value=c)
        return action, notes

    if isinstance(action, MeasureAction):
        new_sweeps: dict[str, Any] = {}
        for ax, raw in action.sweeps.items():
            if ax == "working_points":
                flo, fhi = LIMITS["drive_frequency"]
                tlo, thi = LIMITS["duration"]
                wps, changed = [], False
                for pt in raw:
                    try:
                        f_d, t_g = float(pt[0]), float(pt[1])
                    except Exception:
                        wps.append(pt)
                        continue
                    cf, ct = _clip_scalar(f_d, flo, fhi), _clip_scalar(t_g, tlo, thi)
                    if timing_grid_s:
                        ct = float(_snap_values_to_grid(np.atleast_1d(ct), timing_grid_s, tlo, thi)[0])
                    changed = changed or (cf != f_d or ct != t_g)
                    wps.append([cf, ct])
                if changed:
                    notes.append("working_points: clamped into instrument ranges"
                                 + (" and snapped onto the schedule grid" if timing_grid_s else ""))
                new_sweeps[ax] = wps
            elif ax == "ramsey_detuning" and np.isscalar(raw):
                lo, hi = LIMITS["ramsey_detuning"]
                c = _clip_scalar(raw, lo, hi)
                if c != float(raw):
                    notes.append(f"ramsey_detuning: {float(raw):.4g}->{c:.4g} (clamped)")
                new_sweeps[ax] = c
            else:
                coerced, note = _coerce_axis(ax, raw, grid=timing_grid_s)
                new_sweeps[ax] = coerced
                if note:
                    notes.append(note)
        slo, shi = LIMITS["shots"]
        shots = int(min(max(action.shots, slo), shi))
        if shots != action.shots:
            notes.append(f"shots: {action.shots}->{shots} (clamped)")
        if notes:
            action = MeasureAction(node=action.node, sweeps=new_sweeps, shots=shots)
        return action, notes

    return action, notes


def augment_prereq_hints(errors: list[str]) -> list[str]:
    """Append the missing prerequisite to prerequisite-ordering rejections. Other errors pass
    through unchanged."""
    out = []
    for e in errors:
        hint = next((h for key, h in _PREREQ_HINTS.items() if key in e), "")
        out.append(e + hint)
    return out
