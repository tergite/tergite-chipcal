"""Scripted calibration baseline: a conventional, procedural bring-up of the whole chip.

The routine a physicist would script, using only what the observation shows:

1. **Discovery** (:mod:`chipcal.baselines.discovery`): resonators, f01 and f12 of every qubit.
2. **Readout calibration** of every qubit pair (the assignment matrix the chevron fit needs).
3. **Per coupler**, in chain order:
   a. bias scout -- coupler DC spectroscopy across the bias range in a narrow window around the
      control qubit's f01; park where the qubit line moves steeply with bias (a strong
      parametric lever arm) with good contrast, outside the avoided crossing;
   b. cz_parametrization around |f01(target) - f12(control)| to find a |11> -> |20> transfer blob;
      on a flat map, try the next bias candidate;
   c. cz_chevron around the blob over durations up to ~3x the parametrization probe; the row with the
      cleanest first transfer peak and return gives the CZ frequency, and its first return the duration;
   d. cz_calibration over a 3x3 grid of working points; commit the point with conditional phase
      closest to pi;
   e. cz_local_phases; commit both single-qubit phase corrections;
   f. benchmark; then walk the drive amplitude down (x1/2, x1/4) through c-f, since a strong
      parametric drive leaks, and keep the best benchmarked gate.
4. **Recovery**: re-bias moves leak flux into neighbours through crosstalk. Re-benchmark stale
   gates, re-tune failing gates in place at the current joint bias, and as a last resort nudge the
   weakest failing gate's bias on a fixed visible lattice (never using hidden device parameters).

It runs through :class:`~chipcal.envs.chain_agent_env.ChainAgentEnv` exactly as an LLM would, so the
turn cap, lab-time budget, guardrails and score apply unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr

from chipcal.baselines.discovery import DiscoveryFailed, ScriptedDiscovery
from chipcal.core.actions import LIMITS
from chipcal.envs.observation_analysis import analyze_dataset
from chipcal.sim.physics import wrap_angle

DEG = np.pi / 180.0
PARAMETRIZATION_DURATION = 280e-9
BLOB_THRESHOLD = 0.12
BIAS_LATTICE_A = (0.1e-3, 0.3e-3, 0.6e-3, 0.9e-3)


class EpisodeOver(Exception):
    pass


class BudgetLow(Exception):
    """Few enough turns are left that the best configurations must be restored now."""


# ----------------------------------------------------------------- analysis helpers
def detect_blob(ds: xr.Dataset, control_var: str, threshold: float = BLOB_THRESHOLD) -> Optional[dict]:
    """Population-transfer blob in a parametrization scan: the max of the 3x3-smoothed control P2,
    if it clears the map's median by ``threshold``; else None."""
    shots = float(ds.attrs["shots"])
    p2 = ds[control_var].values[2].astype(float) / shots
    if p2.ndim == 3:
        p2 = p2[0]
    pad = np.pad(p2, 1, mode="edge")
    smooth = sum(pad[1 + di: 1 + di + p2.shape[0], 1 + dj: 1 + dj + p2.shape[1]]
                 for di in (-1, 0, 1) for dj in (-1, 0, 1)) / 9.0
    strength = float(smooth.max() - np.median(smooth))
    if strength < threshold:
        return None
    i, j = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    return {"frequency": float(ds["cz_pulse_frequencies"].values[i]),
            "amplitude": float(ds["cz_pulse_amplitudes"].values[j]), "strength": strength}


def fringe_phase(phases_deg: np.ndarray, p1: np.ndarray) -> float:
    """Least-squares phase of y = a cos(phi) + b sin(phi) + c -> atan2(b, a)."""
    x = np.asarray(phases_deg) * DEG
    design = np.column_stack([np.cos(x), np.sin(x), np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, p1, rcond=None)
    return float(np.arctan2(coef[1], coef[0]))


def calibration_phase_shifts(ds: xr.Dataset, target_var: str) -> list[dict]:
    """Per working point: the control-on minus control-off target fringe shift (the conditional phase)."""
    shots = float(ds.attrs["shots"])
    phases = ds["ramsey_phases"].values
    out = []
    for w in range(ds.sizes["working_points"]):
        p1 = ds[target_var].values[1, w].astype(float) / shots
        out.append({"frequency": float(ds.attrs["wp_frequencies"][w]), "duration": float(ds.attrs["wp_durations"][w]),
                    "shift": float(wrap_angle(fringe_phase(phases, p1[1]) - fringe_phase(phases, p1[0])))})
    return out


def local_phase_estimates(ds: xr.Dataset, control_var: str, target_var: str) -> tuple[float, float]:
    """(control, target) phases the gate pulse accumulates: gate-on minus gate-off fringe shifts."""
    shots = float(ds.attrs["shots"])
    phases = ds["local_phases"].values
    est = {}
    for swap, var in ((1, control_var), (0, target_var)):
        p1 = ds[var].values[1, :, swap].astype(float) / shots
        est[swap] = wrap_angle(fringe_phase(phases, p1[1]) - fringe_phase(phases, p1[0]))
    return float(est[1]), float(est[0])


def first_return(ds: xr.Dataset, control_var: str, min_peak: float = 0.3) -> Optional[dict]:
    """The CZ point of a chevron: the frequency row with the cleanest first |20> transfer peak followed by
    a return to |11>, and the duration of that first return (one full |11> <-> |20> cycle)."""
    shots = float(ds.attrs["shots"])
    p2 = ds[control_var].values[2].astype(float) / shots            # (frequency, duration)
    freqs = np.asarray(ds["cz_pulse_frequencies"].values, dtype=float)
    durations = np.asarray(ds["cz_pulse_durations"].values, dtype=float)
    kernel = np.ones(3) / 3
    best = None
    for i, row in enumerate(p2):
        smooth = np.convolve(np.pad(row, 1, mode="edge"), kernel, mode="valid")
        if smooth.max() < min_peak:
            continue
        above = np.flatnonzero(smooth >= 0.6 * smooth.max())
        peak = int(above[0] + np.argmax(smooth[above[0]: above[-1] + 1]))
        # first local minimum after the first transfer peak
        ret = None
        for j in range(peak + 1, len(smooth) - 1):
            if smooth[j] <= smooth[j - 1] and smooth[j] <= smooth[j + 1]:
                ret = j
                break
        if ret is None or smooth[ret] > 0.5 * smooth[peak]:
            continue
        score = float(smooth[peak] - smooth[ret])
        if best is None or score > best["score"]:
            best = {"frequency": float(freqs[i]), "duration": float(durations[ret]), "score": score,
                    "peak_p2": float(smooth[peak]), "return_p2": float(smooth[ret])}
    return best


def bias_from_ridge(ds: xr.Dataset, control_var: str, max_pull_hz: float = 6e6) -> Optional[float]:
    """The best parking current of a coupler DC spectroscopy map (see ``ranked_biases``)."""
    ranked = ranked_biases(ds, control_var, max_pull_hz=max_pull_hz)
    return ranked[0] if ranked else None


def ranked_biases(ds: xr.Dataset, control_var: str, max_pull_hz: float = 6e6, k: int = 3,
                  separation_a: float = 0.5e-3) -> list[float]:
    """Up to ``k`` parking currents from a coupler DC spectroscopy map, at least ``separation_a`` apart, ranked by
    |d f_qubit / d I| x contrast over rows with a resolved line not pulled into the avoided crossing. A steep
    line means a strong parametric lever arm; distinct ridge points let a flat bias be abandoned for another."""
    shots = float(ds.attrs["shots"])
    p1 = ds[control_var].values[1].astype(float) / shots
    currents = np.asarray(ds["dc_currents"].values, dtype=float)
    freqs = np.asarray(ds["qubit_frequencies"].values, dtype=float)
    if p1.ndim != 2 or currents.size < 5 or freqs.size < 3:
        return []
    peak_idx = np.argmax(p1, axis=1)
    peak_freqs = freqs[peak_idx]
    contrast = p1[np.arange(p1.shape[0]), peak_idx] - np.median(p1, axis=1)
    slope = np.gradient(peak_freqs, currents)
    pull = np.abs(peak_freqs - np.median(peak_freqs))
    valid = np.ones(currents.size, dtype=bool)
    edge = min(2, currents.size // 8)
    if edge:
        valid[:edge] = valid[-edge:] = False
    valid &= contrast > max(0.08, 0.25 * float(np.nanmax(contrast)))
    dispersive = valid & (pull <= max_pull_hz)
    if dispersive.any():
        valid = dispersive
    if not valid.any():
        return []
    score = (np.abs(slope) / (float(np.nanmax(np.abs(slope[valid]))) + 1e-12)
             * contrast / (float(np.nanmax(contrast[valid])) + 1e-12))
    score[~valid] = -np.inf
    out: list[float] = []
    for index in np.argsort(-score):
        if not np.isfinite(score[index]) or len(out) == k:
            break
        if all(abs(currents[index] - c) >= separation_a for c in out):
            out.append(float(currents[index]))
    return out


def confusion_from_readout(ds: xr.Dataset, var: str) -> np.ndarray:
    """Column-stochastic assignment matrix from a readout_calibration dataset (prepared 0, 1, 2)."""
    counts = ds[var].values.astype(float)            # (state, prepared)
    return counts / counts.sum(axis=0, keepdims=True)


# ----------------------------------------------------------------- the agent
@dataclass
class ScriptedLog:
    events: list = field(default_factory=list)
    f_lcb: dict = field(default_factory=dict)
    discovery_ok: bool = False


class ScriptedChainAgent:
    """Deterministic scripted policy over a ChainAgentEnv (discovery + every coupler + recovery)."""

    def __init__(self, env, *, shots: int = 128, max_biases: int = 3, max_passes: int = 3,
                 scout_bias_points: int = 49, scout_freq_span: float = 20e6, scout_freq_points: int = 41,
                 param_freq_span: float = 40e6, param_freq_points: int = 41,
                 param_amp_center: float = 0.4, param_amp_span: float = 0.7, param_amp_points: int = 15,
                 param_durations: tuple = (PARAMETRIZATION_DURATION, 600e-9),
                 chevron_freq_span: float = 8e6, chevron_freq_points: int = 21,
                 chevron_dur: tuple = (224e-9, 400e-9, 26), blob_threshold: float = BLOB_THRESHOLD):
        self.env = env
        self.shots = int(shots)
        self.max_biases = int(max_biases)
        self.max_passes = int(max_passes)
        self.scout = (int(scout_bias_points), float(scout_freq_span), int(scout_freq_points))
        self.param = (float(param_freq_span), int(param_freq_points), float(param_amp_center),
                      float(param_amp_span), int(param_amp_points))
        self.param_durations = tuple(param_durations)
        self.chevron = (float(chevron_freq_span), int(chevron_freq_points), tuple(chevron_dur))
        self.blob_threshold = float(blob_threshold)
        self.log = ScriptedLog()
        self.confusion: dict[str, np.ndarray] = {}
        self._bias_search: dict[int, dict] = {}
        self.outcome = None
        # Best measured configuration of every coupler: {i: (estimate, config dict)}.
        self.best: dict[int, tuple[float, dict]] = {}
        self.reserve_turns = 6 * self.inner.n_couplers + 2

    # -- env plumbing ------------------------------------------------------------
    @property
    def inner(self):
        return self.env._env

    def step(self, action: dict, *, guard: bool = True):
        if guard and self.env.turn >= self.env.max_turns - self.reserve_turns:
            raise BudgetLow()
        out = self.env.step(action)
        self.outcome = out
        if out.terminated or out.truncated:
            raise EpisodeOver()
        return out

    def measure(self, coupler: str, node: str, sweeps: dict, shots: Optional[int] = None) -> Optional[xr.Dataset]:
        self.step({"coupler": coupler, "node": node, "sweeps": sweeps, "shots": int(shots or self.shots)})
        raw = self.env.last_raw_result
        if raw is None or raw.info.get("invalid"):
            self.log.events.append({"rejected": node, "coupler": coupler, "why": raw.info.get("invalid") if raw else None})
            return None
        return raw.dataset

    def commit(self, coupler: str, param: str, value: float) -> bool:
        self.step({"coupler": coupler, "commit": {"param": param, "value": float(value)}})
        raw = self.env.last_raw_result
        return bool(raw is not None and not raw.info.get("invalid"))

    def benchmark(self, coupler: str) -> Optional[float]:
        self.step({"coupler": coupler, "benchmark": True})
        raw = self.env.last_raw_result
        value = None if raw is None else raw.info.get("F_lcb")
        self.log.f_lcb[coupler] = value
        estimate = float(value) if value else None           # only an identifiable benchmark is evidence
        i = self.env._label_index(coupler)
        if estimate is not None and i is not None and self.inner.configs[i].is_complete():
            if i not in self.best or estimate > self.best[i][0]:
                self.best[i] = (estimate, self.inner.configs[i].as_dict())
        return value

    def labels(self, i: int) -> tuple[str, str, str]:
        c = self.env._ctx_raw["couplers"][i]
        return c["label"], c["control"], c["target"]

    def qoi(self, key: str, qubit: str) -> Optional[float]:
        return (self.env._ctx_raw.get(key) or {}).get(qubit)

    # -- stages ------------------------------------------------------------------
    def discover(self, observation) -> None:
        ScriptedDiscovery(self.step).run(observation)
        self.log.discovery_ok = True

    def calibrate_readout(self) -> None:
        for i in range(self.inner.n_couplers):
            label, ctrl, tgt = self.labels(i)
            if ctrl in self.confusion and tgt in self.confusion:
                continue
            ds = self.measure(label, "readout_calibration", {}, shots=1024)
            if ds is not None:
                for q in (ctrl, tgt):
                    self.confusion.setdefault(q, confusion_from_readout(ds, f"y_{q}"))

    def scout_bias(self, i: int) -> list[float]:
        """Bias candidates for coupler ``i``: the ridge pick first, then a visible lattice around it."""
        label, ctrl, _ = self.labels(i)
        f01 = self.qoi("clock_freqs:f01", ctrl)
        profile = self.env._env.measurement_profile
        n_bias, span, n_freq = self.scout
        lo, hi = LIMITS["dc_current"]
        bias_span = min(hi - lo, profile.spectroscopy_max_bias_span_a if profile else hi - lo) * 0.98
        n_bias = min(n_bias, profile.spectroscopy_max_bias_points if profile else n_bias)
        span = min(span, profile.spectroscopy_max_span_hz if profile else span)
        ranked: list[float] = []
        if f01 is not None:
            ds = self.measure(label, "coupler_dc_spectroscopy", {
                "dc_currents": [0.0, bias_span, n_bias],
                # the line is pulled toward lower frequency as the coupler approaches it from above
                "qubit_frequencies": [float(f01) - 0.4 * span, span, n_freq]}, shots=max(self.shots, 256))
            if ds is not None:
                ranked = ranked_biases(ds, f"y_{ctrl}", k=self.max_biases)
        first = ranked[0] if ranked else 0.0
        out = ranked + [first + s * d for d in BIAS_LATTICE_A[1:3] for s in (1, -1)]
        return [float(np.clip(b, lo, hi)) for b in out][: self.max_biases]

    def parametrize(self, i: int) -> Optional[dict]:
        label, ctrl, tgt = self.labels(i)
        f01_t, f12_c = self.qoi("clock_freqs:f01", tgt), self.qoi("clock_freqs:f12", ctrl)
        if f01_t is None or f12_c is None:
            return None
        center = abs(f01_t - f12_c)
        f_span, f_n, a_c, a_span, a_n = self.param
        # The bare detuning misses the dressed resonance at some biases, and a weak lever arm needs ~1 us
        # pulses: if the standard probes find no transfer, retry once over the adjacent windows (the profile
        # caps one window's span), with a wider amplitude range and a longer pulse.
        wide = (0.5, 0.9, a_n)
        passes = [(center, (a_c, a_span, a_n), self.param_durations)] + [
            (center + offset, wide, (self.param_durations[0], 1e-6)) for offset in (0.0, -f_span, f_span)]
        for f_center, amps, durations in passes:
            for duration in durations:
                ds = self.measure(label, "cz_parametrization", {
                    "cz_pulse_frequencies": [f_center, f_span, f_n],
                    "cz_pulse_amplitudes": list(amps),
                    "cz_pulse_duration": float(duration)}, shots=100)
                if ds is None:
                    continue
                blob = detect_blob(ds, f"y_{ctrl}", threshold=self.blob_threshold)
                if blob is not None:
                    blob["param_duration"] = float(duration)
                    return blob
        return None

    def chevron_and_commit(self, i: int, blob: dict, window: Optional[tuple[float, float]] = None) -> Optional[dict]:
        """Chevron at the blob amplitude, over durations up to ~3x the parametrization probe (whose
        strongest transfer marks half a |11> <-> |20> cycle, so the CZ return sits near twice it), or over
        ``window``. Commits amplitude, frequency and the first-return duration; returns that point."""
        label, ctrl, _ = self.labels(i)
        f_span, f_n, _ = self.chevron
        grid = self.timing_grid
        t_lo, t_hi = window or (40e-9, min(3.0 * blob["param_duration"] + 100e-9, LIMITS["duration"][1]))
        t_lo, t_hi = max(t_lo, 20e-9), min(t_hi, LIMITS["duration"][1])
        n = int(np.clip((t_hi - t_lo) / (2 * grid), 30, 120))
        step = max(grid, round((t_hi - t_lo) / (n - 1) / grid) * grid)
        n = int((t_hi - t_lo) // step) + 1
        durations = [t_lo + 0.5 * (n - 1) * step, (n - 1) * step, n]
        ds = self.measure(label, "cz_chevron", {
            "cz_pulse_frequencies": [blob["frequency"], f_span, f_n],
            "cz_pulse_durations": durations, "cz_pulse_amplitude": blob["amplitude"]})
        if ds is None:
            return None
        cand = first_return(ds, f"y_{ctrl}")
        if cand is None:
            return None
        ok = (self.commit(label, "cz_pulse_amplitude", blob["amplitude"])
              and self.commit(label, "cz_pulse_frequency", cand["frequency"])
              and self.commit(label, "cz_pulse_duration", cand["duration"]))
        return cand if ok else None

    @property
    def timing_grid(self) -> float:
        profile = self.env._env.measurement_profile
        return profile.timing_grid_s if profile is not None else 4e-9

    def conditional_phase(self, i: int) -> None:
        label, _, tgt = self.labels(i)
        k = self.inner.configs[i]
        grid = self.env._env.measurement_profile.timing_grid_s if self.env._env.measurement_profile else 4e-9
        wps = [[k.cz_pulse_frequency + df, float(np.clip(round((k.cz_pulse_duration + dt) / grid) * grid, *LIMITS["duration"]))]
               for df in (-0.6e6, 0.0, 0.6e6) for dt in (-12e-9, 0.0, 12e-9)]
        ds = self.measure(label, "cz_calibration", {"working_points": wps})
        if ds is None:
            return
        best = min(calibration_phase_shifts(ds, f"y_{tgt}"), key=lambda s: abs(abs(s["shift"]) - np.pi))
        self.commit(label, "cz_pulse_frequency", best["frequency"])
        self.commit(label, "cz_pulse_duration", best["duration"])

    def local_phases(self, i: int) -> None:
        label, ctrl, tgt = self.labels(i)
        ds = self.measure(label, "cz_local_phases", {})
        if ds is None:
            return
        phi_c, phi_t = local_phase_estimates(ds, f"y_{ctrl}", f"y_{tgt}")
        self.commit(label, "control_local_phase", phi_c % (2 * np.pi))
        self.commit(label, "target_local_phase", phi_t % (2 * np.pi))

    def finish_gate(self, i: int) -> bool:
        self.conditional_phase(i)
        self.local_phases(i)
        return self.inner.configs[i].is_complete()

    def tune_gate(self, i: int, amplitude_steps: tuple = (1.0, 0.5, 0.25)) -> Optional[float]:
        """Gate parameters at the coupler's current bias. A strong parametric drive is fast but leaks and
        drives other sidebands, so after the first gate the amplitude is walked down (each step slower:
        the chevron window is rescaled with 1/amplitude) and every candidate is benchmarked; the best
        measured gate is restored. Returns its benchmark estimate, or None if no gate was built."""
        label = self.labels(i)[0]
        blob = self.parametrize(i)
        if blob is None:
            return None
        best, previous = None, None
        for scale in amplitude_steps:
            trial = dict(blob, amplitude=blob["amplitude"] * scale)
            window = None
            if previous is not None:
                expected = previous["duration"] * previous["amplitude"] / trial["amplitude"]
                window = (0.5 * expected, 1.8 * expected)
                if window[0] > LIMITS["duration"][1]:
                    break
            cand = self.chevron_and_commit(i, trial, window)
            if cand is None or not self.finish_gate(i):
                break
            estimate = self.benchmark_estimate(label)
            previous = dict(cand, amplitude=trial["amplitude"])
            if estimate is not None and (best is None or estimate > best[0]):
                best = (estimate, self.inner.configs[i].as_dict())
            elif best is not None and estimate is not None and estimate < best[0] - 0.02:
                break                                    # clearly worse: stop walking down
        if best is None:
            return None
        if self.inner.configs[i].as_dict() != best[1]:
            self.restore(i, best[1])
            self.benchmark(label)
        return best[0]

    def benchmark_estimate(self, label: str) -> Optional[float]:
        """Benchmark and return its certified estimate F_lcb, or None when the RB was not identifiable."""
        lcb = self.benchmark(label)
        return float(lcb) if lcb else None

    def restore(self, i: int, config: dict, *, guard: bool = True) -> None:
        label = self.labels(i)[0]
        current = self.inner.configs[i].as_dict()
        params = ("parking_current", "cz_pulse_frequency", "cz_pulse_amplitude", "cz_pulse_duration",
                  "control_local_phase", "target_local_phase")
        if all(current.get(p) == config.get(p) for p in params):
            return
        for param in params:
            if config.get(param) is not None:
                self.step({"coupler": label, "commit": {"param": param, "value": float(config[param])}}, guard=guard)

    def restore_best(self) -> None:
        """Commit every coupler's best measured configuration (unguarded: this is the reserve)."""
        for i in range(self.inner.n_couplers):
            if i in self.best and not self.env._terminated:
                self.restore(i, self.best[i][1], guard=False)

    def calibrate_coupler(self, i: int, *, scout: bool = True) -> Optional[float]:
        label = self.labels(i)[0]
        candidates = self.scout_bias(i) if scout else [self.inner.configs[i].parking_current]
        best = None
        for bias in candidates:
            if bias is None:
                continue
            if bias != self.inner.configs[i].parking_current and not self.commit(label, "parking_current", bias):
                continue
            estimate = self.tune_gate(i)
            if estimate is None:
                continue
            if best is None or estimate > best[0]:
                best = (estimate, self.inner.configs[i].as_dict())
            if estimate >= self.inner.f_star:
                break
        if best is not None and self.inner.configs[i].as_dict() != best[1]:
            self.restore(i, best[1])
            self.benchmark(label)
        return None if best is None else best[0]

    # -- recovery ----------------------------------------------------------------
    def _lcb(self, j: int) -> Optional[float]:
        b = self.inner.benchmarks[j]
        return None if b is None else b.f_lcb

    def reconcile_stale(self) -> None:
        for j in range(self.inner.n_couplers):
            if self.inner.benchmarks[j] is None and self.inner.configs[j].is_complete():
                self.benchmark(self.labels(j)[0])

    def failing(self) -> list[int]:
        out = []
        for j in range(self.inner.n_couplers):
            lcb = self._lcb(j)
            if (lcb is None and not self.inner.configs[j].is_complete()) or (lcb is not None and lcb < self.inner.f_star):
                out.append(j)
        return out

    def nudge_bias(self, i: int) -> None:
        """Move coupler ``i``'s bias one step on a fixed visible lattice; reverse direction when the
        last move did not improve its benchmark."""
        label = self.labels(i)[0]
        cur = float(self.inner.configs[i].parking_current or 0.0)
        state = self._bias_search.get(i, {"direction": 1, "tried": [cur], "last": None})
        lcb = self._lcb(i)
        improved = lcb is not None and state["last"] is not None and lcb > state["last"] + 1e-4
        direction = state["direction"] if (improved or state["last"] is None) else -state["direction"]
        lo, hi = LIMITS["dc_current"]
        nudged = cur
        for offset in [direction * d for d in BIAS_LATTICE_A] + [-direction * d for d in BIAS_LATTICE_A]:
            cand = float(np.clip(cur + offset, lo, hi))
            if not any(math.isclose(cand, t, abs_tol=1e-12) for t in state["tried"]):
                nudged = cand
                break
        state.update(direction=1 if nudged > cur else -1, last=lcb)
        state["tried"].append(nudged)
        self._bias_search[i] = state
        self.commit(label, "parking_current", nudged)
        self.tune_gate(i)

    # -- whole episode -------------------------------------------------------------
    def run(self, observation):
        try:
            if (observation.context.get("discovery") or {}).get("ready") is False:
                self.discover(observation)
            self.calibrate_readout()
            for i in range(self.inner.n_couplers):
                self.calibrate_coupler(i)
            self.reconcile_stale()
            passes = 0
            while True:
                todo = self.failing()
                incomplete = any(not self.inner.configs[j].is_complete() for j in todo)
                # Keep searching the bias lattice while a gate is still missing (BudgetLow ends it).
                if not todo or (passes >= self.max_passes and not incomplete):
                    break
                passes += 1
                for j in todo:
                    self.tune_gate(j)
                self.reconcile_stale()
                todo = self.failing()
                if todo:
                    weakest = min(todo, key=lambda j: -1.0 if self._lcb(j) is None else self._lcb(j))
                    self.nudge_bias(weakest)
                    self.reconcile_stale()
            self.restore_best()
            self.step({"stop": True}, guard=False)
        except BudgetLow:
            try:
                self.restore_best()
                if not self.env._terminated:
                    self.step({"stop": True}, guard=False)
            except EpisodeOver:
                pass
            self.log.events.append({"ended": "BudgetLow"})
        except (EpisodeOver, DiscoveryFailed) as exc:
            self.log.events.append({"ended": type(exc).__name__, "detail": str(exc)})
            if not self.env._terminated:
                try:
                    self.env.step({"stop": True})
                except Exception:
                    pass
        return self.env.score()


def run_scripted_episode(tier: str, seed: int, **agent_kwargs) -> dict:
    """Score of the scripted baseline on chip ``seed`` of ``tier``."""
    from chipcal import tiers
    env = tiers.make_env(tier)
    observation = env.reset(seed=int(seed), chain=tiers.sample(tier, seed).chain)
    agent = ScriptedChainAgent(env, **agent_kwargs)
    score = agent.run(observation)
    return {**score, "events": agent.log.events, "f_lcb_seen": agent.log.f_lcb}
