"""Finite-bandwidth first-light spectroscopy; policy state contains no truth frequencies.

An effective noisy response model, not a pulse compiler or a resonator Hamiltonian.
RF = LO + IF. The default profile permits a 500 MHz total scan span and signed IF in
[-250, 250] MHz. These are explicit profile limits, not claims about any instrument.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

NODES = {"resonator_spectroscopy": "readout", "qubit_01_spectroscopy": "f01",
         "qubit_12_spectroscopy": "f12"}
PROMPT = """DISCOVERY (when context contains discovery). Before any coupler action, find each
qubit's readout resonator, f01 and f12 and commit them.
Nodes: resonator_spectroscopy measures |S21| transmission of the addressed qubit's readout feedline;
each resonator on that feedline appears as a dip, and dips carry no qubit label. qubit_01_spectroscopy
and qubit_12_spectroscopy apply a drive tone on the addressed qubit's drive line, read out through
that qubit's committed resonator, and return the excited-state response versus drive frequency for
a fixed ladder of drive amplitudes; the analysis reports the weakest amplitude that resolved a line.
qubit_12_spectroscopy first prepares |1> with a pulse at the committed f01.
Order per qubit: commit the resonator before that qubit's qubit_01_spectroscopy, and f01 before its
qubit_12_spectroscopy. Qubits are otherwise independent of each other. Committing a frequency
unsets that qubit's downstream discovery commits and invalidates coupler benchmarks.
Design frequencies in DISCOVERY STATE are uncertain estimates, not calibrated values.
Local oscillators: one readout LO per feedline, one drive LO per qubit; RF = LO + IF. A feedline
carries at most 5 qubits. A chip with one feedline has the readout LO "readout"; a larger chip has
"readout:fl0", "readout:fl1", ..., with their qubits listed in DISCOVERY STATE readout_feedlines.
Both scan endpoints must lie inside the addressed channel's reachable_rf_hz (boundary included),
the span must not exceed max_span_from_current_lo_hz, the f01 preparation pulse for f12 must lie
inside the drive LO band, and the committed resonator inside the readout LO band. LOs start at
design-derived defaults. Changing an LO does not change committed RF frequencies.
A commit_frequency must match a measured peak or fitted centre of a scan of that transition with
point spacing <= 2 MHz.
Syntax. Set an LO: {"configure_lo":{"channel":"readout" (or "readout:fl1") or "drive:q0","frequency_hz":number}}.
Scan: {"qubit":"q0","node":name,"sweeps":{"frequencies":[center_Hz,full_span_Hz,n]},"shots":256}.
Commit: {"qubit":"q0","commit_frequency":{"transition":"readout" or "f01" or "f12","frequency_hz":number}}.
DISCOVERY STATE alone governs these three scan nodes (reachable_rf_hz,
max_span_from_current_lo_hz, shots in [64,8192]); the coupler MEASUREMENT LIMITS and coupler
grammar limits do not apply to them.
"""


@dataclass(frozen=True)
class DiscoverySettings:
    design_sigma_hz: float = 100e6
    design_bound_hz: float = 300e6
    max_span_hz: float = 500e6
    max_if_hz: float = 250e6
    readout_width_hz: float = 2e6
    drive_width_hz: float = 4e6
    # Readout is a notch on a shared feedline: |S21| sits near its off-resonant
    # baseline and drops at each resonator.
    s21_baseline: float = 1.
    resonator_depth: float = .65
    # Two-tone spectroscopy sweeps a fixed drive-amplitude ladder, as calibration-tool
    # nodes commonly do: the agent chooses the window, the node chooses the power.
    spec_amplitude_min: float = 1e-3
    spec_amplitude_max: float = 8e-3
    spec_amplitude_points: int = 5
    # The 1-2 line answers weakly to a tone configured for 0-1, so its ladder is
    # stronger and shorter; scale times efficiency keeps both ladders in the same
    # saturation range.
    spec_amplitude_12_scale: float = 3.5
    spec_amplitude_12_points: int = 3
    drive_efficiency_12: float = .25
    saturation_amplitude: float = 2.5e-3
    max_response: float = .8
    stark_shift_hz: float = .4e6

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("discovery settings must be finite and positive")
        if self.resonator_depth >= 1:
            raise ValueError("a notch cannot remove more than the transmitted power")

    def spec_amplitudes(self, transition):
        """The drive-amplitude ladder one two-tone scan sweeps, weakest first."""
        if transition == "f12":
            return self.spec_amplitude_12_scale*np.linspace(
                self.spec_amplitude_min, self.spec_amplitude_max, self.spec_amplitude_12_points)
        return np.linspace(self.spec_amplitude_min, self.spec_amplitude_max, self.spec_amplitude_points)


QUBITS_PER_FEEDLINE = 5


def feedlines(n_qubits, per_line=QUBITS_PER_FEEDLINE):
    """Qubit indices of each readout feedline: consecutive groups of at most ``per_line``, as even as possible."""
    return [[int(i) for i in group] for group in np.array_split(np.arange(n_qubits), -(-n_qubits // per_line))]


def is_action(action):
    return isinstance(action, dict) and (bool({"configure_lo", "commit_frequency"} & action.keys())
                                         or (isinstance(action.get("node"), str) and action["node"] in NODES))


def action_schemas():
    number = {"type": "number"}
    qubit = {"type": "string", "pattern": "^q[0-9]+$"}
    def obj(props):
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    return [obj({"configure_lo": obj({"channel": {"type": "string"}, "frequency_hz": number})}),
            obj({"qubit": qubit, "commit_frequency": obj({"transition": {"enum": list(NODES.values())}, "frequency_hz": number})}),
            obj({"qubit": qubit, "node": {"enum": list(NODES)},
                 "sweeps": obj({"frequencies": {"type": "array", "items": number, "minItems": 3, "maxItems": 3}}),
                 "shots": {"type": "integer", "minimum": 64, "maximum": 8192}})]


def baseline(values, dip):
    """The off-resonant level of one trace.

    A quartile, not the median, estimates it: a refinement window can be narrower
    than the line it refines, and there the median sits on the line itself.
    """
    return float(np.percentile(values, 75 if dip else 25))


def response(values, dip):
    """The positive deviation from that level: notch depth, or driven response."""
    level = baseline(values, dip)
    return level-values if dip else values-level


def lorentzian(x, frequency, width, height, floor):
    return floor+height/(1+((x-frequency)/width)**2)


def fit_line(x, deviation, index):
    """Least-squares Lorentzian around one resolved peak (the node's analysis fit).

    The largest sample is only as accurate as the sweep spacing, and a
    power-broadened line is flat enough at the top for noise to move that sample by
    several points. The shoulders carry the centre, so the fit is what gets committed.
    """
    spacing = float(np.max(np.diff(x)))
    height = float(deviation[index])
    # Half maximum of the contiguous run around this peak. A feedline sweep holds
    # several resolved lines, so every sample above half maximum is not one line.
    below = np.flatnonzero(deviation < .5*height)
    left = below[below < index]
    right = below[below > index]
    guess = max(float(x[right[0] if right.size else -1]-x[left[-1] if left.size else 0])/2, spacing)
    window = np.abs(x-x[index]) <= 3*guess
    if int(window.sum()) < 5:
        return None
    try:
        (frequency, width, height, floor), _ = curve_fit(
            lorentzian, x[window], deviation[window], p0=(x[index], guess, height, 0.),
            bounds=((x[0], spacing/4, height/4, -height), (x[-1], 4*guess, 4*height, height)),
            maxfev=4000)
    except (RuntimeError, ValueError):
        return None
    if abs(frequency-x[index]) > 2*width:
        return None
    return {"frequency_hz": float(frequency), "width_hz": float(width),
            "height": float(height), "floor": float(floor)}


def peak_candidates(x, deviation, sigma):
    """Resolved peaks of one trace, measured against observed noise only."""
    peaks, _ = find_peaks(deviation, prominence=max(.08, 6*sigma), distance=2)
    return [{"frequency_hz": float(x[k]), "contrast": float(deviation[k])}
            for k in peaks if deviation[k] > max(.10, 6*sigma)]


def analyze(ds):
    """Reduce a scan to measured peaks plus the single trace a chart or an agent reads."""
    x = ds.frequency.values
    y = ds.signal.values
    sigma = float(ds.attrs["noise_sigma"])
    dip = ds.attrs["node"] == "resonator_spectroscopy"
    extra = {}
    if y.ndim == 1:
        deviation = response(y, dip)
        note = ("Readout candidates have no qubit identity; validate with addressed drive spectroscopy."
                if dip else "Candidates are drive lines of this readout assignment, not necessarily of this qubit.")
    else:
        amplitudes = ds.drive_amplitude.values
        deviations = [response(row, dip) for row in y]
        heights = [float(np.max(d)) for d in deviations]
        # Read the weakest tone that still answers strongly: the louder rows are
        # power broadened and ac-Stark shifted away from the bare frequency.
        index = next(i for i, h in enumerate(heights) if h >= .5*max(heights))
        y, deviation = y[index], deviations[index]
        note = ("Trace is the weakest amplitude that resolved the line; louder rows broaden and shift it.")
        extra = {"drive_amplitude": float(amplitudes[index]),
                 "drive_amplitudes": [float(a) for a in amplitudes]}
    candidates = peak_candidates(x, deviation, sigma)
    strongest = max(candidates, key=lambda c: c["contrast"], default=None)
    fit = None if strongest is None else fit_line(
        x, deviation, int(np.argmin(np.abs(x-strongest["frequency_hz"]))))
    return {"node": ds.attrs["node"], "qubit": ds.attrs["qubit"],
            "status": "CANDIDATES" if candidates else "NO_RESOLVED_PEAK",
            "candidates": candidates, "fit": fit, "resolution_hz": float(np.max(np.diff(x))),
            "observable": "|S21| transmission, dipping at a resonator" if dip
                          else "excited-state response versus drive frequency and amplitude",
            **extra,
            "frequency_hz": x.tolist(), "signal": y.tolist(), "note": note}


class OneQubitDiscovery:
    def __init__(self, chain, rng, settings):
        self.settings = settings
        self.rng = rng
        self.labels = [f"q{i}" for i in range(chain.n_qubits)]
        n = len(self.labels)
        # Each feedline is a multiplexed bus with its own readout LO and the full readout band.
        groups = feedlines(n)
        names = ["readout"] if len(groups) == 1 else [f"readout:fl{k}" for k in range(len(groups))]
        self.feedline = {self.labels[i]: name for name, group in zip(names, groups) for i in group}
        # Random physical ordering and bounded fabrication shifts on each bus.
        ro_design = np.empty(n)
        for group in groups:
            ro_design[group] = rng.permutation(np.linspace(6.5e9, 7.3e9, len(group)))
        ro_true = ro_design + self._offset(n)
        # Avoid unresolved resonator collisions on a feedline without making assignment follow order.
        for _ in range(1000):
            if all(len(g) < 2 or np.min(np.diff(np.sort(ro_true[g]))) > 18e6 for g in groups):
                break
            ro_true = ro_design + self._offset(n)
        else:
            raise ValueError("cannot place resolvable resonators")
        f01 = np.array([q.f01 for q in chain.qubits])
        f12 = np.array([q.f01 + q.alpha for q in chain.qubits])
        f01_design = f01 + self._offset(n)
        f12_design = f01_design + (f12-f01) + rng.normal(0, 25e6, n)
        self._truth = {q: dict(readout=ro_true[i], f01=f01[i], f12=f12[i]) for i, q in enumerate(self.labels)}
        self.design = {q: dict(readout=float(ro_design[i]), f01=float(f01_design[i]), f12=float(f12_design[i])) for i,q in enumerate(self.labels)}
        # Each LO starts at the centre of its channel's design prior: a feedline's readout LO
        # at the mean of its design slots, each drive LO on that qubit's own design f01.
        self.lo = {**{name: float(np.mean(ro_design[g])) for name, g in zip(names, groups)},
                   **{f"drive:{q}": float(f01_design[i]) for i,q in enumerate(self.labels)}}
        self.feedlines = {name: [self.labels[i] for i in g] for name, g in zip(names, groups)}
        self.committed = {q: {} for q in self.labels}
        self.evidence = {}

    def _offset(self, n):
        s = self.settings
        # Truncated Gaussian; rejection avoids artificial mass at the bounds.
        values = self.rng.normal(0, s.design_sigma_hz, n)
        while np.any(np.abs(values) > s.design_bound_hz):
            mask = np.abs(values) > s.design_bound_hz
            values[mask] = self.rng.normal(0, s.design_sigma_hz, int(mask.sum()))
        return values

    @property
    def ready(self):
        return all(len(v) == 3 for v in self.committed.values())

    def accuracy(self, tolerance=1e6):
        """Privileged evaluation: fraction of transitions committed within tolerance [Hz]."""
        pairs = [(self.committed[q].get(k), v) for q, truth in self._truth.items() for k, v in truth.items()]
        return sum(c is not None and abs(c - v) <= tolerance for c, v in pairs)/max(1, len(pairs))

    def visible(self):
        """Policy-visible discovery state, with the band arithmetic already done.

        A scan is legal when both endpoints fall inside this channel's reachable_rf_hz, so
        that interval and the span it admits are published rather than left to be derived
        from the LO and the signed IF limit.
        """
        s = self.settings
        span = min(s.max_span_hz, 2*s.max_if_hz)
        buses = dict(readout_bus=self.labels) if len(self.feedlines) == 1 else dict(readout_feedlines=self.feedlines)
        return deepcopy(dict(design_frequencies_hz=self.design, lo_frequencies_hz=self.lo,
                    committed_frequencies_hz=self.committed, ready=self.ready,
                    max_span_hz=s.max_span_hz, signed_if_limits_hz=[-s.max_if_hz, s.max_if_hz],
                    design_uncertainty_hz=s.design_bound_hz,
                    reachable_rf_hz={channel: [lo-s.max_if_hz, lo+s.max_if_hz] for channel, lo in self.lo.items()},
                    max_span_from_current_lo_hz=span,
                    if_band_endpoints_inclusive=True,
                    lo_positions_to_cover_design_prior=int(np.ceil(2*s.design_bound_hz/span)),
                    **buses))

    @staticmethod
    def _frequency(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 2e9 <= value <= 9e9:
            raise ValueError("frequency_hz must be finite and in [2, 9] GHz")
        return float(value)

    def _band(self, channel, frequencies):
        values = np.asarray(frequencies, dtype=float)
        lo = self.lo[channel]
        worst = float(values[int(np.argmax(np.abs(values-lo)))])
        # The boundary is included to 1 Hz: well above float rounding (about 1e-6 Hz at 5 GHz), which differs
        # between CPUs and maths libraries and would otherwise decide a scan that ends exactly on the band edge.
        if abs(worst-lo) > self.settings.max_if_hz + 1.:
            raise ValueError(
                f"RF {worst:.6g} Hz is outside the {channel} LO signed IF band "
                f"[{lo-self.settings.max_if_hz:.6g}, {lo+self.settings.max_if_hz:.6g}] Hz "
                f"(boundary included; LO {lo:.6g} Hz, IF {worst-lo:+.6g} Hz, limit "
                f"+/-{self.settings.max_if_hz:.6g} Hz); retune configure_lo or narrow/move the scan")

    def execute(self, action):
        """Validate before mutation; return dataset, measured info, and simulation cost."""
        if "configure_lo" in action:
            if set(action) != {"configure_lo"} or set(action["configure_lo"]) != {"channel", "frequency_hz"}:
                raise ValueError("configure_lo must be the only action")
            value = action["configure_lo"]
            channel = value["channel"]
            if channel not in self.lo:
                raise ValueError(f"LO channel must be one of {list(self.lo)}")
            self.lo[channel] = self._frequency(value["frequency_hz"])
            return None, {"configured_lo": deepcopy(value)}, 2.
        q = action.get("qubit")
        if q not in self.labels:
            raise ValueError("discovery needs a valid qubit label")
        if "commit_frequency" in action:
            if set(action) != {"qubit", "commit_frequency"} or set(action["commit_frequency"]) != {"transition", "frequency_hz"}:
                raise ValueError("commit_frequency must be the only action")
            value = action["commit_frequency"]
            transition = value["transition"]
            f = self._frequency(value["frequency_hz"])
            evidence = self.evidence.get((q, transition))
            measured = [p["frequency_hz"] for p in evidence["candidates"]] if evidence else []
            if evidence and evidence.get("fit"):
                measured.append(evidence["fit"]["frequency_hz"])
            if evidence is None or evidence["resolution_hz"] > 2e6 or not any(abs(m-f) <= evidence["resolution_hz"] for m in measured):
                raise ValueError("commit needs a measured peak or its fitted centre from a scan with <=2 MHz spacing; refine the peak first")
            changed = self.committed[q].get(transition) != f
            self.committed[q][transition] = f
            if changed:
                downstream = {"readout": ["f01", "f12"], "f01": ["f12"], "f12": []}[transition]
                for key in downstream:
                    self.committed[q].pop(key, None)
                    self.evidence.pop((q, key), None)
            return None, {"frequency_committed": {"qubit": q, **value}, "frequency_changed": changed}, 1.
        if set(action) != {"qubit", "node", "sweeps", "shots"} or set(action["sweeps"]) != {"frequencies"}:
            raise ValueError("scan requires qubit, node, sweeps.frequencies, shots only")
        transition = NODES[action["node"]]
        triple = action["sweeps"]["frequencies"]
        if not isinstance(triple, list) or len(triple) != 3:
            raise ValueError("frequencies must be [center_Hz, full_span_Hz, point_count]")
        center, span, count = triple
        self._frequency(center)
        if isinstance(span, bool) or not isinstance(span, (int, float)) or not np.isfinite(span) or not 0 < span <= self.settings.max_span_hz:
            raise ValueError("scan span must be positive and <=500 MHz (profile limit)")
        if type(count) is not int or not 5 <= count <= 1001:
            raise ValueError("point_count must be integer in [5,1001]")
        shots = action["shots"]
        if type(shots) is not int or not 64 <= shots <= 8192:
            raise ValueError("shots must be integer in [64,8192]")
        x = np.linspace(center-span/2, center+span/2, count)
        self._frequency(float(x[0])); self._frequency(float(x[-1]))
        self._band(self.feedline[q] if transition == "readout" else f"drive:{q}", x)
        known = self.committed[q]
        if transition != "readout":
            if "readout" not in known:
                raise ValueError("discover and commit the resonator before qubit spectroscopy")
            self._band(self.feedline[q], [known["readout"]])
        if transition == "f12":
            if "f01" not in known:
                raise ValueError("discover and commit f01 before preparing f12")
            self._band(f"drive:{q}", [known["f01"]])
        s = self.settings
        lorentz = lambda frequency, width: 1/(1+((x-frequency)/width)**2)
        sigma = .12/np.sqrt(shots)
        if transition == "readout":
            # Notch transmission: every resonator on the feedline multiplies its own
            # dip into |S21|. Complex IQ is unnecessary for this effective model.
            transmission = np.ones(count)
            for other, truth in self._truth.items():
                if self.feedline[other] == self.feedline[q]:
                    transmission *= 1-s.resonator_depth*lorentz(truth["readout"], s.readout_width_hz)
            signal = s.s21_baseline*transmission
            dims, coords, rows = ("frequency",), {"frequency": x}, 1
        else:
            amplitudes = s.spec_amplitudes(transition)
            efficiency = s.drive_efficiency_12 if transition == "f12" else 1.
            rows = int(amplitudes.size)
            signal = np.zeros((rows, count))
            for other, truth in self._truth.items():
                if self.feedline[other] != self.feedline[q]:
                    continue                      # read out through another feedline
                drive_weight = 1. if other == q else .10
                readout_weight = 1/(1+((known["readout"]-truth["readout"])/s.readout_width_hz)**2)
                preparation = 1. if transition == "f01" else 1/(1+((known["f01"]-truth["f01"])/s.drive_width_hz)**2)
                for row, amplitude in enumerate(amplitudes):
                    # Steady-state saturation of a driven line: a louder tone answers
                    # louder but power broadens and ac-Stark shifts it, so amplitude
                    # buys detection and costs accuracy.
                    saturation = (amplitude*efficiency*drive_weight/s.saturation_amplitude)**2
                    driven = saturation/(1+saturation)
                    signal[row] += (s.max_response*readout_weight*preparation*driven
                                    * lorentz(truth[transition]-s.stark_shift_hz*driven,
                                              s.drive_width_hz*np.sqrt(1+saturation)))
            dims, coords = ("drive_amplitude", "frequency"), {"drive_amplitude": amplitudes, "frequency": x}
        signal = signal+self.rng.normal(0, sigma, signal.shape)
        ds = xr.Dataset({"signal": (dims, signal)}, coords=coords,
                        attrs=dict(node=action["node"], qubit=q, shots=shots, noise_sigma=sigma, discovery=True))
        self.evidence[q, transition] = analyze(ds)
        return ds, {"node": action["node"], "qubit": q}, 3.+rows*count*shots*2e-5
