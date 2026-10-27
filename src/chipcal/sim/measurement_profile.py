"""Explicit experiment limits, separate from instrument specifications.

The caps below are configurable experiment policy (span, point counts, timing
grid, LO/NCO band, time costs), not universal instrument limits. No minimum point
count is imposed: small adaptive windows remain cheap.
"""

from dataclasses import dataclass

import numpy as np

from chipcal.core.actions import MeasureAction, axis_values


@dataclass(frozen=True)
class MeasurementProfile:
    name: str = "default_v1"
    spectroscopy_max_span_hz: float = 20e6
    spectroscopy_max_frequency_points: int = 201
    spectroscopy_max_bias_points: int = 101
    spectroscopy_max_bias_span_a: float = 5e-3
    # Set the LO of the addressed drive channel to check a concrete compiled band.
    qubit_lo_hz: float | None = None
    nco_limit_hz: float = 500e6
    timing_grid_s: float = 4e-9  # conservative schedule policy; not a DAC sample limit
    enforce_timing_grid: bool = True
    spectroscopy_reset_s: float = 200e-6
    spectroscopy_readout_s: float = 3e-6
    bias_ramp_a_per_s: float = 40e-6
    # Per-acquisition overheads are explicit estimates.
    spectroscopy_overhead_s: float = 15.0
    bias_settle_s: float = 0.5
    # Optional whole-workflow caps (None disables the span cap).
    gate_max_span_hz: float | None = None
    gate_max_axis_points: int = 201

    def __post_init__(self):
        values = [self.spectroscopy_max_span_hz, self.spectroscopy_max_frequency_points,
                  self.spectroscopy_max_bias_points, self.spectroscopy_max_bias_span_a,
                  self.nco_limit_hz, self.timing_grid_s]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("profile limits must be positive and finite")
        if self.qubit_lo_hz is not None and not np.isfinite(self.qubit_lo_hz):
            raise ValueError("qubit LO must be finite")
        if not np.isfinite([self.spectroscopy_reset_s, self.spectroscopy_readout_s,
                            self.bias_ramp_a_per_s, self.spectroscopy_overhead_s, self.bias_settle_s]).all():
            raise ValueError("timing cost parameters must be finite")
        if min(self.spectroscopy_reset_s, self.spectroscopy_readout_s, self.spectroscopy_overhead_s, self.bias_settle_s) < 0 or self.bias_ramp_a_per_s <= 0:
            raise ValueError("costs must be nonnegative and bias ramp rate positive")
        if self.gate_max_span_hz is not None and (not np.isfinite(self.gate_max_span_hz) or self.gate_max_span_hz <= 0):
            raise ValueError("gate span cap must be finite and positive")
        if not isinstance(self.gate_max_axis_points,int) or self.gate_max_axis_points < 1:
            raise ValueError("gate point cap must be a positive integer")

    def validate(self, action, default_spectroscopy_duration=4e-6):
        from chipcal.core.actions import CommitAction

        errors = []
        durations = []
        if isinstance(action, CommitAction) and action.param == "cz_pulse_duration":
            durations = [action.value]
        if isinstance(action, MeasureAction):
            sw = action.sweeps
            if self.gate_max_span_hz is not None:
                if "cz_pulse_frequencies" in sw and np.ptp(axis_values(sw["cz_pulse_frequencies"])) > self.gate_max_span_hz + 1e-6:
                    errors.append(f"{self.name}: CZ frequency span exceeds {self.gate_max_span_hz:g} Hz; recenter separate windows")
                for key in ("cz_pulse_frequencies", "cz_pulse_amplitudes", "cz_pulse_durations", "delays", "ramsey_phases", "local_phases"):
                    if key in sw and axis_values(sw[key]).size > self.gate_max_axis_points:
                        errors.append(f"{self.name}: {key} exceeds {self.gate_max_axis_points} points")
            if action.node in ("coupler_dc_spectroscopy", "qubit_spectroscopy"):
                if "spec_pulse_duration" not in sw:
                    durations.append(default_spectroscopy_duration)
                f = axis_values(sw["qubit_frequencies"])
                if np.ptp(f) > self.spectroscopy_max_span_hz + 1e-6:
                    errors.append(f"{self.name}: spectroscopy span exceeds {self.spectroscopy_max_span_hz:g} Hz; use separately centred windows")
                if f.size > self.spectroscopy_max_frequency_points:
                    errors.append(f"{self.name}: spectroscopy has too many frequency points")
                if self.qubit_lo_hz is not None and np.max(np.abs(f - self.qubit_lo_hz)) > self.nco_limit_hz:
                    errors.append(f"{self.name}: frequency outside configured LO +/- NCO band")
                if "dc_currents" in sw:
                    i = axis_values(sw["dc_currents"])
                    if i.size > self.spectroscopy_max_bias_points or np.ptp(i) > self.spectroscopy_max_bias_span_a + 1e-15:
                        errors.append(f"{self.name}: spectroscopy bias sweep exceeds configured point/span budget")
            for key in ("cz_pulse_duration", "cz_pulse_durations", "spec_pulse_duration", "delays"):
                if key in sw:
                    durations.extend(axis_values(sw[key]))
            if "working_points" in sw:
                durations.extend(float(x[1]) for x in sw["working_points"])
        if self.enforce_timing_grid and durations:
            units = np.asarray(durations) / self.timing_grid_s
            if not np.allclose(units, np.rint(units), rtol=0, atol=1e-7):
                errors.append(f"{self.name}: pulse durations must lie on the {self.timing_grid_s:g} s schedule grid")
        return errors

    def spectroscopy_cost(self, action, start_current, default_duration=4e-6):
        """Shot time plus traversed-current ramp time, including restoration.

        Follows an external DC loop and an inner frequency loop. Both qubits are
        read in one shot; no extra per-qubit shot multiplier is charged.
        """
        sw = action.sweeps
        start = 0.0 if start_current is None else float(start_current)
        currents = axis_values(sw.get('dc_currents', [start]))
        frequencies = axis_values(sw['qubit_frequencies'])
        duration = float(sw.get('spec_pulse_duration', default_duration))
        ramp = np.abs(np.diff(np.r_[start, currents, start])).sum()/self.bias_ramp_a_per_s
        settle = currents.size*self.bias_settle_s if 'dc_currents' in sw else 0
        return float(self.spectroscopy_overhead_s + ramp + settle + currents.size*frequencies.size*action.shots*(self.spectroscopy_reset_s + duration + self.spectroscopy_readout_s))

    def describe(self):
        """The limits as shown to the agent: values only, no provenance."""
        from dataclasses import asdict
        return {**asdict(self), "limits_kind": "configurable experiment policy"}


def default_profile():
    """The benchmark's measurement policy: the default caps plus a 40 MHz CZ-window span cap."""
    return MeasurementProfile(name="default_v1", gate_max_span_hz=40e6)


def spectroscopy_controls_errors(action, spectroscopy):
    if not isinstance(action, MeasureAction):
        return []
    controls = {"drive_qubit", "spec_pulse_duration", "spec_pulse_amplitude"} & action.sweeps.keys()
    if controls and spectroscopy is None:
        return ["spectroscopy pulse controls require SpectroscopySettings on the environment"]
    from chipcal.sim.multimode import MultimodeSettings
    if controls and action.sweeps.get("transition", 1) == 2 and not isinstance(spectroscopy, MultimodeSettings):
        return ["pulse controls for prepared-state f12 spectroscopy are not implemented"]
    return []
