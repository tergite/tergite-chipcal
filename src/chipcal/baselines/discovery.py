"""Scripted discovery: find every qubit's resonator, f01 and f12 from visible measurements only.

The routine is a conventional bring-up sequence:

1. Sweep each readout feedline in overlapping LO windows and collect every dip.
2. For each qubit, try the dips nearest its design readout frequency: refine the dip, commit it,
   then search for f01 in two wide windows around the design f01.
3. When a drive line answers, refine it and commit f01; search f12 around f01 plus the design
   anharmonicity, refine and commit it.

It reads only the observation (design frequencies, DISCOVERY STATE, the node analysis) and never
touches hidden device state.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

Step = Callable[[dict], Any]   # action -> StepOutcome


class DiscoveryFailed(RuntimeError):
    pass


def measured_frequency(analysis: dict) -> float:
    """The fitted line centre, or the strongest sample when no fit converged."""
    fit = analysis.get("fit")
    if fit:
        return float(fit["frequency_hz"])
    return float(max(analysis["candidates"], key=lambda p: p["contrast"])["frequency_hz"])


class ScriptedDiscovery:
    def __init__(self, step: Step, *, shots: int = 512, readout_windows: Optional[list[float]] = None,
                 readout_span: float = 450e6, f01_offsets: tuple[float, ...] = (-170e6, 170e6),
                 f01_span: float = 400e6):
        self._step = step
        self.shots = shots
        self.readout_windows = readout_windows
        self.readout_span = readout_span
        self.f01_offsets = f01_offsets
        self.f01_span = f01_span
        self.last = None

    def step(self, action: dict):
        out = self._step(action)
        self.last = out
        if out.terminated or out.truncated:
            raise DiscoveryFailed("episode ended during discovery")
        return out

    def max_if_hz(self) -> float:
        """The signed IF limit of the drive and readout LOs (from DISCOVERY STATE)."""
        state = self.last.observation.context.get("discovery") if self.last is not None else None
        limits = (state or {}).get("signed_if_limits_hz") or [-250e6, 250e6]
        return float(min(abs(limits[0]), abs(limits[1])))

    def lo(self, channel: str, frequency: float):
        return self.step({"configure_lo": {"channel": channel, "frequency_hz": float(frequency)}})

    def scan(self, qubit: str, node: str, center: float, span: float, count: int = 401):
        out = self.step({"qubit": qubit, "node": node,
                         "sweeps": {"frequencies": [float(center), float(span), int(count)]}, "shots": self.shots})
        if out.info.get("invalid"):
            return {}                     # a rejected scan measured nothing (do not read the stale analysis)
        return out.observation.last_analysis or {}

    def commit(self, qubit: str, transition: str, frequency: float):
        return self.step({"qubit": qubit, "commit_frequency": {"transition": transition, "frequency_hz": float(frequency)}})

    def _feedline_dips(self, design: dict, channel: str = "readout") -> list[float]:
        slots = [v["readout"] for v in design.values()]
        windows = self.readout_windows
        if windows is None:
            lo, hi = min(slots) - 300e6, max(slots) + 300e6
            step = self.readout_span * 0.9
            windows, center = [], lo + self.readout_span / 2
            while center - self.readout_span / 2 < hi:
                windows.append(center)
                center += step
        dips: list[float] = []
        for center in windows:
            self.lo(channel, center)
            analysis = self.scan(next(iter(design)), "resonator_spectroscopy", center, self.readout_span)
            for peak in analysis.get("candidates", []):
                if all(abs(peak["frequency_hz"] - x) > 5e6 for x in dips):
                    dips.append(peak["frequency_hz"])
        return dips

    def _qubit(self, qubit: str, values: dict, dips: list[float], channel: str = "readout") -> bool:
        for readout in sorted(dips, key=lambda p: abs(p - values["readout"])):
            self.lo(channel, readout)
            analysis = self.scan(qubit, "resonator_spectroscopy", readout, 12e6, 121)
            if not analysis.get("candidates"):
                continue
            self.commit(qubit, "readout", measured_frequency(analysis))
            for shift in self.f01_offsets:
                center = values["f01"] + shift
                self.lo(f"drive:{qubit}", center)
                analysis = self.scan(qubit, "qubit_01_spectroscopy", center, self.f01_span)
                if not analysis.get("candidates"):
                    continue
                peak = measured_frequency(analysis)
                self.lo(f"drive:{qubit}", peak)
                analysis = self.scan(qubit, "qubit_01_spectroscopy", peak, 20e6, 101)
                if not analysis.get("candidates"):
                    continue
                f01 = measured_frequency(analysis)
                self.commit(qubit, "f01", f01)
                f12_center = f01 + values["f12"] - values["f01"]
                # The design anharmonicity is uncertain by tens of MHz: search +-150 MHz around it, with the
                # drive LO placed so that the window and the f01 preparation tone both fit its IF band.
                band = self.max_if_hz() * 0.999
                lo = min(max(f12_center, f01 - band), f01 + band)          # as close to f12 as the f01 tone allows
                low, high = max(f12_center - 150e6, lo - band), min(f12_center + 150e6, lo + band)
                self.lo(f"drive:{qubit}", lo)
                analysis = self.scan(qubit, "qubit_12_spectroscopy", (low + high) / 2, high - low, int((high - low) / 1e6) + 1)
                if not analysis.get("candidates"):
                    continue
                analysis = self.scan(qubit, "qubit_12_spectroscopy", measured_frequency(analysis), 16e6, 81)
                if not analysis.get("candidates"):
                    continue
                self.commit(qubit, "f12", measured_frequency(analysis))
                return True
        return False

    def run(self, observation) -> Any:
        """Discover every qubit; return the last StepOutcome. Raises DiscoveryFailed."""
        state = observation.context.get("discovery") or {}
        if state.get("ready"):
            return None
        design = state["design_frequencies_hz"]
        lines = state.get("readout_feedlines") or {"readout": list(design)}
        for channel, qubits in lines.items():
            on_line = {q: design[q] for q in qubits}
            dips = self._feedline_dips(on_line, channel)
            for qubit, values in on_line.items():
                if not self._qubit(qubit, values, dips, channel):
                    raise DiscoveryFailed(f"no consistent resonator/f01/f12 found for {qubit}")
        return self.last
