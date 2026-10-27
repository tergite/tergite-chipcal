"""Committed configuration k and the deterministic reset map g.

Field names follow the quantity names of the tergite-autocalibration tool. The reset
map encodes the calibration dependency graph: committing an upstream coordinate unsets
everything downstream, so backtracking is just another commit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

# Dependency depth of each committable coordinate.  Committing a coordinate
# unsets every coordinate at a strictly deeper level (and always invalidates a
# standing benchmark result).
COMMIT_LEVELS: dict[str, int] = {
    "parking_current": 0,
    "cz_pulse_frequency": 1,
    "cz_pulse_amplitude": 1,
    "cz_pulse_duration": 2,
    "control_local_phase": 3,
    "target_local_phase": 3,
}

COMMITTABLE = tuple(COMMIT_LEVELS)


@dataclass
class CommittedConfig:
    """k: the fully observed control configuration. None = not set.

    Local-phase convention, which every backend must implement identically: a
    committed local phase is an ESTIMATE OF THE TRUE ACCUMULATED PHASE of the flux
    pulse, not its negation. A backend's gate fidelity must therefore be maximal
    when the committed value EQUALS the phase the pulse really accumulates, which
    is what ``observation_analysis`` reports for ``cz_local_phases`` (a positive
    fringe shift). Getting this sign wrong does not fail loudly: it silently turns a
    correct measurement into a de-calibration, and it is invisible at working points
    whose true phases sit near 0 or pi.
    """

    parking_current: Optional[float] = None        # A
    cz_pulse_frequency: Optional[float] = None     # Hz
    cz_pulse_amplitude: Optional[float] = None     # normalized 0..1
    cz_pulse_duration: Optional[float] = None      # s
    control_local_phase: Optional[float] = None    # rad, estimate of the accumulated phase
    target_local_phase: Optional[float] = None     # rad, estimate of the accumulated phase

    def commit(self, param: str, value: float) -> "CommittedConfig":
        """The reset map g(k, commit(chi <- v)): set chi, unset deeper levels."""
        if param not in COMMIT_LEVELS:
            raise ValueError(f"unknown commit target {param!r}; expected one of {COMMITTABLE}")
        level = COMMIT_LEVELS[param]
        out = CommittedConfig(**{f.name: getattr(self, f.name) for f in fields(self)})
        setattr(out, param, float(value))
        for name, lvl in COMMIT_LEVELS.items():
            if lvl > level:
                setattr(out, name, None)
        return out

    def is_complete(self) -> bool:
        """Enough committed to define a CZ gate (phases default to 0)."""
        return all(
            getattr(self, name) is not None
            for name in ("parking_current", "cz_pulse_frequency", "cz_pulse_amplitude", "cz_pulse_duration")
        )

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
