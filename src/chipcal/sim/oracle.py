from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import xarray as xr
from chipcal.sim.params import DeviceParams

SQ_GATE_TIME = 40e-9


@dataclass(frozen=True)
class IRBProtocol:
    lengths: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128)
    n_seeds: int = 7
    shots: int = 500
    seed_jitter: float = 0.0015  # per-seed multiplicative spread of the decay


STANDARD_IRB = IRBProtocol()


@dataclass
class IRBResult:
    f_hat: float
    f_sigma: float
    f_lcb: float
    p_ref: float
    p_int: float
    dataset: xr.Dataset = field(repr=False)


def _single_qubit_error(theta: DeviceParams) -> float:
    """Coherence-limited average error of one single-qubit gate, t/(6 T1) + t/(3 T2), averaged over both qubits."""
    per_qubit = [1.0 / (6.0 * t1) + 1.0 / (3.0 * t2) for t1, t2 in ((theta.t1_c, theta.t2_c), (theta.t1_t, theta.t2_t))]
    return float(SQ_GATE_TIME * np.mean(per_qubit))


def _spam_visibility(theta: DeviceParams) -> float:
    """SPAM-limited RB amplitude from the readout confusion diagonals."""
    diag = 0.5 * (theta.confusion_c[:2, :2].diagonal().mean() + theta.confusion_t[:2, :2].diagonal().mean())
    return float(np.clip(2.0 * diag - 1.0, 0.1, 1.0))


def rho(f: float) -> float:
    """Fidelity in "nines", -log10(1 - f), capped at 4."""
    return float(np.clip(-np.log10(max(1.0 - f, 1e-12)), 0.0, 4.0))
