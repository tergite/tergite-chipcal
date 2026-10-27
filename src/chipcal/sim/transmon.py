"""Exact transmon spectrum in the charge basis (Koch et al., PRA 76, 042319, 2007).

H = 4 E_C (n - n_g)^2 - E_J cos(phi), diagonalised in the charge basis n = -N..N. Used to
derive a qubit's anharmonicity from its f01 and E_J/E_C, and its charge-parity line splitting
(a parity switch moves the offset charge n_g by 1/2).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

_NCUT = 30


def _levels(ratio: float, offset_charge: float, n_levels: int = 4) -> np.ndarray:
    """Lowest energies of the transmon with E_C = 1 and E_J = ``ratio``."""
    n = np.arange(-_NCUT, _NCUT + 1)
    h = np.diag(4.0 * (n - offset_charge) ** 2) - 0.5 * ratio * (np.eye(n.size, k=1) + np.eye(n.size, k=-1))
    return np.linalg.eigvalsh(h)[:n_levels]


@lru_cache(maxsize=4096)
def _mean_transitions(ratio: float) -> tuple[float, float]:
    """(f01, f12) in units of E_C at n_g = 1/4, where the first-order charge dispersion vanishes."""
    e = _levels(ratio, 0.25)
    return float(e[1] - e[0]), float(e[2] - e[1])


def from_f01(f01_hz: float, ej_over_ec: float) -> dict:
    """E_C, E_J and the anharmonicity of a transmon with this f01 [Hz] and E_J/E_C."""
    e01, e12 = _mean_transitions(round(float(ej_over_ec), 9))
    ec = float(f01_hz) / e01
    return {"ec_hz": ec, "ej_hz": ec * float(ej_over_ec), "alpha_hz": ec * (e12 - e01), "ej_over_ec": float(ej_over_ec)}


def parity_splits_hz(ec_hz: float, ej_over_ec: float, offset_charge: float) -> tuple[float, float]:
    """(f01, f12) full line separation between the two charge-parity branches at offset charge n_g [Hz]."""
    a, b = _levels(float(ej_over_ec), float(offset_charge)), _levels(float(ej_over_ec), float(offset_charge) + 0.5)
    d01 = abs((a[1] - a[0]) - (b[1] - b[0]))
    d12 = abs((a[2] - a[1]) - (b[2] - b[1]))
    return float(d01 * ec_hz), float(d12 * ec_hz)


def max_parity_splits_hz(ec_hz: float, ej_over_ec: float) -> tuple[float, float]:
    """The largest splits over offset charge (at n_g = 0)."""
    return parity_splits_hz(ec_hz, ej_over_ec, 0.0)
