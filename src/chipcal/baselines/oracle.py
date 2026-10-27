"""Privileged CZ ceiling: the best gate fidelity a coupler admits (evaluation only).

Not a baseline -- it reads the hidden device. It searches the CZ pulse frequency, amplitude and
duration (coarse grid, then Nelder-Mead) with the two local phases chosen optimally from the
channel, at a fixed bias (``coupler_ceiling``) or over the coupler's own bias as well
(``coupler_best_ceiling``), so the benchmark chips, tier targets and baselines can be compared
with what the physics allows.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from chipcal.core.actions import LIMITS
from chipcal.sim import chain as chainmod
from chipcal.sim.multimode import Cell
from chipcal.sim.multimode import noise_branches

PHASES = np.linspace(0, 2 * np.pi, 73, endpoint=False)


def _channel(cell, frequency, amplitude, duration):
    comp = [cell.idx(), cell.idx(target=1), cell.idx(1), cell.idx(1, target=1)]
    vectors = cell.dress[:, comp]
    inputs = np.array([np.outer(vectors[:, i], vectors[:, j].conj()) for i in range(4) for j in range(4)])
    pair = cell.pair_density(cell.flux_pulse(inputs, frequency, amplitude, duration))
    pc = [0, 1, 3, 4]
    return pair[:16, pc, :][:, :, pc].reshape(4, 4, 4, 4)


def fidelity_best_phases(cells, frequency, amplitude, duration) -> tuple[float, float, float, float]:
    """(F, leakage, control phase, target phase): the parity-branch-averaged CZ fidelity maximised over the
    committed local phases (on a 5-degree grid)."""
    if amplitude <= 0 or duration < 8e-9 or frequency <= 0:
        return 0.0, 1.0, 0.0, 0.0
    zc, zt = np.meshgrid(PHASES, PHASES, indexing="ij")
    target = np.stack([np.ones_like(zc), np.exp(1j * zt), np.exp(1j * zc), -np.exp(1j * (zc + zt))])
    ent, survival = 0.0, 0.0
    for weight, cell in cells:
        ch = _channel(cell, frequency, amplitude, duration)
        diag = np.array([[ch[i, j, i, j] for j in range(4)] for i in range(4)])
        ent = ent + weight * np.einsum("iab,ij,jab->ab", target.conj(), diag, target).real
        survival += weight * sum(np.trace(ch[i, i]).real for i in range(4))
    k = np.unravel_index(int(np.argmax(ent)), ent.shape)
    return (float(np.clip((ent[k] + survival) / 20, 0, 1)), float(1 - survival / 4),
            float(PHASES[k[0]]), float(PHASES[k[1]]))


def _coarse_grid(cell, amplitudes, span, n_freq, durations) -> tuple[list, float]:
    """(F, frequency, amplitude, duration) of a pulse grid on the noise-free ``cell``, best first, and its d20."""
    e = cell.energies
    d20 = abs(e[cell.idx(1, 0, 1)] - e[cell.idx(2)])
    grid = []
    for amp in amplitudes:
        for f in d20 + np.linspace(-span, span, n_freq):
            for dur in durations:
                grid.append((fidelity_best_phases([(1.0, cell)], f, amp, dur)[0], f, amp, dur))
    grid.sort(reverse=True)
    return grid, d20


def coupler_ceiling(chain, index: int, currents, *, time_s: float = 0.0, frames=None,
                    amplitudes=(0.1, 0.2, 0.35, 0.5), max_duration: float = 800e-9, span: float = 16e6) -> dict:
    """Best CZ of coupler ``index`` at the joint bias ``currents`` (privileged search). The coarse grid
    uses the noise-free cell; the refinement and the reported fidelity average every noise branch."""
    theta = chainmod.coupler_theta(chain, index, list(currents), time_s=time_s, frames=frames)
    bias = 0.0 if currents[index] is None else float(currents[index])
    cells = [(w, Cell(th, bias)) for w, th in noise_branches(theta)]
    grid, d20 = _coarse_grid(Cell(theta, bias), amplitudes, span, 17, np.arange(48e-9, max_duration, 40e-9))
    best = (0.0, grid[0][1:])
    scale = np.array([0.5e6, 0.03, 10e-9])
    for _, *x0 in grid[:3]:
        x0 = np.asarray(x0)
        res = minimize(lambda z: -fidelity_best_phases(cells, *(x0 + z * scale))[0], np.zeros(3),
                       method="Nelder-Mead", options=dict(maxfev=150, xatol=0.02, fatol=1e-6))
        if -res.fun > best[0]:
            best = (-res.fun, tuple(x0 + res.x * scale))
    f, (frequency, amplitude, duration) = best[0], best[1]
    F, leak, zc, zt = fidelity_best_phases(cells, frequency, amplitude, duration)
    return {"coupler": index, "bias": bias, "fidelity": F, "leakage": leak, "cz_pulse_frequency": float(frequency),
            "cz_pulse_amplitude": float(amplitude), "cz_pulse_duration": float(duration),
            "control_local_phase": zc, "target_local_phase": zt, "d20_hz": float(d20)}


# The ceiling is searched in the standard tunable-coupler regime: the coupler parked at least this far above the
# higher of its two qubits. Reaching a point below them sweeps the coupler through both qubit frequencies, which
# devices avoid and where the model's constant couplings and coupler coherence are least trustworthy.
MIN_COUPLER_ABOVE_QUBITS_HZ = 0.5e9


def bias_for_flux(chain, index: int, currents, flux: float) -> float:
    """The current of coupler ``index`` that puts it at ``flux`` [Phi_0], every other current held."""
    A = chain.crosstalk
    others = sum(A[index, j] * (0.0 if c is None else float(c)) for j, c in enumerate(currents) if j != index)
    return float(chain.couplers[index].i_sweet + (flux - others) / A[index, index])


def flux_scan(chain, index: int, currents=None, n: int = 12, margin_hz: float = MIN_COUPLER_ABOVE_QUBITS_HZ) -> np.ndarray:
    """``n`` coupler fluxes [Phi_0] from near the sweet spot (flat, so excluded) down to where the bare coupler
    frequency comes within ``margin_hz`` of its higher qubit, plus the flux ``currents`` sets when it is in that
    range. The frequency is even and periodic in the flux, so this half period covers every distinct operating
    point of the regime."""
    c = chain.couplers[index]
    floor = max(chain.qubits[q].f01 for q in chain.pair(index)) + margin_hz
    frequency = lambda phi: c.fc0 * (np.cos(np.pi * phi) ** 2 + c.d_asym ** 2 * np.sin(np.pi * phi) ** 2) ** 0.25  # noqa: E731
    phi = np.linspace(0.04, 0.5, 461)
    allowed = phi[frequency(phi) >= floor]
    if not allowed.size:
        return allowed
    scan = np.linspace(allowed[0], allowed[-1], n)
    if currents is not None:
        here = abs((float(chainmod.coupler_fluxes(chain, currents)[index]) + 0.5) % 1.0 - 0.5)
        if frequency(here) >= floor:
            scan = np.sort(np.append(scan, here))
    return scan


def coupler_best_ceiling(chain, index: int, currents, *, fluxes=None, keep: int = 2,
                         amplitudes=(0.1, 0.2, 0.35, 0.5, 0.7), max_duration: float = 800e-9) -> dict:
    """Best CZ of coupler ``index`` over its own bias too, the other currents held (privileged search).

    A coarse noise-free pulse grid ranks the fluxes of ``flux_scan``; ``coupler_ceiling`` then runs at the ``keep``
    best biases. The other couplers stay where ``currents`` puts them, since the agent can null their crosstalk by
    re-biasing this coupler. ``scan`` records the coarse fidelity at each flux."""
    lo, hi = LIMITS["dc_current"]
    scan = []
    for flux in flux_scan(chain, index, currents) if fluxes is None else fluxes:
        trial = list(currents)
        trial[index] = bias_for_flux(chain, index, currents, flux)
        if not lo <= trial[index] <= hi:
            continue
        theta = chainmod.coupler_theta(chain, index, trial)
        grid, _ = _coarse_grid(Cell(theta, trial[index]), amplitudes[1:], 16e6, 9, np.arange(48e-9, max_duration, 60e-9))
        scan.append((grid[0][0], float(flux), trial))
    best = None
    for _, flux, trial in sorted(scan, key=lambda s: s[0], reverse=True)[:keep]:
        out = coupler_ceiling(chain, index, trial, amplitudes=amplitudes, max_duration=max_duration)
        if best is None or out["fidelity"] > best["fidelity"]:
            best = {**out, "flux": flux}
    return {**best, "scan": [[round(flux, 4), round(f, 5)] for f, flux, _ in scan]}
