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


# ---------------------------------------------------------------------------------------------- full ceiling (v0.2)
# The v0.1 ceiling searches one CZ route (|11>-|20>, first harmonic) in the standard coupler regime. The full search
# below tries every route the simulator supports, over the coupler's whole flux range, and checks each gate's
# robustness over the offset-charge states a session passes through. Privileged, evaluation only.
SECTORS = ("via_20", "via_02", "via_20/2", "via_02/2")
FULL_AMPLITUDES = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
FULL_DURATIONS = np.round(np.r_[np.arange(16e-9, 816e-9, 16e-9), np.arange(848e-9, 2.0e-6 + 1e-12, 48e-9)], 12)
MAX_HYBRIDIZATION = 0.02          # coupler share of a dressed qubit |1>: above this the qubit is not a clean qubit
ROBUST_DRAWS = 24                 # offset-charge states (300 s each) the robustness check spans: 2 h of a session


def sector_drive(cell, sector: str) -> float:
    """Drive frequency of a CZ route: |11>-|20> (the 2 on the control) or |11>-|02> (on the target), driven at the
    exchange frequency or at half of it through the coupler's second harmonic (``/2``)."""
    e = cell.energies
    level = cell.idx(2) if sector.startswith("via_20") else cell.idx(target=2)
    exchange = abs(e[level] - e[cell.idx(1, target=1)])
    return exchange / (2 if sector.endswith("/2") else 1)


def hybridization(cell) -> float:
    """The largest coupler share of the two qubits' dressed |1> states (0: clean qubits)."""
    return max(float(abs(cell.dress[cell.idx(bus=1), cell.idx(1)]) ** 2),
               float(abs(cell.dress[cell.idx(bus=1), cell.idx(target=1)]) ** 2))


def _chevron_returns(cell, f0: float, amplitude: float, durations, keep: int = 2) -> list:
    """Full-swing returns of a chevron from |11>: P(11) back up after a real transfer, ranked by P(11) - leakage."""
    rho = cell.state(1, 1)
    freqs = f0 + np.linspace(-3e6, 3e6, 7)
    p11 = np.zeros((len(freqs), len(durations)))
    swap = np.zeros_like(p11)
    for j, d in enumerate(durations):
        pop = np.real(np.diagonal(cell.flux_pulse_many(rho, freqs, amplitude, d), axis1=-2, axis2=-1))
        p11[:, j] = pop[:, cell.idx(1, target=1)]
        swap[:, j] = pop[:, cell.idx(2)] + pop[:, cell.idx(target=2)]
    leak = 1 - p11 - swap
    out = []
    for k, f in enumerate(freqs):
        s = swap[k]
        if s.max() < 0.5:
            continue
        peak = int(np.argmax(s > 0.5 * s.max()))
        for j in range(peak + 1, len(durations) - 1):
            if p11[k, j] >= p11[k, j - 1] and p11[k, j] >= p11[k, j + 1] and swap[k, j] < 0.2:
                out.append((float(p11[k, j] - leak[k, j]), float(f), float(durations[j])))
                break
    return sorted(out, reverse=True)[:keep]


def _refine(cells, frequency: float, amplitude: float, duration: float) -> tuple:
    """Local Nelder-Mead on (frequency, amplitude, duration), the noise branches averaged, phases optimal."""
    x0 = np.array([frequency, amplitude, duration])
    scale = np.array([0.5e6, max(0.2 * amplitude, 0.005), 12e-9])
    lo, hi = LIMITS["duration"]

    def objective(z):
        f, a, d = x0 + z * scale
        if not (0 < a <= 1.0 and lo <= d <= hi and f > 0):
            return 0.0
        try:
            return -fidelity_best_phases(cells, f, a, d)[0]
        except FloatingPointError:
            return 0.0
    res = minimize(objective, np.zeros(3), method="Nelder-Mead",
                   options=dict(maxfev=60, xatol=0.05, fatol=2e-4, initial_simplex=np.vstack([np.zeros(3), np.eye(3)])))
    f, a, d = x0 + res.x * scale
    F, leak, zc, zt = fidelity_best_phases(cells, f, a, d)
    return F, leak, float(f), float(a), float(d), zc, zt


def robustness(chain, index: int, currents, gate: dict, draws: int = ROBUST_DRAWS) -> dict:
    """A fixed gate (phases included) over ``draws`` offset-charge states, sampled mid-interval."""
    from chipcal.core.config import CommittedConfig
    cfg = CommittedConfig(parking_current=gate["bias"], cz_pulse_frequency=gate["cz_pulse_frequency"],
                          cz_pulse_amplitude=gate["cz_pulse_amplitude"], cz_pulse_duration=gate["cz_pulse_duration"],
                          control_local_phase=gate["control_local_phase"], target_local_phase=gate["target_local_phase"])
    interval = chain.realism.offset_charge_interval_s if getattr(chain, "realism", None) is not None else 300.0
    fs = np.array([chainmod.gate_fidelity(chain, index, currents, cfg, time_s=(k + 0.5) * interval)[0]
                   for k in range(draws)])
    return {"robust_mean": float(fs.mean()), "robust_min": float(fs.min()), "robust_p10": float(np.percentile(fs, 10)),
            "robust_frac95": float((fs >= 0.95).mean())}


def _screen_flux(chain, index: int, currents, flux: float, amplitudes=FULL_AMPLITUDES, durations=FULL_DURATIONS):
    """Chevron screen of every route at one coupler flux: ([(score, flux, currents, route, amp, f, d)], skip reason)."""
    lo, hi = LIMITS["dc_current"]
    trial = list(currents)
    trial[index] = bias_for_flux(chain, index, currents, float(flux))
    if not lo <= trial[index] <= hi:
        return [], "current out of range"
    cell = Cell(chainmod.coupler_theta(chain, index, trial), trial[index])
    if hybridization(cell) > MAX_HYBRIDIZATION:
        return [], "hybridized"
    out = []
    for sector in SECTORS:
        f0 = sector_drive(cell, sector)
        if not LIMITS["drive_frequency"][0] <= f0 <= LIMITS["drive_frequency"][1]:
            continue
        for amp in amplitudes:
            try:
                out += [(score, float(flux), trial, sector, amp, f, d)
                        for score, f, d in _chevron_returns(cell, f0, amp, durations)]
            except FloatingPointError:
                break
    return out, None


def _finish_gate(args) -> dict | None:
    """Refine one screened point on the full noisy channel and check it over offset-charge states."""
    chain, index, currents, (score, flux, trial, sector, amp, f, d) = args
    cells = [(w, Cell(th, trial[index])) for w, th in noise_branches(chainmod.coupler_theta(chain, index, trial))]
    try:
        F, leak, f, a, d, zc, zt = _refine(cells, f, amp, d)
    except FloatingPointError:
        return None
    g = {"sector": sector, "flux": flux, "bias": float(trial[index]), "fidelity": F, "leakage": leak,
         "cz_pulse_frequency": f, "cz_pulse_amplitude": a, "cz_pulse_duration": d,
         "control_local_phase": zc, "target_local_phase": zt, "screen": score}
    g.update(robustness(chain, index, trial, g))
    return g


def coupler_full_ceiling(chain, index: int, currents, *, n_flux: int = 16, keep: int = 8, legacy: dict | None = None,
                         amplitudes=FULL_AMPLITUDES, durations=FULL_DURATIONS, map_fn=map) -> dict:
    """Best CZ of coupler ``index`` over every route (``SECTORS``), its whole flux range and the environment's pulse
    limits, the other currents held. Chevrons from |11> screen (flux, route, amplitude); the ``keep`` best returns
    (at most two per flux and route) are refined on the full noisy channel; each refined gate (and ``legacy``, the
    v0.1 ceiling gate, if given) is checked over offset-charge states. ``best`` maximises the fidelity at time zero
    (the v0.1 definition), ``robust`` the mean over offset-charge states. ``map_fn`` (e.g. a process pool's map)
    parallelises the flux screen and the refinements."""
    from functools import partial
    fluxes = [float(x) for x in np.linspace(0.02, 0.5, n_flux)]
    screened, skipped = [], []
    for flux, (points, reason) in zip(fluxes, map_fn(partial(_screen_flux, chain, index, list(currents),
                                                             amplitudes=amplitudes, durations=durations), fluxes)):
        screened += points
        if reason:
            skipped.append([round(flux, 4), reason])
    screened.sort(key=lambda s: -s[0])
    picked, seen = [], {}
    for s in screened:
        key = (s[1], s[3])
        if seen.get(key, 0) < 2:
            picked.append(s)
            seen[key] = seen.get(key, 0) + 1
        if len(picked) == keep:
            break
    gates = [g for g in map_fn(_finish_gate, [(chain, index, list(currents), p) for p in picked]) if g is not None]
    if legacy is not None:
        g = {"sector": "via_20", "legacy_v0.1": True, **{k: legacy[k] for k in (
            "flux", "bias", "fidelity", "leakage", "cz_pulse_frequency", "cz_pulse_amplitude", "cz_pulse_duration",
            "control_local_phase", "target_local_phase") if k in legacy}}
        trial = list(currents)
        trial[index] = g["bias"]
        g.update(robustness(chain, index, trial, g))
        gates.append(g)
    best = max(gates, key=lambda g: g["fidelity"]) if gates else None
    robust = max(gates, key=lambda g: g["robust_mean"]) if gates else None
    return {"coupler": index, "best": best, "robust": robust, "gates": gates, "n_screened": len(screened),
            "skipped_fluxes": skipped}
