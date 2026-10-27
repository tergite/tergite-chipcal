"""A linear chain of fixed-frequency transmons and flux-tunable couplers, with hidden flux crosstalk.

    Q0 --C0-- Q1 --C1-- Q2 --C2-- Q3 --C3-- Q4

of ``n_qubits`` fixed-frequency transmons joined by ``n_qubits - 1`` flux-tunable
couplers (the *couplers-only* tunability track: qubits have no flux knob, only the
couplers do, so the "sweet spots" to find are the coupler operating points).

Beyond the single-coupler cell, the chain adds a **hidden current->flux crosstalk matrix** ``A``::

    Phi = A . I        (Phi: coupler fluxes [Phi_0], I: coupler bias currents [A])

``A`` is diagonally dominant with nearest-neighbour off-diagonals.  With ``A``
diagonal the chain is independent single-coupler tune-ups; with off-diagonals,
biasing one coupler shifts its neighbours' fluxes, so a neighbour whose CZ gate was
already calibrated at its old flux silently degrades. Finding the **set** of
operating points is a coupled fixed point, and re-biasing one coupler triggers a
re-calibration cascade through its neighbours.

Each coupler's gate is scored by synthesising an *effective*
:class:`~chipcal.sim.params.DeviceParams` whose ``i_sweet`` is shifted by the
crosstalk contribution of the neighbours' committed currents, then running the
single-coupler multimode model on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from chipcal.sim import physics
from chipcal.core.config import CommittedConfig
from chipcal.sim.params import DeviceParams

CHAIN_FAMILIES = ("nominal", "strong_crosstalk")


@dataclass
class QubitParams:
    """One fixed-frequency transmon in the chain (no flux knob)."""

    f01: float
    alpha: float
    t1: float
    t2: float
    confusion: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass
class CouplerParams:
    """One flux-tunable SQUID coupler joining two adjacent qubits.

    ``i_sweet`` is the coupler's *own* bias current at flux 0 with every neighbour
    line at zero current; the crosstalk offset from biased neighbours is applied on
    top of it by :func:`coupler_theta`.  ``g_qc_left`` / ``g_qc_right`` are the
    couplings to the lower- / higher-index adjacent qubit.
    """

    fc0: float
    d_asym: float
    i_sweet: float
    i_period: float
    ac_flux_per_amp: float
    g_qc_left: float
    g_qc_right: float
    kappa_shift: float
    stark_left: float
    stark_right: float
    tls: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    multimode: object | None = None
    # Reduced pulse waveform and integration settings (SI units).
    pulse_wait_s: float = 4e-9
    pulse_rise_fall_s: float = 16e-9
    pulse_max_step_s: float = 0.30e-9
    dist_lowpass_tau: float = 0.0
    dist_droop_amp: float = 0.0
    dist_droop_tau: float = 0.0


@dataclass
class ChainDeviceParams:
    """Hidden latent state of a coupler chain: qubits, couplers, crosstalk.

    ``crosstalk`` is the ``(n_couplers, n_couplers)`` current->flux transfer matrix
    ``A`` of ``Phi = A . I``.  Its diagonal is ``1 / i_period`` (the self-transfer,
    so a diagonal ``A`` reproduces ``physics.flux_dc`` exactly) and the
    nearest-neighbour off-diagonals are the hidden crosstalk the agent never sees.
    """

    qubits: list[QubitParams]
    couplers: list[CouplerParams]
    crosstalk: np.ndarray
    family: str = "nominal"
    flux_transfer: object | None = None
    # Opt-in realism (chipcal.sim.realism.Realism): calibrated frames, drift, charge parity, neighbour pull.
    realism: object | None = None
    # The qubit pair (a, b), a < b, that each coupler joins; None is the chain, coupler i joining qubits i, i+1.
    edges: tuple | None = None

    def __post_init__(self) -> None:
        n_q = len(self.qubits)
        n_c = len(self.couplers)
        if self.edges is None and n_c != n_q - 1:
            raise ValueError(f"a chain of {n_q} qubits needs {n_q - 1} couplers, got {n_c}")
        if self.edges is not None:
            self.edges = tuple((int(a), int(b)) for a, b in self.edges)
            if len(self.edges) != n_c or any(not 0 <= a < b < n_q for a, b in self.edges):
                raise ValueError(f"edges must give {n_c} qubit pairs (a, b) with 0 <= a < b < {n_q}")
        A = np.asarray(self.crosstalk, dtype=float)
        if A.shape != (n_c, n_c):
            raise ValueError(f"crosstalk must be ({n_c}, {n_c}), got {A.shape}")
        self.crosstalk = A
        if self.flux_transfer is not None and not np.allclose(self.flux_transfer.dc, A, rtol=1e-12, atol=0):
            raise ValueError("chain DC matrix and FluxTransfer.dc must agree")
        if self.flux_transfer is not None and not np.allclose(self.flux_transfer.offset,
                -np.diag(A)*np.array([c.i_sweet for c in self.couplers]),rtol=1e-12,atol=1e-15):
            raise ValueError("FluxTransfer offset must match the chain sweet-current convention")

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    @property
    def n_couplers(self) -> int:
        return len(self.couplers)

    def pair(self, i: int) -> tuple[int, int]:
        """(left, right) qubit indices of coupler ``i``: the qubits its ``g_qc_left`` and ``g_qc_right`` couple."""
        return self.edges[i] if self.edges is not None else (i, i + 1)

    def coupler_qubits(self, i: int) -> tuple[int, int]:
        """(control, target) qubit indices for coupler ``i``: control = the higher
        frequency qubit (matching the single-coupler convention)."""
        a, b = self.pair(i)
        return (a, b) if self.qubits[a].f01 >= self.qubits[b].f01 else (b, a)


# --------------------------------------------------------------------------- flux
def coupler_fluxes(chain: ChainDeviceParams, currents) -> np.ndarray:
    """True coupler fluxes [Phi_0] for a bias-current vector ``currents`` [A].

    ``Phi_i = A_ii (I_i - i_sweet_i) + sum_{j != i} A_ij I_j``; with ``A`` diagonal
    this is exactly ``physics.flux_dc`` per coupler.  ``None`` entries (a coupler
    whose bias has not been committed) are treated as zero line current.
    """
    A = chain.crosstalk
    n = chain.n_couplers
    I = np.array([0.0 if c is None else float(c) for c in currents], dtype=float)
    phi0 = np.array([-A[i, i] * chain.couplers[i].i_sweet for i in range(n)])
    return A @ I + phi0


def coupler_theta(chain: ChainDeviceParams, i: int, currents, *, time_s: float = 0.0,
                  frames=None) -> DeviceParams:
    """Effective single-coupler :class:`DeviceParams` for coupler ``i`` (see ``_base_theta``).

    With ``chain.realism`` (chipcal.sim.realism) the two qubits also carry: their f01 at wall
    time ``time_s`` (drift), the pull of every *other* coupler they touch at its current (this coupler's own pull is
    in the Cell's Hamiltonian), their charge-parity splits, and the offset of the true dressed f01 from the
    calibrated frame ``frames[q]`` [Hz] the CZ and Ramsey nodes run in. ``chain.realism is None``: exactly the base.
    """
    base = _base_theta(chain, i, currents)
    realism = getattr(chain, "realism", None)
    if realism is None:
        return base
    from dataclasses import replace
    ctrl, tgt = chain.coupler_qubits(i)
    pulls = coupler_pulls(chain, currents) if realism.neighbour_pull else {}
    pulls12 = coupler_pulls(chain, currents, level=2) if realism.neighbour_pull else {}
    other = lambda q, table: sum(p[q] for j, p in table.items() if j != i and q in p)  # noqa: E731
    total = qubit_frequencies(chain, currents, time_s, pulls=pulls)
    changes = {}
    for role, q in (("c", ctrl), ("t", tgt)):
        # The other couplers move f01 by their f01 pull and f12 by their f12 pull (alpha = f12 - f01).
        pull01, pull12 = other(q, pulls), other(q, pulls12)
        changes[f"f01_{role}"] = getattr(base, f"f01_{role}") + realism.drift(q, time_s) + pull01
        changes[f"alpha_{role}"] = getattr(base, f"alpha_{role}") + pull12 - pull01
        changes[f"parity_{role}"], changes[f"parity12_{role}"] = realism.parity_splits(q, time_s)
        if realism.calibrated_frames and frames is not None and frames[q] is not None:
            changes[f"frame_offset_{role}"] = float(total[q] - frames[q])
    return replace(base, calibrated_frames=bool(realism.calibrated_frames), **changes)


def coupler_pulls(chain: ChainDeviceParams, currents, level: int = 1) -> dict:
    """{coupler j: {qubit: dressed-minus-bare f01 (level 1) or f12 (level 2) that coupler j induces at its current}}."""
    from chipcal.sim.multimode import Cell
    from chipcal.sim.realism import qubit_pulls
    out = {}
    for j in range(chain.n_couplers):
        theta = _base_theta(chain, j, currents)
        current = 0.0 if currents[j] is None else float(currents[j])
        ctrl, tgt = chain.coupler_qubits(j)
        pc, pt = qubit_pulls(Cell(theta, current), level)
        out[j] = {ctrl: pc, tgt: pt}
    return out


def dressed_qubits(chain: ChainDeviceParams, currents, time_s: float = 0.0) -> list:
    """The chain's qubits as a measurement finds them at these currents: f01 and f12 dressed by every coupler they
    touch, plus drift (realism only). Discovery and the disclosed QOIs aim at these lines."""
    from dataclasses import replace
    f01 = qubit_frequencies(chain, currents, time_s)
    p12 = coupler_pulls(chain, currents, level=2)
    out = []
    for q, qubit in enumerate(chain.qubits):
        f12 = qubit.f01 + qubit.alpha + sum(p.get(q, 0.0) for p in p12.values())
        f12 += chain.realism.drift(q, time_s) if chain.realism is not None else 0.0
        out.append(replace(qubit, f01=f01[q], alpha=f12 - f01[q]))
    return out


def qubit_frequencies(chain: ChainDeviceParams, currents, time_s: float = 0.0, *, pulls=None) -> list[float]:
    """True dressed f01 of every chain qubit [Hz]: bare + drift at ``time_s`` + the pull of each coupler it touches.
    This is what a perfect calibration of that qubit would find at these currents (realism only)."""
    realism = chain.realism
    pulls = coupler_pulls(chain, currents) if pulls is None else pulls
    out = []
    for q, qubit in enumerate(chain.qubits):
        drift = realism.drift(q, time_s) if realism is not None else 0.0
        out.append(float(qubit.f01 + drift + sum(p.get(q, 0.0) for p in pulls.values())))
    return out


def _base_theta(chain: ChainDeviceParams, i: int, currents) -> DeviceParams:
    """Effective single-coupler :class:`DeviceParams` for coupler ``i``.

    The neighbours' committed currents enter only through a shifted ``i_sweet`` so
    that, for the *single* current ``I_i`` a node sweeps or commits,
    ``physics.flux_dc(I_i, theta_eff)`` equals the true crosstalk-affected flux
    ``Phi_i(I_i)``.  Concretely, with ``c_i = sum_{j!=i} A_ij I_j`` the off-diagonal
    contribution from neighbours,

        i_sweet_eff = i_sweet_i - c_i * i_period_i ,

    which collapses to ``i_sweet_i`` when no neighbour is biased.  Everything else
    (the SQUID law, couplings, Stark/kappa, TLS, readout) is the coupler's intrinsic
    physics, so the single-coupler model runs unchanged on the returned theta.
    """
    A = chain.crosstalk
    n = chain.n_couplers
    cpl = chain.couplers[i]
    I = np.array([0.0 if c is None else float(c) for c in currents], dtype=float)
    c_i = float(sum(A[i, j] * I[j] for j in range(n) if j != i))
    i_sweet_eff = cpl.i_sweet - c_i * cpl.i_period

    ctrl, tgt = chain.coupler_qubits(i)
    qc, qt = chain.qubits[ctrl], chain.qubits[tgt]
    # left/right couplings map onto (control, target) by which adjacent qubit each is
    g_left, g_right = cpl.g_qc_left, cpl.g_qc_right
    s_left, s_right = cpl.stark_left, cpl.stark_right
    if ctrl == chain.pair(i)[0]:  # control is the left qubit
        g_qc_c, g_qc_t = g_left, g_right
        stark_c, stark_t = s_left, s_right
    else:                       # control is the higher-index (right) qubit
        g_qc_c, g_qc_t = g_right, g_left
        stark_c, stark_t = s_right, s_left

    multimode = cpl.multimode
    if multimode is not None and chain.flux_transfer is not None:
        from dataclasses import replace
        from chipcal.sim.multimode import ExtraMode
        transfer = chain.flux_transfer
        modes = list(multimode.extra_modes)
        fluxes = coupler_fluxes(chain, currents)
        for j in range(n):
            shared = set(chain.pair(j)) & set(chain.pair(i))
            if j == i or not shared:
                continue                  # only couplers sharing a qubit with coupler i
            neighbor = chain.couplers[j]
            common_qubit = shared.pop()
            coupling = neighbor.g_qc_left if common_qubit == chain.pair(j)[0] else neighbor.g_qc_right
            modes.append(ExtraMode(frequency=neighbor.fc0,
                coupling_to_bus=0., alpha=multimode.bus_alpha, levels=3,
                t1=multimode.bus_t1,t2=multimode.bus_t2,
                coupling_to_control=coupling if common_qubit == ctrl else 0.,
                coupling_to_target=coupling if common_qubit == tgt else 0.,
                fc0=neighbor.fc0,d_asym=neighbor.d_asym,dc_flux=fluxes[j],
                dc_flux_per_a=A[j,i],reference_current=I[i],
                ac_flux_per_amp=transfer.ac[j,i],ac_delay=transfer.delay_s[j,i],
                ac_lowpass_tau=transfer.lowpass_tau_s[j,i]))
        # The diagonal AC transfer calibrates the addressed line; it need not
        # equal the DC current-to-flux gain. Neighbors are unobserved modes.
        if abs(transfer.ac[i,i].imag) > 1e-15:
            raise ValueError("self AC transfer must be real; put its phase in delay_s")
        multimode = replace(multimode,extra_modes=tuple(modes),
                            ac_gain=float(transfer.ac[i,i].real/cpl.ac_flux_per_amp),
                            ac_delay=float(transfer.delay_s[i,i]),
                            ac_lowpass_tau=float(transfer.lowpass_tau_s[i,i]))

    return DeviceParams(
        f01_c=qc.f01, f01_t=qt.f01, alpha_c=qc.alpha, alpha_t=qt.alpha,
        t1_c=qc.t1, t1_t=qt.t1, t2_c=qc.t2, t2_t=qt.t2,
        fc0=cpl.fc0, d_asym=cpl.d_asym,
        i_sweet=i_sweet_eff, i_period=cpl.i_period, ac_flux_per_amp=cpl.ac_flux_per_amp,
        g_qc_c=g_qc_c, g_qc_t=g_qc_t,
        kappa_shift=cpl.kappa_shift, stark_c=stark_c, stark_t=stark_t,
        tls=cpl.tls, confusion_c=qc.confusion, confusion_t=qt.confusion,
        family=chain.family, multimode=multimode,
        pulse_wait_s=cpl.pulse_wait_s, pulse_rise_fall_s=cpl.pulse_rise_fall_s,
        pulse_max_step_s=cpl.pulse_max_step_s,
        dist_lowpass_tau=cpl.dist_lowpass_tau, dist_droop_amp=cpl.dist_droop_amp,
        dist_droop_tau=cpl.dist_droop_tau,
    )


def coupler_qubit_labels(i: int, chain: ChainDeviceParams,
                         names: Optional[list[str]] = None) -> tuple[str, str]:
    """(control_label, target_label) for coupler ``i`` -- the ``qubits`` arg the
    measurement nodes expect, ordered control-first."""
    names = names or [f"q{j}" for j in range(chain.n_qubits)]
    ctrl, tgt = chain.coupler_qubits(i)
    return names[ctrl], names[tgt]


# ----------------------------------------------------------------- true fidelity
def gate_fidelity(chain: ChainDeviceParams, i: int, currents,
                  config: CommittedConfig, backend: str = "multimode", *, time_s: float = 0.0,
                  frames=None) -> tuple[float, dict]:
    """Privileged true fidelity of coupler ``i``'s committed gate at the joint bias.

    Reuses the single-coupler scorer on the crosstalk-shifted effective theta, so
    re-biasing a neighbour (which moves ``currents`` and hence ``i_sweet_eff``)
    degrades this gate automatically -- the recalibration cascade, for free.
    """
    theta = coupler_theta(chain, i, currents, time_s=time_s, frames=frames)
    if backend == "multimode":
        from chipcal.sim.multimode import gate_fidelity as multimode_fidelity
        return multimode_fidelity(config, theta)
    raise ValueError("Only multimode is supported")


def chain_gate_fidelities(chain: ChainDeviceParams, currents,
                          configs: list[CommittedConfig],
                          backend: str = "multimode", *, time_s: float = 0.0, frames=None) -> list[float]:
    """True per-coupler fidelities at the current joint bias (reward/eval only)."""
    return [gate_fidelity(chain, i, currents, configs[i], backend=backend, time_s=time_s, frames=frames)[0]
            for i in range(chain.n_couplers)]


# --------------------------------------------------------------------- the prior
class ChainPrior:
    """Domain-randomization prior over coupler chains.

    Qubit frequencies alternate between a high and a low band so adjacent qubits are
    well detuned (collision-free by construction -- frequency-collision families are
    a deferred difficulty lever).  ``crosstalk_scale`` sets the nearest-neighbour
    off-diagonal magnitude as a fraction of the self-transfer; 0 reproduces four
    independent single-coupler devices.
    """

    HIGH_BAND = 4.8e9
    LOW_BAND = 4.2e9

    def __init__(self, n_qubits: int = 5, tls_rate: float = 0.0,
                 crosstalk_scale: float = 0.12):
        if n_qubits < 2:
            raise ValueError("a chain needs at least 2 qubits")
        self.n_qubits = int(n_qubits)
        self.tls_rate = float(tls_rate)
        self.crosstalk_scale = float(crosstalk_scale)

    def _confusion(self, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
        c = np.eye(3)
        for j in range(3):
            off = [k for k in range(3) if k != j]
            errs = rng.uniform(lo, hi, size=2)
            c[off, j] = errs
            c[j, j] = 1.0 - errs.sum()
        return c

    def sample(self, rng: np.random.Generator, family: str = "nominal") -> ChainDeviceParams:
        if family not in CHAIN_FAMILIES:
            raise ValueError(f"unknown family {family!r}; expected one of {CHAIN_FAMILIES}")
        n_q = self.n_qubits
        n_c = n_q - 1

        qubits = []
        for q in range(n_q):
            band = self.HIGH_BAND if q % 2 == 0 else self.LOW_BAND
            f01 = rng.normal(band, 40e6)
            alpha = rng.uniform(-250e6, -180e6)
            t1 = float(np.exp(rng.normal(np.log(60e-6), 0.35)))
            t2 = float(min(2 * t1, np.exp(rng.normal(np.log(80e-6), 0.4))))
            confusion = self._confusion(rng, 0.01, 0.08)
            qubits.append(QubitParams(f01=f01, alpha=alpha, t1=t1, t2=t2, confusion=confusion))

        couplers = []
        for c in range(n_c):
            fc0 = rng.normal(7.8e9, 0.2e9)
            d_asym = rng.uniform(0.0, 0.15)
            i_sweet = rng.uniform(-0.8e-3, 0.8e-3)
            i_period = rng.normal(2.2e-3, 0.3e-3)
            ac_flux_per_amp = rng.normal(0.25, 0.04)
            g_qc_left = rng.normal(70e6, 8e6)
            g_qc_right = rng.normal(70e6, 8e6)
            kappa_shift = rng.normal(3e6, 1e6)
            stark_left = rng.normal(2e7, 8e6)
            stark_right = rng.normal(2e7, 8e6)
            n_tls = rng.poisson(self.tls_rate)
            tls = np.column_stack([
                rng.uniform(4.0e9, 8.5e9, size=n_tls),
                rng.uniform(0.05, 0.5, size=n_tls),
                rng.normal(15e6, 5e6, size=n_tls).clip(min=3e6),
            ]) if n_tls else np.zeros((0, 3))
            couplers.append(CouplerParams(
                fc0=fc0, d_asym=d_asym, i_sweet=i_sweet, i_period=i_period,
                ac_flux_per_amp=ac_flux_per_amp, g_qc_left=g_qc_left, g_qc_right=g_qc_right,
                kappa_shift=kappa_shift, stark_left=stark_left, stark_right=stark_right, tls=tls,
            ))

        scale = self.crosstalk_scale * (2.0 if family == "strong_crosstalk" else 1.0)
        A = self._crosstalk_matrix(rng, couplers, scale)
        return ChainDeviceParams(qubits=qubits, couplers=couplers, crosstalk=A, family=family)

    def _crosstalk_matrix(self, rng: np.random.Generator,
                          couplers: list[CouplerParams], scale: float, edges=None) -> np.ndarray:
        """Diagonally dominant current->flux transfer ``A``: diagonal ``1/i_period``,
        nearest-neighbour off-diagonals a signed fraction ``~scale`` of the row's
        self-transfer; non-adjacent couplers a much weaker second-neighbour leak. Neighbours are adjacent
        in the chain, or with ``edges`` couplers sharing a qubit (second neighbours: sharing a neighbour)."""
        n = len(couplers)
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = 1.0 / couplers[i].i_period
        near = None if edges is None else [[bool(set(edges[i]) & set(edges[j])) for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if near is None:
                    dist = abs(i - j)
                else:
                    dist = 1 if near[i][j] else 2 if any(near[i][k] and near[k][j] for k in range(n)) else 3
                if dist == 1:
                    frac = rng.uniform(0.5, 1.0) * scale
                elif dist == 2:
                    frac = rng.uniform(0.05, 0.2) * scale
                else:
                    frac = 0.0
                if frac:
                    A[i, j] = rng.choice([-1.0, 1.0]) * frac * A[i, i]
        return A

    def initial_parking_currents(self, chain: ChainDeviceParams,
                                 rng: np.random.Generator) -> list[float]:
        """Initial per-coupler bias guesses (the env reset context): a quarter-flux
        point near each coupler's own sweet spot, ignorant of crosstalk."""
        out = []
        for c in chain.couplers:
            frac = rng.uniform(0.18, 0.28)
            out.append(float(c.i_sweet + frac * c.i_period))
        return out
