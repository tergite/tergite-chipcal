"""Opt-in realism for the simulated chain: effects a stationary, perfectly framed model leaves out.

The base simulator is a stationary device whose gates and Ramseys remove each qubit's idle phase
at its *true* dressed frequency. Physical devices differ in ways that make calibration harder, and
this module adds the following effects:

- **Calibrated frames.** Single-qubit pulses and virtual-Z frames run at a *calibrated* clock.
  After a coupler bias move, or after frequency drift, the true frequency is off that clock, and
  every CZ, Ramsey and RB sequence accumulates 2 pi delta t of phase until the qubit is retuned.
- **Frequency drift.** A linear drift plus a random walk of each qubit frequency, on wall time.
- **Neighbour pull.** A qubit is dressed by *every* coupler it touches: moving one coupler moves
  the qubit under its other coupler's gate.
- **Charge-parity splitting.** Quasiparticle tunnelling switches the junction's charge parity, so
  a transition is split into two lines that shots average over. The split follows the exact
  transmon spectrum (Koch et al., PRA 76, 042319, 2007) of each qubit's own E_C and E_J/E_C, and
  changes as the offset charge drifts.
- **Telegraph defects.** Two-level defects near a qubit switch at random and shift its frequency
  by a few kHz (Burnett et al., npj Quantum Inf. 5, 54, 2019).
- **Coherence and readout spread.** T1 and T2 spread, and imperfect three-state readout.

All of it enters the same Hamiltonian, solver and readout chain as physics; nothing plants an
answer. ``ChainDeviceParams.realism`` is ``None`` by default, and then no code path changes.
Every sampled range lives in :class:`RealismProfile`; docs/parameters.md gives their sources.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Sequence

import numpy as np

# Single-qubit gate layers per two-qubit Clifford (both qubits in parallel), and one layer's duration
# (8.25 single-qubit gates per two-qubit Clifford, Barends et al., Nature 508, 500, 2014).
CLIFFORD_1Q_LAYERS = 5
SQ_LAYER_S = 40e-9


@dataclass(frozen=True)
class QubitNoise:
    """One qubit's non-Markovian frequency noise.

    ``parity_split_hz`` / ``parity_split_12_hz``: the *largest* charge-parity splitting of f01 / f12 (full line
    separation, at offset charge 0); the split at offset charge n_g is that times |cos(2 pi n_g)|. The two branches
    are equally likely shot to shot and shift f12 with the opposite sign of f01.
    ``drift_rate_hz_per_s``: linear drift of f01 (and f12) with wall time. ``diffusion_hz_per_sqrt_s``: random-walk
    drift. ``tls_jumps_hz`` / ``tls_rates_hz``: random-telegraph frequency jumps from defects near the qubit, each a
    full jump size and a switching rate. Every path is seeded and the same on every evaluation."""
    parity_split_hz: float = 0.0
    parity_split_12_hz: float = 0.0
    drift_rate_hz_per_s: float = 0.0
    diffusion_hz_per_sqrt_s: float = 0.0
    tls_jumps_hz: tuple = ()
    tls_rates_hz: tuple = ()

    def __post_init__(self):
        values = [self.parity_split_hz, self.parity_split_12_hz, self.drift_rate_hz_per_s, self.diffusion_hz_per_sqrt_s,
                  *self.tls_jumps_hz, *self.tls_rates_hz]
        if not np.isfinite(values).all() or min(self.parity_split_hz, self.parity_split_12_hz, self.diffusion_hz_per_sqrt_s) < 0:
            raise ValueError("qubit noise must be finite; splits and diffusion non-negative")
        if len(self.tls_jumps_hz) != len(self.tls_rates_hz) or any(r < 0 for r in self.tls_rates_hz):
            raise ValueError("one non-negative switching rate per telegraph jump required")


@lru_cache(maxsize=4096)
def _telegraph_flips(seed: int, qubit: int, index: int, rate: float, horizon: float) -> tuple:
    """Seeded Poisson switching times of one telegraph fluctuator up to ``horizon`` seconds."""
    rng = np.random.default_rng([seed, qubit, index, 307])
    times, t = [], 0.0
    while rate > 0:
        t += rng.exponential(1.0 / rate)
        if t > horizon:
            break
        times.append(t)
    return tuple(times), bool(rng.random() < 0.5)


@dataclass(frozen=True)
class Realism:
    """Device-level realism switches. ``qubit_noise`` has one entry per chain qubit."""
    qubit_noise: tuple[QubitNoise, ...] = ()
    seed: int = 0
    # Gates, Ramseys and RB run in a calibrated frame (see module docstring).
    calibrated_frames: bool = True
    # Frequency error of a frame calibration (at reset, and the Ramsey retune after a move) [Hz].
    calibration_sigma_hz: float = 2e3
    # After a parking-current change, retune both adjacent qubits' frames at this cost [s].
    retune_on_move: bool = True
    retune_seconds: float = 60.0
    # Wall time per agent turn beyond the instrument time [s]; drift runs on wall time.
    turn_latency_s: float = 0.0
    # A qubit is dressed by every coupler it touches, not only the one whose gate is simulated.
    neighbour_pull: bool = True
    # Clifford lengths of the benchmark; None uses the simulator's standard protocol.
    rb_lengths: tuple | None = None
    drift_step_s: float = 60.0
    # The offset charge of every junction is redrawn after each interval [s] (slow offset-charge drift).
    offset_charge_interval_s: float = 300.0

    def __post_init__(self):
        if min(self.calibration_sigma_hz, self.retune_seconds, self.turn_latency_s) < 0 or self.drift_step_s <= 0:
            raise ValueError("invalid realism timing or calibration settings")
        if self.offset_charge_interval_s <= 0:
            raise ValueError("offset-charge interval must be positive")
        if self.rb_lengths is not None and (len(self.rb_lengths) < 3 or min(self.rb_lengths) < 0):
            raise ValueError("rb_lengths needs at least three non-negative Clifford lengths")

    def noise(self, q: int) -> QubitNoise:
        return self.qubit_noise[q] if q < len(self.qubit_noise) else QubitNoise()

    def drift(self, q: int, time_s: float) -> float:
        """f01 offset of qubit ``q`` at wall time ``time_s`` [Hz]: linear drift, a seeded random walk (linear in time
        between grid steps, so continuous), and the seeded telegraph jumps."""
        n = self.noise(q)
        time_s = float(time_s)
        offset = n.drift_rate_hz_per_s * time_s
        if n.diffusion_hz_per_sqrt_s > 0 and time_s > 0:
            steps = time_s / self.drift_step_s
            k = int(np.floor(steps))
            rng = np.random.default_rng([int(self.seed), int(q), 101])
            increments = rng.normal(0.0, n.diffusion_hz_per_sqrt_s * np.sqrt(self.drift_step_s), size=k + 1)
            path = np.concatenate([[0.0], np.cumsum(increments)])
            offset += path[k] + (steps - k) * (path[k + 1] - path[k])
        horizon = 3600.0 * (int(time_s // 3600.0) + 1)
        for index, (jump, rate) in enumerate(zip(n.tls_jumps_hz, n.tls_rates_hz)):
            flips, start_up = _telegraph_flips(int(self.seed), int(q), index, float(rate), horizon)
            up = start_up ^ (bisect_right(flips, time_s) % 2 == 1)
            offset += (0.5 if up else -0.5) * float(jump)
        return float(offset)

    def parity_splits(self, q: int, time_s: float) -> tuple[float, float]:
        """(f01, f12) charge-parity line separation of qubit ``q`` at wall time ``time_s`` [Hz]."""
        n = self.noise(q)
        if not (n.parity_split_hz or n.parity_split_12_hz):
            return 0.0, 0.0
        interval = int(float(time_s) // self.offset_charge_interval_s)
        offset_charge = np.random.default_rng([int(self.seed), int(q), interval, 503]).random()
        c = abs(np.cos(2 * np.pi * offset_charge))
        return float(n.parity_split_hz * c), float(n.parity_split_12_hz * c)


# (f01, f12) splits [Hz] below which both charge-parity branches are simulated as one.
PARITY_RESOLVABLE_HZ = (200.0, 5e3)


def parity_branches(theta):
    """[(weight, theta)] over the charge-parity branches of ``theta``'s two qubits (one entry when none split).

    In a branch the qubit's f01 moves by s*delta01/2 and its f12 by -s*delta12/2; its calibrated frame does not move,
    so the frame offset (true minus frame) moves with f01."""
    branches = [(1.0, theta)]
    for role in ("c", "t"):
        d01, d12 = getattr(theta, f"parity_{role}"), getattr(theta, f"parity12_{role}")
        # A split below these is unresolvable: over a 50 us Ramsey a 200 Hz f01 split costs < 0.1 %
        # contrast, and a 5 kHz f12 split moves the |20> phase of a 1 us gate by < 0.02 rad.
        if d01 < PARITY_RESOLVABLE_HZ[0] and d12 < PARITY_RESOLVABLE_HZ[1]:
            continue
        split = []
        for weight, th in branches:
            for s in (-1.0, 1.0):
                f01 = getattr(th, f"f01_{role}") + s * d01 / 2
                alpha = getattr(th, f"alpha_{role}") - s * d12 / 2 - s * d01 / 2
                split.append((weight / 2, replace(th, **{f"f01_{role}": f01, f"alpha_{role}": alpha,
                                                       f"frame_offset_{role}": getattr(th, f"frame_offset_{role}") + s * d01 / 2})))
        branches = split
    return branches


def frame_rb_contraction(offset_c_hz: float, offset_t_hz: float, seconds: float) -> float:
    """Depolarizing parameter of the idle Z error a frame offset leaves per Clifford of length ``seconds``.

    A two-qubit unitary U contributes p = (|Tr U|^2 - 1)/15 under twirling; Rz(a) x Rz(b) has
    |Tr U| = 4 |cos(a/2) cos(b/2)|."""
    a, b = 2 * np.pi * offset_c_hz * seconds, 2 * np.pi * offset_t_hz * seconds
    return float(np.clip((16 * (np.cos(a / 2) * np.cos(b / 2)) ** 2 - 1) / 15, 0.0, 1.0))


def clifford_seconds(gate_s: float, cz_per_clifford: float = 1.5) -> float:
    return float(cz_per_clifford * gate_s + CLIFFORD_1Q_LAYERS * SQ_LAYER_S)


# ------------------------------------------------------------------ the default profile
@dataclass(frozen=True)
class RealismProfile:
    """Sampling ranges of the realism effects, all from published devices (docs/parameters.md lists each source)."""
    t1_s: tuple[float, float] = (15e-6, 110e-6)
    t2_over_t1: tuple[float, float] = (0.6, 2.0)
    # Three-state assignment probabilities P(read j | prepared j), preparation error included.
    p00: tuple[float, float] = (0.97, 0.995)
    p11: tuple[float, float] = (0.90, 0.97)
    p22: tuple[float, float] = (0.92, 0.96)
    # Charge parity: each qubit's largest split follows from its E_C and E_J/E_C (exact transmon spectrum).
    charge_parity: bool = True
    drift_rate_hz_per_h: tuple[float, float] = (-1e3, 1e3)
    diffusion_hz_per_sqrt_h: tuple[float, float] = (0.0, 2e3)
    # Telegraph defects per qubit (0..max), each with a jump size and a log-uniform switching rate.
    tls_per_qubit_max: int = 2
    tls_jump_hz: tuple[float, float] = (1e3, 20e3)
    tls_rate_hz: tuple[float, float] = (2e-5, 2e-3)
    calibration_sigma_hz: float = 2e3
    retune_seconds: float = 60.0
    turn_latency_s: float = 0.0
    offset_charge_interval_s: float = 300.0
    rb_lengths: tuple | None = None

    @classmethod
    def from_dict(cls, values: dict) -> "RealismProfile":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items() if k in known})


def confusion(rng: np.random.Generator, profile: RealismProfile) -> np.ndarray:
    """Column-stochastic 3x3 P(read i | prepared j): |0> errs mostly to |1>; |1> errs 70/30 to |0> (decay) and |2>;
    |2> errs mostly to |1> (decay during readout)."""
    p00, p11, p22 = (rng.uniform(*profile.p00), rng.uniform(*profile.p11), rng.uniform(*profile.p22))
    c = np.zeros((3, 3))
    c[:, 0] = [p00, (1 - p00) * 0.9, (1 - p00) * 0.1]
    c[:, 1] = [(1 - p11) * 0.7, p11, (1 - p11) * 0.3]
    c[:, 2] = [(1 - p22) * 0.2, (1 - p22) * 0.8, p22]
    return c


def sample_qubit_noise(rng: np.random.Generator, profile: RealismProfile, qubit=None) -> QubitNoise:
    split = split12 = 0.0
    if profile.charge_parity and qubit is not None:
        from chipcal.sim import transmon
        ratio = transmon_ratio(qubit)
        ec = transmon.from_f01(qubit.f01, ratio)["ec_hz"]
        split, split12 = transmon.max_parity_splits_hz(ec, ratio)
    n_tls = int(rng.integers(0, profile.tls_per_qubit_max + 1))
    jumps = tuple(float(rng.uniform(*profile.tls_jump_hz)) for _ in range(n_tls))
    lo, hi = np.log(profile.tls_rate_hz[0]), np.log(profile.tls_rate_hz[1])
    rates = tuple(float(np.exp(rng.uniform(lo, hi))) for _ in range(n_tls))
    return QubitNoise(parity_split_hz=float(split), parity_split_12_hz=float(split12),
                      drift_rate_hz_per_s=float(rng.uniform(*profile.drift_rate_hz_per_h)) / 3600,
                      diffusion_hz_per_sqrt_s=float(rng.uniform(*profile.diffusion_hz_per_sqrt_h)) / 60,
                      tls_jumps_hz=jumps, tls_rates_hz=rates)


def transmon_ratio(qubit) -> float:
    """E_J/E_C of a qubit with this f01 and anharmonicity (inverting the exact transmon spectrum)."""
    from scipy.optimize import brentq
    from chipcal.sim import transmon
    target = float(qubit.alpha) / float(qubit.f01)
    g = lambda r: transmon.from_f01(qubit.f01, r)["alpha_hz"] / float(qubit.f01) - target  # noqa: E731
    lo, hi = 5.0, 400.0
    if g(lo) * g(hi) > 0:
        return hi if abs(g(hi)) < abs(g(lo)) else lo
    return float(brentq(g, lo, hi, xtol=1e-6))


def realize(chain, seed: int, profile: RealismProfile | None = None):
    """``chain`` with the profile's coherence, readout and frequency noise (a new ChainDeviceParams)."""
    profile = profile or RealismProfile()
    rng = np.random.default_rng([int(seed), 211])
    qubits, noise = [], []
    for q in chain.qubits:
        t1 = float(rng.uniform(*profile.t1_s))
        qubits.append(replace(q, t1=t1, t2=float(min(2 * t1, t1 * rng.uniform(*profile.t2_over_t1))),
                              confusion=confusion(rng, profile)))
        noise.append(sample_qubit_noise(rng, profile, q))
    realism = Realism(qubit_noise=tuple(noise), seed=int(seed), calibration_sigma_hz=profile.calibration_sigma_hz,
                      retune_seconds=profile.retune_seconds, turn_latency_s=profile.turn_latency_s,
                      rb_lengths=profile.rb_lengths, offset_charge_interval_s=profile.offset_charge_interval_s)
    return replace(chain, qubits=qubits, realism=realism)


def qubit_pulls(cell, level: int = 1) -> tuple[float, float]:
    """(control, target) dressed-minus-bare f01 (``level`` 1) or f12 (``level`` 2) of a Cell at its bias [Hz]."""
    t, e = cell.theta, cell.energies
    if level == 1:
        return (float(e[cell.idx(1)] - e[cell.idx()] - t.f01_c), float(e[cell.idx(target=1)] - e[cell.idx()] - t.f01_t))
    return (float(e[cell.idx(2)] - e[cell.idx(1)] - t.f12_c), float(e[cell.idx(target=2)] - e[cell.idx(target=1)] - t.f12_t))


def mixture(values: Sequence[tuple[float, np.ndarray]]) -> np.ndarray:
    return sum(w * np.asarray(v) for w, v in values)
