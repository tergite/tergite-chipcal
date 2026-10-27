"""Multilevel, open-system cell shared by spectroscopy and CZ calibration.

H/h = sum(w_j n_j + alpha_j n_j(n_j-1)/2) + sum(g_ij a_i† a_j + h.c.).
The bus frequency follows the asymmetric SQUID at the *instantaneous* flux.
No dispersive denominator, planted CZ ridge, Stark coefficient, or TLS penalty
is used. Frequencies are Hz. The capacitive interaction uses the excitation-
conserving RWA; a total-excitation cutoff is explicit and configurable.

Finite flux pulses use midpoint exponentials with Strang-split Lindblad loss.
The microwave spectroscopy solver uses the same operators in the drive frame.
Default lifetimes, bus anharmonicity and line parameters are generic assumptions;
see docs/physics.md for scope and docs/parameters.md for sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment
from scipy.signal import convolve, windows

from chipcal.sim import physics

PROTOCOL_DESCRIPTION = (
    "Spectroscopy drives qubit 0=control or 1=target and reads both qubits; the coupler has no readout. "
    "Optional spec_pulse_duration [s] and spec_pulse_amplitude [DAC]. "
    "Flux pulses use soft-square (Hann-smoothed) envelopes. "
    "cz_calibration applies a final X on the control when control_on=1. "
    "Local-phase convention: a committed local phase is the phase the gate pulse accumulates on that qubit "
    "(the gate-on minus gate-off Ramsey fringe shift); the gate removes it. "
    "Reference 2Q RB contains an average of 1.5 CZ gates per Clifford."
)


@dataclass(frozen=True)
class ExtraMode:
    """Unobserved TLS (levels=2) or oscillator, with explicit coupling in Hz."""
    frequency: float
    coupling_to_bus: float
    alpha: float = 0.0
    t1: float = 5e-6
    t2: float = 3e-6
    levels: int = 2
    coupling_to_control: float = 0.0
    coupling_to_target: float = 0.0
    # Nonzero fc0 describes a neighboring tunable coupler, not a fixed TLS.
    fc0: float = 0.0
    d_asym: float = 0.1
    dc_flux: float = 0.0
    dc_flux_per_a: float = 0.0
    reference_current: float = 0.0
    ac_flux_per_amp: complex = 0j
    ac_delay: float = 0.0
    ac_lowpass_tau: float = 0.0

    def __post_init__(self):
        if not np.isfinite([self.frequency, self.coupling_to_bus, self.alpha, self.t1, self.t2,
                           self.coupling_to_control, self.coupling_to_target, self.fc0,
                           self.d_asym, self.dc_flux, self.dc_flux_per_a, self.reference_current,
                           self.ac_flux_per_amp, self.ac_delay, self.ac_lowpass_tau]).all():
            raise ValueError("extra-mode parameters must be finite")
        if self.frequency <= 0 or not 0 < self.t2 <= 2*self.t1 or self.levels not in (2, 3, 4):
            raise ValueError("invalid extra-mode frequency, lifetime, or truncation")
        if min(self.fc0, self.ac_delay, self.ac_lowpass_tau) < 0 or not 0 <= self.d_asym <= 1:
            raise ValueError("invalid extra-mode flux response")


@dataclass(frozen=True)
class MultimodeSettings:
    bus_alpha: float = -200e6
    bus_t1: float = 10e-6
    bus_t2: float = 5e-6
    direct_exchange_hz: float = 0.0
    extra_modes: tuple[ExtraMode, ...] = ()
    max_excitations: int = 2
    # Numerical controls; acquisition point counts do not change these.
    max_step: float = 0.125e-9
    carrier_samples: int = 128
    dissipation_step: float = 2e-9
    spectroscopy_segments: int = 12
    # "dressed_ladder" (spectroscopy_many, 48 envelope segments) or the legacy
    # "full_liouvillian" (spectroscopy, spectroscopy_segments); tests check that they agree.
    spectroscopy_model: str = "dressed_ladder"
    # "batched" (flux_pulse_many) or "reference" (the original per-substep solver).
    pulse_solver: str = "batched"
    pulse_duration: float = 6e-6
    pulse_amplitude: float = 0.003
    rabi_hz_per_amp: float = 20e6
    # Complex AC response of the addressed line, separate from DC transfer.
    ac_gain: float = 1.0
    ac_delay: float = 0.0
    ac_lowpass_tau: float = 0.0
    xy_crosstalk: complex = 0j
    iq_sigma: float = 0.12
    readout_frequencies: tuple[float, float] = (6.5e9, 6.7e9)
    readout_coupling_hz: float = 60e6
    readout_linewidth_hz: float = 2e6
    # Capacitive qubit-coupler coupling scales as sqrt(omega_c): g_qc(omega_c) = g_qc * sqrt(omega_c/g_reference_hz),
    # so g_qc_c/g_qc_t are the couplings at this coupler frequency. 0 keeps g constant (the default).
    g_reference_hz: float = 0.0
    # Quasi-static (1/f) flux noise of the coupler loop, rms [Phi_0]. The gate channel and its benchmark are averaged
    # over the flux offsets +-sigma (two-point Gauss-Hermite, exact to second order); measurement nodes neglect it
    # (its population effect is far below shot noise). 0: no flux noise.
    flux_noise_phi0: float = 0.0

    def __post_init__(self):
        numeric = [self.bus_alpha, self.bus_t1, self.bus_t2, self.direct_exchange_hz,
                   self.max_step, self.dissipation_step, self.pulse_duration,
                   self.pulse_amplitude, self.rabi_hz_per_amp, self.ac_gain,
                   self.ac_delay, self.ac_lowpass_tau, self.iq_sigma,
                   *self.readout_frequencies, self.readout_coupling_hz, self.readout_linewidth_hz]
        if not np.isfinite(numeric).all() or not np.isfinite(self.xy_crosstalk):
            raise ValueError("multimode parameters must be finite")
        if not 0 < self.bus_t2 <= 2*self.bus_t1 or min(self.max_step, self.dissipation_step, self.pulse_duration, self.rabi_hz_per_amp, self.readout_linewidth_hz) <= 0:
            raise ValueError("invalid lifetimes or integration/pulse settings")
        if self.max_excitations not in (2, 3, 4) or not isinstance(self.spectroscopy_segments, int) or self.spectroscopy_segments < 1:
            raise ValueError("excitation cutoff must be 2, 3 or 4; segments must be positive")
        if not isinstance(self.carrier_samples, int) or self.carrier_samples < 16:
            raise ValueError("at least 16 time samples per carrier required")
        if min(self.ac_gain, self.ac_delay, self.ac_lowpass_tau, self.iq_sigma) < 0 or not 0 <= self.pulse_amplitude <= .1:
            raise ValueError("invalid drive/readout parameters")
        if min(self.g_reference_hz, self.flux_noise_phi0) < 0 or not np.isfinite([self.g_reference_hz, self.flux_noise_phi0]).all():
            raise ValueError("coupling reference frequency and flux noise must be finite and non-negative")
        if len(self.extra_modes) > 2:
            raise ValueError("at most two extra modes in the local cell")
        if self.spectroscopy_model not in ("dressed_ladder", "full_liouvillian"):
            raise ValueError("spectroscopy_model must be dressed_ladder or full_liouvillian")
        if self.pulse_solver not in ("batched", "reference"):
            raise ValueError("pulse_solver must be batched or reference")


@lru_cache(maxsize=256)
def _soft_square_samples(count):
    window = windows.hann(max(3, count//2))
    return convolve(np.ones(count), window, mode="same") / window.sum()


def soft_square(t, duration):
    """Soft-square envelope on a 1 GS/s DAC grid, then interpolated.

    N square samples convolved with a Hann window of length N//2 (mode='same'), as
    common pulse libraries define it. It has approximately half-amplitude endpoints;
    it is not a zero-ended fixed-length cosine ramp.
    """
    count = max(4, int(round(duration / 1e-9)))
    values = _soft_square_samples(count)
    return np.interp(np.asarray(t), np.arange(count)*duration/count, values)


def liouvillian(h, collapses):
    """C-order vectorization of -i[H,rho] + sum D[L]rho; H is Hz."""
    eye = np.eye(h.shape[0])
    result = -2j*np.pi*(np.kron(h, eye) - np.kron(eye, h.T))
    for c in collapses:
        rate = c.conj().T @ c
        result += np.kron(c, c.conj()) - .5*(np.kron(rate, eye) + np.kron(eye, rate.T))
    return result


def conjugate(u, rho):
    return u @ rho @ u.conj().T


class Cell:
    def __init__(self, theta, current=0.0, settings=None):
        self.theta = theta
        self.settings = settings or getattr(theta, "multimode", None) or MultimodeSettings()
        s = self.settings
        if np.asarray(theta.tls).size:
            raise ValueError("legacy scalar TLS penalties have no physical coupling; use MultimodeSettings.extra_modes")
        self.flux = float(physics.flux_dc(current, theta))
        self.extra_fluxes = [m.dc_flux + m.dc_flux_per_a*(current-m.reference_current) for m in s.extra_modes]
        self.levels = (s.max_excitations+1,)*3 + tuple(m.levels for m in s.extra_modes)
        self.basis = np.array([x for x in product(*(range(n) for n in self.levels)) if sum(x) <= s.max_excitations])
        self.index = {tuple(x): i for i, x in enumerate(self.basis)}
        self.dim = len(self.basis)
        self.a = []
        for mode in range(len(self.levels)):
            a = np.zeros((self.dim, self.dim))
            for j, state in enumerate(self.basis):
                if state[mode]:
                    low = state.copy(); low[mode] -= 1
                    a[self.index[tuple(low)], j] = np.sqrt(state[mode])
            self.a.append(a)
        self.n = [np.diag(self.basis[:, j].astype(float)) for j in range(len(self.levels))]
        self.total = sum(self.n)
        # Unit-scale qubit-coupler exchange, for g_qc(omega_c) during flux pulses.
        exchange = [self.a[i].T @ self.a[j] for i, j in ((0, 1), (1, 2))]
        self.v_qc = theta.g_qc_c*(exchange[0] + exchange[0].T) + theta.g_qc_t*(exchange[1] + exchange[1].T)
        self.h0 = self.hamiltonian(self.flux)
        # Assign eigenstates one-to-one to bare labels. Preparation/virtual Z
        # use selectively addressed dressed qubit states at the parking bias.
        e, v = np.linalg.eigh(self.h0)
        rows, cols = linear_sum_assignment(-abs(v)**2)
        order = cols[np.argsort(rows)]
        self.energies, self.dress = e[order], v[:, order]
        # Fix eigenvector phase so ideal selective pulse conventions are stable.
        self.dress = self.dress.astype(complex)
        self.dress *= np.exp(-1j*np.angle(np.diag(self.dress)))[None, :]
        self.collapses = []
        t1 = [theta.t1_c, s.bus_t1, theta.t1_t, *(m.t1 for m in s.extra_modes)]
        t2 = [theta.t2_c, s.bus_t2, theta.t2_t, *(m.t2 for m in s.extra_modes)]
        for a, n, t, u in zip(self.a, self.n, t1, t2):
            if not 0 < u <= 2*t:
                raise ValueError("each mode requires 0 < T2 <= 2 T1")
            rate = max(0, 1/u-1/(2*t))
            self.collapses.extend([a/np.sqrt(t), n*np.sqrt(2*rate)])
        self.dissipator = liouvillian(np.zeros_like(self.h0), self.collapses)
        self._damping_cache = {}

    def coupling_scale(self, omega):
        """g_qc(omega_c)/g_qc: sqrt(omega_c/g_reference_hz), or 1 with a constant coupling."""
        ref = self.settings.g_reference_hz
        return np.sqrt(np.asarray(omega, float)/ref) if ref > 0 else 1.0

    def dressed_f01(self, flux):
        """(control, target) dressed f01 at ``flux`` [Hz], the eigenstates assigned to bare labels as at parking."""
        e, v = np.linalg.eigh(self.hamiltonian(flux))
        rows, cols = linear_sum_assignment(-abs(v)**2)
        e = e[cols[np.argsort(rows)]]
        return e[self.idx(1)] - e[self.idx()], e[self.idx(target=1)] - e[self.idx()]

    def frame_energies(self):
        """(control, target) frequencies whose idle phase the frame removes [Hz]: the true dressed f01, or with
        ``theta.calibrated_frames`` the calibrated clock (true minus ``frame_offset``), so a stale
        calibration leaves 2 pi offset t of phase in every pulse."""
        fc, ft = self.energies[self.idx(1)], self.energies[self.idx(target=1)]
        if getattr(self.theta, "calibrated_frames", False):
            fc, ft = fc - self.theta.frame_offset_c, ft - self.theta.frame_offset_t
        return fc, ft

    def idx(self, control=0, bus=0, target=0):
        return self.index[(control, bus, target) + (0,)*len(self.settings.extra_modes)]

    def hamiltonian(self, flux):
        t, s = self.theta, self.settings
        bus = float(physics.omega_c(flux, t))
        freq = [t.f01_c, bus, t.f01_t,
                *(self.extra_frequency(m,f) for m,f in zip(s.extra_modes,self.extra_fluxes))]
        scale = self.coupling_scale(bus) if s.g_reference_hz > 0 else 1.
        alpha = [t.alpha_c, s.bus_alpha, t.alpha_t, *(m.alpha for m in s.extra_modes)]
        h = np.diag(sum(f*self.basis[:, j] + .5*b*self.basis[:, j]*(self.basis[:, j]-1)
                        for j, (f, b) in enumerate(zip(freq, alpha))))
        edges = [(0, 1, t.g_qc_c*scale if scale != 1. else t.g_qc_c), (1, 2, t.g_qc_t*scale if scale != 1. else t.g_qc_t),
                 (0, 2, s.direct_exchange_hz)]
        edges += [(1, j, m.coupling_to_bus) for j, m in enumerate(s.extra_modes, 3)]
        edges += [(0, j, m.coupling_to_control) for j, m in enumerate(s.extra_modes, 3)]
        edges += [(2, j, m.coupling_to_target) for j, m in enumerate(s.extra_modes, 3)]
        for i, j, g in edges:
            term = self.a[i].T @ self.a[j]
            h += g*(term + term.T)
        return h

    @staticmethod
    def extra_frequency(mode, flux):
        if not mode.fc0:
            return mode.frequency
        return mode.fc0*(np.cos(np.pi*flux)**2+mode.d_asym**2*np.sin(np.pi*flux)**2)**.25

    def state(self, control=0, target=0, dressed=True):
        v = np.eye(self.dim)[:, self.idx(control, target=target)]
        if dressed:
            v = self.dress @ v
        return np.outer(v, v.conj())

    def damping(self, duration):
        key = round(float(duration), 18)
        if key not in self._damping_cache:
            self._damping_cache[key] = expm(self.dissipator*duration)
        return self._damping_cache[key]

    def apply_damping(self, rho, duration):
        shape = rho.shape
        return (rho.reshape(-1, self.dim**2) @ self.damping(duration).T).reshape(shape)

    def idle(self, rho, duration):
        return (rho.reshape(-1, self.dim**2) @ expm(liouvillian(self.h0, self.collapses)*duration).T).reshape(rho.shape)

    def flux_pulse(self, rho, frequency, amplitude, duration, *, rotating=True):
        """Propagate arbitrary density matrices, including channel matrix units.

        Loss acts on all modes throughout the trajectory. Refining max_step and
        dissipation_step controls independent coherent and open-system errors.
        Evaluated by ``flux_pulse_many`` as a one-point batch.
        """
        return self.flux_pulse_many(rho, [frequency], [amplitude], duration, rotating=rotating)[0]

    def _step_table(self, base, tiny, lo, hi, resolution=.25e6, bus0=None):
        """Midpoint step propagators exp(-2 pi i (base + x n_bus) tiny) on a grid of x [Hz].

        Without extra modes the pulse Hamiltonian depends on one scalar, the
        instantaneous bus shift x. ``flux_pulse_many`` takes the nearest grid
        propagator and corrects the residual shift with a symmetric diagonal phase,
        which keeps every step exactly unitary.
        """
        if hi - lo < 1.:
            lo, hi = lo - .5, hi + .5
        # 0.25 MHz spacing: (2 pi tiny dx)^2/8 ~ 1e-10 per substep at the carrier grid.
        grid = np.linspace(lo, hi, int(np.clip(np.ceil((hi - lo)/resolution), 64, 8192)) + 1)
        h = base[None] + grid[:, None, None]*self.n[1][None]
        if self.settings.g_reference_hz > 0 and bus0 is not None:
            # The exchange follows the coupler: g(omega_c) = g sqrt(omega_c/ref), still one scalar per step.
            h = h + (self.coupling_scale(bus0 + grid) - self.coupling_scale(bus0))[:, None, None]*self.v_qc[None]
        e, v = np.linalg.eigh(h)
        return grid, (v*np.exp(-2j*np.pi*e*tiny)[:, None, :]) @ v.conj().swapaxes(-1, -2)

    @staticmethod
    def _ordered_product(steps):
        """U_S ... U_1 for steps[..., k, :, :] = U_(k+1), by pairwise batched products."""
        while steps.shape[-3] > 1:
            count = steps.shape[-3]
            even = count - count % 2
            pairs = steps[..., 1:even:2, :, :] @ steps[..., 0:even:2, :, :]
            steps = np.concatenate((pairs, steps[..., even:, :, :]), axis=-3) if count % 2 else pairs
        return steps[..., 0, :, :]

    def flux_pulse_many(self, rho, frequencies, amplitudes, duration, *, rotating=True):
        """``flux_pulse`` for a batch of (frequency, amplitude) pairs sharing one duration.

        Returns shape (batch, *rho.shape). Every member uses the time grid of the
        highest frequency, so none is integrated more coarsely than
        ``flux_pulse_reference`` integrates it alone.
        """
        s, t = self.settings, self.theta
        frequencies, amplitudes = np.broadcast_arrays(np.asarray(frequencies, float).ravel(),
                                                      np.asarray(amplitudes, float).ravel())
        batch = frequencies.size
        if (not np.isfinite(frequencies).all() or not np.isfinite(amplitudes).all()
                or not np.isfinite(duration) or duration < 0 or (frequencies <= 0).any()):
            raise ValueError("finite pulse parameters, frequency > 0, duration >= 0 required")
        rho = np.asarray(rho)
        if duration == 0:
            return np.broadcast_to(rho, (batch, *rho.shape)).copy()
        if batch > 16:
            # Larger batches overflow cache in the per-block step arrays (measured optimum 16).
            return np.concatenate([self.flux_pulse_many(rho, frequencies[i:i+16], amplitudes[i:i+16], duration,
                                                        rotating=rotating) for i in range(0, batch, 16)])
        if s.extra_modes or s.pulse_solver == "reference":
            return np.stack([self.flux_pulse_reference(rho, f, a, duration, rotating=rotating)
                             for f, a in zip(frequencies, amplitudes)])
        blocks = max(1, int(np.ceil(duration/s.dissipation_step)))
        dt = duration/blocks
        substeps = max(1, int(np.ceil(dt/min(s.max_step, 1/(s.carrier_samples*frequencies.max())))))
        tiny = dt/substeps
        centers = (np.arange(blocks*substeps)+.5)*tiny
        response = s.ac_gain*np.exp(-2j*np.pi*frequencies*s.ac_delay)/(1+2j*np.pi*frequencies*s.ac_lowpass_tau)
        envelope = soft_square(centers, duration)
        if t.dist_lowpass_tau > 0:
            from scipy.signal import lfilter
            decay = np.exp(-tiny/t.dist_lowpass_tau)
            envelope = lfilter([1-decay], [1, -decay], envelope)
        if t.dist_droop_amp and t.dist_droop_tau > 0:
            from scipy.signal import lfilter
            decay = np.exp(-tiny/t.dist_droop_tau)
            envelope -= t.dist_droop_amp*lfilter([1-decay], [1, -decay], envelope)
        frame_hz = t.f01_t
        base = self.h0 - frame_hz*self.total
        bus0 = float(physics.omega_c(self.flux, t))
        # The flux excursion bounds the bus shift, and so the interpolation grid.
        peak = float(np.max(abs(t.ac_flux_per_amp*amplitudes*response))*np.max(abs(envelope)))
        shifts = physics.omega_c(self.flux + np.linspace(-peak, peak, 4097), t) - bus0
        grid, table = self._step_table(base, tiny, float(shifts.min()), float(shifts.max()), bus0=bus0)
        dx = grid[1] - grid[0]
        # With g(omega_c) the step Hamiltonian is not linear in the bus shift; interpolate instead.
        exact_shift = not self.settings.g_reference_hz > 0
        bus_number = np.diag(self.n[1]).astype(float)
        rho = np.broadcast_to(rho, (batch, *rho.shape)).copy()
        lead = (1,)*(rho.ndim - 3)
        for block in range(blocks):
            window = slice(block*substeps, (block+1)*substeps)
            carrier = np.real(response[:, None]*np.exp(2j*np.pi*frequencies[:, None]*centers[None, window]))
            flux = self.flux + t.ac_flux_per_amp*amplitudes[:, None]*envelope[None, window]*carrier
            shift = physics.omega_c(flux, t) - bus0
            index = np.clip(np.rint((shift - grid[0])/dx).astype(int), 0, len(grid) - 1)
            steps = table[index]
            if exact_shift:
                # H(x) is linear in x: exp(-i a N) U(x_k) exp(-i a N), a = pi (x - x_k) tiny, is
                # unitary and agrees with exp(-2 pi i H(x) tiny) to third order in tiny.
                half = np.exp(-1j*np.pi*(shift - grid[index])[..., None]*tiny*bus_number)
                steps = half[..., :, None]*steps*half[..., None, :]
            else:
                position = np.clip((shift - grid[0])/dx, 0, len(grid) - 1 - 1e-9)
                lower = position.astype(int)
                weight = (position - lower)[..., None, None]
                steps = (1 - weight)*table[lower] + weight*table[lower + 1]
            u = self._ordered_product(steps).reshape(batch, *lead, self.dim, self.dim)
            rho = self.apply_damping(rho, dt/2)
            rho = u @ rho @ u.conj().swapaxes(-1, -2)
            rho = self.apply_damping(rho, dt/2)
        if rotating:
            # Remove local idle phase, retaining static ZZ and all exchange.
            fc, ft = self.frame_energies()
            local = fc*self.basis[:, 0] + ft*self.basis[:, 2] + frame_hz*self.basis[:, 1]
            if self.basis.shape[1] > 3:
                local += frame_hz*self.basis[:, 3:].sum(axis=1)
            phase = np.exp(2j*np.pi*(local - frame_hz*self.basis.sum(axis=1))*duration)
            rho = conjugate(self.dress.conj().T, rho)
            rho = phase[:, None]*rho*phase.conj()[None, :]
        return rho

    def flux_pulse_reference(self, rho, frequency, amplitude, duration, *, rotating=True):
        """The original per-substep eigendecomposition solver, kept to validate the fast path."""
        s, t = self.settings, self.theta
        if not np.isfinite([frequency, amplitude, duration]).all() or duration < 0 or frequency <= 0:
            raise ValueError("finite pulse parameters, frequency > 0, duration >= 0 required")
        if duration == 0:
            return rho.copy()
        blocks = max(1, int(np.ceil(duration/s.dissipation_step)))
        dt = duration/blocks
        substeps = max(1, int(np.ceil(dt/min(s.max_step, 1/(s.carrier_samples*frequency)))))
        tiny = dt/substeps
        centers = (np.arange(blocks*substeps)+.5)*tiny
        response = s.ac_gain*np.exp(-2j*np.pi*frequency*s.ac_delay)/(1+2j*np.pi*frequency*s.ac_lowpass_tau)
        envelope = soft_square(centers, duration)
        # Apply existing calibrated envelope distortion, if provided, causally.
        if t.dist_lowpass_tau > 0:
            from scipy.signal import lfilter
            decay = np.exp(-tiny/t.dist_lowpass_tau)
            envelope = lfilter([1-decay], [1, -decay], envelope)
        if t.dist_droop_amp and t.dist_droop_tau > 0:
            from scipy.signal import lfilter
            decay = np.exp(-tiny/t.dist_droop_tau)
            envelope -= t.dist_droop_amp*lfilter([1-decay], [1, -decay], envelope)
        flux = self.flux + t.ac_flux_per_amp*amplitude*envelope*np.real(response*np.exp(2j*np.pi*frequency*centers))
        # Common excitation frame improves conditioning and cancels at readout.
        frame_hz = t.f01_t
        base = self.h0 - frame_hz*self.total
        bus0 = float(physics.omega_c(self.flux, t))
        bus_values = physics.omega_c(flux, t)
        extra_shifts = []
        for mode, bias in zip(s.extra_modes, self.extra_fluxes):
            pickup = mode.ac_flux_per_amp*np.exp(-2j*np.pi*frequency*mode.ac_delay)/(1+2j*np.pi*frequency*mode.ac_lowpass_tau)
            f = bias + amplitude*envelope*np.real(pickup*np.exp(2j*np.pi*frequency*centers))
            extra_shifts.append(np.broadcast_to(self.extra_frequency(mode,f)-self.extra_frequency(mode,bias), centers.shape))
        for block in range(blocks):
            rho = self.apply_damping(rho, dt/2)
            h = base[None] + (bus_values[block*substeps:(block+1)*substeps]-bus0)[:, None, None]*self.n[1]
            if s.g_reference_hz > 0:
                h = h + (self.coupling_scale(bus_values[block*substeps:(block+1)*substeps])
                         - self.coupling_scale(bus0))[:, None, None]*self.v_qc
            for j, shift in enumerate(extra_shifts,3):
                h += shift[block*substeps:(block+1)*substeps,None,None]*self.n[j]
            e, v = np.linalg.eigh(h)
            us = (v*np.exp(-2j*np.pi*e*tiny)[:, None, :]) @ v.swapaxes(-1, -2)
            u = np.eye(self.dim, dtype=complex)
            for step in us:
                u = step @ u
            rho = conjugate(u, rho)
            rho = self.apply_damping(rho, dt/2)
        if rotating:
            # Remove local idle phase, retaining static ZZ and all exchange.
            fc, ft = self.frame_energies()
            local = fc*self.basis[:, 0] + ft*self.basis[:, 2] + frame_hz*self.basis[:, 1]
            if self.basis.shape[1] > 3:
                local += frame_hz*self.basis[:, 3:].sum(axis=1)
            phase = np.exp(2j*np.pi*(local-frame_hz*self.basis.sum(axis=1))*duration)
            rho = conjugate(self.dress.conj().T, rho)
            rho = phase[:, None]*rho*phase.conj()[None, :]
        return rho

    def spectroscopy(self, frequency, *, drive=0, transition=1, duration=None, amplitude=None):
        s = self.settings
        if drive not in (0, 1) or transition not in (1, 2):
            raise ValueError("drive 0/1 and transition 1/2 required")
        duration = s.pulse_duration if duration is None else duration
        amplitude = s.pulse_amplitude if amplitude is None else amplitude
        if not 0 <= amplitude <= .1 or duration <= 0:
            raise ValueError("spectroscopy requires positive duration and amplitude in [0,.1]")
        q = 0 if drive == 0 else 2
        rho = self.state(control=int(transition == 2 and q == 0), target=int(transition == 2 and q == 2))
        lowering = self.a[q].astype(complex) + s.xy_crosstalk*self.a[2-q]
        drive_h = (lowering+lowering.conj().T)*amplitude*s.rabi_hz_per_amp/2
        h = self.h0 - frequency*self.total
        segments = s.spectroscopy_segments if duration <= 6e-6 else 1
        envelopes = soft_square((np.arange(segments)+.5)*duration/segments, duration) if segments > 1 else [1.]
        for env in envelopes:
            rho = (expm(liouvillian(h + env*drive_h, self.collapses)*duration/segments) @ rho.ravel()).reshape(self.dim, self.dim)
        # Local qubit occupation: a dark bus has no acquisition channel.
        return self.marginals(rho)

    def spectroscopy_many(self, frequencies, *, drive=0, transition=1, duration=None, amplitude=None, segments=48):
        """``spectroscopy`` for many drive frequencies in the addressed transmon's dressed ladder.

        Energies, drive matrix elements and loss operators are those of the full cell,
        projected onto the dressed |0>, |1>, |2> of the driven qubit, so line positions and
        the coupler anticrossing are exact; transitions out of that ladder enter at
        (g/Delta)^2. The 9x9 Liouvillian makes a finely segmented envelope affordable.
        Returns (control, target) marginals, each (len(frequencies), 3).
        """
        s = self.settings
        if drive not in (0, 1) or transition not in (1, 2):
            raise ValueError("drive 0/1 and transition 1/2 required")
        duration = s.pulse_duration if duration is None else duration
        amplitude = s.pulse_amplitude if amplitude is None else amplitude
        if not 0 <= amplitude <= .1 or duration <= 0:
            raise ValueError("spectroscopy requires positive duration and amplitude in [0,.1]")
        frequencies = np.atleast_1d(np.asarray(frequencies, float))
        q = 0 if drive == 0 else 2
        # The bus partners keep the hybridized |2>/|1,1_bus> pair near an anticrossing.
        labels = ([self.idx(), self.idx(1), self.idx(2), self.idx(bus=1), self.idx(1, bus=1)] if q == 0 else
                  [self.idx(), self.idx(target=1), self.idx(target=2), self.idx(bus=1), self.idx(bus=1, target=1)])
        ladder = self.dress[:, labels]
        lowering = self.a[q].astype(complex) + s.xy_crosstalk*self.a[2-q]
        drive_h = ladder.conj().T @ ((lowering+lowering.conj().T)*amplitude*s.rabi_hz_per_amp/2) @ ladder
        collapses = [ladder.conj().T @ c @ ladder for c in self.collapses]
        n = len(labels)
        energies = self.energies[labels]
        excitations = self.basis[labels, :3].sum(axis=1).astype(float)
        # L(f, env) = L(0, env) + f*frame: the drive frame only adds a diagonal term.
        eye = np.eye(n)
        frame = 2j*np.pi*(np.kron(np.diag(excitations), eye) - np.kron(eye, np.diag(excitations))).diagonal()
        envelopes = soft_square((np.arange(segments)+.5)*duration/segments, duration)
        rho = np.zeros((len(frequencies), n, n), complex)
        rho[:, transition-1, transition-1] = 1
        vec = rho.reshape(len(frequencies), n*n)
        step = duration/segments
        for env in envelopes:
            generator = liouvillian(np.diag(energies) + env*drive_h, collapses)
            batch = np.broadcast_to(generator, (len(frequencies), n*n, n*n)).copy()
            batch[:, np.arange(n*n), np.arange(n*n)] += frequencies[:, None]*frame[None]
            vec = np.einsum('bij,bj->bi', expm(batch*step), vec)
        full = ladder[None] @ vec.reshape(-1, n, n) @ ladder.conj().T[None]
        return self.marginals(full)

    def marginals(self, rho):
        p = np.diagonal(rho, axis1=-2, axis2=-1).real
        if p.min() < -1e-7 or not np.allclose(p.sum(axis=-1), 1, atol=1e-6):
            raise FloatingPointError("multimode propagation lost positivity or trace")
        result = []
        for q in (0, 2):
            # Third output is leakage >=2 when testing a higher cutoff.
            result.append(np.stack([p[..., self.basis[:, q] == j].sum(-1) for j in (0, 1)] +
                                   [p[..., self.basis[:, q] >= 2].sum(-1)], axis=-1))
        return tuple(result)

    def pair_density(self, rho):
        """Trace hidden modes, retaining both local qutrits for analysis pulses."""
        result = np.zeros((*rho.shape[:-2], 9, 9), complex)
        for i, a in enumerate(self.basis):
            for j, b in enumerate(self.basis):
                if a[0] < 3 and a[2] < 3 and b[0] < 3 and b[2] < 3 and np.array_equal(a[[1, *range(3, len(a))]], b[[1, *range(3, len(b))]]):
                    result[..., 3*a[0]+a[2], 3*b[0]+b[2]] += rho[..., i, j]
        return result


def rotation(angle, phase=0.):
    """Ideal calibrated selective 01 pulse; |2> is untouched."""
    u = np.eye(3, dtype=complex)
    u[:2, :2] = [[np.cos(angle/2), -1j*np.exp(-1j*phase)*np.sin(angle/2)],
                  [-1j*np.exp(1j*phase)*np.sin(angle/2), np.cos(angle/2)]]
    return u


def pair_populations(rho):
    p = np.diagonal(rho, axis1=-2, axis2=-1).real.reshape(*rho.shape[:-2], 3, 3)
    return p.sum(-1), p.sum(-2)


def noise_branches(theta, settings=None):
    """[(weight, theta)] over the charge-parity branches and the quasi-static flux offsets of ``theta``."""
    from dataclasses import replace
    from chipcal.sim.realism import parity_branches
    branches = parity_branches(theta)
    s = settings or getattr(theta, "multimode", None) or MultimodeSettings()
    sigma = float(getattr(s, "flux_noise_phi0", 0.0))
    if sigma > 0:
        # A flux offset d moves the coupler as a sweet-current shift of -d * i_period.
        branches = [(w / 2, replace(th, i_sweet=th.i_sweet - sign * sigma * th.i_period))
                    for w, th in branches for sign in (-1.0, 1.0)]
    return branches


def gate_fidelity(config, theta, settings=None):
    """Average computational CZ fidelity from the actual trace-decreasing channel.

    Uses all 16 logical matrix units, including off-diagonal coherences and
    leakage, with no second global leakage penalty or planted optimum. With
    charge-parity splitting or quasi-static flux noise the channel is the mixture
    of the noise branches, so the fidelity and diagnostics are their average.
    """
    if not config.is_complete():
        return 0., {"incomplete": True}
    branches = noise_branches(theta, settings)
    if len(branches) > 1:
        results = [(w, _gate_fidelity_one(config, th, settings)) for w, th in branches]
        value = float(sum(w*r[0] for w, r in results))
        diag = dict(results[0][1][1])
        for key in ("leakage", "seepage", "computational_survival"):
            diag[key] = float(sum(w*r[1][key] for w, r in results))
        diag["noise_branches"] = len(results)
        return value, diag
    return _gate_fidelity_one(config, theta, settings)


def _gate_fidelity_one(config, theta, settings=None):
    cell = Cell(theta, config.parking_current, settings)
    comp = [cell.idx(), cell.idx(target=1), cell.idx(1), cell.idx(1, target=1)]
    vectors = cell.dress[:, comp]
    inputs = np.array([np.outer(vectors[:, i], vectors[:, j].conj()) for i in range(4) for j in range(4)])
    # Leakage is observable qubit occupation >=2. Hidden bus excitations must
    # be traced out, not treated as directly measured leakage or postselection.
    leak_mask = (cell.basis[:,0] >= 2) | (cell.basis[:,2] >= 2)
    outside = conjugate(cell.dress, np.diag(leak_mask.astype(float)))
    inputs = np.concatenate((inputs, (outside/leak_mask.sum())[None]), axis=0)
    output = cell.flux_pulse(inputs, config.cz_pulse_frequency, config.cz_pulse_amplitude, config.cz_pulse_duration)
    pair = cell.pair_density(output)
    pair_comp = [0,1,3,4]
    channel = pair[:16,pair_comp,:][:,:,pair_comp].reshape(4,4,4,4)
    zc, zt = config.control_local_phase or 0., config.target_local_phase or 0.
    # A committed local phase is an ESTIMATE OF THE TRUE ACCUMULATED PHASE, so the
    # target must be dressed with exp(+i z): fidelity peaks when the committed value
    # equals the phase the pulse really accumulates. That is the convention
    # observation_analysis reports (positive fringe shift). With exp(-i z) the
    # fidelity would peak at the negation, so committing a correctly measured phase
    # would de-calibrate the gate (tests/test_local_phase_convention.py).
    target = np.array([1, np.exp(1j*zt), np.exp(1j*zc), -np.exp(1j*(zc+zt))])
    entanglement = sum(target[i].conjugate()*target[j]*channel[i,j,i,j] for i in range(4) for j in range(4)).real
    survival = sum(np.trace(channel[i,i]).real for i in range(4))
    value = float(np.clip((entanglement+survival)/20, 0, 1))
    return value, {"model": "multimode_lindblad_v1", "leakage": float(1-survival/4),
                   "seepage": float(np.trace(pair[-1][np.ix_(pair_comp,pair_comp)]).real),
                   "computational_survival": float(survival/4), "fidelity_definition": "unconditional_average_CZ_channel_fidelity"}
