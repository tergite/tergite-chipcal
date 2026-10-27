from __future__ import annotations
import numpy as np
from scipy.special import j1
from chipcal.sim.params import DeviceParams
TWO_PI = 2.0 * np.pi

def flux_dc(i_dc, theta: DeviceParams):
    """DC current -> coupler flux in Phi_0 units."""
    return (np.asarray(i_dc, dtype=float) - theta.i_sweet) / theta.i_period


def omega_c(flux, theta: DeviceParams):
    """Tunable coupler frequency [Hz] of an asymmetric SQUID with asymmetry d:
    fc0 * [cos^2(pi Phi) + d^2 sin^2(pi Phi)]^(1/4)."""
    phi = np.asarray(flux, dtype=float)
    c2 = np.cos(np.pi * phi) ** 2
    s2 = np.sin(np.pi * phi) ** 2
    return theta.fc0 * (c2 + theta.d_asym**2 * s2) ** 0.25


def coupler_j(flux, theta: DeviceParams):
    """Effective qubit-qubit exchange J(Phi) [Hz] via virtual coupler excitation."""
    wc = omega_c(flux, theta)
    return 0.5 * theta.g_qc_c * theta.g_qc_t * (
        1.0 / (theta.f01_c - wc) + 1.0 / (theta.f01_t - wc)
    )


def dj_dflux(flux, theta: DeviceParams, h: float = 1e-4):
    """Numerical dJ/dPhi [Hz / Phi_0]; the parametric-drive lever arm."""
    return (coupler_j(np.asarray(flux) + h, theta) - coupler_j(np.asarray(flux) - h, theta)) / (2 * h)


def g_eff(amp, i_dc, theta: DeviceParams):
    """Effective |11> <-> |20> coupling [Hz] at normalized drive amplitude `amp`.

    First-order sideband: |dJ/dPhi| * (delta_Phi / 2), with Bessel-type
    saturation 2*J1(x)/x (-> 1 as x -> 0) in x = pi * delta_Phi.  At the sweet
    spot dJ/dPhi ~ 0: the bias is dead and the parametrization map is flat.
    """
    amp = np.asarray(amp, dtype=float)
    phi = flux_dc(i_dc, theta)
    slope = np.abs(dj_dflux(phi, theta))
    dphi = theta.ac_flux_per_amp * amp
    x = np.pi * dphi
    x_safe = np.where(np.abs(x) < 1e-9, 1.0, x)
    sat = np.where(np.abs(x) < 1e-9, 1.0, 2.0 * j1(x_safe) / x_safe)
    return slope * (dphi / 2.0) * sat


def transition_frequency(amp, i_dc, theta: DeviceParams):
    """Dressed |11>-|20> transition [Hz]: bare via_20 splitting + static-J
    dressing at this bias + drive-induced quadratic shift (chevron curvature)."""
    amp = np.asarray(amp, dtype=float)
    phi = flux_dc(i_dc, theta)
    return theta.delta_via20 + 2.0 * coupler_j(phi, theta) + theta.kappa_shift * amp**2


def amplitude_11(g_hz, delta_hz, t):
    """Two-level |11> amplitude after drive time t (eq. 5 generalized).

    H/hbar = [[0, g], [g, delta]] in angular units; returns the complex c11.
    |c11|^2 is the return probability, 1-|c11|^2 the |20> transfer, and
    arg(c11) the acquired conditional phase (pi at a full on-resonance cycle).
    """
    g = TWO_PI * np.asarray(g_hz, dtype=float)
    d = TWO_PI * np.asarray(delta_hz, dtype=float)
    t = np.asarray(t, dtype=float)
    omega = np.sqrt(g**2 + (d / 2.0) ** 2)
    omega_safe = np.where(omega == 0.0, 1.0, omega)
    c = np.cos(omega * t) + 1j * (d / (2.0 * omega_safe)) * np.sin(omega * t)
    return np.exp(-1j * d * t / 2.0) * np.where(omega == 0.0, 1.0 + 0j, c)


def transfer_probability(g_hz, delta_hz, t):
    """P(|11> -> |20>) of eq. (5)."""
    return 1.0 - np.abs(amplitude_11(g_hz, delta_hz, t)) ** 2


def conditional_phase(g_hz, delta_hz, t):
    """Conditional phase [rad] acquired by |11> (target: pi)."""
    return np.angle(amplitude_11(g_hz, delta_hz, t))


def local_phases(amp, t, theta: DeviceParams):
    """Drive-induced single-qubit phases (control, target) [rad] over the gate."""
    amp = np.asarray(amp, dtype=float)
    return theta.stark_c * amp**2 * t, theta.stark_t * amp**2 * t


def tls_penalty(i_dc, theta: DeviceParams):
    """Extra gate error from TLS near the coupler frequency at this bias."""
    if theta.tls.shape[0] == 0:
        return np.zeros_like(np.asarray(i_dc, dtype=float))
    wc = np.asarray(omega_c(flux_dc(i_dc, theta), theta))[..., None]
    f, s, w = theta.tls[:, 0], theta.tls[:, 1], theta.tls[:, 2]
    lor = s * w**2 / ((wc - f) ** 2 + w**2)
    return lor.sum(axis=-1)


def relax_populations(p, t, t1):
    """First-order amplitude damping of qutrit populations over time t.

    p has the state axis last: (..., 3).  |2> decays to |1> at 2/T1, |1> to
    |0> at 1/T1 (harmonic-oscillator scaling).
    """
    p = np.asarray(p, dtype=float).copy()
    e1 = np.exp(-t / t1)
    e2 = np.exp(-2.0 * t / t1)
    p2 = p[..., 2] * e2
    from2 = p[..., 2] - p2
    p1 = p[..., 1] * e1 + from2
    p0 = 1.0 - p1 - p2
    return np.stack([p0, p1, p2], axis=-1)


def wrap_angle(phi):
    """Wrap to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(phi, dtype=float)))
