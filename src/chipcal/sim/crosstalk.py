"""DC, AC-flux and microwave transfer are distinct physical quantities.

Rows are victim modes, columns are source lines. DC has units Phi0/A; AC has
units Phi0/DAC. Complex microwave entries are local Rabi amplitude ratios.
No DC matrix is silently reused as an AC matrix or a frequency-shift matrix.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FluxTransfer:
    dc: np.ndarray
    ac: np.ndarray
    delay_s: np.ndarray
    lowpass_tau_s: np.ndarray
    offset: np.ndarray

    def __post_init__(self):
        dc, ac = np.asarray(self.dc, float), np.asarray(self.ac, complex)
        delay, tau, offset = map(lambda x: np.asarray(x, float), (self.delay_s, self.lowpass_tau_s, self.offset))
        if dc.ndim != 2 or dc.shape[0] != dc.shape[1] or ac.shape != dc.shape or delay.shape != dc.shape or tau.shape != dc.shape or offset.shape != dc.shape[:1]:
            raise ValueError("square DC/AC/delay/tau matrices and one offset per victim required")
        if not all(np.isfinite(x).all() for x in (dc, ac, delay, tau, offset)) or np.any(delay < 0) or np.any(tau < 0) or np.any(np.diag(dc) == 0):
            raise ValueError("finite transfers, nonzero DC diagonal and causal delays/time constants required")
        for name, value in zip(("dc", "ac", "delay_s", "lowpass_tau_s", "offset"), (dc, ac, delay, tau, offset)):
            value = value.copy(); value.flags.writeable = False
            object.__setattr__(self, name, value)

    @property
    def dc_fraction(self):
        """A_ij/A_jj: flux in victim per flux generated in the source's loop."""
        return self.dc/np.diag(self.dc)[None, :]

    def response(self, frequency_hz):
        frequency = np.asarray(frequency_hz, float)
        if not np.isfinite(frequency).all():
            raise ValueError("finite frequencies required")
        omega = 2*np.pi*frequency[..., None, None]
        return self.ac*np.exp(-1j*omega*self.delay_s)/(1+1j*omega*self.lowpass_tau_s)

    def dc_flux(self, currents):
        return np.asarray(currents) @ self.dc.T + self.offset

    def flux(self, time_s, currents, amplitudes, frequencies_hz, phases_rad=None):
        """Steady sinusoidal response, including interference from all sources.

        This helper is not a transient filter for arbitrary pulse edges; the
        Cell solver separately resolves its envelope and the addressed carrier.
        """
        amps, freq = np.asarray(amplitudes), np.asarray(frequencies_hz)
        n = self.dc.shape[1]
        if amps.shape != (n,) or freq.shape != (n,):
            raise ValueError("one amplitude and carrier frequency per source required")
        phases = np.zeros(n) if phases_rad is None else np.asarray(phases_rad)
        # H(victim, source) evaluated at each source's own frequency.
        h = np.column_stack([self.response(f)[..., j] for j, f in enumerate(freq)])
        carriers = amps*np.exp(1j*(2*np.pi*np.asarray(time_s)[..., None]*freq+phases))
        return self.dc_flux(currents) + np.real(carriers @ h.T)


def compensate_dc(transfer, desired_flux):
    """Solve physical flux targets; do not invert an elementwise frequency pull."""
    return np.linalg.solve(transfer.dc, np.asarray(desired_flux)-transfer.offset)


def apply_xy_crosstalk(rabi_amplitudes, transfer):
    """Complex Rabi amplitudes add before populations are computed (20 log10 dB)."""
    return np.asarray(rabi_amplitudes, complex) @ np.asarray(transfer, complex).T
