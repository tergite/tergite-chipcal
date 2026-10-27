"""Hidden parameters theta of one qubit-coupler-qubit cell.

- The coupler is an asymmetric-SQUID transmon:
  omega_c = f_c0 * [cos^2(pi*Phi) + d^2 sin^2(pi*Phi)]^(1/4) (Koch et al. 2007).
- Bias currents are in amperes.
- The CZ transition is the |11> <-> |20> path at |f01_target - f12_control|.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

@dataclass
class DeviceParams:
    """Static latent theta. Frequencies in Hz, times in s, currents in A.

    Flux is in units of Phi_0; "amp" means the normalized AC pulse amplitude
    (the 0..1 DAC scale swept by cz_parametrization).
    """

    # Qubits (control = higher-frequency qubit, target = lower)
    f01_c: float
    f01_t: float
    alpha_c: float
    alpha_t: float
    t1_c: float
    t1_t: float
    t2_c: float
    t2_t: float

    # Coupler: omega_c(Phi) = fc0 * [cos^2(pi Phi) + d^2 sin^2(pi Phi)]^(1/4)
    fc0: float
    d_asym: float

    # Flux-line transfers
    i_sweet: float          # DC current at Phi = 0 (coupler sweet spot) [A]
    i_period: float         # DC current per flux quantum [A / Phi_0]
    ac_flux_per_amp: float  # AC flux per unit normalized amplitude [Phi_0]

    # Qubit-coupler couplings
    g_qc_c: float
    g_qc_t: float

    # Drive-induced shift of the transition (chevron curvature): Hz per amp^2
    kappa_shift: float
    # Drive-induced single-qubit (Stark) phase rates: rad/s per amp^2
    stark_c: float
    stark_t: float

    # Control-line flux-pulse distortion (LTI), all default to an ideal line:
    #   finite bandwidth (one-pole low-pass that rounds edges) + bias-tee droop
    #   (slow exponential that sags the plateau and leaves an undershoot tail).
    # This warps the *commanded* flux envelope, so it degrades the true gate and
    # skews every measured plot (the flux-pulse predistortion problem).
    dist_lowpass_tau: float = 0.0   # s; one-pole low-pass time constant (0 = off)
    dist_droop_amp: float = 0.0     # fraction of plateau lost to slow droop [0, 0.6]
    dist_droop_tau: float = 0.0     # s; droop (bias-tee) time constant

    # TLS defects: array (n, 3) of (frequency [Hz], strength [0..1], width [Hz])
    tls: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    # Readout confusion, C[i, j] = P(measure i | prepared j), columns sum to 1
    confusion_c: np.ndarray = field(default_factory=lambda: np.eye(3))
    confusion_t: np.ndarray = field(default_factory=lambda: np.eye(3))

    family: str = "nominal"
    # Opt-in multilevel physical settings, shared by every calibration node.
    multimode: object | None = None
    # Reduced pulse waveform and integration settings (SI units).
    pulse_wait_s: float = 4e-9
    pulse_rise_fall_s: float = 16e-9
    pulse_max_step_s: float = 0.30e-9
    # Realism (chipcal.sim.realism; all zero/False on the default path). Charge-parity splitting of each
    # qubit's f01 / f12 [Hz, full separation], and the offset of its true dressed f01 from the calibrated
    # frame [Hz]. With ``calibrated_frames`` the CZ and Ramsey frames run at that calibrated clock.
    parity_c: float = 0.0
    parity_t: float = 0.0
    parity12_c: float = 0.0
    parity12_t: float = 0.0
    frame_offset_c: float = 0.0
    frame_offset_t: float = 0.0
    calibrated_frames: bool = False

    @property
    def f12_c(self) -> float:
        return self.f01_c + self.alpha_c

    @property
    def f12_t(self) -> float:
        return self.f01_t + self.alpha_t

    @property
    def delta_via20(self) -> float:
        """Bare |11> <-> |20> transition frequency (via_20 path)."""
        return abs(self.f01_t - self.f12_c)


# Scalar fabrication / coherence parameters that may be overridden after a seed
# draw. Array fields (tls, confusion) and the family label are excluded.
TUNABLE_FIELDS = (
    "f01_c", "f01_t", "alpha_c", "alpha_t",
    "t1_c", "t1_t", "t2_c", "t2_t",
    "fc0", "d_asym", "i_sweet", "i_period", "ac_flux_per_amp",
    "g_qc_c", "g_qc_t", "kappa_shift", "stark_c", "stark_t",
    "dist_lowpass_tau", "dist_droop_amp", "dist_droop_tau",
)
