# What the simulator models

## Summary

Each tunable coupler and its two neighbouring fixed-frequency transmons form a cell. The cell is simulated
as three coupled anharmonic (Kerr) oscillators, with frequencies in hertz. The coupler frequency follows the
asymmetric-SQUID law of its flux. DC bias currents set that flux through a hidden, slightly cross-coupled
current-to-flux matrix.

The CZ gate is the parametric scheme used with flux-tunable couplers: the coupler flux is modulated near the
$\lvert 11\rangle \leftrightarrow \lvert 20\rangle$ energy difference, under a soft-square envelope (McKay et al. 2016;
Caldwell et al. 2018; Sete et al. 2021). The resulting
dynamics are integrated in time with every side effect included: dressing shifts, conditional and local phases,
leakage, other sideband transitions and coupler population. Energy relaxation and dephasing of all three modes
enter through a Lindblad master equation.

Each committed gate is scored by a final interleaved randomized benchmark that the environment runs when the
episode ends. The benchmark is derived from the gate's full quantum channel, so its fit estimates the gate's
leakage-aware average fidelity to a CZ, dressed with the committed single-qubit phase corrections. The
agent sees only finite-shot, readout-corrupted measurements: spectroscopy, chevrons, Ramsey-type phase scans,
and its own benchmarks.

Before any coupler work, the agent must discover each qubit's readout resonator, $f_{01}$ and $f_{12}$. Discovery
uses a phenomenological line model with local-oscillator band limits. Realism adds slow and non-Markovian effects
on top: charge-parity splitting, telegraph and drift noise of the qubit frequencies, calibrated-frame errors,
dressing by neighbouring couplers, quasi-static flux noise, and imperfect three-state readout.

## The model in detail

**Hamiltonian.** For the qubit–coupler–qubit cell (Yan et al. 2018), each mode a transmon truncated to a Kerr
oscillator (Koch et al. 2007; Krantz et al. 2019), with frequencies in hertz:

```math
\frac{H}{h} = \sum_{j} \left[ \omega_j\, \hat n_j + \frac{\alpha_j}{2}\, \hat n_j \left(\hat n_j - 1\right) \right]
+ \sum_{\langle i,j \rangle} g_{ij} \left( \hat a_i^\dagger \hat a_j + \hat a_j^\dagger \hat a_i \right)
```

- $\omega_j$ is the frequency and $\alpha_j$ the anharmonicity of mode $j$; $g_{ij}$ couples each qubit to the coupler.
- Three levels per mode, at most two excitations in total.
- The excitation-conserving (rotating-wave) exchange plus a diagonal flux modulation conserves excitation
  number, so this truncation is exact for the CZ.

**Coupler.** Its frequency follows the asymmetric-SQUID law (Koch et al. 2007; Krantz et al. 2019):

```math
\omega_c(\Phi) = \omega_{\max} \left[ \cos^2(\pi \Phi) + d^2 \sin^2(\pi \Phi) \right]^{1/4},
\qquad \boldsymbol{\Phi} = A\, \mathbf{I}
```

- $\Phi$ is in units of the flux quantum $\Phi_0$, $d$ is the SQUID asymmetry, and $\mathbf{I}$ the vector of bias
  currents.
- $A$ is the hidden DC current-to-flux matrix: its diagonal is 1 / (current per flux quantum), and its
  nearest-neighbour terms are 0.25–1 % of that.
- The flux pulse sees a separate AC transfer.

**CZ pulse.**
- A flux modulation at the commanded frequency and amplitude under a soft-square envelope, integrated with
  128 time samples per carrier period and Strang-split dissipation.
- A fast batched solver is used for every measurement. Tests check it against the per-step reference
  solver; it agrees to $\sim 10^{-5}$ in the density matrix and stays exactly trace-preserving.

**Gate fidelity** (what the benchmark measures; also reported unscored as the true fidelity). Over the dressed
computational subspace ($d = 4$), the average gate fidelity of a possibly trace-decreasing channel (Pedersen et
al. 2007; Wood & Gambetta 2018):

```math
F = \frac{\sum_k \left\lvert \operatorname{Tr}\!\left( U^\dagger K_k \right) \right\rvert^2 + \operatorname{Tr} \mathcal{E}(\mathbf{1})}{d\,(d + 1)}
= \frac{\sum_k \left\lvert \operatorname{Tr}\!\left( U^\dagger K_k \right) \right\rvert^2 + \operatorname{Tr} \mathcal{E}(\mathbf{1})}{20}
```

- $\mathcal{E}$ is the gate channel restricted to the computational subspace, with Kraus operators $K_k$; $U$ is the
  CZ dressed with the committed local phases.
- Leakage of either qubit into $\lvert 2\rangle$ makes $\mathcal{E}$ trace-decreasing and counts as error.
- Coupler excitations left at the end of the gate are traced out.

**Measurements.**
- Every node returns multinomial three-state counts per qubit. Each qubit's readout has its own
  column-stochastic assignment matrix; a dispersive IQ model with a fixed classifier sits on top.
- Spectroscopy nodes solve the driven, dissipative cell. Chevron and parametrization nodes run the same flux
  pulse as the gate.
- Ramsey-type nodes read out at the qubit's calibrated frame.

**Benchmark.** Interleaved randomized benchmarking (IRB; Magesan et al. 2012), modelled as a Clifford twirl of
the true gate channel (not a compiled sequence). It includes leakage and seepage (Wood & Gambetta 2018).

It measures two survival curves: the probability that a random sequence of $m$ two-qubit Cliffords returns the
qubits to their initial state.
- Reference: random Clifford sequences. Each Clifford is compiled with on average 1.5 of the committed CZs
  (Barends et al. 2014), so a bad CZ also degrades the reference curve.
- Interleaved: the same sequences with the committed CZ inserted after every Clifford.

Each curve is fitted with $A\,p^m + B$, which gives the decays $p_{\mathrm{ref}}$ and $p_{\mathrm{int}}$; the extra
decay of the interleaved curve is the error of the CZ. A separate one-gate probe measures $S$, the probability
that the CZ keeps a computational state inside the computational subspace ($S = 1$ for a gate that never leaks).

The benchmark reports three numbers:

| Field | Symbol | Meaning |
|---|---|---|
| `F_hat` | $\hat F$ | The estimated average gate fidelity of the committed CZ, $\hat F = \left(3\,p_{\mathrm{int}}/p_{\mathrm{ref}} + S\right)/4$. Without leakage ($S = 1$) this is the standard two-qubit IRB estimate $1 - \tfrac{3}{4}\left(1 - p_{\mathrm{int}}/p_{\mathrm{ref}}\right)$. The $\hat F$ of the final benchmark is the score. |
| `F_sigma` | $\sigma_F$ | The uncertainty of $\hat F$: the larger of the error propagated from the two fits and the spread of $\hat F$ when the random sequences are bootstrap-resampled, widened by the Birge ratio $\sqrt{\max(1, \chi^2_\nu)}$ when a single exponential does not fit the curves exactly. |
| `F_lcb` | $F_{\mathrm{lcb}}$ | A conservative lower bound, $F_{\mathrm{lcb}} = \hat F - 2\,\sigma_F$: for Gaussian errors the true fidelity lies above it with about 97.7 % confidence. The agent's own benchmarks pass when $F_{\mathrm{lcb}} \ge 0.95$. |

The fit is marked unreliable when the reference decay is too fast to identify ($p_{\mathrm{ref}} \le 0.5$) or the
uncertainty is too large ($\sigma_F \ge 0.1$). Then no $\hat F$ is reported, $F_{\mathrm{lcb}} = 0$, and a final
benchmark of that kind scores 0.25.

**Discovery.**
- Readout is a shared feedline whose resonators appear as unlabelled $\lvert S_{21} \rvert$ notches. Two-tone
  spectroscopy answers through the qubit's committed resonator, so assigning resonators to qubits is inferable
  from addressed drive responses. A drive tone leaks 10 % into the other qubits' lines.
- Lines power-broaden and Stark-shift with the tone amplitude.
- Design frequencies deviate from the true ones by $\sigma = 100$ MHz, truncated at $\pm 300$ MHz.

## Realism (always on in the benchmark tiers)

| Effect | Model |
|---|---|
| Charge-parity splitting | The exact charge-basis transmon spectrum of each qubit's $E_C$ and $E_J/E_C$ gives the split, times $\lvert \cos 2\pi n_g \rvert$. The offset charge $n_g$ is redrawn every 300 s. Every acquisition and the gate channel average the two parity branches equally. |
| Frequency noise | Seeded linear drift, a random walk, and 0–2 telegraph defects per qubit (1–20 kHz jumps), all on wall-clock time. |
| Calibrated frames | Gates, Ramseys and RB run at a calibrated clock with a 2 kHz error. A bias move costs a 60 s Ramsey retune of both adjacent qubits. Drift between retunes appears as frame error in every phase-sensitive measurement and in the gate. |
| Neighbour dressing | A qubit's $f_{01}$ and $f_{12}$ are shifted by every coupler it touches, not only the one being calibrated. |
| Quasi-static flux noise | 5–11 $\mu\Phi_0$ rms per coupler loop. The gate channel and the benchmark average over $\pm\sigma$ flux offsets (two-point Gauss–Hermite, exact to second order). It changes CZ fidelity by $\lesssim 3 \times 10^{-3}$. |
| Coherence and readout spread | $T_1$, $T_2$ and three-state assignment drawn per qubit from published ranges (docs/parameters.md). |

## Simplifications

- Exchange coupling in the rotating-wave approximation; Bloch–Siegert shifts are omitted.
- No direct qubit–qubit coupling, so residual idle ZZ does not cancel at a zero-ZZ bias point.
- Constant $g_{qc}$ (the $g_{qc} \propto \sqrt{\omega_c}$ scaling is available as an option but off).
- The SQUID dispersion has no $-E_C$ correction (the transmon result is $\omega \approx \sqrt{8 E_J E_C} - E_C$;
  Koch et al. 2007).
- Each coupler is simulated in an isolated qubit–coupler–qubit cell. Neighbouring couplers enter only through
  flux crosstalk and frequency dressing: no spectator qubits, no spectator ZZ, no AC flux crosstalk.
- Single-qubit gates and state preparation are ideal and instantaneous.
- Zero-temperature Lindblad dissipation with frequency-independent $T_1$/$T_2$ and local dissipators.
- No TLS defects coupled to the coupler; telegraph defects only shift the qubit frequencies.
- Readout assignment errors are independent per qubit (no readout crosstalk).
- Quasi-static flux noise is omitted from the measurement nodes; its population effect is far below shot
  noise.
- The benchmark is a Clifford-twirl model with gate-independent single-qubit error, not a sequence simulation.
- Discovery uses a phenomenological line model decoupled from the gate Hamiltonian; its outcome feeds the
  agent's context ($f_{01}$, $f_{12}$) but not the gate physics.
- Pulse distortions (low-pass, bias-tee droop) are implemented but off.

**Numerical accuracy.** With the default solver settings, a gate fidelity is within about $10^{-3}$ of a fully
converged integration (the error comes mostly from the 128 samples per modulation period). The error is
deterministic, so comparisons between agents are unaffected.

Full references are listed in [parameters.md](parameters.md#other-references).
