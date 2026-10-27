# Parameters and their basis

Every chip is sampled from the ranges below (`chipcal/tiers.py`, `chipcal/sim/realism.py`).

Sampling is uniform within each range unless noted.

## Qubits (fixed-frequency transmons)

| Parameter | Range | Basis |
|---|---|---|
| f01 | two alternating bands, 4.0–4.4 GHz and 4.6–5.0 GHz | generic (typical fixed-frequency transmons; neighbours detuned by design) |
| E_J/E_C | 36.5–100 | WACQT devices: Burnett et al. 2019, Kosen et al. 2022 |
| Anharmonicity | derived from f01 and E_J/E_C (exact charge-basis spectrum), giving about −160 to −300 MHz | standard transmon theory (Koch et al. 2007); consistent with WACQT devices (Kosen et al. 2022 and 2024) |
| T1 | 15–110 µs | WACQT devices: Kosen et al. 2022 and 2024, Burnett et al. 2019; the lower end follows Arute et al. 2019 |
| T2 / T1 | 0.6–2.0, with T2 ≤ 2 T1 | WACQT devices: Kosen et al. 2022 and 2024, Burnett et al. 2019 |
| Minimum \|f01(target) − f12(control)\| | 100 MHz (collision filter) | generic |

## Readout (three-state assignment, preparation error included)

| Parameter | Range | Basis |
|---|---|---|
| P(0\|0) | 0.97–0.995 | WACQT: Chen et al. 2023; also Sung et al. 2021, Blok et al. 2021 |
| P(1\|1) | 0.90–0.97 | WACQT: Chen et al. 2023; also Sung et al. 2021, Blok et al. 2021 |
| P(2\|2) | 0.92–0.96 | WACQT three-state readout: Chen et al. 2023; also Blok et al. 2021 |
| How errors split between states | \|0⟩ errors 90 % to \|1⟩; \|1⟩ errors 70/30 to \|0⟩/\|2⟩; \|2⟩ errors 80 % to \|1⟩ | generic (relaxation during readout dominates) |

## Tunable couplers

| Parameter | Range | Basis |
|---|---|---|
| Maximum frequency | 6.47–7.92 GHz | WACQT devices: Kosen et al. 2022 and 2024 |
| Anharmonicity | −90 to −75 MHz | WACQT: Kosen et al. 2022; also Sung et al. 2021 |
| T1 | 8–12 µs | typical published values, e.g. Sung et al. 2021 |
| T2 | 1–6 µs | typical published values, e.g. Sung et al. 2021 |
| Qubit–coupler coupling g | 27–72.5 MHz overall; per tier: 2q1c 45–72, 3q2c 35–72, 5q4c 27–72 MHz | WACQT: Kosen et al. 2022; also Sung et al. 2021 |
| SQUID asymmetry d | 0.03–0.08 | generic |
| Current per flux quantum | 2.4–3.6 mA | WACQT: Kosen et al. 2024 |
| Sweet-spot current offset | −1 to +1 mA | generic |
| AC flux per unit pulse amplitude | 0.15–0.35 Φ0 | generic |
| Nearest-neighbour DC flux crosstalk | 3q2c 0.25–0.5 %, 5q4c 0.5–1 % of the self-coupling; next-nearest 0.05–0.2 × that | WACQT: Kosen et al. 2024 (the upper values correspond to lines without crosstalk shielding); also Dai et al. 2021 |
| Quasi-static flux noise (rms) | 5–11 µΦ0 | typical published 1/f flux noise, e.g. Bylander et al. 2011, Krantz et al. 2019 |

## Realism: slow and non-Markovian effects

| Parameter | Range | Basis |
|---|---|---|
| Charge-parity line splitting | follows from each qubit's E_C and E_J/E_C (exact spectrum), times \|cos 2π n_g\| | standard transmon theory (Koch et al. 2007); consistent with WACQT measurements (Burnett et al. 2019) and with Ristè et al. 2013, Serniak et al. 2018 |
| Offset charge n_g | uniform in [0, 1), redrawn every 300 s | slow offset-charge drift is commonly reported, e.g. Christensen et al. 2019; the 300 s interval is generic |
| Parity switching | shot-to-shot, 50/50 between the branches | published switching times are much shorter than an acquisition, e.g. Ristè et al. 2013, Serniak et al. 2018 |
| Telegraph frequency jumps | 0–2 defects per qubit, each a 1–20 kHz jump, switching rate log-uniform in 2×10⁻⁵ to 2×10⁻³ Hz | WACQT: Burnett et al. 2019 |
| Linear frequency drift | −1 to +1 kHz/h | order of magnitude, WACQT: Burnett et al. 2019 |
| Frequency random walk | 0–2 kHz/√h | order of magnitude, WACQT: Burnett et al. 2019 |
| Frame calibration error (Ramsey) | 2 kHz rms | WACQT Ramsey tracking: Burnett et al. 2019 |
| Retune time after a bias move | 60 s | generic |
| Single-qubit layers per two-qubit Clifford | 5 × 40 ns | standard Clifford decompositions, e.g. Barends et al. 2014 |
| CZ gates per two-qubit Clifford | 1.5 on average | standard Clifford decompositions, e.g. Barends et al. 2014 |

## Measurement and time costs

| Parameter | Value | Basis |
|---|---|---|
| Spectroscopy window | ≤ 20 MHz and ≤ 201 points per scan; bias ≤ 5 mA and ≤ 101 points | generic experiment policy |
| CZ frequency window | ≤ 40 MHz per scan | generic |
| Points per measurement | ≤ 5000 (product of all sweep axes) | generic (bounds simulation time) |
| Pulse timing grid | 4 ns | generic |
| Node overhead / shot time / DC settle / benchmark | 15 s / 250 µs / 20 s / 270 s | generic |
| Discovery: design-frequency uncertainty | σ 100 MHz, truncated at ±300 MHz | generic (fabrication spread of junction-based frequencies) |
| Discovery: LO band | ±250 MHz IF, 500 MHz maximum span | generic |
| Lab-time budget | 8 h simulated per episode | generic |

## Main sources (WACQT)

- J. J. Burnett et al., "Decoherence benchmarking of superconducting qubits", npj Quantum Inf. 5, 54 (2019),
  doi:10.1038/s41534-019-0168-5.
- S. Kosen et al., "Building blocks of a flip-chip integrated superconducting quantum processor", Quantum Sci.
  Technol. 7, 035018 (2022), doi:10.1088/2058-9565/ac734b.
- L. Chen et al., "Transmon qubit readout fidelity at the threshold for quantum error correction without a
  quantum-limited amplifier", npj Quantum Inf. 9, 26 (2023), doi:10.1038/s41534-023-00689-6.
- S. Kosen et al., "Signal Crosstalk in a Flip-Chip Quantum Processor", PRX Quantum 5, 030350 (2024),
  doi:10.1103/PRXQuantum.5.030350.

## Other references

- F. Arute et al., "Quantum supremacy using a programmable superconducting processor", Nature 574, 505 (2019), doi:10.1038/s41586-019-1666-5.
- R. Barends et al., "Superconducting quantum circuits at the surface code threshold for fault tolerance", Nature 508, 500 (2014), doi:10.1038/nature13171.
- M. S. Blok et al., "Quantum Information Scrambling on a Superconducting Qutrit Processor", Phys. Rev. X 11, 021010 (2021), doi:10.1103/PhysRevX.11.021010.
- J. Bylander et al., "Noise spectroscopy through dynamical decoupling with a superconducting flux qubit", Nat. Phys. 7, 565 (2011), doi:10.1038/nphys1994.
- S. A. Caldwell et al., "Parametrically Activated Entangling Gates Using Transmon Qubits", Phys. Rev. Applied 10, 034050 (2018), doi:10.1103/PhysRevApplied.10.034050.
- B. G. Christensen et al., "Anomalous charge noise in superconducting qubits", Phys. Rev. B 100, 140503(R) (2019), doi:10.1103/PhysRevB.100.140503.
- X. Dai et al., "Calibration of Flux Crosstalk in Large-Scale Flux-Tunable Superconducting Quantum Circuits", PRX Quantum 2, 040313 (2021), doi:10.1103/PRXQuantum.2.040313.
- J. Koch et al., "Charge-insensitive qubit design derived from the Cooper pair box", Phys. Rev. A 76, 042319 (2007), doi:10.1103/PhysRevA.76.042319.
- P. Krantz et al., "A quantum engineer's guide to superconducting qubits", Appl. Phys. Rev. 6, 021318 (2019), doi:10.1063/1.5089550.
- E. Magesan et al., "Efficient Measurement of Quantum Gate Error by Interleaved Randomized Benchmarking", Phys. Rev. Lett. 109, 080505 (2012), doi:10.1103/PhysRevLett.109.080505.
- D. C. McKay et al., "Universal Gate for Fixed-Frequency Qubits via a Tunable Bus", Phys. Rev. Applied 6, 064007 (2016), doi:10.1103/PhysRevApplied.6.064007.
- L. H. Pedersen, N. M. Møller and K. Mølmer, "Fidelity of quantum operations", Phys. Lett. A 367, 47 (2007), doi:10.1016/j.physleta.2007.02.069.
- D. Ristè et al., "Millisecond charge-parity fluctuations and induced decoherence in a superconducting transmon qubit", Nat. Commun. 4, 1913 (2013), doi:10.1038/ncomms2936.
- K. Serniak et al., "Hot Nonequilibrium Quasiparticles in Transmon Qubits", Phys. Rev. Lett. 121, 157701 (2018), doi:10.1103/PhysRevLett.121.157701.
- E. A. Sete et al., "Parametric-Resonance Entangling Gates with a Tunable Coupler", Phys. Rev. Applied 16, 024050 (2021), doi:10.1103/PhysRevApplied.16.024050.
- Y. Sung et al., "Realization of High-Fidelity CZ and ZZ-Free iSWAP Gates with a Tunable Coupler", Phys. Rev. X 11, 021058 (2021), doi:10.1103/PhysRevX.11.021058.
- C. J. Wood and J. M. Gambetta, "Quantification and characterization of leakage errors", Phys. Rev. A 97, 032306 (2018), doi:10.1103/PhysRevA.97.032306.
- F. Yan et al., "Tunable Coupling Scheme for Implementing High-Fidelity Two-Qubit Gates", Phys. Rev. Applied 10, 054062 (2018), doi:10.1103/PhysRevApplied.10.054062.
