---
license: apache-2.0
pretty_name: tergite-chipcal benchmark chips v0.2
task_categories:
- reinforcement-learning
tags:
- llm-agents
- quantum-computing
- superconducting-qubits
- calibration
- physics
- benchmark
size_categories:
- n<1K
configs:
- config_name: 2q1c
  data_files:
  - split: test
    path: 2q1c-test.jsonl
- config_name: 3q2c
  data_files:
  - split: test
    path: 3q2c-test.jsonl
- config_name: 5q4c
  data_files:
  - split: test
    path: 5q4c-test.jsonl
---

# tergite-chipcal benchmark chips v0.2

**300 frozen, simulated superconducting-qubit chips for evaluating LLM agents that calibrate CZ gates from scratch.**

[tergite-chipcal](https://github.com/tergite/tergite-chipcal) is a multi-turn agent environment. Each episode gives the agent an uncharacterised chip
of fixed-frequency transmons joined by flux-tunable couplers. The agent has to find every readout resonator and
qubit frequency, then calibrate a CZ gate on every coupler. It works only through lab measurements such as
spectroscopy, chevrons, Ramsey phase scans and randomized benchmarking, and each one costs simulated lab time.
The score is the minimum interleaved-RB fidelity over the chip's couplers.

This dataset holds the evaluation chips: the full hidden parameters of every test chip, so all results are
reported on the same devices. Every chip can be calibrated: a privileged oracle certifies that each of its
couplers admits a CZ fidelity of at least 0.96, and the chip's per-coupler ceilings ship with it. The ceilings are
lower bounds (see the repository's docs/scoring.md).

**v0.2** keeps the chips of v0.1 (same seeds and hidden parameters) and recomputes the 2q1c ceilings with a full
search: both CZ routes (|11⟩ ↔ |20⟩ and |11⟩ ↔ |02⟩) and their half tones, the coupler's whole flux range
(qubit–coupler hybridization at most 2 %), amplitudes up to 1.0 and durations up to 2 µs, at time zero. Every 2q1c
chip now admits at least 0.975. The v0.1 value is kept per chip as `ceiling_v0_1`. 3q2c and 5q4c keep the v0.1
search (|11⟩ ↔ |20⟩ only, coupler parked above the qubits) for now.

> **Simulated data only.** No row describes a physical device. Physics model v0.1, not validated against
> hardware; see the repository's docs/physics.md for what is modelled and simplified. Built with the help of AI
> coding assistants.

## Contents

| Config | Chip | Couplers | Turn cap | Chips |
|---|---|---|---|---|
| `2q1c` | 2 qubits | 1 | 150 | 100 |
| `3q2c` | 3 qubits | 2 | 250 | 100 |
| `5q4c` | 5 qubits | 4 | 400 | 100 |

Each tier holds the first 100 test seeds (from 3000000 up) whose every coupler reaches the 0.96 ceiling; the
selection rejected 2 (2q1c), 3 (3q2c) and 13 (5q4c) seeds. The chips' ceilings (minimum over couplers) have
medians 0.992 (full search), 0.982 and 0.976 (v0.1 search).

Each row is one chip:

| Field | Description |
|---|---|
| `device` | Hidden parameters in SI units (Hz, s, A): qubit frequencies, anharmonicities, T1/T2, three-state readout confusion; coupler SQUID parameters, couplings, flux crosstalk; realism noise (charge-parity splits, telegraph defects, drift, calibrated-frame errors) |
| `ceiling` | Per coupler: the oracle's best CZ fidelity over bias and pulse, with the bias, flux, pulse frequency, amplitude, duration and local phases that reach it; for 2q1c also the route and the gate's mean, minimum and fraction ≥ 0.95 over 24 offset-charge states |
| `ceiling_v0_1` | 2q1c only: the v0.1 ceiling, for comparison |
| `tier`, `seed`, `split` | Chip size, generator seed, seed split (`test`) |
| `generator`, `version` | Generator version (`chipcal-chips-v0.1`, unchanged), dataset version (`v0.2`) |

`manifest.json` holds the SHA-256 of every file, the selection rule and the ceiling search of each tier. Total size is 2.1 MB.

## Usage

```python
from datasets import load_dataset
chips = load_dataset("tergite/tergite-chipcal", "2q1c", split="test")
```

To run episodes, rebuild the typed device with the tergite-chipcal package (`import chipcal`) and check the files against the manifest:

```python
from chipcal import splits
tier, seed, device = splits.load("2q1c-test.jsonl")[0]
```

```bash
python -m chipcal.splits verify .   # checksums, plus regeneration of the first chips of each file
```

**Do not train on these chips.** Draw training seeds from the train split (1000000–1999999,
`chipcal.tiers.SPLITS`). The parameters are public by design: the agent never sees them, and the benchmark
measures calibration from measurements.

## Baselines (2q1c)

All rows on the 100 chips except three models on the first 10 (Qwen3.8-27B, Gemma 4 31B without the playbook, Ising
Calibration 1.5). Verified entries are on the
[leaderboard](https://github.com/tergite/tergite-chipcal/blob/main/LEADERBOARD.md), ranked by calibration rate (every
final gate F ≥ 0.95). Score / ceiling divides each final gate by the best its coupler admits. Every model turn is in
[tergite/tergite-chipcal-runs](https://huggingface.co/datasets/tergite/tergite-chipcal-runs).

| Policy | Calibrated (every final F ≥ 0.95), 95 % CI | Score: min final-RB F, mean (95 % CI) | Score / ceiling |
|---|---|---|---|
| Gemma 4 31B + calibration playbook (guided: an unpublished strategy prompt) | **64 / 100** (54–73 %) | 0.821 (0.765–0.874) | 0.828 |
| Scripted calibration routine | **44 / 100** (35–54 %) | 0.853 (0.807–0.892) | 0.860 |
| Gemma 4 31B, no strategy prompt | **2 / 10** (6–51 %) | 0.437 (0.221–0.653) | 0.440 |
| Qwen3.8-27B, no strategy prompt | **0 / 10** (0–28 %) | 0.315 (0.250–0.445) | 0.318 |
| NVIDIA Ising Calibration 1.5 (31B), no strategy prompt, score pending: no reasoning tokens produced | **0 / 10** (0–28 %) | 0.000 (0.000–0.000) | 0.000 |
| Discovery, then a blind default gate | **0 / 100** (0–4 %) | 0.255 (0.250–0.266) | 0.257 |
| *Privileged oracle (physics ceiling, full search; a lower bound)* | *100 / 100* | *median 0.992 (0.975–0.997)* | *1* |

Every chip can be calibrated, yet the best entry calibrates 64 of 100 and the raw models at most 2 of 10: the
headroom is in the agent, not the physics. The 3q2c and 5q4c tiers are harder settings with the same score. They add
hidden flux crosstalk between couplers and weaker couplings.

## Beyond the frozen chips

The generator is procedural: every seed is a new chip, and a tier name `<n>q-<difficulty>` scales it to any size
(2 to 64 qubits; a chain up to 5, beyond that a grid whose rows are readout feedlines of at most 5) and three
difficulty presets that weaken couplings, raise crosstalk and worsen readout and coherence. Generate a training
set, optionally certified calibratable by the same oracle:

```bash
python scripts/generate_chips.py --tier 12q-hard --n 500 --out chips
python scripts/generate_chips.py --tier 8q-extreme --n 50 --min-ceiling 0.96 --workers 64 --out chips
```

`hard` and `extreme` chips above 10 qubits are experimental: how many can be calibrated is not yet
characterised, and exploration is welcome.

## Provenance

- Generated by the tergite-chipcal simulator from fixed seeds, with no human or hardware data.
- The measurement nodes and the CZ calibration chain follow
  [tergite-autocalibration](https://github.com/tergite/tergite-autocalibration).
- The simulator started from the pulse-level simulator in [tergite-backend](https://github.com/tergite/tergite-backend)
  and was extended to three-level transmons in a qubit-coupler-qubit system.
- Inspired by NVIDIA's work on AI for quantum calibration: the QCalEval benchmark (S. Cao et al.,
  [arXiv:2604.25884](https://arxiv.org/abs/2604.25884), 2026) and the NVIDIA Ising Calibration models. No code, data
  or prompts from either are used.

Tergite is developed by a core group of collaborators; Chalmers Next Labs AB (CNL) manages and maintains this
project. We are grateful for the collaboration of Chalmers Next Labs AB – WACQT Quantum Technology Testbed, and of
the Quantum Technology Laboratory (QTL) and the Applied Quantum Physics Laboratory (AQP) at Chalmers University of
Technology, MC2.

**Owner:** Chalmers Next Labs AB · **Created:** 2026-10-26 (v0.2: 2026-11-02) · **License:** Apache-2.0

Model and product names (such as NVIDIA Ising Calibration, Gemma and Qwen) belong to their respective owners. They
are used only to identify the work and the models referred to; no affiliation or endorsement is implied.

## Citation

```bibtex
@software{tergite_chipcal_2026,
  title   = {tergite-chipcal: a simulated superconducting-qubit chip-calibration environment for LLM agents},
  author  = {{Chalmers Next Labs AB}},
  year    = {2026},
  version = {0.2.0}
}
```
