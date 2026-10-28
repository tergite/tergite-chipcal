---
license: apache-2.0
pretty_name: tergite-chipcal benchmark chips v0.1
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

# tergite-chipcal benchmark chips v0.1

**300 frozen, simulated superconducting-qubit chips for evaluating LLM agents that calibrate CZ gates from scratch.**

[tergite-chipcal](https://github.com/tergite/tergite-chipcal) is a multi-turn agent environment. Each episode gives the agent an uncharacterised chip
of fixed-frequency transmons joined by flux-tunable couplers. The agent has to find every readout resonator and
qubit frequency, then calibrate a CZ gate on every coupler. It works only through lab measurements such as
spectroscopy, chevrons, Ramsey phase scans and randomized benchmarking, and each one costs simulated lab time.
The score is the minimum interleaved-RB fidelity over the chip's couplers.

This dataset holds the evaluation chips: the full hidden parameters of every test chip, so all results are
reported on the same devices. Every chip can be calibrated: a privileged oracle certifies that each of its
couplers admits a CZ fidelity of at least 0.96, and the chip's per-coupler ceilings ship with it.

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
medians 0.986, 0.982 and 0.976.

Each row is one chip:

| Field | Description |
|---|---|
| `device` | Hidden parameters in SI units (Hz, s, A): qubit frequencies, anharmonicities, T1/T2, three-state readout confusion; coupler SQUID parameters, couplings, flux crosstalk; realism noise (charge-parity splits, telegraph defects, drift, calibrated-frame errors) |
| `ceiling` | Per coupler: the oracle's best CZ fidelity over bias and pulse, with the bias, flux, pulse frequency, amplitude, duration and local phases that reach it |
| `tier`, `seed`, `split` | Chip size, generator seed, seed split (`test`) |
| `generator`, `version` | Generator version (`chipcal-chips-v0.1`), dataset version |

`manifest.json` holds the SHA-256 of every file and the selection rule. Total size is 2.0 MB.

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

All rows on the 100 chips except Qwen, a pilot on the first 8; verified entries are on the
[leaderboard](https://github.com/tergite/tergite-chipcal/blob/main/LEADERBOARD.md).
Score / ceiling divides each final gate by the best its coupler admits.

| Policy | Score: min final-RB F, mean (95 % CI) | Score / ceiling | Calibrated (F ≥ 0.95) |
|---|---|---|---|
| Discovery, then a blind default gate | 0.255 (0.250–0.266) | 0.259 | 0 / 100 |
| Scripted calibration routine | 0.853 (0.807–0.892) | 0.866 | 44 / 100 |
| Qwen3.8-27B, no strategy prompt | 0.331 (0.250–0.493) | 0.337 | 0 / 8 |
| *Privileged oracle (physics ceiling)* | *median 0.986 (0.961–0.996)* | *1* | *100 / 100* |

Every chip can be calibrated, yet the scripted routine calibrates 44 of 100 and the model none of its 8: the
headroom is in the agent, not the physics. The 3q2c and 5q4c tiers are harder settings with the same score. They add hidden
flux crosstalk between couplers and weaker couplings.

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

Tergite is developed by a core group of collaborators; Chalmers Next Labs AB (CNL) manages and maintains this
project. We are grateful for the collaboration of Chalmers Next Labs AB – WACQT Quantum Technology Testbed, and of
the Quantum Technology Laboratory (QTL) and the Applied Quantum Physics Laboratory (AQP) at Chalmers University of
Technology, MC2.

**Owner:** Chalmers Next Labs AB · **Created:** 2026-10-26 · **License:** Apache-2.0

## Citation

```bibtex
@software{tergite_chipcal_2026,
  title   = {tergite-chipcal: a simulated superconducting-qubit chip-calibration environment for LLM agents},
  author  = {{Chalmers Next Labs AB}},
  year    = {2026},
  version = {0.1.1}
}
```
