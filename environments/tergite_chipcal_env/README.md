# tergite-chipcal-env

Calibrate a simulated superconducting-qubit chip, from first light to CZ gates, as a multi-turn verifiers
environment. See the [tergite-chipcal repository](https://github.com/tergite/tergite-chipcal) for the task, the action
grammar, the score and the [leaderboard](https://github.com/tergite/tergite-chipcal/blob/main/LEADERBOARD.md).

| Taskset option | Default | Meaning |
|---|---|---|
| `tier` | `2q1c` | `2q1c`, `3q2c` or `5q4c`, or a procedural tier such as `8q-hard` |
| `split` | `test` | `train`, `validation` or `test` (the frozen benchmark: every chip certified calibratable) |
| `n_chips` | 100 | chips per eval |
| `first_seed` | split start | first chip seed |
| `visual` | true | attach a plot of each measurement (needs a vision-language model) |
| `max_turns` | tier cap | override the turn cap |

**Reward:** `chip_score` = the minimum over couplers of the fidelity from a final interleaved randomized
benchmark of each final committed gate. A benchmark that is not identifiable counts 0.25; an incomplete gate
counts 0.

**Metrics:** `normalized_score` (score / ceiling), `min_true_fidelity`, `success`, `discovery_fraction`, `invalid_action_rate`,
`parse_failure_rate`, `lab_time_h`.

**Turn cap:** set the agent seat's `max_turns` at least as high as the tier's turn cap. A lower value ends the
conversation early; the chip is then scored as it stands.

## Development

Tergite is developed by a core group of collaborators. [Chalmers Next Labs AB (CNL)](https://chalmersnextlabs.se/) takes on the
role of managing and maintaining this project.

We are grateful for the collaboration of:

- [Chalmers Next Labs AB](https://chalmersnextlabs.se/) – WACQT Quantum Technology Testbed
- Chalmers University of Technology, MC2 –
  [Quantum Technology Laboratory (QTL)](https://www.chalmers.se/en/departments/mc2/research/quantum-technology/)
- Chalmers University of Technology, MC2 – [Applied Quantum Physics Laboratory (AQP)](https://github.com/aqp-mc2-chalmers)
