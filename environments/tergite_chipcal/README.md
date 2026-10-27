# tergite-chipcal

Calibrate a simulated superconducting-qubit chip, from first light to CZ gates, as a multi-turn verifiers
environment. See the [tergite-chipcal repository](../../README.md) for the task, the action grammar and the score.

| Taskset option | Default | Meaning |
|---|---|---|
| `tier` | `2q1c` | `2q1c`, `3q2c` or `5q4c` |
| `split` | `test` | `train`, `validation` or `test` (benchmark chips are the first seeds of `test`) |
| `n_chips` | 100 | chips per eval |
| `first_seed` | split start | first chip seed |
| `visual` | true | attach a plot of each measurement (needs a vision-language model) |
| `max_turns` | tier cap | override the turn cap |

**Reward:** `chip_score` = the minimum over couplers of the fidelity from a final interleaved randomized
benchmark of each final committed gate. A benchmark that is not identifiable counts 0.25; an incomplete gate
counts 0.

**Metrics:** `min_true_fidelity`, `success`, `discovery_fraction`, `invalid_action_rate`,
`parse_failure_rate`, `lab_time_h`.

**Turn cap:** set the agent seat's `max_turns` at least as high as the tier's turn cap. A lower value ends the
conversation early; the chip is then scored as it stands.
