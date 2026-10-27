# Scoring

## The score

When the episode ends, the environment benchmarks every coupler's final committed gate once more, at the final
bias currents and wall-clock time. It uses interleaved randomized benchmarking with a fixed protocol: Clifford
lengths 0–128, 21 random sequences per length and 1000 shots.

    F_i   = fitted fidelity of the final benchmark of coupler i      (one fit; no min or max over repeats)
    score = min_i F_i

- **The weakest gate decides.** A chip is only as useful as its worst gate, and the minimum cannot be traded
  off between couplers.
- **Unidentifiable benchmarks.** If the reference decay collapses to the 1/4 survival floor within a few
  Cliffords (p_ref ≤ 0.5), or the fit fails or leaves σ ≥ 0.1, the coupler scores **0.25**, the fidelity of a
  fully depolarising gate. Off-resonant, idle and broken gates land here. A single exponential does not fit
  leakage-dressed decays exactly; a poor fit (reduced χ² > 1) widens σ by √χ² instead of discarding the
  benchmark.
- **Incomplete gates.** A coupler with any of parking current, CZ frequency, amplitude or duration uncommitted
  scores **0**.
- **The agent's own benchmarks do not count.** They are measurements that help the agent decide. The score
  is always a fresh, final benchmark, so repeating benchmarks until a lucky draw gains nothing.
- **Reproducible.** The final benchmark's random stream depends only on the episode seed and the coupler, so
  a given final configuration always gets the same score.

**Success** means every coupler has F_i ≥ 0.95.

**Every benchmark chip can succeed.** A privileged oracle (`chipcal.baselines.oracle.coupler_best_ceiling`) reads
the hidden device and searches each coupler's bias, CZ pulse frequency, amplitude and duration, and local phases
for the best gate fidelity C_i the physics allows. The bias search stays in the standard tunable-coupler regime,
with the coupler at least 0.5 GHz above both its qubits, so C_i is a conservative bound. The frozen split keeps only chips whose every coupler has
C_i ≥ 0.96, a margin above 0.95 for RB fit noise. Each chip's C_i ship with it, and results also report

    score / ceiling = min_i F_i / C_i

so a score reads against what its chip allows.

## Reported alongside

| Metric | Meaning |
|---|---|
| `final_rb_fidelity_per_coupler` | the F_i above |
| `normalized_score` | score / ceiling, min_i F_i / C_i (frozen benchmark chips) |
| `min_true_fidelity` | the simulator's true average CZ fidelity (minimum over couplers), unscored |
| `success` | every F_i ≥ f_star (0.95) |
| `discovery_fraction` | fraction of resonator / f01 / f12 commits within 1 MHz of the truth |
| `achieved_min_f_lcb` | the agent's own last certified lower bounds (minimum over couplers) |
| `n_turns`, `n_invalid_actions`, parse failures | grammar skill, kept separate from calibration skill |
| `lab_time_s` | simulated instrument time used |

Average gate fidelity has a high floor. The identity scores 0.4 against a CZ, and any non-entangling gate with
suitable single-qubit phases reaches 0.6. So the true fidelity of a useless gate can look respectable, while
its randomized benchmark collapses to the floor. The score follows the benchmark.

## Reward-validity probes

`chipcal.baselines.probes` runs degenerate policies:

| Probe | Expected score |
|---|---|
| Stop immediately | 0 (no gate) |
| Spam invalid actions | 0 |
| Random grammar-valid actions | 0, or 0.25 if a random gate happens to be complete |
| Skip discovery, commit a blind gate | 0 (coupler actions are rejected until discovery is complete) |
| Discover, then commit a blind default gate | about 0.25, more on a chip where the blind pulse happens to entangle |
| Discover, blind gate, then benchmark 20 times | the same: the agent's benchmarks do not enter the score |
