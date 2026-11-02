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
- **The device keeps drifting, by design.** The final benchmark sees the chip as it is when the episode ends,
  including its slow drifts: every qubit's offset charge is redrawn every 300 s of lab time, which changes its
  charge-parity splitting ([parameters](parameters.md)). On qubits with a small E_J/E_C this splits the |2⟩
  level by up to a few MHz, enough to detune a CZ tuned in a calmer charge state. Transmons in the lab behave the same
  way, so a gate that is good only in some charge states scores by luck; a robust gate (typically a shorter
  pulse) scores reliably. An agent's own benchmark is a snapshot taken when it starts, while it occupies 270 s of
  lab time, so it may describe a different charge state from the one scored.

**Success** (a calibrated chip) means every coupler has F_i ≥ 0.95.

## The leaderboard ranks by calibration rate

The headline number of an entry is its **calibration rate**: the fraction of frozen chips it calibrates, with a
95 % Wilson interval. The mean score is the tie-break and is reported beside it. This matches how a calibration
engineer judges a run (did I get a usable gate?). Fidelity is a compressed scale with a high floor, so a mean
score gives a broken run most of a point: a gate at 0.865 has about 13 times the error of a 0.99 gate yet counts
0.865, and a policy that calibrates more chips can still have the lower mean when its misses are broken gates
(0 or 0.25) rather than mediocre ones.

The rate is a threshold metric, so chips near 0.95 can fall either side by drift luck, and entries on only 10
chips have wide intervals (8 / 10 is 49–94 %). RL training should keep the continuous score as the reward; a 0/1
reward is too sparse. The reward and the headline differ on purpose.

## The physics ceiling

**Every benchmark chip can succeed.** A privileged oracle reads the hidden device and searches each coupler for the
best CZ fidelity C_i it can find. The frozen split keeps only chips whose every coupler has C_i ≥ 0.96, a margin
above 0.95 for RB fit noise. Each chip's C_i ship with it, and results also report

    score / ceiling = min_i F_i / C_i

so a score reads against what its chip allows.

**Benchmark v0.2 (2q1c): the full search.** `chipcal.baselines.oracle.coupler_full_ceiling`, with every other
coupler at its reset current, searches:

- four CZ routes: the |11⟩ ↔ |20⟩ resonance (the control qubit's second level) and |11⟩ ↔ |02⟩ (the target's),
  each driven at its own frequency or at half of it (a half tone);
- the coupler's whole flux range, above or below the qubits, as long as qubit–coupler hybridization stays at
  most 2 % (the coupler stays a coupler);
- pulse amplitudes up to 1.0 and durations up to 2 µs, as the environment allows;
- the local phases of each gate.

C_i is the best fidelity at time zero. Every 2q1c chip has C_i ≥ 0.975 (median 0.992). The winning routes are
|11⟩ ↔ |02⟩ on 52 chips, |11⟩ ↔ |20⟩ on 41 and a half tone on 7. Each ceiling gate is also checked over 24
offset-charge states: it stays at or above 0.95 in all of them on 99 chips, and in 79 % of them on seed 3000086.
The v0.1 value of each ceiling is kept in the data as `ceiling_v0_1`.

**The ceiling is still a lower bound.** The search is a grid plus local refinement, so a policy can exceed its
chip's ceiling (score / ceiling above 1). One simulator simplification matters for gates with the coupler parked
below the qubits, which win on many chips: the qubit–coupler coupling g does not fall with the coupler's frequency
([physics](physics.md#simplifications)), which makes those gates somewhat optimistic.

**3q2c and 5q4c keep the v0.1 search** until the full search runs on them: one route (|11⟩ ↔ |20⟩, first
harmonic), the coupler parked at least 0.5 GHz above both qubits, amplitudes up to 0.7 and durations below
800 ns, at time zero. `data/v0.2/manifest.json` records which search each tier uses.

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
