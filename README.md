# tergite-chipcal

**Calibrate a simulated superconducting-qubit chip, end to end, from first light to CZ gates.**

tergite-chipcal is a multi-turn environment for LLM agents and RL, built on a physics simulator of fixed-frequency
transmons joined by flux-tunable couplers. An episode hands the agent an uncharacterised chip. The agent must
first find every qubit's readout resonator and transition frequencies, then calibrate a CZ gate on every coupler.
Its tools are the measurements an experimentalist would use: spectroscopy, parametrization maps, chevrons, Ramsey
phase scans and randomized benchmarking. Each measurement costs simulated lab time, and the result comes back as
finite-shot data and a plot.

The score is a final randomized benchmark of the gates the agent leaves committed. The only way to score well is
to build a good CZ gate.

> **Physics model v0.1, not validated against hardware.** See [docs/physics.md](docs/physics.md) for what is
> modelled and what is simplified. Built with the help of AI coding assistants.

**It needs a vision-language model.** After each measurement the agent receives a plot of the data as an image.
For a text-only model, pass `--text-only` (or `--taskset.visual false` on the Prime Intellect hub); a text
analysis of the same data then replaces the plot. To see exactly what a model receives, export it without a
model ([below](#see-what-the-model-sees)).

**What makes it different**
- **Procedural, as big and as hard as you want.** Every seed is a new chip. Beyond the three benchmark tiers,
  any name like `12q-extreme` is a procedural tier: up to 64 qubits, as a grid on multiplexed feedlines, at
  three difficulty presets ([below](#procedural-chips)). The training distribution never runs out.
- **Solvable by construction.** A privileged oracle certifies that every coupler of every benchmark chip admits a
  CZ fidelity of at least 0.96. Each chip ships with its ceiling, and results are reported against it.
- **A verifiable reward.** Hidden device parameters never reach the agent, its own benchmarks do not count, and
  degenerate policies score 0 ([docs/scoring.md](docs/scoring.md)).
- **A real calibration workflow.** The measurement nodes and the CZ calibration chain follow the
  [tergite-autocalibration](https://github.com/tergite/tergite-autocalibration) repository. The simulator started
  from the pulse-level simulator in [tergite-backend](https://github.com/tergite/tergite-backend); its Hamiltonian
  was extended to three-level transmons in a qubit-coupler-qubit system.

| Tier | Chip | Couplers | Turn cap |
|---|---|---|---|
| `2q1c` | 2 qubits | 1 | 150 |
| `3q2c` | 3 qubits | 2 | 250 |
| `5q4c` | 5 qubits | 4 | 400 |
| `<n>q-<difficulty>` | 2–64 qubits: a chain up to 5, a grid beyond | chain n − 1; 2×5 grid 13 | 50 + 100 per coupler |

## Install

```bash
pip install -e .            # numpy, scipy, xarray, matplotlib
pip install -e ".[llm]"     # + openai, for running a model
```

The distribution is `tergite-chipcal`; the Python package is imported as `chipcal`.

## Run a model

Serve a vision-language model behind any OpenAI-compatible endpoint (for example vLLM), then:

```bash
python -m chipcal.run --tier 2q1c --seed 3000000 --out runs/demo \
    --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.8-27B
python -m chipcal.run ... --text-only        # a text-only model: text analysis instead of plots
```

`runs/demo/transcript.json` holds every turn, `runs/demo/run.json` holds the score, and `runs/demo/vision/`
holds every plot the model was shown.

To run on your own machine instead, [docs/local-ollama.md](docs/local-ollama.md) serves Qwen3.8 with vision
through Ollama and starts an episode against it.

This runner sends the model the system prompt and one self-contained observation per turn. The observation
carries the committed configuration, a measurement notebook and the session history. Actions are JSON in
text, with guided decoding against the action schema when the server supports `response_format`.

### Prime Intellect Environments Hub / verifiers

`environments/tergite_chipcal` is a verifiers (v1) environment. The environment plays the user's side of the
conversation: the first user turn is the full observation, and later turns are compact updates plus the plot
of the latest measurement.

```bash
pip install -e environments/tergite_chipcal
prime env eval tergite-chipcal --taskset.tier 2q1c --taskset.n-chips 30
```

The reward is `chip_score`, the score below. The metrics also report the score / ceiling, the true minimum
fidelity, success, discovery accuracy, invalid-action rate, parse-failure rate and lab time. On the test split of
a named tier the tasks are the frozen benchmark chips.

### See what the model sees

`scripts/export_observations.py` plays an episode with the scripted calibration routine, or replays a recorded
transcript, and writes the model's input turn by turn without a model or a GPU: `system_prompt.txt`, one
`turn_NNN.txt` per turn and the PNG attached to it.

```bash
python scripts/export_observations.py --tier 2q1c --seed 3000000 --out runs/seen
python scripts/export_observations.py --tier 8q-hard --seed 1000042 --max-turns 40 --out runs/seen-8q
python scripts/export_observations.py --replay runs/demo/transcript.json --tier 2q1c --seed 3000000 --out runs/replay
```

## Procedural chips

The three benchmark tiers are three points on a dial. A tier name `<n>q-<difficulty>` turns the dial: n qubits
(2 to 64) from the same physical prior, at one of three presets. Up to 5 qubits the chip is a chain. Beyond, it is a
grid: each row is one readout feedline of at most 5 qubits, couplers join neighbours along each row and between
rows, and the qubit frequencies alternate as a checkerboard so every coupled pair is detuned. `10q-hard` is a
2×5 grid with 13 couplers.

| Preset | Qubit–coupler g | Weak links | Flux crosstalk (nearest neighbour) | Readout error | T1 | Min CZ detuning |
|---|---|---|---|---|---|---|
| `nominal` | 27–72 MHz | – | 0.5–1 % | 1–5 % | 15–110 µs | 100 MHz |
| `hard` | 20–50 MHz | 25 % redrawn at 12–20 MHz | 1–2 % | 3–8 % | 15–60 µs | 80 MHz |
| `extreme` | 15–40 MHz | 50 % redrawn at 9–15 MHz | 2–4 % | 5–12 % | 10–40 µs | 60 MHz |

Weak couplings narrow and slow the CZ window, so a gate fights decoherence. Stronger crosstalk means re-biasing one
coupler detunes its neighbours. Worse readout and shorter T1 blur every measurement. Readout is multiplexed on
feedlines of at most 5 qubits, each with its own readout LO, so discovery stays a realistic resonator-assignment
problem at any size. A procedural tier allows 50 turns plus 100 per coupler, and 2 h of lab time per coupler.

A procedural tier name works wherever a tier name does:

```bash
python -m chipcal.run --tier 12q-hard --seed 1000042 --out runs/12q --base-url ... --model ...
prime env eval tergite-chipcal --taskset.tier 8q-extreme --taskset.split train
python scripts/generate_chips.py --tier 8q-hard --n 50 --out chips          # 50 chips, full parameters
python scripts/generate_chips.py --tier 6q-hard --n 10 --min-ceiling 0.96 --workers 32 --out chips   # certified
python scripts/evaluate.py --policy scripted --tier 8q-hard --data chips --n 10 --out results/8q
```

Harder presets can produce chips whose physics allows no 0.95 gate. `--min-ceiling` runs the privileged oracle on
every coupler (about 25 CPU-minutes each) and keeps only chips that can be calibrated, as the benchmark does.

> **Experimental:** `hard` and `extreme` chips above 10 qubits. How many of them can be calibrated, and how the
> scripted routine copes with them, is not yet characterised. Each coupler is simulated in its own
> qubit-coupler-qubit cell, so a large chip has no spectator-qubit physics beyond flux crosstalk and frequency
> dressing ([docs/physics.md](docs/physics.md)). Exploration and reports are welcome.

## The task

**Observation.** Each turn is a text block plus, after a measurement, a PNG plot of the measured data. The
text block contains:
- turn and lab-time budget;
- the chip status;
- the committed configuration of every coupler, with all six fields shown;
- the discovery state (design frequencies, local-oscillator bands, committed frequencies);
- the measurement limits;
- the last event;
- a notebook of empirical fits.

Hidden device parameters never appear.

**Actions.** One JSON object per turn:

```json
{"configure_lo": {"channel": "readout", "frequency_hz": 6.8e9}}
{"qubit": "q0", "node": "resonator_spectroscopy", "sweeps": {"frequencies": [6.8e9, 4.5e8, 401]}, "shots": 512}
{"qubit": "q0", "commit_frequency": {"transition": "f01", "frequency_hz": 4.19e9}}
{"coupler": "c0", "node": "cz_chevron", "sweeps": {"cz_pulse_frequencies": [2.1e8, 8e6, 21], "cz_pulse_durations": [2.2e-7, 4e-7, 26], "cz_pulse_amplitude": 0.5}, "shots": 128}
{"coupler": "c0", "commit": {"param": "cz_pulse_duration", "value": 2.4e-7}}
{"coupler": "c0", "benchmark": true}
{"stop": true}
```

Sweeps are `[center, full_span, point_count]` in SI units. A value in the wrong unit (for example 220 ns written
as 2.2e2 or 2.2e8 instead of 2.2e-7) is rejected with the expected unit stated. Other out-of-range values are
clamped, and the agent is told so. Invalid actions are shown back to the agent and cost a turn, with no score penalty.

The measurement nodes and committed quantities follow the vocabulary of the open-source tergite-autocalibration
tool:
- discovery: `resonator_spectroscopy`, `qubit_01_spectroscopy`, `qubit_12_spectroscopy`;
- coupler: `coupler_dc_spectroscopy`, `cz_parametrization`, `cz_chevron`, `cz_calibration`, `cz_local_phases`;
- single-qubit: `qubit_spectroscopy`, `qubit_t1`, `qubit_ramsey`, `readout_calibration`;
- `benchmark` (interleaved RB).

The prompt states the task, the score, the grammar, the units and what each node does. It contains no
calibration strategy.

**Score.** When the episode ends (stop, turn cap or lab-time budget), the environment runs one final
interleaved randomized benchmark (21 random sequences, 1000 shots) of every coupler's final committed gate, at
the final bias currents:

    score = min over couplers of F_RB,   F_RB = the fitted fidelity of that final benchmark

See [docs/scoring.md](docs/scoring.md).

## Benchmark v0.1

`data/v0.1/` freezes the evaluation chips. For each tier it holds the first 100 test seeds whose every coupler
admits a CZ fidelity of at least 0.96, as certified by a privileged oracle that searches each coupler's bias and
pulse (`chipcal.baselines.oracle`). Every chip is stored with its full hidden parameters, its per-coupler ceilings
and a checksum. Check it with `python -m chipcal.splits verify data/v0.1`. Training should draw seeds from the
train split (`chipcal.tiers.SPLITS`).

So the 0.95 calibration bar is reachable on every benchmark chip, and each score is also reported as
**score / ceiling**: each final gate against the best its coupler admits ([docs/scoring.md](docs/scoring.md)).

Reported results are for the **2q1c** tier: 2 qubits and 1 coupler, starting from full discovery. The 3q2c
and 5q4c tiers ship as harder settings with the same score.

| Policy (2q1c) | Chips × samples | Score: min final-RB F, mean (95 % CI) | Score / ceiling | Calibrated (F ≥ 0.95) | Parse failures / turn |
|---|---|---|---|---|---|
| Stop immediately | 12 × 1 | 0.000 (0.000–0.000) | 0.000 | 0 / 12 | – |
| Random valid actions | 12 × 1 | 0.000 (0.000–0.000) | 0.000 | 0 / 12 | – |
| Discovery, then a blind default gate | 12 × 1 | 0.294 (0.250–0.381) | 0.298 | 0 / 12 | – |
| Scripted calibration routine | 100 × 1 | 0.853 (0.807–0.892) | 0.866 | 44 / 100 | – |
| Qwen3.8-27B, no strategy prompt (pilot)<sup>†</sup> | 8 × 1 | 0.331 (0.250–0.493) | 0.337 | 0 / 8 | 0.003 |
| *Physics ceiling (privileged oracle, best bias and pulse)* | *100* | *median 0.986 (0.961–0.996)* | *1* | *100 / 100* | – |

The scripted routine runs on all 100 chips; the probe rows cover the first 12 and Qwen the first 8 (a pilot).
Every chip can be calibrated, yet the scripted routine calibrates 44 of 100: the gap to the ceiling is the
agent's, not the physics'. The scripted routine reads only its own measurements; its search ranges were tuned on
chips of this simulator's distribution.

`python scripts/evaluate.py --summarize results/...` regenerates the table from the episode logs.

<sup>†</sup> **Note on the Qwen row.** The model's score can be raised substantially by putting a calibration guideline
in the system prompt, i.e. the physics and the order of the calibration chain, pasted in as extra knowledge.
This is deliberately not done here: the v0.1 result measures the raw model's ability to calibrate from the
task description alone, and no strategy prompt is published at this stage. To run your own ablation, pass a
text file with `--prompt-addendum` to `scripts/evaluate.py`.


## Physics

See [docs/physics.md](docs/physics.md) for what is modelled and simplified, and
[docs/parameters.md](docs/parameters.md) for every sampled range and its published source.
[docs/examples.md](docs/examples.md) walks through one successful calibration, turn by turn.

## Acknowledgements

This project was sponsored by:

-   [Knut and Alice Wallenberg Foundation](https://kaw.wallenberg.org/en) under the [Wallenberg Center for Quantum Technology (WACQT)](https://www.chalmers.se/en/centres/wacqt/) project at [Chalmers University of Technology](https://www.chalmers.se)

## License

Apache-2.0. See [LICENSE](LICENSE).
