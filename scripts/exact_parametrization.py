"""Compare a cz_parametrization the agent measured with the exact, noise-free map behind it.

    python scripts/exact_parametrization.py runs/demo --turn 3 --out docs/example-cz-parametrization.png

Panel (a) is the plot the agent saw (from the run's vision/ folder). Panel (b) is the exact probability that
the control qubit ends in |2> on the same grid: no shot noise and no readout error. Panel (c) is the same on a
fine grid. The chip is taken from the run's checkpoint (episode.pkl) in its final state, so slow drifts can make
(b) differ slightly from the moment of the measurement.
"""
from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from chipcal.core.actions import MeasureAction
from chipcal.envs import multimode_nodes
from chipcal.sim.realism import parity_branches


def row(job):
    """Exact control-|2> population along the frequency axis at one amplitude, averaged over parity branches."""
    theta, config, bias, freqs, amplitude, duration = job
    action = MeasureAction(node="cz_parametrization", sweeps={
        "dc_currents": [bias, 0.0, 1], "cz_pulse_frequencies": [float(np.mean(freqs)), float(np.ptp(freqs)), len(freqs)],
        "cz_pulse_amplitudes": [amplitude, 0.0, 1], "cz_pulse_duration": duration}, shots=1)
    total = np.zeros(len(freqs))
    for weight, branch in parity_branches(theta):
        _, control, _, _, _ = multimode_nodes._populations(branch, config, action)
        total += weight * np.asarray(control)[..., 2].ravel()
    return total


def exact_map(theta, config, bias, freqs, amps, duration, workers):
    with ProcessPoolExecutor(workers) as pool:
        return np.array(list(pool.map(row, [(theta, config, bias, freqs, a, duration) for a in amps], chunksize=1)))


def executed_axis(axes, name):
    axis = axes[name]
    return np.linspace(axis["min"], axis["max"], axis["n"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", type=Path, help="one episode directory (transcript.json, episode.pkl, vision/)")
    parser.add_argument("--turn", type=int, required=True, help="the turn of the cz_parametrization")
    parser.add_argument("--fine", type=int, nargs=2, default=(101, 145), metavar=("N_AMP", "N_FREQ"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    turns = json.loads((args.episode / "transcript.json").read_text())["turns"]
    turn = next(t for t in turns if t["index"] == args.turn)
    action, analysis = turn["action"], turn["observation"]["last_analysis"]
    if action.get("node") != "cz_parametrization" or analysis.get("node") != "cz_parametrization":
        parser.error(f"turn {args.turn} is not a cz_parametrization")
    with open(args.episode / "episode.pkl", "rb") as handle:
        chain_env = pickle.load(handle)._env
    index = chain_env.coupler_labels().index(action["coupler"])
    theta = chain_env._theta(index)
    # The executed grid (after any guardrail clamping), not the requested one.
    axes = analysis["axes"]
    bias = float(axes["dc_currents"]["min"])
    freqs, amps = executed_axis(axes, "cz_pulse_frequencies"), executed_axis(axes, "cz_pulse_amplitudes")
    duration = float(action["sweeps"].get("cz_pulse_duration", 280e-9))
    config = replace(chain_env.configs[index], parking_current=bias)
    coarse = exact_map(theta, config, bias, freqs, amps, duration, args.workers)
    fine_freqs = np.linspace(freqs[0], freqs[-1], args.fine[1])
    fine_amps = np.linspace(amps[0], amps[-1], args.fine[0])
    fine = exact_map(theta, config, bias, fine_freqs, fine_amps, duration, args.workers)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    seen = sorted(args.episode.glob(f"vision/*/turn_{args.turn:03d}_{action['coupler']}_cz_parametrization.png"))
    fig, panels = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.05, 1, 1.12]))
    if seen:
        panels[0].imshow(mpimg.imread(seen[0]))
    panels[0].axis("off")
    panels[0].set_title(f"(a) what the agent saw: {len(amps)} × {len(freqs)} points,\n"
                        f"{analysis.get('shots')} shots each, three-state readout errors", fontsize=10, loc="left")
    for ax, (a, f, p), title in ((panels[1], (amps, freqs, coarse), "(b) same grid, exact P(control in |2>):\nno shot noise, no readout error"),
                                 (panels[2], (fine_amps, fine_freqs, fine), f"(c) exact, fine grid ({len(fine_amps)} × {len(fine_freqs)})")):
        mesh = ax.pcolormesh(a, f / 1e6, p.T, shading="nearest", vmin=0, vmax=1, cmap="viridis", rasterized=True)
        ax.set_xlabel("Pulse amplitude")
        ax.set_ylabel("Drive frequency (MHz)")
        ax.set_title(title, fontsize=10, loc="left")
    fig.colorbar(mesh, ax=panels[2], label="P2")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
