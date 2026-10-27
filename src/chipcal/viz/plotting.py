from __future__ import annotations
import os, re, tempfile
from pathlib import Path
from typing import Optional
import numpy as np
import xarray as xr

def safe_stem(text: str) -> str:
    """Filesystem-safe stem for plot names."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip()).strip("_")
    return stem or "plot"


def _plt():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "chipcal_mplconfig"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    _plt().close(fig)
    return path


def _vars(ds: xr.Dataset) -> tuple[str, Optional[str]]:
    names = [name for name in ds.data_vars if name.startswith("y_")]
    if not names:
        raise ValueError("dataset has no data variables")
    return names[0], names[1] if len(names) > 1 else None


def _prob(ds: xr.Dataset, var: str) -> np.ndarray:
    shots = float(ds.attrs.get("shots", 1.0))
    return ds[var].values.astype(float) / max(shots, 1.0)


def _imshow(ax, x, y, z, xlabel: str, ylabel: str, title: str, vmin=None, vmax=None):
    # Place measured samples at their actual coordinates, including irregular
    # adaptive grids. imshow stretched endpoints and treated every interval alike.
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    ix, iy = np.argsort(x), np.argsort(y)
    x, y, z = x[ix], y[iy], z[np.ix_(iy, ix)]
    if min(len(x), len(y)) == 1 or np.any(np.diff(x) == 0) or np.any(np.diff(y) == 0):
        # A one-frequency refinement is a legitimate acquisition. A mesh would
        # have zero height, hiding every measured point; show the samples instead.
        xx, yy = np.meshgrid(x, y)
        im = ax.scatter(xx.ravel(), yy.ravel(), c=z.ravel(), marker="s", s=40,
                        vmin=vmin, vmax=vmax, rasterized=True)
    else:
        im = ax.pcolormesh(x, y, z, shading="nearest", vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{len(x)} × {len(y)} measured points")
    return im


def _frequency_tiles(freqs: np.ndarray, n_tiles: int) -> list[np.ndarray]:
    n = int(len(freqs))
    if n == 0:
        return []
    n_tiles = max(1, min(int(n_tiles), n))
    return [chunk for chunk in np.array_split(np.arange(n), n_tiles) if len(chunk)]


# One recessive hue for a measured trace, one accent for what analysis picked out.
_TRACE = "#1f6f8b"
_ACCENT = "#d1495b"


def _overlay_analysis(ax, frequencies, trace, analysis, dip: bool) -> None:
    """Draw what the node analysis found in one trace: its peaks and its fitted line.

    A first-light feedline sweep can show five dips at once, so only the fitted line
    is labelled; the rest are dashed markers and the trace stays readable.
    """
    from chipcal.envs.one_qubit_discovery import baseline, lorentzian

    candidates = analysis.get("candidates") or []
    fit = analysis.get("fit")
    # Markers sit behind the trace: a 2 MHz line inside a 450 MHz sweep is a few
    # pixels wide, and a rule drawn over it would be all the reader sees.
    for candidate in candidates:
        ax.axvline(candidate["frequency_hz"] * 1e-9, color=_ACCENT, lw=.9, ls=(0, (4, 3)),
                   alpha=.5, zorder=.5)
    if fit is None:
        strongest = max(candidates, key=lambda c: c["contrast"], default=None)
        if strongest is not None:
            ax.axvline(strongest["frequency_hz"] * 1e-9, color=_ACCENT, lw=1.0, alpha=.7, zorder=.5,
                       label=f"peak {strongest['frequency_hz'] * 1e-9:.6f} GHz")
        return
    # The fit is drawn over the window it was taken from, not extrapolated across
    # a sweep that may hold four other resonators.
    span = 5 * fit["width_hz"]
    x = np.linspace(max(fit["frequency_hz"] - span, frequencies[0]),
                    min(fit["frequency_hz"] + span, frequencies[-1]), 400)
    level = baseline(trace, dip)
    curve = lorentzian(x, fit["frequency_hz"], fit["width_hz"], fit["height"], fit["floor"])
    ax.axvline(fit["frequency_hz"] * 1e-9, color=_ACCENT, lw=1.0, alpha=.7, zorder=.5)
    ax.plot(x * 1e-9, level - curve if dip else level + curve, color=_ACCENT, lw=1.9, alpha=.9,
            zorder=3, label=f"fit {fit['frequency_hz'] * 1e-9:.6f} GHz · "
                            f"half-width {fit['width_hz'] * 1e-6:.2f} MHz")


def plot_discovery(ds: xr.Dataset, path: Path) -> Path:
    """Render a first-light scan the way the instrument's own analysis reads it.

    A readout scan is |S21| against frequency, where a resonator is a notch. A
    two-tone scan is the frequency-by-amplitude map it actually measures, so power
    broadening and the neighbours that only answer to a loud tone stay visible, with
    the amplitude row the analysis read drawn underneath it.
    """
    from chipcal.envs.one_qubit_discovery import analyze

    plt = _plt()
    node = str(ds.attrs.get("node", "dataset"))
    shots = int(ds.attrs.get("shots", 0) or 0)
    dip = node == "resonator_spectroscopy"
    raw = ds.frequency.values
    frequencies = raw * 1e-9
    values = np.asarray(ds.signal.values, dtype=float)
    analysis = analyze(ds)
    trace = np.asarray(analysis["signal"], dtype=float)
    header = f"{node} · {ds.attrs.get('qubit', '')}"

    if values.ndim == 1:
        fig, ax = plt.subplots(figsize=(9.0, 4.4))
        ax.plot(frequencies, values, color=_TRACE, lw=1.3, marker=".", ms=2.6, zorder=2)
        _overlay_analysis(ax, raw, values, analysis, dip)
        ax.set_xlabel("Readout frequency (GHz)" if dip else "Drive frequency (GHz)")
        ax.set_ylabel("|S21| (a.u.)" if dip else "Excited-state response")
        ax.set_title(f"{header} · {len(frequencies)} points · {shots} shots")
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=9)
        return _finish(fig, path)

    # Pulse amplitudes are volts; millivolts keep the row labels short.
    amplitudes = ds.drive_amplitude.values * 1e3
    read_at = float(analysis.get("drive_amplitude", ds.drive_amplitude.values[0])) * 1e3
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=True,
                                      gridspec_kw=dict(height_ratios=(2.0, 1.0), hspace=.09))
    mesh = top.pcolormesh(frequencies, amplitudes, values, shading="nearest",
                          cmap="viridis", rasterized=True)
    top.set_yticks(amplitudes)
    top.set_yticklabels([f"{a:.1f}" for a in amplitudes])
    top.set_ylabel("Spectroscopy amplitude (mV)")
    top.set_title(f"{header} · {len(amplitudes)} × {len(frequencies)} points · {shots} shots")
    fig.colorbar(mesh, ax=[top, bottom], label="Excited-state response", pad=.02)
    found = analysis.get("fit") or max(analysis.get("candidates") or [],
                                       key=lambda c: c["contrast"], default=None)
    if found is not None:
        # Where the analysis read the line: this frequency, in this amplitude row. The
        # white pass underneath keeps the cross legible over both ends of the ramp.
        for colour, width in (("white", 4.2), (_ACCENT, 2.0)):
            top.plot(found["frequency_hz"] * 1e-9, read_at, marker="x", ls="none",
                     ms=12, mec=colour, mew=width)
    bottom.plot(frequencies, trace, color=_TRACE, lw=1.3, zorder=2,
                label=f"{read_at:.1f} mV row read by analysis")
    _overlay_analysis(bottom, raw, trace, analysis, dip)
    bottom.set_xlabel("Drive frequency (GHz)")
    bottom.set_ylabel("Response")
    bottom.grid(alpha=.18)
    bottom.legend(frameon=False, fontsize=9)
    return _finish(fig, path)


def plot_dataset(ds: xr.Dataset, out_dir: str | Path, stem: Optional[str] = None) -> Path:
    """Render one environment dataset to a PNG and return the output path."""
    node = str(ds.attrs.get("node", "dataset"))
    out_dir = Path(out_dir)
    path = out_dir / f"{safe_stem(stem or node)}.png"
    if ds.attrs.get("discovery"):
        return plot_discovery(ds, path)
    control_var, target_var = _vars(ds)
    plt = _plt()

    if node == "coupler_dc_spectroscopy":
        from chipcal.envs.observation_analysis import spectroscopy_probe_var
        currents = ds["dc_currents"].values * 1e6
        freqs = ds["qubit_frequencies"].values * 1e-9
        probe_var, indirect = spectroscopy_probe_var(ds, control_var, target_var)
        # P1 is a probability: pin the colour scale so a pure-noise field reads as noise
        # instead of being auto-stretched into apparent structure.
        if indirect and target_var:
            drive = "control" if ds.attrs.get("drive_qubit") == 0 else "target"
            fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
            for ax, var in zip(axes, (control_var, target_var)):
                probed = " (probed)" if var == probe_var else ""
                im = _imshow(ax, freqs, currents, _prob(ds, var)[1],
                             "Qubit drive frequency (GHz)", "DC current (uA)",
                             f"{var[2:]} P1{probed}; drive {drive}",
                             vmin=0, vmax=1)
                fig.colorbar(im, ax=ax, label="P1")
            return _finish(fig, path)
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        im = _imshow(ax, freqs, currents, _prob(ds, probe_var)[1],
                     "Qubit drive frequency (GHz)", "DC current (uA)",
                     f"{probe_var[2:]} P1 spectroscopy", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label="P1")
        return _finish(fig, path)

    if node == "cz_parametrization":
        p2 = _prob(ds, control_var)[2]
        currents = ds["dc_currents"].values * 1e6
        freqs = ds["cz_pulse_frequencies"].values * 1e-6
        amps = ds["cz_pulse_amplitudes"].values
        n_show = min(int(p2.shape[0]), 4)
        fig, axes = plt.subplots(1, n_show, figsize=(5.2 * n_show, 4.5), squeeze=False)
        vmin, vmax = float(np.min(p2[:n_show])), float(np.max(p2[:n_show]))
        ims = []
        for idx in range(n_show):
            ax = axes[0, idx]
            ims.append(_imshow(
                ax,
                amps,
                freqs,
                p2[idx],
                "Pulse amplitude",
                "Drive frequency (MHz)",
                f"Control P2 transfer, I={currents[idx]:.1f} uA",
                vmin=vmin,
                vmax=vmax,
            ))
        fig.colorbar(ims[-1], ax=axes.ravel().tolist(), label="P2")
        return _finish(fig, path)

    if node == "cz_chevron":
        p2 = _prob(ds, control_var)[2]
        freqs = ds["cz_pulse_frequencies"].values * 1e-6
        durations = ds["cz_pulse_durations"].values * 1e9
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        im = _imshow(ax, durations, freqs, p2, "Pulse duration (ns)", "Drive frequency (MHz)", "Chevron: control P2 transfer")
        fig.colorbar(im, ax=ax, label="P2")
        return _finish(fig, path)

    if node == "cz_calibration":
        if target_var is None:
            target_var = control_var
        p1 = _prob(ds, target_var)[1]
        phases = ds["ramsey_phases"].values
        n_wp = int(ds.sizes["working_points"])
        cols = min(3, n_wp)
        rows = int(np.ceil(n_wp / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.2 * rows), squeeze=False)
        freqs = ds.attrs.get("wp_frequencies", [None] * n_wp)
        times = ds.attrs.get("wp_durations", [None] * n_wp)
        for w in range(n_wp):
            ax = axes[w // cols, w % cols]
            ax.plot(phases, p1[w, 0], marker="o", ms=3, label="control off")
            ax.plot(phases, p1[w, 1], marker="o", ms=3, label="control on")
            f_txt = f"{float(freqs[w]) * 1e-6:.2f} MHz" if freqs[w] is not None else f"wp {w}"
            t_txt = f"{float(times[w]) * 1e9:.1f} ns" if times[w] is not None else ""
            ax.set_title(f"{f_txt}, {t_txt}")
            ax.set_xlabel("Ramsey phase (deg)")
            ax.set_ylabel("Target P1")
            ax.set_ylim(-0.05, 1.05)
            if w == 0:
                ax.legend(frameon=False, fontsize=8)
        for w in range(n_wp, rows * cols):
            axes[w // cols, w % cols].axis("off")
        return _finish(fig, path)

    if node == "cz_local_phases":
        if target_var is None:
            target_var = control_var
        phases = ds["local_phases"].values
        p1_c = _prob(ds, control_var)[1]
        p1_t = _prob(ds, target_var)[1]
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
        panels = [
            ("Target local phase", p1_t[:, 0]),
            ("Control local phase", p1_c[:, 1]),
        ]
        for ax, (title, arr) in zip(axes, panels):
            ax.plot(phases, arr[0], marker="o", ms=3, label="CZ off")
            ax.plot(phases, arr[1], marker="o", ms=3, label="CZ on")
            ax.set_title(title)
            ax.set_xlabel("Analysis phase (deg)")
            ax.set_ylabel("P1")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(frameon=False, fontsize=8)
        return _finish(fig, path)

    if node == "cz_rb":
        counts = ds["y_survival"].values.astype(float)
        shots = float(ds.attrs.get("shots", 1.0))
        lengths = ds["number_of_cliffords"].values
        y = counts.mean(axis=1) / shots
        err = counts.std(axis=1, ddof=1) / shots / np.sqrt(counts.shape[1])
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for idx, label in enumerate(("reference", "interleaved")):
            ax.errorbar(lengths, y[idx], yerr=err[idx], marker="o", capsize=2, label=label)
        ax.set_xlabel("Number of Cliffords")
        ax.set_ylabel("Survival")
        ax.set_title("Standardized IRB samples")
        ax.set_ylim(0.2, 1.02)
        ax.legend(frameon=False)
        return _finish(fig, path)

    # Generic fallback: first variable, first state, flattened to 2D.
    arr = _prob(ds, control_var)[0]
    img = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr[None, :]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    im = ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_title(f"{node}: {control_var} state 0")
    fig.colorbar(im, ax=ax)
    return _finish(fig, path)
