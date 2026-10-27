"""Measurement sequences on the shared multimode cell (node names follow tergite-autocalibration).

The y_* variables remain three-state counts for existing policies. iq_* and
iq_std_* are complex integrated-IQ statistics of the very same simulated shots.
No hidden Hamiltonian parameters or exact populations enter those variables.

The independent slices of a sweep (DC-spectroscopy bias rows, chevron durations, parametrization grid chunks)
run on CHIPCAL_SIM_WORKERS processes (default 1, serial). The populations are identical either way, and the
shots are drawn afterwards in this process, so a measurement does not depend on the worker count. The workers
start by forkserver, which re-imports the calling script: guard its entry point with ``if __name__ == "__main__"``.
"""
import multiprocessing
import multiprocessing.util
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr

from chipcal.core.actions import axis_values
from chipcal.sim import readout
from chipcal.sim.multimode import Cell, conjugate, rotation, pair_populations


_POOL = None


def _workers() -> int:
    return max(1, int(os.environ.get("CHIPCAL_SIM_WORKERS", "1")))


def _map(fn, jobs):
    """``[fn(*job) for job in jobs]``, on the simulation worker pool when there is one."""
    global _POOL
    jobs = list(jobs)
    if _workers() == 1 or len(jobs) < 2:
        return [fn(*job) for job in jobs]
    if _POOL is None:
        # forkserver, not fork: an episode often runs inside another pool's worker, whose threads make fork unsafe.
        _POOL = ProcessPoolExecutor(_workers(), mp_context=multiprocessing.get_context("forkserver"))
        multiprocessing.util.Finalize(None, _close_pool, exitpriority=10)
    return list(_POOL.map(fn, *zip(*jobs)))


def _close_pool():
    """Stop the simulation workers. A process that runs episodes inside another pool exits by joining its children,
    and a waiting shutdown of this pool never returns there, so its (idle) workers are terminated instead."""
    if _POOL is not None:
        workers = list(_POOL._processes.values())
        _POOL.shutdown(wait=False, cancel_futures=True)
        for worker in workers:
            worker.terminate()


def _spectroscopy_row(theta, bias, frequencies, drive, transition, duration, amplitude):
    return Cell(theta, bias).spectroscopy_many(frequencies, drive=drive, transition=transition, duration=duration,
                                               amplitude=amplitude)


def _pulse_marginals(theta, bias, frequencies, amplitudes, duration):
    cell = Cell(theta, bias)
    return cell.marginals(cell.flux_pulse_many(cell.state(1, 1), frequencies, amplitudes, duration))


def _ramsey_input(cell, measured=1, control_on=0):
    v = np.zeros(cell.dim, complex)
    if measured == 1:
        v[cell.idx(control_on)] = 1/np.sqrt(2)
        v[cell.idx(control_on, target=1)] = -1j/np.sqrt(2)
    else:
        v[cell.idx()] = 1/np.sqrt(2)
        v[cell.idx(1)] = -1j/np.sqrt(2)
    v = cell.dress @ v
    return np.outer(v, v.conj())


def _analysis(cell, rho, phases, measured=1, undo_control=False):
    pair = cell.pair_density(rho)
    results = [[], []]
    for phase in phases:
        analyze = rotation(np.pi/2, np.deg2rad(phase))
        u = np.kron(rotation(np.pi) if undo_control else np.eye(3), analyze) if measured == 1 else np.kron(analyze, np.eye(3))
        pc, pt = pair_populations(conjugate(u, pair))
        results[0].append(pc); results[1].append(pt)
    return tuple(np.asarray(x) for x in results)


def _iq_centers(cell, qubit):
    """Linear dispersive readout away from a qubit–resonator collision.

    The bus dresses the addressed qubit frequency and its local oscillator
    strength. This changes the readout pull even though the fabricated qubit is
    fixed. The reference classifier is calibrated at zero applied current.
    """
    s = cell.settings
    q = 0 if qubit == 0 else 2
    idx = cell.idx(1) if q == 0 else cell.idx(target=1)
    f = cell.energies[idx]
    alpha = cell.theta.alpha_c if q == 0 else cell.theta.alpha_t
    fr = s.readout_frequencies[qubit]
    det = f-fr
    g = s.readout_coupling_hz*abs(cell.dress[idx, idx])
    if min(abs(det), abs(det+alpha)) < 5*abs(g):
        raise ValueError("readout resonator collision requires an explicit resonator mode; dispersive IQ model is invalid here")
    chi = g*g*alpha/(det*(det+alpha))
    pulls = chi*(2*np.arange(3)-1)
    # Drive at nominal fr, fixed complex gain and cable phase.
    return np.exp(0.35j)*(1 - .85/(1+2j*pulls/s.readout_linewidth_hz))


def _acquire(node, axes, pc, pt, cells, theta, shots, rng, qubits, noise, attrs):
    shape = pc.shape[:-1]
    data = {}
    reference = Cell(theta, 0.)
    for q, (label, pops, confusion) in enumerate(zip(qubits, (pc, pt), (theta.confusion_c, theta.confusion_t))):
        # Existing confusion is an empirical stochastic readout/preparation
        # channel, followed by finite-SNR IQ and a fixed reference classifier.
        latent_counts = readout.measure(pops, confusion, shots, rng, noise=noise).reshape(-1, 3)
        counts = np.zeros_like(latent_counts)
        means = np.zeros(len(counts), complex)
        deviations = np.zeros((len(counts), 2))
        reference_centers = _iq_centers(reference, q)
        for i, (count, cell) in enumerate(zip(latent_counts, cells)):
            centers = _iq_centers(cell, q)
            # Scale noise to reference separation: sigma denotes a fraction of
            # neighboring pointer separation, not an instrument voltage spec.
            sigma = cell.settings.iq_sigma * min(abs(np.diff(reference_centers)))
            states = np.repeat(np.arange(3), count)
            iq = centers[states] + sigma*(rng.normal(size=shots) + 1j*rng.normal(size=shots))
            classified = abs(iq[:, None]-reference_centers[None]).argmin(axis=1)
            counts[i] = np.bincount(classified, minlength=3)
            means[i] = iq.mean()
            deviations[i] = [iq.real.std(), iq.imag.std()]
        dims = tuple(axes)
        data[f"y_{label}"] = (("state", *dims), np.moveaxis(counts.reshape(*shape, 3), -1, 0))
        # Real I/Q components avoid complex-to-JSON / NetCDF conversion traps.
        data[f"iq_{label}"] = ((*dims, "quadrature"), np.stack((means.real, means.imag), -1).reshape(*shape, 2))
        data[f"iq_std_{label}"] = ((*dims, "quadrature"), deviations.reshape(*shape, 2))
    return xr.Dataset(data, coords={"state": [0,1,2], **axes, "quadrature": [0, 1]},
        attrs={"node": node, "shots": shots, "simulation_model": "multimode_lindblad_v1",
               "coupler_readout": False, "readout_model": "dispersive_iq_fixed_classifier_v1",
               "iq_units": "normalized complex voltage", "acquisition": "same-shot classified counts and integrated IQ statistics",
               "quadrature_labels": ["I", "Q"],
               **attrs})


def run_node(theta, k, action, rng, qubits, noise=None):
    """Simulate one node and acquire its shots. With charge-parity splitting (chipcal.sim.realism) every shot
    sees one parity branch at random, so the populations the readout samples are the branches' equal mixture."""
    from chipcal.sim.realism import parity_branches
    branches = parity_branches(theta)
    axes, pcs, pts, cells, attrs = _populations(branches[0][1], k, action)
    if len(branches) > 1:
        pcs, pts = branches[0][0]*pcs, branches[0][0]*pts
        for weight, branch in branches[1:]:
            _, pc, pt, _, _ = _populations(branch, k, action)
            pcs, pts = pcs + weight*pc, pts + weight*pt
    return _acquire(action.node, axes, pcs, pts, cells, theta, action.shots, rng, qubits, noise, attrs)


def _populations(theta, k, action):
    """(axes, control populations, target populations, cells, attrs) of one node, before readout."""
    node, sw = action.node, action.sweeps
    current = k.parking_current if k.parking_current is not None else 0.
    cell = Cell(theta, current)
    cells, pcs, pts = [], [], []
    attrs = {}

    def append(c, populations):
        cells.append(c); pcs.append(populations[0]); pts.append(populations[1])

    if node in ("coupler_dc_spectroscopy", "qubit_spectroscopy"):
        currents = axis_values(sw.get("dc_currents", [current]))
        freqs = axis_values(sw["qubit_frequencies"])
        axes = {"dc_currents": currents, "qubit_frequencies": freqs} if node.startswith("coupler") else {"qubit_frequencies": freqs}
        drive, transition = int(sw.get("drive_qubit", 0)), int(sw.get("transition", 1))
        ladder = cell.settings.spectroscopy_model == "dressed_ladder"
        rows = _map(_spectroscopy_row, [(theta, float(bias), freqs, drive, transition, sw.get("spec_pulse_duration"),
                                         sw.get("spec_pulse_amplitude")) for bias in currents]) if ladder else None
        for row, bias in enumerate(currents):
            c = Cell(theta, float(bias))
            if ladder:
                pc, pt = rows[row]
                for i in range(len(freqs)):
                    append(c, (pc[i], pt[i]))
                continue
            for frequency in freqs:
                append(c, c.spectroscopy(frequency, drive=drive, transition=transition,
                       duration=sw.get("spec_pulse_duration"), amplitude=sw.get("spec_pulse_amplitude")))
        attrs.update(drive_qubit=drive, transition=transition,
                     spectroscopy_model="multimode_dressed_ladder_v1" if ladder else "multimode_lindblad_v1",
                     spec_pulse_duration=sw.get("spec_pulse_duration", cell.settings.pulse_duration),
                     spec_pulse_amplitude=sw.get("spec_pulse_amplitude", cell.settings.pulse_amplitude))
    elif node == "cz_parametrization":
        axes = {"dc_currents": axis_values(sw.get("dc_currents", [current])),
                "cz_pulse_frequencies": axis_values(sw["cz_pulse_frequencies"]),
                "cz_pulse_amplitudes": axis_values(sw["cz_pulse_amplitudes"])}
        duration = float(sw.get("cz_pulse_duration", 280e-9))
        f, a = np.meshgrid(axes["cz_pulse_frequencies"], axes["cz_pulse_amplitudes"], indexing="ij")
        chunks = list(zip(np.array_split(f.ravel(), _workers()), np.array_split(a.ravel(), _workers())))
        jobs = [(theta, float(bias), fc, ac, duration) for bias in axes["dc_currents"] for fc, ac in chunks if fc.size]
        parts = iter(_map(_pulse_marginals, jobs))
        for bias in axes["dc_currents"]:
            c = Cell(theta, bias)
            done = [next(parts) for fc, _ in chunks if fc.size]
            pc, pt = np.concatenate([d[0] for d in done]), np.concatenate([d[1] for d in done])
            for i in range(f.size):
                append(c, (pc[i], pt[i]))
        attrs["cz_pulse_duration"] = duration
    elif node == "cz_chevron":
        axes = {"cz_pulse_frequencies": axis_values(sw["cz_pulse_frequencies"]),
                "cz_pulse_durations": axis_values(sw["cz_pulse_durations"])}
        amp = float(sw.get("cz_pulse_amplitude", k.cz_pulse_amplitude))
        freqs, durations = axes["cz_pulse_frequencies"], axes["cz_pulse_durations"]
        grid = np.empty((len(freqs), len(durations), 2, 3))
        columns = _map(_pulse_marginals, [(theta, current, freqs, amp, duration) for duration in durations])
        for j, (pc, pt) in enumerate(columns):
            grid[:, j, 0], grid[:, j, 1] = pc, pt
        for i in range(len(freqs)):
            for j in range(len(durations)):
                append(cell, (grid[i, j, 0], grid[i, j, 1]))
        attrs["cz_pulse_amplitude"] = amp
    elif node == "cz_calibration":
        points = sw["working_points"]
        phases = axis_values(sw.get("ramsey_phases", (0., 360., 25)))
        axes = {"working_points": np.arange(len(points)), "control_ons": [0,1], "ramsey_phases": phases}
        for frequency, duration in points:
            for on in (0,1):
                rho = cell.flux_pulse(_ramsey_input(cell, control_on=on), frequency, k.cz_pulse_amplitude, duration)
                pair = _analysis(cell, rho, phases, undo_control=bool(on))
                for pc, pt in zip(*pair):
                    append(cell, (pc,pt))
        attrs.update(wp_frequencies=[float(p[0]) for p in points], wp_durations=[float(p[1]) for p in points], control_analysis="X after flux pulse when control_on=1")
    elif node == "cz_local_phases":
        phases = axis_values(sw.get("local_phases", (0.,360.,25)))
        axes = {"gate_modes": [0,1], "swap": [0,1], "local_phases": phases}
        for on in (0,1):
            for swap in (0,1):
                rho = cell.flux_pulse(_ramsey_input(cell, measured=1-swap), k.cz_pulse_frequency,
                        k.cz_pulse_amplitude if on else 0., k.cz_pulse_duration)
                for pc, pt in zip(*_analysis(cell, rho, phases, measured=1-swap)):
                    append(cell, (pc,pt))
    elif node in ("qubit_t1", "qubit_ramsey"):
        delays = axis_values(sw["delays"])
        axes = {"delays": delays}
        detuning = float(sw.get("ramsey_detuning", 1e6))
        # Parallel independent characterization prepares both qubits. The
        # Ramsey sequence uses separate addressed experiments to stay in the
        # validated excitation sector and to distinguish spectator response.
        for delay in delays:
            if node == "qubit_t1":
                # Bare local excitation resolves coherent hybridization beating;
                # this is the broadband-preparation characterization protocol.
                append(cell, cell.marginals(cell.idle(cell.state(1,1,dressed=False), delay)))
            else:
                result = []
                for q in (0,1):
                    inp = _ramsey_input(cell, measured=q)
                    rho = cell.idle(inp, delay)
                    rho = conjugate(cell.dress.conj().T, rho)
                    # The clock is the qubit's calibrated frame: the dressed f01 (an ideal
                    # calibration) or, with realism, the calibrated clock with its error.
                    frequency = cell.frame_energies()[q]
                    phase = 360*(frequency+detuning)*delay
                    result.append(_analysis(cell, rho, [-phase], measured=q)[q][0])
                append(cell, tuple(result))
        attrs.update(ramsey_detuning=detuning, preparation="broadband local X for T1; selective dressed X90 for Ramsey")
    elif node == "readout_calibration":
        axes = {"prepared": [0,1,2]}
        for state in (0,1,2):
            append(cell, (np.eye(3)[state], np.eye(3)[state]))
    else:
        raise ValueError(f"multimode backend does not implement discovery node {node!r}")
    shape = tuple(len(v) for v in axes.values())
    return axes, np.array(pcs).reshape(*shape,3), np.array(pts).reshape(*shape,3), cells, attrs
