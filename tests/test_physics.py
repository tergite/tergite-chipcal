from dataclasses import replace

import numpy as np
import pytest

from chipcal import tiers
from chipcal.core.config import CommittedConfig
from chipcal.sim import chain as chainmod
from chipcal.sim.multimode import Cell, gate_fidelity, noise_branches
from chipcal.sim.multimode_rb import run_irb
from chipcal.sim.realism import frame_rb_contraction, parity_branches


def reference_point(tier="2q1c", seed=3_000_001, index=0, bias=None):
    chain = tiers.sample(tier, seed).chain
    currents = chainmod.ChainPrior(n_qubits=chain.n_qubits).initial_parking_currents(chain, np.random.default_rng(seed))
    if bias is not None:
        currents[index] = bias
    return chain, currents, chainmod.coupler_theta(chain, index, list(currents))


def test_fast_solvers_match_reference():
    _, currents, theta = reference_point()
    cell = Cell(theta, currents[0], replace(theta.multimode, spectroscopy_segments=48))
    rho = cell.state(1, 1)
    f0 = abs(theta.f01_t - theta.f12_c)
    batch = cell.flux_pulse_many(rho, [f0, f0 + 6e6], [.5, .3], 300e-9)
    for i, (f, a) in enumerate([(f0, .5), (f0 + 6e6, .3)]):
        # A batch shares its highest member's (finer) time grid, so it differs from the per-member reference
        # by that solver's own discretization error (both sit ~2e-3 from a fully converged solution).
        np.testing.assert_allclose(batch[i], cell.flux_pulse_reference(rho, f, a, 300e-9), atol=2e-4)
    f01 = cell.energies[cell.idx(target=1)] - cell.energies[cell.idx()]
    freqs = f01 + np.linspace(-3e6, 3e6, 5)
    control, target = cell.spectroscopy_many(freqs, drive=1, amplitude=.003)
    full = [cell.spectroscopy(f, drive=1, amplitude=.003) for f in freqs]
    np.testing.assert_allclose(control, [c for c, _ in full], atol=5e-3)
    np.testing.assert_allclose(target, [t for _, t in full], atol=5e-3)


def test_fast_pulse_stays_trace_preserving_when_the_coupler_sweeps_through_the_qubits():
    # Bias 0 parks this coupler between the qubits; a strong pulse swings it over ~5 GHz.
    chain = tiers.sample("2q1c", 3_000_000).chain
    theta = chainmod.coupler_theta(chain, 0, [0.0])
    cell = Cell(theta, 0.0)
    ref = Cell(theta, 0.0, replace(theta.multimode, pulse_solver="reference"))
    f0 = abs(theta.f01_t - theta.f12_c)
    fast = cell.flux_pulse_many(cell.state(1, 1), [f0], [.9], 400e-9)[0]
    assert np.trace(fast).real == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.eigvalsh(fast).min() > -1e-9
    np.testing.assert_allclose(fast, ref.flux_pulse_reference(ref.state(1, 1), f0, .9, 400e-9), atol=5e-5)
    cell.marginals(fast)   # the positivity/trace guard accepts it


def test_noise_branches_are_a_normalised_mixture():
    _, _, theta = reference_point("5q4c", 3_000_002, index=1)
    branches = noise_branches(theta)
    assert sum(w for w, _ in branches) == pytest.approx(1.0)
    assert len(branches) == 2 * len(parity_branches(theta))   # flux noise doubles the parity branches


@pytest.fixture(scope="module")
def best_gate():
    """A good CZ on one test chip, found once by the privileged oracle (a coarse search)."""
    from chipcal.baselines.oracle import coupler_ceiling
    chain, currents, theta = reference_point()
    best = coupler_ceiling(chain, 0, currents, amplitudes=(0.35, 0.5), max_duration=480e-9, span=12e6)
    config = CommittedConfig(parking_current=currents[0], cz_pulse_frequency=best["cz_pulse_frequency"],
                             cz_pulse_amplitude=best["cz_pulse_amplitude"], cz_pulse_duration=best["cz_pulse_duration"],
                             control_local_phase=best["control_local_phase"], target_local_phase=best["target_local_phase"])
    return theta, currents, best, config


@pytest.mark.slow
def test_local_phase_convention_and_incomplete_gates(best_gate):
    theta, currents, best, config = best_gate
    assert best["fidelity"] > 0.9
    f, diag = gate_fidelity(config, theta)
    assert f == pytest.approx(best["fidelity"], abs=2e-3)
    assert 0 <= diag["leakage"] < 0.05
    negated = replace(config, control_local_phase=-config.control_local_phase,
                      target_local_phase=-config.target_local_phase)
    assert gate_fidelity(negated, theta)[0] < f - 0.05
    assert gate_fidelity(CommittedConfig(parking_current=currents[0]), theta)[0] == 0.0


@pytest.mark.slow
def test_irb_lower_bound_is_conservative(best_gate):
    theta, currents, best, config = best_gate
    truth = gate_fidelity(config, theta)[0]
    rng = np.random.default_rng(0)
    results = [run_irb(theta, config, rng) for _ in range(20)]
    assert np.mean([r.f_hat for r in results]) == pytest.approx(truth, abs=0.01)
    assert sum(r.f_lcb > truth for r in results) <= 2


def test_irb_refuses_an_unidentifiable_reference_decay():
    _, currents, theta = reference_point()
    bad = CommittedConfig(parking_current=currents[0], cz_pulse_frequency=50e6, cz_pulse_amplitude=0.9,
                          cz_pulse_duration=1.5e-6, control_local_phase=1.0, target_local_phase=2.0)
    result = run_irb(theta, bad, np.random.default_rng(1))
    assert result.f_lcb <= max(gate_fidelity(bad, theta)[0], 0.0) + 0.05


def test_frame_offsets_contract_rb_like_a_z_rotation():
    assert frame_rb_contraction(0.0, 0.0, 200e-9) == pytest.approx(1.0)
    assert frame_rb_contraction(1e5, 0.0, 200e-9) < frame_rb_contraction(1e4, 0.0, 200e-9) < 1.0


def test_realism_drift_is_seeded_continuous_and_telegraph():
    chain = tiers.sample("3q2c", 3_000_003).chain
    r = chain.realism
    for q in range(chain.n_qubits):
        assert r.drift(q, 1000.0) == r.drift(q, 1000.0)
        assert abs(r.drift(q, 1000.0) - r.drift(q, 1000.001)) < 1e3   # only a telegraph flip can jump
        assert abs(r.drift(q, 0.0)) <= 0.5 * sum(r.noise(q).tls_jumps_hz) + 1e-9
    split = [r.parity_splits(q, t) for q in range(chain.n_qubits) for t in (0.0, 299.0, 301.0)]
    assert all(s01 >= 0 and s12 >= s01 for s01, s12 in split)


def test_a_shared_qubit_is_dressed_by_both_couplers_on_f01_and_f12():
    chain, currents, _ = reference_point("3q2c", 3_000_004)
    middle = 1
    dressed = chainmod.dressed_qubits(chain, currents)[middle]
    for index in (0, 1):
        theta = chainmod.coupler_theta(chain, index, currents)
        cell = Cell(theta, currents[index])
        role = "c" if chain.coupler_qubits(index)[0] == middle else "t"
        idx1 = cell.idx(1) if role == "c" else cell.idx(target=1)
        idx2 = cell.idx(2) if role == "c" else cell.idx(target=2)
        e = cell.energies
        # Pulls add to first order; the residue is second order (kHz), far below the MHz-scale
        # error of ignoring a neighbour's f12 pull.
        assert e[idx1] - e[cell.idx()] == pytest.approx(dressed.f01, abs=20e3)
        assert e[idx2] - e[idx1] == pytest.approx(dressed.f01 + dressed.alpha, abs=20e3)


def test_ceiling_bias_scan_keeps_every_grid_coupler_above_its_own_qubits():
    import numpy as np
    from chipcal import tiers
    from chipcal.baselines.oracle import MIN_COUPLER_ABOVE_QUBITS_HZ, flux_scan
    chain = tiers.sample("10q-hard", 3_000_000).chain
    for i in range(chain.n_couplers):
        c = chain.couplers[i]
        phi = flux_scan(chain, i)
        fc = c.fc0 * (np.cos(np.pi * phi) ** 2 + c.d_asym ** 2 * np.sin(np.pi * phi) ** 2) ** 0.25
        assert phi.size and fc.min() >= max(chain.qubits[q].f01 for q in chain.pair(i)) + MIN_COUPLER_ABOVE_QUBITS_HZ
