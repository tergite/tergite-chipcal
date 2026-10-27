import json

import numpy as np
import pytest

from chipcal import tiers
from chipcal.sim import transmon
from chipcal.sim.device_manifest import encode
from chipcal.sim.realism import transmon_ratio


def test_a_seed_names_one_chip_forever():
    for tier in tiers.TIERS:
        a = json.dumps(encode(tiers.sample(tier, 3_000_000)), sort_keys=True)
        b = json.dumps(encode(tiers.sample(tier, 3_000_000)), sort_keys=True)
        assert a == b


def test_splits_are_disjoint_and_enforced():
    ranges = sorted(tiers.SPLITS.values())
    assert all(hi < lo for (_, hi), (lo, _) in zip(ranges, ranges[1:]))
    with pytest.raises(ValueError):
        tiers.sample("2q1c", 3_000_000, split="train")
    with pytest.raises(ValueError):
        tiers.sample("2q1c", 42)
    assert tiers.split_of(1_000_005) == "train"


@pytest.mark.parametrize("tier", list(tiers.TIERS))
def test_sizes_coupling_and_realism(tier):
    t = tiers.TIERS[tier]
    chain = tiers.sample(tier, 3_000_001).chain
    assert chain.n_qubits == t.n_qubits and chain.n_couplers == t.n_qubits - 1
    for c in chain.couplers:
        assert t.g_qc_hz[0] <= c.g_qc_left <= t.g_qc_hz[1]
        assert c.multimode.flux_noise_phi0 > 0
    assert chain.realism is not None and len(chain.realism.qubit_noise) == t.n_qubits
    assert tiers.sample(tier, 3_000_001, realism=False).chain.realism is None


def test_anharmonicity_follows_the_exact_transmon_spectrum():
    for seed in range(3_000_000, 3_000_010):
        for q in tiers.sample("5q4c", seed).chain.qubits:
            ratio = transmon_ratio(q)
            assert 36.0 <= ratio <= 101.0
            assert transmon.from_f01(q.f01, ratio)["alpha_hz"] == pytest.approx(q.alpha, rel=1e-4)


def test_no_frequency_collisions():
    for seed in range(3_000_000, 3_000_050):
        qubits = tiers.sample("5q4c", seed, realism=False).chain.qubits
        for a, b in zip(qubits, qubits[1:]):
            control, target = (a, b) if a.f01 >= b.f01 else (b, a)
            assert abs(target.f01 - (control.f01 + control.alpha)) >= tiers.TIERS["5q4c"].min_cz_detuning_hz


def test_transmon_parity_split_matches_a_published_device():
    # A 3.76 GHz, -278 MHz transmon (E_J/E_C 36.5) has a ~109 kHz parity split (Burnett et al. 2019).
    params = transmon.from_f01(3.76e9, 36.5)
    assert params["alpha_hz"] == pytest.approx(-278e6, rel=0.01)
    split01, split12 = transmon.max_parity_splits_hz(params["ec_hz"], 36.5)
    assert split01 == pytest.approx(109e3, rel=0.05)
    assert 20 < split12 / split01 < 30
    # Splits vanish exponentially with E_J/E_C.
    assert transmon.max_parity_splits_hz(transmon.from_f01(4.5e9, 100)["ec_hz"], 100)[0] < 10.0
    assert np.isclose(transmon.parity_splits_hz(params["ec_hz"], 36.5, 0.25)[0], 0.0, atol=1.0)


def test_procedural_tiers_scale_size_and_difficulty():
    hard = tiers.get("8q-hard")                         # a 2x4 grid: 6 couplers along the rows, 4 between them
    assert hard.layout == "grid" and hard.n_couplers == 10
    assert hard.max_turns == 50 + 100 * 10 and hard.budget_seconds == 20 * 3600
    chain = tiers.sample("8q-hard", 1_000_042).chain
    assert chain.n_qubits == 8 and chain.n_couplers == 10 and chain.edges == tuple(hard.edges)
    for i in range(chain.n_couplers):                   # every coupled pair clears the CZ collision filter
        control, target = (chain.qubits[q] for q in chain.coupler_qubits(i))
        assert abs(target.f01 - (control.f01 + control.alpha)) >= hard.min_cz_detuning_hz
    assert tiers.get("5q-hard").layout == "chain" and tiers.get("5q-hard").n_couplers == 4
    g = [x for c in chain.couplers for x in (c.g_qc_left, c.g_qc_right)]
    assert all(hard.weak_g_qc_hz[0] <= x <= hard.g_qc_hz[1] for x in g)
    assert any(x < hard.g_qc_hz[0] for x in g)          # some links drawn from the weak band
    # The named tiers are untouched by the procedural knobs.
    assert tiers.get("5q4c") is tiers.TIERS["5q4c"]
    for name in ("1q-hard", "8q-impossible", "65q-nominal", "five"):
        with pytest.raises(ValueError):
            tiers.get(name)
