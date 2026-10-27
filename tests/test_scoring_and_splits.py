import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from chipcal import scoring, splits, tiers
from chipcal.baselines.discovery import ScriptedDiscovery
from chipcal.baselines.probes import invalid_spam, random_valid_actions, stop_immediately

ROOT = Path(__file__).resolve().parents[1]


def test_score_is_the_minimum_final_rb_fidelity():
    assert scoring.episode_score([0.97, 0.91]) == pytest.approx(0.91)
    assert scoring.episode_score([]) == 0.0


def test_final_rb_scores_an_idle_gate_at_the_floor_and_a_good_gate_near_its_true_fidelity():
    from chipcal.core.config import CommittedConfig
    from chipcal.sim import chain as chainmod
    from chipcal.sim.multimode import gate_fidelity
    chain = tiers.sample("2q1c", 3_000_001).chain
    theta = chainmod.coupler_theta(chain, 0, [1.325e-3])
    rng = np.random.default_rng(0)
    idle = CommittedConfig(parking_current=1.325e-3, cz_pulse_frequency=312.4e6, cz_pulse_amplitude=0.001,
                           cz_pulse_duration=376e-9, control_local_phase=0.0, target_local_phase=0.0)
    assert scoring.final_rb_fidelity(theta, idle, rng)[0] == scoring.RB_FLOOR
    good = CommittedConfig(parking_current=1.325e-3, cz_pulse_frequency=312.382e6, cz_pulse_amplitude=0.2,
                           cz_pulse_duration=376e-9, control_local_phase=0.0999, target_local_phase=6.182)
    f, details = scoring.final_rb_fidelity(theta, good, rng)
    assert details["fit_reliable"] and f == pytest.approx(gate_fidelity(good, theta)[0], abs=0.01)
    assert scoring.final_rb_fidelity(theta, CommittedConfig(parking_current=1e-3), rng)[0] == 0.0


@pytest.mark.parametrize("probe", [stop_immediately, invalid_spam])
def test_degenerate_policies_score_zero(probe):
    for seed in (3_000_000, 3_000_001):
        assert probe("2q1c", seed)["score"] == 0.0


def test_random_actions_score_at_most_the_floor():
    for seed in (3_000_000, 3_000_001):
        assert random_valid_actions("2q1c", seed)["score"] <= scoring.RB_FLOOR


# Includes chips whose design anharmonicity is far off (3q2c 3000008, 3000016) or large (5q4c 3000012).
@pytest.mark.parametrize("tier,seeds", [("2q1c", (3_000_000, 3_000_001, 3_000_002)),
                                        ("3q2c", (3_000_008, 3_000_016)), ("5q4c", (3_000_012,))])
def test_scripted_discovery_finds_every_line(tier, seeds):
    for seed in seeds:
        env = tiers.make_env(tier)
        obs = env.reset(seed=seed, chain=tiers.sample(tier, seed).chain)
        ScriptedDiscovery(env.step).run(obs)
        assert env._env.discovery.ready and env._env.discovery.accuracy() == 1.0
        assert env._n_invalid == 0


def test_frozen_split_verifies_and_every_chip_is_calibratable():
    directory = ROOT / "data" / "v0.1"
    if not directory.exists():
        pytest.skip("no frozen split built")
    assert splits.verify(directory, regenerate=2) == []
    manifest = json.loads((directory / "manifest.json").read_text())
    for tier in tiers.TIERS:
        rows = [json.loads(line) for line in (directory / f"{tier}-test.jsonl").read_text().splitlines()]
        assert len(rows) == manifest["files"][f"{tier}-test.jsonl"]["chips"]
        assert all(len(r["ceiling"]) == tiers.TIERS[tier].n_qubits - 1 for r in rows)
        assert min(c["fidelity"] for r in rows for c in r["ceiling"]) >= splits.MIN_CEILING
        assert [r["seed"] for r in rows] == [chip["seed"] for chip in splits.benchmark(tier)]


def test_build_keeps_only_chips_whose_every_coupler_reaches_the_ceiling(tmp_path):
    lo = tiers.SPLITS["test"][0]
    fidelities = {"2q1c": [[0.97], [0.94], [0.99]], "3q2c": [[0.97, 0.98], [0.97, 0.95], [0.96, 0.99]],
                  "5q4c": [[0.97] * 4, [0.97, 0.97, 0.9, 0.97], [0.98] * 4]}
    lines = [json.dumps({"tier": tier, "seed": lo + k, "coupler": c, "search": "best_bias", "fidelity": f,
                         **{key: 0.0 for key in splits.CEILING_FIELDS if key != "fidelity"}})
             for tier, chips in fidelities.items() for k, chip in enumerate(chips) for c, f in enumerate(chip)]
    (tmp_path / "ceilings.jsonl").write_text("\n".join(lines))
    index = tmp_path / "index.json"
    manifest = splits.build(tmp_path / "out", 2, [tmp_path / "ceilings.jsonl"], index_path=index)
    kept = json.loads(index.read_text())["tiers"]
    assert {tier: [c["seed"] for c in chips] for tier, chips in kept.items()} == {t: [lo, lo + 2] for t in fidelities}
    assert manifest["files"]["2q1c-test.jsonl"]["seeds_scanned"] == 3


def test_normalized_score_compares_each_gate_with_its_own_ceiling():
    assert splits.normalized_score({"c0": 0.95, "c1": 0.90}, [0.99, 0.91]) == pytest.approx(0.95 / 0.99)
    assert splits.normalized_score({"c0": 0.95}, None) is None


@pytest.mark.skipif(importlib.util.find_spec("verifiers") is None, reason="verifiers not installed")
def test_verifiers_environment_resolves():
    import sys
    sys.path.insert(0, str(ROOT / "environments" / "tergite_chipcal_env"))
    from verifiers.v1.utils.loaders import environment_class, taskset_class
    import tergite_chipcal_env
    assert taskset_class("tergite_chipcal_env") is tergite_chipcal_env.ChipcalTaskset
    assert environment_class("tergite_chipcal_env") is tergite_chipcal_env.ChipcalEnv
    tasks = tergite_chipcal_env.ChipcalTaskset(tergite_chipcal_env.ChipcalTasksetConfig(id="tergite_chipcal_env", n_chips=2)).load()
    assert [t.data.seed for t in tasks] == [chip["seed"] for chip in splits.benchmark("2q1c")[:2]]
