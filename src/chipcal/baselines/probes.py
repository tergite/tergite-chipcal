"""Degenerate policies that must score at the floor -- the reward-validity probes.

If any of these scores well, the score can be gamed. Each one runs through the same
ChainAgentEnv as an LLM would.
"""
from __future__ import annotations

import numpy as np

from chipcal import tiers
from chipcal.baselines.discovery import DiscoveryFailed, ScriptedDiscovery
from chipcal.core.config import COMMITTABLE

# A plausible-looking but blind gate: mid-range values, never measured.
DEFAULT_GATE = {"parking_current": 0.0, "cz_pulse_frequency": 300e6, "cz_pulse_amplitude": 0.5,
                "cz_pulse_duration": 200e-9, "control_local_phase": 0.0, "target_local_phase": 0.0}


def _episode(tier, seed, **overrides):
    env = tiers.make_env(tier, **overrides)
    return env, env.reset(seed=int(seed), chain=tiers.sample(tier, seed).chain)


def _finish(env):
    if not env._terminated:
        env.step({"stop": True})
    return env.score()


def stop_immediately(tier, seed):
    env, _ = _episode(tier, seed)
    return _finish(env)


def skip_discovery_commit_defaults(tier, seed):
    """Commit a blind gate on every coupler without discovering anything (rejected), then stop."""
    env, _ = _episode(tier, seed)
    for c in range(env._env.n_couplers):
        for param in COMMITTABLE:
            env.step({"coupler": f"c{c}", "commit": {"param": param, "value": DEFAULT_GATE[param]}})
    return _finish(env)


def discover_then_commit_defaults(tier, seed):
    """Honest discovery, then a blind default gate on every coupler, then stop."""
    env, obs = _episode(tier, seed)
    try:
        ScriptedDiscovery(env.step).run(obs)
    except DiscoveryFailed:
        return _finish(env)
    for c in range(env._env.n_couplers):
        for param in COMMITTABLE:
            if not env._terminated:
                env.step({"coupler": f"c{c}", "commit": {"param": param, "value": DEFAULT_GATE[param]}})
    return _finish(env)


def benchmark_spam(tier, seed, repeats=20):
    """Discover, commit a blind gate, then benchmark repeatedly (the score must ignore lucky RB draws)."""
    env, obs = _episode(tier, seed)
    try:
        ScriptedDiscovery(env.step).run(obs)
    except DiscoveryFailed:
        return _finish(env)
    for c in range(env._env.n_couplers):
        for param in COMMITTABLE:
            env.step({"coupler": f"c{c}", "commit": {"param": param, "value": DEFAULT_GATE[param]}})
    for _ in range(repeats):
        if env._terminated:
            break
        env.step({"coupler": "c0", "benchmark": True})
    return _finish(env)


def random_valid_actions(tier, seed, n=60):
    """Grammar-valid actions with random values in the instrument limits."""
    rng = np.random.default_rng(int(seed) + 17)
    env, _ = _episode(tier, seed)
    for _ in range(n):
        if env._terminated:
            break
        c = f"c{rng.integers(env._env.n_couplers)}"
        kind = rng.integers(4)
        if kind == 0:
            action = {"qubit": f"q{rng.integers(env._env.n_qubits)}", "node": "resonator_spectroscopy",
                      "sweeps": {"frequencies": [float(rng.uniform(6.2e9, 7.6e9)), 200e6, 101]}, "shots": 256}
        elif kind == 1:
            param = COMMITTABLE[rng.integers(len(COMMITTABLE))]
            value = {"parking_current": rng.uniform(-3e-3, 3e-3), "cz_pulse_frequency": rng.uniform(1e7, 1e9),
                     "cz_pulse_amplitude": rng.uniform(0, 1), "cz_pulse_duration": rng.uniform(8e-9, 2e-6)}.get(
                param, rng.uniform(0, 2 * np.pi))
            action = {"coupler": c, "commit": {"param": param, "value": float(value)}}
        elif kind == 2:
            action = {"coupler": c, "benchmark": True}
        else:
            action = {"configure_lo": {"channel": "readout", "frequency_hz": float(rng.uniform(6.2e9, 7.6e9))}}
        env.step(action)
    return _finish(env)


def invalid_spam(tier, seed, n=40):
    env, _ = _episode(tier, seed)
    for _ in range(n):
        if env._terminated:
            break
        env.step({"coupler": "c0", "node": "no_such_node", "sweeps": {}, "shots": 1})
    return _finish(env)


PROBES = {
    "stop_immediately": stop_immediately,
    "skip_discovery_commit_defaults": skip_discovery_commit_defaults,
    "discover_then_commit_defaults": discover_then_commit_defaults,
    "benchmark_spam": benchmark_spam,
    "random_valid_actions": random_valid_actions,
    "invalid_spam": invalid_spam,
}
