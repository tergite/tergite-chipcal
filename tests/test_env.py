import json
import re

import numpy as np
import pytest

from chipcal import tiers
from chipcal.baselines.discovery import ScriptedDiscovery
from chipcal.envs.chain_agent_env import chain_window_json_schema, default_chain_window_system_prompt


def make(tier="2q1c", seed=3_000_000, **overrides):
    env = tiers.make_env(tier, **overrides)
    return env, env.reset(seed=seed, chain=tiers.sample(tier, seed).chain)


def hidden_numbers(chain):
    """Every hidden scalar of a chip, for leak checks."""
    values = []
    for q in chain.qubits:
        values += [q.f01, q.alpha, q.f01 + q.alpha, q.t1, q.t2, *np.ravel(q.confusion)]
    for c in chain.couplers:
        values += [c.fc0, c.d_asym, c.i_sweet, c.i_period, c.ac_flux_per_amp, c.g_qc_left, c.g_qc_right,
                   c.multimode.bus_t1, c.multimode.bus_t2, c.multimode.bus_alpha]
    values += list(np.ravel(chain.crosstalk))
    return [v for v in values if v not in (0.0, 1.0)]


def numbers_in(text):
    return [float(m) for m in re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?|-?\d+e[-+]?\d+", text)]


def assert_no_hidden_values(text, chain, rel=1e-6):
    shown = np.array(numbers_in(text))
    for value in hidden_numbers(chain):
        if shown.size:
            assert not np.any(np.isclose(shown, value, rtol=rel, atol=0)), f"hidden value {value!r} appears in the observation"


@pytest.mark.parametrize("tier", list(tiers.TIERS))
def test_reset_observation_shows_no_hidden_parameter(tier):
    env, obs = make(tier)
    chain = env._env.chain
    text = obs.to_prompt_text(visual_measurement=True) + json.dumps(obs.to_dict())
    assert_no_hidden_values(text, chain)
    assert "initial_parking_currents" in obs.context and all(v is None for v in obs.context["initial_parking_currents"].values())
    assert "clock_freqs:f01" not in obs.context


def test_observations_stay_free_of_hidden_values_through_discovery():
    env, obs = make()
    chain = env._env.chain
    ScriptedDiscovery(env.step).run(obs)
    for turn in env.transcript.turns:
        text = json.dumps(turn.observation) + json.dumps(turn.tool_result)
        # True discovery lines sit within the measurement error of committed values, so only
        # check the device parameters that no measurement returns exactly.
        assert_no_hidden_values(text, chain)


def test_discovery_gates_coupler_actions_and_invalid_actions_are_free():
    env, obs = make()
    out = env.step({"coupler": "c0", "benchmark": True})
    assert not env.transcript.turns[-1].valid
    assert out.reward == 0.0 and env.turn == 1
    assert "discovery" in str(env.last_raw_result.info["invalid"])
    out = env.step({"coupler": "c0", "node": "cz_chevron"})             # malformed: no sweeps/shots
    assert not env.transcript.turns[-1].valid and out.reward == 0.0
    out = env.step({"bogus": 1})
    assert not env.transcript.turns[-1].valid and out.reward == 0.0


def test_stop_scores_the_minimum_true_fidelity():
    env, obs = make()
    ScriptedDiscovery(env.step).run(obs)
    env.step({"coupler": "c0", "commit": {"param": "parking_current", "value": 5e-4}})
    out = env.step({"stop": True})
    assert out.terminated
    score = env.score()
    assert score["min_true_fidelity"] == 0.0 == out.reward       # an incomplete gate scores 0
    assert score["discovery_fraction"] == 1.0 and score["success"] is False


def test_turn_cap_forces_a_scored_stop():
    env, obs = make(max_turns=3)
    for _ in range(3):
        out = env.step({"configure_lo": {"channel": "readout", "frequency_hz": 6.8e9}})
    assert out.terminated and env.transcript.terminated
    assert env.score()["n_turns"] == 3


def test_prompt_has_no_strategy_playbook():
    prompt = default_chain_window_system_prompt(native_thinking=True, discovery=True).lower()
    for phrase in ("playbook", "invariant", "abs(f01(target)-f12(control))", "earliest", "hypothesis",
                   "measure_template", "commit_candidate"):
        assert phrase not in prompt
    for required in ("minimum over couplers", "final interleaved randomized benchmark", "cz_chevron", "commit_frequency", "[center,full_span,point_count]"):
        assert required in prompt


def test_schema_and_tool_schema_cover_discovery():
    env, _ = make()
    schema = env.tool_schema()
    assert schema["json_schema"] == chain_window_json_schema(discovery=True)
    assert len(schema["json_schema"]["oneOf"]) == 8


def test_measurement_datasets_carry_no_hidden_state():
    env, obs = make()
    ScriptedDiscovery(env.step).run(obs)
    env.step({"coupler": "c0", "node": "readout_calibration", "sweeps": {}, "shots": 256})
    ds = env.last_raw_result.dataset
    assert ds is not None
    forbidden = {"theta", "truth", "parity", "fit_frequency", "candidate_frequencies"}
    assert not forbidden & {k.lower() for k in ds.attrs}
    assert not any(k.startswith("p_") for k in ds.attrs)


def test_an_unidentifiable_benchmark_reports_no_fidelity_estimate():
    env, obs = make()
    ScriptedDiscovery(env.step).run(obs)
    for param, value in (("parking_current", 1.325e-3), ("cz_pulse_frequency", 312e6), ("cz_pulse_amplitude", 0.001),
                         ("cz_pulse_duration", 376e-9)):
        env.step({"coupler": "c0", "commit": {"param": param, "value": value}})
    env.step({"coupler": "c0", "benchmark": True})                      # an idle "gate": RB collapses to the floor
    info = env.last_raw_result.info
    assert info["fit_reliable"] is False and info["F_hat"] is None and info["F_lcb"] == 0.0
    assert "not identifiable" in env.transcript.turns[-1].observation["last_event"]


def test_an_interrupted_episode_resumes_exactly(tmp_path):
    """A job that dies mid-episode resumes from its checkpoint and ends exactly as an uninterrupted run."""
    from chipcal.envs.agent_env import Completion
    from chipcal.run import run

    script = [{"configure_lo": {"channel": "readout", "frequency_hz": 6.8e9}},
              {"qubit": "q0", "node": "resonator_spectroscopy", "sweeps": {"frequencies": [6.8e9, 4.5e8, 201]}, "shots": 256},
              {"configure_lo": {"channel": "readout", "frequency_hz": 7.1e9}},
              {"qubit": "q0", "node": "resonator_spectroscopy", "sweeps": {"frequencies": [7.1e9, 4.5e8, 201]}, "shots": 256},
              {"stop": True}]

    class Scripted:
        def __init__(self, fail_at=None):
            self.calls, self.fail_at = 0, fail_at

        def generate(self, messages, *, images=None, schema=None, temperature=0):
            if self.calls == self.fail_at:
                raise TimeoutError("job time limit")
            action = script[min(self.calls, len(script) - 1)]
            self.calls += 1
            return Completion(text=json.dumps(action), raw={})

    whole = run("2q1c", 3_000_000, tmp_path / "whole", client=Scripted(), save_measurements=False)
    with pytest.raises(TimeoutError):
        run("2q1c", 3_000_000, tmp_path / "split", client=Scripted(fail_at=3), save_measurements=False)
    client = Scripted()
    client.calls = 3
    resumed = run("2q1c", 3_000_000, tmp_path / "split", client=client, save_measurements=False)
    assert [t.action for t in resumed.turns] == [t.action for t in whole.turns]
    assert resumed.score["score"] == whole.score["score"]
    assert json.loads((tmp_path / "split" / "run.json").read_text())["segments"] == 2
    again = run("2q1c", 3_000_000, tmp_path / "split", client=Scripted(fail_at=0), save_measurements=False)
    assert len(again.turns) == len(whole.turns)          # a completed episode is not rerun


def test_a_failing_analysis_costs_one_observation_not_the_episode(monkeypatch):
    import numpy as np
    from chipcal.envs import chain_agent_env
    env, obs = make()
    def broken(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")
    ScriptedDiscovery(env.step).run(obs)
    monkeypatch.setattr(chain_agent_env, "analyze_dataset", broken)
    out = env.step({"coupler": "c0", "node": "readout_calibration", "sweeps": {}, "shots": 256})
    assert out.observation.last_analysis["status"] == "ANALYSIS_FAILED" and not env._terminated


def test_wrong_units_are_rejected_and_clamps_are_reported():
    env, _ = make(discovery=False)
    env.step({"coupler": "c0", "commit": {"param": "parking_current", "value": 1e-5}})
    chevron = {"coupler": "c0", "node": "cz_chevron", "shots": 64,
               "sweeps": {"cz_pulse_frequencies": [4.26e8, 1.4e7, 5], "cz_pulse_amplitude": 0.25}}
    # nanoseconds written 1e15 too large: rejected, with the unit in the message the agent sees
    outcome = env.step({**chevron, "sweeps": {**chevron["sweeps"], "cz_pulse_durations": [5.2e8, 9.6e8, 5]}})
    assert not outcome.observation.session_history[-1]["result"]["valid"]
    assert "wrong units?" in outcome.observation.last_event and "seconds" in outcome.observation.last_event
    # a sweep that only overhangs the range runs clamped, and the agent is told (both observation formats)
    outcome = env.step({**chevron, "sweeps": {**chevron["sweeps"], "cz_pulse_durations": [1.9e-6, 4e-7, 5]}})
    result = outcome.observation.session_history[-1]["result"]
    assert result["valid"] and any("clamped" in note for note in result["coerced"])
    assert "guardrail: cz_pulse_durations: sweep clamped" in outcome.observation.to_turn_text()


def test_a_large_chip_reads_out_on_feedlines_of_at_most_five_qubits():
    env, obs = make("8q-hard", seed=1_000_042)
    state = obs.context["discovery"]
    assert state["readout_feedlines"] == {"readout:fl0": ["q0", "q1", "q2", "q3"], "readout:fl1": ["q4", "q5", "q6", "q7"]}
    # A feedline sweep shows the dips of its own four resonators only.
    discovery = env._env.discovery
    lo = state["lo_frequencies_hz"]["readout:fl1"]
    out = env.step({"qubit": "q5", "node": "resonator_spectroscopy", "sweeps": {"frequencies": [lo, 5e8, 501]}, "shots": 512})
    dips = [p["frequency_hz"] for p in out.observation.last_analysis["candidates"]]
    on_line = [discovery._truth[q]["readout"] for q in state["readout_feedlines"]["readout:fl1"]]
    assert all(any(abs(d - f) < 3e6 for f in on_line) for d in dips)
    ScriptedDiscovery(env.step).run(out.observation)
    assert env._env.discovery.ready and env._env.discovery.accuracy() == 1.0 and env._n_invalid == 0
    # A chip of five qubits or fewer keeps the single shared "readout" LO.
    assert "readout" in make("5q4c")[1].context["discovery"]["lo_frequencies_hz"]
