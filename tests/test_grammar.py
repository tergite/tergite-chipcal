import pytest

from chipcal.core.actions import CommitAction, MeasureAction, coerce_action, parse_action, unit_slips, validate
from chipcal.core.config import CommittedConfig
from chipcal.envs.chain_agent_env import _window_action_structure_errors, chain_window_json_schema, expand_action


def test_expand_action_orders_commits_before_the_main_action():
    batch = {"commits": [{"coupler": "c0", "commit": {"param": "parking_current", "value": 1e-4}}],
             "then": {"coupler": "c0", "benchmark": True}}
    assert expand_action(batch) == [batch["commits"][0], batch["then"]]
    assert expand_action({"stop": True}) == [{"stop": True}]


@pytest.mark.parametrize("bad", [{"commits": []}, {"commits": "x"}, {"commits": [{}], "extra": 1}])
def test_malformed_batch_stays_one_action(bad):
    assert expand_action(bad) == [bad]


def test_one_action_kind_per_object():
    assert _window_action_structure_errors({"coupler": "c0", "benchmark": True, "stop": True})
    assert _window_action_structure_errors({"coupler": "c0", "node": "cz_chevron", "sweeps": {}, "shots": True})
    assert _window_action_structure_errors({"coupler": "c0", "commit": {"param": "target_local_phase", "value": 9.0}})
    assert not _window_action_structure_errors({"coupler": "c0", "commit": {"param": "cz_pulse_amplitude", "value": .4}})


def test_schema_variants():
    assert len(chain_window_json_schema()["oneOf"]) == 5
    assert len(chain_window_json_schema(discovery=True)["oneOf"]) == 8


def test_guardrails_flag_unit_slips_and_clamp_range_errors():
    # nanoseconds written as seconds: flagged for rejection, never rescaled or clamped onto an edge
    slip = parse_action({"node": "cz_chevron", "sweeps": {"cz_pulse_frequencies": [2.0e8, 8e6, 21],
                         "cz_pulse_durations": [2.2e2, 4e2, 26], "cz_pulse_amplitude": 0.4}, "shots": 128})
    errors = unit_slips(slip)
    assert len(errors) == 1 and "cz_pulse_durations: wrong units?" in errors[0] and "seconds" in errors[0]
    assert unit_slips(CommitAction("cz_pulse_frequency", 407.0)) and unit_slips(CommitAction("cz_pulse_duration", 220.0))
    # a sweep that only overhangs the range, too many shots, an amplitude above 1: clamped, not rejected
    act = parse_action({"node": "cz_chevron", "sweeps": {"cz_pulse_frequencies": [2.0e8, 8e6, 21],
                        "cz_pulse_durations": [1.9e-6, 4e-7, 26], "cz_pulse_amplitude": 0.4}, "shots": 99999})
    assert not unit_slips(act)
    fixed, notes = coerce_action(act, CommittedConfig(parking_current=0.0), timing_grid_s=4e-9)
    assert isinstance(fixed, MeasureAction) and fixed.shots == 4096 and notes
    assert not validate(fixed, CommittedConfig(parking_current=0.0))
    fixed, notes = coerce_action(CommitAction("cz_pulse_amplitude", 3.0))
    assert fixed.value == 1.0 and notes and not unit_slips(CommitAction("cz_pulse_amplitude", 3.0))


def test_reset_map_unsets_deeper_levels():
    k = CommittedConfig(parking_current=1e-4, cz_pulse_frequency=2e8, cz_pulse_amplitude=.4,
                        cz_pulse_duration=2e-7, control_local_phase=1., target_local_phase=2.)
    k2 = k.commit("cz_pulse_frequency", 2.1e8)
    assert k2.cz_pulse_amplitude == .4 and k2.cz_pulse_duration is None and k2.control_local_phase is None
    assert k.commit("parking_current", 0.0).cz_pulse_frequency is None


def test_guardrails_snap_a_committed_duration_onto_the_timing_grid():
    fixed, notes = coerce_action(CommitAction("cz_pulse_duration", 70.559e-9), timing_grid_s=4e-9)
    assert fixed.value == pytest.approx(72e-9) and notes
    same, notes = coerce_action(CommitAction("cz_pulse_duration", 72e-9), timing_grid_s=4e-9)
    assert same.value == pytest.approx(72e-9) and not notes
