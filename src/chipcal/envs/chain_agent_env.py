"""Numeric-window observations, action history and inference rollouts."""
from __future__ import annotations
import copy
import json
import math

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from chipcal.sim import oracle
from chipcal.core.config import CommittedConfig
from chipcal.core.actions import LIMITS, NODE_AXES
from chipcal.envs.agent_env import Completion, EpisodeAbort, LLMClient, REASON_INSTRUCTION, RewardConfig, StepOutcome, Transcript, Turn
from chipcal.llm.parsing import _fmt_a, _fmt_float, _fmt_hz, _fmt_s, _json_safe, extract_json_action
from chipcal.sim.chain import ChainPrior
from chipcal.envs.chain_env import ChainCalibrationEnv
from chipcal.core.config import COMMITTABLE, COMMIT_LEVELS
from chipcal.envs.env import StepResult
from chipcal.envs.one_qubit_discovery import is_action, action_schemas, PROMPT as DISCOVERY_PROMPT
from chipcal.envs.observation_analysis import analysis_prompt_block, analyze_dataset, annotate_rb_identifiability
_PARTIAL_CREDIT_ALPHA = 0.3
_MAX_LEDGER_CANDIDATES = 48
_MAX_BENCHMARK_CONFIGS = 24
_MAX_INSTRUMENT_NOTEBOOK_ENTRIES = 32
# Both persistent sections sit ahead of the per-turn state under `stable_prefix`, so any byte
# that moves in them costs a full prompt re-prefill on the server. Trimming them in steps
# rather than one row at a time keeps a prompt's prefix identical to the previous turn's.
_NOTEBOOK_LOW_WATER = 0.6
_HISTORY_OMIT_STEP = 8
_EXPERIMENTAL_NOTEBOOK_FIELDS: dict[str, tuple[str, ...]] = {'coupler_dc_spectroscopy': ('ridge_frequency_span_Hz', 'ridge_contrast', 'top_ridge_rows', 'tls', 'sampling', 'probe_var'), 'cz_parametrization': ('max_p2', 'median_p2', 'contrast', 'top_transfer_points', 'exchange_state', 'no_pulse_reference', 'leakage_onset_amplitude', 'clean_amplitude_max'), 'cz_chevron': ('max_p2', 'median_p2', 'contrast', 'transfer_return_candidates', 'exchange_state', 'no_pulse_reference', 'median_leakage'), 'swap_chevron': ('max_swap_probability', 'median_swap_probability', 'contrast', 'transfer_return_candidates'), 'cz_calibration': ('ranked_working_points',), 'cz_repeated': ('number_of_czs', 'best_points', 'p11_retained', 'no_pulse_reference', 'exchange_state', 'median_leakage'), 'cz_local_phases': ('phase_corrections_rad',), 'cz_rb': ('survival', 'fit_diagnostics'), 'qubit_spectroscopy': ('peaks', 'unresolved_qubits'), 'qubit_t1': ('estimates',), 'qubit_ramsey': ('estimates',), 'readout_calibration': ('qubits',)}
_DISCOVERY_NODES = ("resonator_spectroscopy", "qubit_01_spectroscopy", "qubit_12_spectroscopy")
for _discovery_node in _DISCOVERY_NODES:
    # The fitted centre and the amplitude row it was read at are measured reductions of
    # the same trace, and the protocol asks the policy to commit the fit.
    _EXPERIMENTAL_NOTEBOOK_FIELDS[_discovery_node] = ("qubit", "candidates", "fit", "drive_amplitude", "resolution_hz")
ActionJSON = dict[str, Any]
_CHAIN_NODES = ['coupler_dc_spectroscopy', 'cz_parametrization', 'cz_chevron', 'cz_calibration', 'cz_local_phases', 'qubit_spectroscopy', 'qubit_t1', 'qubit_ramsey', 'readout_calibration']
_ACTION_MODES = frozenset({'grounded', 'hybrid', 'window'})

MAX_BATCH_COMMITS = 4
_BOOKKEEPING_KEYS = ('commit', 'commit_frequency', 'configure_lo')


def _is_bookkeeping_schema(variant: dict[str, Any]) -> bool:
    return bool(set(variant.get('required') or ()) & set(_BOOKKEEPING_KEYS))


def _batch_schema(variants: list[dict[str, Any]]) -> dict[str, Any]:
    """One generation may carry the values already read off the last plot plus the next action.

    A commit records a number the policy decided when it read the previous measurement, so
    spending a whole generation on it buys no new information. Batching removes those calls
    without changing what the environment executes: the driver still steps every entry
    separately, so rewards, the ledger and the notebook see exactly the same sequence.
    """
    book = [v for v in variants if _is_bookkeeping_schema(v)]
    rest = [v for v in variants if not _is_bookkeeping_schema(v)]
    return {'type': 'object', 'required': ['commits'], 'additionalProperties': False,
            'properties': {'commits': {'type': 'array', 'minItems': 1, 'maxItems': MAX_BATCH_COMMITS,
                                       'items': {'oneOf': book}},
                           'then': {'oneOf': rest}}}


def expand_action(action: Any) -> list[Any]:
    """The sequence a batched action stands for; anything else passes through unchanged.

    Callers run this unconditionally, so a policy that never batches behaves as before.
    A malformed batch is returned whole and fails the ordinary structure check, which keeps
    one invalid generation worth exactly one invalid step.
    """
    if not isinstance(action, dict) or 'commits' not in action:
        return [action]
    commits = action.get('commits')
    if not isinstance(commits, list) or not commits or set(action) - {'commits', 'then'}:
        return [action]
    steps = list(commits)
    if action.get('then') is not None:
        steps.append(action['then'])
    return steps


def chain_window_json_schema(discovery=False) -> dict[str, Any]:
    """Standard JSON Schema for guided decoding on the numeric-window surface.

    JSON Schema guarantees one syntactically exclusive action and basic scalar types.  The
    environment remains the authoritative semantic mask for node-specific required axes,
    instrument ranges, grid size, and configuration dependencies.
    """
    numeric_tree = {'description': 'A scalar, an explicit coordinate list, or [center, full_span, point_count]. Every flat three-number list is interpreted as that triple, with inclusive endpoints center-full_span/2 and center+full_span/2; it is never [min,max,n].', 'anyOf': [{'type': 'number'}, {'type': 'array', 'minItems': 1, 'items': {'anyOf': [{'type': 'number'}, {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number'}}]}}]}
    coupler = {'type': 'string', 'pattern': '^c[0-9]+$'}
    schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'title': 'ChainWindowAction', 'oneOf': [{'type': 'object', 'required': ['coupler', 'node', 'sweeps', 'shots'], 'properties': {'coupler': coupler, 'node': {'type': 'string', 'enum': list(_CHAIN_NODES)}, 'sweeps': {'type': 'object', 'additionalProperties': numeric_tree}, 'shots': {'type': 'integer', 'minimum': int(LIMITS['shots'][0]), 'maximum': int(LIMITS['shots'][1])}}, 'additionalProperties': False}, {'type': 'object', 'required': ['coupler', 'commit'], 'properties': {'coupler': coupler, 'commit': {'type': 'object', 'required': ['param', 'value'], 'properties': {'param': {'type': 'string', 'enum': list(COMMITTABLE)}, 'value': {'type': 'number'}}, 'additionalProperties': False}}, 'additionalProperties': False}, {'type': 'object', 'required': ['coupler', 'benchmark'], 'properties': {'coupler': coupler, 'benchmark': {'const': True}}, 'additionalProperties': False}, {'type': 'object', 'required': ['stop'], 'properties': {'stop': {'const': True}}, 'additionalProperties': False}]}
    if discovery:
        schema["oneOf"].extend(action_schemas())
    schema['oneOf'].append(_batch_schema(schema['oneOf']))
    return schema

def chain_window_tool_schema() -> dict[str, Any]:
    """Action grammar for the autonomous numeric-window policy.

    The surface has no measurement templates and no candidate-commit operation: the
    policy must emit the physical experiment window (or physical value to commit) itself.  Deterministic code only
    validates the request against :mod:`chipcal.core.actions` and executes it.

    A sweep triple is ``[center, span, points]`` and therefore makes all three
    consequential experiment-design choices policy outputs.  Explicit value lists remain
    available for irregular grids such as ``working_points``.  Repeating a measurement or
    benchmark is the retry operation; there is no controller-selected retry macro.
    """
    return _json_safe({'contract': 'The model chooses the target coupler, measurement node, every sweep center/span/point count, shots, and every physical commit value. Code only rejects invalid requests and executes valid ones.', 'measure': {'nodes': _CHAIN_NODES, 'axes': {name: NODE_AXES[name] for name in _CHAIN_NODES}, 'limits': LIMITS, 'sweep_coordinate_contract': {'triple': '[center, full_span, integer_point_count]', 'inclusive_endpoints': 'center - full_span/2, center + full_span/2', 'units': 'SI units specified by the node axis (A, Hz, s, rad, normalized)', 'not_min_max': 'the first two values are NOT [minimum, maximum]', 'three_value_rule': 'every flat numeric list of length 3 is a triple, never three explicit samples'}, 'form': {'coupler': '<c0..cN>', 'node': '<measurement node>', 'sweeps': {'<required/optional axis>': '[center, full_span, point_count], an explicit value list, or a node-defined scalar'}, 'shots': '<explicit integer>'}, 'note': 'All required axes and shots must be explicit. To retry, emit another complete measurement action and choose its window again (it may equal the previous one). Both endpoints must lie inside the disclosed instrument limits.'}, 'commit': {'params': list(COMMITTABLE), 'dependency_levels': COMMIT_LEVELS, 'reset_rule': 'Committing one parameter preserves fields at the same/shallower level and sets every deeper-level field to null. The resulting complete config and downstream_resets are returned after every valid commit.', 'form': {'coupler': '<c0..cN>', 'commit': {'param': '<physical parameter>', 'value': '<physical float>'}}, 'note': 'The model emits the physical value; no candidate ID is accepted.'}, 'benchmark': {'form': {'coupler': '<c0..cN>', 'benchmark': True}, 'note': 'To retry an uncertain benchmark, emit this action again.'}, 'stop': {'form': {'stop': True}, 'note': 'Ends the episode; the score is computed from the final committed configuration.'}, 'batch': {'form': {'commits': ['<commit / commit_frequency / configure_lo action>', '...'], 'then': '<measurement, benchmark or stop action>'}, 'note': f'Optional, and the normal way to avoid wasting a turn. One reply may record up to {MAX_BATCH_COMMITS} values you already read off the latest plot and launch the next action in the same JSON; commits run first, in order, then `then`. Never batch a commit whose value depends on the result of `then`.'}, 'json_schema': chain_window_json_schema()})

def _window_action_structure_errors(action: Any) -> list[str]:
    """Strict surface checks layered in front of the existing physics validator.

    ``ChainCalibrationEnv`` remains the single source of truth for numeric bounds and
    node/config dependencies.  These checks prevent permissive parser precedence from
    silently accepting two actions in one object, and guarantee that the policy—not a
    default in Python—selects ``sweeps`` and ``shots``.
    """
    if not isinstance(action, dict):
        return ['window action must be a JSON object']
    kinds = []
    if 'node' in action:
        kinds.append('measure')
    if 'commit' in action:
        kinds.append('commit')
    if 'benchmark' in action:
        kinds.append('benchmark')
    if 'stop' in action:
        kinds.append('stop')
    if len(kinds) != 1:
        return [f'window action must contain exactly one action kind, found {kinds}']
    kind = kinds[0]
    allowed = {'measure': {'coupler', 'node', 'sweeps', 'shots'}, 'commit': {'coupler', 'commit'}, 'benchmark': {'coupler', 'benchmark'}, 'stop': {'stop'}}[kind]
    extra = sorted(set(action) - allowed)
    if extra:
        return [f"unexpected field(s) for {kind} action: {', '.join(extra)}"]
    if kind != 'stop' and (not isinstance(action.get('coupler'), str)):
        return [f'{kind} action requires a coupler label']
    if kind == 'stop' and action.get('stop') is not True:
        return ['stop must be true']
    if kind == 'benchmark' and action.get('benchmark') is not True:
        return ['benchmark must be true']
    if kind == 'measure':
        if not isinstance(action.get('node'), str):
            return ['measurement node must be a string']
        if not isinstance(action.get('sweeps'), dict):
            return ['measurement must explicitly provide a sweeps object']
        shots = action.get('shots')
        if isinstance(shots, bool) or not isinstance(shots, int):
            return ['measurement must explicitly provide integer shots']
        for axis, raw in action['sweeps'].items():
            stack = [raw]
            while stack:
                value = stack.pop()
                if isinstance(value, (list, tuple)):
                    stack.extend(value)
                elif isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(float(value))):
                    return [f'sweep {axis} must contain only finite physical numbers']
            if isinstance(raw, (list, tuple)) and len(raw) == 3 and all((isinstance(value, (int, float)) and (not isinstance(value, bool)) for value in raw)):
                _, span, points = raw
                if float(span) <= 0.0:
                    return [f'sweep {axis} span must be positive']
                if isinstance(points, bool) or int(points) != points or int(points) < 2:
                    return [f'sweep {axis} point_count must be an integer >= 2']
    if kind == 'commit':
        commit = action.get('commit')
        if not isinstance(commit, dict) or set(commit) != {'param', 'value'}:
            return ['commit must contain exactly param and value']
        value = commit.get('value')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(float(value))):
            return ['commit value must be a finite physical number']
        if commit.get('param') in {'control_local_phase', 'target_local_phase'} and (not 0.0 <= float(value) <= 2.0 * math.pi):
            return ['committed local phase must lie in [0, 2*pi] radians']
    return []

def _fmt_commit_value(param: str, value: Any) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    if param == 'parking_current':
        return _fmt_a(float(value))
    if param == 'cz_pulse_frequency':
        return _fmt_hz(float(value))
    if param == 'cz_pulse_duration':
        return _fmt_s(float(value))
    return _fmt_float(float(value))

@dataclass
class ChainObservation:
    """Everything the policy may see for the whole chip -- provably theta-free."""
    turn: int
    max_turns: int
    remaining_budget_seconds: float
    total_time: float
    objective_f_star: float
    context: dict[str, Any]
    couplers: list[dict[str, Any]]
    min_f_lcb: float
    all_pass: bool
    best_min_f_lcb: Optional[float]
    qoi_lines: list[str]
    last_event: Optional[str]
    last_analysis: Optional[dict[str, Any]] = None
    last_analysis_coupler: Optional[str] = None
    last_analysis_node: Optional[str] = None
    trace_tail: list[dict[str, Any]] = field(default_factory=list)
    calibration_ledger: list[dict[str, Any]] = field(default_factory=list)
    instrument_notebook: list[dict[str, Any]] = field(default_factory=list)
    session_history: list[dict[str, Any]] = field(default_factory=list)
    history_truncated_turns: int = 0
    history_contract: Optional[str] = None
    measurement_dataset: Any = field(default=None, repr=False, compare=False)
    # Image files a backend drew for the latest measurement; the simulator leaves it empty and
    # the policy renders the dataset instead.
    measurement_images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload.pop('measurement_dataset', None)
        return _json_safe(payload)

    def to_prompt_text(self, ctx_raw: Optional[dict]=None, *, visual_measurement: bool=False, stable_prefix: bool=False) -> str:
        """Observation text. ``stable_prefix`` emits the same lines with the append-only notebook,
        ledger and history first and the per-turn state last, so a server prefix cache reuses
        most of the previous turn's prompt instead of diverging at the first line."""
        text = self._prompt_lines(visual_measurement)
        if not stable_prefix:
            return '\n'.join(text[0] + text[1])
        return '\n'.join(text[1] + ['CURRENT STATE (this turn):'] + text[0])

    def to_turn_text(self, *, visual_measurement: bool = False) -> str:
        """Compact per-turn update for chat-style multi-turn loops, where the conversation already holds the
        history: the chip state and the latest result, without the persistent notebook, ledger, history,
        limits and protocol (those are in the first, full observation)."""
        state, _ = self._prompt_lines(visual_measurement)
        keep = []
        for line in state:
            if line.startswith('MEASUREMENT LIMITS') or line == self.context.get('spectroscopy_protocol'):
                continue
            if line.startswith('DISCOVERY STATE: '):
                d = self.context.get('discovery') or {}
                line = 'DISCOVERY STATE: ' + json.dumps({k: d.get(k) for k in ('committed_frequencies_hz', 'lo_frequencies_hz', 'reachable_rf_hz', 'ready')}, sort_keys=True)
            keep.append(line)
        return '\n'.join(keep)

    def _prompt_lines(self, visual_measurement: bool) -> tuple[list[str], list[str]]:
        """(per-turn state lines, persistent append-only lines), in the legacy order."""
        metric = self.context.get('fidelity_label') or 'F_lcb'
        lines = [f'step {self.turn} of {self.max_turns}; remaining_budget_seconds={self.remaining_budget_seconds:.1f}; total_time={self.total_time:.1f}s', f'OBJECTIVE: at the end the environment benchmarks every coupler once more; the score is the MIN over couplers of those final RB fidelities; calibrated means every one >= f_star={self.objective_f_star:.4f}. Benchmarked {metric} values below are your own measurements.', self.context.get('coupler_orientation') or 'COUPLER ORIENTATION: every pair below is control -> target; control is the higher-f01 qubit.', 'CHIP STATUS:']
        lines += ['  ' + row['line'] for row in self.couplers]
        discovery = self.context.get('discovery')
        if discovery:
            lines.append('DISCOVERY STATE: ' + json.dumps(discovery, sort_keys=True))
        if self.context.get('measurement_profile'):
            # While discovery is incomplete the observation carries two limit sets. Say which
            # one governs which node family, or the narrower coupler caps read as universal.
            scope = ' (coupler nodes only; discovery scans are governed by DISCOVERY STATE)' if discovery and (not discovery.get('ready')) else ''
            lines.append(f'MEASUREMENT LIMITS{scope}: ' + json.dumps(self.context['measurement_profile'], sort_keys=True))
        if self.context.get('spectroscopy_protocol'):
            lines.append(self.context['spectroscopy_protocol'])
        lines.append('COMMITTED CONFIGURATION (SI units; null means NOT committed; all six fields are shown after every action):')
        for row in self.couplers:
            config = row.get('config') or {param: None for param in COMMITTABLE}
            missing = row.get('missing') or [param for param in COMMITTABLE if config.get(param) is None]
            lines.append('  ' + str(row['label']) + ': ' + json.dumps(config, sort_keys=True, separators=(',', ':')) + f" core_complete={str(bool(row.get('core_complete', False))).lower()}" + ' missing=' + json.dumps(missing, separators=(',', ':')))
        tail = '' if self.all_pass else '   [unbenchmarked/stale couplers shown as 0]'
        lines.append(f'MIN benchmarked {metric} = {self.min_f_lcb:.4f}{tail}')
        if self.qoi_lines:
            lines.append('1Q QOIs:')
            lines += ['  ' + q for q in self.qoi_lines]
        if self.last_event:
            event = self.last_event
            if visual_measurement:
                event = event.replace('see structured analysis below', 'see attached plot')
            lines.append('last event: ' + event)
        if visual_measurement and self.measurement_dataset is not None:
            tag = self.last_analysis_node or 'measurement'
            if self.last_analysis_coupler:
                tag = f'{self.last_analysis_coupler} - {tag}'
            lines.append(f'LATEST MEASUREMENT PLOT ATTACHED ({tag}). Read the measured pixels, axes, color scale, uncertainty/noise, peaks, ridges, fringes, and edge effects yourself. The instrument notebook below supplies only finite-shot empirical summaries and ordinary curve fits; it never exposes hidden simulator state.')
        if self.last_analysis and (not visual_measurement):
            tag = self.last_analysis_node or 'measurement'
            if self.last_analysis_coupler:
                tag = f'{self.last_analysis_coupler} - {tag}'
            lines.append(f'structured observation analysis ({tag}):\n' + analysis_prompt_block(self.last_analysis))
        state, lines = lines, []
        if self.instrument_notebook:
            lines.append('PERSISTENT INSTRUMENT NOTEBOOK (chronological; measured counts/declared coordinates and empirical curve fits only; no hidden truth or simulator posterior):')
            for entry in self.instrument_notebook:
                lines.append('  ' + json.dumps(entry, sort_keys=True, separators=(',', ':')))
        if self.calibration_ledger:
            lines.append('PERSISTENT CALIBRATION LEDGER (policy-visible evidence only):')
            for row in self.calibration_ledger:
                rendered = copy.deepcopy(row)
                if visual_measurement:
                    rendered.pop('candidates', None)
                    if isinstance(rendered.get('last_measurement'), dict):
                        rendered['last_measurement'].pop('status', None)
                lines.append('  ' + json.dumps(rendered, sort_keys=True, separators=(',', ':')))
        if self.session_history:
            lines.append(f"SESSION HISTORY [{self.history_contract or 'unspecified'}] (policy actions and observed results; oldest to newest):")
            for row in self.session_history:
                rendered = copy.deepcopy(row)
                if visual_measurement and isinstance(rendered.get('result'), dict):
                    rendered['result'].pop('analysis', None)
                lines.append('  ' + json.dumps(rendered, sort_keys=True, separators=(',', ':')))
        if self.history_truncated_turns:
            lines.append(f'history compaction summarized {self.history_truncated_turns} middle turn(s); the persistent ledger above remains complete')
        return state, lines

class ChainAgentEnv:
    """Multi-turn agentic-RL wrapper. One episode = discovery plus the CZ tune-up of a whole chip.

        env = ChainAgentEnv(n_qubits=3, discovery=True, max_turns=200)
        obs = env.reset(seed=3)
        out = env.step({"coupler": "c0", "node": "cz_chevron", "sweeps": {...}, "shots": 128})
        ...
        out = env.step({"stop": True})   # terminal reward = min_i true CZ fidelity
        env.transcript.to_dict()
    """

    def __init__(self, *, backend: str='multimode', max_turns: int=200, budget_seconds: float=28800.0, lambda_per_minute: float=0.0025, f_star: float=0.98, reward_mode: str='final_rb', invalid_penalty: float=0.0, shaping: bool=False, partial_credit_alpha: float=0.0, pc_linear_floor: bool=False, trace_tail: int=8, noise=None, n_qubits: int=5, crosstalk_scale: float=0.12, family: str='nominal', scout_bias: bool=True, scout_qoi: bool=False, prior: Optional[ChainPrior]=None, duration_limits: Optional[tuple[float, float]]=None, action_mode: str='window', history_char_budget: int=64000, notebook_char_budget: Optional[int]=None, guardrails: bool=False, spectroscopy=None, measurement_profile=None, discovery=False):
        if action_mode != 'window':
            raise ValueError('Only window actions are supported')
        if action_mode not in _ACTION_MODES:
            raise ValueError(f'action_mode must be one of {sorted(_ACTION_MODES)}, got {action_mode!r}')
        self.action_mode = str(action_mode)
        self.max_turns = int(max_turns)
        self.trace_tail = max(int(trace_tail), self.max_turns)
        if int(history_char_budget) < 2000:
            raise ValueError('history_char_budget must be at least 2000 characters')
        self.history_char_budget = int(history_char_budget)
        # None keeps the legacy entry-count cap only; RL bounds the notebook's characters too.
        self.notebook_char_budget = None if notebook_char_budget is None else int(notebook_char_budget)
        self.reward_cfg = RewardConfig(lambda_per_minute, invalid_penalty, shaping)
        self.partial_credit_alpha = float(partial_credit_alpha)
        self.pc_linear_floor = bool(pc_linear_floor)
        self.f_star = float(f_star)
        self.scout_bias = bool(scout_bias)
        self.scout_qoi = bool(scout_qoi)
        prior = prior or ChainPrior(n_qubits=n_qubits, crosstalk_scale=crosstalk_scale)
        self._env = ChainCalibrationEnv(prior=prior, backend=backend, budget_seconds=budget_seconds, lambda_per_minute=lambda_per_minute, f_star=f_star, reward_mode=reward_mode, family=family, n_qubits=n_qubits, debug=False, noise=noise, scout_bias=self.scout_bias, scout_qoi=self.scout_qoi, duration_limits=duration_limits, guardrails=guardrails, spectroscopy=spectroscopy, measurement_profile=measurement_profile, discovery=discovery)
        self.guardrails = bool(guardrails)
        self.transcript = Transcript()
        self._ctx_raw: dict[str, Any] = {}
        self._reset_state()

    def tool_schema(self) -> dict[str, Any]:
        schema = chain_window_tool_schema()
        settings = self._env.discovery_settings
        if settings:
            schema["discovery"] = DISCOVERY_PROMPT
            schema["json_schema"] = chain_window_json_schema(discovery=True)
            # The measure block below describes coupler nodes. Discovery scans are a separate
            # node family with its own span/IF/shot limits, so scope the coupler ones here
            # rather than leaving two contradictory limit sets side by side.
            schema["measure"]["limits_scope"] = (
                "The nodes, axes and limits in this measure block govern coupler measurement actions only. "
                f"The discovery nodes {sorted(_DISCOVERY_NODES)} are governed by DISCOVERY STATE: span up to "
                f"{settings.max_span_hz:.6g} Hz, RF inside the addressed channel's reachable_rf_hz (signed IF "
                f"+/-{settings.max_if_hz:.6g} Hz, boundary included), and shots in [64,8192].")
        return schema

    def _reset_state(self) -> None:
        self.turn = 0
        self.total_time = 0.0
        self._terminated = False
        self.last_raw_result: Optional[StepResult] = None
        self._last_event: Optional[str] = None
        self._last_analysis: Optional[dict[str, Any]] = None
        self._last_analysis_coupler: Optional[str] = None
        self._last_analysis_node: Optional[str] = None
        self._last_measurement_dataset = None
        self._last_measurement_images: list[str] = []
        self._best_min_f_lcb: Optional[float] = None
        self._trace: list[dict[str, Any]] = []
        self._session_history: list[dict[str, Any]] = []
        self._instrument_notebook: list[dict[str, Any]] = []
        self._phi_prev = 0.0
        self._n_invalid = 0
        self._terminal_comps: Optional[dict[str, Any]] = None
        self._reference_currents: Optional[list[Optional[float]]] = None
        self._candidate_registry: dict[str, dict[str, Any]] = {}
        self._ledger: dict[str, dict[str, Any]] = {str(c['label']): {'label': str(c['label']), 'measurements': 0, 'commit_actions': 0, 'parameters_written': 0, 'benchmarks': 0, 'attempts': 0, 'unobserved_rewrites': 0, 'redundant_commits': 0, 'unsupported_writes': 0, 'params_since_evidence': set(), 'supported_params': set(), 'last_measurement': None, 'last_benchmark': None, 'best_benchmark_lcb': None, 'benchmark_config_history': [], 'candidates': []} for c in self._ctx_raw.get('couplers', [])}

    def reset(self, seed: Optional[int]=None, theta=None, chain=None) -> ChainObservation:
        chain = chain if chain is not None else theta
        ctx, info = self._env.reset(seed=seed, chain=chain)
        self._ctx_raw = ctx
        self.family = info.get('family')
        self.transcript = Transcript()
        self._reset_state()
        self._reference_currents = list(self._env.reference_currents)
        self._last_event = ctx.get('first_event') or ('chip is undiscovered and un-tuned' if ctx.get('discovery') else 'chip is un-tuned')
        if self.reward_cfg.shaping:
            self._phi_prev = oracle.rho(self._env.privileged_min_fidelity())
        return self._observe()

    @staticmethod
    def _experimental_notebook_entry(*, turn: int, coupler: Optional[str], node: Optional[str], analysis: dict[str, Any], action: ActionJSON) -> dict[str, Any]:
        """Make a lab-note from observed counts without exposing a model inversion.

        ``analyze_dataset`` contains both conventional experimental reductions and, for a
        chevron, an optional GridBelief posterior.  The latter is useful to scripted agents
        but is deliberately not copied here.  The explicit allow-list above makes that
        information boundary reviewable instead of relying on a fragile deny-list.
        """
        node_name = str(node or analysis.get('node') or 'measurement')
        entry: dict[str, Any] = {'turn': int(turn), 'coupler': coupler, 'node': node_name, 'provenance': 'finite_shot_counts_and_declared_sweep_coordinates_only', 'method': 'empirical_summary_or_standard_curve_fit_not_simulator_inversion', 'shots': analysis.get('shots', action.get('shots')), 'axes': copy.deepcopy(analysis.get('axes') or {}), 'status': analysis.get('status'), 'kind': analysis.get('kind')}
        if node_name in ('resonator_spectroscopy', 'qubit_01_spectroscopy', 'qubit_12_spectroscopy'):
            entry['provenance'] = 'measured_noisy_response_and_declared_sweep_coordinates_only'
        if isinstance(action, dict) and 'sweeps' in action:
            entry['requested_sweeps'] = copy.deepcopy(action.get('sweeps') or {})
        for field_name in _EXPERIMENTAL_NOTEBOOK_FIELDS.get(node_name, ()):
            if field_name in analysis:
                entry[field_name] = copy.deepcopy(analysis[field_name])
        return _json_safe(entry)

    def _record_instrument_notebook(self, coupler: Optional[str], node: Optional[str], analysis: dict[str, Any], action: ActionJSON) -> None:
        self._instrument_notebook.append(self._experimental_notebook_entry(turn=self.turn, coupler=coupler, node=node, analysis=analysis, action=action))
        del self._instrument_notebook[:-_MAX_INSTRUMENT_NOTEBOOK_ENTRIES]
        if self.notebook_char_budget is not None:
            # Oldest entries go first; committed config and the ledger keep the durable facts.
            # Trim to a low-water mark rather than to the budget, so the notebook is rewritten
            # every few turns instead of every turn (see _NOTEBOOK_LOW_WATER).
            packed = lambda: len(json.dumps(self._instrument_notebook, sort_keys=True, separators=(',', ':')))
            if packed() > self.notebook_char_budget:
                floor = int(self.notebook_char_budget * _NOTEBOOK_LOW_WATER)
                while len(self._instrument_notebook) > 1 and packed() > floor:
                    del self._instrument_notebook[0]

    @staticmethod
    def _candidate_values(node: str, analysis: dict[str, Any], action: ActionJSON) -> list[dict[str, float]]:
        """Translate measured, policy-visible analysis rows into committable candidates.

        This layer deliberately performs no hidden-parameter inference.  Every returned
        number is copied from ``analysis`` (or a scalar sweep value the policy itself sent),
        which makes candidate-ID actions auditable and removes floating-point transcription
        from the LLM's job without giving it privileged information.
        """
        rows: list[dict[str, float]] = []
        if node == 'coupler_dc_spectroscopy':
            raw_biases = list((analysis.get('tls') or {}).get('clean_biases_A') or [])
            raw_biases += [r.get('current_A') for r in analysis.get('top_ridge_rows', [])]
            seen: set[float] = set()
            for value in raw_biases:
                if isinstance(value, (int, float)) and float(value) not in seen:
                    seen.add(float(value))
                    rows.append({'parking_current': float(value)})
        elif node == 'cz_parametrization':
            duration = (action.get('sweeps') or {}).get('cz_pulse_duration')
            for row in analysis.get('top_transfer_points', []):
                values = {'cz_pulse_frequency': row.get('frequency_Hz'), 'cz_pulse_amplitude': row.get('amplitude')}
                if isinstance(duration, (int, float)):
                    values['cz_pulse_duration'] = float(duration)
                if all((isinstance(v, (int, float)) for v in values.values())):
                    rows.append({k: float(v) for k, v in values.items()})
        elif node in ('cz_chevron', 'swap_chevron'):
            amplitude = (action.get('sweeps') or {}).get('cz_pulse_amplitude')
            belief_candidate = analysis.get('belief_map_candidate') or {}
            belief_values = {'cz_pulse_frequency': belief_candidate.get('cz_pulse_frequency'), 'cz_pulse_amplitude': amplitude, 'cz_pulse_duration': belief_candidate.get('cz_pulse_duration')}
            if all((isinstance(v, (int, float)) for v in belief_values.values())):
                rows.append({k: float(v) for k, v in belief_values.items()})
            axes = analysis.get('axes') or {}
            freq_axis = axes.get('cz_pulse_frequencies') or {}
            freq_min = freq_axis.get('min')
            if isinstance(freq_min, (int, float)) and isinstance(amplitude, (int, float)):
                rows.append({'cz_pulse_frequency': float(freq_min) - 4000000.0, 'cz_pulse_amplitude': float(amplitude), 'cz_pulse_duration': 5e-08})
            for row in analysis.get('transfer_return_candidates', []):
                values = {'cz_pulse_frequency': row.get('frequency_Hz'), 'cz_pulse_duration': row.get('return_duration_s')}
                if all((isinstance(v, (int, float)) for v in values.values())):
                    rows.append({k: float(v) for k, v in values.items()})
        elif node == 'cz_calibration':
            for row in analysis.get('ranked_working_points', []):
                values = {'cz_pulse_frequency': row.get('frequency_Hz'), 'cz_pulse_duration': row.get('duration_s')}
                if all((isinstance(v, (int, float)) for v in values.values())):
                    rows.append({k: float(v) for k, v in values.items()})
        elif node == 'cz_local_phases':
            phase = analysis.get('phase_corrections_rad') or {}
            if isinstance(phase.get('control'), (int, float)) and isinstance(phase.get('target'), (int, float)):
                rows.append({'control_local_phase': float(phase['control']), 'target_local_phase': float(phase['target'])})
        return rows[:10] if node in ('cz_chevron', 'swap_chevron') else rows[:4]

    def _register_candidates(self, coupler: Optional[str], node: Optional[str], analysis: dict[str, Any], action: ActionJSON) -> None:
        if coupler not in self._ledger or not node:
            return
        self._register_candidate_values(str(coupler), str(node), self._candidate_values(node, analysis, action))

    def _register_candidate_values(self, coupler: str, source: str, values_list: list[dict[str, float]]) -> None:
        """Replace one evidence source's executable candidates while retaining others.

        Measured/posterior estimates can fall outside the action grammar (for
        example a chevron return duration above the global 2 us limit).  Such a row
        is evidence, but it is not a legal commit target and must never be advertised
        as a policy option.
        """
        if coupler not in self._ledger:
            return
        bounded_params = {'parking_current': 'dc_current', 'cz_pulse_frequency': 'drive_frequency', 'cz_pulse_amplitude': 'amplitude', 'cz_pulse_duration': 'duration'}
        executable_values = []
        for values in values_list:
            if any((param not in COMMITTABLE or not isinstance(value, (int, float)) or (not math.isfinite(float(value))) for param, value in values.items())):
                continue
            outside = False
            for param, limit_name in bounded_params.items():
                if param not in values:
                    continue
                lo, hi = LIMITS[limit_name]
                if not lo <= float(values[param]) <= hi:
                    outside = True
                    break
            if not outside:
                executable_values.append({param: float(value) for param, value in values.items()})
        ledger = self._ledger[coupler]
        preserved = []
        for old in ledger['candidates']:
            if old.get('source_node') == source:
                self._candidate_registry.pop(old['candidate_id'], None)
            else:
                preserved.append(old)
        candidates = []
        for rank, values in enumerate(executable_values):
            candidate_id = f'{coupler}:t{self.turn}:{source}:{rank}'
            row = {'candidate_id': candidate_id, 'source_node': source, 'rank': rank, 'commit': values}
            candidates.append(row)
            self._candidate_registry[candidate_id] = {'coupler': coupler, 'commit': dict(values), 'source_node': source, 'turn': self.turn, 'rank': rank}
        durable_sources = {'benchmark_config_history', 'best_benchmark_config', 'restore_previous_config', 'benchmark_bias_search'}
        durable = [row for row in preserved if row.get('source_node') in durable_sources]
        ordinary = [row for row in preserved if row.get('source_node') not in durable_sources]
        combined = candidates + durable + ordinary
        ledger['candidates'] = combined[:_MAX_LEDGER_CANDIDATES]
        for dropped in combined[_MAX_LEDGER_CANDIDATES:]:
            self._candidate_registry.pop(dropped['candidate_id'], None)
        ledger['supported_params'] = {param for row in ledger['candidates'] for param in row['commit']}

    def _invalidate_candidates(self, labels: list[str], *, clear_support: bool=True, preserve_sources: Optional[set[str]]=None) -> None:
        preserve_sources = preserve_sources or set()
        for label in labels:
            ledger = self._ledger.get(label)
            if ledger is None:
                continue
            kept = []
            for old in ledger['candidates']:
                if old.get('source_node') in preserve_sources:
                    kept.append(old)
                else:
                    self._candidate_registry.pop(old['candidate_id'], None)
            ledger['candidates'] = kept
            if clear_support:
                ledger['supported_params'] = {param for row in kept for param in row['commit']}

    def _consume_candidate_params(self, label: str, candidate_id: str, params: set[str]) -> None:
        """Consume only the selected fields, retaining the rest of a candidate transaction."""
        ledger = self._ledger.get(label)
        spec = self._candidate_registry.get(candidate_id)
        if ledger is None or spec is None:
            return
        for param in params:
            spec['commit'].pop(param, None)
        kept = []
        for row in ledger['candidates']:
            if row.get('candidate_id') == candidate_id:
                for param in params:
                    row['commit'].pop(param, None)
                if not row['commit']:
                    self._candidate_registry.pop(candidate_id, None)
                    continue
            kept.append(row)
        ledger['candidates'] = kept
        ledger['supported_params'] = {param for row in kept for param in row['commit']}

    def _register_bias_search_candidates(self, coupler: str) -> None:
        """Expose bounded local-search moves around the current committed bias.

        Called after both benchmarks and bias commits.  Recentering is deterministic
        arithmetic over policy-visible state; choosing whether/where to move remains the
        policy's decision.  This lets a recovery controller take multiple bounded steps
        without privately adding SI floats between measurements.
        """
        idx = self._label_index(coupler)
        if idx is None:
            return
        current = self._env.configs[idx].parking_current
        if not isinstance(current, (int, float)):
            return
        lo, hi = LIMITS['dc_current']
        values = []
        seen = {float(current)}
        for offset in (0.0001, -0.0001, 0.0003, -0.0003, 0.0006, -0.0006, 0.0009, -0.0009):
            value = float(max(lo, min(hi, float(current) + offset)))
            if value not in seen:
                seen.add(value)
                values.append({'parking_current': value})
        self._register_candidate_values(coupler, 'benchmark_bias_search', values)

    def _register_restore_candidate(self, coupler: str, previous: dict[str, Any]) -> None:
        """Retain the previously visible complete gate config across a deliberate re-bias."""
        values = {param: float(previous[param]) for param in COMMITTABLE if param != 'parking_current' and isinstance(previous.get(param), (int, float))}
        if values:
            self._register_candidate_values(coupler, 'restore_previous_config', [values])

    def _register_best_config_candidate(self, coupler: str, f_lcb: Any) -> None:
        """Persist the best previously benchmarked config as policy-visible recovery state."""
        if not isinstance(f_lcb, (int, float)):
            return
        ledger = self._ledger.get(coupler)
        idx = self._label_index(coupler)
        if ledger is None or idx is None:
            return
        best = ledger.get('best_benchmark_lcb')
        if isinstance(best, (int, float)) and float(f_lcb) <= float(best):
            return
        config = self._env.configs[idx].as_dict()
        values = {param: float(config[param]) for param in COMMITTABLE if isinstance(config.get(param), (int, float))}
        if values:
            ledger['best_benchmark_lcb'] = float(f_lcb)
            self._register_candidate_values(coupler, 'best_benchmark_config', [values])

    def _register_benchmark_config_history(self, coupler: str, f_lcb: Any) -> None:
        """Expose a bounded ranked memory of configurations actually benchmarked.

        A policy can restore the best point found during a bias search from this serialized
        history instead of private state; both the configuration and its outcome were observed.
        """
        if not isinstance(f_lcb, (int, float)):
            return
        ledger = self._ledger.get(coupler)
        idx = self._label_index(coupler)
        if ledger is None or idx is None:
            return
        config = self._env.configs[idx].as_dict()
        values = {param: float(config[param]) for param in COMMITTABLE if isinstance(config.get(param), (int, float))}
        if not values:
            return
        history = ledger.setdefault('benchmark_config_history', [])
        for row in history:
            if row['commit'] == values:
                row['F_lcb'] = float(f_lcb)          # the latest result, not the best of repeated draws
                break
        else:
            history.append({'F_lcb': float(f_lcb), 'commit': values})
        history.sort(key=lambda row: float(row['F_lcb']), reverse=True)
        del history[_MAX_BENCHMARK_CONFIGS:]
        self._register_candidate_values(coupler, 'benchmark_config_history', [dict(row['commit']) for row in history])

    def _candidate_id_from_reference(self, coupler: Any, reference: Any) -> Optional[str]:
        """Resolve a hybrid candidate ID or a grounded policy-visible option index."""
        candidate_id = str(reference)
        return candidate_id if candidate_id in self._candidate_registry else None

    def _requested_commits(self, action: ActionJSON) -> dict[str, float]:
        if not isinstance(action, dict):
            return {}
        commit = action.get('commit')
        if isinstance(commit, dict) and commit.get('param') in COMMITTABLE and isinstance(commit.get('value'), (int, float)):
            return {str(commit['param']): float(commit['value'])}
        return {}

    def _update_ledger(self, action: ActionJSON, res: StepResult, invalid: bool, requested: dict[str, float], previous: dict[str, Any]) -> None:
        coupler = res.info.get('coupler') or (action.get('coupler') if isinstance(action, dict) else None)
        ledger = self._ledger.get(str(coupler))
        if ledger is None or invalid:
            return
        node = res.info.get('node')
        if node:
            ledger['measurements'] += 1
            ledger['params_since_evidence'].clear()
            ledger['last_measurement'] = {'turn': self.turn, 'node': node, 'status': (self._last_analysis or {}).get('status')}
        if requested:
            ledger['commit_actions'] += 1
            actual = res.info.get('committed')
            writes = actual if isinstance(actual, dict) else requested
            ledger['parameters_written'] += len(writes)
            for param, value in writes.items():
                if param not in ledger['supported_params']:
                    ledger['unsupported_writes'] += 1
                else:
                    ledger['supported_params'].remove(param)
                if param in ledger['params_since_evidence']:
                    ledger['unobserved_rewrites'] += 1
                old = previous.get(param)
                if isinstance(old, (int, float)):
                    tol = max(1e-12, 1e-09 * max(abs(float(old)), abs(float(value))))
                    if abs(float(old) - float(value)) <= tol:
                        ledger['redundant_commits'] += 1
                ledger['params_since_evidence'].add(param)
            invalidated = list(res.info.get('benchmarks_invalidated') or [])
            self._invalidate_candidates(sorted({label for label in invalidated if label != coupler}))
            is_candidate = isinstance(action, dict) and 'commit_candidate' in action
            if is_candidate:
                candidate_id = self._candidate_id_from_reference(coupler, action.get('commit_candidate'))
                if candidate_id is not None:
                    self._consume_candidate_params(str(coupler), candidate_id, set(requested))
            if 'parking_current' in requested:
                self._invalidate_candidates([str(coupler)], clear_support=True, preserve_sources={'coupler_dc_spectroscopy', 'benchmark_bias_search', 'best_benchmark_config', 'benchmark_config_history'})
                self._register_restore_candidate(str(coupler), previous)
            elif not is_candidate:
                self._invalidate_candidates([str(coupler)], clear_support=False)
        if 'F_lcb' in res.info:
            ledger['benchmarks'] += 1
            ledger['attempts'] += 1
            ledger['params_since_evidence'].clear()
            ledger['supported_params'].clear()
            ledger['last_benchmark'] = {'turn': self.turn, 'F_lcb': res.info.get('F_lcb'), 'passed': bool(res.info.get('passed'))}

    def _ledger_view(self) -> list[dict[str, Any]]:
        rows = []
        for label in self._env.coupler_labels():
            state = self._ledger.get(label)
            if state is None:
                continue
            candidates = copy.deepcopy(state['candidates'])
            rows.append({'label': label, 'measurements': state['measurements'], 'commit_actions': state['commit_actions'], 'parameters_written': state['parameters_written'], 'attempts': state['attempts'], 'unobserved_rewrites': state['unobserved_rewrites'], 'redundant_commits': state['redundant_commits'], 'unsupported_writes': state['unsupported_writes'], 'last_measurement': state['last_measurement'], 'last_benchmark': state['last_benchmark'], 'candidates': candidates})
        return _json_safe(rows)

    @staticmethod
    def _brief_history_row(row: dict[str, Any]) -> dict[str, Any]:
        """Deterministic evidence summary used only after the context cap is reached."""
        out = copy.deepcopy(row)
        result = out.get('result')
        if isinstance(result, dict) and isinstance(result.get('analysis'), dict):
            analysis = result['analysis']
            result['analysis'] = {'compacted': True, 'status': analysis.get('status'), 'keys': sorted((str(key) for key in analysis))}
        return out

    @staticmethod
    def _history_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        nodes: dict[str, int] = {}
        invalid = 0
        cost = 0.0
        for row in rows:
            action = row.get('action') if isinstance(row, dict) else {}
            action = action if isinstance(action, dict) else {}
            if 'node' in action:
                kind = 'measure'
                node = str(action.get('node'))
                nodes[node] = nodes.get(node, 0) + 1
            elif 'commit' in action:
                kind = 'commit'
            elif action.get('benchmark'):
                kind = 'benchmark'
            elif action.get('stop'):
                kind = 'stop'
            else:
                kind = 'invalid_or_other'
            kinds[kind] = kinds.get(kind, 0) + 1
            result = row.get('result') if isinstance(row, dict) else {}
            if isinstance(result, dict):
                invalid += int(not bool(result.get('valid', True)))
                cost += float(result.get('cost_seconds', 0.0) or 0.0)
        return {'history_compaction': True, 'omitted_turns': len(rows), 'turn_range': [rows[0].get('turn'), rows[-1].get('turn')] if rows else [], 'action_counts': kinds, 'measurement_counts': nodes, 'invalid_actions': invalid, 'cost_seconds': round(cost, 6), 'note': 'middle turns summarized at the configured context budget'}

    def _history_view(self) -> tuple[list[dict[str, Any]], int]:
        """Return full history until the explicit character budget, then first+latest+summary."""
        rows = _json_safe(self._session_history)
        if self.action_mode != 'window' or not rows:
            return ([], 0)
        packed = json.dumps(rows, sort_keys=True, separators=(',', ':'))
        if len(packed) <= self.history_char_budget:
            return (rows, 0)
        first = [self._brief_history_row(row) for row in rows[:2]]
        tail: list[dict[str, Any]] = []
        for row in reversed(rows[2:]):
            candidate_tail = [self._brief_history_row(row)] + tail
            omitted_n = max(0, len(rows) - len(first) - len(candidate_tail))
            marker = self._history_summary(rows[len(first):len(first) + omitted_n])
            candidate = first + ([marker] if omitted_n else []) + candidate_tail
            if len(json.dumps(candidate, sort_keys=True, separators=(',', ':'))) > self.history_char_budget:
                break
            tail = candidate_tail
        omitted_n = max(0, len(rows) - len(first) - len(tail))
        # Filling the budget exactly would fold one more row into the marker every turn, so the
        # marker and the tail start would move and no two prompts would share a prefix.
        # Quantising the fold keeps those bytes identical between rewrites; in between, the
        # view only grows at its end, which a prefix cache serves for free.
        if omitted_n:
            omitted_n = min(len(rows) - len(first), -(-omitted_n//_HISTORY_OMIT_STEP)*_HISTORY_OMIT_STEP)
            tail = [self._brief_history_row(row) for row in rows[len(first) + omitted_n:]]
        omitted = rows[len(first):len(first) + omitted_n]
        view = first + ([self._history_summary(omitted)] if omitted else []) + tail
        return (_json_safe(view), omitted_n)

    def _record_window_history(self, action: Any, res: StepResult, *, analysis: Optional[dict[str, Any]], event: Optional[str], forced_stop: bool) -> None:
        info = res.info if isinstance(res.info, dict) else {}
        result: dict[str, Any] = {'valid': not bool(info.get('invalid')), 'cost_seconds': round(float(res.cost_seconds), 6), 'remaining_budget_seconds': round(float(self._env.tau), 6)}
        for key in ('configured_lo', 'frequency_committed', 'node', 'coupler', 'committed', 'benchmarks_invalidated', 'F_hat', 'F_sigma', 'F_lcb', 'fit_reliable', 'reference_decay', 'passed', 'benchmark_identifiable', 'benchmark_fit_reasons', 'invalid', 'coerced', 'duration_clamped_to', 'config_after', 'downstream_resets', 'still_unset'):
            if key in info:
                result[key] = copy.deepcopy(info[key])
        if analysis is not None:
            result['analysis'] = copy.deepcopy(analysis)
        if event:
            result['event'] = str(event)
        if forced_stop:
            result['forced_stop_after_action'] = True
        self._session_history.append({'turn': int(self.turn), 'action': copy.deepcopy(action), 'result': result})

    def step(self, action: ActionJSON, *, model_text: Optional[str]=None, model_logprobs: Optional[list[float]]=None, model_analysis: Optional[dict]=None) -> StepOutcome:
        if self._terminated:
            raise RuntimeError('episode is over; call reset()')
        policy_observation = self._observe().to_dict()
        self.turn += 1
        requested = self._requested_commits(action)
        executed_action = action
        previous: dict[str, Any] = {}
        coupler_idx = self._label_index(action.get('coupler') if isinstance(action, dict) else None)
        if coupler_idx is not None:
            previous = self._env.configs[coupler_idx].as_dict()
        try:
            structure_errors = [] if is_action(action) else _window_action_structure_errors(action)
            if structure_errors:
                res = StepResult(None, cost_seconds=0.0, reward=0.0, terminated=False, truncated=False, info={'invalid': structure_errors})
            else:
                res = self._env.step(action)
        except EpisodeAbort:
            raise
        except Exception as exc:
            res = StepResult(None, cost_seconds=0.0, reward=0.0, terminated=False, truncated=False, info={'invalid': [f'parse error: {exc}']})
        if self._env.discovery is not None:
            discovery_state = self._env.discovery.visible()
            self._ctx_raw["discovery"] = discovery_state
            for transition in ("f01", "f12"):
                self._ctx_raw[f"clock_freqs:{transition}"] = {
                    q: values[transition] for q, values in self._env.discovery.committed.items()
                    if transition in values}
        invalid = bool(res.info.get('invalid'))
        if invalid:
            self._n_invalid += 1
        committed = res.info.get('committed')
        coupler = res.info.get('coupler')
        if not invalid and committed is not None and (coupler_idx is not None):
            config_after = self._env.configs[coupler_idx].as_dict()
            downstream_resets = [param for param in COMMITTABLE if previous.get(param) is not None and config_after.get(param) is None]
            res.info['config_after'] = _json_safe(config_after)
            res.info['downstream_resets'] = downstream_resets
            res.info['still_unset'] = [param for param in COMMITTABLE if config_after.get(param) is None]
        self.total_time = self._env.total_time
        step_analysis: Optional[dict[str, Any]] = None
        if res.dataset is not None:
            self._last_measurement_dataset = res.dataset
            self._last_measurement_images = [str(path) for path in (res.info.get('images') or [])]
            try:
                self._last_analysis = analyze_dataset(res.dataset, self._coupler_context(coupler))
            except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
                # A fit that fails numerically must cost the agent one analysis, never the episode.
                self._last_analysis = {'node': res.dataset.attrs.get('node'), 'status': 'ANALYSIS_FAILED',
                                       'error': f'{type(exc).__name__}: {exc}'}
            if self._last_analysis.get('node') == 'cz_rb':
                self._last_analysis = annotate_rb_identifiability(self._last_analysis, res.info)
                fit = self._last_analysis['fit_diagnostics']
                res.info['benchmark_identifiable'] = bool(fit['identifiable'])
                res.info['benchmark_fit_reasons'] = list(fit['reasons'])
            step_analysis = copy.deepcopy(self._last_analysis)
            self._last_analysis_coupler = coupler
            self._last_analysis_node = res.info.get('node') or self._last_analysis.get('node')
            self._record_instrument_notebook(coupler, self._last_analysis_node, self._last_analysis, executed_action)
            self._register_candidates(coupler, self._last_analysis_node, self._last_analysis, executed_action)
        self._update_ledger(action, res, invalid, requested, previous)
        if not invalid and 'F_lcb' in res.info and (coupler is not None):
            self._register_benchmark_config_history(str(coupler), res.info.get('F_lcb'))
            self._register_best_config_candidate(str(coupler), res.info.get('F_lcb'))
            self._register_bias_search_candidates(str(coupler))
        # A discovery scan addresses a qubit, not a coupler: name the element it
        # actually measured so neither the model nor a reader sees "on None".
        element = coupler if coupler is not None else (
            action.get('qubit') if isinstance(action, dict) else None)
        self._last_event = self._event_text(res, invalid, committed, element)
        notes = (res.info or {}).get('coerced') if isinstance(res.info, dict) else None
        if self._last_event and notes:
            # Guardrail repairs change what was executed: say so in the line every observation format carries.
            self._last_event += ' | guardrail: ' + '; '.join(str(note) for note in notes)
        action_event = self._last_event
        reward, comps = self._reward(res, invalid, committed)
        self.transcript.total_reward += reward
        mn, _ = self._min_lcb_and_pass()
        self._best_min_f_lcb = mn if self._best_min_f_lcb is None else max(self._best_min_f_lcb, mn)
        self._trace.append({'turn': self.turn, 'label': self._label(action, res), 'cost': round(float(res.cost_seconds), 3), 'F_lcb': res.info.get('F_lcb'), 'invalidated': res.info.get('benchmarks_invalidated'), 'invalid': invalid})
        action_res = res
        terminated = bool(res.terminated)
        truncated = bool(res.truncated)
        if terminated or truncated or self.turn >= self.max_turns:
            self._terminated = True
            if not (terminated or truncated):
                res = self._env.step({'stop': True})
                terminated = True
                term_reward, _ = self._terminal_reward(res)
                reward += term_reward
                self.transcript.total_reward += term_reward
                comps['forced_stop'] = True
                self._last_event = 'reached turn cap - forced stop'
            self.transcript.terminated = terminated
            self.transcript.truncated = truncated
            self.transcript.terminal_reward = res.reward
            self.transcript.final_config = _json_safe([k.as_dict() for k in self._env.configs])
        self.last_raw_result = action_res
        self._record_window_history(action, action_res, analysis=step_analysis, event=action_event, forced_stop=action_res is not res)
        obs = self._observe()
        tool_result = _json_safe({'info': action_res.info, 'cost_seconds': action_res.cost_seconds})
        if executed_action is not action:
            tool_result['executed_action'] = _json_safe(executed_action)
        if action_res is not res:
            tool_result['forced_stop'] = _json_safe({'info': res.info, 'cost_seconds': res.cost_seconds})
        self.transcript.turns.append(Turn(index=self.turn, observation=obs.to_dict(), action=_json_safe(action), valid=not invalid, reward=reward, reward_components=comps, tool_result=tool_result, model_text=model_text, model_logprobs=model_logprobs, model_analysis=model_analysis, policy_observation=policy_observation))
        return StepOutcome(obs, reward, terminated, truncated, _json_safe(res.info))

    def _reward(self, res: StepResult, invalid: bool, committed) -> tuple[float, dict]:
        if res.terminated or res.truncated:
            return self._terminal_reward(res)
        reward, comps = (0.0, {})
        if self.reward_cfg.shaping and committed is not None:
            phi_now = oracle.rho(self._env.privileged_min_fidelity())
            reward = phi_now - self._phi_prev
            comps['shaping'] = reward
            self._phi_prev = phi_now
        if invalid:
            reward -= self.reward_cfg.invalid_penalty
            comps['invalid_penalty'] = -self.reward_cfg.invalid_penalty
        return (reward, comps)

    def _pc_credit(self, v) -> float:
        """Per-coupler credit for the partial-certification bonus.

        Plain ``rho(F_lcb)`` unless ``pc_linear_floor``: rho is nearly flat over low LCBs, so the
        bonus sees almost no contrast between a barely-benchmarked and a half-tuned gate. The
        floor replaces the sub-f_star branch with a linear ramp to
        rho(f_star): continuous at f_star, order-preserving, and identical to rho above f_star
        (rho is convex with slope > the ramp's there), so the certification optimum is untouched."""
        f = 0.0 if v is None else float(v)
        r = oracle.rho(f)
        if not self.pc_linear_floor or self.f_star <= 0.0:
            return r
        return max(r, oracle.rho(self.f_star) * f / self.f_star)

    def _terminal_reward(self, res: StepResult) -> tuple[float, dict]:
        terminal_true = float(res.reward)
        closure = 0.0 - self._phi_prev if self.reward_cfg.shaping else 0.0
        lcbs = self._env._valid_lcbs()
        rhos = [oracle.rho(0.0 if v is None else float(v)) for v in lcbs] if lcbs else []
        credits = [self._pc_credit(v) for v in lcbs] if lcbs else []
        mean_credit = sum(credits) / len(credits) if credits else 0.0
        min_rho = min(rhos) if rhos else 0.0
        bonus = self.partial_credit_alpha * (mean_credit - min_rho) if credits else 0.0
        comps = {'terminal_true': terminal_true, 'shaping_closure': closure, 'partial_credit': bonus, 'mean_pc_credit': mean_credit, 'n_valid_lcbs': sum((1 for v in lcbs if v is not None)), 'n_certified': sum((1 for v in lcbs if v is not None and float(v) >= self.f_star)), 'n_certified_090': sum((1 for v in lcbs if v is not None and float(v) >= 0.9)), 'all_pass_090': float(bool(lcbs) and all((v is not None and float(v) >= 0.9 for v in lcbs))), 'mean_rho_lcb': sum(rhos) / len(rhos) if rhos else 0.0, 'sim_time_s': float(self._env.total_time), 'n_invalid_actions': int(self._n_invalid), 'n_measurements': int(sum((v['measurements'] for v in self._ledger.values()))), 'n_commit_actions': int(sum((v['commit_actions'] for v in self._ledger.values()))), 'n_parameters_written': int(sum((v['parameters_written'] for v in self._ledger.values()))), 'n_attempts': int(sum((v['attempts'] for v in self._ledger.values()))), 'n_unobserved_rewrites': int(sum((v['unobserved_rewrites'] for v in self._ledger.values()))), 'n_redundant_commits': int(sum((v['redundant_commits'] for v in self._ledger.values()))), 'n_unsupported_writes': int(sum((v['unsupported_writes'] for v in self._ledger.values())))}
        self._terminal_comps = comps
        return (terminal_true + closure + bonus, comps)

    def terminal_stats(self) -> Optional[dict[str, Any]]:
        """Episode-outcome diagnostics for the trainer's per-iter learning curve.

        ``None`` until the episode has terminated.  Keys: ``partial_credit``,
        ``n_valid_lcbs``, ``n_certified``, ``mean_rho_lcb``, ``sim_time_s``,
        ``n_invalid_actions`` (plus the terminal reward components)."""
        return dict(self._terminal_comps) if self._terminal_comps else None

    def _min_lcb_and_pass(self) -> tuple[float, bool]:
        lcbs = self._env._valid_lcbs()
        eff = [0.0 if v is None else float(v) for v in lcbs]
        mn = min(eff) if eff else 0.0
        ap = bool(lcbs) and all((v is not None and v >= self.f_star for v in lcbs))
        return (mn, ap)

    def _coupler_context(self, coupler_label: Optional[str]) -> dict[str, Any]:
        """The per-coupler context ``analyze_dataset`` needs (control/target labels + any
        disclosed 1Q QOIs for that pair), built from the reset context."""
        couplers = self._ctx_raw.get('couplers', [])
        idx = self._label_index(coupler_label)
        if idx is None or idx >= len(couplers):
            return {'qubits': ('q0', 'q1')}
        c = couplers[idx]
        ctrl, tgt = (c['control'], c['target'])
        out: dict[str, Any] = {'qubits': (ctrl, tgt)}
        f01 = self._ctx_raw.get('clock_freqs:f01') or {}
        f12 = self._ctx_raw.get('clock_freqs:f12') or {}
        if ctrl in f01 and tgt in f01 and (ctrl in f12) and (tgt in f12):
            out['clock_freqs:f01'] = {ctrl: f01[ctrl], tgt: f01[tgt]}
            out['clock_freqs:f12'] = {ctrl: f12[ctrl], tgt: f12[tgt]}
        confusion = self._ctx_raw.get('confusion') or {}
        if ctrl in confusion and tgt in confusion:
            out['confusion'] = {ctrl: confusion[ctrl], tgt: confusion[tgt]}
        return out

    def action_errors(self, action) -> list[str]:
        """Why this action would be rejected, without measuring it.

        Empty means the env would accept it. This is the same verdict ``step``
        applies, including the device's schedule grid and NCO limits.
        """
        return self._env.action_errors(action)

    def committed(self, coupler: str) -> CommittedConfig:
        """The configuration committed so far for one coupler, e.g. 'c0'.

        Useful to validate a proposed window against what is already committed.
        """
        index = self._label_index(coupler)
        if index is None:
            raise KeyError(f'unknown coupler {coupler!r}; have {self._env.coupler_labels()}')
        return self._env.configs[index]

    def _label_index(self, coupler_label: Optional[str]) -> Optional[int]:
        if coupler_label is None:
            return None
        try:
            return self._env.coupler_labels().index(str(coupler_label))
        except ValueError:
            return None

    def _coupler_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        couplers = self._ctx_raw.get('couplers', [])
        for i, c in enumerate(couplers):
            k = self._env.configs[i]
            b = self._env.benchmarks[i]
            config = _json_safe(k.as_dict())
            missing = [param for param in COMMITTABLE if config.get(param) is None]
            bits = []
            bits.append(f'bias={_fmt_a(k.parking_current)}' if k.parking_current is not None else 'bias=UNSET')
            if k.cz_pulse_frequency is not None:
                bits.append(f'fd={_fmt_hz(k.cz_pulse_frequency)}')
            if k.cz_pulse_amplitude is not None:
                bits.append(f'amp={_fmt_float(k.cz_pulse_amplitude)}')
            if k.cz_pulse_duration is not None:
                bits.append(f'dur={_fmt_s(k.cz_pulse_duration)}')
            if b is not None and b.f_lcb is not None:
                status = 'PASS' if b.f_lcb >= self.f_star else 'FAIL'
                metric = self._ctx_raw.get('fidelity_label') or 'F_lcb'
                sigma = getattr(b, 'f_sigma', None)
                fl = f'{metric}={b.f_lcb:.4f}' + (f' +/- {sigma:.4f}' if metric == 'F' and sigma else '') + f' {status}'
                f_lcb: Optional[float] = float(b.f_lcb)
            elif k.is_complete():
                prior_benchmarks = int((self._ledger.get(str(c['label'])) or {}).get('benchmarks', 0))
                fl = 'STALE (configuration changed since benchmark; re-benchmark)' if prior_benchmarks else 'UNBENCHMARKED (core gate complete; phase nulls execute as zero correction)'
                status = 'STALE' if prior_benchmarks else 'UNBENCHMARKED'
                f_lcb = None
            else:
                fl = 'INCOMPLETE (commit missing prerequisites before benchmark)'
                status = 'INCOMPLETE'
                f_lcb = None
            line = f"{c['label']} [control={c['control']} -> target={c['target']}] " + ' '.join(bits) + ' | ' + fl
            rows.append({'label': c['label'], 'control': c['control'], 'target': c['target'], 'f_lcb': f_lcb, 'status': status, 'line': line, 'config': config, 'missing': missing, 'core_complete': bool(k.is_complete()), 'phase_corrections_committed': {'control': config.get('control_local_phase') is not None, 'target': config.get('target_local_phase') is not None}})
        return rows

    def _qoi_lines(self) -> list[str]:
        ctx = self._ctx_raw
        if self.scout_qoi or 'clock_freqs:f01' not in ctx:
            return ['NOT disclosed - earn f01/f12/T1/readout with qubit_spectroscopy/qubit_t1/qubit_ramsey/readout_calibration before calibrating']
        f01 = ctx.get('clock_freqs:f01', {})
        f12 = ctx.get('clock_freqs:f12', {})
        t1 = ctx.get('t1_time', {})
        lines = []
        for q in ctx.get('qubits', []):
            bits = []
            if q in f01:
                bits.append(f'f01={float(f01[q]) * 1e-09:.4f}GHz')
            if q in f12:
                bits.append(f'f12={float(f12[q]) * 1e-09:.4f}GHz')
            if q in t1:
                bits.append(f'T1={float(t1[q]) * 1000000.0:.0f}us')
            lines.append(f'{q}: ' + ' '.join(bits))
        return lines

    def _event_text(self, res: StepResult, invalid: bool, committed, coupler) -> Optional[str]:
        info = res.info
        if invalid:
            return f"INVALID action rejected: {info.get('invalid')}"
        if res.terminated or res.truncated:
            return None
        if committed is not None:
            rendered = ', '.join((f'{param}={_fmt_commit_value(param, value)}' for param, value in committed.items()))
            prefix = 'committed candidate transaction' if info.get('candidate_id') else 'committed'
            txt = f'{prefix} {rendered} on {coupler}'
            resets = list(info.get('downstream_resets') or [])
            if resets:
                txt += ' -> dependency reset set downstream field(s) to null: ' + ','.join((str(param) for param in resets))
            inval = info.get('benchmarks_invalidated') or []
            others = [c for c in inval if c != coupler]
            if others:
                txt += f" -> flux crosstalk INVALIDATED benchmarks on {','.join(inval)}"
            elif inval:
                txt += ' -> its own standing benchmark is now stale'
            if info.get('duration_clamped_to'):
                txt += f" (duration clamped to {info['duration_clamped_to']})"
            return txt
        if 'F_lcb' in info:
            fl = info['F_lcb']
            if info.get('fit_reliable') is False:
                return (f"benchmarked {coupler}: RB not identifiable (reference decay p_ref={float(info.get('reference_decay', 0.0)):.3f}; "
                        "the survival curves reach the 1/4 floor within a few Cliffords; both curves contain the committed CZ, "
                        "on average 1.5 per reference Clifford). No fidelity estimate; it counts as no benchmark.")
            if fl is None and info.get('benchmark_identifiable') is False and info.get('F_hat') is not None:
                reasons = '; '.join(info.get('benchmark_fit_reasons') or ['fit quality failed'])
                sigma = info.get('F_sigma')
                leak = (info.get('rb_analysis') or {}).get('leakage_per_clifford')
                return (f"benchmarked {coupler}: RB UNIDENTIFIABLE ({reasons}); F_hat={float(info['F_hat']):.4f}"
                        + (f", F_sigma={float(sigma):.4f}" if sigma is not None else '')
                        + (f", leakage {float(leak):.3f} per reference Clifford" if leak is not None else '')
                        + '. It counts as no benchmark.')
            if fl is None:
                return f'benchmarked {coupler}: IRB produced no valid F_lcb'
            if info.get('benchmark_identifiable') is False:
                reasons = '; '.join(info.get('benchmark_fit_reasons') or ['fit quality failed'])
                return f"benchmarked {coupler}: RB UNIDENTIFIABLE ({reasons}); F_hat={float(info.get('F_hat', 0.0)):.4f}, F_sigma={float(info.get('F_sigma', 0.0)):.4f}, F_lcb={fl:.4f}."
            metric = self._ctx_raw.get('fidelity_label') or 'F_lcb'
            sigma = info.get('F_sigma') if metric == 'F' else None
            return (f"benchmarked {coupler}: {metric}={fl:.4f}" + (f" +/- {float(sigma):.4f}" if sigma else '')
                    + f" {('PASS' if info.get('passed') else 'FAIL')}")
        if info.get('frequency_committed'):
            fc = info['frequency_committed']
            return f"committed {fc['qubit']} {fc['transition']} = {float(fc['frequency_hz']) / 1e9:.6f} GHz"
        node = info.get('node')
        if node:
            where = f' on {coupler}' if coupler is not None else ''
            return f'measured {node}{where} (see structured analysis below)'
        return None

    def _observe(self) -> ChainObservation:
        mn, ap = self._min_lcb_and_pass()
        session_history, history_truncated = self._history_view()
        return ChainObservation(turn=self.turn, max_turns=self.max_turns, remaining_budget_seconds=float(self._env.tau), total_time=float(self._env.total_time), objective_f_star=self.f_star, context=_json_safe(self._ctx_raw), couplers=self._coupler_rows(), min_f_lcb=mn, all_pass=ap, best_min_f_lcb=self._best_min_f_lcb, qoi_lines=self._qoi_lines(), last_event=self._last_event, last_analysis=_json_safe(self._last_analysis), last_analysis_coupler=self._last_analysis_coupler, last_analysis_node=self._last_analysis_node, trace_tail=list(self._trace[-self.trace_tail:]), calibration_ledger=self._ledger_view(), instrument_notebook=copy.deepcopy(self._instrument_notebook), session_history=session_history, history_truncated_turns=history_truncated, history_contract='window_full_history_v1', measurement_dataset=self._last_measurement_dataset, measurement_images=list(self._last_measurement_images))

    @staticmethod
    def _label(action: ActionJSON, res: StepResult) -> str:
        if not isinstance(action, dict):
            return str(action)
        tag = f"@{action.get('coupler')}" if action.get('coupler') else ''
        if 'node' in action:
            return f"measure:{action['node']}{tag}"
        if 'commit' in action:
            commit = action['commit']
            param = commit.get('param') if isinstance(commit, dict) else commit
            return f'commit:{param}{tag}'
        if action.get('benchmark'):
            return f'benchmark{tag}'
        if action.get('stop'):
            return 'stop'
        return '?'

    def checkpoint(self) -> dict[str, Any]:
        return {'configs': copy.deepcopy(self._env.configs), 'benchmarks': copy.deepcopy(self._env.benchmarks), 'benchmark_provenance': copy.deepcopy(self._env._benchmark_provenance), 'tau': self._env.tau, 'total_time': self._env.total_time, 'wall_clock': getattr(self._env, 'wall_clock', 0.0), 'frames': copy.deepcopy(getattr(self._env, '_frames', None)), 'realism_rng_state': copy.deepcopy(self._env._realism_rng.bit_generator.state) if getattr(self._env, '_realism', None) is not None else None, 'done': self._env._done, 'rng_state': copy.deepcopy(self._env._rng.bit_generator.state), 'phi_prev': self._phi_prev, 'best_min': self._best_min_f_lcb, 'turn': self.turn, 'terminated': self._terminated, 'last_raw_result': copy.deepcopy(self.last_raw_result), 'last_event': self._last_event, 'last_analysis': copy.deepcopy(self._last_analysis), 'last_analysis_coupler': self._last_analysis_coupler, 'last_analysis_node': self._last_analysis_node, 'last_measurement_dataset': copy.deepcopy(self._last_measurement_dataset), 'trace': copy.deepcopy(self._trace), 'session_history': copy.deepcopy(self._session_history), 'instrument_notebook': copy.deepcopy(self._instrument_notebook), 'ledger': copy.deepcopy(self._ledger), 'candidate_registry': copy.deepcopy(self._candidate_registry), 'n_invalid': self._n_invalid, 'terminal_comps': copy.deepcopy(self._terminal_comps), 'transcript': copy.deepcopy(self.transcript)}

    def restore(self, cp: dict[str, Any]) -> None:
        self._env.configs = copy.deepcopy(cp['configs'])
        self._env._benchmarks = copy.deepcopy(cp['benchmarks'])
        self._env._benchmark_provenance = copy.deepcopy(cp.get('benchmark_provenance', [None] * len(self._env._benchmarks)))
        self._env._enforce_benchmark_provenance()
        self._env.tau = cp['tau']
        self._env.total_time = cp['total_time']
        self._env.wall_clock = cp.get('wall_clock', 0.0)
        self._env._frames = copy.deepcopy(cp.get('frames'))
        if cp.get('realism_rng_state') is not None:
            self._env._realism_rng.bit_generator.state = copy.deepcopy(cp['realism_rng_state'])
        self._env._done = cp['done']
        if 'rng_state' in cp:
            self._env._rng.bit_generator.state = copy.deepcopy(cp['rng_state'])
        self._phi_prev = cp['phi_prev']
        self._best_min_f_lcb = cp['best_min']
        self.turn = cp['turn']
        self._terminated = cp.get('terminated', cp['done'])
        self.last_raw_result = copy.deepcopy(cp.get('last_raw_result'))
        self._last_event = cp.get('last_event')
        self._last_analysis = copy.deepcopy(cp.get('last_analysis'))
        self._last_analysis_coupler = cp.get('last_analysis_coupler')
        self._last_analysis_node = cp.get('last_analysis_node')
        self._last_measurement_dataset = copy.deepcopy(cp.get('last_measurement_dataset'))
        self._trace = copy.deepcopy(cp.get('trace', []))
        self._session_history = copy.deepcopy(cp.get('session_history', []))
        self._instrument_notebook = copy.deepcopy(cp.get('instrument_notebook', []))
        self._ledger = copy.deepcopy(cp.get('ledger', self._ledger))
        self._candidate_registry = copy.deepcopy(cp.get('candidate_registry', {}))
        self._n_invalid = int(cp.get('n_invalid', 0))
        self._terminal_comps = copy.deepcopy(cp.get('terminal_comps'))
        self.transcript = copy.deepcopy(cp.get('transcript', self.transcript))

    def score(self) -> dict[str, Any]:
        """Privileged eval-only scoring (see chipcal.scoring for the benchmark metric)."""
        gt = self._env.ground_truth()
        lcbs = self._env._valid_lcbs()
        eff = [0.0 if v is None else float(v) for v in lcbs]
        from chipcal.scoring import episode_score
        true_min = float(gt['min_true_fidelity'])
        engine = getattr(self._env, 'discovery', None)
        labels = self._env.coupler_labels()
        final = {labels[i]: f for i, (f, _) in enumerate(self._env.final_rb_fidelities())}
        score = {'score': episode_score(final.values()), 'final_rb_fidelity_per_coupler': final,
                 'final_rb_details': {labels[i]: d for i, (_, d) in enumerate(self._env.final_rb_fidelities())},
                 'min_true_fidelity': true_min, 'true_fidelity_per_coupler': gt['true_fidelity_per_coupler'],
                 'success': bool(final) and all(f >= self.f_star for f in final.values()),
                 'f_star': self.f_star,
                 'achieved_min_f_lcb': min(eff) if eff else 0.0, 'certified_f_lcb_per_coupler': eff,
                 'discovery_fraction': None if engine is None else float(engine.accuracy()),
                 'n_turns': int(self.turn), 'n_invalid_actions': int(self._n_invalid),
                 'lab_time_s': float(self._env.total_time), 'family': gt['family'],
                 'metric_revision': 'min_final_rb_v1'}
        self.transcript.score = score
        return score

def run_chain_agent_episode(env: ChainAgentEnv, policy: Callable[[ChainObservation], ActionJSON], *, seed: Optional[int]=None, chain=None, do_score: bool=False, on_turn: Optional[Callable[[Transcript], None]]=None) -> Transcript:
    """Drive one whole-chip episode with a callable ``policy(ChainObservation) -> action json``."""
    obs = env.reset(seed=seed, chain=chain)
    out = StepOutcome(obs, 0.0, False, False, {})
    while not (out.terminated or out.truncated):
        action = policy(out.observation)
        out = env.step(action, model_text=getattr(policy, 'last_text', None), model_analysis=getattr(policy, 'last_analysis', None))
        env.transcript.turns[-1].model_usage = copy.deepcopy(getattr(policy, 'last_usage', None))
        if on_turn is not None:
            on_turn(env.transcript)
    if do_score:
        env.score()
    return env.transcript
CHAIN_WINDOW_DECISION_FORMS = 'Decision forms (field names only; every physical number must come from your own measurements; there are no prompt defaults):\nmeasure: {"coupler":"cN","node":"<measurement node>","sweeps":{"<axis>":"[center,full_span,point_count]"},"shots":"<integer>"}\ncommit: {"coupler":"cN","commit":{"param":"<physical parameter>","value":"<physical value>"}}\nbenchmark: {"coupler":"cN","benchmark":true}\nstop: {"stop":true}\nUnits and grid: every number is in SI base units -- seconds, hertz, amperes, radians, and normalized amplitude (1 ns is 1e-09 s, 1 us is 1e-06 s, 1 MHz is 1e+06 Hz, 1 GHz is 1e+09 Hz, 1 mA is 1e-03 A). Pulse durations must be integer multiples of the schedule timing grid stated in the measurement profile, including both endpoints and the step of a [center,full_span,point_count] axis. A fixed scalar sweep coordinate is a JSON number, not [value,0,1]. Coupler orientation is control -> target, with control the higher-f01 qubit.\n\n'
CHAIN_WINDOW_NODE_REFERENCE = ('MEASUREMENT NODES (what each one does; every node returns 3-state readout counts of both qubits of the addressed coupler at every sweep point, plus integrated IQ statistics):\n'
    '- coupler_dc_spectroscopy: sweeps the coupler bias current (dc_currents, A) and a drive tone frequency (qubit_frequencies, Hz) applied to drive_qubit (0=control, 1=target). Optional spec_pulse_duration [s], spec_pulse_amplitude [DAC].\n'
    '- qubit_spectroscopy: drive-frequency scan at the committed bias; transition 1 drives |0>-|1>, transition 2 prepares |1> and drives |1>-|2>.\n'
    '- cz_parametrization: prepares |11>, applies a flux pulse of fixed duration (cz_pulse_duration, default 280 ns) modulated at cz_pulse_frequencies with amplitude cz_pulse_amplitudes, optionally at several dc_currents.\n'
    '- cz_chevron: prepares |11>, applies the flux pulse at a fixed amplitude (cz_pulse_amplitude, or the committed one) over cz_pulse_frequencies x cz_pulse_durations.\n'
    '- cz_calibration: at the committed amplitude, for each working point [frequency_Hz, duration_s], a Ramsey sequence on the target with the control in |0> or |1> (control_ons) around the flux pulse, read out at analysis phases ramsey_phases [deg].\n'
    '- cz_local_phases: at the committed gate, a Ramsey sequence on each qubit (swap selects which qubit is measured) with the flux pulse off or on (gate_modes), read out at analysis phases local_phases [deg].\n'
    '- qubit_t1: excites both qubits and waits for delays [s]. qubit_ramsey: Ramsey sequences versus delays [s] with an artificial ramsey_detuning [Hz].\n'
    '- readout_calibration: prepares |0>, |1> and |2> and reads them out.\n'
    'benchmark: interleaved randomized benchmarking of the committed CZ on that coupler; reports F_hat, F_sigma and F_lcb = F_hat - 2 F_sigma, or no estimate when the decay is not identifiable (curves at the 1/4 floor).\n'
    'Commits: parking_current [A] sets the coupler DC bias; cz_pulse_frequency [Hz], cz_pulse_amplitude, cz_pulse_duration [s] define the CZ flux pulse; control_local_phase and target_local_phase [rad] are the single-qubit phase corrections applied with the gate.\n\n')
CHAIN_WINDOW_OBJECTIVE = ('OBJECTIVE AND SCORE: the episode ends when you stop, at the turn cap, or when the lab-time budget runs out. The environment then runs one final interleaved randomized benchmark of every coupler\'s final committed gate at the final bias currents, and the score is the MINIMUM over couplers of those fitted fidelities. A gate whose final benchmark is not identifiable (survival at the 1/4 floor) scores 0.25; a coupler with an incomplete configuration scores 0. Your own benchmarks during the episode are measurements for you; they do not enter the score. The chip counts as calibrated when every coupler reaches f_star. Re-biasing one coupler can shift its neighbours through hidden flux crosstalk.\n\n')
DEFAULT_CHAIN_WINDOW_SYSTEM_PROMPT = ('You are calibrating every CZ coupler of a simulated superconducting-qubit chip. On each turn choose one action: a measurement on a coupler, a commit of one physical parameter, a benchmark, or stop. Deterministic code only validates your action against the grammar and the instrument limits and executes it.\n\n'
    + CHAIN_WINDOW_OBJECTIVE + CHAIN_WINDOW_DECISION_FORMS + CHAIN_WINDOW_NODE_REFERENCE
    + "SWEEP COORDINATES: [center,full_span,point_count] has inclusive endpoints center-full_span/2 and center+full_span/2 in the axis's SI units. It is NOT [minimum,maximum,points], and span is not a half-width. Every flat three-number list is interpreted as this triple, never as three explicit samples.\n\n"
    + 'Use only information in the observation. Hidden device parameters are never available. Emit exactly one JSON object.')
CHAIN_WINDOW_JSON_REASON_INSTRUCTION = 'Return exactly one JSON object with two keys: reason and action, and no prose or markdown. reason must contain the string fields evidence, state, and next_step. action must be exactly one executable action from the action grammar. Keep all reasoning inside JSON strings.'
CHAIN_WINDOW_NATIVE_THINKING_INSTRUCTION = "Reason in the model's native thinking channel and finish within its token cap. After the thinking channel closes, return exactly one executable action as a bare JSON object, with no prose, markdown fence, or wrapper object."
CHAIN_WINDOW_DISCOVERY_THINKING_INSTRUCTION = CHAIN_WINDOW_NATIVE_THINKING_INSTRUCTION

def default_chain_window_system_prompt(tool_schema: Optional[dict[str, Any]]=None, *, reason: bool=False, reason_json: bool=False, native_thinking: bool=False, discovery: bool=False, discovery_phase: bool=False) -> str:
    """Canonical prompt for ``ChainAgentEnv(action_mode="window")``.

    Training and evaluation should both build their prompt here, so the model sees the same
    instructions in both.

    ``discovery_phase`` selects the short native-thinking variant used while the chip is
    still undiscovered, so the reasoning budget is not spent auditing a coupler ladder that
    has no committed evidence yet.
    """
    if sum((bool(mode) for mode in (reason, reason_json, native_thinking))) > 1:
        raise ValueError('reason, reason_json, and native_thinking are mutually exclusive')
    if discovery_phase and not (discovery and native_thinking):
        raise ValueError('discovery_phase requires discovery and native_thinking')
    msg = (DISCOVERY_PROMPT + "\n\n" if discovery else "") + DEFAULT_CHAIN_WINDOW_SYSTEM_PROMPT
    if reason:
        msg += '\n\n' + REASON_INSTRUCTION
    elif reason_json:
        msg += '\n\n' + CHAIN_WINDOW_JSON_REASON_INSTRUCTION
    elif native_thinking:
        msg += '\n\n' + (CHAIN_WINDOW_DISCOVERY_THINKING_INSTRUCTION if discovery_phase else CHAIN_WINDOW_NATIVE_THINKING_INSTRUCTION)
    if tool_schema is not None:
        msg += '\n\nAction grammar:\n' + json.dumps(tool_schema)
    return msg

def chain_llm_policy(client: LLMClient, *, system_prompt: Optional[Any]=None, tool_schema: Optional[dict]=None, response_schema: Optional[dict]=None, temperature: float=0.0, visual_observations: bool=False, vision_dir: Optional[str]=None, stable_prefix: bool=False) -> Callable[[ChainObservation], ActionJSON]:
    """Wrap an :class:`~chipcal.envs.agent_env.LLMClient` into a whole-chip policy callable.

    ``system_prompt`` may be a string or a ``callable(observation) -> str``, so an episode
    that changes phase -- first-light discovery, then coupler calibration -- can carry the
    instruction block that applies to the phase it is actually in.
    """
    base = system_prompt or DEFAULT_CHAIN_WINDOW_SYSTEM_PROMPT
    suffix = ''
    if tool_schema is not None:
        suffix += '\n\nAction grammar:\n' + json.dumps(tool_schema)
    if visual_observations:
        suffix += '\n\nThe latest finite-shot measurement is supplied as a plot image. Notebook entries contain only measured-count summaries and standard curve fits; hidden device state is never included. The committed configuration, resets, units, limits and action history remain textual.'

    def _system_message(obs: ChainObservation) -> str:
        return (base(obs) if callable(base) else base) + suffix
    plot_root = Path(vision_dir or 'diagnostics/chain_vision')
    episode_serial = -1
    cached_dataset_id: Optional[int] = None
    cached_plot: Optional[Path] = None

    def _policy(obs: ChainObservation) -> ActionJSON:
        nonlocal episode_serial, cached_dataset_id, cached_plot
        if obs.turn == 0:
            episode_serial += 1
            cached_dataset_id = None
            cached_plot = None
        messages = [{'role': 'system', 'content': _system_message(obs)}, {'role': 'user', 'content': obs.to_prompt_text(visual_measurement=visual_observations, **({'stable_prefix': True} if stable_prefix else {}))}]
        images = None
        supplied = getattr(obs, 'measurement_images', None) or []
        if visual_observations and supplied:
            # Figures supplied with the observation take precedence over rendering the dataset.
            cached_plot = Path(supplied[0])
            images = [Path(path).read_bytes() for path in supplied]
        elif visual_observations and obs.measurement_dataset is not None:
            dataset_id = id(obs.measurement_dataset)
            if cached_dataset_id != dataset_id or cached_plot is None:
                from chipcal.llm.parsing import render_observation_image
                tag = '_'.join((part for part in (f'turn_{obs.turn:03d}', obs.last_analysis_coupler, obs.last_analysis_node) if part))
                cached_plot = render_observation_image(obs.measurement_dataset, plot_root / f'episode_{episode_serial:03d}', tag)
                if cached_plot is None or not cached_plot.is_file():
                    raise RuntimeError('visual observation rendering did not produce a PNG')
                cached_dataset_id = dataset_id
            images = [cached_plot.read_bytes()]
        completion = client.generate(messages, images=images, schema=response_schema, temperature=temperature)
        _policy.last_text = completion.text
        _policy.last_usage = completion.raw
        _policy.last_plot_path = str(cached_plot) if images else None
        try:
            from chipcal.llm.parsing import extract_json_analysis
            _policy.last_analysis = extract_json_analysis(completion.text)
            return extract_json_action(completion.text)
        except Exception:
            _policy.last_analysis = None
            return {'__unparseable__': True}
    _policy.last_text = None
    _policy.last_usage = None
    _policy.last_analysis = None
    _policy.last_plot_path = None
    return _policy
