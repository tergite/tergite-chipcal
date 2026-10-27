"""ChainCalibrationEnv: joint CZ tune-up of every coupler of a qubit chain.

One episode tunes up *all* ``n_couplers`` CZ gates of a
:class:`~chipcal.sim.chain.ChainDeviceParams` under one budget. A **hidden
flux-crosstalk matrix** couples the coupler biases (``Phi = A . I``), so

* every action is addressed to a coupler (``"coupler": "c2"``); it runs the existing
  measurement/commit/benchmark machinery on that coupler's effective theta, and
* re-biasing one coupler shifts its neighbours' fluxes -- so the env invalidates the
  benchmarks of every coupler whose flux moved (a stale/valid calibration ledger),
  and a neighbour's committed gate genuinely degrades.

The terminal reward is driven by the weakest gate. ``reward_mode="final_rb"`` (the
benchmark score, :mod:`chipcal.scoring`) is the minimum over couplers of a final
interleaved-RB fit that the environment runs on every coupler's final committed gate.
``"true_fidelity"`` is the minimum of the simulator's true average CZ fidelities instead. ``"nines"`` is ``rho(min_i F_lcb[i]) - lambda*T`` over the
standing RB lower bounds, and ``"binary"`` rewards all-pass. You cannot trade one gate
off against another, and re-biasing to fix one gate can break its neighbours.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from chipcal.sim import chain as chainmod
from chipcal.envs import nodes
from chipcal.sim import oracle
from chipcal.core.actions import (
    BenchmarkAction,
    CommitAction,
    MeasureAction,
    StopAction,
    augment_prereq_hints,
    axis_values,
    coerce_action,
    unit_slips,
    parse_action,
    validate,
)
from chipcal.core.config import COMMITTABLE, CommittedConfig
from chipcal.envs.env import CostModel, DC_CURRENT_LIMIT_A, StepResult
from chipcal.sim.readout import NoiseModel

# A benchmark certifies one committed gate under one joint-bias configuration.  Any
# actual change on a bias line coupled into that gate changes its Hamiltonian, so the
# benchmark is stale. (A flux-change threshold would let several individually
# sub-threshold moves accumulate while a cached passing LCB survived.)
#
# Crosstalk support is read directly from the device matrix, avoiding cancellation
# in a before/after flux subtraction.  Current equality is exact because the same
# committed floats form the provenance fingerprint.


def _coupler_index(raw: Any, n_couplers: int) -> Optional[int]:
    """Resolve the ``coupler`` field of an action ("c2" / "C2" / 2) to an index."""
    if raw is None:
        return None
    if isinstance(raw, (int, np.integer)):
        i = int(raw)
    elif isinstance(raw, str):
        s = raw.strip().lower().lstrip("c")
        if not s.isdigit():
            return None
        i = int(s)
    else:
        return None
    return i if 0 <= i < n_couplers else None


class ChainCalibrationEnv:
    """Hidden-parameter MDP over a coupler chain's joint CZ tune-up."""

    def __init__(
        self,
        prior: Optional[chainmod.ChainPrior] = None,
        backend: str = "multimode",
        budget_seconds: float = 14400.0,
        lambda_per_minute: float = 0.01,
        f_star: float = 0.98,
        reward_mode: str = "nines",
        family: str = "nominal",
        n_qubits: int = 5,
        debug: bool = False,
        noise: Optional[NoiseModel] = None,
        scout_bias: bool = False,
        scout_qoi: bool = False,
        duration_limits: Optional[tuple[float, float]] = None,
        guardrails: bool = False,
        spectroscopy=None,
        measurement_profile=None,
        discovery=False,
    ):
        if backend != "multimode":
            raise NotImplementedError(f"backend {backend!r}: only multimode is supported")
        if reward_mode not in ("final_rb", "true_fidelity", "nines", "binary"):
            raise ValueError("reward_mode must be 'final_rb', 'true_fidelity', 'nines' or 'binary'")
        if duration_limits is not None:
            lo, hi = float(duration_limits[0]), float(duration_limits[1])
            if not (0.0 < lo < hi):
                raise ValueError(f"duration_limits must be 0 < lo < hi, got {duration_limits}")
            duration_limits = (lo, hi)
        self.prior = prior or chainmod.ChainPrior(n_qubits=n_qubits)
        self.backend = backend
        if backend == "multimode":
            from chipcal.sim.multimode import MultimodeSettings
            spectroscopy = spectroscopy or MultimodeSettings()
            if not isinstance(spectroscopy, MultimodeSettings):
                raise ValueError("multimode requires MultimodeSettings, not the legacy spectroscopy settings")
        self.spectroscopy = spectroscopy
        from .one_qubit_discovery import DiscoverySettings
        self.discovery_settings = (DiscoverySettings() if discovery is True else discovery) or None
        self.discovery = None
        self.measurement_profile = measurement_profile
        self.budget_seconds = budget_seconds
        self.lambda_per_minute = lambda_per_minute
        self.f_star = f_star
        self.reward_mode = reward_mode
        self.family = family
        self.n_qubits = int(n_qubits)
        self.debug = debug
        self.noise = noise
        self.scout_bias = scout_bias
        self.scout_qoi = scout_qoi
        # Optional gate-duration window [s]: when set, a committed ``cz_pulse_duration`` is
        # clamped into it (a weak-coupling operating point that would need a slower gate
        # then under-rotates and benchmarks lower). ``None`` keeps the global 8 ns-2 us limit.
        self.duration_limits = duration_limits
        # Grammar guardrails (chipcal.core.actions): reject wrong-unit values with the unit stated
        # (unit_slips), clamp the remaining out-of-range numeric values into LIMITS (coerce_action),
        # and state the missing prerequisite in ordering rejections.
        # Off => strict grammar.
        self.guardrails = bool(guardrails)
        self.cost_model = CostModel()

        self.chain: Optional[chainmod.ChainDeviceParams] = None
        self.configs: list[CommittedConfig] = []
        self._benchmarks: list[Optional[oracle.IRBResult]] = []
        # Parallel measurement-time fingerprints.  A benchmark without matching
        # provenance is never considered valid, even if some mutation bypassed the
        # normal commit/invalidation path.
        self._benchmark_provenance: list[Optional[dict[str, Any]]] = []
        self.reference_currents: list[float] = []
        self.tau = budget_seconds
        self.total_time = 0.0
        self._rng = np.random.default_rng()
        self._done = True

    # ------------------------------------------------------------------ labels
    @property
    def n_couplers(self) -> int:
        return self.chain.n_couplers if self.chain is not None else self.n_qubits - 1

    def qubit_labels(self) -> list[str]:
        return [f"q{i}" for i in range(self.n_qubits)]

    def coupler_labels(self) -> list[str]:
        return [f"c{i}" for i in range(self.n_couplers)]

    def _currents(self) -> list[Optional[float]]:
        """The committed bias-current vector (None where a coupler is unbiased)."""
        return [k.parking_current for k in self.configs]

    @property
    def benchmarks(self) -> list[Optional[oracle.IRBResult]]:
        """Standing IRB results, fail-closed against their measurement provenance.

        Several policies inspect this list directly, so validation lives at the
        property boundary as well as in ``_valid_lcbs``.  This makes an out-of-band
        config mutation stale the affected result before any controller can act on it.
        """
        self._enforce_benchmark_provenance()
        return self._benchmarks

    def _benchmark_fingerprint(self, i: int) -> dict[str, Any]:
        """Exact controls under which coupler ``i``'s standing IRB was measured."""
        currents = self._currents()
        return {
            "config": tuple(
                (param, getattr(self.configs[i], param))
                for param in COMMITTABLE
            ),
            "coupled_currents": tuple(
                (j, currents[j])
                for j in range(self.n_couplers)
                if float(self.chain.crosstalk[i, j]) != 0.0
            ),
        }

    def _clear_benchmark(self, i: int) -> bool:
        """Clear both a standing result and its provenance; return whether one existed."""
        existed = self._benchmarks[i] is not None
        self._benchmarks[i] = None
        self._benchmark_provenance[i] = None
        return existed

    def _enforce_benchmark_provenance(self) -> list[str]:
        """Fail closed every standing result not measured under the current controls."""
        if self.chain is None or not self._benchmarks:
            return []
        if len(self._benchmark_provenance) != len(self._benchmarks):
            # A legacy/malformed restored state has no trustworthy provenance.
            self._benchmark_provenance = [None] * len(self._benchmarks)
        labels = self.coupler_labels()
        stale: list[str] = []
        for i, result in enumerate(self._benchmarks):
            provenance = self._benchmark_provenance[i]
            if result is None:
                # Keep the parallel state canonical even if external code cleared the
                # public list entry directly.
                self._benchmark_provenance[i] = None
                continue
            if provenance is None or provenance != self._benchmark_fingerprint(i):
                self._clear_benchmark(i)
                stale.append(labels[i])
        return stale

    # ------------------------------------------------------------------- reset
    def reset(self, seed: Optional[int] = None,
              chain: Optional[chainmod.ChainDeviceParams] = None):
        self._rng = np.random.default_rng(seed)
        self._episode_seed = 0 if seed is None else int(seed)
        self.final_rb = None
        self.chain = chain if chain is not None else self.prior.sample(self._rng, family=self.family)
        from chipcal.sim.spectroscopy import validate_calibration_tls
        validate_calibration_tls(self.spectroscopy, [c.tls for c in self.chain.couplers])
        if self.backend == "multimode":
            from dataclasses import replace
            self.chain = replace(self.chain, couplers=[replace(c, multimode=c.multimode or self.spectroscopy) for c in self.chain.couplers])
        self.n_qubits = self.chain.n_qubits
        n_c = self.chain.n_couplers
        self._benchmarks = [None] * n_c
        self._benchmark_provenance = [None] * n_c
        self.tau = self.budget_seconds
        self.total_time = 0.0
        self._done = False

        guesses = self.prior.initial_parking_currents(self.chain, self._rng)
        realism = getattr(self.chain, "realism", None)
        # Immutable evaluator reference.  The guesses are sampled for every reset so the device
        # and RNG stream are already identical in scout/non-scout modes; scout mode merely hides
        # them from the policy.  Using literal uncommitted ``None`` lines as a reference evaluates
        # some couplers at zero-useful-flux and can yield a vacuous ceiling of 0.
        self.reference_currents = [float(value) for value in guesses]
        if self.scout_bias:
            self.configs = [CommittedConfig() for _ in range(n_c)]
        else:
            self.configs = [CommittedConfig().commit("parking_current", guesses[i]) for i in range(n_c)]
        # Realism (chipcal.sim.realism): calibrated qubit frames, set as if single-qubit Ramseys had just run,
        # and the wall clock drift runs on. Its own RNG keeps the default stream untouched.
        self._realism = getattr(self.chain, "realism", None)
        self.wall_clock = 0.0
        self._frames = None
        if self._realism is not None:
            self._realism_rng = np.random.default_rng([int(self._realism.seed), 0 if seed is None else int(seed), 17])
            self._frames = [None] * self.n_qubits
            self._calibrate_frames(range(self.n_qubits))

        qlabels, clabels = self.qubit_labels(), self.coupler_labels()
        couplers_ctx = []
        for i in range(n_c):
            ctrl, tgt = self.chain.coupler_qubits(i)
            couplers_ctx.append({"label": clabels[i], "control": qlabels[ctrl], "target": qlabels[tgt]})

        context: dict[str, Any] = {
            "qubits": qlabels,
            "couplers": couplers_ctx,
            "n_couplers": n_c,
            "objective": "all_pass",
            "budget_seconds": self.tau,
            "f_star": self.f_star,
            "bias_limits_A": (-DC_CURRENT_LIMIT_A, DC_CURRENT_LIMIT_A),
        }
        if self.measurement_profile is not None:
            context["measurement_profile"] = self.measurement_profile.describe()
        if self.backend == "multimode":
            from chipcal.sim.multimode import PROTOCOL_DESCRIPTION
            context["spectroscopy_protocol"] = PROTOCOL_DESCRIPTION
        # 1Q QOIs are built unconditionally so the sampled device and RNG order stay
        # identical between disclosed-prior and blackbox runs.  In ``scout_qoi`` mode
        # the policy gets only labels and must earn these values with the
        # qubit_spectroscopy/T1/Ramsey/readout nodes.
        # With realism the lines a measurement finds are dressed by the couplers at the currents actually
        # applied at reset (an uncommitted line carries zero current).
        seen = chainmod.dressed_qubits(self.chain, self._currents())
        qoi = {
            "clock_freqs:f01": {qlabels[q]: seen[q].f01 + self._rng.normal(0, 1e4)
                                for q in range(self.n_qubits)},
            "clock_freqs:f12": {
                qlabels[q]: seen[q].f01 + seen[q].alpha
                + self._rng.normal(0, 2e4)
                for q in range(self.n_qubits)
            },
            "t1_time": {qlabels[q]: self.chain.qubits[q].t1 * np.exp(self._rng.normal(0, 0.05))
                        for q in range(self.n_qubits)},
            "confusion": {qlabels[q]: np.clip(
                self.chain.qubits[q].confusion + self._rng.normal(0, 0.005, (3, 3)), 0.0, 1.0
            ) for q in range(self.n_qubits)},
        }
        if self.scout_qoi or self.discovery_settings:
            context["characterize_qubits"] = qlabels
        else:
            context["clock_freqs:f01"] = qoi["clock_freqs:f01"]
            context["clock_freqs:f12"] = qoi["clock_freqs:f12"]
            context["t1_time"] = qoi["t1_time"]
        if self.scout_bias:
            context["initial_parking_currents"] = {clabels[i]: None for i in range(n_c)}
        else:
            context["initial_parking_currents"] = {clabels[i]: guesses[i] for i in range(n_c)}

        if self.discovery_settings:
            from .one_qubit_discovery import OneQubitDiscovery
            from dataclasses import replace as _replace
            truth_chain = _replace(self.chain, qubits=seen)
            self.discovery = OneQubitDiscovery(truth_chain, self._rng, self.discovery_settings)
            context["discovery"] = self.discovery.visible()
        info = {"family": self.chain.family}
        if self.debug:
            info["chain"] = self.chain
        return context, info

    def _action_errors(self, act, i: int) -> list[str]:
        """Every reason this parsed action would be rejected, for coupler ``i``.

        This is the whole verdict, not just ``validate``: the measurement
        profile's schedule grid and NCO limits, and the spectroscopy controls,
        are part of whether an action can be measured at all.
        """
        errors = validate(act, self.configs[i])
        if not errors:
            from chipcal.sim.measurement_profile import spectroscopy_controls_errors
            errors.extend(spectroscopy_controls_errors(act, self.spectroscopy))
            if self.measurement_profile is not None:
                errors.extend(self.measurement_profile.validate(act,
                    default_spectroscopy_duration=self.spectroscopy.pulse_duration if self.spectroscopy is not None else 4e-6))
        return errors

    def action_errors(self, action) -> list[str]:
        """Why this action would be rejected, without measuring it.

        An empty list means the env would accept it. Prefer this over
        ``validate``, which does not know about the measurement profile.
        """
        if not isinstance(action, dict):
            return ["action must be a dict"]
        if action.get("stop"):
            return [] if set(action) == {"stop"} and action["stop"] is True else ["stop must be the only action"]
        from .one_qubit_discovery import is_action
        if is_action(action):
            if self.discovery is None:
                return ["discovery is not enabled"]
            from copy import deepcopy
            try:
                deepcopy(self.discovery).execute(action)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                return [str(exc)]
            return []
        if self.discovery is not None and not self.discovery.ready:
            return ["complete one-qubit discovery before coupler actions"]
        i = _coupler_index(action.get("coupler"), self.n_couplers)
        if i is None:
            return [f"action needs a valid 'coupler' label in {self.coupler_labels()}"]
        try:
            act = parse_action({k: v for k, v in action.items() if k != "coupler"})
        except Exception as exc:
            return [f"parse error: {exc}"]
        return self._action_errors(act, i)

    # -------------------------------------------------------------------- step
    def step(self, action) -> StepResult:
        if self._done:
            raise RuntimeError("episode is over; call reset()")
        self._enforce_benchmark_provenance()
        if not isinstance(action, dict):
            return self._result(None, cost=1.0, info={"invalid": ["action must be a dict"]})

        # stop needs no coupler
        if action.get("stop"):
            if set(action) != {"stop"} or action["stop"] is not True:
                return self._result(None, 1., {"invalid": ["stop must be the only action"]})
            return self._terminate()

        from .one_qubit_discovery import is_action
        if is_action(action):
            if self.discovery is None:
                return self._result(None, 1., {"invalid": ["discovery is not enabled"]})
            # Execute on a copy so budget exhaustion cannot apply a partial transaction.
            from copy import deepcopy
            candidate = deepcopy(self.discovery)
            try:
                ds, info, cost = candidate.execute(action)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                return self._result(None, 1., {"invalid": [str(exc)]})
            if self._would_exhaust(cost):
                return self._timeout(self.tau)
            self.discovery = candidate
            if info.get("frequency_changed"):
                for j in range(self.n_couplers):
                    self._clear_benchmark(j)
                info["benchmarks_invalidated"] = self.coupler_labels()
            info["discovery"] = self.discovery.visible()
            return self._result(ds, cost, info)
        if self.discovery is not None and not self.discovery.ready:
            return self._result(None, 1., {"invalid": ["complete resonator, f01 and f12 discovery for all qubits before coupler calibration"]})

        i = _coupler_index(action.get("coupler"), self.n_couplers)
        if i is None:
            return self._result(None, cost=1.0, info={
                "invalid": [f"action needs a valid 'coupler' label in {self.coupler_labels()}"]})

        try:
            act = parse_action({k: v for k, v in action.items() if k != "coupler"})
        except Exception as exc:
            return self._result(None, cost=1.0, info={"invalid": [f"parse error: {exc}"]})

        coerce_notes: list[str] = []
        if self.guardrails:
            # A wrong SI prefix is the policy's error to fix: reject with the unit, never guess it.
            slips = unit_slips(act)
            if slips:
                return self._result(None, cost=1.0, info={"invalid": slips, "coupler": self.coupler_labels()[i]})
            profile = self.measurement_profile
            grid = (profile.timing_grid_s
                    if profile is not None and profile.enforce_timing_grid else None)
            act, coerce_notes = coerce_action(act, self.configs[i], timing_grid_s=grid)

        errors = self._action_errors(act, i)
        if errors:
            if self.guardrails:
                errors = augment_prereq_hints(errors)
            return self._result(None, cost=1.0, info={"invalid": errors, "coupler": self.coupler_labels()[i]})

        if isinstance(act, StopAction):
            return self._terminate()
        if isinstance(act, CommitAction):
            res = self._commit(i, act)
        elif isinstance(act, BenchmarkAction):
            res = self._benchmark(i)
        else:
            res = self._measure(i, act)
        # Surface any range-clamping so the policy sees the value it actually executed.
        if coerce_notes and getattr(res, "info", None) is not None:
            res.info.setdefault("coerced", []).extend(coerce_notes)
        return res

    # ----------------------------------------------------------------- actions
    def _commit(self, i: int, act: CommitAction) -> StepResult:
        clabel = self.coupler_labels()[i]
        cost = self.cost_model.commit_seconds
        invalidated: list[str] = []
        value = act.value
        clamped = False
        if act.param == "parking_current":
            old_currents = self._currents()
            old_bias = self.configs[i].parking_current
            if old_bias != value:
                self.configs[i] = self.configs[i].commit(act.param, value)
                cost += self.cost_model.dc_settle
                invalidated = self._invalidate_on_flux_change(old_currents, self._currents())
                if self._realism is not None and self._realism.retune_on_move:
                    # Re-centre both adjacent qubits' frames (a Ramsey each) after every move.
                    cost += self._realism.retune_seconds
                    retuned = self._calibrate_frames(self.chain.coupler_qubits(i))
        else:
            had_benchmark = self._benchmarks[i] is not None
            if act.param == "cz_pulse_duration" and self.duration_limits is not None:
                lo, hi = self.duration_limits
                v = float(np.clip(value, lo, hi))
                clamped = v != value
                value = v
            self.configs[i] = self.configs[i].commit(act.param, value)
            # a gate-param commit only invalidates this coupler's standing benchmark
            if had_benchmark:
                self._clear_benchmark(i)
                invalidated = [clabel]
        info = {"committed": {act.param: value}, "coupler": clabel}
        if act.param == "parking_current" and self._realism is not None and self._realism.retune_on_move \
                and old_bias != value:
            info["qubits_retuned"] = sorted(retuned)
        if clamped:
            info["duration_clamped_to"] = self.duration_limits
        if invalidated:
            info["benchmarks_invalidated"] = invalidated
        return self._result(None, cost=cost, info=info)

    def _calibrate_frames(self, qubits) -> dict:
        """Re-measure the calibrated frame of ``qubits`` (a Ramsey): their true dressed f01 now, plus the
        calibration's error. Returns {qubit label: frame Hz}."""
        true = chainmod.qubit_frequencies(self.chain, self._currents(), self.wall_clock)
        labels = self.qubit_labels()
        out = {}
        for q in qubits:
            self._frames[q] = float(true[q] + self._realism_rng.normal(0.0, self._realism.calibration_sigma_hz))
            out[labels[q]] = self._frames[q]
        return out

    def _theta(self, i: int, currents=None):
        """Coupler ``i``'s effective device at the present wall time and calibrated frames."""
        currents = self._currents() if currents is None else currents
        return chainmod.coupler_theta(self.chain, i, currents, time_s=self.wall_clock, frames=self._frames)

    def _invalidate_on_flux_change(self, old_currents, new_currents) -> list[str]:
        """Stale every standing benchmark coupled to an actually changed bias line.

        Benchmark validity is about whether the physical configuration changed, not
        whether one incremental flux move crossed an arbitrary threshold.  In
        particular, retaining a benchmark across multiple small moves creates an
        accumulation loophole.  Exact same-current recommits remain no-ops.
        """
        changed_lines = [
            i for i, (old, new) in enumerate(zip(old_currents, new_currents))
            if old != new
        ]
        if not changed_lines:
            return []

        labels = self.coupler_labels()
        invalidated = []
        for j in range(self.n_couplers):
            affected = any(
                float(self.chain.crosstalk[j, i]) != 0.0
                for i in changed_lines
            )
            if self._benchmarks[j] is not None and affected:
                self._clear_benchmark(j)
                invalidated.append(labels[j])
        return invalidated

    def _benchmark(self, i: int) -> StepResult:
        if not self.configs[i].is_complete():
            missing = [k for k in ("parking_current", "cz_pulse_frequency", "cz_pulse_amplitude", "cz_pulse_duration")
                       if getattr(self.configs[i], k) is None]
            return self._result(None, 1.0, {"invalid": [f"benchmark needs a complete gate; commit {', '.join(missing)} first"],
                                            "coupler": self.coupler_labels()[i]})
        cost = self.cost_model.benchmark_seconds
        if self._would_exhaust(cost):
            return self._timeout(cost)
        theta = self._theta(i)
        if self.backend == "multimode":
            from chipcal.sim.multimode_rb import run_irb
            if self._realism is not None and self._realism.rb_lengths:
                result = run_irb(theta, self.configs[i], self._rng,
                                 oracle.IRBProtocol(lengths=tuple(self._realism.rb_lengths)))
            else:
                result = run_irb(theta, self.configs[i], self._rng)
        else:
            raise ValueError("Only multimode is supported")
        self._benchmarks[i] = result
        self._benchmark_provenance[i] = self._benchmark_fingerprint(i)
        reliable = bool(result.dataset.attrs.get("fit_reliable", True))
        info = {
            "coupler": self.coupler_labels()[i],
            # A fit of curves that collapsed to the 1/4 floor gives a meaningless F_hat: report none.
            "F_hat": result.f_hat if reliable else None, "F_sigma": result.f_sigma if reliable else None,
            "F_lcb": result.f_lcb, "fit_reliable": reliable, "reference_decay": round(float(result.p_ref), 4),
            "passed": result.f_lcb >= self.f_star,
            "chip_status": self._chip_status(),
        }
        return self._result(result.dataset, cost=cost, info=info)

    def _measure(self, i: int, act: MeasureAction) -> StepResult:
        cost = self._measurement_cost(act)
        if self.measurement_profile is not None and act.node in ("coupler_dc_spectroscopy", "qubit_spectroscopy"):
            cost = self.measurement_profile.spectroscopy_cost(act, self.configs[i].parking_current,
                        self.spectroscopy.pulse_duration if self.spectroscopy is not None else 4e-6)
        if self._would_exhaust(cost):
            return self._timeout(cost)
        theta = self._theta(i)
        labels = chainmod.coupler_qubit_labels(i, self.chain, self.qubit_labels())
        ds = nodes.run_node(theta, self.configs[i], act, self._rng, labels,
                            backend=self.backend, noise=self.noise, spectroscopy=self.spectroscopy)
        return self._result(ds, cost=cost, info={"node": act.node, "coupler": self.coupler_labels()[i]})

    # ---------------------------------------------------------------- helpers
    def _measurement_cost(self, act: MeasureAction) -> float:
        n_points, n_bias = 1, 0
        for ax, raw in act.sweeps.items():
            if ax == "working_points":
                n_points *= len(list(raw))
            elif ax == "dc_currents":
                vals = axis_values(raw)
                n_bias = vals.size
                n_points *= vals.size
            elif ax in ("cz_pulse_duration", "cz_pulse_amplitude", "ramsey_phases", "local_phases"):
                continue
            else:
                n_points *= axis_values(raw).size
        if act.node == "cz_calibration":
            n_points *= 2 * axis_values(act.sweeps.get("ramsey_phases", (210.0, 420.0, 30))).size
        if act.node == "cz_local_phases":
            n_points *= 4 * axis_values(act.sweeps.get("local_phases", (180.0, 360.0, 45))).size
        return self.cost_model.measurement_cost(n_points, act.shots, n_bias)

    def _would_exhaust(self, cost: float) -> bool:
        return cost > self.tau

    def _spend(self, cost: float) -> float:
        spent = min(cost, self.tau)
        self.tau -= spent
        self.total_time += spent
        return spent

    def _result(self, ds, cost: float, info: dict) -> StepResult:
        spent = self._spend(cost)
        if getattr(self, "_realism", None) is not None:
            # Drift runs on wall time: the instrument's and the agent's.
            self.wall_clock += spent + self._realism.turn_latency_s
        if self.tau <= 0:
            return self._timeout(spent, base_info=info)
        return StepResult(dataset=ds, cost_seconds=spent, reward=0.0,
                          terminated=False, truncated=False, info=info)

    # ------------------------------------------------------------- evaluation
    def _valid_lcbs(self) -> list[Optional[float]]:
        self._enforce_benchmark_provenance()
        return [b.f_lcb if b is not None else None for b in self._benchmarks]

    def _chip_status(self) -> dict:
        labels = self.coupler_labels()
        lcbs = self._valid_lcbs()
        return {
            "benchmarked": [labels[j] for j, v in enumerate(lcbs) if v is not None],
            "passing": [labels[j] for j, v in enumerate(lcbs) if v is not None and v >= self.f_star],
            "all_pass": all(v is not None and v >= self.f_star for v in lcbs),
        }

    def _terminal_reward(self) -> tuple[float, dict]:
        lcbs = self._valid_lcbs()
        time_penalty = self.lambda_per_minute * self.total_time / 60.0
        # missing/stale benchmark -> that gate counts as failing (0).
        eff = [0.0 if v is None else float(v) for v in lcbs]
        min_lcb = min(eff) if eff else 0.0
        all_pass = all(v is not None and v >= self.f_star for v in lcbs)
        true_fs = None
        if self.reward_mode == "final_rb":
            from chipcal.scoring import episode_score
            final = self.final_rb_fidelities()
            r = episode_score(f for f, _ in final)
            true_fs = self.chain_fidelities()
        elif self.reward_mode == "true_fidelity":
            from chipcal.scoring import episode_score
            true_fs = self.chain_fidelities()
            r = episode_score(true_fs)
        elif self.reward_mode == "binary":
            r = float(all_pass) * (1.0 + 0.5 * max(0.0, 1.0 - self.total_time / self.budget_seconds))
        else:
            r = oracle.rho(min_lcb) - time_penalty
        info = {
            "F_lcb_per_coupler": {self.coupler_labels()[j]: lcbs[j] for j in range(self.n_couplers)},
            "min_F_lcb": min_lcb,
            "all_pass": all_pass,
            "total_time": self.total_time,
            "time_penalty_nines": time_penalty,
        }
        if self.final_rb is not None:
            info["final_rb_fidelity_per_coupler"] = {self.coupler_labels()[j]: self.final_rb[j][0]
                                                     for j in range(self.n_couplers)}
        if self.debug or true_fs is not None:
            true_fs = true_fs if true_fs is not None else self.chain_fidelities()
            info["true_fidelity_per_coupler"] = {self.coupler_labels()[j]: true_fs[j]
                                                 for j in range(self.n_couplers)}
            info["min_true_fidelity"] = min(true_fs) if true_fs else 0.0
        return r, info

    def _terminate(self) -> StepResult:
        self._done = True
        r, info = self._terminal_reward()
        return StepResult(dataset=None, cost_seconds=0.0, reward=r,
                          terminated=True, truncated=False, info=info)

    def _timeout(self, cost: float, base_info: Optional[dict] = None) -> StepResult:
        self._spend(min(cost, self.tau))
        self._done = True
        r, info = self._terminal_reward()
        info["timeout"] = True
        if base_info:
            info.update(base_info)
        return StepResult(dataset=None, cost_seconds=cost, reward=r,
                          terminated=False, truncated=True, info=info)

    def final_rb_fidelities(self) -> list[tuple[float, dict]]:
        """The scoring benchmark (chipcal.scoring) of every coupler's final gate, run once and cached. Its random
        stream depends only on the episode seed, so a given final configuration always scores the same."""
        if self.final_rb is None:
            from chipcal.scoring import final_rb_fidelity
            self.final_rb = [final_rb_fidelity(self._theta(i), self.configs[i],
                                               np.random.default_rng([self._episode_seed, 991, i]))
                             for i in range(self.n_couplers)]
        return self.final_rb

    def chain_fidelities(self) -> list[float]:
        """Privileged true per-coupler fidelities at the current joint bias."""
        return chainmod.chain_gate_fidelities(self.chain, self._currents(), self.configs,
                                              backend=self.backend, time_s=getattr(self, "wall_clock", 0.0),
                                              frames=getattr(self, "_frames", None))

    def privileged_min_fidelity(self) -> float:
        """Privileged weakest-gate true fidelity (reward-path shaping potential)."""
        fs = self.chain_fidelities()
        return float(min(fs)) if fs else 0.0

    def ground_truth(self, *, reference_currents=None) -> dict:
        """Oracle-only diagnostics for evaluation (never shown to a policy)."""
        true_fs = self.chain_fidelities()
        labels = self.coupler_labels()
        return {
            "true_fidelity_per_coupler": {labels[i]: true_fs[i] for i in range(self.n_couplers)},
            "min_true_fidelity": min(true_fs) if true_fs else 0.0,
            "family": self.chain.family,
        }
