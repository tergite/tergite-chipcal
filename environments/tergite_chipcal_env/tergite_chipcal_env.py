"""tergite-chipcal-env: calibrate a simulated superconducting-qubit chip (verifiers v1 environment).

The environment plays the user's side of a multi-turn conversation. Each user turn carries the chip's
observation (text, plus a plot of the latest measurement); the model replies with one JSON action;
the environment executes it in the simulator and replies with the next observation. The episode
ends when the model emits {"stop": true}, at the tier's turn cap, or when the simulated lab-time
budget runs out. The environment then benchmarks every coupler's final gate with interleaved RB; the reward
is the minimum, over all couplers, of those fitted fidelities (chipcal.scoring).

    prime env install tergite/tergite-chipcal-env
    vf-eval tergite-chipcal-env -m <model> --env.taskset.tier 2q1c --env.taskset.n-chips 30
    vf-eval tergite-chipcal-env -m <model> --env.taskset.tier 8q-hard --env.taskset.split train    (procedural chip)

Actions are JSON in text: there is no tool calling. Guided decoding (JSON schema) is not applied on
this path; a reply that does not parse costs one turn and is shown back as an invalid action.
"""
from __future__ import annotations

import asyncio
import base64
from typing import Any

import verifiers.v1 as vf

from chipcal import splits, tiers
from chipcal.envs.chain_agent_env import default_chain_window_system_prompt, expand_action
from chipcal.llm.parsing import extract_json_action, render_observation_image


class ChipcalTasksetConfig(vf.TasksetConfig):
    tier: str = "2q1c"
    """2q1c, 3q2c or 5q4c, or a procedural tier "<n>q-<difficulty>" such as 8q-hard (chipcal.tiers.get)."""
    split: str = "test"
    """train, validation or test. The test split of a named tier is the frozen benchmark: chips whose every
    coupler admits a CZ fidelity >= 0.96 (chipcal.splits)."""
    n_chips: int = 100
    first_seed: int | None = None
    """First chip seed of a contiguous range; defaults to the frozen benchmark, or the start of the split."""
    visual: bool = True
    """Attach a plot of each measurement (needs a vision-language model)."""
    max_turns: int | None = None
    """Override the tier's turn cap."""


class ChipcalData(vf.TaskData):
    tier: str
    seed: int
    visual: bool = True
    max_turns: int | None = None


class ChipcalState(vf.State):
    score: float = 0.0
    normalized_score: float = float("nan")
    min_true_fidelity: float = 0.0
    success: bool = False
    discovery_fraction: float = 0.0
    certified_min_f_lcb: float = 0.0
    n_turns: int = 0
    n_invalid: int = 0
    n_unparseable: int = 0
    lab_time_s: float = 0.0


class ChipcalTask(vf.Task[ChipcalData, ChipcalState, vf.TaskConfig]):
    @vf.reward(weight=1.0)
    async def chip_score(self, trace: vf.Trace) -> float:
        """Min over couplers of the final interleaved-RB fidelity (chipcal.scoring)."""
        return float(trace.state.score)

    @vf.metric
    async def normalized_score(self, trace: vf.Trace) -> float:
        """Min over couplers of final RB fidelity / that coupler's ceiling (frozen benchmark chips; NaN otherwise)."""
        return float(trace.state.normalized_score)

    @vf.metric
    async def min_true_fidelity(self, trace: vf.Trace) -> float:
        return float(trace.state.min_true_fidelity)

    @vf.metric
    async def success(self, trace: vf.Trace) -> float:
        return float(trace.state.success)

    @vf.metric
    async def discovery_fraction(self, trace: vf.Trace) -> float:
        return float(trace.state.discovery_fraction)

    @vf.metric
    async def invalid_action_rate(self, trace: vf.Trace) -> float:
        return trace.state.n_invalid / max(trace.state.n_turns, 1)

    @vf.metric
    async def parse_failure_rate(self, trace: vf.Trace) -> float:
        return trace.state.n_unparseable / max(trace.state.n_turns, 1)

    @vf.metric
    async def lab_time_h(self, trace: vf.Trace) -> float:
        return trace.state.lab_time_s / 3600.0


class ChipcalTaskset(vf.Taskset[ChipcalTask, ChipcalTasksetConfig]):
    def load(self) -> list[ChipcalTask]:
        tier = tiers.get(self.config.tier).name
        if self.config.split == "test" and self.config.first_seed is None and tier in tiers.TIERS:
            seeds = [chip["seed"] for chip in splits.benchmark(tier)][:self.config.n_chips]
        else:
            lo, hi = tiers.SPLITS[self.config.split]
            first = lo if self.config.first_seed is None else int(self.config.first_seed)
            if not (lo <= first and first + self.config.n_chips - 1 <= hi):
                raise ValueError("chip seeds fall outside the split")
            seeds = range(first, first + self.config.n_chips)
        prompt = default_chain_window_system_prompt(discovery=True)
        env = tiers.make_env(tier)
        system = prompt + "\n\nAction grammar:\n" + _json(env.tool_schema())
        return [ChipcalTask(ChipcalData(id=f"{tier}-{seed}", name=f"{tier} chip {seed}",
                                        prompt=None, system_prompt=system, tier=tier, seed=seed,
                                        visual=self.config.visual, max_turns=self.config.max_turns))
                for seed in seeds]


def _json(value) -> str:
    import json
    return json.dumps(value)


def _turn_message(observation, *, first: bool, visual: bool, workdir) -> list[dict[str, Any]]:
    """The user turn for one observation: the full observation first, compact updates after
    (the conversation itself carries the history)."""
    text = (observation.to_prompt_text(visual_measurement=visual) if first
            else observation.to_turn_text(visual_measurement=visual))
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if visual and observation.measurement_dataset is not None:
        path = render_observation_image(observation.measurement_dataset, workdir, f"turn_{observation.turn:03d}")
        if path is not None:
            data = base64.b64encode(path.read_bytes()).decode()
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
    return [{"role": "user", "content": parts}]


class ChipcalEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        import tempfile
        from pathlib import Path
        data = task.data
        env = tiers.make_env(data.tier, **({"max_turns": data.max_turns} if data.max_turns else {}))
        chain = tiers.sample(data.tier, data.seed).chain
        observation = await asyncio.to_thread(env.reset, seed=int(data.seed), chain=chain)
        unparseable = 0
        with tempfile.TemporaryDirectory() as workdir:
            async with agents.agent.interaction(task) as interaction:
                first = True
                while not env._terminated:
                    segment = await interaction.turn(_turn_message(observation, first=first, visual=data.visual,
                                                                   workdir=Path(workdir)))
                    first = False
                    if segment.terminated:
                        break
                    try:
                        batch = expand_action(extract_json_action(segment.last_reply))
                    except Exception:
                        unparseable += 1
                        batch = [{"__unparseable__": True}]
                    for action in batch:
                        outcome = await asyncio.to_thread(env.step, action, model_text=segment.last_reply)
                        observation = outcome.observation
                        if outcome.terminated or outcome.truncated:
                            break
                if not env._terminated:
                    await asyncio.to_thread(env.step, {"stop": True})
                score = await asyncio.to_thread(env.score)
                state = interaction.trace.state
                state.score = float(score["score"])
                normalized = splits.normalized_score(score["final_rb_fidelity_per_coupler"],
                                                     splits.ceiling(data.tier, data.seed))
                state.normalized_score = float("nan") if normalized is None else float(normalized)
                state.min_true_fidelity = float(score["min_true_fidelity"])
                state.success = bool(score["success"])
                state.discovery_fraction = float(score["discovery_fraction"] or 0.0)
                state.certified_min_f_lcb = float(score["achieved_min_f_lcb"])
                state.n_turns = int(score["n_turns"])
                state.n_invalid = int(score["n_invalid_actions"])
                state.n_unparseable = unparseable
                state.lab_time_s = float(score["lab_time_s"])


__all__ = ["ChipcalEnv", "ChipcalTaskset"]
