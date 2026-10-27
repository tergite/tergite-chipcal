from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable
from chipcal.llm.parsing import _json_safe

class EpisodeAbort(RuntimeError):
    """Raised by a backend to end the episode now; ``step`` must never absorb it as an
    invalid action (a user interrupt, for instance)."""


@dataclass
class Completion:
    """A model turn. ``token_logprobs`` is what on-policy LLM RL needs; a training
    client must populate it."""

    text: str
    token_logprobs: Optional[list[float]] = None
    tokens: Optional[list[int]] = None
    raw: Optional[dict[str, Any]] = None


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic chat interface (any OpenAI-compatible server, local HF, ...).

    A *training* client (local HF policy) must return ``token_logprobs``; an *eval*
    client may leave them ``None``.  ``images`` is the optional VLM hook -- the text
    controller leaves it unused; structured count-derived summaries are the main
    policy surface, while images remain optional diagnostics."""

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        images: Optional[list[bytes]] = None,
        schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> Completion: ...


@dataclass
class Turn:
    index: int
    observation: dict[str, Any]
    action: Any
    valid: bool
    reward: float
    reward_components: dict[str, Any]
    tool_result: dict[str, Any]
    model_text: Optional[str] = None
    model_logprobs: Optional[list[float]] = None
    model_analysis: Optional[dict[str, Any]] = None
    # Exact observation supplied to the policy before ``action``.  ``observation`` is the
    # post-action observation for the next turn, retained for backwards compatibility.
    policy_observation: Optional[dict[str, Any]] = None
    # Transport telemetry only; never appended to the policy's observation.
    model_usage: Optional[dict[str, Any]] = None


@dataclass
class Transcript:
    turns: list[Turn] = field(default_factory=list)
    total_reward: float = 0.0
    terminal_reward: Optional[float] = None
    terminated: bool = False
    truncated: bool = False
    final_config: dict[str, Any] = field(default_factory=dict)
    score: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "turns": [t.__dict__ for t in self.turns],
            "total_reward": self.total_reward,
            "terminal_reward": self.terminal_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "final_config": self.final_config,
            "score": self.score,
        })


@dataclass
class StepOutcome:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class RewardConfig:
    lambda_per_minute: float = 0.01
    invalid_penalty: float = 0.05
    shaping: bool = True


REASON_INSTRUCTION = (
    "Before acting, write ONE short line of reasoning that cites the structured observation "
    "analysis -- name the specific candidate (frequency/duration/amplitude/bias) you choose "
    "from the ranked lists and why. Then emit the single JSON action inside a ```json fenced "
    "code block. Keep the reasoning to one sentence."
)
