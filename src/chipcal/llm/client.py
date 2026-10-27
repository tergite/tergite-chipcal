from __future__ import annotations
import base64
from typing import Any, Optional
from chipcal.envs.agent_env import Completion

def _b64_images(images: Optional[list[bytes]]) -> list[str]:
    return [base64.b64encode(img).decode("ascii") for img in (images or [])]


def _attach_last_user(messages: list[dict[str, Any]]) -> int:
    """Index of the last ``user`` message (for attaching images), or -1."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def _usage_dict(resp: Any) -> dict[str, Any]:
    """Best-effort JSON-safe usage/metadata snapshot (never the raw SDK object)."""
    out: dict[str, Any] = {}
    usage = getattr(resp, "usage", None)
    if usage is not None:
        for k in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens",
                  "total_tokens", "cache_read_input_tokens"):
            v = getattr(usage, k, None)
            if v is not None:
                out[k] = int(v)
    for k in ("model", "stop_reason", "finish_reason"):
        v = getattr(resp, k, None)
        if isinstance(v, (str, int)):
            out[k] = v
    choices = getattr(resp, "choices", None)
    if choices:
        finish = getattr(choices[0], "finish_reason", None)
        if isinstance(finish, str):
            out["finish_reason"] = finish
    details = getattr(usage, "completion_tokens_details", None)
    count = getattr(details, "reasoning_tokens", None)
    if isinstance(count, int):
        out["reasoning_tokens"] = count
    return out


class OpenAIClient:
    """OpenAI Chat Completions eval client (optional token logprobs).

    ``want_logprobs=True`` populates ``Completion.token_logprobs`` from the response's
    per-token ``logprob`` values.  ``json_mode`` requests a JSON-object response (the system
    prompt already says "JSON"); a ``schema`` upgrades that to a strict json_schema format.
    ``two_pass_reasoning`` gives local reasoning models an unconstrained native-thinking call,
    then a separate thinking-disabled, schema-constrained action call.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 2048,
        want_logprobs: bool = False,
        top_logprobs: int = 0,
        top_p: float = 1.0,
        top_k: int = 0,
        json_mode: bool = True,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        include_reasoning_in_text: bool = False,
        reasoning_effort: Optional[str] = None,
        two_pass_reasoning: bool = False,
        reasoning_max_tokens: Optional[int] = None,
        action_max_tokens: int = 512,
        action_checklist: str = "",
        client: Any = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.max_tokens = int(max_tokens)
        self.want_logprobs = bool(want_logprobs)
        self.top_logprobs = int(top_logprobs)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.json_mode = bool(json_mode)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.include_reasoning_in_text = bool(include_reasoning_in_text)
        self.reasoning_effort = reasoning_effort
        self.two_pass_reasoning = bool(two_pass_reasoning)
        self.reasoning_max_tokens = int(reasoning_max_tokens or max_tokens)
        self.action_max_tokens = int(action_max_tokens)
        self.action_checklist = str(action_checklist)
        self._client = client
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = float(timeout)

    def _ensure_client(self) -> Any:
        if self._client is None:
            import openai  # lazy

            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def _to_messages(self, messages: list[dict[str, Any]], images: Optional[list[bytes]]) -> list[dict[str, Any]]:
        out = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
        b64 = _b64_images(images)
        if b64:
            idx = _attach_last_user(out)
            if idx >= 0:
                text = out[idx]["content"]
                content: list[dict[str, Any]] = []
                if isinstance(text, str) and text:
                    content.append({"type": "text", "text": text})
                content.extend(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{d}"}} for d in b64
                )
                out[idx]["content"] = content
        return out

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        images: Optional[list[bytes]] = None,
        schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> Completion:
        client = self._ensure_client()
        mapped_messages = self._to_messages(messages, images)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": mapped_messages,
            "temperature": float(temperature),
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        extra_body: dict[str, Any] = {}
        if self.top_k > 0:
            extra_body["top_k"] = self.top_k
        if self.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = self.chat_template_kwargs
        if extra_body:
            kwargs["extra_body"] = extra_body
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.want_logprobs:
            kwargs["logprobs"] = True
            if self.top_logprobs:
                kwargs["top_logprobs"] = self.top_logprobs
        response_format = None
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "action", "schema": schema, "strict": True},
            }
        elif self.json_mode:
            response_format = {"type": "json_object"}
        if self.two_pass_reasoning:
            reasoning_kwargs = dict(kwargs)
            reasoning_kwargs["max_tokens"] = self.reasoning_max_tokens
            # Pass 1 is unconstrained native reasoning. Applying the action grammar here would
            # once again compete with the opening <think> channel.
            reasoning_kwargs.pop("response_format", None)
            reasoning_resp = client.chat.completions.create(**reasoning_kwargs)
            reasoning_choice = reasoning_resp.choices[0]
            reasoning_message = reasoning_choice.message
            reasoning = (
                getattr(reasoning_message, "reasoning_content", None)
                or getattr(reasoning_message, "reasoning", None)
                or ""
            )
            proposal = getattr(reasoning_message, "content", None) or ""
            memo = reasoning or proposal
            if proposal and proposal.strip() != memo.strip():
                memo += "\n\nProposed final response from the reasoning pass:\n" + proposal
            action_instruction = (
                "\n\nThe native reasoning pass is complete. Its deliberation follows and may "
                "end abruptly at the token cap:\n\n<reasoning_memo>\n" + memo +
                "\n</reasoning_memo>\n\nNow serialize the single best next action. Re-check "
                "it against the original observation and action grammar. Return only the JSON "
                "object: no thinking, explanation, markdown, or wrapper."
            )
            if self.action_checklist:
                action_instruction += "\n\nAction validation checklist:\n" + self.action_checklist
            # Vision is paid for only in the reasoning pass. The serialization pass gets the
            # exact original text plus the completed memo and therefore cannot reinterpret the
            # plot or spend a second vision-prefill.
            action_messages = self._to_messages(messages, None)
            last_user = next(
                (index for index in range(len(action_messages) - 1, -1, -1)
                 if action_messages[index].get("role") == "user"),
                None,
            )
            if last_user is None or not isinstance(action_messages[last_user].get("content"), str):
                raise ValueError("two-pass OpenAI reasoning requires a final text user message")
            action_messages[last_user]["content"] += action_instruction
            action_kwargs = {
                "model": self.model,
                "messages": action_messages,
                "temperature": 0.0,
                "max_tokens": self.action_max_tokens,
                "top_p": 1.0,
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                        "preserve_thinking": False,
                    },
                },
            }
            if response_format is not None:
                action_kwargs["response_format"] = response_format
            action_resp = client.chat.completions.create(**action_kwargs)
            action_choice = action_resp.choices[0]
            action_text = getattr(action_choice.message, "content", None) or ""
            traced_text = f"<think>\n{memo}\n</think>\n\n{action_text}"
            return Completion(
                text=traced_text,
                token_logprobs=(
                    _openai_token_logprobs(action_choice) if self.want_logprobs else None
                ),
                tokens=None,
                raw={
                    "two_pass_reasoning": True,
                    "reasoning": _usage_dict(reasoning_resp),
                    "action": _usage_dict(action_resp),
                },
            )
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = getattr(choice.message, "content", None) or ""
        reasoning = (
            getattr(choice.message, "reasoning_content", None)
            or getattr(choice.message, "reasoning", None)
            or ""
        )
        if self.include_reasoning_in_text and reasoning:
            text = f"<think>\n{reasoning}\n</think>\n\n{text}"
        return Completion(
            text=text,
            token_logprobs=_openai_token_logprobs(choice) if self.want_logprobs else None,
            tokens=None,
            raw=_usage_dict(resp),
        )


def _openai_token_logprobs(choice: Any) -> Optional[list[float]]:
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    if not content:
        return None
    out: list[float] = []
    for tok in content:
        v = tok.get("logprob") if isinstance(tok, dict) else getattr(tok, "logprob", None)
        if v is not None:
            out.append(float(v))
    return out or None
