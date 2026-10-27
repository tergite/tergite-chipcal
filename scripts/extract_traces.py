"""Print candidate thinking-trace excerpts from model episodes, for the README.

    python scripts/extract_traces.py results/qwen-pilot-2q1c/episodes --kind commit_frequency --max 5

Each excerpt is one turn: the event the model had just seen, its reasoning (the native thinking
channel, trimmed), the action it emitted, and what the environment answered.
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

KINDS = {
    "configure_lo": lambda a: "configure_lo" in a,
    "commit_frequency": lambda a: "commit_frequency" in a,
    "measure": lambda a: "node" in a,
    "commit": lambda a: "commit" in a,
    "benchmark": lambda a: a.get("benchmark") is True,
    "stop": lambda a: a.get("stop") is True,
}


def reasoning(text: str | None) -> str:
    if not text:
        return ""
    match = re.search(r"<think>(.*?)</think>", text, re.S)
    body = match.group(1) if match else text.split("</think>")[0]
    return " ".join(body.split())


def excerpts(directory: Path, kind: str | None, invalid_only: bool):
    for transcript in sorted(directory.glob("*/transcript.json")):
        turns = json.loads(transcript.read_text())["turns"]
        for previous, turn in zip([None] + turns[:-1], turns):
            action = turn.get("action") or {}
            if not isinstance(action, dict) or not turn.get("model_text"):
                continue
            if kind and not KINDS[kind](action):
                continue
            if invalid_only and turn.get("valid", True):
                continue
            seen = (previous or {}).get("observation", {}).get("last_event") if previous else "episode start"
            yield {"episode": transcript.parent.name, "turn": turn["index"], "seen": seen,
                   "thinking": reasoning(turn["model_text"]), "action": action,
                   "result": turn["observation"].get("last_event"), "valid": turn.get("valid")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--kind", choices=sorted(KINDS))
    parser.add_argument("--invalid", action="store_true", help="only rejected actions")
    parser.add_argument("--max", type=int, default=5)
    parser.add_argument("--width", type=int, default=1200, help="characters of reasoning to keep")
    args = parser.parse_args()
    for n, ex in enumerate(excerpts(args.directory, args.kind, args.invalid)):
        if n == args.max:
            break
        thinking = ex["thinking"] if len(ex["thinking"]) <= args.width else ex["thinking"][: args.width] + " [...]"
        print(f"### {ex['episode']} turn {ex['turn']}\n")
        print(f"**Saw:** {ex['seen']}\n")
        print("**Thinking:**\n\n" + textwrap.indent(textwrap.fill(thinking, 110), "> ") + "\n")
        print(f"**Action:** `{json.dumps(ex['action'])}`\n")
        print(f"**Result:** {ex['result']}\n")


if __name__ == "__main__":
    main()
