"""Export what a model sees, turn by turn, without running a model: the observation text and the plot.

    python scripts/export_observations.py --tier 2q1c --seed 3000000 --out runs/seen
    python scripts/export_observations.py --tier 8q-hard --seed 1000042 --max-turns 40 --out runs/seen-8q
    python scripts/export_observations.py --replay runs/demo/transcript.json --tier 2q1c --seed 3000000 --out runs/replay

The actions come from the scripted calibration routine, or from a recorded transcript with --replay. The
observations go through the same policy wrapper as `chipcal.run`, so the files are exactly the model's
input: system_prompt.txt once; then, for every turn, turn_NNN.txt (the user message) and the PNG attached to it
under episode_000/. index.jsonl lists turn, text file, image and the action taken. --text-only exports the
text-only variant, where a text analysis replaces the plot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chipcal import tiers
from chipcal.envs.agent_env import Completion
from chipcal.envs.chain_agent_env import chain_llm_policy
from chipcal.run import system_prompt


class Recorder:
    """An LLM client stand-in that writes each request to disk and answers with a fixed action."""

    def __init__(self, out: Path):
        self.out = out
        self.turn = 0

    def generate(self, messages, *, images=None, schema=None, temperature=0.0, **_):
        if not (self.out / "system_prompt.txt").exists():
            (self.out / "system_prompt.txt").write_text(messages[0]["content"])
        (self.out / f"turn_{self.turn:03d}.txt").write_text(messages[1]["content"])
        return Completion(text='{"stop": true}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", type=tiers.tier_name, default="2q1c")
    parser.add_argument("--seed", type=int, default=tiers.SPLITS["test"][0])
    parser.add_argument("--replay", type=Path, help="transcript.json whose actions to replay")
    parser.add_argument("--max-turns", type=int, help="stop exporting after this many turns")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = tiers.make_env(args.tier)
    observation = env.reset(seed=args.seed, chain=tiers.sample(args.tier, args.seed).chain)
    recorder = Recorder(args.out)
    policy = chain_llm_policy(recorder, system_prompt=system_prompt(), tool_schema=env.tool_schema(),
                              visual_observations=not args.text_only, vision_dir=str(args.out))
    index = open(args.out / "index.jsonl", "w")
    step = env.step

    class Done(BaseException):          # not an Exception, so the scripted routine cannot swallow it
        pass

    def recording_step(action, **kwargs):
        """Show the current observation to the recorder, then take the action."""
        nonlocal observation
        if args.max_turns is not None and recorder.turn >= args.max_turns:
            raise Done
        policy(observation)
        image = policy.last_plot_path and str(Path(policy.last_plot_path).relative_to(args.out))
        index.write(json.dumps({"turn": recorder.turn, "text": f"turn_{recorder.turn:03d}.txt", "image": image,
                                "action": action}) + "\n")
        recorder.turn += 1
        outcome = step(action, **kwargs)
        observation = outcome.observation
        return outcome

    env.step = recording_step
    try:
        if args.replay:
            for turn in json.loads(args.replay.read_text())["turns"]:
                if env.step(turn["action"]).terminated:
                    break
        else:
            from chipcal.baselines.scripted import ScriptedChainAgent
            ScriptedChainAgent(env).run(observation)
    except Done:
        pass
    index.close()
    print(f"{recorder.turn} turns exported to {args.out}")


if __name__ == "__main__":
    main()
