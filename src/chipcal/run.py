"""Run one benchmark episode with an LLM behind an OpenAI-compatible server.

    python -m chipcal.run --tier 2q1c --seed 3000000 --out runs/demo \\
        --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.8-27B
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

from chipcal import tiers
from chipcal.envs.chain_agent_env import chain_llm_policy, default_chain_window_system_prompt, expand_action
from chipcal.viz.plotting import plot_dataset


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def system_prompt(discovery: bool = True, native_thinking: bool = True):
    """The system prompt (discovery protocol first when discovery is on)."""
    return default_chain_window_system_prompt(native_thinking=native_thinking, discovery=discovery)


def openai_client(model, base_url, *, api_key='EMPTY', thinking_tokens=4096, top_p=.8, timeout=1800,
                  reasoning_effort=None, two_pass=True):
    """An OpenAI-compatible client for a vLLM-served reasoning VLM: an unconstrained thinking pass, then a
    schema-constrained action pass."""
    from chipcal.llm.client import OpenAIClient
    return OpenAIClient(model=model, base_url=base_url, api_key=api_key, timeout=timeout, max_tokens=thinking_tokens,
                        top_p=top_p, top_k=20, chat_template_kwargs={'enable_thinking': True},
                        include_reasoning_in_text=True, reasoning_effort=reasoning_effort,
                        two_pass_reasoning=two_pass, reasoning_max_tokens=thinking_tokens, action_max_tokens=512)


def _checkpoint(env, path: Path) -> None:
    """Pickle the whole environment (state, RNG streams, transcript) atomically."""
    temporary = path.with_suffix('.tmp')
    with open(temporary, 'wb') as handle:
        pickle.dump(env, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def run(tier, seed, out, *, client, visual=True, temperature=.7, max_turns=None, save_measurements=True, resume=True,
        discovery=True, prompt_addendum=None):
    """One episode of ``tier`` on chip ``seed``; writes run.json, transcript.json and plots under ``out``.

    The environment is checkpointed after every model turn (``episode.pkl``). With ``resume`` an interrupted
    episode (a job that hit its time limit) continues from its last checkpoint, and a completed one is returned
    as recorded, so long episodes can span several chained jobs.

    ``discovery=False`` starts from disclosed single-qubit QOIs (no discovery stage); ``prompt_addendum`` is
    extra text appended to the system prompt (for ablations; the benchmark prompt has none)."""
    out = Path(out)
    tier = tiers.get(tier)
    checkpoint = out / 'episode.pkl'
    previous = json.loads((out / 'run.json').read_text()) if (out / 'run.json').exists() else None
    if previous is not None and not resume:
        raise FileExistsError(out)
    if previous is not None and previous.get('status') == 'complete' and checkpoint.exists():
        with open(checkpoint, 'rb') as handle:
            return pickle.load(handle).transcript
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(tier=tier.name, seed=int(seed), split=tiers.split_of(seed), generator=tiers.GENERATOR_VERSION,
                    visual=visual, temperature=temperature, started=time.time(), status='running',
                    resumed=bool(previous), segments=int((previous or {}).get('segments', 0)) + 1,
                    discovery=bool(discovery),
                    prompt_addendum_sha256=hashlib.sha256(prompt_addendum.encode()).hexdigest() if prompt_addendum else None)
    atomic_json(out / 'run.json', metadata)
    try:
        if previous is not None and checkpoint.exists():
            with open(checkpoint, 'rb') as handle:
                env = pickle.load(handle)
            observation = env._observe()
        else:
            device = tiers.sample(tier, seed)
            env = tiers.make_env(tier, discovery=bool(discovery), **({'max_turns': max_turns} if max_turns else {}))
            observation = env.reset(seed=int(seed), chain=device.chain)
        prompt = system_prompt(discovery=bool(discovery)) + (f'\n\n{prompt_addendum.strip()}\n' if prompt_addendum else '')
        policy = chain_llm_policy(client, system_prompt=prompt,
                                  tool_schema=env.tool_schema(), response_schema=env.tool_schema()['json_schema'],
                                  temperature=temperature, visual_observations=visual, vision_dir=str(out / 'vision'))
        finished = False
        while not finished:
            batch = expand_action(policy(observation))
            text, analysis, usage = policy.last_text, policy.last_analysis, policy.last_usage
            for position, action in enumerate(batch):
                # Only the first entry of a batch was generated; the rest carry no model text.
                first = position == 0
                outcome = env.step(action, model_text=text if first else None, model_analysis=analysis if first else None)
                if first:
                    env.transcript.turns[-1].model_usage = usage
                observation = outcome.observation
                turn = env.transcript.turns[-1]
                ds = observation.measurement_dataset
                if save_measurements and ds is not None and turn.valid and isinstance(action, dict) and 'node' in action:
                    from chipcal.sim.device_manifest import encode
                    atomic_json(out / f'measurement-{turn.index:03d}.json', encode(ds.to_dict(data='array')))
                    plot_dataset(ds, out / 'plots', stem=f'turn-{turn.index:03d}')
                print(json.dumps({'turn': turn.index, 'valid': turn.valid, 'action': action}), flush=True)
                if outcome.terminated or outcome.truncated:
                    finished = True
                    break
            _checkpoint(env, checkpoint)
        env.score()
        _checkpoint(env, checkpoint)
        atomic_json(out / 'transcript.json', env.transcript.to_dict())
        metadata.update(status='complete', finished=time.time(), score=env.transcript.score)
        atomic_json(out / 'run.json', metadata)
        return env.transcript
    except BaseException as error:
        metadata.update(status='failed', finished=time.time(), error=f'{type(error).__name__}: {error}')
        atomic_json(out / 'run.json', metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tier', type=tiers.tier_name, default='2q1c', help='2q1c, 3q2c, 5q4c, or procedural <n>q-<difficulty>')
    parser.add_argument('--seed', type=int, default=tiers.SPLITS['test'][0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model', default='Qwen/Qwen3.8-27B')
    parser.add_argument('--api-key', default='EMPTY')
    parser.add_argument('--thinking-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=.7)
    parser.add_argument('--top-p', type=float, default=.8)
    parser.add_argument('--max-turns', type=int, help='override the tier turn cap')
    parser.add_argument('--request-timeout', type=float, default=1800)
    parser.add_argument('--text-only', action='store_true', help='no plot images (text analysis instead)')
    args = parser.parse_args()
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.request_timeout <= 0 or args.thinking_tokens < 1:
        parser.error('invalid sampling or timeout settings')
    client = openai_client(args.model, args.base_url, api_key=args.api_key, thinking_tokens=args.thinking_tokens,
                           top_p=args.top_p, timeout=args.request_timeout)
    run(args.tier, args.seed, args.out, client=client, visual=not args.text_only, temperature=args.temperature,
        max_turns=args.max_turns)


if __name__ == '__main__':
    main()
