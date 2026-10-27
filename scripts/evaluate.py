"""Evaluate a policy on the frozen benchmark chips and print the results table.

    python scripts/evaluate.py --policy scripted --tier 2q1c --n 30 --workers 16 --out results/scripted
    python scripts/evaluate.py --policy probe:stop_immediately --tier 3q2c --n 30 --out results/stop
    python scripts/evaluate.py --policy model --model Qwen/Qwen3.8-27B --base-url http://host:8000/v1 \\
        --tier 2q1c --n 30 --samples 3 --workers 8 --out results/qwen
    python scripts/evaluate.py --policy scripted --tier 8q-hard --data chips --n 10 --out results/8q   (generated chips)
    python scripts/evaluate.py --summarize results/scripted results/qwen

Each episode is one JSON line (tier, seed, sample, score, ...). The summary reports the mean score with
a 95 % bootstrap interval over chips, the mean score / ceiling (each final gate against the best its coupler
admits, for the frozen benchmark chips), the number of chips calibrated, and grammar statistics kept apart
from skill.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from chipcal import splits, tiers

FIELDS = ("score", "min_true_fidelity", "success", "discovery_fraction", "achieved_min_f_lcb", "n_turns",
          "n_invalid_actions", "lab_time_s")


def _chips(tier: str, n: int, data: Path | None):
    """The first n chips of the frozen split (<tier>-test.jsonl) or of generated chips (<tier>.jsonl) in data."""
    for name in (f"{tier}-test.jsonl", f"{tier}.jsonl"):
        if data is not None and (data / name).exists():
            return [(t, s, d.chain) for t, s, d in splits.load(data / name)[:n]]
    lo = tiers.SPLITS["test"][0]
    return [(tier, seed, tiers.sample(tier, seed).chain) for seed in range(lo, lo + n)]


def run_one(job: dict) -> dict:
    tier, seed, chain, sample = job["tier"], job["seed"], job["chain"], job["sample"]
    t0 = time.time()
    policy = job["policy"]
    extra = {}
    if policy == "scripted":
        from chipcal.baselines.scripted import ScriptedChainAgent
        env = tiers.make_env(tier)
        observation = env.reset(seed=int(seed), chain=chain)
        score = ScriptedChainAgent(env).run(observation)
    elif policy.startswith("probe:"):
        from chipcal.baselines.probes import PROBES
        score = PROBES[policy.split(":", 1)[1]](tier, seed)
    elif policy == "model":
        import tempfile
        from chipcal.run import openai_client, run
        client = openai_client(job["model"], job["base_url"], api_key=job["api_key"], thinking_tokens=job["thinking_tokens"])
        out = Path(job["out"]) / "episodes" / f"{tier}-{seed}-{sample}"
        transcript = run(tier, seed, out, client=client, visual=not job["text_only"], temperature=job["temperature"],
                         save_measurements=False, discovery=job.get("discovery", True),
                         prompt_addendum=job.get("prompt_addendum"))
        score = transcript.score
        unparseable = sum(1 for t in transcript.turns if isinstance(t.action, dict) and t.action.get("__unparseable__"))
        extra["parse_failures"] = unparseable
    else:
        raise ValueError(f"unknown policy {policy!r}")
    row = {"tier": tier, "seed": int(seed), "sample": int(sample), "policy": policy, "wall_s": round(time.time() - t0, 1)}
    row.update({k: score.get(k) for k in FIELDS})
    row["final_rb_fidelity_per_coupler"] = score.get("final_rb_fidelity_per_coupler")
    row["normalized_score"] = splits.normalized_score(score.get("final_rb_fidelity_per_coupler"),
                                                      splits.ceiling(tier, seed))
    row.update(extra)
    return row


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, values.size)).mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(paths) -> str:
    rows = []
    for p in paths:
        for f in sorted(Path(p).glob("*.jsonl")):
            rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    lines = ["| policy | tier | chips × samples | score (95 % CI) | score / ceiling | calibrated | min true F | discovery "
             "| invalid / turn | parse fail / turn |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted({(r["policy"], r["tier"]) for r in rows}):
        group = [r for r in rows if (r["policy"], r["tier"]) == key]
        # Average samples per chip first, then bootstrap over chips.
        per_chip = {}
        for r in group:
            per_chip.setdefault(r["seed"], []).append(r["score"])
        mean, lo, hi = bootstrap_ci([np.mean(v) for v in per_chip.values()])
        turns = sum(r["n_turns"] or 0 for r in group) or 1
        parse = sum(r.get("parse_failures", 0) for r in group)
        normalized = [r["normalized_score"] for r in group if r.get("normalized_score") is not None]
        lines.append(f"| {key[0]} | {key[1]} | {len(per_chip)} × {round(len(group) / len(per_chip))} | {mean:.3f} ({lo:.3f}–{hi:.3f}) "
                     f"| {np.mean(normalized) if normalized else float('nan'):.3f} "
                     f"| {sum(bool(r['success']) for r in group)} / {len(group)} | {np.mean([r['min_true_fidelity'] for r in group]):.3f} "
                     f"| {np.mean([r['discovery_fraction'] or 0 for r in group]):.2f} | {sum(r['n_invalid_actions'] or 0 for r in group) / turns:.3f} "
                     f"| {parse / turns:.3f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default="scripted")
    parser.add_argument("--tier", type=tiers.tier_name, action="append")
    parser.add_argument("--n", type=int, default=30, help="chips per tier (the first n frozen test chips)")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard", default="0/1", help="i/k: run every k-th episode starting at i (one --out per shard)")
    parser.add_argument("--data", type=Path, default=Path("data/v0.1"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--thinking-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--no-discovery", action="store_true", help="start from disclosed single-qubit QOIs")
    parser.add_argument("--prompt-addendum", type=Path, help="text file appended to the system prompt (ablations)")
    parser.add_argument("--summarize", nargs="+", type=Path, help="print the table for these result directories")
    args = parser.parse_args()
    if args.summarize:
        print(summarize(args.summarize))
        return
    if args.out is None:
        parser.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for tier in args.tier or sorted(tiers.TIERS):
        for t, seed, chain in _chips(tier, args.n, args.data):
            for sample in range(args.samples):
                jobs.append(dict(tier=t, seed=seed, chain=chain, sample=sample, policy=args.policy, model=args.model,
                                 base_url=args.base_url, api_key=args.api_key, thinking_tokens=args.thinking_tokens,
                                 temperature=args.temperature, text_only=args.text_only, out=str(args.out),
                                 discovery=not args.no_discovery,
                                 prompt_addendum=args.prompt_addendum.read_text() if args.prompt_addendum else None))
    i, k = map(int, args.shard.split("/"))
    jobs = jobs[i::k]
    name = args.policy.replace(":", "-").replace("/", "-")
    sink_path = args.out / f"{name}.jsonl"
    # Re-running into the same --out (a chained job after a time limit) skips recorded episodes; interrupted
    # model episodes resume from their checkpoints.
    done = set()
    if sink_path.exists():
        done = {(r["tier"], r["seed"], r["sample"]) for r in map(json.loads, sink_path.read_text().splitlines()) if r}
    jobs = [j for j in jobs if (j["tier"], j["seed"], j["sample"]) not in done]
    with open(sink_path, "a") as sink, ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                # One broken episode must not take down the others; it is not recorded, so the next
                # (chained) run retries it from its checkpoint.
                print(json.dumps({"episode_error": f"{type(exc).__name__}: {exc}"}), flush=True)
                continue
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(json.dumps(row), flush=True)
    print(summarize([args.out]))


if __name__ == "__main__":
    main()
