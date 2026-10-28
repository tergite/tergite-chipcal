"""The public leaderboard: build, validate and verify submissions, and render LEADERBOARD.md.

    python scripts/leaderboard.py new --from results/my-model --name "My model" --kind model --model org/model \\
        --track raw --submitter "My lab" --out leaderboard/submissions/2026-11-01-my-model
    python scripts/leaderboard.py validate
    python scripts/leaderboard.py verify leaderboard/submissions/2026-11-01-my-model --chips 2
    python scripts/leaderboard.py render            # --check: fail if LEADERBOARD.md is out of date

A submission is a folder in leaderboard/submissions/ holding meta.json, episodes.jsonl (the rows of
scripts/evaluate.py) and, for a model, actions.jsonl (every episode's actions, taken from its transcripts). Every
number on the board is recomputed here from the rows. The simulator is deterministic given a chip and an action
sequence, so verify replays the actions of a few chips (a model) or reruns the policy on them (code) and checks the
recorded scores. See leaderboard/README.md for the rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import bootstrap_ci, run_one  # noqa: E402

from chipcal import splits, tiers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "leaderboard" / "submissions"
BOARD = ROOT / "LEADERBOARD.md"
KINDS = ("model", "code")
TRACKS = {"raw": "task description only", "guided": "with a disclosed strategy prompt",
          "baseline": "code baseline, no model"}
REQUIRED = ("name", "kind", "track", "tier", "benchmark", "chipcal_version", "date", "submitter")
SCORE_TOLERANCE = 1e-6


def load(folder: Path) -> tuple[dict, list[dict], dict]:
    meta = json.loads((folder / "meta.json").read_text())
    rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines() if line.strip()]
    actions = {}
    if (folder / "actions.jsonl").exists():
        for line in (folder / "actions.jsonl").read_text().splitlines():
            if line.strip():
                a = json.loads(line)
                actions[(a["seed"], a["sample"])] = a["actions"]
    return meta, rows, actions


def validate(folder: Path) -> list[str]:
    """Problems with one submission (empty when it is valid)."""
    problems = []
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9.-]*", folder.name):
        problems.append("folder name must be YYYY-MM-DD-<lowercase-slug>")
    try:
        meta, rows, actions = load(folder)
    except (OSError, ValueError, KeyError) as error:
        return problems + [f"unreadable submission: {error}"]
    problems += [f"meta.json: missing {k}" for k in REQUIRED if not meta.get(k)]
    if meta.get("kind") not in KINDS:
        problems.append(f"meta.json: kind must be one of {KINDS}")
    if meta.get("track") not in TRACKS:
        problems.append(f"meta.json: track must be one of {list(TRACKS)}")
    if meta.get("kind") == "model" and not meta.get("model"):
        problems.append("meta.json: a model entry names its model")
    if meta.get("kind") == "code" and not meta.get("policy"):
        problems.append("meta.json: a code entry names its scripts/evaluate.py --policy")
    if meta.get("track") == "guided" and not meta.get("prompt_addendum_sha256"):
        problems.append("meta.json: the guided track discloses prompt_addendum_sha256")
    if meta.get("benchmark") != splits.VERSION:
        problems.append(f"meta.json: benchmark must be {splits.VERSION}")
    tier = meta.get("tier")
    if tier not in tiers.TIERS:
        return problems + [f"meta.json: tier must be one of {sorted(tiers.TIERS)}"]
    frozen = {c["seed"] for c in splits.benchmark(tier)}
    keys = [(r.get("seed"), r.get("sample")) for r in rows]
    if not rows:
        problems.append("episodes.jsonl is empty")
    if len(set(keys)) != len(keys):
        problems.append("episodes.jsonl repeats an episode (seed, sample)")
    for r in rows:
        if r.get("tier") != tier or r.get("seed") not in frozen:
            problems.append(f"episode {r.get('tier')} {r.get('seed')} is not a frozen {tier} chip")
        if any(k not in r for k in ("score", "success", "n_turns", "n_invalid_actions")):
            problems.append(f"episode {r.get('seed')} lacks score, success, n_turns or n_invalid_actions")
    if meta.get("kind") == "model":
        problems += [f"actions.jsonl: no actions for episode {k}" for k in keys if k not in actions]
    return problems


def summary(meta: dict, rows: list[dict]) -> dict:
    """The board's numbers, recomputed from the episode rows."""
    tier = meta["tier"]
    per_chip: dict[int, list[float]] = {}
    for r in rows:
        per_chip.setdefault(r["seed"], []).append(r["score"])
    mean, lo, hi = bootstrap_ci([np.mean(v) for v in per_chip.values()])
    normalized = []
    for r in rows:
        ceilings = splits.ceiling(tier, r["seed"])
        finals = r.get("final_rb_fidelity_per_coupler") or ({"c0": r["score"]} if len(ceilings) == 1 else None)
        if finals is not None:
            normalized.append(splits.normalized_score(finals, ceilings))
    turns = sum(r["n_turns"] or 0 for r in rows) or 1
    return {"chips": len(per_chip), "samples": round(len(rows) / len(per_chip)), "mean": mean, "lo": lo, "hi": hi,
            "normalized": float(np.mean(normalized)) if normalized else None,
            "calibrated": sum(bool(r["success"]) for r in rows), "episodes": len(rows),
            "invalid": sum(r["n_invalid_actions"] or 0 for r in rows) / turns,
            "parse": sum(r.get("parse_failures", 0) for r in rows) / turns,
            "complete": set(per_chip) == {c["seed"] for c in splits.benchmark(tier)}}


def replay(tier: str, seed: int, actions: list, discovery: bool = True) -> float:
    """The score of an episode replayed from its actions, without its model."""
    env = tiers.make_env(tier, discovery=discovery)
    env.reset(seed=int(seed), chain=tiers.sample(tier, seed).chain)
    for action in actions:
        if env._terminated or env.step(action).terminated:
            break
    if not env._terminated:
        env.step({"stop": True})
    return float(env.score()["score"])


def verify(folder: Path, chips: int) -> list[str]:
    """Recompute the scores of ``chips`` episodes, chosen from the folder name, and compare them with the rows."""
    meta, rows, actions = load(folder)
    rng = random.Random(int(hashlib.sha256(folder.name.encode()).hexdigest(), 16))
    problems = []
    for r in rng.sample(rows, min(chips, len(rows))):
        if meta["kind"] == "model":
            got = replay(meta["tier"], r["seed"], actions[(r["seed"], r["sample"])], meta.get("discovery", True))
        else:
            got = run_one(dict(tier=meta["tier"], seed=r["seed"], chain=tiers.sample(meta["tier"], r["seed"]).chain,
                               sample=r["sample"], policy=meta["policy"]))["score"]
        ok = abs(got - r["score"]) <= SCORE_TOLERANCE
        print(f"{folder.name}: chip {r['seed']} sample {r['sample']}: recorded {r['score']:.6f}, "
              f"recomputed {got:.6f} {'ok' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            problems.append(f"chip {r['seed']}: recorded {r['score']} but the episode scores {got}")
    return problems


def render() -> str:
    entries = []
    for folder in sorted(p for p in SUBMISSIONS.iterdir() if p.is_dir()):
        meta, rows, _ = load(folder)
        entries.append((meta, summary(meta, rows), folder.relative_to(ROOT).as_posix()))
    lines = ["# Leaderboard", "",
             "Generated by `python scripts/leaderboard.py render` from [leaderboard/submissions](leaderboard/submissions);",
             "do not edit by hand. To add an entry, see [leaderboard/README.md](leaderboard/README.md). Every number is",
             "recomputed from the submitted episodes, and a model entry's actions can be replayed to verify its scores.", ""]
    for tier in sorted({m["tier"] for m, _, _ in entries}, key=list(tiers.TIERS).index):
        chips = splits.benchmark(tier)
        ceilings = [min(c["ceiling"]) for c in chips]
        header = ["| # | Entry | Track | Chips × samples | Score: min final-RB F, mean (95 % CI) | Score / ceiling "
                  "| Calibrated (F ≥ 0.95) | Invalid / turn | Parse fail / turn | Evidence | Date |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]

        def row(rank, meta, s, path):
            name = f"[{meta['name']}]({path})"
            evidence = "actions, replayable" if meta["kind"] == "model" else f"code: `{meta['policy']}`"
            parse = f"{s['parse']:.3f}" if meta["kind"] == "model" else "–"
            norm = f"{s['normalized']:.3f}" if s["normalized"] is not None else "–"
            return (f"| {rank} | {name} | {meta['track']} | {s['chips']} × {s['samples']} | {s['mean']:.3f} "
                    f"({s['lo']:.3f}–{s['hi']:.3f}) | {norm} | {s['calibrated']} / {s['episodes']} "
                    f"| {s['invalid']:.3f} | {parse} | {evidence} | {meta['date']} |")

        tier_entries = [(m, s, p) for m, s, p in entries if m["tier"] == tier]
        complete = sorted((e for e in tier_entries if e[1]["complete"]), key=lambda e: -e[1]["mean"])
        partial = sorted((e for e in tier_entries if not e[1]["complete"]), key=lambda e: -e[1]["mean"])
        lines += [f"## {tier} ({len(chips)} frozen chips, benchmark {splits.VERSION})", "", *header]
        lines += [row(i + 1, *e) for i, e in enumerate(complete)]
        lines += [f"| – | *Physics ceiling (privileged oracle)* | – | {len(chips)} | *median {np.median(ceilings):.3f} "
                  f"({min(ceilings):.3f}–{max(ceilings):.3f})* | *1* | *{len(chips)} / {len(chips)}* | – | – | – | – |", ""]
        if partial:
            lines += ["Partial entries (not every frozen chip; not ranked):", "", *header]
            lines += [row("–", *e) for e in partial]
            lines += [""]
    lines += ["Tracks: " + "; ".join(f"**{k}**: {v}" for k, v in TRACKS.items()) + ".", ""]
    return "\n".join(lines)


def new(args) -> None:
    """A submission folder from a scripts/evaluate.py results directory."""
    src, out = Path(args.source), Path(args.out)
    rows = [r for f in sorted(src.glob("*.jsonl")) for r in map(json.loads, filter(str.strip, f.read_text().splitlines()))
            if "score" in r]
    if not rows:
        raise SystemExit(f"no episode rows in {src}/*.jsonl")
    tier = rows[0]["tier"]
    out.mkdir(parents=True, exist_ok=False)
    meta = {"name": args.name, "kind": args.kind, "track": args.track, "tier": tier, "benchmark": splits.VERSION,
            "chipcal_version": args.chipcal_version, "date": out.name[:10], "submitter": args.submitter,
            **({"model": args.model} if args.kind == "model" else {"policy": rows[0]["policy"]}),
            **({"prompt_addendum_sha256": args.prompt_sha256} if args.prompt_sha256 else {}),
            "settings": json.loads(args.settings), "notes": args.notes, "link": args.link}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: (r["seed"], r["sample"]))))
    if args.kind == "model":
        lines = []
        for r in sorted(rows, key=lambda r: (r["seed"], r["sample"])):
            transcript = src / "episodes" / f"{r['tier']}-{r['seed']}-{r['sample']}" / "transcript.json"
            actions = [turn["action"] for turn in json.loads(transcript.read_text())["turns"]]
            lines.append(json.dumps({"seed": r["seed"], "sample": r["sample"], "actions": actions}) + "\n")
        (out / "actions.jsonl").write_text("".join(lines))
    print(f"wrote {out}; check meta.json, then run validate, verify and render")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("--from", dest="source", required=True, help="a scripts/evaluate.py --out directory")
    n.add_argument("--out", required=True, help="leaderboard/submissions/YYYY-MM-DD-<slug>")
    n.add_argument("--name", required=True)
    n.add_argument("--kind", choices=KINDS, required=True)
    n.add_argument("--model")
    n.add_argument("--track", choices=list(TRACKS), required=True)
    n.add_argument("--submitter", required=True)
    n.add_argument("--chipcal-version", default="0.1.0")
    n.add_argument("--prompt-sha256", help="sha256 of the --prompt-addendum file (guided track)")
    n.add_argument("--settings", default="{}", help='JSON, e.g. {"temperature": 0.7, "thinking_tokens": 4096}')
    n.add_argument("--notes", default="")
    n.add_argument("--link", default=None)
    sub.add_parser("validate")
    v = sub.add_parser("verify")
    v.add_argument("folders", nargs="+", type=Path)
    v.add_argument("--chips", type=int, default=2)
    r = sub.add_parser("render")
    r.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.cmd == "new":
        new(args)
    elif args.cmd == "validate":
        problems = {f.name: validate(f) for f in sorted(SUBMISSIONS.iterdir()) if f.is_dir()}
        for name, found in problems.items():
            print(f"{name}: " + ("ok" if not found else "\n  " + "\n  ".join(found)))
        raise SystemExit(1 if any(problems.values()) else 0)
    elif args.cmd == "verify":
        problems = [p for folder in args.folders for p in verify(folder, args.chips)]
        raise SystemExit(1 if problems else 0)
    else:
        text = render() + "\n"
        if args.check:
            if not BOARD.exists() or BOARD.read_text() != text:
                raise SystemExit("LEADERBOARD.md is out of date: run python scripts/leaderboard.py render")
            print("LEADERBOARD.md is up to date")
        else:
            BOARD.write_text(text)
            print(f"wrote {BOARD}")


if __name__ == "__main__":
    main()
