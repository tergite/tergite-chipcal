"""The public leaderboard: build, validate and verify submissions, and render LEADERBOARD.md.

    python scripts/leaderboard.py new --from results/my-model --name "My model" --kind model --model org/model \\
        --track raw --submitter "My lab" --out leaderboard/submissions/2026-11-01-my-model
    python scripts/leaderboard.py validate
    python scripts/leaderboard.py verify leaderboard/submissions/2026-11-01-my-model --chips 2
    python scripts/leaderboard.py render            # LEADERBOARD.md + docs/leaderboard-*.svg; --check: fail if stale

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


def load_entries() -> list[tuple[dict, dict, str]]:
    """Every submission as (meta, summary, path relative to the repository)."""
    entries = []
    for folder in sorted(p for p in SUBMISSIONS.iterdir() if p.is_dir()):
        meta, rows, _ = load(folder)
        entries.append((meta, summary(meta, rows), folder.relative_to(ROOT).as_posix()))
    return entries


def render() -> str:
    entries = load_entries()
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
            if meta.get("caveat"):
                name += f" ({meta['caveat']})"
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


PLOT_TIERS = ("2q1c",)
THEMES = {
    # Transparent background, so the chart sits on GitHub's own light or dark page.
    "light": {"primary": "#1f2328", "secondary": "#59636e", "muted": "#818b98", "grid": "#e1e6eb",
              "model": "#2a78d6", "baseline": "#59636e", "ring": "#ffffff", "band": "#2a78d6", "band_opacity": 0.14},
    "dark": {"primary": "#f0f6fc", "secondary": "#9198a1", "muted": "#7d8590", "grid": "#2a3038",
             "model": "#3987e5", "baseline": "#9198a1", "ring": "#0d1117", "band": "#3987e5", "band_opacity": 0.28},
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"


def plot_path(tier: str, theme: str) -> Path:
    return ROOT / "docs" / f"leaderboard-{tier}-{theme}.svg"


def _text_width(text: str, size: float) -> float:
    """A conservative width estimate for the sans-serif stack (no font metrics in a static SVG)."""
    narrow, wide = set("ijlI.,:;'()[] |·1"), set("mwMW@%")
    return size * sum(0.3 if c in narrow else 0.85 if c in wide else 0.62 if c.isupper() else 0.55 for c in text)


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def plot(tier: str, entries: list[tuple[dict, dict, str]], theme: str) -> str:
    """The tier's board as a static SVG: every entry's mean score with its 95 % CI, beside the physics ceiling."""
    c = THEMES[theme]
    chips = splits.benchmark(tier)
    ceilings = [min(ch["ceiling"]) for ch in chips]
    tier_entries = sorted((e for e in entries if e[0]["tier"] == tier), key=lambda e: (-e[1]["mean"], e[0]["name"]))
    rows = [{"name": "Physics ceiling (privileged oracle)", "sub": f"{len(chips)} chips · best bias and pulse",
             "mean": float(np.median(ceilings)), "lo": min(ceilings), "hi": max(ceilings), "kind": "ceiling",
             "value": f"{np.median(ceilings):.3f}", "note": f"{len(chips)} / {len(chips)} calibratable"}]
    for meta, s, _ in tier_entries:
        scope = f"{s['chips']} chips" if s["complete"] else f"{s['chips']} of {len(chips)} chips, partial"
        rows.append({"name": meta["name"], "sub": f"{scope} · {meta['track']}", "mean": s["mean"], "lo": s["lo"],
                     "hi": s["hi"], "kind": meta["kind"], "value": f"{s['mean']:.3f}",
                     "note": f"{s['calibrated']} / {s['episodes']} calibrated", "caveat": meta.get("caveat")})

    pad, row_h, name_size, sub_size = 16, 44, 13, 11
    label_w = max(max(_text_width(r["name"], name_size), _text_width(r["sub"], sub_size),
                      _text_width(r.get("caveat") or "", sub_size)) for r in rows) + 24
    value_w = max(_text_width(r["note"], sub_size) for r in rows) + 28
    width = 880
    x0, x1 = pad + label_w, width - pad - value_w
    top = 96
    heights = [row_h + (14 if r.get("caveat") else 0) for r in rows]
    starts = [top + sum(heights[:i]) for i in range(len(rows))]
    height = top + sum(heights) + 52
    sx = lambda v: x0 + (x1 - x0) * v  # noqa: E731

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height:.0f}" '
           f'viewBox="0 0 {width} {height:.0f}" font-family="{FONT}" role="img" '
           f'aria-label="tergite-chipcal {tier} leaderboard: mean score with 95 % confidence interval per entry">']
    out.append(f'<text x="{pad}" y="26" font-size="15" font-weight="600" fill="{c["primary"]}">'
               f'tergite-chipcal leaderboard · {tier}</text>')
    out.append(f'<text x="{pad}" y="46" font-size="12" fill="{c["secondary"]}">Score: minimum final-RB CZ fidelity '
               f'over couplers, mean over the frozen chips with 95 % CI. Higher is better.</text>')
    # Legend: model vs code baseline vs ceiling, marks beside text-coloured labels.
    lx = pad
    for kind, label in (("model", "language model"), ("baseline", "code baseline"), ("ceiling", "physics ceiling")):
        cy = 70
        if kind == "ceiling":
            out.append(f'<rect x="{lx}" y="{cy - 5}" width="18" height="10" rx="2" fill="{c["band"]}" fill-opacity="{c["band_opacity"]}"/>')
            out.append(f'<line x1="{lx + 9}" y1="{cy - 5}" x2="{lx + 9}" y2="{cy + 5}" stroke="{c["band"]}" stroke-width="2"/>')
            lx += 24
        else:
            out.append(f'<circle cx="{lx + 5}" cy="{cy}" r="5" fill="{c[kind]}"/>')
            lx += 16
        out.append(f'<text x="{lx}" y="{cy + 4}" font-size="12" fill="{c["secondary"]}">{label}</text>')
        lx += _text_width(label, 12) + 22

    plot_top, plot_bottom = top, top + sum(heights)
    # Gridlines and reference lines span every row.
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{sx(v):.1f}" y1="{plot_top}" x2="{sx(v):.1f}" y2="{plot_bottom}" '
                   f'stroke="{c["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{sx(v):.1f}" y="{plot_bottom + 18}" font-size="11" text-anchor="middle" '
                   f'fill="{c["muted"]}">{v:g}</text>')
    out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{plot_bottom + 38}" font-size="11" text-anchor="middle" '
               f'fill="{c["secondary"]}">score (min final-RB fidelity)</text>')
    for v, label, anchor in ((0.25, "0.25: fully depolarising gate", "start"), (0.95, "calibrated: F ≥ 0.95", "end")):
        out.append(f'<line x1="{sx(v):.1f}" y1="{plot_top - 6}" x2="{sx(v):.1f}" y2="{plot_bottom}" '
                   f'stroke="{c["muted"]}" stroke-width="1"/>')
        dx = 4 if anchor == "start" else -4
        out.append(f'<text x="{sx(v) + dx:.1f}" y="{plot_top - 8}" font-size="11" text-anchor="{anchor}" '
                   f'fill="{c["secondary"]}">{label}</text>')

    for i, r in enumerate(rows):
        cy = starts[i] + row_h / 2
        if i:
            out.append(f'<line x1="{pad}" y1="{starts[i]}" x2="{width - pad}" y2="{starts[i]}" '
                       f'stroke="{c["grid"]}" stroke-width="1" stroke-opacity="0.6"/>')
        style = ' font-style="italic"' if r["kind"] == "ceiling" else ""
        out.append(f'<text x="{pad}" y="{cy - 3}" font-size="{name_size}"{style} fill="{c["primary"]}">{_esc(r["name"])}</text>')
        out.append(f'<text x="{pad}" y="{cy + 12}" font-size="{sub_size}" fill="{c["muted"]}">{_esc(r["sub"])}</text>')
        if r.get("caveat"):
            out.append(f'<text x="{pad}" y="{cy + 26}" font-size="{sub_size}" font-style="italic" '
                       f'fill="{c["secondary"]}">{_esc(r["caveat"])}</text>')
        if r["kind"] == "ceiling":
            out.append(f'<rect x="{sx(r["lo"]):.1f}" y="{cy - 9}" width="{max(sx(r["hi"]) - sx(r["lo"]), 2):.1f}" '
                       f'height="18" rx="3" fill="{c["band"]}" fill-opacity="{c["band_opacity"]}"/>')
            out.append(f'<line x1="{sx(r["mean"]):.1f}" y1="{cy - 9}" x2="{sx(r["mean"]):.1f}" y2="{cy + 9}" '
                       f'stroke="{c["band"]}" stroke-width="2" stroke-linecap="round"/>')
        else:
            color = c["model"] if r["kind"] == "model" else c["baseline"]
            if sx(r["hi"]) - sx(r["lo"]) > 1:
                out.append(f'<line x1="{sx(r["lo"]):.1f}" y1="{cy}" x2="{sx(r["hi"]):.1f}" y2="{cy}" stroke="{color}" '
                           f'stroke-width="2" stroke-linecap="round" stroke-opacity="0.55"/>')
            if r.get("caveat"):  # provisional: hollow, so it reads as not yet a settled score
                out.append(f'<circle cx="{sx(r["mean"]):.1f}" cy="{cy}" r="4.5" fill="{c["ring"]}" stroke="{color}" '
                           f'stroke-width="2"/>')
            else:
                out.append(f'<circle cx="{sx(r["mean"]):.1f}" cy="{cy}" r="5" fill="{color}" stroke="{c["ring"]}" '
                           f'stroke-width="2"/>')
        vx = width - pad
        out.append(f'<text x="{vx}" y="{cy - 3}" font-size="{name_size}" font-weight="600" text-anchor="end" '
                   f'style="font-variant-numeric: tabular-nums" fill="{c["primary"]}">{r["value"]}</text>')
        out.append(f'<text x="{vx}" y="{cy + 12}" font-size="{sub_size}" text-anchor="end" '
                   f'fill="{c["muted"]}">{_esc(r["note"])}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def outputs() -> dict[Path, str]:
    """Everything render writes: LEADERBOARD.md and the README plots."""
    entries = load_entries()
    files = {BOARD: render() + "\n"}
    for tier in PLOT_TIERS:
        if any(m["tier"] == tier for m, _, _ in entries):
            for theme in THEMES:
                files[plot_path(tier, theme)] = plot(tier, entries, theme)
    return files


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
        files = outputs()
        if args.check:
            stale = [p.relative_to(ROOT).as_posix() for p, text in files.items() if not p.exists() or p.read_text() != text]
            if stale:
                raise SystemExit(f"{', '.join(stale)} out of date: run python scripts/leaderboard.py render")
            print("LEADERBOARD.md and the plots are up to date")
        else:
            for path, text in files.items():
                path.write_text(text)
                print(f"wrote {path}")


if __name__ == "__main__":
    main()
