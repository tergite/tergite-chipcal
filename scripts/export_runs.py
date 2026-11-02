"""Build the Hugging Face dataset of evaluation runs (tergite/tergite-chipcal-runs) from leaderboard submissions.

Every model entry on the leaderboard links its full run here. To publish a new score:

    python scripts/export_runs.py add leaderboard/submissions/2026-11-01-my-model --results results/my-model
    python scripts/export_runs.py measure leaderboard/submissions/2026-11-01-my-model     # replay; slow, see below
    python scripts/export_runs.py card
    hf upload tergite/tergite-chipcal-runs build/hf-runs . --repo-type dataset

``add`` copies the submission files, every episode's transcript and run summary, the plots the model was shown
and, for runs that saved them (scripts/evaluate.py does by default), the simulated data of every measurement.
``measure`` is needed only for runs without saved measurements: it replays each episode from its actions (the
simulator is deterministic), saves the measurement data and checks that the replay reaches the recorded score. One
episode can take an hour; ``--index`` replays a single episode so a cluster array can run them in parallel.
``card`` writes README.md from the run folders, so the card always matches what is uploaded. ``check`` fails when
a model entry on the leaderboard has no run folder here.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from leaderboard import ROOT, SCORE_TOLERANCE, load, load_entries, rank_key, summary  # noqa: E402

from chipcal import tiers  # noqa: E402
from chipcal.run import measurement_node, save_measurement  # noqa: E402

OUT = ROOT / "build" / "hf-runs"
REPO = "tergite/tergite-chipcal-runs"
GITHUB = "https://github.com/tergite/tergite-chipcal"


def run_dir(meta: dict, submission: Path, out: Path) -> Path:
    return out / meta["tier"] / submission.name


def episode_id(row: dict) -> str:
    return f"{row['tier']}-{row['seed']}-{row['sample']}"


def add(submission: Path, results: Path, out: Path) -> None:
    """Copy a submission and its episodes (transcripts, run summaries, the plots the model saw) into the dataset."""
    meta, rows, _ = load(submission)
    if meta["kind"] != "model":
        raise SystemExit("only model entries have runs to export")
    dest = run_dir(meta, submission, out)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("meta.json", "episodes.jsonl", "actions.jsonl"):
        shutil.copy2(submission / name, dest / name)
    for row in rows:
        src, ep = results / "episodes" / episode_id(row), dest / "episodes" / episode_id(row)
        if not (src / "transcript.json").exists():
            raise SystemExit(f"missing transcript: {src / 'transcript.json'}")
        ep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "run.json", ep / "run.json")
        with open(src / "transcript.json", "rb") as f_in, gzip.open(ep / "transcript.json.gz", "wb", 9) as f_out:
            shutil.copyfileobj(f_in, f_out)
        # Only files of turns the final transcript records as measurements: a run that died between saving a
        # measurement and checkpointing redoes that turn on resume, and the model may then choose differently.
        turns = json.loads((src / "transcript.json").read_text())["turns"]
        measured = {(t["index"], measurement_node(t["action"])) for t in turns
                    if t["valid"] and measurement_node(t["action"])}
        turn_of = lambda name: int(name.split("_")[1])  # noqa: E731  turn_<n>_...
        for path in sorted((src / "measurements").glob("turn_*.json.gz")):  # saved during the run: no replay needed
            if (turn_of(path.name), path.name.split("_", 2)[2][:-len(".json.gz")]) in measured:
                (ep / "measurements").mkdir(exist_ok=True)
                shutil.copy2(path, ep / "measurements" / path.name)
        images = ep / "images"
        # A resumed episode keeps its later plots in a second folder; turn numbers are global, so they merge.
        for png in sorted((src / "vision").glob("episode_*/*.png")):
            if any(turn_of(png.name) == n and png.stem.endswith(node) for n, node in measured):
                images.mkdir(exist_ok=True)
                shutil.copy2(png, images / png.name)
    print(f"added {dest.relative_to(out)}: {len(rows)} episodes")


def measure(submission: Path, out: Path, index: int | None = None) -> list[str]:
    """Replay episodes, save every measurement's simulated data, and confirm the recorded scores."""
    meta, rows, actions = load(submission)
    dest = run_dir(meta, submission, out)
    rows = sorted(rows, key=lambda r: (r["seed"], r["sample"]))
    problems = []
    for row in rows if index is None else [rows[index]]:
        folder = dest / "episodes" / episode_id(row) / "measurements"
        folder.mkdir(parents=True, exist_ok=True)
        env = tiers.make_env(meta["tier"], discovery=meta.get("discovery", True))
        env.reset(seed=int(row["seed"]), chain=tiers.sample(meta["tier"], row["seed"]).chain)
        for action in actions[(row["seed"], row["sample"])]:
            if env._terminated:
                break
            outcome = env.step(action)
            turn, ds = env.transcript.turns[-1], outcome.observation.measurement_dataset
            if ds is not None and turn.valid and measurement_node(action):
                save_measurement(folder, turn.index, action, ds)
            if outcome.terminated:
                break
        if not env._terminated:
            env.step({"stop": True})
        got = float(env.score()["score"])
        ok = abs(got - row["score"]) <= SCORE_TOLERANCE
        print(f"{submission.name}: chip {row['seed']}: recorded {row['score']:.6f}, replayed {got:.6f} "
              f"{'ok' if ok else 'MISMATCH'}; {len(list(folder.glob('*.json.gz')))} measurements", flush=True)
        if not ok:
            problems.append(f"chip {row['seed']}: recorded {row['score']} but the replay scores {got}")
    return problems


def _fmt_settings(settings: dict) -> str:
    names = {"temperature": "temperature", "thinking_tokens": "thinking budget (tokens)", "visual": "plots shown",
             "model_revision": "model revision", "serving": "serving"}
    return "\n".join(f"| {names.get(k, k)} | `{v}` |" for k, v in settings.items())


def card(out: Path) -> str:
    """The dataset card, generated from the run folders present in ``out``."""
    runs = []
    for meta_path in sorted(out.glob("*/*/meta.json")):
        folder = meta_path.parent
        meta, rows, _ = load(folder)
        runs.append((meta, summary(meta, rows), rows, folder.relative_to(out).as_posix()))
    runs.sort(key=lambda r: (r[0]["tier"], *rank_key((r[0], r[1], r[3]))))
    n_episodes = sum(len(r[2]) for r in runs)
    n_images = len(list(out.glob("*/*/episodes/*/images/*.png")))
    n_measure = len(list(out.glob("*/*/episodes/*/measurements/*.json.gz")))

    lines = ["---", "license: apache-2.0", "pretty_name: tergite-chipcal evaluation runs", "task_categories:",
             "- reinforcement-learning", "tags:", "- llm-agents", "- quantum-computing", "- superconducting-qubits",
             "- calibration", "- benchmark", "- agent-trajectories", "size_categories:", "- n<1K", "configs:"]
    for meta, _, _, path in runs:
        lines += [f"- config_name: {path.split('/')[-1]}", "  data_files:", "  - split: episodes",
                  f"    path: {path}/episodes.jsonl"]
    lines += ["---", "", "# tergite-chipcal evaluation runs", "",
              f"**Every turn of every language-model run on the [tergite-chipcal]({GITHUB}) benchmark: what the model was "
              "shown, what it thought and did, the plots it saw, and the simulated data behind each measurement.**", "",
              f"[tergite-chipcal]({GITHUB}) is a multi-turn environment in which an agent calibrates a simulated "
              "superconducting-qubit chip from scratch. It must first find every qubit's readout resonator and "
              "transition frequencies, then calibrate a CZ gate on every coupler, using only lab measurements "
              "(spectroscopy, chevrons, Ramsey phase scans, randomized benchmarking) that cost simulated lab time. "
              "The score is the minimum over couplers of a final interleaved randomized-benchmarking fidelity of the "
              "gates the agent leaves committed; a chip whose every final gate reaches 0.95 counts as calibrated, "
              "and the leaderboard ranks entries by the fraction of chips they calibrate.", "",
              f"This dataset holds the full record behind each model entry on the [leaderboard]({GITHUB}/blob/main/LEADERBOARD.md): "
              f"{len(runs)} runs, {n_episodes} episodes, {n_images} plots and {n_measure} simulated measurements. "
              "The chips themselves are in [tergite/tergite-chipcal](https://huggingface.co/datasets/tergite/tergite-chipcal).", "",
              "> **Simulated data only.** No file describes a physical device. Physics model v0.1, not validated "
              "against hardware.", "", "## Runs", "",
              "| Run | Model | Track | Chips × samples | Calibrated (every final F ≥ 0.95), 95 % CI | Score: mean (95 % CI) | Invalid actions / turn | Note |",
              "|---|---|---|---|---|---|---|---|"]
    for meta, s, _, path in runs:
        rev = meta.get("settings", {}).get("model_revision")
        model = f"[{meta['model']}](https://huggingface.co/{meta['model']})" + (f" @ `{rev[:8]}`" if rev else "")
        lines.append(f"| [{meta['name']}]({path}) | {model} | {meta['track']} | {s['chips']} × {s['samples']} | "
                     f"**{s['calibrated']} / {s['episodes']}** ({100 * s['rate_lo']:.0f}–{100 * s['rate_hi']:.0f} %) | "
                     f"{s['mean']:.3f} ({s['lo']:.3f}–{s['hi']:.3f}) | "
                     f"{s['invalid']:.3f} | {meta.get('caveat') or ''} |")
    lines += ["", "Raw track: the model gets only the task description, no strategy prompt. Guided track: a strategy "
              "prompt is appended to the system prompt, identified by its sha256 in `meta.json`. Scores are recomputed "
              "from `episodes.jsonl` exactly as on the leaderboard.", ""]

    for meta, s, rows, path in runs:
        lines += [f"### {meta['name']}", "", f"`{path}` · {meta['date']} · tergite-chipcal {meta['chipcal_version']}"
                  + (f" · {meta['notes']}" if meta.get("notes") else ""), ""]
        if meta.get("caveat"):
            lines += [f"> **{meta['caveat'][0].upper() + meta['caveat'][1:]}.**", ""]
        if meta.get("settings"):
            lines += ["| Setting | Value |", "|---|---|", _fmt_settings(meta["settings"]), ""]
        lines += ["| Chip | Score | True min fidelity | Discovery | Turns | Invalid actions | Lab time (h) |",
                  "|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: (r["seed"], r["sample"])):
            lab = f"{r['lab_time_s'] / 3600:.2f}" if r.get("lab_time_s") is not None else "–"
            lines.append(f"| {r['seed']} | {r['score']:.3f} | {r.get('min_true_fidelity', 0):.3f} | "
                         f"{r.get('discovery_fraction', 0):.2f} | {r['n_turns']} | {r['n_invalid_actions']} | {lab} |")
        lines += ["", "*True min fidelity* is the simulator's exact gate fidelity, reported but not scored. "
                  "*Discovery* is the fraction of the chip's frequencies the agent found.", ""]

    lines += ["## Layout", "", "```",
              "<tier>/<run>/",
              "  meta.json            model, revision, settings, notes (leaderboard format)",
              "  episodes.jsonl       one row per episode: score, fidelities, discovery, turns, invalid actions",
              "  actions.jsonl        one row per episode: the full action sequence, replayable",
              "  episodes/<tier>-<seed>-<sample>/",
              "    run.json             episode summary and final score breakdown",
              "    transcript.json.gz   every turn (below)",
              "    images/turn_<n>_<node>.png             the plot the model was shown after each measurement",
              "    measurements/turn_<n>_<node>.json.gz   the simulated data of each measurement",
              "```", "",
              "**Transcript.** `{\"turns\": [...], \"score\": {...}, \"final_config\": [...]}`. Each turn holds:", "",
              "| Field | Content |", "|---|---|",
              "| `index` | turn number |",
              "| `policy_observation` | the structured state the model was shown: chip status, calibration ledger, notebook, history |",
              "| `model_text` | the model's raw output: its thinking in `<think>…</think>`, then the action JSON |",
              "| `action`, `valid` | the parsed action and whether the environment accepted it |",
              "| `tool_result` | what the environment returned: measurement analysis or an error |",
              "| `model_usage` | token counts of the thinking and the action pass |",
              "| `observation`, `reward`, `reward_components` | environment internals, kept for completeness |", "",
              "One model call may return a batch of actions; they run as consecutive turns, and only the first turn of "
              "a batch carries `model_text` and `model_usage`.", "",
              "**Measurements.** `{\"turn\", \"action\", \"dataset\"}`, where `dataset` is an "
              "[xarray](https://docs.xarray.dev) `Dataset.to_dict(data=\"list\")`: the swept axes as coordinates and the "
              "finite-shot outcomes (state populations, signal) as data variables, with the node's settings as "
              "attributes. Complex arrays are stored as `{\"real\", \"imag\"}`, NaN as `null`. They were regenerated by "
              "replaying the actions, which also re-confirmed every score.", "",
              "## Use", "", "```python", "import gzip, json", "from datasets import load_dataset", "",
              f"rows = load_dataset(\"{REPO}\", \"{runs[0][3].split('/')[-1] if runs else '<run>'}\", split=\"episodes\")   # one row per chip", "",
              "ep = \"<tier>/<run>/episodes/<tier>-<seed>-<sample>\"   # after downloading the repo",
              "transcript = json.load(gzip.open(f\"{ep}/transcript.json.gz\"))",
              "for turn in transcript[\"turns\"][:5]:",
              "    print(turn[\"index\"], turn[\"action\"], (turn[\"model_text\"] or \"\")[:200])", "",
              "import xarray as xr", "",
              "def to_xarray(d):",
              "    unpack = lambda v: [complex(r, i) for r, i in zip(v[\"real\"], v[\"imag\"])] if isinstance(v, dict) and \"real\" in v else v",
              "    for group in (\"coords\", \"data_vars\"):",
              "        for var in d.get(group, {}).values():",
              "            var[\"data\"] = unpack(var[\"data\"])",
              "    return xr.Dataset.from_dict(d)", "",
              "m = json.load(gzip.open(f\"{ep}/measurements/turn_003_resonator_spectroscopy.json.gz\"))",
              "print(m[\"action\"], to_xarray(m[\"dataset\"]))", "```", "",
              "## Verify a score", "",
              "The simulator is deterministic given a chip and an action sequence, so any score here can be recomputed:", "",
              "```bash", f"git clone {GITHUB} && cd tergite-chipcal && pip install -e .",
              "python scripts/leaderboard.py verify leaderboard/submissions/<run> --chips 10", "```", "",
              "A run replays exactly under the tergite-chipcal version in its `meta.json`; every run here has been "
              "replayed in full.", "",
              "## How runs are added", "",
              f"A model entry joins the leaderboard as a pull request carrying its actions ([how to submit]({GITHUB}/blob/main/leaderboard/README.md)). "
              "When it is merged, the maintainers add its full run here with `scripts/export_runs.py`, which also "
              "writes this card, so the leaderboard and this dataset always list the same model entries.", "",
              "## Limitations", "",
              "- Runs on fewer than all 100 frozen chips are partial: compare their confidence intervals, not only means.",
              "- Sampling settings (temperature, thinking budget) follow each model card and differ between runs; see each run's settings.",
              "- Physics model v0.1 is a simulator, not validated against hardware; a good score here does not show that a model can calibrate hardware.", "",
              "## Licenses", "",
              "The data is released under Apache-2.0. Transcripts contain outputs of third-party models, used under their "
              "licenses: Qwen3.8 and Gemma 4 (Apache-2.0) and NVIDIA Ising Calibration 1.5 (OpenMDW-1.1, which places "
              "no restrictions on outputs). Model and product names belong to their respective owners and identify the "
              "models evaluated; no affiliation or endorsement is implied.", "",
              "**Owner:** Chalmers Next Labs AB · **License:** Apache-2.0", "",
              "## Citation", "", "```bibtex", "@software{tergite_chipcal_2026,",
              "  title   = {tergite-chipcal: a simulated superconducting-qubit chip-calibration environment for LLM agents},",
              "  author  = {{Chalmers Next Labs AB}},", "  year    = {2026},", "  version = {0.2.0}", "}", "```", ""]
    return "\n".join(lines)


def check(out: Path) -> list[str]:
    """Model entries on the leaderboard without a run folder here, or whose link does not point at it."""
    problems = []
    for meta, _, path in load_entries():
        if meta["kind"] != "model":
            continue
        name = Path(path).name
        if not (out / meta["tier"] / name / "meta.json").exists():
            problems.append(f"{name}: no run folder in {out} (run export_runs.py add)")
        else:
            bare = [e.name for e in sorted((out / meta["tier"] / name / "episodes").glob("*"))
                    if not any((e / "measurements").glob("*.json.gz"))]
            if bare:
                problems.append(f"{name}: no measurements for {', '.join(bare)} (run export_runs.py measure, or "
                                "ignore if those episodes made none)")
        expected = f"https://huggingface.co/datasets/{REPO}/tree/main/{meta['tier']}/{name}"
        if meta.get("link") != expected:
            problems.append(f"{name}: meta.json link should be {expected}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("submission", type=Path)
    a.add_argument("--results", type=Path, required=True, help="the scripts/evaluate.py --out directory of the run")
    m = sub.add_parser("measure")
    m.add_argument("submission", type=Path)
    m.add_argument("--index", type=int, help="replay only this episode (rows sorted by seed, sample)")
    sub.add_parser("card")
    sub.add_parser("check")
    args = parser.parse_args()
    if args.cmd == "add":
        add(args.submission, args.results, args.out)
    elif args.cmd == "measure":
        raise SystemExit(1 if measure(args.submission, args.out, args.index) else 0)
    elif args.cmd == "card":
        (args.out / "README.md").write_text(card(args.out))
        print(f"wrote {args.out / 'README.md'}")
    else:
        problems = check(args.out)
        print("\n".join(problems) if problems else "every model entry has its run")
        raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
