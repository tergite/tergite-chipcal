"""Frozen benchmark splits: the exact chips of a benchmark version, stored explicitly.

Seeds name chips through ``chipcal.tiers.sample``, but a stored split does not depend on the
random-number generator staying bit-identical across numpy versions: each row carries the
chip's full hidden parameters, and evaluation loads the chip from the row.

A benchmark chip must be calibratable: the privileged oracle (``chipcal.baselines.oracle``,
``scripts/oracle_ceiling.py``) finds every coupler's best CZ over its pulse and its own bias, and
``build`` keeps the first test seeds whose every coupler reaches ``MIN_CEILING``. Each row stores
those per-coupler ceilings, so a score can be read against what its chip allows. ``build`` also
writes the seeds and ceilings into the package (``chipcal/benchmark/``), for code without data/.

    python -m chipcal.splits build --out data/v0.1 --n 100 --ceilings ceilings/*.jsonl
    python -m chipcal.splits verify data/v0.1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from chipcal import tiers
from chipcal.sim.device_manifest import decode, encode

VERSION = "v0.1"
# Every coupler of a benchmark chip admits at least this CZ fidelity, a margin above f_star = 0.95 for RB
# fit noise and finite calibration precision.
MIN_CEILING = 0.96
CEILING_FIELDS = ("fidelity", "leakage", "bias", "flux", "cz_pulse_frequency", "cz_pulse_amplitude",
                  "cz_pulse_duration", "control_local_phase", "target_local_phase")
INDEX = Path(__file__).parent / "benchmark" / f"{VERSION}.json"


def row(tier: str, seed: int) -> dict:
    device = tiers.sample(tier, seed)
    return {"version": VERSION, "generator": tiers.GENERATOR_VERSION, "tier": tier, "seed": int(seed),
            "split": tiers.split_of(seed), "device": encode(device)}


def _line(value: dict) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def read_ceilings(paths) -> dict:
    """{(tier, seed): {coupler: oracle row}} from ``scripts/oracle_ceiling.py`` output (best-bias search)."""
    out: dict = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r.get("search") == "best_bias":
                out.setdefault((r["tier"], r["seed"]), {})[r["coupler"]] = r
    return out


def build(out: Path, n: int, ceiling_paths, split: str = "test", index_path: Path = INDEX) -> dict:
    """The first ``n`` seeds of ``split`` per tier whose every coupler ceiling reaches ``MIN_CEILING``. Seeds
    are scanned in order; a seed without a ceiling for every coupler ends the scan (compute more first)."""
    out.mkdir(parents=True, exist_ok=True)
    lo, _ = tiers.SPLITS[split]
    ceilings = read_ceilings(ceiling_paths)
    manifest = {"version": VERSION, "generator": tiers.GENERATOR_VERSION, "split": split, "min_ceiling": MIN_CEILING,
                "selection": "first seeds whose every coupler admits a CZ fidelity >= min_ceiling (privileged "
                             "best-bias oracle, chipcal.baselines.oracle.coupler_best_ceiling)", "files": {}}
    index = {"version": VERSION, "min_ceiling": MIN_CEILING, "tiers": {}}
    for tier in tiers.TIERS:
        n_c = tiers.TIERS[tier].n_couplers
        rows, seed = [], lo
        while len(rows) < n:
            found = ceilings.get((tier, seed), {})
            if len(found) < n_c:
                raise ValueError(f"{tier}: {len(rows)} of {n} chips selected; seed {seed} has no ceiling for every "
                                 f"coupler; compute ceilings beyond it")
            couplers = [{k: found[c][k] for k in CEILING_FIELDS} for c in range(n_c)]
            if min(c["fidelity"] for c in couplers) >= MIN_CEILING:
                rows.append({**row(tier, seed), "ceiling": couplers})
            seed += 1
        text = "\n".join(_line(r) for r in rows) + "\n"
        path = out / f"{tier}-{split}.jsonl"
        path.write_text(text)
        manifest["files"][path.name] = {"tier": tier, "chips": n, "first_seed": lo, "seeds_scanned": seed - lo,
                                        "sha256": hashlib.sha256(text.encode()).hexdigest()}
        index["tiers"][tier] = [{"seed": r["seed"], "ceiling": [c["fidelity"] for c in r["ceiling"]]} for r in rows]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    index_path.parent.mkdir(exist_ok=True)
    index_path.write_text(json.dumps(index, indent=1) + "\n")
    return manifest


def benchmark(tier: str) -> list[dict]:
    """The frozen benchmark chips of ``tier``: [{"seed", "ceiling": per-coupler fidelity}], in order."""
    return json.loads(INDEX.read_text())["tiers"][tier]


def ceiling(tier: str, seed: int):
    """Per-coupler ceilings of a frozen benchmark chip, or None for any other chip."""
    if tier not in tiers.TIERS or not INDEX.exists():
        return None
    return next((c["ceiling"] for c in benchmark(tier) if c["seed"] == int(seed)), None)


def normalized_score(final_per_coupler: dict, ceilings) -> float | None:
    """min over couplers of F_i / ceiling_i: each final gate against the best its coupler admits."""
    if not ceilings or not final_per_coupler:
        return None
    return min(f / c for f, c in zip((final_per_coupler[k] for k in sorted(final_per_coupler, key=_coupler_index)),
                                     ceilings))


def _coupler_index(label: str) -> int:
    return int(label.lstrip("c"))


def load(path: Path):
    """[(tier, seed, ChainDevice)] of one split file."""
    out = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        out.append((r["tier"], r["seed"], decode(r["device"])))
    return out


def _same(a, b, rtol: float = 1e-12) -> bool:
    """Equal JSON values, with floats equal to ``rtol``: CPUs and maths libraries differ in the last bit."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], rtol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= rtol * max(abs(a), abs(b))
    return a == b


def verify(directory: Path, regenerate: int = 3) -> list[str]:
    """Checksums match, and the first ``regenerate`` chips of each file regenerate (to float rounding)."""
    problems = []
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        text = (directory / name).read_text()
        if hashlib.sha256(text.encode()).hexdigest() != info["sha256"]:
            problems.append(f"{name}: checksum mismatch")
        for line in text.splitlines()[:regenerate]:
            stored = json.loads(line)
            stored.pop("ceiling", None)
            if not _same(json.loads(_line(row(stored["tier"], stored["seed"]))), stored):
                problems.append(f"{name}: seed {stored['seed']} no longer regenerates")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--n", type=int, default=100)
    b.add_argument("--ceilings", type=Path, nargs="+", required=True, help="scripts/oracle_ceiling.py output")
    b.add_argument("--split", default="test", choices=sorted(tiers.SPLITS))
    v = sub.add_parser("verify")
    v.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.cmd == "build":
        print(json.dumps(build(args.out, args.n, args.ceilings, args.split), indent=2))
    else:
        problems = verify(args.directory)
        print("\n".join(problems) or "ok")
        raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
