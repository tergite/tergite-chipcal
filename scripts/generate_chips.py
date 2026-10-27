"""Generate procedural chips: any size, any difficulty preset, optionally certified calibratable.

    python scripts/generate_chips.py --tier 8q-hard --n 50 --out chips
    python scripts/generate_chips.py --tier 12q-extreme --n 20 --split validation --out chips
    python scripts/generate_chips.py --tier 6q-hard --n 10 --min-ceiling 0.96 --workers 32 --out chips

Writes <out>/<tier>.jsonl in the format of the frozen benchmark (full hidden parameters per row), so
`scripts/evaluate.py --data <out> --tier <tier>` runs a policy on exactly these chips, and prints one line per chip
(couplings, weak links, feedlines). The tier name is "<n>q-<difficulty>" (chipcal.tiers.get; presets in
chipcal.tiers.DIFFICULTY). --min-ceiling runs the privileged oracle on every coupler (about 25 CPU-minutes each)
and keeps only chips whose every coupler admits a CZ of at least that fidelity, storing the ceilings in the rows.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from chipcal import splits, tiers
from chipcal.envs.one_qubit_discovery import feedlines


def ceiling(job):
    from chipcal.baselines.oracle import coupler_best_ceiling
    from chipcal.sim.chain import ChainPrior
    tier, seed, index = job
    chain = tiers.sample(tier, seed).chain
    currents = ChainPrior(n_qubits=chain.n_qubits).initial_parking_currents(chain, np.random.default_rng(seed))
    out = coupler_best_ceiling(chain, index, currents)
    return {k: out[k] for k in splits.CEILING_FIELDS}


def summary(row: dict, tier: tiers.Tier) -> str:
    chain = tiers.sample(tier, row["seed"]).chain
    g = [x for c in chain.couplers for x in (c.g_qc_left, c.g_qc_right)]
    weak = sum(x < tier.g_qc_hz[0] for x in g)
    text = (f"seed {row['seed']}: {chain.n_qubits} qubits ({tier.layout}), {chain.n_couplers} couplers, "
            f"g {min(g) / 1e6:.0f}-{max(g) / 1e6:.0f} MHz ({weak} weak links), {len(feedlines(chain.n_qubits))} feedline(s)")
    if "ceiling" in row:
        text += f", ceiling {min(c['fidelity'] for c in row['ceiling']):.3f}"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", type=tiers.tier_name, required=True, help="e.g. 8q-hard, 12q-extreme, 5q-nominal")
    parser.add_argument("--n", type=int, default=10, help="chips to write")
    parser.add_argument("--split", default="train", choices=sorted(tiers.SPLITS))
    parser.add_argument("--first", type=int, help="first seed (default: the start of the split)")
    parser.add_argument("--min-ceiling", type=float, help="keep only chips whose every coupler admits this CZ fidelity")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tier = tiers.get(args.tier)
    lo, hi = tiers.SPLITS[args.split]
    seed = lo if args.first is None else args.first
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        while len(rows) < args.n:
            # A batch of candidate seeds; with a ceiling filter, about as many as the workers can certify at once.
            batch = list(range(seed, min(hi + 1, seed + (args.n - len(rows) if args.min_ceiling is None
                                                         else max(1, args.workers // tier.n_couplers)))))
            if not batch:
                raise SystemExit(f"the {args.split} split ran out of seeds")
            seed = batch[-1] + 1
            candidates = [{k: v for k, v in splits.row(tier.name, s).items() if k != "version"} for s in batch]
            if args.min_ceiling is not None:
                jobs = [(tier.name, s, c) for s in batch for c in range(tier.n_couplers)]
                found = list(pool.map(ceiling, jobs))
                for k, row in enumerate(candidates):
                    row["ceiling"] = found[k * tier.n_couplers:(k + 1) * tier.n_couplers]
                candidates = [r for r in candidates if min(c["fidelity"] for c in r["ceiling"]) >= args.min_ceiling]
            for row in candidates[:args.n - len(rows)]:
                rows.append(row)
                print(summary(row, tier), flush=True)
    path = args.out / f"{tier.name}.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n" for r in rows))
    print(f"{len(rows)} chips written to {path}")


if __name__ == "__main__":
    main()
