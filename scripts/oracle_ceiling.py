"""Privileged physics ceilings: the best CZ each test chip's couplers admit, over each coupler's own bias.

    python scripts/oracle_ceiling.py --tier 2q1c --n 8 --workers 16 > ceilings-2q1c.jsonl
    python scripts/oracle_ceiling.py --tier 5q4c --first 3000000 --n 400 --shard 3/16 --workers 64 > part-3.jsonl

One JSON line per coupler. The other couplers sit at the reset bias. ``--reset-bias`` searches the pulse only, at
the reset bias. ``--shard i/k`` keeps every k-th (seed, coupler) job, so k cluster jobs split one pool.
``python -m chipcal.splits build --ceilings ...`` selects the frozen benchmark chips from these lines.

``--full`` runs the v0.2 search (``chipcal.baselines.oracle.coupler_full_ceiling``): every CZ route, the whole flux range,
the environment's pulse limits, and robustness over offset-charge states. It runs the chips one at a time with
``--workers`` processes for each chip's flux screen; ``--frozen <file>`` takes the chips (and their v0.1 ceiling gates,
kept as candidates) from a frozen split:

    python scripts/oracle_ceiling.py --full --frozen data/v0.1/2q1c-test.jsonl --shard 0/25 --workers 16 > full-0.jsonl
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from chipcal import tiers
from chipcal.baselines.oracle import coupler_best_ceiling, coupler_ceiling, coupler_full_ceiling
from chipcal.sim.chain import ChainPrior


def job(args):
    tier, seed, index, reset_only = args
    chain = tiers.sample(tier, seed).chain
    currents = ChainPrior(n_qubits=chain.n_qubits).initial_parking_currents(chain, np.random.default_rng(seed))
    out = (coupler_ceiling if reset_only else coupler_best_ceiling)(chain, index, currents)
    return {"tier": tier, "seed": seed, "search": "reset_bias" if reset_only else "best_bias", **out}


def full(args):
    i, k = map(int, args.shard.split("/"))
    if args.frozen:
        chips = [json.loads(line) for line in args.frozen.read_text().splitlines() if line.strip()]
        jobs = [(c["tier"], c["seed"], j, c["ceiling"][j]) for c in chips for j in range(len(c["ceiling"]))]
    else:
        n_c = tiers.get(args.tier).n_couplers
        jobs = [(args.tier, seed, j, None) for seed in range(args.first, args.first + args.n) for j in range(n_c)]
    with ProcessPoolExecutor(args.workers) as pool:
        for tier, seed, index, legacy in jobs[i::k]:
            t0 = time.time()
            chain = tiers.sample(tier, seed).chain
            # the other couplers at their reset currents, as in the v0.1 search
            currents = ChainPrior(n_qubits=chain.n_qubits).initial_parking_currents(chain, np.random.default_rng(seed))
            out = coupler_full_ceiling(chain, index, currents, legacy=legacy, map_fn=pool.map)
            print(json.dumps({"tier": tier, "seed": seed, "search": "full_v0.2", "seconds": round(time.time() - t0),
                              "legacy_fidelity": None if legacy is None else legacy["fidelity"], **out}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default="2q1c", type=tiers.tier_name)
    parser.add_argument("--first", type=int, default=tiers.SPLITS["test"][0])
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--shard", default="0/1", help="i/k: run every k-th job starting at i")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reset-bias", action="store_true")
    parser.add_argument("--full", action="store_true", help="the v0.2 search: every route, flux range and robustness")
    parser.add_argument("--frozen", type=Path, help="--full: take the chips from this frozen split file")
    args = parser.parse_args()
    if args.full:
        return full(args)
    i, k = map(int, args.shard.split("/"))
    n_c = tiers.get(args.tier).n_couplers
    jobs = [(args.tier, seed, c, args.reset_bias) for seed in range(args.first, args.first + args.n) for c in range(n_c)]
    with ProcessPoolExecutor(args.workers) as pool:
        # Each line carries its seed and coupler, so lines are written as jobs finish: the file shows true progress.
        for future in as_completed([pool.submit(job, j) for j in jobs[i::k]]):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
