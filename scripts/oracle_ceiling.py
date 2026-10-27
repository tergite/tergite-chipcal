"""Privileged physics ceilings: the best CZ each test chip's couplers admit, over each coupler's own bias.

    python scripts/oracle_ceiling.py --tier 2q1c --n 8 --workers 16 > ceilings-2q1c.jsonl
    python scripts/oracle_ceiling.py --tier 5q4c --first 3000000 --n 400 --shard 3/16 --workers 64 > part-3.jsonl

One JSON line per coupler. The other couplers sit at the reset bias. ``--reset-bias`` searches the pulse only, at
the reset bias. ``--shard i/k`` keeps every k-th (seed, coupler) job, so k cluster jobs split one pool.
``python -m chipcal.splits build --ceilings ...`` selects the frozen benchmark chips from these lines.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from chipcal import tiers
from chipcal.baselines.oracle import coupler_best_ceiling, coupler_ceiling
from chipcal.sim.chain import ChainPrior


def job(args):
    tier, seed, index, reset_only = args
    chain = tiers.sample(tier, seed).chain
    currents = ChainPrior(n_qubits=chain.n_qubits).initial_parking_currents(chain, np.random.default_rng(seed))
    out = (coupler_ceiling if reset_only else coupler_best_ceiling)(chain, index, currents)
    return {"tier": tier, "seed": seed, "search": "reset_bias" if reset_only else "best_bias", **out}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default="2q1c", type=tiers.tier_name)
    parser.add_argument("--first", type=int, default=tiers.SPLITS["test"][0])
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--shard", default="0/1", help="i/k: run every k-th job starting at i")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reset-bias", action="store_true")
    args = parser.parse_args()
    i, k = map(int, args.shard.split("/"))
    n_c = tiers.get(args.tier).n_couplers
    jobs = [(args.tier, seed, c, args.reset_bias) for seed in range(args.first, args.first + args.n) for c in range(n_c)]
    with ProcessPoolExecutor(args.workers) as pool:
        # Each line carries its seed and coupler, so lines are written as jobs finish: the file shows true progress.
        for future in as_completed([pool.submit(job, j) for j in jobs[i::k]]):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
