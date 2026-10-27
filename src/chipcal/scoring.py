"""The benchmark score: the minimum, over couplers, of each gate's final interleaved-RB fidelity.

When the episode ends, the environment benchmarks every coupler's final committed gate once more,
at the final bias currents and wall-clock time, with a fixed high-statistics interleaved-RB protocol.
The agent's own benchmarks during the episode do not count, so repeating them until a lucky draw
gains nothing. Each coupler's fidelity is the F of that single final fit:

    F_i   = F_hat of the final IRB fit of coupler i
    score = min_i F_i

If the final fit is not identifiable (the survival curves collapse to the 1/4 floor within a few
Cliffords), the coupler scores RB_FLOOR = 0.25, the fidelity of a fully depolarising gate. A coupler
with an incomplete configuration (any of parking current, CZ frequency, amplitude or duration
uncommitted) scores 0. The simulator's true average gate fidelity is reported alongside, unscored.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from chipcal.sim.oracle import IRBProtocol

# More seeds and shots than the agent's benchmark, so the score's statistical spread is small.
SCORING_IRB = IRBProtocol(lengths=(0, 1, 2, 4, 8, 16, 32, 64, 128), n_seeds=21, shots=1000)
RB_FLOOR = 0.25


def final_rb_fidelity(theta, config, rng: np.random.Generator) -> tuple[float, dict]:
    """(F, details) of one coupler's final scoring benchmark."""
    from chipcal.sim.multimode_rb import run_irb
    if not config.is_complete():
        return 0.0, {"incomplete": True}
    result = run_irb(theta, config, rng, SCORING_IRB)
    reliable = bool(result.dataset.attrs.get("fit_reliable", True))
    value = float(result.f_hat) if reliable else RB_FLOOR
    return value, {"F_hat": float(result.f_hat), "F_sigma": float(result.f_sigma), "fit_reliable": reliable,
                   "reference_decay": float(result.p_ref)}


def episode_score(fidelities: Iterable[float]) -> float:
    values = [float(f) for f in fidelities]
    return min(values) if values else 0.0
