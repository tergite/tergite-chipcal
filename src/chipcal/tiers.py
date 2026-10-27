"""Benchmark tiers and the seeded chip generator.

Three tiers share one physical prior and differ in size and coupling:

    tier   chip        couplers   g_qc range      flux crosstalk   turn cap
    2q1c   2 qubits    1          45-72 MHz       -                150
    3q2c   3 qubits    2          35-72 MHz       0.25-0.5 %       250
    5q4c   5 qubits    4          27-72 MHz       0.5-1 %          400

Every episode starts with discovery (resonators, f01, f12) and runs with realism on
(chipcal.sim.realism). Seeds are split into disjoint train / validation / test ranges;
``sample(tier, seed)`` is a pure function of its arguments, so a seed names one chip forever.
The ranges and their sources are listed in docs/parameters.md.

Procedural tiers extend the same generator to any chip size and difficulty: ``get("12q-hard")`` is a
12-qubit chip at the ``hard`` preset of ``DIFFICULTY``. Up to 5 qubits it is a chain; beyond, a grid whose
rows are its readout feedlines (at most 5 qubits each), with couplers along each row and between rows.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import re

import numpy as np

from chipcal.envs.curriculum import ChainDevice
from chipcal.envs.one_qubit_discovery import feedlines
from chipcal.sim.chain import ChainDeviceParams, ChainPrior, QubitParams
from chipcal.sim.multimode import MultimodeSettings
from chipcal.sim import transmon
from chipcal.sim.realism import RealismProfile, realize

GENERATOR_VERSION = "chipcal-chips-v0.1"

# Disjoint seed ranges [lo, hi] (inclusive). The frozen test split uses the first seeds of "test".
SPLITS = {"train": (1_000_000, 1_999_999), "validation": (2_000_000, 2_009_999), "test": (3_000_000, 3_009_999)}


@dataclass(frozen=True)
class Tier:
    name: str
    n_qubits: int
    g_qc_hz: tuple[float, float]
    crosstalk_scale: float
    max_turns: int
    budget_seconds: float = 8 * 3600.0
    f_star: float = 0.95
    # Overrides of PRIOR, and a fraction of qubit-coupler links redrawn from a weak-coupling band.
    prior: tuple[tuple[str, tuple[float, float]], ...] = ()
    weak_g_qc_hz: tuple[float, float] = (9e6, 20e6)
    weak_fraction: float = 0.0
    min_cz_detuning_hz: float = 100e6
    layout: str = "chain"

    @property
    def edges(self) -> list[tuple[int, int]]:
        """The qubit pair of every coupler."""
        return layout_edges(self.n_qubits, self.layout)

    @property
    def n_couplers(self) -> int:
        return len(self.edges)


def grid_positions(n_qubits: int) -> dict[int, tuple[int, int]]:
    """(row, column) of every qubit of a grid: each row is one readout feedline, filled left to right."""
    return {q: (r, c) for r, row in enumerate(feedlines(n_qubits)) for c, q in enumerate(row)}


def layout_edges(n_qubits: int, layout: str) -> list[tuple[int, int]]:
    """Coupled qubit pairs: neighbours along a chain, or along each grid row and then between rows."""
    if layout == "chain":
        return [(q, q + 1) for q in range(n_qubits - 1)]
    rows = feedlines(n_qubits)
    along = [(q, q + 1) for row in rows for q in row[:-1]]
    across = [(rows[r][c], rows[r + 1][c]) for r in range(len(rows) - 1) for c in range(min(len(rows[r]), len(rows[r + 1])))]
    return along + across


TIERS = {t.name: t for t in (
    Tier("2q1c", 2, (45e6, 72e6), 0.0, 150),
    Tier("3q2c", 3, (35e6, 72e6), 0.005, 250),
    Tier("5q4c", 5, (27e6, 72e6), 0.01, 400),
)}

# Physical prior shared by all tiers (generic ranges; see docs/parameters.md).
PRIOR = {
    # Fixed-frequency transmons alternate between two bands so neighbours are detuned.
    "frequency_bands_hz": ((4.0e9, 4.4e9), (4.6e9, 5.0e9)),
    # E_J/E_C of the transmon design; the anharmonicity follows from it and f01 (exact spectrum).
    "ej_over_ec": (36.5, 100.0),
    "t1_s": (15e-6, 110e-6),
    "t2_over_t1": (0.6, 2.0),
    # Tunable coupler (SQUID transmon) above the qubits.
    "coupler_max_frequency_hz": (6.47e9, 7.92e9),
    "coupler_t1_s": (8e-6, 12e-6),
    "coupler_t2_s": (1e-6, 6e-6),
    "coupler_alpha_hz": (-90e6, -75e6),
    "coupler_asymmetry": (0.03, 0.08),
    "i_period_a": (2.4e-3, 3.6e-3),
    "i_sweet_a": (-1.0e-3, 1.0e-3),
    "ac_flux_per_amp": (0.15, 0.35),
    "readout_off_diagonal": (0.01, 0.05),
    # Quasi-static flux noise of each coupler loop, rms [Phi_0] (realism only).
    "flux_noise_phi0": (5e-6, 11e-6),
}
# Difficulty presets of the procedural tiers "<n>q-<difficulty>". "nominal" is the benchmark prior at the 5q4c
# coupling range; harder presets weaken the couplings (part of them into a weak band where the CZ window is
# narrow and slow), raise the flux crosstalk, worsen readout and coherence, and crowd the CZ detuning.
DIFFICULTY = {
    "nominal": dict(g_qc_hz=(27e6, 72e6), crosstalk_scale=0.01),
    "hard": dict(g_qc_hz=(20e6, 50e6), crosstalk_scale=0.02, weak_fraction=0.25, weak_g_qc_hz=(12e6, 20e6),
                 prior=(("readout_off_diagonal", (0.03, 0.08)), ("t1_s", (15e-6, 60e-6))),
                 min_cz_detuning_hz=80e6),
    "extreme": dict(g_qc_hz=(15e6, 40e6), crosstalk_scale=0.04, weak_fraction=0.5, weak_g_qc_hz=(9e6, 15e6),
                    prior=(("readout_off_diagonal", (0.05, 0.12)), ("t1_s", (10e-6, 40e-6)),
                           ("coupler_asymmetry", (0.02, 0.12))),
                    min_cz_detuning_hz=60e6),
}
MAX_QUBITS = 64


def get(name: str | Tier) -> Tier:
    """A named benchmark tier, or a procedural one such as "8q-hard": a chain up to 5 qubits, a grid beyond.

    A procedural tier allows 50 turns plus 100 per coupler and 2 h of lab time per coupler (8 h at least)."""
    if isinstance(name, Tier):
        return name
    if name in TIERS:
        return TIERS[name]
    match = re.fullmatch(r"(\d+)q-(\w+)", name)
    if not match or match.group(2) not in DIFFICULTY or not 2 <= int(match.group(1)) <= MAX_QUBITS:
        raise ValueError(f"unknown tier {name!r}: use one of {sorted(TIERS)} or '<n>q-<difficulty>' with 2 <= n <= "
                         f"{MAX_QUBITS} and difficulty in {sorted(DIFFICULTY)}")
    n = int(match.group(1))
    layout = "chain" if n <= 5 else "grid"
    n_c = len(layout_edges(n, layout))
    return Tier(name, n, max_turns=50 + 100 * n_c, budget_seconds=max(8.0, 2.0 * n_c) * 3600.0, layout=layout,
                **DIFFICULTY[match.group(2)])


def tier_name(name: str) -> str:
    """argparse type: a valid tier name."""
    return get(name).name


def _check_seed(seed: int, split: str) -> None:
    lo, hi = SPLITS[split]
    if not lo <= int(seed) <= hi:
        raise ValueError(f"seed {seed} is outside the {split} split [{lo}, {hi}]")


def split_of(seed: int) -> str:
    for name, (lo, hi) in SPLITS.items():
        if lo <= int(seed) <= hi:
            return name
    raise ValueError(f"seed {seed} is in no split")


def sample(tier: str | Tier, seed: int, *, split: str | None = None, realism: RealismProfile | None | bool = True) -> ChainDevice:
    """The chip of ``tier`` named by ``seed``. ``realism=False`` returns the stationary base device."""
    tier = get(tier)
    _check_seed(seed, split or split_of(seed))
    rng = np.random.default_rng([int(seed), 7])
    prior_ranges = {**PRIOR, **dict(tier.prior)}
    u = lambda key: float(rng.uniform(*prior_ranges[key]))  # noqa: E731
    edges = tier.edges
    # Coupled qubits sit in alternate bands: by index along a chain, as a checkerboard on a grid.
    band = [q % 2 for q in range(tier.n_qubits)] if tier.layout == "chain" else \
        [sum(rc) % 2 for _, rc in sorted(grid_positions(tier.n_qubits).items())]
    for _ in range(1000):
        lines = [(float(rng.uniform(*PRIOR["frequency_bands_hz"][band[q]])), u("ej_over_ec")) for q in range(tier.n_qubits)]
        lines = [(f01, transmon.from_f01(f01, ratio)["alpha_hz"]) for f01, ratio in lines]
        if all(abs(min(a, b)[0] - (max(a, b)[0] + max(a, b)[1])) >= tier.min_cz_detuning_hz
               for a, b in ((lines[i], lines[j]) for i, j in edges)):
            break
    else:
        raise RuntimeError("no collision-free frequency assignment")
    qubits = []
    for f01, alpha in lines:
        t1 = u("t1_s")
        confusion = np.eye(3)
        for j in range(3):
            other = [k for k in range(3) if k != j]
            confusion[other, j] = rng.uniform(*prior_ranges["readout_off_diagonal"], size=2)
            confusion[j, j] = 1 - confusion[other, j].sum()
        qubits.append(QubitParams(f01, float(alpha), t1, float(min(2 * t1, t1 * u("t2_over_t1"))), confusion))
    # The chain prior supplies one coupler per edge (its own qubits are not used).
    prior = ChainPrior(n_qubits=len(edges) + 1, tls_rate=0, crosstalk_scale=tier.crosstalk_scale)
    raw = prior.sample(np.random.default_rng([int(seed), 11]))
    couplers = []
    for c in raw.couplers:
        bus_t1 = u("coupler_t1_s")
        flux_noise = u("flux_noise_phi0")
        settings = MultimodeSettings(bus_t1=bus_t1, bus_t2=min(2 * bus_t1, u("coupler_t2_s")),
                                     bus_alpha=u("coupler_alpha_hz"), flux_noise_phi0=flux_noise if realism else 0.0)
        couplers.append(replace(c, fc0=u("coupler_max_frequency_hz"), d_asym=u("coupler_asymmetry"),
                                i_period=u("i_period_a"), i_sweet=u("i_sweet_a"), ac_flux_per_amp=u("ac_flux_per_amp"),
                                g_qc_left=float(rng.uniform(*tier.g_qc_hz)), g_qc_right=float(rng.uniform(*tier.g_qc_hz)),
                                multimode=settings))
    if tier.weak_fraction:
        weak = np.random.default_rng([int(seed), 17])
        couplers = [replace(c, **{side: float(weak.uniform(*tier.weak_g_qc_hz))
                                  for side in ("g_qc_left", "g_qc_right") if weak.random() < tier.weak_fraction})
                    for c in couplers]
    grid = None if tier.layout == "chain" else tuple(edges)
    matrix = prior._crosstalk_matrix(np.random.default_rng([int(seed), 13]), couplers, tier.crosstalk_scale, grid)
    chain = ChainDeviceParams(qubits, couplers, matrix, edges=grid)
    if realism:
        chain = realize(chain, int(seed), realism if isinstance(realism, RealismProfile) else RealismProfile())
    mean_g = float(np.mean([.5 * (c.g_qc_left + c.g_qc_right) for c in chain.couplers]))
    return ChainDevice(chain, "nominal", None, tier.f_star, tier.crosstalk_scale, mean_g)


def make_env(tier: str | Tier, **overrides):
    """A ChainAgentEnv configured for ``tier``: discovery on, guardrails on, final-RB score."""
    from chipcal.envs.chain_agent_env import ChainAgentEnv
    from chipcal.sim.measurement_profile import default_profile
    tier = get(tier)
    kwargs = dict(n_qubits=tier.n_qubits, crosstalk_scale=tier.crosstalk_scale, max_turns=tier.max_turns,
                  budget_seconds=tier.budget_seconds, f_star=tier.f_star, reward_mode="final_rb",
                  invalid_penalty=0.0, shaping=False, partial_credit_alpha=0.0, discovery=True,
                  guardrails=True, scout_bias=True, measurement_profile=default_profile(),
                  history_char_budget=24000, notebook_char_budget=8000)
    kwargs.update(overrides)
    return ChainAgentEnv(**kwargs)
