"""Three-state readout channel.

q(x; theta, k) = C(theta) p_phys(x; theta, k), then counts ~ Multinomial(N, q).
The confusion matrix C is column-stochastic: C[i, j] = P(measure i | true j).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    """Tunable, robust measurement-noise knobs applied to the observation channel.

    All default to a no-op, are clipped to safe ranges, and act at the probability
    level so every node's plot picks them up uniformly.  They smear what the policy
    *measures* without touching the gate's true fidelity (readout/analog noise does
    not change the physics).

    - visibility         contrast in (0, 1]; q <- v*q + (1-v)*uniform (depolarize).
    - thermal_population  residual upward |0>->|1> excitation fraction [0, 0.5].
    - readout_error       extra symmetric 3-state assignment error [0, 0.6],
                          compounded on top of the device confusion matrix.
    - prob_sigma          analog Gaussian noise std added to probabilities before
                          sampling [0, 0.5] (graininess beyond binomial shot noise).
    """

    visibility: float = 1.0
    thermal_population: float = 0.0
    readout_error: float = 0.0
    prob_sigma: float = 0.0

    def clipped(self) -> "NoiseModel":
        return NoiseModel(
            visibility=float(np.clip(self.visibility, 0.05, 1.0)),
            thermal_population=float(np.clip(self.thermal_population, 0.0, 0.5)),
            readout_error=float(np.clip(self.readout_error, 0.0, 0.6)),
            prob_sigma=float(np.clip(self.prob_sigma, 0.0, 0.5)),
        )

    def is_identity(self) -> bool:
        return (self.visibility >= 1.0 and self.thermal_population <= 0.0
                and self.readout_error <= 0.0 and self.prob_sigma <= 0.0)


def _depolarize(p: np.ndarray, visibility: float, thermal: float) -> np.ndarray:
    """Contrast loss toward uniform plus residual |0>->|1> excitation; renormalized."""
    p = np.array(p, dtype=float)
    n = p.shape[-1]
    if visibility < 1.0:
        p = visibility * p + (1.0 - visibility) * (1.0 / n)
    if thermal > 0.0 and n >= 2:
        moved = thermal * p[..., 0]
        p[..., 0] -= moved
        p[..., 1] += moved
    s = p.sum(axis=-1, keepdims=True)
    return p / np.where(s <= 0.0, 1.0, s)


def _assignment_error_matrix(n: int, r: float) -> np.ndarray:
    """Column-stochastic symmetric assignment error: 1-r on the diagonal."""
    e = np.full((n, n), r / max(n - 1, 1))
    np.fill_diagonal(e, 1.0 - r)
    return e


def apply_confusion(p_phys: np.ndarray, confusion: np.ndarray) -> np.ndarray:
    """Map true qutrit populations (..., 3) through the readout confusion."""
    q = np.einsum("ij,...j->...i", confusion, np.asarray(p_phys, dtype=float))
    q = np.clip(q, 0.0, None)
    return q / q.sum(axis=-1, keepdims=True)


def sample_counts(q: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Multinomial counts with the state axis last: (..., 3) -> (..., 3) ints."""
    q = np.asarray(q, dtype=float)
    flat = q.reshape(-1, q.shape[-1])
    counts = np.stack([rng.multinomial(shots, row) for row in flat])
    return counts.reshape(q.shape)


def measure(
    p_phys: np.ndarray,
    confusion: np.ndarray,
    shots: int,
    rng: np.random.Generator,
    noise: "NoiseModel | None" = None,
) -> np.ndarray:
    """The full observation channel C p_phys -> multinomial counts, optionally with extra noise."""
    if noise is None or noise.is_identity():
        return sample_counts(apply_confusion(p_phys, confusion), shots, rng)
    noise = noise.clipped()
    p_phys = _depolarize(p_phys, noise.visibility, noise.thermal_population)
    if noise.readout_error > 0.0:
        confusion = _assignment_error_matrix(confusion.shape[0], noise.readout_error) @ confusion
    q = apply_confusion(p_phys, confusion)
    if noise.prob_sigma > 0.0:
        q = np.clip(q + rng.normal(0.0, noise.prob_sigma, size=q.shape), 0.0, None)
        q = q / q.sum(axis=-1, keepdims=True)
    return sample_counts(q, shots, rng)


def multinomial_loglik(counts: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Log-likelihood (up to the count-only normalization) summed over states.

    counts (..., 3) against probabilities q (broadcastable, state axis last).
    """
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1.0)
    return (np.asarray(counts) * np.log(q)).sum(axis=-1)
