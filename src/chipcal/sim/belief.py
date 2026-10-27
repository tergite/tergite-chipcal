"""Grid belief over (g, Delta) for chevron-style data.

A coarse 2D grid posterior over the effective coupling g and the (dressed,
amplitude-inclusive) transition frequency Delta_eff at a fixed drive amplitude:
an exact Bayes update with the multinomial readout likelihood against a
two-level |11> <-> |20> forward model.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from chipcal.sim import physics, readout


class GridBelief:
    """b(g, Delta) on a rectangular grid; uniform prior by default."""

    def __init__(self, g_grid: np.ndarray, delta_grid: np.ndarray, log_prior: np.ndarray | None = None):
        self.g_grid = np.asarray(g_grid, dtype=float)
        self.delta_grid = np.asarray(delta_grid, dtype=float)
        shape = (self.g_grid.size, self.delta_grid.size)
        self.log_post = np.zeros(shape) if log_prior is None else np.array(log_prior, dtype=float)
        self._normalize()

    @classmethod
    def from_context(cls, delta_center: float, delta_span: float = 30e6, n_delta: int = 61,
                     g_max: float = 12e6, n_g: int = 49) -> "GridBelief":
        """Prior centered on the transition predicted from single-qubit quantities:
        g uniform on [~0, g_max], Delta uniform on a window around center."""
        g = np.linspace(1e4, g_max, n_g)
        d = delta_center + np.linspace(-delta_span / 2, delta_span / 2, n_delta)
        return cls(g, d)

    def _normalize(self) -> None:
        self.log_post -= self.log_post.max()
        z = np.exp(self.log_post).sum()
        self.log_post -= np.log(z)

    @property
    def posterior(self) -> np.ndarray:
        return np.exp(self.log_post)

    # ---------------------------------------------------------------- updates
    def update_from_chevron(self, dataset: xr.Dataset, control_var: str, confusion: np.ndarray) -> None:
        """Exact Bayes update from a cz_chevron counts dataset.

        Forward model per hypothesis (g, Delta): two-level transfer
        at each (f_d, t) grid point -> control populations [0, P_ret, P_tr] ->
        known confusion -> multinomial log-likelihood of the observed counts.
        """
        counts = np.moveaxis(dataset[control_var].values, 0, -1)  # (nF, nT, 3)
        freqs = dataset["cz_pulse_frequencies"].values
        times = dataset["cz_pulse_durations"].values

        g = self.g_grid[:, None, None, None]
        delta = freqs[None, None, :, None] - self.delta_grid[None, :, None, None]
        t = times[None, None, None, :]

        p_tr = physics.transfer_probability(g, delta, t)            # (nG, nD, nF, nT)
        p_states = np.stack([np.zeros_like(p_tr), 1.0 - p_tr, p_tr], axis=-1)
        q = readout.apply_confusion(p_states, confusion)
        ll = readout.multinomial_loglik(counts[None, None], q)      # (nG, nD, nF, nT)
        self.log_post += ll.sum(axis=(-1, -2))
        self._normalize()

    def update_from_parametrization_slice(self, dataset: xr.Dataset, control_var: str,
                                          confusion: np.ndarray, amp_index: int,
                                          duration: float) -> None:
        """Update from one amplitude column of a cz_parametrization scan
        (fixed t): same channel, frequency axis only."""
        counts = np.moveaxis(dataset[control_var].values, 0, -1)[0, :, amp_index]  # (nF, 3)
        freqs = dataset["cz_pulse_frequencies"].values

        g = self.g_grid[:, None, None]
        delta = freqs[None, None, :] - self.delta_grid[None, :, None]
        p_tr = physics.transfer_probability(g, delta, duration)
        p_states = np.stack([np.zeros_like(p_tr), 1.0 - p_tr, p_tr], axis=-1)
        q = readout.apply_confusion(p_states, confusion)
        ll = readout.multinomial_loglik(counts[None, None], q)
        self.log_post += ll.sum(axis=-1)
        self._normalize()

    # ------------------------------------------------------------ diagnostics
    def map_estimate(self) -> tuple[float, float]:
        i, j = np.unravel_index(np.argmax(self.log_post), self.log_post.shape)
        return float(self.g_grid[i]), float(self.delta_grid[j])

    def mean_std(self) -> tuple[tuple[float, float], tuple[float, float]]:
        p = self.posterior
        pg = p.sum(axis=1)
        pd = p.sum(axis=0)
        mg = float((pg * self.g_grid).sum())
        md = float((pd * self.delta_grid).sum())
        sg = float(np.sqrt((pg * (self.g_grid - mg) ** 2).sum()))
        sd = float(np.sqrt((pd * (self.delta_grid - md) ** 2).sum()))
        return (mg, sg), (md, sd)

    def credible_contains(self, g_true: float, delta_true: float, n_sigma: float = 3.0) -> bool:
        """True if the truth lies within n_sigma posterior std of the mean,
        never tighter than the grid can resolve (half a cell)."""
        (mg, sg), (md, sd) = self.mean_std()
        res_g = 0.5 * float(np.diff(self.g_grid).mean())
        res_d = 0.5 * float(np.diff(self.delta_grid).mean())
        return (
            abs(g_true - mg) <= n_sigma * sg + res_g
            and abs(delta_true - md) <= n_sigma * sd + res_d
        )

    def commit_candidates(self) -> dict:
        """Induced predictive over the decision variables:
        t_g* = 1/(2 g_MAP) (full exchange-and-return cycle), f_d* = Delta_MAP."""
        g_map, d_map = self.map_estimate()
        return {"cz_pulse_frequency": d_map, "cz_pulse_duration": 1.0 / (2.0 * g_map)}

    def feasibility_mass(self, t_max: float = 1e-6, g_min: float | None = None) -> float:
        """Posterior mass on 'a CZ is reachable here': g large enough that a
        full cycle fits the pulse-length limit (a flat
        map concentrates b on g ~ 0 and this mass collapses -> dismiss)."""
        g_req = max(1.0 / (2.0 * t_max), 0.0 if g_min is None else g_min)
        return float(self.posterior[self.g_grid >= g_req, :].sum())
