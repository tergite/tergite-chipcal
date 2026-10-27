from dataclasses import dataclass, field
from typing import Any, Optional
import xarray as xr

DC_CURRENT_LIMIT_A: float = 3.1e-3


@dataclass
class CostModel:
    """Simulated lab-time economics (seconds): compile/arm overhead per node, slow DC
    bias settling, a ~quarter-millisecond shot cycle, and an expensive standardized
    IRB. Generic estimates."""

    node_overhead: float = 15.0
    point_overhead: float = 2e-3
    shot_seconds: float = 250e-6
    dc_settle: float = 20.0
    commit_seconds: float = 0.0
    benchmark_seconds: float = 270.0

    def measurement_cost(self, n_points: int, shots: int, n_bias_changes: int) -> float:
        return (
            self.node_overhead
            + n_bias_changes * self.dc_settle
            + n_points * (self.point_overhead + shots * self.shot_seconds)
        )


@dataclass
class StepResult:
    dataset: Optional[xr.Dataset]
    cost_seconds: float
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
