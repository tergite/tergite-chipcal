from dataclasses import dataclass
from chipcal.sim.chain import ChainDeviceParams
from chipcal.sim.readout import NoiseModel

@dataclass
class ChainDevice:
    """One matched-group chain chip for a curriculum stage.  ``theta`` aliases ``chain`` so
    rollout loops that pass ``device.theta`` drive the chain env unchanged."""

    chain: ChainDeviceParams
    family: str
    noise: NoiseModel | None
    f_star: float
    crosstalk_scale: float
    mean_g_qc: float

    @property
    def theta(self) -> ChainDeviceParams:
        return self.chain

    @property
    def g_qc(self) -> float:            # for g_bin() metric break-outs, mirroring Device
        return self.mean_g_qc
