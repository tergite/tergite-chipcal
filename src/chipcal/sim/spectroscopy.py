import numpy as np


def validate_calibration_tls(settings, legacy_tls):
    """Reject scalar TLS penalty triples on a multimode device.

    Multimode devices model defects as physical ``ExtraMode`` entries of their settings; a
    parallel scalar penalty model would contradict them.
    """
    from chipcal.sim.multimode import MultimodeSettings
    if isinstance(settings, MultimodeSettings):
        if any(np.asarray(t).size for t in legacy_tls):
            raise ValueError("multimode requires physical ExtraMode parameters, not scalar TLS penalties")
        return
    if settings is not None and (getattr(settings, "tls", ()) or any(np.asarray(t).size for t in legacy_tls)):
        raise ValueError("TLS defects are only supported as multimode ExtraMode parameters")
