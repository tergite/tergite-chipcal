from dataclasses import fields, is_dataclass, replace
import hashlib, json
from pathlib import Path
import numpy as np

def registry():
    from chipcal.sim.chain import ChainDeviceParams, QubitParams, CouplerParams
    from chipcal.sim.params import DeviceParams
    from chipcal.sim.multimode import MultimodeSettings, ExtraMode
    from chipcal.sim.crosstalk import FluxTransfer
    from chipcal.sim.readout import NoiseModel
    from chipcal.sim.measurement_profile import MeasurementProfile
    from chipcal.envs.curriculum import ChainDevice
    from chipcal.sim.realism import QubitNoise, Realism
    return {c.__name__: c for c in (ChainDeviceParams, QubitParams, CouplerParams,
        DeviceParams, MultimodeSettings, ExtraMode, FluxTransfer, NoiseModel,
        MeasurementProfile, ChainDevice, QubitNoise, Realism)}


def encode(value):
    if is_dataclass(value):
        if type(value).__name__ not in registry():
            raise ValueError(f"unsupported device type {type(value).__name__}")
        later = _OMIT_AT_DEFAULT.get(type(value).__name__, set())
        # Optional realism fields are written only when they differ from their default, so a device
        # without realism encodes (and hashes) the same whether or not those fields exist.
        return {"type": type(value).__name__, **{f.name: encode(getattr(value, f.name)) for f in fields(value)
                                                 if not (f.name in later and _is_default(f, getattr(value, f.name)))}}
    if isinstance(value, np.ndarray):
        return {"type": "array", "shape": list(value.shape), "values": encode(value.tolist())}
    if isinstance(value, (complex, np.complexfloating)):
        return {"type": "complex", "real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, tuple):
        return {"type": "tuple", "values": [encode(x) for x in value]}
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("device values must be finite")
        return value
    raise ValueError(f"unsupported device value {type(value).__name__}")


def _is_default(f, value):
    from dataclasses import MISSING
    if f.default is not MISSING:
        return value is f.default or (type(value) is type(f.default) and value == f.default)
    return False


_OMIT_AT_DEFAULT = {"ChainDeviceParams": {"realism", "edges"},
                    "DeviceParams": {"parity_c", "parity_t", "parity12_c", "parity12_t", "frame_offset_c",
                                     "frame_offset_t", "calibrated_frames"},
                    "MultimodeSettings": {"g_reference_hz", "flux_noise_phi0"}}
_LATER_FIELDS = {**_OMIT_AT_DEFAULT,
                 "MultimodeSettings": {"spectroscopy_model", "pulse_solver"} | _OMIT_AT_DEFAULT["MultimodeSettings"]}


def decode(value):
    if isinstance(value, list):
        return [decode(x) for x in value]
    if not isinstance(value, dict):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("device values must be finite")
        return value
    kind = value.get("type")
    if kind is None:
        return {k: decode(v) for k, v in value.items()}
    if kind == "array" and set(value) == {"type", "shape", "values"}:
        data = np.asarray(decode(value["values"]))
        shape = value["shape"]
        if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError("invalid array shape")
        return data.reshape(shape)
    if kind == "complex" and set(value) == {"type", "real", "imag"}:
        return complex(decode(value["real"]), decode(value["imag"]))
    if kind == "tuple" and set(value) == {"type", "values"}:
        return tuple(decode(x) for x in value["values"])
    cls = registry().get(kind)
    names = {f.name for f in fields(cls)} if cls is not None else set()
    # Optional fields; absent means the dataclass default.
    optional = _LATER_FIELDS.get(kind, set())
    if cls is None or not ({"type", *names} - optional) <= set(value) <= {"type", *names}:
        raise ValueError(f"unknown type or missing/extra fields: {kind}")
    return cls(**{k: decode(v) for k, v in value.items() if k != "type"})


def validate(spec, backend=None, n_qubits=None):
    if not isinstance(spec, dict) or set(spec) != {"schema", "backend", "device", "measurement_profile", "spectroscopy"}:
        raise ValueError("expected a complete chipcal-device-v1 document")
    if spec["schema"] != "chipcal-device-v1" or spec["backend"] != "multimode":
        raise ValueError("unsupported device schema/backend")
    if backend and backend != spec["backend"]:
        raise ValueError("device backend differs from requested backend")
    json.dumps(spec, allow_nan=False)
    device = decode(spec["device"])
    from chipcal.envs.curriculum import ChainDevice
    from chipcal.sim.chain import ChainDeviceParams, CouplerParams, QubitParams
    from chipcal.sim.multimode import MultimodeSettings, Cell
    from chipcal.sim.chain import coupler_theta
    from chipcal.sim.readout import NoiseModel
    from chipcal.sim.measurement_profile import MeasurementProfile
    if not isinstance(device, ChainDevice) or not isinstance(device.chain, ChainDeviceParams):
        raise ValueError("expected ChainDevice with ChainDeviceParams")
    chain = device.chain
    if not 1 <= len(chain.qubits) <= 5 or (n_qubits and len(chain.qubits) != n_qubits):
        raise ValueError("device qubit count does not match launch settings (1–5)")
    if not 0 < device.f_star <= 1 or device.crosstalk_scale < 0:
        raise ValueError("invalid fidelity target/crosstalk scale")
    if not np.isfinite(chain.crosstalk).all() or np.any(np.diag(chain.crosstalk) == 0):
        raise ValueError("finite crosstalk with nonzero diagonal required")
    for q in chain.qubits:
        if not isinstance(q, QubitParams) or q.f01 <= 0 or not 0 < q.t2 <= 2*q.t1:
            raise ValueError("qubits need positive frequency and 0 < T2 <= 2 T1")
        if q.confusion.shape != (3, 3) or np.any(q.confusion < 0) or not np.allclose(q.confusion.sum(axis=0), 1):
            raise ValueError("confusion matrices must be nonnegative, 3x3 and column-stochastic")
    for i, c in enumerate(chain.couplers):
        if not isinstance(c, CouplerParams) or c.fc0 <= 0 or c.i_period <= 0 or not 0 <= c.d_asym <= 1:
            raise ValueError("invalid coupler frequency, period or asymmetry")
        if (c.pulse_wait_s < 0 or c.pulse_rise_fall_s < 0 or c.pulse_max_step_s <= 0
                or c.dist_lowpass_tau < 0 or c.dist_droop_tau < 0 or not 0 <= c.dist_droop_amp <= .6):
            raise ValueError("invalid pulse timing or distortion")
        if c.g_qc_left < 0 or c.g_qc_right < 0 or c.ac_flux_per_amp <= 0:
            raise ValueError("nonnegative couplings and positive AC flux gain required")
        if c.tls.ndim != 2 or c.tls.shape[1] != 3:
            raise ValueError("TLS must have shape (n,3)")
        if not isinstance(c.multimode, MultimodeSettings):
            raise ValueError("explicit MultimodeSettings required on every coupler")
        if spec["backend"] == "multimode":
            if c.tls.size:
                raise ValueError("multimode needs explicit ExtraMode defects, not legacy TLS penalties")
            Cell(coupler_theta(chain, i, np.zeros(len(chain.couplers))))
    if device.noise is not None:
        if not isinstance(device.noise, NoiseModel) or device.noise != device.noise.clipped():
            raise ValueError("noise values exceed their supported ranges")
    profile, spectroscopy = decode(spec["measurement_profile"]), decode(spec["spectroscopy"])
    if profile is not None and not isinstance(profile, MeasurementProfile):
        raise ValueError("invalid measurement profile")
    if spec["backend"] == "multimode" and spectroscopy is not None:
        raise ValueError("multimode uses its own spectroscopy settings")
    # Derived reporting value, not an independent hidden physics knob.
    device.mean_g_qc = float(np.mean([.5*(c.g_qc_left+c.g_qc_right) for c in chain.couplers])) if chain.couplers else 0.
    return device, {"measurement_profile": profile, "spectroscopy": spectroscopy}


def load(path, backend=None, n_qubits=None):
    spec = json.loads(Path(path).read_text()) if isinstance(path, (str, Path)) else path
    return validate(spec, backend, n_qubits)


def save(spec, directory, seed):
    if not directory:
        return
    raw = json.dumps(spec, indent=2, allow_nan=False)
    folder = Path(directory)
    if folder.name.startswith("transcripts_"):
        folder = folder.parent
    target = folder / f"device-manifest-{seed}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(raw + '\n')
    target.with_suffix('.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest() + '\n')
