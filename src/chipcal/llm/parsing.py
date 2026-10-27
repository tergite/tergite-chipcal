from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Optional
import numpy as np
import xarray as xr

def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _fmt_hz(v: float) -> str:
    return f"{v * 1e-6:.3f} MHz ({v:.9g} Hz)"


def _fmt_ghz(v: float) -> str:
    return f"{v * 1e-9:.6f} GHz ({v:.9g} Hz)"


def _fmt_s(v: float) -> str:
    return f"{v * 1e9:.1f} ns ({v:.9g} s)"


def _fmt_a(v: float) -> str:
    return f"{v * 1e6:.1f} uA ({v:.9g} A)"


def _fmt_float(v: Optional[float], kind: str = "") -> str:
    if v is None:
        return "unset"
    if kind == "hz":
        return _fmt_hz(v)
    if kind == "s":
        return _fmt_s(v)
    if kind == "a":
        return _fmt_a(v)
    return f"{v:.9g}"


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the top-level JSON object from a model response."""
    cleaned = text.strip()
    # Native-thinking models can return ``<think>...braces included...</think>`` before the
    # executable JSON. Parse only the post-thinking final channel so braces in prose cannot be
    # mistaken for the action object.
    think_end = cleaned.lower().rfind("</think>")
    if think_end >= 0:
        cleaned = cleaned[think_end + len("</think>"):].strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response did not contain a JSON object")
        obj = json.loads(cleaned[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("response JSON must be an object")
    return obj


def extract_json_action(text: str) -> dict[str, Any]:
    """Parse an action object from a model response."""

    obj = extract_json_object(text)
    if "action" in obj and isinstance(obj["action"], dict):
        obj = obj["action"]
    return obj


def extract_json_analysis(text: str) -> Optional[dict[str, Any]]:
    """Return optional structured analysis/reasoning from a wrapped response."""

    obj = extract_json_object(text)
    analysis = obj.get("analysis")
    if not isinstance(analysis, dict):
        analysis = obj.get("reason")
    return analysis if isinstance(analysis, dict) else None


def render_observation_outputs(
    ds: xr.Dataset,
    out_dir: Path,
    stem: str,
    *,
    dense: bool = False,
) -> dict[str, Any]:
    """Render dataset PNG diagnostics and return their paths."""

    try:
        from chipcal.viz.plotting import plot_dataset, safe_stem
    except Exception:
        return {}
    return {"main": str(plot_dataset(ds, out_dir, stem=safe_stem(stem)))}


def render_observation_image(ds: xr.Dataset, out_dir: Path, stem: str) -> Optional[Path]:
    """Render one dataset PNG for a VLM prompt; return None if plotting is absent."""

    paths = render_observation_outputs(ds, out_dir, stem, dense=False)
    main = paths.get("main")
    return Path(main) if main else None
