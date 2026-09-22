"""English documentation is provided in README.md."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import numpy as np
from settings import BANDS_NM, MATERIALS

WAVES = np.asarray(BANDS_NM, dtype=float)
G0, G1 = 0.089, 0.1245
LEE_METHOD = "Lee-D2-8th-A1=1-A2=1/pi-1.03-2.4-1.04-5.4-n1.34-nadir"

def _vector4(value: Any, name: str) -> np.ndarray:
    """English documentation is provided in README.md."""
    out = np.asarray(value, dtype=float)
    if out.shape != (4,):
        raise ValueError(f"{name} must contain four values in wavelength order")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains nonfinite input")
    return out.copy()


def load_water(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """English documentation is provided in README.md."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    header_end = next((i for i, line in enumerate(lines)
                       if line.strip().startswith("\\end_header")), None)
    if header_end is None:
        raise ValueError("Pure-water table must have a \\end_header marker")
    rows = []
    for line in lines[header_end + 1:]:
        value = line.strip()
        if value.startswith("\\end_data"):
            break
        if not value or value.startswith(("!", "#")):
            continue
        parts = value.split()
        if len(parts) != 3:
            raise ValueError(f"Expected wavelength,a,b in pure-water row: {value}")
        wave = float(parts[0])
        if wave not in (440.0, 445.0, 490.0, 560.0, 665.0):
            continue
        rows.append([float(x.replace("D", "E").replace("d", "e")) for x in parts])
    if not rows:
        raise ValueError("Pure-water table has no data")
    data = np.asarray(rows, dtype=float)
    if (not np.all(np.isfinite(data)) or np.any(np.diff(data[:, 0]) <= 0)
            or np.any(data[:, 1:] < 0)):
        raise ValueError("Pure-water table must be finite, ordered and nonnegative")
    if data[0, 0] > WAVES[0] or data[-1, 0] < WAVES[-1]:
        raise ValueError("Pure-water table does not cover all four bands")
    if not np.array_equal(data[:, 0], [440.0, 445.0, 490.0, 560.0, 665.0]):
        raise ValueError("Expected water-table support rows for four-band sampling")
    aw = np.interp(WAVES, data[:, 0], data[:, 1])
    bw = np.interp(WAVES, data[:, 0], data[:, 2])
    return WAVES.copy(), aw, 0.49997 * bw


def rrs_from_Rrs(Rrs: Any) -> np.ndarray:
    """English documentation is provided in README.md."""
    value = np.asarray(Rrs, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return value / (0.52 + 1.7 * value)


def Rrs_from_rrs(rrs: Any) -> np.ndarray:
    """English documentation is provided in README.md."""
    value = np.asarray(rrs, dtype=float)
    denominator = 1.0 - 1.7 * value
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = 0.52 * value / denominator
    return np.where(np.isfinite(value) & (denominator != 0) & np.isfinite(out),
                    out, np.nan)


def make_context(a: Any, bb: Any, solar_zenith: float,
                 deep_rrs_above: Any = None) -> dict[str, Any]:
    """English documentation is provided in README.md."""
    a = _vector4(a, "a")
    bb = _vector4(bb, "bb")
    solar = float(solar_zenith)
    if not math.isfinite(solar) or not 0 <= solar < 90:
        raise ValueError("solar_zenith must be in [0,90) degrees in air")
    kappa = a + bb
    # The model rationale and parameter policy are documented in README.md.
    if np.any(kappa == 0):
        raise ValueError("Zero a+bb is outside the real formula domain")
    u = bb / kappa
    arg_c, arg_b = 1.0 + 2.4 * u, 1.0 + 5.4 * u
    if np.any(arg_c < 0) or np.any(arg_b < 0):
        raise ValueError("Lee square-root domain invalid")
    # The model rationale and parameter policy are documented in README.md.
    # The model rationale and parameter policy are documented in README.md.
    mu_s = math.sqrt(1.0 - (math.sin(math.radians(solar)) / 1.34) ** 2)
    duc = 1.03 * np.sqrt(arg_c)
    dub = 1.04 * np.sqrt(arg_b)
    xc = (1.0 / mu_s + duc) * kappa
    xb = (1.0 / mu_s + dub) * kappa
    dp = (G0 + G1 * u) * u
    if not np.all(np.isfinite([xc, xb, dp])):
        raise ValueError("Nonfinite Lee context")
    context: dict[str, Any] = dict(a=a, bb=bb, u=u, kappa=kappa, duc=duc,
        dub=dub, xc=xc, xb=xb, rrs_dp=dp, solar_zenith=solar, n_water=1.34,
        cos_view_water=1.0, method=LEE_METHOD, rrs_dp_source="iop_u_polynomial")
    if deep_rrs_above is not None:
        deep = _vector4(deep_rrs_above, "deep_rrs_above")
        converted = rrs_from_Rrs(deep)
        context["rrs_deep_converted"] = converted
        context["rrs_dp_max_abs_difference"] = float(np.max(np.abs(dp - converted)))
        context["rrs_dp"] = converted
        context["rrs_dp_source"] = "measured_deep_Rrs_interface_transform"
    return context


def forward(z: Any, rho4: Any, context: dict[str, Any]) -> np.ndarray:
    """English documentation is provided in README.md."""
    rho = _vector4(rho4, "rho4")
    depth = np.asarray(z, dtype=float)[..., None]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # The model rationale and parameter policy are documented in README.md.
        # The model rationale and parameter policy are documented in README.md.
        water = context["rrs_dp"] * (-np.expm1(-context["xc"] * depth))
        bottom = (rho / np.pi) * np.exp(-context["xb"] * depth)
        below = water + bottom
    return Rrs_from_rrs(below)


def c4(observed, predicted):
    """English documentation is provided in README.md."""
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    if observed.shape != (4,) or predicted.shape[-1:] != (4,):
        raise ValueError("Exactly four bands are required")
    if not np.isfinite(observed).all() or np.any(observed == 0):
        raise ValueError("Original C4 requires four finite nonzero inputs")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # The model rationale and parameter policy are documented in README.md.
        value = np.mean(((predicted-observed)/observed)**2, axis=-1)
    return np.where(np.isfinite(value), value, np.inf)


def components(pair, fraction):
    """English documentation is provided in README.md."""
    result = dict.fromkeys(MATERIALS, 0.0)
    result[pair[0]] = float(fraction)
    result[pair[1]] = float(1-fraction)
    return result
