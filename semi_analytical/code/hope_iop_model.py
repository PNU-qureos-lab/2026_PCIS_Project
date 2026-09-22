"""English documentation is provided in README.md."""
from __future__ import annotations

from typing import Any
import itertools
import numpy as np

from model import WAVES, G0, G1, Rrs_from_rrs, _vector4

HOPE_METHOD = "HOPE-PGX-X550-S0.015-Y0.5-4band-nonnegative-SD-15th"
A0 = np.array([0.98902, 0.7558, 0.3433, 0.70225], dtype=float)
A1 = np.array([0.0018, 0.0256, 0.0659, 0.13615], dtype=float)
S = 0.015
Y = 0.5
# The model rationale and parameter policy are documented in README.md.
# The model rationale and parameter policy are documented in README.md.
SEED_FACTORS = (0.5, 1.0, 1.5)

HOPE_CONFIG = {
    "method": HOPE_METHOD,
    "bands_nm": WAVES.tolist(),
    "a0": A0.tolist(), "a1": A1.tolist(),
    "S_nm_inverse": S, "Y": Y,
    "P_reference_nm": 440.0, "G_reference_nm": 440.0, "X_reference_nm": 550.0,
    "parameter_bounds": {"P":[0.0,None], "G":[0.0,None], "X":[0.0,None]},
    "formula_domain": "P>0 for ln(P); G>=0; X>=0; nonzero denominators; finite real evaluation",
    "initial_P": "0.072 * (Rrs(440)/Rrs(550))**(-1.62)",
    "initial_G": "P_in", "initial_X": "30 * aw(640) * Rrs(640)",
    "seed_factors": list(SEED_FACTORS),
    "start_count": 27, "x_scale": "physical PGX coordinates; initial scales used for diagnostics only",
    "initialization_wavelengths": "only 443/490/560/665; 440 extrapolated from 443/490; 550 interpolated from 490/560; 640 interpolated from 560/665",
    "seed_bounds_policy": "formula starts unchanged; no projection or clipping",
    "solver": "nonnegative_pgx_feasible_steepest_descent",
    "gradient": "analytic_chain_rule; no finite-difference step",
    "objective": "mean(((Rrs_deep_model - Rrs_deep_observed)/Rrs_deep_observed)**2)",
    "g0": G0, "g1": G1, "interface_above_numerator": 0.52,
    "interface_above_denominator_coefficient": 1.7,
    "coefficient_sources": ["Lee 1998 Table 2", "Lee 1999 equations 16-19",
                            "Dekker 2011 HOPE equations 8-10"],
    "coefficient_interpolation": "443:440/450 nm linear;665:660/670 nm linear",
    "water_coefficients": "C:/HE60/data/H2OabDefaults_SEAwater.txt; bbw=0.49997*bw; resampled to four bands before initialization",
    "fit_deep_source": "PGX model only; no observed deep override",
    "HF_deep_source": "original observed deep Rrs transformed by existing interface",
    "selection": "converged, PGX nonnegative and finite real-formula minimum C_IOP; no extra absorption-sign filtering",
    "near_best_relative_cost_tolerance": 0.01,
    "near_best_absolute_cost_tolerance": 1e-12,
    "QAA_fallback": False,
    "jacobian_diagnostic": "singular values of residual Jacobian @ diag(x_scale)",
}


def sample_initialization_spectrum(wavelengths: Any, values: Any,
                                   target: float) -> tuple[float, dict[str, Any]]:
    """English documentation is provided in README.md."""
    w, v = np.asarray(wavelengths, dtype=float), np.asarray(values, dtype=float)
    if (w.shape != (4,) or not np.array_equal(w, WAVES) or v.shape != w.shape
            or not np.isfinite(w).all() or not np.isfinite(v).all()
            or np.any(np.diff(w) <= 0) or np.any(v <= 0)):
        raise ValueError('Initialization requires ONLY 443,490,560,665 nm with finite positive values')
    exact = np.flatnonzero(w == target)
    if len(exact):
        j = int(exact[0])
        return float(v[j]), dict(method='native', left_nm=float(w[j]), right_nm=float(w[j]), weight_right=0.0)
    if target < w[0]:
        if target != 440.0:
            raise ValueError('Only 440 nm extrapolation from 443/490 is allowed')
        i, j, method = 0, 1, 'linear_extrapolation'
    elif target > w[-1]:
        raise ValueError('Initialization wavelength exceeds available spectrum')
    else:
        j = int(np.searchsorted(w, target))
        i, method = j - 1, 'linear_interpolation'
    weight = float((target - w[i]) / (w[j] - w[i]))
    value = float(v[i] + weight * (v[j] - v[i]))
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Interpolated/extrapolated initialization input is not positive')
    return value, dict(method=method, left_nm=float(w[i]), right_nm=float(w[j]), weight_right=weight)


def formula_initialization(initialization: dict[str, Any]) -> dict[str, Any]:
    """English documentation is provided in README.md."""
    waves = initialization['wavelengths_nm']
    values, provenance = {}, {}
    for label, source, target in (
            ('Rrs_440', 'Rrs', 440.0), ('Rrs_550', 'Rrs', 550.0),
            ('Rrs_640', 'Rrs', 640.0), ('aw_640', 'aw', 640.0)):
        values[label], details = sample_initialization_spectrum(waves, initialization[source], target)
        provenance.update({f'{label}_{key}': value for key, value in details.items()})
    p = float(0.072 * (values['Rrs_440'] / values['Rrs_550']) ** -1.62)
    x = float(30.0 * values['aw_640'] * values['Rrs_640'])
    base = np.array([p, p, x])
    if not np.isfinite(base).all() or np.any(base <= 0):
        raise ValueError('Formula P/G/X baseline must be finite and strictly positive')
    factors = list(itertools.product(SEED_FACTORS, repeat=3))
    return dict(base=base, factors=factors, starts=[base * f for f in factors],
                metadata=dict(P_in=p, G_in=p, X_in=x, **values, **provenance))


def water_coefficients(water_aw4: Any, water_bbw4: Any) -> tuple[np.ndarray, np.ndarray]:
    """English documentation is provided in README.md."""
    aw = _vector4(water_aw4, "water_aw4")
    bbw = _vector4(water_bbw4, "water_bbw4")
    if np.any(aw < 0) or np.any(bbw < 0):
        raise ValueError("Pure-water aw and bbw must be nonnegative")
    return aw, bbw


def spectrum(pgx: Any, water_aw4: Any, water_bbw4: Any) -> dict[str, np.ndarray]:
    """English documentation is provided in README.md."""
    params = np.asarray(pgx, dtype=float)
    if params.shape != (3,) or not np.all(np.isfinite(params)):
        raise ValueError("P,G,X must contain three finite values")
    p, g, x = params
    if np.any(params < 0):
        raise ValueError("P,G,X must be nonnegative")
    if p <= 0:
        raise ValueError("ln(P) requires P>0; no additional physical lower bound")
    aw, bbw = water_coefficients(water_aw4, water_bbw4)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        a_phi = (A0 + A1 * np.log(p)) * p
        a_g = g * np.exp(-S * (WAVES - 440.0))
        bbp = x * (550.0 / WAVES) ** Y
        a = aw + a_phi + a_g
        bb = bbw + bbp
        kappa = a + bb
        u = bb / kappa
        rrs_model = (G0 + G1 * u) * u
        Rrs_model = Rrs_from_rrs(rrs_model)
    result = dict(a_phi=a_phi, a_g=a_g, bbp=bbp, a=a, bb=bb, u=u,
                  rrs_deep_model=rrs_model, Rrs_deep_model=Rrs_model)
    if not all(np.all(np.isfinite(v)) for v in result.values()):
        raise ValueError("Nonfinite PGX spectrum or deep reflection")
    if np.any(kappa == 0) or np.any(1.0 - 1.7 * rrs_model == 0):
        raise ValueError("Zero denominator outside the real formula domain")
    return result


def relative_residual(pgx: Any, observed: Any, aw4: Any, bbw4: Any) -> np.ndarray:
    """English documentation is provided in README.md."""
    raw = _vector4(observed, "deep_Rrs_observed")
    if np.any(raw <= 0):
        raise ValueError("Observed deep Rrs must be strictly positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        residual = (spectrum(pgx, aw4, bbw4)["Rrs_deep_model"] - raw) / raw
    if not np.all(np.isfinite(residual)):
        raise ValueError("Nonfinite relative residual")
    return residual
