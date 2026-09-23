"""English documentation is provided in README.md."""
from __future__ import annotations

import math
import time
from typing import Any
import numpy as np
from steepest_descent import solve_pgx, SOLVER_CONFIG, TRACE_COLUMNS

from model import WAVES, _vector4, make_context
from hope_iop_model import (
    HOPE_METHOD, HOPE_CONFIG,
    spectrum, water_coefficients, relative_residual, formula_initialization,
)

HOPE_CONFIG.update(SOLVER_CONFIG)


def _boundary_flags(pgx: np.ndarray) -> dict[str, bool]:
    """English documentation is provided in README.md."""
    flags = np.asarray(pgx) == 0.0
    return dict(P_at_lower_bound=bool(flags[0]), G_at_lower_bound=bool(flags[1]),
                X_at_lower_bound=bool(flags[2]), any_lower_bound=bool(np.any(flags)))


def _jacobian_diagnostics(jac: Any, x_scale: np.ndarray) -> dict[str, Any]:
    """English documentation is provided in README.md."""
    output = dict(jac_singular_1=math.nan, jac_singular_2=math.nan,
                  jac_singular_3=math.nan, jac_condition=math.nan, jac_rank=0)
    matrix = np.asarray(jac, dtype=float)
    if matrix.shape != (4, 3) or not np.all(np.isfinite(matrix)):
        return output
    values = np.linalg.svd(matrix * x_scale[np.newaxis, :], compute_uv=False)
    tol = max(matrix.shape) * np.finfo(float).eps * values[0]
    condition = float(values[0] / values[-1]) if values[-1] > 0 else math.inf
    output.update(jac_singular_1=float(values[0]), jac_singular_2=float(values[1]),
                  jac_singular_3=float(values[2]), jac_condition=condition,
                  jac_rank=int(np.sum(values > tol)))
    return output


def _start_row(start_id: int, start: tuple[float, ...]) -> dict[str, Any]:
    """English documentation is provided in README.md."""
    return dict(start_id=start_id, P_start=start[0], G_start=start[1], X_start=start[2],
        P=math.nan, G=math.nan, X=math.nan, C_IOP=math.inf,
        solver_success=False, model_defined=False, parameter_feasible=False,
        physical_valid=False, accepted=False, selected=False,
        G_negative=False, X_negative=False, physical_components_negative=False,
        solver_status=-99, solver_message="not_started", nfev=0, njev=0, iterations=0,
        residual_calls=0, invalid_residual_calls=0, optimality=math.nan,
        projected_gradient_inf=math.nan, raw_gradient_inf=math.nan,
        P_at_lower_bound=False, G_at_lower_bound=False, X_at_lower_bound=False,
        any_lower_bound=False, jac_singular_1=math.nan, jac_singular_2=math.nan,
        jac_singular_3=math.nan, jac_condition=math.nan, jac_rank=0,
        failure_reason="", elapsed_seconds=0.0, hope_method=HOPE_METHOD)


def fit_deep_iops(deep_rrs_above: Any, water_aw4: Any,
                  water_bbw4: Any, initialization_spectrum: Any = None) -> dict[str, Any]:
    """English documentation is provided in README.md."""
    t_all = time.perf_counter()
    rows: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    summary: dict[str, Any] = dict(
        success=False, status="uncomputed", failure_reason="", hope_method=HOPE_METHOD,
        P=math.nan, G=math.nan, X=math.nan, S=HOPE_CONFIG["S_nm_inverse"], Y=HOPE_CONFIG["Y"],
        P_min=0.0, G_min=0.0, X_min=0.0, P_zero_defined=False,
        X_reference_nm=550.0, C_IOP=math.inf, selected_start_id=0,
        global_minimum_certified=False, physical_bounds_applied=True,
        start_count=27, attempted_start_count=0, converged_start_count=0,
        physical_start_count=0, accepted_start_count=0, residual_calls_total=0,
        invalid_residual_calls_total=0, nfev_total=0, njev_total=0,
        P_at_lower_bound=False, G_at_lower_bound=False, X_at_lower_bound=False,
        any_lower_bound=False, jac_singular_1=math.nan, jac_singular_2=math.nan,
        jac_singular_3=math.nan, jac_condition=math.nan, jac_rank=0,
        near_best_start_count=0, near_best_P_span=math.nan, near_best_G_span=math.nan,
        near_best_X_span=math.nan, elapsed_seconds=0.0,
    )
    result = dict(parameters=summary, multistart=rows, trajectory=trajectory, spectrum=None)
    try:
        raw = _vector4(deep_rrs_above, "deep_Rrs_observed")
        if np.any(raw <= 0):
            raise ValueError("Observed deep Rrs must contain four strictly positive values")
        aw, bbw = water_coefficients(water_aw4, water_bbw4)
        if initialization_spectrum is None:
            raise ValueError('Four-band initialization spectrum is required; no fixed PGX fallback')
        initialization = formula_initialization(initialization_spectrum)
        summary.update(initialization['metadata'])
    except (KeyError, ValueError, TypeError) as exc:
        summary.update(status="invalid_input", failure_reason=str(exc),
                       elapsed_seconds=time.perf_counter() - t_all)
        return result

    starts = initialization['starts']
    x_scale = initialization['base']
    for start_id, start in enumerate(starts, 1):
        t_start = time.perf_counter()
        formula_start = np.asarray(start).copy()
        start = formula_start.copy()
        row = _start_row(start_id, start)
        row.update({f'{name}_formula_start': float(value)
                    for name, value in zip(('P', 'G', 'X'), formula_start)})
        row.update(P_start_projected=False,
                   P_projection_delta=float(start[0] - formula_start[0]))
        row.update(initialization['metadata'])
        row.update({f'{name}_factor': float(value) for name, value
                    in zip(('P', 'G', 'X'), initialization['factors'][start_id - 1])})
        rows.append(row)
        try:
            opt = solve_pgx(start, raw, aw, bbw, x_scale)
            row.update(P=float(opt.x[0]), G=float(opt.x[1]), X=float(opt.x[2]),
                solver_success=bool(opt.success), solver_status=int(opt.status),
                solver_message=str(opt.message), nfev=int(opt.nfev), njev=int(opt.njev),
                iterations=int(opt.nit), residual_calls=int(opt.nfev),
                invalid_residual_calls=int(opt.invalid_evaluations),
                optimality=float(opt.optimality),
                projected_gradient_inf=float(opt.projected_gradient_inf),
                raw_gradient_inf=float(opt.raw_gradient_inf),
                parameter_feasible=bool(np.all(np.isfinite(opt.x)) and np.all(opt.x >= 0) and opt.x[0] > 0),
                **_boundary_flags(opt.x),
                **_jacobian_diagnostics(opt.jac, x_scale))
            row.update({f'{name}_solver_initial': float(value)
                        for name, value in zip(('P','G','X'),opt.trace[0,5:8])})
            for values in opt.trace:
                saved = dict(zip(TRACE_COLUMNS, map(float,values)))
                saved.update(start_id=start_id, iteration=int(saved['iteration']),
                             nfev=int(saved['nfev']), line_evaluations=int(saved['line_evaluations']),
                             event='initial' if saved['iteration']==0 else 'accepted_move')
                trajectory.append(saved)
            try:
                # The model rationale and parameter policy are documented in README.md.
                # The model rationale and parameter policy are documented in README.md.
                checked = relative_residual(opt.x, raw, aw, bbw)
                cost = float(np.mean(checked ** 2))
                spec = spectrum(opt.x, aw, bbw)
                signs_ok = all(np.all(spec[k] >= 0) for k in ('a_phi','a_g','bbp','a','bb'))
                # The model rationale and parameter policy are documented in README.md.
                # The model rationale and parameter policy are documented in README.md.
                row.update(C_IOP=cost, model_defined=bool(math.isfinite(cost)),
                    physical_valid=bool(signs_ok), G_negative=bool(opt.x[1] < 0),
                    X_negative=bool(opt.x[2] < 0), physical_components_negative=not bool(signs_ok))
                if not row["model_defined"]:
                    row["failure_reason"] = "nonfinite_final_cost"
            except (ValueError, OverflowError, FloatingPointError) as exc:
                row["failure_reason"] = "final_formula_check: " + str(exc)
            row["accepted"] = row["solver_success"] and row["model_defined"] and row["parameter_feasible"]
            if not row["parameter_feasible"]:
                row["failure_reason"] = "PGX_nonnegative_or_log_domain_violation"
            if not row["solver_success"]:
                row["failure_reason"] = "solver_not_converged: " + str(opt.message)
        except (ValueError, RuntimeError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as exc:
            row.update(solver_message=str(exc), failure_reason="solver_exception: " + str(exc))
        row.update(elapsed_seconds=time.perf_counter() - t_start)

    summary.update(attempted_start_count=len(rows),
        projected_start_count=sum(r['P_start_projected'] for r in rows),
        unique_feasible_start_count=len({(r['P_start'], r['G_start'], r['X_start']) for r in rows}),
        converged_start_count=sum(r["solver_success"] for r in rows),
        physical_start_count=sum(r["physical_valid"] for r in rows),
        accepted_start_count=sum(r["accepted"] for r in rows),
        residual_calls_total=sum(r["residual_calls"] for r in rows),
        invalid_residual_calls_total=sum(r["invalid_residual_calls"] for r in rows),
        nfev_total=sum(r["nfev"] for r in rows), njev_total=sum(r["njev"] for r in rows))
    accepted = [r for r in rows if r["accepted"]]
    candidates = accepted or [r for r in rows if r["model_defined"] and r["parameter_feasible"]]
    best = min(candidates, key=lambda r: (r["C_IOP"], r["start_id"])) if candidates else None
    if best is not None:
        names = ["P", "G", "X", "C_IOP", "P_at_lower_bound", "G_at_lower_bound",
            "X_at_lower_bound", "any_lower_bound", "jac_singular_1", "jac_singular_2",
            "jac_singular_3", "jac_condition", "jac_rank"]
        summary.update({name: best[name] for name in names})
    if accepted:
        best["selected"] = True
        summary.update(success=True, status="ok", selected_start_id=best["start_id"])
        result["spectrum"] = spectrum([best["P"], best["G"], best["X"]], aw, bbw)
        # The model rationale and parameter policy are documented in README.md.
        near_tolerance = max(HOPE_CONFIG["near_best_absolute_cost_tolerance"],
                             HOPE_CONFIG["near_best_relative_cost_tolerance"] * best["C_IOP"])
        near = [r for r in accepted if r["C_IOP"] <= best["C_IOP"] + near_tolerance]
        summary["near_best_start_count"] = len(near)
        for name in ("P", "G", "X"):
            summary[f"near_best_{name}_span"] = max(r[name] for r in near) - min(r[name] for r in near)
    else:
        summary.update(status="no_accepted_start", failure_reason="No converged and real-formula-defined start")
    summary["elapsed_seconds"] = time.perf_counter() - t_all
    return result


def derive_contexts(deep_inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """English documentation is provided in README.md."""
    output: dict[str, Any] = dict(contexts={}, iops=[], parameters=[], multistart=[], trajectory=[], failures=[])
    for key, deep in sorted(deep_inputs.items()):
        try:
            # The model rationale and parameter policy are documented in README.md.
            solar = float(deep["solar_zenith"])
            if not math.isfinite(solar) or not 0 <= solar < 90:
                raise ValueError("solar_zenith must be in [0,90) degrees in air")
            fitted = fit_deep_iops(deep["rrs"], deep["a_water"], deep["bb_water"],
                                   deep.get('initialization_spectrum'))
            print(f"PGX {key}: {fitted['parameters']['accepted_start_count']}/27 converged; "
                  f"C_IOP={fitted['parameters']['C_IOP']:.10g}", flush=True)
            parameters = dict(water_key=key, solar_zenith=solar,
                              **fitted["parameters"])
            output["parameters"].append(parameters)
            output["multistart"].extend(dict(water_key=key, **row) for row in fitted["multistart"])
            output['trajectory'].extend(dict(water_key=key, **row) for row in fitted['trajectory'])
            if not parameters["success"]:
                output["failures"].append(dict(water_key=key, stage="hope_pgx",
                    status=parameters["status"], reason=parameters["failure_reason"]))
                continue
            spec = fitted["spectrum"]
            # The model rationale and parameter policy are documented in README.md.
            # The model rationale and parameter policy are documented in README.md.
            ctx = make_context(spec["a"], spec["bb"], solar, deep["rrs"])
            output["contexts"][key] = ctx
            for j, wavelength in enumerate(WAVES):
                output["iops"].append(dict(water_key=key, wavelength=int(wavelength),
                    solar_zenith=solar,
                    aw=float(deep["a_water"][j]), bbw=float(deep["bb_water"][j]),
                    a_hat=float(spec["a"][j]), bb_hat=float(spec["bb"][j]),
                    a_phi=float(spec["a_phi"][j]), a_g=float(spec["a_g"][j]), bbp=float(spec["bbp"][j]),
                    P=parameters["P"], G=parameters["G"], X=parameters["X"],
                    S=parameters["S"], Y=parameters["Y"], C_IOP=parameters["C_IOP"],
                    u=float(spec["u"][j]),
                    Rrs_deep_observed=float(deep["rrs"][j]),
                    Rrs_deep_model=float(spec["Rrs_deep_model"][j]),
                    rrs_deep_model=float(spec["rrs_deep_model"][j]),
                    relative_residual=float((spec["Rrs_deep_model"][j]-deep["rrs"][j])/deep["rrs"][j]),
                    rrs_dp=float(ctx["rrs_dp"][j]),
                    rrs_dp_minus_model=float(ctx["rrs_dp"][j]-spec["rrs_deep_model"][j]),
                    xc=float(ctx["xc"][j]), xb=float(ctx["xb"][j]),
                    rrs_dp_source=ctx["rrs_dp_source"], hope_method=HOPE_METHOD))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            # The model rationale and parameter policy are documented in README.md.
            if output["parameters"] and output["parameters"][-1]["water_key"] == key:
                output["parameters"][-1].update(success=False, status="context_or_input_failure",
                                                failure_reason=str(exc))
            else:
                output["parameters"].append(dict(water_key=key, solar_zenith=math.nan,
                    **fit_deep_iops([0, 0, 0, 0], [0]*4, [0]*4)["parameters"]))
                output["parameters"][-1].update(status="context_or_input_failure", failure_reason=str(exc))
            output["contexts"].pop(key, None)
            output["iops"] = [row for row in output["iops"] if row["water_key"] != key]
            output["failures"].append(dict(water_key=key, stage="hope_pgx",
                status="context_or_input_failure", reason=str(exc)))
    return output
