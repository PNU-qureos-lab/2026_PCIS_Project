"""2단계: 깊은 물 조건별 HOPE P·G·X를 추정한 뒤 기존 H/F 문맥을 준비한다.

입력은 깊은 물 Rrs 네 밴드와 순수 물 aw/bbw, 태양 천정각이다. 정답 총 IOP,
수심, 바닥 비율을 읽지 않는다. QAA 계산·초기값·실패 대체도 사용하지 않는다.
깊은 물 조건 하나당 관측 초기식의 0.5/1.0/1.5배 27개 시작점에서 최급강하법을 수행하고,
수렴하고 실수 수식이 정의되는 후보 중 상대 제곱오차 평균이 최소인 해를 고른다.
P/G/X>=0를 적용하며 P=0은 ln(P)의 정의역에서 제외한다. 양수 P_MIN과
흡수/후방산란 성분의 별도 비음수 제약은 적용하지 않는다.

IOP 피팅 안에서는 항상 P·G·X에서 계산한 모델 깊은 물 Rrs를 사용한다.
선정 후 make_context에는 관측 깊은 물 Rrs를 넘겨 기존 H/F의 물기둥 항을 유지한다.
이 둘의 차이를 iops.csv에 함께 저장한다. H/F를 돌리는 동안 a·bb는 고정된다.

derive_contexts의 반환값은 contexts(성공 조건별 D2 사전), iops(파장별 행),
parameters(조건별 상태/파라미터), multistart(시작점별 모든 결과), failures(실패표).
초기식은 허용 네 밴드만 보간·외삽하여 계산한다. 모든 시작점을 원값 그대로 사용한다.
CSV/JSON 기록은 run_all.py가 맡는다. 이 파일은 입력 파일과 정답 파일을 읽지 않는다.
"""
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
    """추정값이 실제 0 경계에 놓였는지 기록한다. 임의의 근접 허용차는 없다.

    P는 로그 때문에 유효해에서 0일 수 없지만 CSV 세 변수 형식은 일관되게 둔다.
    """
    flags = np.asarray(pgx) == 0.0
    return dict(P_at_lower_bound=bool(flags[0]), G_at_lower_bound=bool(flags[1]),
                X_at_lower_bound=bool(flags[2]), any_lower_bound=bool(np.any(flags)))


def _jacobian_diagnostics(jac: Any, x_scale: np.ndarray) -> dict[str, Any]:
    """잔차 Jacobian의 열을 변수 스케일로 조정해 특이값과 조건수를 구한다.

    J_scaled=J*diag(P_in,G_in,X_in)는 초기식 기준 변화에 대한 잔차 민감도다.
    매우 작은 최소 특이값/큰 조건수는 매개변수 분리가 약할 가능성을 뜻한다.
    이는 오차 막대나 관측 노이즈의 추정이 아니며 후보 선정에도 쓰지 않는다.
    """
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
    """성공·실패 모두 같은 열을 쓰도록 시작점 진단 행의 기본값을 준비한다."""
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
    """깊은 물 한 조건의 P·G·X를 27개 시작점으로 추정하고 전체 이력을 반환한다.

    목적함수 C_IOP는 네 밴드 상대오차 제곱의 평균이다. 깊은 물의 관측 Rrs와
    PGX로 계산한 모델 Rrs를 모두 수면 위 단위에서 비교한다. 참 a/bb는 쓰지 않는다.
    초기값은 P_in=0.072*(Rrs440/Rrs550)^(-1.62), G_in=P_in,
    X_in=30*aw640*Rrs640이며 각 변수에 0.5,1,1.5를 곱한 27개를 모두 실행한다.

    solve_pgx는 원래 PGX 좌표의 음의 해석적 기울기로 이동한다. G/X=0에서는
    음수로 나가려는 성분만 정지시키며 나머지 허용 방향은 계속 최적화한다.
    보폭은 비음수 제약을 만족하는 직선 구간에서 C_IOP를 최소화하여 정한다.
    이전 보폭의 배수, Armijo 상수, 정규화 좌표의 전처리는 없다.
    경계해는 원래 기울기 전체가 아닌 허용 방향의 기울기(KKT 잔차)로 수렴 판정한다.
    TRF나 다른 최적화로 대신 풀지 않는다. 종료 상태가 실패인 후보는 성공 후보로
    바꿔 쓰지 않는다. 27개 중 수렴하고 수식이 정의되는 최소 C_IOP를 선택한다.
    다중 시작점은 국소해 누락을 줄이는 장치이며 전역 최솟값의 증명은 아니다.
    저장 이력은 첫 20회/100회 간격/최종이며 실제 전체 반복 횟수도 별도로 기록한다.
    """
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
                # 아래 검산은 원래 NumPy 분광식으로 한다. 최급강하 내부 계산과 독립 대조다.
                # solver nfev는 비용/해석적 기울기의 평가 횟수이며 이 검산은 포함하지 않는다.
                checked = relative_residual(opt.x, raw, aw, bbw)
                cost = float(np.mean(checked ** 2))
                spec = spectrum(opt.x, aw, bbw)
                signs_ok = all(np.all(spec[k] >= 0) for k in ('a_phi','a_g','bbp','a','bb'))
                # a_phi 등 성분의 부호는 보고용이다. PGX 자체의 비음수 조건은
                # parameter_feasible로 따로 검사하며, 두 조건을 혼동하지 않는다.
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
        # 비슷한 비용의 시작점 분산은 진단이다. 1e-12는 0 근처 비용의 수치 비교용이다.
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
    """깊은 물 조건마다 한 번 피팅하고, H/F 문맥과 저장할 진단 행들을 반환한다.

    입력은 rrs(수면 위 Rrs), a_water, bb_water, solar_zenith와 초기식용 네 밴드다.
    rrs를 변환해 받는다고 오해하지 않아야 한다. 실제 정답 IOP 등 추가 입력 필드는
    읽지 않는다. 실패 조건은 contexts에서 제외하고 failures에 기록하므로 호출자는
    해당 water_key의 얕은 물 사례를 실패로 남기고 다른 조건을 계속 처리할 수 있다.
    """
    output: dict[str, Any] = dict(contexts={}, iops=[], parameters=[], multistart=[], trajectory=[], failures=[])
    for key, deep in sorted(deep_inputs.items()):
        try:
            # 태양각 오류는 피팅 전에 검사한다. 실패할 문맥에 27회 최적화를 낭비하지 않는다.
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
            # 이 시점에만 관측 깊은 물 Rrs를 H/F 깊은 물 항에 넣는다.
            # P·G·X 피팅의 순방향 예측에는 make_context를 호출하지 않는다.
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
            # 피팅 성공 이후 태양각/문맥 연결 실패도 조건 단위로 기록한다.
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
