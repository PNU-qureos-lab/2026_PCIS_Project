"""심수역 기준 반사도로 CHL·TSM과 고유광학특성(IOP)을 추정합니다.

수면 위 Rrs를 수면 아래 rrs로 변환하고, 주어진 g0·g1으로 u=bb/(a+bb)를
역산합니다. 내장 성분 스펙트럼을 이용한 모형의 u가 이 값에 가장 잘 맞도록
엽록소 농도 CHL(mg/m³)과 총 부유물질 농도 TSM(g/m³)을 탐색합니다.
g0·g1은 사전 산정된 입력값이며, 이 과정에서 다시 추정하지 않습니다.

CHL 7개 × TSM 7개 × 초기 이동 간격 2개로 총 98개 초기 조건을 평가합니다.
농도는 각 단위의 0.01을 한 칸으로 하는 정수 격자에서 탐색하며 하한은 0입니다.
오차가 줄면 간격을 두 배로 늘리고, 줄지 않으면 정수 절반으로 줄입니다.
최소 간격의 유효한 8방향 이웃에서 개선이 없을 때 종료하며, 사용자는 계산을
중지할 수 있습니다. 최종 IOP는 같은 분석에 속한 모든 수심 픽셀에 공통으로 쓰입니다.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Callable

import numpy as np

from .engine import _vector, make_context


DIRECTIONS = tuple((i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j)
# 정수 격자 한 칸(tick)은 CHL 0.01 mg/m³ 또는 TSM 0.01 g/m³이다.
# 아래 정수는 실제 농도 시작값을 100배한 값이다.
# CHL: 0.1, 0.5, 1, 1.5, 2, 2.5, 3 mg/m³; TSM: 0.1, 0.5, 1, 1.5, 2, 3, 5 g/m³.
# 시작값 목록이며, 추정 농도의 하한·상한을 뜻하지 않는다.
CHL_LEVEL_TICKS = (10, 50, 100, 150, 200, 250, 300)
TSM_LEVEL_TICKS = (10, 50, 100, 150, 200, 300, 500)
INITIAL_STEP_TICKS = (2, 10)
# 49개 농도 조합을 초기 이동 간격 0.02와 0.1에서 각각 탐색한다.
CONFIGURATIONS = tuple(
    itertools.product(itertools.product(CHL_LEVEL_TICKS, TSM_LEVEL_TICKS), INITIAL_STEP_TICKS)
)


def _check_cancelled(cancelled):
    """사용자 중지는 정상 수렴이 아니므로 InterruptedError로 호출부에 알립니다."""
    if cancelled is not None and cancelled():
        raise InterruptedError("사용자가 IOP 계산을 중지했습니다.")


def _validated_rows(source: dict, name: str, columns: int) -> list:
    """내장 스펙트럼의 열 수, 파장 순서, 유한성 및 성분값의 비음수성을 검사합니다."""
    try:
        rows = source[name]
        values = np.asarray(rows, dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"IOP 자료의 {name} 표를 읽을 수 없습니다.") from exc
    if (
        values.ndim != 2
        or len(values) < 2
        or values.shape[1] < columns
        or not np.isfinite(values).all()
        or np.any(np.diff(values[:, 0]) <= 0)
    ):
        raise ValueError(f"IOP 자료의 {name} 표는 유한한 값과 증가하는 파장을 가져야 합니다.")
    if np.any(values[:, 0] <= 0) or np.any(values[:, 1:columns] < 0):
        raise ValueError(f"IOP 자료의 {name} 표에 음수 성분값 또는 잘못된 파장이 있습니다.")
    # 성분식 전체에서 Python float를 사용해 연산 순서를 일정하게 유지합니다.
    return [[float(value) for value in row] for row in rows]


def interp(rows, wavelength, column=1):
    """파장 양옆의 두 자료를 선형 보간합니다. 표의 범위 밖으로 외삽하지 않습니다."""
    if not rows[0][0] <= wavelength <= rows[-1][0]:
        raise ValueError(f"IOP 스펙트럼이 {wavelength:g} nm를 포함하지 않습니다.")
    for i, row in enumerate(rows):
        if row[0] == wavelength:
            return row[column]
        if row[0] > wavelength:
            previous = rows[i - 1]
            return previous[column] + (row[column] - previous[column]) * (
                wavelength - previous[0]
            ) / (row[0] - previous[0])
    raise ValueError("IOP 스펙트럼을 보간할 수 없습니다.")


class ComponentIOP:
    """내장 성분식으로 CHL·TSM에 따른 흡수계수 a, 후방산란계수 bb, u를 계산합니다.

    고정 모델 기준파장 440/550 nm는 성분식의 상수입니다. 추가 관측 밴드를 읽는 것이
    아닙니다. 각 센서의 4개/6개 파장에서만 목적함수를 평가합니다.
    """

    def __init__(self, source: dict, waves: Any):
        self.waves = tuple(float(w) for w in _vector(waves, "waves"))
        if any(w <= 0 for w in self.waves) or any(
            a >= b for a, b in zip(self.waves, self.waves[1:])
        ):
            raise ValueError("IOP 파장은 양수이며 오름차순이어야 합니다.")
        source = {
            name: _validated_rows(source, name, columns)
            for name, columns in (("water", 3), ("chl", 2), ("mineral_a", 2), ("mineral_b", 2))
        }
        self.source = source
        ac440 = interp(source["chl"], 440)
        self.coefficients = []
        for w in self.waves:
            self.coefficients.append(
                (
                    interp(source["water"], w),
                    interp(source["water"], w, 2),
                    interp(source["chl"], w),
                    0.2 * ac440 * math.exp(-0.014 * (w - 440)),
                    interp(source["mineral_a"], w),
                    interp(source["mineral_b"], w),
                )
            )

    def forward(self, point: tuple[int, int]) -> list[dict]:
        """정수 농도 좌표 (CHL×100, TSM×100)로 성분별 IOP와 합계를 반환합니다.

        CHL은 mg/m³, TSM은 g/m³이며, a·b·bb의 단위는 m^-1입니다.
        흡수 a: 순수해수 + Chl×a_chl + Chl×a_cdom + TSM×a_mineral.
        산란 b의 Chl 성분: Chl×Chl^0.62×550/파장.
        bb: 0.49997×b_water + 0.01833×b_Chl + 0.01833×b_mineral.
        성분 합에는 누적 반올림 오차를 줄이는 math.fsum을 사용합니다.
        """
        if len(point) != 2 or min(point) < 0:
            raise ValueError("Chl/TSM 농도는 음수가 될 수 없습니다.")
        C, T = point[0] / 100, point[1] / 100
        rows = []
        for w, (aw, bw, ac, ad, am, bm) in zip(self.waves, self.coefficients):
            a_components = (aw, C * ac, C * ad, T * am)
            b_components = (bw, C * C**0.62 * 550 / w, 0.0, T * bm)
            bb_components = (
                0.49997 * bw,
                0.01833 * b_components[1],
                0.0,
                0.01833 * b_components[3],
            )
            a, bb = math.fsum(a_components), math.fsum(bb_components)
            if not math.isfinite(a) or not math.isfinite(bb) or not math.isfinite(a + bb):
                raise ArithmeticError(f"Chl={C}, TSM={T}에서 IOP 수치 범위를 초과했습니다.")
            if a + bb == 0:
                raise ArithmeticError("IOP의 a+bb가 0이므로 u를 계산할 수 없습니다.")
            rows.append(
                dict(
                    wavelength_nm=w,
                    a=a,
                    bb=bb,
                    u=bb / (a + bb),
                    a_components=a_components,
                    b_components=b_components,
                    bb_components=bb_components,
                )
            )
        return rows

    def u(self, point):
        """목적함수에 필요한 센서 밴드별 u만 반환합니다."""
        return tuple(row["u"] for row in self.forward(point))


class Objective:
    """밴드별 u 차이의 무가중 제곱합과 같은 농도점의 계산 캐시입니다."""

    def __init__(self, model: ComponentIOP, target, cancelled=None):
        self.model, self.target, self.cancelled = model, tuple(target), cancelled
        self.cache = {}
        self.best_point, self.best_sse = None, math.inf

    def __call__(self, point):
        _check_cancelled(self.cancelled)
        if point not in self.cache:
            value = math.fsum((u - t) ** 2 for u, t in zip(self.model.u(point), self.target))
            if not math.isfinite(value):
                raise ArithmeticError(f"{point}의 IOP 목적함수가 유한하지 않습니다.")
            self.cache[point] = value
            if value < self.best_sse:
                self.best_point, self.best_sse = point, value
        return self.cache[point]


@dataclass
class SearchResult:
    """한 시작점에서 끝난 농도와 SSE, 반복·이동 횟수입니다."""

    point: tuple[int, int]
    sse: float
    iterations: int
    moves: int


def pattern_search(objective, start, initial_step=10, on_iteration=None):
    """비음수 농도 격자에서 8방향 이웃을 비교하며 이동 간격을 조정합니다.

    개선은 엄격한 '<'만 사용합니다. 동점이면 농도 좌표 튜플 순서를 사용합니다.
    한 칸은 CHL 0.01 mg/m³ 또는 TSM 0.01 g/m³이며, 최소 간격에서 유효한
    이웃 모두 개선이 없을 때 종료합니다. 반복 횟수의 상한은 두지 않습니다.
    """
    if min(start) < 0 or initial_step < 1 or not isinstance(initial_step, int):
        raise ValueError("IOP 시작점 또는 초기 이동 간격이 잘못되었습니다.")
    point, step = tuple(start), initial_step
    current = objective(point)
    iterations, moves = 0, 0
    while True:
        iterations += 1
        options = []
        for dc, dt in DIRECTIONS:
            neighbor = (point[0] + dc * step, point[1] + dt * step)
            if min(neighbor) >= 0:
                options.append((objective(neighbor), neighbor))
        candidate_score, candidate = min(options)
        used_step = step
        if candidate_score < current:
            point, current = candidate, candidate_score
            moves += 1
            step *= 2
            action = "move_and_double_step"
        elif step > 1:
            step = max(1, step // 2)
            action = "halve_step"
        else:
            action = "finished_no_improving_0.01_neighbor"
        if on_iteration is not None:
            on_iteration(
                dict(
                    iteration=iterations,
                    point_ticks=list(point),
                    Chl_mg_m3=point[0] / 100,
                    TSM_g_m3=point[1] / 100,
                    sse=current,
                    tested_step=used_step / 100,
                    next_step=step / 100,
                    action=action,
                )
            )
        if action == "finished_no_improving_0.01_neighbor":
            return SearchResult(point, current, iterations, moves)


def target_from_rrs(deep_rrs, g0, g1):
    """주어진 g0·g1을 사용해 심수역의 수면 위 Rrs에서 목표 u를 구합니다.

    rrs=Rrs/(0.52+1.7*Rrs), rrs=g0*u+g1*u² 입니다.
    양의 해는 소거 오차를 줄인 2*rrs/(g0+sqrt(g0²+4*g1*rrs))로 계산합니다.
    여기서 계수는 입력값이며, 계수를 찾거나 최적화하는 연산은 없습니다.
    """
    raw = _vector(deep_rrs, "deep_rrs")
    g0, g1 = _vector(g0, "g0", len(raw)), _vector(g1, "g1", len(raw))
    if np.any(raw <= 0):
        raise ValueError("깊은 물 Rrs는 모든 밴드에서 유한한 양수여야 합니다.")
    converted = tuple(float(r) / (0.52 + 1.7 * float(r)) for r in raw)
    target = []
    for r, c0, c1 in zip(converted, g0, g1):
        discriminant = float(c0) ** 2 + 4 * float(c1) * r
        if discriminant < 0 or not math.isfinite(discriminant):
            raise ValueError("선택한 계수와 깊은 물 Rrs의 u 이차식에 실수해가 없습니다.")
        denominator = float(c0) + math.sqrt(discriminant)
        if denominator == 0:
            raise ValueError("선택한 계수의 u 역산 분모가 0입니다.")
        target.append(2 * r / denominator)
    if not all(0 < u < 1 for u in target):
        raise ValueError("깊은 물 Rrs에서 계산한 u가 유효 범위 0<u<1을 벗어났습니다.")
    return converted, tuple(target)


def fit_iop(
    deep_rrs: Any,
    solar_zenith: float,
    waves: Any,
    g0: Any,
    g1: Any,
    iop_source: dict,
    progress: Callable[[dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """심수역 한 조건의 IOP를 98개 초기 조건에서 추정하고 수심 모형 입력을 만듭니다.

    반환값 context는 같은 영상의 모든 얕은 물 픽셀에 공통으로 사용합니다.
    parameters는 Chl/TSM과 탐색 진단, band_rows는 밴드별 IOP·Rrs·계수 기록,
    multistart는 초기 조건별 98개 탐색 결과, neighbors는 최종 농도에서 각 축의
    농도 단위로 0.01만큼 떨어진 이웃점의 검증 결과입니다.
    progress는 탐색 완료 때 {'stage':'iop','completed':n,'total':98,'message':...}
    사전을 받습니다. cancelled()가 True이면 InterruptedError를 발생시킵니다.

    이 단계의 관측 입력은 심수역 기준 반사도이며, 실측 수심이나 해저면 스펙트럼은
    사용하지 않습니다. 태양 천정각은 농도 최적화 후 D2 감쇠항 준비에 쓰입니다.
    """
    _check_cancelled(cancelled)
    waves = _vector(waves, "waves")
    raw = _vector(deep_rrs, "deep_rrs", len(waves))
    g0, g1 = _vector(g0, "g0", len(waves)), _vector(g1, "g1", len(waves))
    solar = float(solar_zenith)
    if not math.isfinite(solar) or not 0 <= solar < 90:
        raise ValueError("태양 천정각은 공기 중 0 이상 90 미만의 도 단위여야 합니다.")
    converted, target = target_from_rrs(raw, g0, g1)
    model = ComponentIOP(iop_source, waves)
    objective = Objective(model, target, cancelled)
    results, starts = [], []
    for start_id, (start, step) in enumerate(CONFIGURATIONS, 1):
        _check_cancelled(cancelled)
        result = pattern_search(objective, start, step)
        results.append((result, start, step, start_id))
        starts.append(
            dict(
                start_id=start_id,
                start_Chl_mg_m3=start[0] / 100,
                start_TSM_g_m3=start[1] / 100,
                initial_step=step / 100,
                estimated_Chl_mg_m3=result.point[0] / 100,
                estimated_TSM_g_m3=result.point[1] / 100,
                u_SSE=result.sse,
                u_RMSE=math.sqrt(result.sse / len(waves)),
                iterations=result.iterations,
                accepted_moves=result.moves,
                termination="no_improving_0.01_neighbor",
            )
        )
        if progress is not None:
            progress(
                dict(
                    stage="iop",
                    completed=start_id,
                    total=len(CONFIGURATIONS),
                    message=f"Deep-water IOP calculation: {start_id}/{len(CONFIGURATIONS)} starting conditions completed",
                )
            )
    # SSE가 같으면 농도 좌표 순서로, 그마저 같으면 초기 조건의 실행 순서로 선택합니다.
    best, selected_start, selected_step, selected_id = min(
        results, key=lambda item: (item[0].sse, item[0].point)
    )
    neighbors = []
    for dc, dt in DIRECTIONS:
        _check_cancelled(cancelled)
        point = (best.point[0] + dc, best.point[1] + dt)
        valid = min(point) >= 0
        score = (
            math.fsum((row["u"] - u) ** 2 for row, u in zip(model.forward(point), target))
            if valid
            else None
        )
        if valid and score < best.sse:
            raise AssertionError("선택한 IOP 해의 0.01 이웃에 더 낮은 SSE가 남아 있습니다.")
        neighbors.append(
            dict(
                delta_Chl=dc / 100,
                delta_TSM=dt / 100,
                candidate_Chl=point[0] / 100,
                candidate_TSM=point[1] / 100,
                valid_nonnegative=valid,
                selected_SSE=best.sse,
                candidate_SSE=score,
                SSE_difference=score - best.sse if valid else None,
                improves=bool(valid and score < best.sse),
            )
        )
    for row in starts:
        row["selected"] = row["start_id"] == selected_id
    fitted = model.forward(best.point)
    context = make_context(
        [row["a"] for row in fitted], [row["bb"] for row in fitted], solar, waves, g0, g1, raw
    )
    parameter = dict(
        Chl_mg_m3=best.point[0] / 100,
        TSM_g_m3=best.point[1] / 100,
        Chl_hat=best.point[0] / 100,
        TSM_hat=best.point[1] / 100,
        solar_zenith=solar,
        u_SSE=best.sse,
        u_RMSE=math.sqrt(best.sse / len(waves)),
        selected_start_id=selected_id,
        selected_start_Chl=selected_start[0] / 100,
        selected_start_TSM=selected_start[1] / 100,
        selected_initial_step=selected_step / 100,
        starts_completed=len(results),
        distinct_final_points=len({r.point for r, _, _, _ in results}),
        total_iterations=sum(r.iterations for r, _, _, _ in results),
        unique_evaluations=len(objective.cache),
        termination="no_improving_0.01_neighbor",
        input_scope="infinite_depth_only",
        coefficient_refit=False,
        iop_method="component_model_98_starts",
    )
    band_rows = []
    for j, row in enumerate(fitted):
        below = (float(g0[j]) + float(g1[j]) * row["u"]) * row["u"]
        above = 0.52 * below / (1 - 1.7 * below)
        band_rows.append(
            dict(
                wavelength_nm=float(waves[j]),
                wavelength=float(waves[j]),
                g0=float(g0[j]),
                g1=float(g1[j]),
                a_hat=row["a"],
                bb_hat=row["bb"],
                u=row["u"],
                target_u=target[j],
                u_error=row["u"] - target[j],
                kappa=row["a"] + row["bb"],
                aw=row["a_components"][0],
                a_phi=row["a_components"][1],
                a_g=row["a_components"][2],
                a_mineral=row["a_components"][3],
                bbw=row["bb_components"][0],
                bbp=row["bb_components"][1] + row["bb_components"][3],
                Rrs_deep_observed=float(raw[j]),
                converted_rrs=converted[j],
                Rrs_deep_model=above,
                rrs_deep_model=below,
                relative_residual=(above - raw[j]) / raw[j],
                rrs_dp=float(context["rrs_dp"][j]),
                rrs_dp_minus_model=float(context["rrs_dp"][j]) - below,
                xc=float(context["xc"][j]),
                xb=float(context["xb"][j]),
                rrs_dp_source=context["rrs_dp_source"],
                a_components=list(row["a_components"]),
                b_components=list(row["b_components"]),
                bb_components=list(row["bb_components"]),
            )
        )
    return dict(
        context=context,
        parameters=parameter,
        band_rows=band_rows,
        multistart=starts,
        neighbors=neighbors,
    )
