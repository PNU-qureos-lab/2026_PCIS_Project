"""반사도 모형을 이용해 픽셀별 수심과 해저면 혼합계수를 추정합니다.

입력은 수면 위 원격탐사반사도 Rrs(sr^-1), 고유광학특성(IOP)에서 계산한
감쇠계수, 심수역 기준 반사도, 해저면 반사도 스펙트럼입니다. 센서의 파장과
g0·g1도 인자로 전달하므로 영상 파일 처리나 화면 구성과 독립적으로 동작합니다.

D2 반사도식의 오차를 최소화하도록 수심 z(m)와 혼합계수 F(무차원)를 찾습니다.
수심은 0 이상으로 제한하며, F는 제한하지 않고 범위 이탈 여부를 결과에 기록합니다.
Brent 방법으로 초기 수심을 정한 뒤 수심·혼합계수의 8방향 이웃을 비교합니다.
해저면 2종은 한 쌍의 초기 F 세 개를, 3종은 세 쌍의 초기 후보 아홉 개 중
오차가 작은 다섯 개를 탐색합니다. 최종 반사도는 항상 두 해저면의 선형 혼합입니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize_scalar


NEIGHBOR_OFFSETS = tuple((dz, df) for dz in (-1, 0, 1) for df in (-1, 0, 1) if (dz, df) != (0, 0))
LEE_METHOD = "Lee-D2-8th-A1=1-A2=1/pi-1.03-2.4-1.04-5.4-n1.34-nadir"


@dataclass(frozen=True)
class SolverSettings:
    """수심·혼합계수 탐색의 이동 간격, 초기값 및 종료 기준입니다.

    수심 이동 간격은 초기 Brent 정밀도와 다릅니다. 초기 2.37 m에서 시작하면
    2.27, 2.37, 2.47 m 격자를 사용하며 0 m 원점 격자로 반올림하지 않습니다.
    max_evaluations는 반복 횟수가 아닌 초기 후보당 실제 C4 계산 횟수 제한입니다.
    """

    depth_step: float = 0.1
    fraction_step: float = 0.01
    seed_fractions: tuple[float, ...] = (0.75, 0.5, 0.25)
    top_k: int = 5
    max_evaluations: int = 10000
    initial_xtol: float = 1e-4
    initial_maxiter: int = 500
    near_relative_c4: float = 0.01
    same_solution_atol: float = 1e-9

    def validate(self) -> None:
        """설정의 유한성·허용 범위와 고정된 초기 혼합계수를 검사합니다."""
        for name in ("depth_step", "fraction_step", "initial_xtol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}은 유한한 양수여야 합니다.")
        for name in ("near_relative_c4", "same_solution_atol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name}은 유한한 0 이상의 값이어야 합니다.")
        for name, minimum in (("top_k", 1), ("max_evaluations", 20), ("initial_maxiter", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name}은 {minimum} 이상의 정수여야 합니다.")
        if self.top_k > 9:
            raise ValueError("top_k는 최대 초기 후보 수인 9를 넘을 수 없습니다.")
        if tuple(self.seed_fractions) != (0.75, 0.5, 0.25):
            raise ValueError("초기 혼합계수는 (0.75, 0.5, 0.25)여야 합니다.")


def _vector(value: Any, name: str, size: int | None = None) -> np.ndarray:
    """센서 밴드 순서의 유한한 벡터를 복사합니다. 보간·자르기는 하지 않습니다."""
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or (size is not None and result.shape != (size,)):
        raise ValueError(f"{name}의 밴드 수 또는 배열 형태가 올바르지 않습니다.")
    if size is None and len(result) not in (4, 6):
        raise ValueError(f"{name}에는 Sentinel-2 4밴드 또는 SuperDove 6밴드가 필요합니다.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name}에 NaN 또는 무한대가 포함되어 있습니다.")
    return result.copy()


def rrs_from_Rrs(values: Any) -> np.ndarray:
    """수면 위 Rrs를 수면 아래 rrs로 변환합니다. 두 반사도의 단위는 sr^-1입니다."""
    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return values / (0.52 + 1.7 * values)


def Rrs_from_rrs(values: Any) -> np.ndarray:
    """수면 아래 rrs를 수면 위 Rrs로 변환합니다. 정의되지 않는 값은 NaN입니다.

    변환식은 Rrs=0.52*rrs/(1-1.7*rrs)입니다. 이 함수는 분모가 0인 값과
    비유한 값을 NaN으로 바꾸며, 수심 탐색의 오차 평가에서는 분모<=0도 제외합니다.
    """
    values = np.asarray(values, dtype=float)
    denominator = 1.0 - 1.7 * values
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = 0.52 * values / denominator
    return np.where(np.isfinite(values) & (denominator != 0) & np.isfinite(result), result, np.nan)


def make_context(
    a: Any, bb: Any, solar_zenith: float, waves: Any, g0: Any, g1: Any, deep_rrs_above: Any = None
) -> dict:
    """주어진 IOP와 심수역 기준 반사도로 D2 모형의 공통 계산값을 준비합니다.

    a, bb의 단위는 m^-1, 태양 천정각은 공기 중의 도입니다. 관측 방향은 수직입니다.
    kappa=a+bb, u=bb/kappa, mu_s=sqrt(1-(sin(solar)/1.34)^2)이며,
    xc=(1/mu_s+1.03*sqrt(1+2.4*u))*kappa,
    xb=(1/mu_s+1.04*sqrt(1+5.4*u))*kappa 입니다.

    영상 분석에서는 심수역의 수면 위 Rrs를 deep_rrs_above로 전달합니다. 이를
    수면 아래 값으로 변환해 물기둥 항의 rrs_dp로 사용합니다. g0·g1은 변경하지
    않습니다. deep_rrs_above를 생략한 합성 시험에서는 (g0+g1*u)*u를 사용합니다.
    """
    waves = _vector(waves, "waves")
    n = len(waves)
    if np.any(waves <= 0) or np.any(np.diff(waves) <= 0):
        raise ValueError("파장은 양수이며 오름차순이어야 합니다.")
    a, bb = _vector(a, "a", n), _vector(bb, "bb", n)
    g0, g1 = _vector(g0, "g0", n), _vector(g1, "g1", n)
    solar = float(solar_zenith)
    if not math.isfinite(solar) or not 0 <= solar < 90:
        raise ValueError("태양 천정각은 0 이상 90 미만이어야 합니다.")
    kappa = a + bb
    if np.any(kappa == 0):
        raise ValueError("a+bb=0이면 D2의 u를 계산할 수 없습니다.")
    u = bb / kappa
    arg_c, arg_b = 1.0 + 2.4 * u, 1.0 + 5.4 * u
    if np.any(arg_c < 0) or np.any(arg_b < 0):
        raise ValueError("D2 제곱근의 실수 정의역을 벗어났습니다.")
    mu_s = math.sqrt(1.0 - (math.sin(math.radians(solar)) / 1.34) ** 2)
    duc, dub = 1.03 * np.sqrt(arg_c), 1.04 * np.sqrt(arg_b)
    xc, xb = (1.0 / mu_s + duc) * kappa, (1.0 / mu_s + dub) * kappa
    dp = (g0 + g1 * u) * u
    if not np.all(np.isfinite([xc, xb, dp])):
        raise ValueError("D2 모형의 공통 계산값에 NaN 또는 무한대가 생겼습니다.")
    context = dict(
        a=a,
        bb=bb,
        waves=waves,
        g0=g0,
        g1=g1,
        u=u,
        kappa=kappa,
        duc=duc,
        dub=dub,
        xc=xc,
        xb=xb,
        rrs_dp=dp,
        solar_zenith=solar,
        n_water=1.34,
        cos_view_water=1.0,
        method=LEE_METHOD,
        rrs_dp_source="iop_u_polynomial",
    )
    if deep_rrs_above is not None:
        deep = _vector(deep_rrs_above, "deep_rrs_above", n)
        converted = rrs_from_Rrs(deep)
        if not np.isfinite(converted).all():
            raise ValueError("깊은 물 Rrs 수면 변환 결과가 유한하지 않습니다.")
        context.update(
            rrs_deep_converted=converted,
            rrs_dp_max_abs_difference=float(np.max(np.abs(dp - converted))),
            rrs_dp=converted,
            rrs_dp_source="measured_deep_Rrs_interface_transform",
        )
    return context


def forward(depth: Any, rho: Any, context: dict) -> np.ndarray:
    """수심과 혼합 해저면 반사도로 D2 수면 위 Rrs를 예측합니다.

    rrs=rrs_dp*(1-exp(-xc*z))+(rho/pi)*exp(-xb*z).
    얕은 수심의 소거 오차를 줄이기 위해 첫 항에 -expm1(-xc*z)를 씁니다.
    수심 하나 또는 수심 배열을 받을 수 있습니다. rho는 이미 선택한 두 해저면을
    F*rho_A+(1-F)*rho_B로 섞은 스펙트럼이며 F 범위 제한은 적용하지 않습니다.
    """
    rho = _vector(rho, "rho", len(context["waves"]))
    depth = np.asarray(depth, dtype=float)[..., None]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        water = context["rrs_dp"] * (-np.expm1(-context["xc"] * depth))
        bottom = (rho / np.pi) * np.exp(-context["xb"] * depth)
    return Rrs_from_rrs(water + bottom)


def c4(observed: Any, predicted: Any) -> np.ndarray:
    """N개 밴드의 평균 상대제곱오차 C4를 계산합니다.

    C4=(1/N)*sum(((Rrs_model-Rrs_obs)/Rrs_obs)^2); N=4 또는 6.
    분모 하한, 추가 가중치, 관측값 보정을 적용하지 않습니다. 관측값 0은 오류입니다.
    """
    observed = _vector(observed, "observed")
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape[-1:] != observed.shape or np.any(observed == 0):
        raise ValueError("C4는 같은 밴드 수와 0이 아닌 관측값을 요구합니다.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = np.mean(((predicted - observed) / observed) ** 2, axis=-1)
    return np.where(np.isfinite(result), result, np.inf)


class EvaluationLimit(Exception):
    """초기 후보별 평가 한도 도달을 정상적인 탐색 종료와 구분합니다."""


class _Candidate:
    """초기 F 한 개와 해저면 유형쌍 한 개의 독립된 탐색 상태와 캐시입니다."""

    def __init__(self, observed, context, library, pair, initial_f, settings, cancelled):
        self.observed, self.context = observed, context
        self.materials, self.pair = tuple(library), pair
        self.rho_a, self.rho_b = library[pair[0]], library[pair[1]]
        self.initial_f, self.settings, self.cancelled = initial_f, settings, cancelled
        # seed_id의 문자열 순서로 오차가 같은 후보의 우선순위를 정합니다.
        # 해저면 ID에 밑줄이 들어갈 수 있으므로 쌍은 별도 튜플로 보관합니다.
        self.seed_id = f"{pair[0]}_{pair[1]}_F{initial_f:.2f}"
        self.cache, self.physical_cache, self.depth_terms, self.initial_values = {}, {}, {}, {}
        self.z_origin = None
        self.evaluations = 0
        self.best_key, self.initial_key = None, None
        self.best_cost, self.initial_cost = math.inf, math.inf
        self.initialization_evaluations = 0
        self.initial_status, self.initial_solver_message = "uncomputed", ""
        self.initial_solver_success = False
        self.initial_nit, self.initial_solver_nfev = None, None
        self.initial_boundary_selected = False

    def check_cancelled(self):
        """취소를 수렴으로 바꾸지 않고 호출부까지 전달합니다."""
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("사용자가 수심 계산을 중지했습니다.")

    def coordinates(self, key):
        """초기 수심 원점과 정수 이동 횟수를 실제 수심·F로 바꿉니다."""
        nz, nf = key
        return (
            self.z_origin + nz * self.settings.depth_step,
            round(self.initial_f + nf * self.settings.fraction_step, 12),
        )

    def rho(self, fraction):
        """F와 1-F로 선형 혼합합니다. 음수 또는 1 초과 F도 그대로 둡니다."""
        return fraction * self.rho_a + (1 - fraction) * self.rho_b

    def _evaluate_at(self, z, f):
        """수심(m)·혼합계수 좌표에서 밴드별 상대제곱오차를 순서대로 더합니다."""
        self.check_cancelled()
        z, f = float(z), float(f)
        if not math.isfinite(z) or z < 0 or not math.isfinite(f):
            return math.inf
        physical_key = (z, f)
        if physical_key in self.physical_cache:
            return self.physical_cache[physical_key]
        if self.evaluations >= self.settings.max_evaluations:
            raise EvaluationLimit
        terms = self.depth_terms.get(z)
        if terms is None:
            terms = []
            for j in range(len(self.observed)):
                eb = math.exp(-float(self.context["xb"][j]) * z) / math.pi
                water = -float(self.context["rrs_dp"][j]) * math.expm1(
                    -float(self.context["xc"][j]) * z
                )
                terms.append(
                    (water, eb, float(self.rho_a[j]), float(self.rho_b[j]), float(self.observed[j]))
                )
            self.depth_terms[z] = terms
        total = 0.0
        for water, eb, rho_a, rho_b, obs in terms:
            below = water + (f * rho_a + (1 - f) * rho_b) * eb
            denominator = 1 - 1.7 * below
            # 수면 경계 변환의 분모가 0 이하이면 유효한 예측으로 채택하지 않습니다.
            if denominator <= 0 or not math.isfinite(below):
                total = math.inf
                break
            predicted = 0.52 * below / denominator
            residual = (predicted - obs) / obs
            total += residual * residual
        value = total / len(self.observed)
        if not math.isfinite(value):
            value = math.inf
        self.evaluations += 1
        self.physical_cache[physical_key] = value
        return value

    def evaluate(self, key):
        """같은 격자점은 재계산하지 않고, 평가한 최저 C4의 좌표를 기억합니다."""
        if key not in self.cache:
            self.cache[key] = self._evaluate_at(*self.coordinates(key))
            if self.cache[key] < self.best_cost:
                self.best_key, self.best_cost = key, self.cache[key]
        return self.cache[key]

    def initial_metadata(self):
        """초기 수심 탐색의 성공 여부, 종료 상태 및 평가 횟수를 반환합니다."""
        return dict(
            initial_method="scipy_brent_nonnegative_boundary",
            initial_status=self.initial_status,
            initial_solver_success=self.initial_solver_success,
            initial_solver_message=self.initial_solver_message,
            initial_xtol=self.settings.initial_xtol,
            initial_nit=self.initial_nit,
            initial_solver_nfev=self.initial_solver_nfev,
            initial_boundary_selected=self.initial_boundary_selected,
            initialization_evaluations=self.initialization_evaluations,
        )


def _initialize(seed):
    """초기 F를 고정하고 연속 수심을 Brent로 구한 뒤 격자의 원점으로 삼습니다."""

    def objective(z):
        z = float(z)
        if z not in seed.initial_values:
            seed.initial_values[z] = seed._evaluate_at(z, seed.initial_f)
        return seed.initial_values[z]

    try:
        objective(0.0)  # z>=0 경계는 Brent의 성공 여부와 별개로 먼저 비교합니다.
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            # (0,1)은 출발 두 점입니다. 수심 상한이 1 m라는 뜻이 아닙니다.
            result = minimize_scalar(
                objective,
                method="brent",
                bracket=(0.0, 1.0),
                options={
                    "xtol": seed.settings.initial_xtol,
                    "maxiter": seed.settings.initial_maxiter,
                },
            )
        seed.initial_solver_success = bool(result.success and math.isfinite(float(result.fun)))
        seed.initial_solver_message = str(result.message)
        seed.initial_nit = int(result.nit) if hasattr(result, "nit") else None
        seed.initial_solver_nfev = int(result.nfev) if hasattr(result, "nfev") else None
        if seed.initial_solver_success:
            seed.initial_status = "success"
        elif (
            seed.initial_nit is not None
            and seed.initial_nit >= seed.settings.initial_maxiter
            or "maximum" in str(result.message).lower()
        ):
            seed.initial_status = "maxiter"
        else:
            seed.initial_status = "no_bracket"
    except EvaluationLimit:
        seed.initial_status = "evaluation_limit"
        seed.initial_solver_message = "초기 수심 계산 중 후보별 평가 횟수 제한에 도달했습니다."
    except (ValueError, RuntimeError, OverflowError) as exc:
        seed.initial_status, seed.initial_solver_message = "no_bracket", str(exc)
    seed.initialization_evaluations = seed.evaluations
    finite = [
        (value, z)
        for z, value in seed.initial_values.items()
        if z >= 0 and math.isfinite(z) and math.isfinite(value)
    ]
    if not finite:
        raise ValueError(f"유한한 초기 수심 후보가 없습니다 ({seed.seed_id}, {seed.initial_status}).")
    # 최적화기가 보고한 마지막 점뿐 아니라 실제 평가한 모든 유효 수심을 비교합니다.
    seed.initial_cost, seed.z_origin = min(finite, key=lambda item: (item[0], item[1]))
    seed.initial_boundary_selected = seed.z_origin == 0.0
    seed.initial_key = seed.best_key = (0, 0)
    seed.best_cost = seed.initial_cost
    seed.cache[(0, 0)] = seed.initial_cost


def _optimize(seed):
    """8개 이웃을 모두 평가한 후 최저점으로만 이동합니다. 이동 간격을 줄이지 않습니다."""
    key, cost, cycle, reason = seed.initial_key, seed.initial_cost, 0, "no_c4_improvement"
    try:
        if seed.initial_status == "evaluation_limit":
            raise EvaluationLimit
        while True:
            seed.check_cancelled()
            cycle += 1
            options = []
            for dz, df in NEIGHBOR_OFFSETS:
                neighbor = (key[0] + dz, key[1] + df)
                options.append((seed.evaluate(neighbor), neighbor))
            # C4가 같으면 정수 좌표 튜플의 순서로 다음 위치를 결정합니다.
            next_cost, next_key = min(options, key=lambda item: (item[0], item[1]))
            if not next_cost < cost:
                break
            key, cost = next_key, next_cost
    except EvaluationLimit:
        reason = "evaluation_limit"
        # 8개 비교 도중 한도에 닿아도 그때까지 평가한 더 좋은 점을 버리지 않습니다.
        key, cost = seed.best_key, seed.best_cost
    z, f = seed.coordinates(key)
    fractions = dict.fromkeys(seed.materials, 0.0)
    fractions[seed.pair[0]], fractions[seed.pair[1]] = float(f), float(1 - f)
    rho = seed.rho(f)
    predicted = forward(z, rho, seed.context)
    return dict(
        seed_id=seed.seed_id,
        pair=seed.pair,
        initial_F=seed.initial_f,
        initial_z=seed.z_origin,
        initial_c4=seed.initial_cost,
        depth_m=z,
        z_hat=z,
        F=f,
        fractions=fractions,
        c4=float(cost),
        rms_relative_error_pct=100 * math.sqrt(cost),
        status=reason,
        cycles=cycle,
        evaluations=seed.evaluations,
        **seed.initial_metadata(),
        f_negative=f < 0,
        f_ge1=f >= 1,
        f_outside=(f < 0 or f > 1),
        bottom_reflectance_negative=bool(np.any(rho < 0)),
        bottom_reflectance_above_one=bool(np.any(rho > 1)),
        predicted_rrs=[float(v) for v in predicted],
    )


def _classify(records, settings, materials):
    """최소 C4의 대표 해와 상대 오차 기준 이내의 서로 다른 후보를 구분합니다."""
    ranked = sorted(records, key=lambda row: (row["c4"], row["seed_id"]))
    if not ranked or not math.isfinite(ranked[0]["c4"]):
        raise ValueError("유한한 최종 수심 후보가 없습니다.")
    representatives = []
    for rank, row in enumerate(ranked, 1):
        signature = np.array([row["depth_m"], *(row["fractions"][m] for m in materials)])
        match = next(
            (
                i
                for i, rep in enumerate(representatives)
                if np.allclose(signature, rep[0], rtol=0, atol=settings.same_solution_atol)
            ),
            None,
        )
        row["rank"], row["same_solution_duplicate"] = rank, match is not None
        if match is None:
            match = len(representatives)
            representatives.append((signature, row))
        row["solution_group"] = match + 1
        row["same_solution_as"] = representatives[match][1]["seed_id"]
    winner = ranked[0]
    cutoff = winner["c4"] * (1 + settings.near_relative_c4)
    for row in ranked:
        row["near_best"] = bool(row["c4"] <= cutoff)
        row["relative_c4_gap"] = (
            (row["c4"] - winner["c4"]) / winner["c4"]
            if winner["c4"] > 0
            else (0.0 if row["c4"] == 0 else math.inf)
        )
        row["is_representative"] = row is winner
        row["is_alternative"] = (
            row["near_best"]
            and row["solution_group"] != winner["solution_group"]
            and not row["same_solution_duplicate"]
        )
    alternatives = [row for row in ranked if row["is_alternative"]]
    summary = dict(winner)
    summary.update(
        has_alternative=bool(alternatives),
        alternative_count=len(alternatives),
        unique_solution_count=len(representatives),
        same_solution_duplicate_count=len(ranked) - len(representatives),
        total_evaluations=sum(row["evaluations"] for row in ranked),
        any_evaluation_limit=any(row["status"] == "evaluation_limit" for row in ranked),
        alternative_depth_min=min([winner["depth_m"]] + [r["depth_m"] for r in alternatives]),
        alternative_depth_max=max([winner["depth_m"]] + [r["depth_m"] for r in alternatives]),
        candidates=ranked,
    )
    return summary


def solve_pixel(
    observed: Any,
    context: dict,
    library: dict[str, Any],
    settings: SolverSettings | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """선택한 2종/3종 해저면으로 한 픽셀의 수심과 최적 해저면 유형쌍을 추정합니다.

    observed는 센서 파장 순서의 수면 위 Rrs입니다. library의 키는 해저면 ID,
    값은 같은 파장 순서의 반사도입니다. 해저면 제목에 밑줄이 있어도 안전합니다.
    library의 순서를 해저면 유형쌍의 첫째/둘째 순서로 사용하므로 재현을 위해 설정에 기록합니다.
    반환 fractions는 선택한 모든 해저면을 포함하며 최종 쌍 밖 해저면의 계수는 0입니다.
    total_evaluations는 최적화한 후보들의 평가 수 합계입니다.
    all_evaluations에는 최종 상위 후보에서 탈락한 초기 후보 평가도 포함합니다.

    음수/1 초과 F, 평가 한도, 유사 후보 존재 여부는 결과 표시값입니다. 이를 이유로
    최소 C4 대표를 교체하거나 수심을 보정하지 않습니다. 유사 후보 수심 범위는
    저장한 후보들의 범위이지 통계적 신뢰구간이 아닙니다.
    """
    settings = settings or SolverSettings()
    settings.validate()
    waves = _vector(context["waves"], "context.waves")
    observed = _vector(observed, "observed", len(waves))
    if np.any(observed == 0):
        raise ValueError("관측 Rrs가 0인 밴드는 상대오차를 계산할 수 없습니다.")
    context = dict(context)
    for name in ("xc", "xb", "rrs_dp"):
        context[name] = _vector(context[name], f"context.{name}", len(waves))
    if len(library) not in (2, 3) or any(not isinstance(k, str) or not k for k in library):
        raise ValueError("고유한 이름의 해저면을 정확히 2종 또는 3종 선택해야 합니다.")
    library = {name: _vector(values, f"해저면 {name}", len(waves)) for name, values in library.items()}
    pairs = tuple(combinations(library, 2))
    seeds = [
        _Candidate(observed, context, library, pair, f, settings, cancelled)
        for pair in pairs
        for f in settings.seed_fractions
    ]
    for seed in seeds:
        _initialize(seed)
    # 기본 top_k=5에서 2종은 세 후보 모두, 3종은 아홉 후보 중 상위 다섯 후보를 탐색합니다.
    ranked = sorted(seeds, key=lambda seed: (seed.initial_cost, seed.seed_id))
    selected = ranked[: min(settings.top_k, len(ranked))]
    records = [_optimize(seed) for seed in selected]
    summary = _classify(records, settings, tuple(library))
    summary.update(
        initial_candidate_count=len(seeds),
        optimized_candidate_count=len(selected),
        all_evaluations=sum(seed.evaluations for seed in seeds),
        initial_candidates=[
            dict(
                seed_id=s.seed_id,
                pair=s.pair,
                initial_F=s.initial_f,
                initial_z=s.z_origin,
                initial_c4=s.initial_cost,
                selected=s in selected,
                **s.initial_metadata(),
            )
            for s in ranked
        ],
    )
    return summary
