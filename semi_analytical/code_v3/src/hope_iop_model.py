"""HOPE 계열 P·G·X 분광모형과 깊은 물 순방향 반사도 계산.

이 파일은 역산기나 H/F 탐색기가 아니라, 세 광학 변수에서 스펙트럼을 만드는
순방향 계산이다. 밴드 순서는 443·490·560·665 nm, 흡수·후방산란 단위는 m^-1,
파장은 nm이다. P·G·X를 바꾸면 a·bb와 모델 깊은 물 Rrs가 함께 변한다.

근거: Lee et al. (1998) Table 2의 a0/a1, Lee et al. (1999) 식 (16)–(19)의
분광 매개변수화, Dekker et al. (2011) HOPE 식 (8)–(10)의 S=0.015, Y=0.5.
X는 여기서 550 nm 입자 후방산란이다. 다른 기준파장의 X와 혼용하지 않는다.
443/665 nm의 a0/a1은 원문 440/450, 660/670 nm 두 행 사이의 선형보간값이다.

깊은 물 반사도 관계의 g0/g1=0.089/0.1245 및 수면 변환 0.52/1.7은 비교 대상
기존 model.py에서 가져온다. 따라서 Lee 1999 전체 알고리즘의 그대로 재현이
아니라 HOPE 분광식과 기존 순방향 관계를 결합한 실험이다. QAA 역산과 관측
깊은 물 반사도 대입을 이 계산 안에서 수행하지 않는다. SAMBUCA도 사용하지 않는다.
"""
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
# 사용자가 추가한 제약은 P/G/X>=0입니다. ln(P)가 포함되므로 P=0은 수식에서
# 제외되어 실제 정의역은 P>0, G>=0, X>=0입니다. 기존의 양수 P_MIN은 쓰지 않습니다.
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
    """원본 파장 두 점으로 초기식 입력을 복원한다. 피팅의 4밴드는 바꾸지 않는다.

    440 nm만 최소 443 nm에서 3 nm 외삽을 허용한다(사용자 승인).
    다른 외삽, 중복 파장, 음수/0 입력 및 결과는 오류로 남긴다.
    """
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
    """깊은 물 관측 Rrs로 사용자 지정 초기식의 기준 P_in/G_in/X_in을 계산한다.

    P_in=0.072*(Rrs440/Rrs550)^(-1.62), G_in=P_in, X_in=30*aw640*Rrs640.
    여기의 Rrs는 수면 위 반사도다. aw는 사용자 지정 물 상수표의 네 밴드 값이다.
    490/560으로 550, 560/665로 640을 선형보간한다. aw(640)도 네 밴드 aw에서
    보간하므로 초기식만 몰래 다른 물 상수 파장을 읽는 경로가 없다.
    440은 443/490의 기울기를 이용하여 3 nm 선형외삽한다(사용자 지정).
    모든 보간 양 끝 파장과 가중치를 기록하므로 실제 원본 440 nm 관측과 구분된다.
    각 기준값에 0.5/1.0/1.5를 독립적으로 곱하여 3*3*3=27개 시작점을 만든다.
    G_in=P_in이어도 두 배율은 독립적이므로 최적화 중 P=G로 묶지 않는다.
    시작점을 다른 값으로 투영하거나 자르지 않고 27개 원값을 그대로 사용한다.
    참 총 IOP/참 수심/참 바닥 비율은 이 함수의 입력이 아니다.
    """
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
    """순수 물 입력의 모양·유한성·비음수를 검사한다. 후방산란 bbw를 다시 나누지 않는다."""
    aw = _vector4(water_aw4, "water_aw4")
    bbw = _vector4(water_bbw4, "water_bbw4")
    if np.any(aw < 0) or np.any(bbw < 0):
        raise ValueError("Pure-water aw and bbw must be nonnegative")
    return aw, bbw


def spectrum(pgx: Any, water_aw4: Any, water_bbw4: Any) -> dict[str, np.ndarray]:
    """P·G·X에서 a_phi, a_g, bbp, 총 a·bb와 깊은 물 반사도를 계산한다.

    a_phi=[a0+a1*ln(P)]P; a_g=G*exp[-S*(lambda-440)]; a=aw+a_phi+a_g.
    bbp=X*(550/lambda)^Y; bb=bbw+bbp; u=bb/(a+bb).
    rrs_model=(g0+g1*u)*u; Rrs_model=0.52*rrs_model/(1-1.7*rrs_model).
    ln은 자연로그다. P>0은 수식의 실수 정의역이고 양수 P_MIN은 사용하지 않는다.
    G/X>=0도 이 공개 순방향 함수에서 검사하여 역산 밖의 호출에서도 음수를 막는다.
    P/G/X 비음수 조건과 a_phi>=0는 서로 다르므로 a_phi·총 a/bb의 제약은 추가하지 않는다.
    이 함수에는 관측 Rrs 인자가 없다. 역산 목적함수의 모델값이 관측값으로
    덮이는 실수를 인터페이스 단계에서 막는다.
    """
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
    """깊은 물 네 밴드 상대잔차를 반환한다. 분모 하한이나 관측 보정을 넣지 않는다."""
    raw = _vector4(observed, "deep_Rrs_observed")
    if np.any(raw <= 0):
        raise ValueError("Observed deep Rrs must be strictly positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        residual = (spectrum(pgx, aw4, bbw4)["Rrs_deep_model"] - raw) / raw
    if not np.all(np.isfinite(residual)):
        raise ValueError("Nonfinite relative residual")
    return residual
