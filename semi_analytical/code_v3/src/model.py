"""공통 계산식: D2 순방향 모델, 수면 경계 변환, C4, 세 바닥 계수 표현.

순방향은 수심 z와 바닥 반사도 rho를 주었을 때 Rrs가 얼마인지 계산하는 방향입니다.
역산은 이 계산으로 만든 Rrs가 관측 Rrs에 가까워지도록 z와 F를 찾아가는 일입니다.
3단계 초기 수심과 5단계 반복 최적화는 같은 D2식과 C4를 사용합니다.

D2 출처: 8차/4번알고리즘완성/동시후보_QAA전용_재계산/02_coefficients/D2.
A1=1, A2=1/pi, DuC=1.03*sqrt(1+2.4*u), DuB=1.04*sqrt(1+5.4*u).
굴절률 n=1.34, 수직 하향 관측을 유지하며 계수를 새로 맞추지 않습니다.
모든 배열의 마지막 밴드 순서는 443·490·560·665 nm입니다.
Rrs는 수면 위, rrs는 수면 아래 반사도입니다. rho는 바닥 반사도로 둘과 다릅니다.
"""
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
    """입력을 유한한 실수 네 값으로 검사하고 원본과 분리된 배열로 반환합니다.

    네 밴드가 아닌 배열이나 NaN·무한대는 오류입니다. 값을 보간하거나 바꾸지 않습니다.
    """
    out = np.asarray(value, dtype=float)
    if out.shape != (4,):
        raise ValueError(f"{name} must contain four values in wavelength order")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains nonfinite input")
    return out.copy()


def load_water(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """별도 HydroLight H2OabDefaults 표에서 파장·aw·bbw를 읽는 보조 함수입니다.

    해당 표의 세 열은 파장, 순수해수 흡수 a, 총 산란 b입니다.
    사용자 지정 관계 bbw=0.49997*bw를 사용합니다. b/2로 대체하지 않습니다.
    443 nm는 표의 440·445 nm를 선형보간하고 490·560·665 nm는 해당 행을 읽습니다.
    이 표는 5 nm 간격의 물 상수표이므로 443 nm 상수를 얻을 때만 인접 두 행이
    필요합니다. 관측 반사도/바닥 스펙트럼에 추가 밴드를 사용하는 것이 아닙니다.
    반환값은 네 밴드의 (파장, aw, bbw)이며 나머지 물 상수 행은 계산에 쓰지 않습니다.
    """
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
    """수면 위 Rrs를 수면 아래 rrs로 변환합니다: rrs=Rrs/(0.52+1.7*Rrs).

    입력값을 임의로 보정하는 함수가 아니라 수면 경계의 변환식입니다.
    분모가 잘못된 입력은 비유한 값으로 드러나며 유효성 검사는 호출 쪽에서 합니다.
    """
    value = np.asarray(Rrs, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return value / (0.52 + 1.7 * value)


def Rrs_from_rrs(rrs: Any) -> np.ndarray:
    """수면 아래 rrs를 수면 위 Rrs로 되돌립니다: Rrs=0.52*rrs/(1-1.7*rrs).

    분모가 정확히 0이거나 계산이 유한하지 않으면 NaN을 반환합니다.
    14차에서는 분모의 부호에 따른 물리적 가지 제한도 제거합니다.
    따라서 분모가 음수여도 유한한 실수 계산이면 원래 유리식 값을 사용합니다.
    계산 가능한 음수 예측 Rrs는 그대로 두며 0으로 잘라내지 않습니다.
    """
    value = np.asarray(rrs, dtype=float)
    denominator = 1.0 - 1.7 * value
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = 0.52 * value / denominator
    return np.where(np.isfinite(value) & (denominator != 0) & np.isfinite(out),
                    out, np.nan)


def make_context(a: Any, bb: Any, solar_zenith: float,
                 deep_rrs_above: Any = None) -> dict[str, Any]:
    """IOP와 태양 천정각으로 D2에서 반복 사용할 네 밴드 계수를 준비합니다.

    a·bb 단위는 m^-1, solar_zenith는 공기 중 태양 천정각(도)입니다.
    kappa=a+bb, u=bb/(a+bb), mu_s는 굴절을 반영한 수중 태양각의 코사인입니다.
    xc는 물기둥 항, xb는 바닥 항의 수심에 따른 감쇠에 사용합니다.

    깊은 물 관측 Rrs를 주면 그 값을 수면 아래 rrs로 바꿔 rrs_dp에 넣습니다.
    이는 11차 1번에서 유지한 D2 경로입니다. IOP의 u로 만든 다항식과 차이도 기록합니다.
    HOPE 피팅 중에는 관측값으로 예측을 덮어쓰면 안 됩니다. 이 함수는 피팅을 마친
    a/bb로 H/F 탐색용 계수를 만들 때 관측 깊은 물 인자와 함께 호출합니다.
    깊은 물 입력을 주지 않은 합성 시험 등에서는 (g0+g1*u)*u를 사용합니다.
    이 함수가 반환하는 context를 같은 조건의 모든 seed가 공유합니다.
    """
    a = _vector4(a, "a")
    bb = _vector4(bb, "bb")
    solar = float(solar_zenith)
    if not math.isfinite(solar) or not 0 <= solar < 90:
        raise ValueError("solar_zenith must be in [0,90) degrees in air")
    kappa = a + bb
    # a+bb의 양수 제약은 제거합니다. 정확히 0이면 u의 나눗셈이 정의되지 않습니다.
    if np.any(kappa == 0):
        raise ValueError("Zero a+bb is outside the real formula domain")
    u = bb / kappa
    arg_c, arg_b = 1.0 + 2.4 * u, 1.0 + 5.4 * u
    if np.any(arg_c < 0) or np.any(arg_b < 0):
        raise ValueError("Lee square-root domain invalid")
    # 공기 중 태양각을 수중 각도로 굴절시킨 뒤 그 코사인을 구합니다.
    # 관측 방향은 기존 D2의 수직 방향이므로 관측각 코사인은 1입니다.
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
    """주어진 수심·바닥 반사도에서 수면 위 Rrs 네 값을 예측합니다.

    수면 아래 rrs = rrs_dp*(1-exp(-xc*z)) + (rho/pi)*exp(-xb*z).
    첫 항은 물기둥 기여, 둘째 항은 수심을 통과해 남은 바닥 기여입니다.
    수심이 0이면 물기둥 항이 0이고 바닥 항만 남습니다.
    이 결과를 Rrs_from_rrs로 변환한 뒤 관측 Rrs와 비교합니다.

    z 한 값이면 길이 4 배열, z 여러 값이면 각 수심의 네 밴드 배열을 반환합니다.
    rho4는 F*rho_첫바닥+(1-F)*rho_둘째바닥으로 만든 값이며 음수·1 초과도 자르지 않습니다.
    이 함수 자체에는 탐색 스텝이나 수심 상한을 넣지 않습니다.
    지수 계산 또는 수면 변환이 유효하지 않으면 해당 예측은 NaN으로 남습니다.
    """
    rho = _vector4(rho4, "rho4")
    depth = np.asarray(z, dtype=float)[..., None]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # expm1(x)=exp(x)-1이므로 -expm1(-x)는 1-exp(-x)와 같습니다.
        # 얕은 수심에서 비슷한 두 수를 빼며 생기는 정밀도 손실을 줄이는 동일 수식입니다.
        water = context["rrs_dp"] * (-np.expm1(-context["xc"] * depth))
        bottom = (rho / np.pi) * np.exp(-context["xb"] * depth)
        below = water + bottom
    return Rrs_from_rrs(below)


def c4(observed, predicted):
    """관측 Rrs와 예측 Rrs 사이 네 밴드 상대 제곱오차의 평균을 구합니다.

    C4=(1/4)*sum(((Rrs_model-Rrs_obs)/Rrs_obs)^2).
    각 밴드의 관측값 그대로 나누며 하한값이나 새로운 가중치를 넣지 않습니다.
    따라서 작은 관측값의 상대오차가 크게 반영될 수 있는 성질도 그대로 남습니다.
    관측 Rrs가 정확히 0이면 이 식은 계산할 수 없으므로 오류로 알립니다.
    예측이 NaN·무한대이면 C4를 무한대로 돌려 그 후보가 선택되지 않도록 합니다.
    여기서 계산한 C4는 무차원이며 RMSE(m)와 다른 지표입니다.
    """
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    if observed.shape != (4,) or predicted.shape[-1:] != (4,):
        raise ValueError("Exactly four bands are required")
    if not np.isfinite(observed).all() or np.any(observed == 0):
        raise ValueError("Original C4 requires four finite nonzero inputs")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # 핵심 C4 식입니다. observed 대신 max(observed,하한)을 쓰지 않습니다.
        value = np.mean(((predicted-observed)/observed)**2, axis=-1)
    return np.where(np.isfinite(value), value, np.inf)


def components(pair, fraction):
    """바닥쌍의 F를 [모래, 갈조, 녹조] 공통 계수로 펼칩니다.

    예: pair=(sand,brown), F=0.9이면 sand=0.9, brown=0.1, green=0입니다.
    F=1.2이면 sand=1.2, brown=-0.2, green=0으로 그대로 둡니다.
    쌍의 이름이 달라도 이 세 계수로 비교하면 같은 바닥 구성을 알아볼 수 있습니다.
    """
    result = dict.fromkeys(MATERIALS, 0.0)
    result[pair[0]] = float(fraction)
    result[pair[1]] = float(1-fraction)
    return result
