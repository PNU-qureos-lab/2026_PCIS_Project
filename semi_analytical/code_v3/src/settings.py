"""수치 설정을 한곳에 모으고 합의값과 기존 시험값을 구분합니다.

사용자 확정: 수심 이동 0.1 m, F 이동 0.01, 초기 9개 중 k=5,
최소 C4 대비 1% 이내 다른 후보 표시, F 범위 제약 없음, C4 분모 하한 없음.
종료는 정한 좌표 이동에서 C4 개선 없음이며 절대 C4 목표값을 두지 않습니다.

max_evaluations=10000과 Figure 5 정규화 분모는 기존 시험·제안값입니다.
initial_xtol과 initial_maxiter는 초기 수심 스칼라 계산의 수치 설정입니다.
same_solution_atol은 같은 해의 수치적 동등성 비교용이지 물리적 정확도 기준이 아닙니다.
SETTING_NOTES는 실행 결과 run_config.json에도 저장하여 사용한 조건을 남깁니다.
"""
from dataclasses import dataclass
from pathlib import Path
import math

BANDS_NM = (443, 490, 560, 665)
MATERIALS = ("sand", "seagrass")
PAIRS = (("sand", "seagrass"),)
DEFAULT_DATA = Path("G:/태평양/9차/데이터")
DEFAULT_BOTTOM = Path("G:/태평양/9차/bottom_reflectances")
# Figure 5는 RMSE 확정. 다음 정규화 분모는 기존 미리보기의 제안값입니다.
FIGURE5_DEPTH_RANGE_M = 4.5
FIGURE5_COEFFICIENT_RANGE = 1.0

@dataclass(frozen=True)
class Settings:
    """한 번 실행할 때 각 seed가 공유하는 설정 묶음입니다.

    frozen=True이므로 계산 중 누군가 스텝 등을 바꾸는 일을 막습니다.
    값을 바꾸려면 새 Settings를 만들어야 하며 실행 기록에 그 값이 남습니다.
    일반 실행에서는 아래 기본값을 그대로 사용합니다.
    """
    # [확정] 반복 탐색의 수심 이동 간격(m). 초기 수심 정밀도와 다른 값입니다.
    depth_step: float = 0.1
    # [확정] 반복 탐색의 F 이동 간격. 계산 중 줄이지 않습니다.
    fraction_step: float = 0.01
    # 초기 비율 3개는 그대로 두고, 이후 각각의 시작값에서 ±0.01씩 움직입니다.
    seed_fractions: tuple = (0.75, 0.5, 0.25)
    # [확정] 초기 C4로 선정하여 반복 최적화할 seed 개수입니다.
    top_k: int = 3
    # [기존 시험값] seed당 실제 C4 평가 한도. 초기 수심 계산도 포함합니다.
    max_evaluations: int = 10000
    # [기존 수치해법] 초기 수심 Brent의 상대 허용오차. C4 종료 목표값이 아닙니다.
    initial_xtol: float = 1e-4
    # [기존 수치해법] Brent 내부 최대 반복 수. 전체 C4 평가 한도와 별개입니다.
    initial_maxiter: int = 500
    # [확정] 대표보다 C4가 1% 이내로 큰 다른 해를 추가 후보로 표시합니다.
    near_relative_c4: float = 0.01
    # [수치 비교] 수심·세 바닥 계수가 사실상 같은지 비교하는 기존 허용차입니다.
    same_solution_atol: float = 1e-9

    def validate(self):
        """설정의 형식과 기본 유효성을 검사합니다.

        이 검사는 허용할 수 없는 음수 스텝·잘못된 k 등을 거절합니다.
        과학적으로 최적인 설정인지 자동으로 결정하거나 값을 고치는 함수가 아닙니다.
        """
        if self.depth_step <= 0 or self.fraction_step <= 0:
            raise ValueError("Steps must be positive")
        if self.max_evaluations < 20:
            raise ValueError("At least 20 evaluations are required")
        if not math.isfinite(self.initial_xtol) or self.initial_xtol <= 0:
            raise ValueError("Initial Brent xtol must be finite and positive")
        if self.initial_maxiter < 1:
            raise ValueError("Initial Brent maxiter must be positive")
        if not 1 <= self.top_k <= len(PAIRS) * len(self.seed_fractions):
            raise ValueError("Invalid top_k")
        if self.near_relative_c4 < 0 or self.same_solution_atol < 0:
            raise ValueError("Tolerances cannot be negative")

SETTING_NOTES = {
    "depth_step": "사용자 확정: 0.1 m 고정",
    "fraction_step": "사용자 확정: 0.01 고정",
    "top_k": "사용자 확정: 초기 9개 중 C4 하위 5개",
    "near_relative_c4": "사용자 확정: 최소 C4보다 1% 이내인 다른 해 표시",
    "max_evaluations": "기존 시험용 한도 10000, 초기 계산 포함",
    "initial_xtol": "8차 Brent 상대 허용오차 1e-4; 수심 격자 간격이 아님",
    "initial_maxiter": "기존 SciPy 기본값 500을 명시",
    "same_solution_atol": "기존 수치적 동등성 비교용 1e-9; 물리적 정확도 기준이 아님",
    "search_neighborhood": "현재점의 상하좌우 4개와 대각선 4개를 모두 비교; 최소 C4로 이동",
    "stopping": "주변 8방향 모두 C4 개선 없음; 절대 목표값·스텝 축소 없음",
    "fraction_bounds": "없음; 음수/1 초과도 최소 C4이면 선정 후 표시",
    "c4_denominator_floor": "없음; 정확히 0인 관측값은 계산 불가",
    "figure5_reference_ranges": "기존 제안값 4.5 m와 1; RMSE와 정규화 전 값도 저장",
}
