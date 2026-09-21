"""3단계: 바닥쌍과 초기 F로 9개 seed를 만들고 각각의 초기 수심을 구합니다.

seed는 반복 최적화를 시작할 후보 한 개입니다.
모래–갈조, 모래–녹조, 갈조–녹조 3쌍 × F=0.75·0.50·0.25 = 9개입니다.
이때 F는 쌍의 첫 번째 바닥 계수이고 두 번째 계수는 1-F입니다.
각 seed에서 F를 고정하고 D2의 4밴드 C4를 최소화하는 연속 수심 z를 찾습니다.
'공통수심'은 한 seed의 네 밴드에 공통인 z이며 9개 seed의 z가 같다는 뜻은 아닙니다.
아직 상위 5개 선정이나 수심·F 번갈아 이동은 하지 않습니다.
"""
import math
import numpy as np
from scipy.optimize import minimize_scalar
from settings import PAIRS
from search_state import CandidateState, EvaluationLimit

def initialize_seed(seed):
    """한 seed의 초기 F를 고정한 채 초기 수심을 스칼라 최적화로 구합니다.

    입력·출력: CandidateState 객체에 초기 z·C4·수치해법 상태를 기록하고 C4를 반환합니다.
    출처: G:/태평양/8차/4번알고리즘완성/동시후보_QAA전용_재계산/code/invert.py:108.
    Brent 방법은 격자 전체를 훑는 방법이 아니라 C4를 보며 실수 수심을 탐색합니다.
    bracket=(0,1)은 탐색을 시작할 두 점이지 수심을 1 m 이하로 제한하는 범위가 아닙니다.
    xtol=1e-4는 기존 수치해법의 상대 허용오차이며 0.25 m 이동 스텝과 별개입니다.

    기존 8차 수치해법에는 음수 수심 제한이 없었지만, 현재 코드는 PPT의 z>=0을 유지합니다.
    z=0 경계도 비교하며 음수 수심은 C4=무한대로 거절합니다.
    수치해법이 실패하거나 계산 한도에 닿아도 그때까지의 최저 유한 C4를 보관합니다.
    그 결과를 성공으로 바꿔 기록하지 않고 initial_status에 실패·한도 상태를 남깁니다.
    한 번 초기화한 객체를 다시 초기화하지 못하게 해 계산 횟수 중복을 막습니다.
    """
    if seed.initial_status != "uncomputed":
        raise ValueError("Each candidate may only be initialized once")
    start_evaluations = seed.evaluations

    def objective(z):
        """Brent가 요청한 수심 한 점의 C4를 계산하고 시도 이력을 남깁니다.

        이 단계에서는 F가 initial_f로 고정됩니다. 같은 수심을 다시 물으면 저장값을 씁니다.
        counted는 이번 호출에서 실제 모델 계산 횟수가 늘었는지 나타냅니다.
        """
        z = float(z)
        if z in seed.initial_values:
            return seed.initial_values[z]
        before = seed.evaluations
        value = seed._evaluate_at(z, seed.initial_f)
        seed.initial_values[z] = value
        seed.initial_trials.append(dict(z=z, c4=value, evaluations=seed.evaluations,
                                        counted=seed.evaluations > before))
        return value

    try:
        # Brent가 최소점을 감싸는 구간을 찾지 못하거나 계산 한도에 닿아도
        # 수심 0 경계값과 비교할 수 있도록 먼저 평가합니다.
        objective(0.0)
        # 정의역 밖 점을 무한대 오차로 거절할 때 Brent 내부에서 inf-inf가 생길 수 있습니다.
        # 반환 상태와 초기 수심의 모든 시도는 별도 기록에 그대로 남습니다.
        # 정의역 밖 점을 무한대 오차로 거절할 때 생길 수 있는 수치 경고만 억제합니다.
        # 값을 다른 오차로 바꾸거나 solver 실패 상태를 숨기는 처리는 아닙니다.
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            # F는 고정한 채 네 밴드 공통수심 z 하나를 찾습니다.
            # 시작 bracket의 1.0은 수심 상한이 아니므로 더 깊은 수심도 탐색할 수 있습니다.
            result = minimize_scalar(objective, method="brent", bracket=(0.0, 1.0),
                options={"xtol": seed.settings.initial_xtol,
                         "maxiter": seed.settings.initial_maxiter})
        seed.initial_solver_success = bool(result.success and math.isfinite(float(result.fun)))
        seed.initial_solver_message = str(result.message)
        seed.initial_nit = int(result.nit) if hasattr(result, "nit") else None
        seed.initial_solver_nfev = int(result.nfev) if hasattr(result, "nfev") else None
        if result.success and math.isfinite(float(result.fun)):
            seed.initial_status = "success"
        elif (seed.initial_nit is not None and seed.initial_nit >= seed.settings.initial_maxiter
              or "maximum" in str(result.message).lower()):
            seed.initial_status = "maxiter"
        else:
            seed.initial_status = "no_bracket"
    except EvaluationLimit:
        seed.initial_status = "evaluation_limit"
        seed.initial_solver_message = "Per-seed C4 evaluation limit reached during initialization"
    except (ValueError, RuntimeError, OverflowError) as exc:
        # SciPy 버전에 따라 구간 탐색 실패가 예외 또는 실패 결과로 반환될 수 있습니다.
        # 어느 경우든 초기화 성공으로 바꿔 적지 않습니다.
        seed.initial_status = "no_bracket"
        seed.initial_solver_message = str(exc)
    seed.initialization_evaluations = seed.evaluations - start_evaluations
    # solver가 마지막에 보고한 점만 믿지 않고 실제 계산한 유한한 점들을 비교합니다.
    # 초기화 상태는 별도로 남겨 미완료 결과가 성공처럼 보이지 않게 합니다.
    finite = [(value, z) for z, value in seed.initial_values.items()
              if z >= 0 and math.isfinite(z) and math.isfinite(value)]
    if not finite:
        raise ValueError(f"No finite initial candidate ({seed.initial_status}): {seed.initial_solver_message}")
    seed.initial_cost, seed.z_origin = min(finite, key=lambda item: (item[0], item[1]))
    seed.initial_boundary_selected = seed.z_origin == 0.0
    # 초기 수심 자체가 격자의 원점입니다. 수심 0 m에서 격자를 새로 만들지 않습니다.
    seed.initial_key = seed.best_key = (0, 0)
    seed.best_cost = seed.initial_cost
    seed.cache[seed.initial_key] = seed.initial_cost
    return seed.initial_cost

def build_initial_seeds(observed, context, library, settings):
    """세 쌍 × 세 초기 F의 모든 후보를 생성하고 빠짐없이 초기화합니다.

    관측값·QAA 문맥·바닥 라이브러리·설정만 받습니다. 정답 자료를 받지 않습니다.
    여기서 9개의 초기 C4가 모두 정해져야 4단계에서 같은 기준으로 순위를 매길 수 있습니다.
    """
    settings.validate()
    seeds = [CandidateState(observed, context, library, pair, fraction, settings)
             for pair in PAIRS for fraction in settings.seed_fractions]
    for seed in seeds:
        initialize_seed(seed)
    return seeds
