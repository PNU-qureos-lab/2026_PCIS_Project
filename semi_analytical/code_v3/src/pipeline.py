"""한 관측 사례를 3 → 4 → 5 → 6단계에 순서대로 통과시킵니다.

수식과 탐색 규칙은 각 단계 파일에 있습니다. 여기서는 호출 순서만 연결합니다.
정답 자료를 받지 않아 실제 수심·바닥으로 seed 또는 대표를 고를 수 없습니다.
"""
from dataclasses import asdict
from settings import Settings
from step03_initial_seeds import build_initial_seeds
from step04_select_seeds import select_seeds
from step05_optimize_seeds import optimize_seeds
from step06_final_solution import classify_candidates

def solve(observed, context, library, settings=None):
    """관측 Rrs 한 개의 최종 대표와 모든 중간 기록을 반환합니다.

    observed: 네 밴드 얕은 물 Rrs. context: 2단계의 D2 계수.
    library: 순수 바닥 3종. settings: 고정 스텝·k 등 실행 설정.
    반환 순서: 대표 요약, 순위가 붙은 5개 후보, 이동 기록, 초기 9개 기록.
    총 계산 횟수에는 선정되지 않은 4개 seed의 초기 수심 계산도 포함합니다.
    """
    settings = settings or Settings()
    settings.validate()
    seeds = build_initial_seeds(observed, context, library, settings)  # 3
    selected, initials = select_seeds(seeds, settings)                 # 4
    records, traces = optimize_seeds(selected)                         # 5
    summary, records = classify_candidates(records, settings)          # 6
    summary['total_evaluations'] = sum(seed.evaluations for seed in seeds)
    summary['settings'] = asdict(settings)
    return summary, records, traces, initials

def solve_payload(payload):
    """일반 실행 또는 병렬 작업에서 입력 묶음을 받아 solve를 호출합니다.

    예상되는 계산 오류는 문자열로 돌려주어 run_all.py가 해당 사례의 실패로 기록합니다.
    실패한 사례에 다른 사례의 추정값이나 실제 정답을 채워 넣지 않습니다.
    """
    observed, context, library, settings = payload
    try:
        return solve(observed, context, library, settings)
    except (ValueError, AssertionError, RuntimeError) as exc:
        return repr(exc)
