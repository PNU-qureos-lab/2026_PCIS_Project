"""5단계: 선정된 5개 seed에서 수심·혼합비의 주변 8방향을 비교합니다.

한 위치에서 (Δz,ΔF)=(±0.1,0),(0,±0.01),(±0.1,±0.01)의 8점을 모두 비교합니다.
현재 C4보다 작은 후보 중 최저 C4로 한 번 이동하고, 새 위치에서 다시 8점을 봅니다.
8방향 모두 개선이 없으면 no_c4_improvement로 종료합니다.
초기 후보·F 무제약·4밴드 C4·평가 한도는 기존 9차와 같습니다.
계산 한도로 멈추는 evaluation_limit는 위의 정상 종료와 구분합니다.
"""
import math
import numpy as np
from model import forward, components, MATERIALS
from search_state import EvaluationLimit

# 정수 오프셋에 Settings의 수심·F 이동 간격을 곱해 실제 좌표를 구합니다.
# (0,0)은 현재점이므로 제외합니다. 두 값이 0이 아니면 대각선 동시 이동입니다.
NEIGHBOR_OFFSETS = tuple((dz, df) for dz in (-1, 0, 1) for df in (-1, 0, 1)
                         if (dz, df) != (0, 0))

def optimize_seed(seed):
    """초기화된 한 seed를 고정 스텝으로 이동하고 최종 후보 한 행을 반환합니다.

    key=(nz,nf)는 초기값에서 수심·F를 몇 번 움직였는지를 나타내는 정수 좌표입니다.
    실제 값은 z=z_origin+nz*0.1, F=initial_f+nf*0.01입니다.
    상하좌우 네 방향과 대각선 네 방향을 같은 현재 위치에서 비교합니다.
    현재보다 C4가 엄격히 작을 때만 이동하므로 같은 오차 사이를 왕복하지 않습니다.

    정상 종료는 현재 격자 위치의 주변 8방향에서 더 개선되지 않았다는 뜻입니다.
    전체 수심·혼합비 영역의 전역 최저 C4를 보장하는 판정은 아닙니다.
    한도에 닿으면 이미 실제 평가한 점 중 가장 좋은 점을 보존합니다.
    반환값에는 수심 z_hat, 쌍 기준 F, 세 바닥 계수, C4, 종료 이유, 보정 없는 예측 Rrs,
    계산 횟수와 범위 밖 표시가 포함됩니다. 세부 이동은 seed.history에 남습니다.
    f_ge1은 F>=1 표시이고 f_outside는 F<0 또는 F>1입니다. 순수 바닥 F=1은 범위 안입니다.
    """
    key, cost = seed.initial_key, seed.initial_cost
    if key is None:
        raise ValueError("Initialize before optimization")
    seed.log(key, cost, 0, "initial")
    cycle, reason = 0, "no_c4_improvement"
    try:
        if seed.initial_status == "evaluation_limit":
            raise EvaluationLimit
        while True:
            cycle += 1
            options = []
            # 8점을 모두 평가하기 전에는 현재 위치를 바꾸지 않습니다.
            # 음수 수심 거절·평가 횟수·캐시는 기존 seed.evaluate가 처리합니다.
            for dz, df in NEIGHBOR_OFFSETS:
                neighbor = (key[0] + dz, key[1] + df)
                options.append((seed.evaluate(neighbor), neighbor))
            # 동점이면 기존 좌표 튜플 순서로 고릅니다. 정답으로 동점을 깨지 않습니다.
            next_cost, next_key = min(options, key=lambda item: (item[0], item[1]))
            if not next_cost < cost:
                break
            dz, df = next_key[0] - key[0], next_key[1] - key[1]
            event = "move_diagonal" if dz and df else "move_depth" if dz else "move_fraction"
            key, cost = next_key, next_cost
            # cycle 한 번은 8방향 비교 한 번이며, 실제 이동 방향을 구분해 기록합니다.
            seed.log(key, cost, cycle, event)
    # 오차 개선 없음과 계산 횟수 한도는 의미가 다르므로 종료 이름도 구분합니다.
    except EvaluationLimit:
        reason = "evaluation_limit"
        # 다음 점 평가가 한도에 막혀도 이미 찾은 더 좋은 점은 보존합니다.
        key, cost = seed.best_key, seed.best_cost
    seed.log(key, cost, cycle, reason)
    z, f = seed.coordinates(key)
    iz, _ = seed.coordinates(seed.initial_key)
    pred = forward(z, seed.rho(f), seed.context)
    # 최종 F를 모래·갈조·녹조 공통 계수로 바꿔 서로 다른 바닥쌍도 비교 가능하게 합니다.
    cf = components(seed.pair, f)
    return dict(seed_id=seed.seed_id, pair="_".join(seed.pair),
                initial_F=seed.initial_f, initial_z=iz, initial_c4=seed.initial_cost,
                z_hat=z, F=f, **{f"f_{m}":cf[m] for m in MATERIALS},
                c4=float(cost), rms_relative_error_pct=100*math.sqrt(cost),
                status=reason, cycles=cycle, evaluations=seed.evaluations,
                **seed.initial_metadata(),
                f_negative=f < 0, f_ge1=f >= 1, f_outside=(f < 0 or f > 1),
                fraction_label="negative" if f < 0 else "above_one" if f > 1 else "pure_first" if f == 1 else "pure_second" if f == 0 else "within_range",
                bottom_reflectance_negative=bool(np.any(seed.rho(f) < 0)),
                bottom_reflectance_above_one=bool(np.any(seed.rho(f) > 1)),
                **{f"Rrs_model_{w}":float(v) for w,v in zip((443,490,560,665),pred)})

def optimize_seeds(selected_seeds):
    """선정된 각 seed를 독립적으로 최적화하고 결과와 이동 기록을 모읍니다.

    한 seed에서 찾은 수심·혼합비로 다른 seed의 시작점을 바꾸지 않습니다.
    결과 5개의 순위와 대표 선정은 다음 6단계에서 수행합니다.
    """
    records, traces = [], []
    for seed in selected_seeds:
        records.append(optimize_seed(seed))
        traces.extend(seed.history)
    return records, traces
