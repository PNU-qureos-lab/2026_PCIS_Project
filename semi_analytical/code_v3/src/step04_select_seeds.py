"""4단계: 초기 C4를 비교하여 9개 중 5개 seed만 반복 최적화 대상으로 고릅니다.

초기 수심을 다시 계산하지 않습니다. 최종 정답과의 거리도 보지 않습니다.
선정되지 않은 4개의 초기값과 순위도 seed_ranking.csv에 남깁니다.
"""

def select_seeds(seeds, settings):
    """초기 C4 오름차순으로 정렬하고 앞의 top_k개 객체를 반환합니다.

    동일 C4이면 seed_id 순으로 정해 실행마다 순위가 뒤집히지 않게 합니다.
    반환값 selected는 5단계에 넘길 객체 목록, rows는 9개 전체의 확인용 기록입니다.
    rows의 selected가 True인 후보만 반복 이동합니다.
    이 단계에서는 중복해를 먼저 제거해서 새로운 시작점을 보충하지 않습니다.
    합의한 '초기 9개 중 C4가 작은 5개'를 그대로 적용하고 중복 표시는 6단계에서 합니다.
    """
    # 여기서 보는 점수는 반복 최적화 전 initial_cost입니다.
    # 미리 최적화한 결과나 실제 정답으로 상위 5개를 뽑는 것이 아닙니다.
    ranked = sorted(seeds, key=lambda seed:(seed.initial_cost, seed.seed_id))
    if len(ranked) < settings.top_k:
        raise ValueError("Not enough initialized seeds for top_k")
    rows = []
    for index, seed in enumerate(ranked):
        z, _ = seed.coordinates(seed.initial_key)
        rows.append(dict(seed_id=seed.seed_id, initial_rank=index+1,
            initial_F=seed.initial_f, initial_z=z, initial_c4=seed.initial_cost,
            selected=index < settings.top_k, **seed.initial_metadata(),
            initial_trials=seed.initial_trials))
    # 앞의 다섯 객체만 5단계에 전달하고 아홉 개의 순위 기록은 모두 보관합니다.
    return ranked[:settings.top_k], rows
