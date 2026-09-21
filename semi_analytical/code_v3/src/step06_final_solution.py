"""6단계: 5개 최적화 결과를 비교하여 대표해와 추가 후보를 표시합니다.

대표: C4가 가장 작은 결과. 평균을 내거나 실제 정답을 보고 다시 고르지 않습니다.
같은 해: 수심과 모래·갈조·녹조 세 계수가 모두 같은 결과.
다른 근접 후보: 같은 해가 아니면서 C4가 대표 C4의 1.01배 이하인 결과.
표시 여부와 관계없이 최적화한 5개는 모두 저장합니다.
"""
import math
import numpy as np
from settings import MATERIALS

def classify_candidates(records, settings):
    """최종 C4 순위·동일해 그룹·추가 후보 표시와 대표 요약을 반환합니다.

    records는 5단계 결과 사전 목록입니다. 각 사전에 순위와 표시를 추가합니다.
    반환값 summary는 대표 1개의 정보, ranked는 순위를 붙인 전체 후보 목록입니다.

    동일해 검사는 [z, f_sand, f_brown, f_green]으로 합니다.
    따라서 '갈조–모래 10:90'과 '모래–갈조 90:10'을 이름 때문에 다른 답으로 보지 않습니다.
    수심이 다르면 바닥 계수가 같아도 다른 해입니다.
    same_solution_atol=1e-9는 기존 수치적 동등성 비교용이며 관측 정확도 기준이 아닙니다.

    near_best는 C4 조건만, same_solution_duplicate는 이미 나온 같은 해인지 표시합니다.
    is_alternative는 대표와 다른 해의 그룹에서 처음 나온 근접 후보에만 붙습니다.
    예: 최소 C4=0.01000이면 0.01010까지입니다. 1%는 수심 오차 1%가 아닙니다.
    최소 C4=0이면 0인 후보만 해당합니다. 별도의 작은 상수를 추가하지 않습니다.
    alternative_z_min/max는 저장된 후보의 범위이며 통계적인 신뢰구간이 아닙니다.
    """
    ranked = sorted(records, key=lambda r:(r['c4'],r['seed_id']))
    if not ranked or not math.isfinite(ranked[0]['c4']):
        raise ValueError("No finite optimized candidates")
    representatives = []
    for rank, row in enumerate(ranked, 1):
        # 바닥쌍 이름이나 F 하나만 비교하면 같은 구성을 놓칠 수 있습니다.
        # 수심과 세 바닥 계수를 함께 비교하므로 순수 갈조가 여러 경로에서 나와도 구분됩니다.
        signature = np.array([row['z_hat'], *(row['f_'+m] for m in MATERIALS)])
        match = next((i for i, rep in enumerate(representatives)
                      if np.allclose(signature, rep[0], rtol=0, atol=settings.same_solution_atol)), None)
        row['rank'] = rank
        row['same_solution_duplicate'] = match is not None
        if match is None:
            match = len(representatives)
            representatives.append((signature, row))
        row['solution_group'] = match+1
        row['same_solution_as'] = representatives[match][1]['seed_id']
    winner = ranked[0]
    # 허용 간격은 대표 C4의 1%입니다. C4에 0.01을 더하는 조건이 아닙니다.
    cutoff = winner['c4']*(1+settings.near_relative_c4)
    for row in ranked:
        row['near_best'] = bool(row['c4'] <= cutoff)
        row['relative_c4_gap'] = ((row['c4']-winner['c4'])/winner['c4']
                                  if winner['c4'] > 0 else (0.0 if row['c4'] == 0 else math.inf))
        row['is_representative'] = row['seed_id'] == winner['seed_id']
        # 같은 추가 해가 여러 seed에서 반복되어도 추가 후보 수를 중복해서 세지 않습니다.
        # 하지만 그 원래 seed 행들은 ranked에 그대로 남습니다.
        row['is_alternative'] = row['near_best'] and row['solution_group'] != winner['solution_group'] and not row['same_solution_duplicate']
    alternatives = [row for row in ranked if row['is_alternative']]
    # 대표 요약은 최소 C4 행을 복사합니다. 여러 후보를 평균내지 않습니다.
    summary = dict(winner)
    summary.update(has_alternative=bool(alternatives), alternative_count=len(alternatives),
                   unique_solution_count=len(representatives),
                   same_solution_duplicate_count=len(ranked)-len(representatives),
                   total_evaluations=sum(r['evaluations'] for r in ranked),
                   any_evaluation_limit=any(r['status']=='evaluation_limit' for r in ranked),
                   alternative_z_min=min([winner['z_hat']]+[r['z_hat'] for r in alternatives]),
                   alternative_z_max=max([winner['z_hat']]+[r['z_hat'] for r in alternatives]))
    return summary, ranked
