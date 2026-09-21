"""결과가 계산 규칙을 지켰는지 검사하고 선정 완료 후 실제 정답과 비교합니다.

verify_case는 C4 순위·스텝·횟수를 확인하며 정답을 필요로 하지 않습니다.
evaluate_after_selection은 선정된 대표들의 성능을 계산합니다.
compare_candidates_with_truth는 다섯 후보를 사후 대조하지만 대표를 바꾸지 않습니다.
2위가 실제 정답에 더 가까워도 주요 성능은 C4로 선정한 대표 기준입니다.
"""
import math
import numpy as np
import pandas as pd
from settings import MATERIALS
from io_utils import save_json, save_csv

def verify_case(estimate, candidates, trace, settings, initials=None):
    """한 사례의 저장 결과가 합의된 추정 규칙을 지켰는지 확인합니다.

    초기 C4 상위 k개만 최적화했는지, 최종 대표가 최소 C4인지,
    기록된 수심·F 이동이 설정한 고정 스텝인지, C4가 증가하지 않았는지 검사합니다.
    계산 한도와 근접 후보 표시도 확인합니다.
    실제 정답에 가까운지를 검사하는 함수가 아니며 계산 결과의 정확도를 보증하지 않습니다.
    """
    if not candidates or not math.isfinite(estimate['c4']):
        raise AssertionError('No finite result')
    if len(candidates) != settings.top_k:
        raise AssertionError('Incorrect number of optimized seeds')
    if initials is not None:
        expected = sorted(initials, key=lambda r:(r['initial_c4'], r['seed_id']))[:settings.top_k]
        if len(initials) != 9 or {r['seed_id'] for r in candidates} != {r['seed_id'] for r in expected}:
            raise AssertionError('Seeds do not match the lowest initial C4 ranks')
    if estimate['c4'] != min(r['c4'] for r in candidates):
        raise AssertionError('Representative is not the minimum C4')
    if max(r['evaluations'] for r in candidates) > settings.max_evaluations:
        raise AssertionError('Per-seed evaluation limit exceeded')
    if not math.isclose(sum(estimate['f_'+m] for m in MATERIALS),1.0,abs_tol=1e-10):
        raise AssertionError('Canonical coefficients do not sum to one')
    for row in candidates:
        if row['z_hat'] < 0 or row['c4'] > row['initial_c4']:
            raise AssertionError('Invalid depth or degraded candidate')
        if row['is_alternative'] and row['c4'] > estimate['c4']*(1+settings.near_relative_c4):
            raise AssertionError('Alternative criterion not respected')
    prior = {}
    for row in trace:
        prev = prior.get(row['seed_id'])
        if prev:
            if row['c4'] > prev['c4']:
                raise AssertionError('Accepted trajectory increased C4')
            if row['event'] == 'move_fraction':
                if row['z'] != prev['z'] or not math.isclose(abs(row['F']-prev['F']),settings.fraction_step,abs_tol=1e-9):
                    raise AssertionError('Fraction step changed')
            elif row['event'] == 'move_depth':
                if row['F'] != prev['F'] or not math.isclose(abs(row['z']-prev['z']),settings.depth_step,abs_tol=1e-9):
                    raise AssertionError('Depth step changed')
            elif row['event'] == 'move_diagonal':
                # 대각선도 수심·F 각각의 설정 간격을 정확하게 따라야 합니다.
                if (not math.isclose(abs(row['z']-prev['z']),settings.depth_step,abs_tol=1e-9)
                        or not math.isclose(abs(row['F']-prev['F']),settings.fraction_step,abs_tol=1e-9)):
                    raise AssertionError('Diagonal step changed')
        prior[row['seed_id']] = row


def error_metrics(truth, estimate):
    """정답과 추정의 차이로 n·MAE·RMSE·Bias를 구합니다.

    Bias는 평균(추정-정답)입니다. 비유한 오차는 계산에서 빼고 n에 실제 포함 수를 기록합니다.
    이 결과는 이미 고른 대표의 사후 성능이며 선정 비용 C4와는 구분합니다.
    """
    error = np.asarray(estimate,float)-np.asarray(truth,float)
    finite = np.isfinite(error)
    return dict(n=int(finite.sum()), MAE=float(np.mean(np.abs(error[finite]))),
                RMSE=float(np.sqrt(np.mean(error[finite]**2))),Bias=float(np.mean(error[finite])))


def evaluate_after_selection(rows, output):
    """대표 한 개씩을 전체·바닥 그룹별로 평가하여 metrics.csv와 summary.json에 저장합니다.

    수심과 모래·갈조·녹조 세 계수를 각각 평가합니다.
    C4가 유한한 대표들은 F 범위 밖 여부나 근접 후보 존재 여부로 걸러내지 않습니다.
    실패·한도 종료·음수 F·1 초과 F의 개수도 따로 남깁니다.
    """
    df = pd.DataFrame(rows)
    good = df[np.isfinite(df['c4'])].copy()
    metrics = []
    for group, frame in [('all',good),*list(good.groupby('true_group',sort=True))]:
        for quantity, truth, estimate in [('depth','true_z','z_hat'),
                                        *((m,'true_f_'+m,'f_'+m) for m in MATERIALS)]:
            metrics.append(dict(group=group,quantity=quantity,**error_metrics(frame[truth],frame[estimate])))
    save_csv(metrics,output/'metrics.csv')
    summary = dict(cases=len(df),finite_estimates=len(good),failed_cases=len(df)-len(good),
                   status_counts=df['status'].value_counts().to_dict(),
                   has_alternative_count=int(good['has_alternative'].sum()),
                   fraction_outside_count=int(good['f_outside'].sum()),
                   fraction_negative_count=int(good['f_negative'].sum()),
                   fraction_above_one_count=int((good['F']>1).sum()),
                   selected_at_evaluation_limit=int((good['status']=='evaluation_limit').sum()),
                   cases_with_any_evaluation_limit=int(good['any_evaluation_limit'].sum()),
                   median_c4=float(good['c4'].median()),max_c4=float(good['c4'].max()),
                   min_depth_hat=float(good['z_hat'].min()),max_depth_hat=float(good['z_hat'].max()),
                   depth_metrics=error_metrics(good['true_z'],good['z_hat']),
                   all_estimates_in_metrics=True)
    save_json(summary,output/'summary.json')
    return summary


def compare_candidates_with_truth(estimates_csv, candidates_csv, output_dir):
    """저장된 다섯 후보 모두를 실제 정답에 대조하는 사후 진단입니다.

    수심 절대오차와 세 바닥 계수의 제곱평균제곱근 오차를 각각 구합니다.
    대표보다 수심이 더 가까운지, 바닥이 더 가까운지, 둘 다 가까운지 표시합니다.
    1% 밖 후보도 표에 남겨 '가까운 후보만 확인해서 놓치는 경우'를 볼 수 있습니다.
    후보 순위·대표 CSV·주요 성능을 덮어쓰지 않으며 정답을 이용한 재선정을 하지 않습니다.

    1e-12는 기존 비교표의 반올림 동률 처리값입니다.
    오차가 그 이하이면 정답이라고 판단하는 허용오차가 아닙니다.
    반환 요약과 candidate_truth_summary.json에서 2위가 더 가까운 사례 수를 확인합니다.
    """
    from pathlib import Path
    estimates = pd.read_csv(estimates_csv, float_precision='round_trip')
    candidates = pd.read_csv(candidates_csv, float_precision='round_trip')
    refs = estimates[['case_id','true_z',*[f'true_f_{m}' for m in MATERIALS]]]
    rows = candidates.merge(refs, on='case_id', how='left', validate='many_to_one')
    rows['depth_abs_error_m'] = np.abs(rows.z_hat-rows.true_z)
    squared = [(rows[f'f_{m}']-rows[f'true_f_{m}'])**2 for m in MATERIALS]
    rows['bottom_3component_rmse'] = np.sqrt(np.mean(squared, axis=0))
    baseline = rows.loc[rows.is_representative,
        ['case_id','depth_abs_error_m','bottom_3component_rmse']].rename(columns={
            'depth_abs_error_m':'representative_depth_abs_error_m',
            'bottom_3component_rmse':'representative_bottom_3component_rmse'})
    rows = rows.merge(baseline, on='case_id', validate='many_to_one')
    rows['closer_depth_than_representative'] = (
        rows.depth_abs_error_m < rows.representative_depth_abs_error_m-1e-12)
    rows['closer_bottom_than_representative'] = (
        rows.bottom_3component_rmse < rows.representative_bottom_3component_rmse-1e-12)
    rows['closer_in_both'] = rows.closer_depth_than_representative & rows.closer_bottom_than_representative
    rows = rows.sort_values(['case_id','rank'])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    save_csv(rows.to_dict('records'),output_dir/'candidate_truth_comparison.csv')
    rank2 = rows[rows['rank']==2]
    near = rows[rows.is_alternative]
    counts = dict(cases=int(rows.case_id.nunique()), candidates=len(rows),
        near_alternative_cases=int(near.case_id.nunique()),
        near_with_closer_depth=int(near.loc[near.closer_depth_than_representative,'case_id'].nunique()),
        near_with_closer_bottom=int(near.loc[near.closer_bottom_than_representative,'case_id'].nunique()),
        rank2_closer_depth=int(rank2.closer_depth_than_representative.sum()),
        rank2_closer_bottom=int(rank2.closer_bottom_than_representative.sum()),
        rank2_closer_both=int(rank2.closer_in_both.sum()))
    summary = dict(counts=counts, comparison_tie_allowance=1e-12,
        correctness_tolerance=None, representative_reselected=False,
        main_performance_metrics_changed=False,
        meaning='Post-estimation diagnosis using known truth, not deployable selection performance')
    save_json(summary,output_dir/'candidate_truth_summary.json')
    return summary
