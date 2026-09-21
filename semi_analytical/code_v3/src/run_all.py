"""7단계 전체 실행의 시작 파일입니다. 코드를 읽을 때도 여기서 흐름을 볼 수 있습니다.

1 네 밴드 Rrs 자료 정리 → 2 깊은 물 HOPE P/G/X → 사례마다 3 초기 9개 → 4 상위 5개
→ 5 후보별 반복 최적화 → 6 최종해 선정·기록 → 사후 정답 평가 → 7 그림.
모든 사례가 완료되면 각 단계 이름의 하위 폴더에 CSV·JSON과 그림이 남습니다.

기본 실행: python run_all.py --mode pilot  (기존 검증용 36개와 그림)
전체 실행: python run_all.py --mode full --workers 4  (4,860개)
그림 생략: 위 명령 뒤에 --no-figures 추가.
같은 출력 폴더로 재실행하면 해당 결과 파일을 덮어씁니다.
v2 초기식과 27개 시작점은 hope_iop_model.py, 피팅 연결은 step02_hope_pgx.py,
15차 IOP는 14차의 실제 PGX 좌표 최급강하 + Brent 선 최소화에 PGX>=0를 추가합니다.
P=0은 ln(P)로 제외하며 G/X=0 경계에서는 허용 방향과 KKT 잔차를 사용합니다.
수심/F는 12차 v2의 상위 5개 seed + 수심 0.1 m/F 0.01 고정 8방향 탐색입니다.
F는 제한하지 않으며 수심 정의역·초기화·종료 판정도 12차 v2 그대로입니다.
관측·PGX 초기식·IOP·H/F에는 443/490/560/665 nm만 전달합니다.
기존 D2 모델 수식은 유지하며 이 파일에서는 단계 연결과 저장을 담당합니다.
QAA 기준 결과는 추정이 끝난 뒤 비교에만 읽습니다. QAA 계산을 호출하지 않습니다.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from settings import Settings, SETTING_NOTES, DEFAULT_DATA, DEFAULT_BOTTOM, MATERIALS
from model import WAVES, LEE_METHOD, forward, make_context, c4
from step01_rrs_data import load_dataset, select_cases, prepare_inputs, save_prepared_tables
from step02_hope_pgx import HOPE_METHOD, HOPE_CONFIG, derive_contexts
from pipeline import solve_payload
from evaluation import verify_case, evaluate_after_selection, compare_candidates_with_truth
from io_utils import save_json, save_csv, json_default, CsvStream

ROOT = Path(__file__).resolve().parent
# 방법 이름을 결과 설정에 분리해 기록합니다. IOP의 기울기 수렴과 H/F 격자의
# 주변 개선 없음은 서로 다른 종료 조건이며, 둘을 같은 최급강하법으로 부르지 않습니다.
EXPERIMENT_CONFIG = dict(
    name='15th_hybrid_12th_HF_nonnegative_IOP',
    iop_reference='G:/태평양/14차',
    hf_reference='G:/태평양/12차/iteration1HOPE_v2',
    iop_method='14th PGX steepest descent with nonnegative feasible directions and Brent line search',
    iop_parameter_bounds={'P':[0.0,None], 'G':[0.0,None], 'X':[0.0,None]},
    iop_formula_domain='P>0 for ln(P); G>=0; X>=0',
    iop_solver_entry='steepest_descent.solve_pgx; shared solve_hf is not called by this pipeline',
    hf_method='12th v2 original eight-neighbor fixed-grid descent; top 5 of 9 initial seeds',
    hf_termination='no strictly smaller C4 among eight neighbors; evaluation limit separate',
    hf_continuous_gradient_stationarity_checked=False,
    fraction_bounds=None,
    hf_depth_policy='original 12th v2 nonnegative depth and initialization boundary retained',
    hf_interface_policy='original 12th v2 search requires positive surface-transform denominator',
    global_minimum_certified=False)
STAGE_FOLDERS = {
    1:'01_rrs_data', 2:'02_hope_pgx', 3:'03_initial_seeds', 4:'04_seed_selection',
    5:'05_optimization', 6:'06_final_solution', 7:'07_figures',
}


def solve_with_iop_status(payload):
    """IOP가 실패한 조건을 명시적으로 돌려주고 나머지는 원래 H/F 함수에 넘깁니다.

    pipeline.solve_payload의 기존 입출력 연결 구조를 유지합니다.
    두 번째 값이 실패 사유 문자열이면 그 조건의 얕은 물을 계산하지 않습니다.
    정상 context가 있으면 기존 함수에 같은 네 항목만 전달합니다.
    Windows 병렬 처리에서 사용하므로 중첩 함수가 아닌 모듈 최상위에 정의합니다.
    """
    if isinstance(payload[1], str):
        return payload[1]
    return solve_payload(payload)


def run(args):
    """입력 경로·실행 모드를 받아 1~7단계와 저장·검증을 수행합니다.

    중요한 경계: obs에는 관측만, references에는 실제 정답을 보관합니다.
    solve_payload에 넘기는 값은 관측·HOPE IOP 문맥·바닥 라이브러리·설정뿐입니다.
    6단계가 대표를 고른 뒤에 실제값을 결합하여 평가표와 그림을 만듭니다.

    출력 폴더 01~07은 사용자 지정 단계와 같습니다.
    03에는 초기 9개, 04에는 초기 순위와 selected, 05에는 이동 경로,
    06에는 대표·전체 후보·근접 후보·사후 정답 대조, 07에는 그림을 저장합니다.
    --workers는 서로 다른 사례를 병렬로 계산하는 수입니다. seed 수 k와는 다릅니다.
    run_config.json은 사용 설정, run_manifest.json은 실행 결과와 코드 해시를 남깁니다.
    """
    started = time.perf_counter()
    # 코드를 G:로 옮겼다면 기본 결과도 그 코드 폴더의 runs 아래에 저장됩니다.
    # 입력 데이터 경로 자체는 settings.py의 G:/태평양/9차 경로를 사용합니다.
    output = (args.output or ROOT/'runs'/args.mode).resolve()
    for source in (args.data.resolve(),args.bottom.resolve()):
        if output==source or source in output.parents:
            raise ValueError('Output must not be inside a source-data directory')
    output.mkdir(parents=True,exist_ok=True)
    folders = {number:output/name for number,name in STAGE_FOLDERS.items()}
    for folder in folders.values():
        folder.mkdir(exist_ok=True)
    settings = Settings(top_k=args.top_k)
    settings.validate()

    # 1. 관측 Rrs·깊은 물·바닥 라이브러리 정리. 참값은 references에만 둡니다.
    dataset = load_dataset(args.data,args.bottom,cache_path=args.cache)
    cases = select_cases(dataset,args.mode)
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError('No selected cases')
    observations, deep_inputs, references = prepare_inputs(dataset,cases)
    save_prepared_tables(observations,deep_inputs,dataset['bottoms'],references,dataset['audit'],folders[1])
    # 깊은 물 정답은 별도 평가 파일로만 보관합니다. 아래 derive_contexts에는
    # prepare_inputs에서 분리한 deep_inputs만 전달하므로 이 총 a/bb를 볼 수 없습니다.
    # pilot의 얕은 물은 1조건이어도 깊은 물 IOP 산점도는 27조건 전체를 평가합니다.
    save_csv([dict(water_key=key, solar_zenith=row['solar_zenith'],
                   **{f'{field}_{int(w)}':float(row[field][j])
                      for field in ('a_true','bb_true') for j,w in enumerate(WAVES)})
              for key,row in sorted(dataset['infinite'].items())],
             folders[1]/'deep_reference_truth.csv')
    print(f'1/7 Rrs data prepared: {len(observations)} cases',flush=True)

    # 2. 깊은 물 Rrs + aw/bbw만으로 P/G/X를 맞춥니다. 최적화 내부의 깊은 물
    # 반사도는 모델값입니다. 성공한 뒤 H/F context는 기존 관측 deep 항을 유지합니다.
    iop_started = time.perf_counter()
    print(f'2/7 Fitting HOPE P/G/X: {len(deep_inputs)} conditions x 27 starts',flush=True)
    iop_result = derive_contexts(deep_inputs)
    contexts, iops = iop_result['contexts'], iop_result['iops']
    save_csv(iops,folders[2]/'iops.csv')
    save_csv(iop_result['parameters'],folders[2]/'pgx_parameters.csv')
    save_csv(iop_result['multistart'],folders[2]/'multistart.csv')
    save_csv(iop_result['trajectory'],folders[2]/'pgx_iteration_trace.csv')
    save_json(iop_result['failures'],folders[2]/'failures.json')
    iop_seconds = time.perf_counter()-iop_started
    print(f'2/7 HOPE P/G/X: {len(contexts)}/{len(deep_inputs)} conditions; '
          f'{iop_seconds:.2f} seconds',flush=True)
    save_json(dict(experiment=EXPERIMENT_CONFIG,settings=asdict(settings),setting_notes=SETTING_NOTES,
        data=str(args.data),bottom=str(args.bottom),mode=args.mode,case_count=len(cases),
        bands_nm=WAVES,iop_method=HOPE_METHOD,hope_config=HOPE_CONFIG,lee_method=LEE_METHOD,
        stage_folders=STAGE_FOLDERS,
        selection='Minimum original C4; F unbounded; denominator has no floor',
        truth_usage='Evaluation only, after step06 final selection'),output/'run_config.json')

    paths = {
        'initial_candidates':folders[3]/'initial_candidates.csv',
        'initial_depth_trials':folders[3]/'initial_depth_trials.csv',
        'seed_ranking':folders[4]/'seed_ranking.csv',
        'optimized_candidates':folders[5]/'optimized_candidates.csv',
        'trajectory':folders[5]/'trajectory.csv',
        'candidates':folders[6]/'candidates.csv',
        'alternatives':folders[6]/'alternatives.csv',
        'bands':folders[6]/'bands.csv',
    }
    streams = {name:CsvStream(path) for name,path in paths.items()}
    estimates, failures = [], []
    # 추정기에 전달하는 네 항목입니다. references의 정답 수심·비율·총 IOP가 없습니다.
    payloads = ((row['rrs'],contexts.get(row['water_key'],
                    f"IOP estimation failed for {row['water_key']}; see 02_hope_pgx/failures.json"),
                 dataset['bottoms'],settings)
                for row in observations)
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers>1 else None
    # 각 관측 사례에서 pipeline.solve가 3 → 4 → 5 → 6을 순서대로 수행합니다.
    solutions = pool.map(solve_with_iop_status,payloads,chunksize=4) if pool else map(solve_with_iop_status,payloads)
    selection_fields = {'rank','same_solution_duplicate','solution_group','same_solution_as',
                        'near_best','relative_c4_gap','is_representative','is_alternative'}
    try:
        for index,(obs,solution) in enumerate(zip(observations,solutions),1):
            ref = references[obs['case_id']]
            common = {key:value for key,value in ref.items() if key not in ('a_true','bb_true')}
            try:
                if isinstance(solution,str):
                    raise RuntimeError(solution)
                estimate, candidates, traces, initials = solution
                estimate.pop('settings')
                verify_case(estimate,candidates,traces,settings,initials)
                # 6번에서 대표를 선정한 뒤에만 실제 수심·혼합비를 결합합니다.
                combined = {**common,**estimate}
                # 이미 대표를 고른 뒤의 진단입니다.
                # 실제 바닥·수심을 넣었을 때 D2와 원본 Rrs가 얼마나 다른지 확인합니다.
                # 이 진단의 C4로 위에서 고른 대표를 다시 선정하지 않습니다.
                true_rho = sum(ref['true_f_'+m]*dataset['bottoms'][m] for m in MATERIALS)
                hope_baseline = forward(ref['true_z'],true_rho,contexts[obs['water_key']])
                true_context = make_context(ref['a_true'],ref['bb_true'],ref['solar_zenith'],
                                             deep_inputs[obs['water_key']]['rrs'])
                true_baseline = forward(ref['true_z'],true_rho,true_context)
                combined['c4_at_truth_using_hope'] = float(c4(obs['rrs'],hope_baseline))
                combined['c4_at_truth_using_true_iops'] = float(c4(obs['rrs'],true_baseline))
                estimates.append(combined)
                for initial in initials:
                    trials = initial.pop('initial_trials',[])
                    streams['initial_depth_trials'].write([
                        dict(case_id=obs['case_id'],seed_id=initial['seed_id'],initial_F=initial['initial_F'],**trial)
                        for trial in trials])
                    streams['initial_candidates'].write([dict(case_id=obs['case_id'],
                        **{k:v for k,v in initial.items() if k not in ('initial_rank','selected')})])
                streams['seed_ranking'].write([dict(case_id=obs['case_id'],**row) for row in initials])
                streams['optimized_candidates'].write([dict(case_id=obs['case_id'],
                    **{k:v for k,v in row.items() if k not in selection_fields}) for row in candidates])
                streams['trajectory'].write([dict(case_id=obs['case_id'],**row) for row in traces])
                streams['candidates'].write([dict(case_id=obs['case_id'],**row) for row in candidates])
                streams['alternatives'].write([dict(case_id=obs['case_id'],**row)
                                               for row in candidates if row['is_alternative']])
                streams['bands'].write([dict(case_id=obs['case_id'],wavelength=int(w),
                    Rrs_obs=float(obs['rrs'][j]), Rrs_model=estimate[f'Rrs_model_{int(w)}'],
                    Rrs_at_truth_hope=float(hope_baseline[j]),Rrs_at_truth_true_iop=float(true_baseline[j]))
                    for j,w in enumerate(WAVES)])
            except (ValueError,AssertionError,RuntimeError) as exc:
                failures.append(dict(case_id=obs['case_id'],error=repr(exc)))
                estimates.append(dict(**common,c4=math.nan,z_hat=math.nan,F=math.nan,
                    f_sand=math.nan,f_brown=math.nan,f_green=math.nan,status='failed',
                    has_alternative=False,f_outside=False,f_negative=False,any_evaluation_limit=False))
            if index==1 or index%100==0 or index==len(observations):
                print(json.dumps(dict(stages='3-6',done=index,total=len(observations),
                                      failures=len(failures)),ensure_ascii=False),flush=True)
    finally:
        for stream in streams.values():
            stream.close()
        if pool:
            pool.shutdown()

    estimates_path = folders[6]/'estimates.csv'
    save_csv(estimates,estimates_path)
    save_json(failures,folders[6]/'failures.json')
    # 정답 대조는 6단계의 결과를 저장한 다음 수행합니다.
    # 2위가 정답에 더 가까운 경우도 별도 표에 남고 대표 선정 규칙은 유지됩니다.
    evaluation_dir = folders[6]/'evaluation'
    evaluation_dir.mkdir(exist_ok=True)
    summary = evaluate_after_selection(estimates,evaluation_dir)
    if summary['finite_estimates']:
        compare_candidates_with_truth(estimates_path,paths['candidates'],evaluation_dir)

    # 7. 저장된 최종 결과만 읽어서 그림을 만듭니다. 역산을 다시 하지 않습니다.
    figure_paths = []
    if not args.no_figures:
        from step07_figures import make_figures
        from plot_hope_iops import make_iop_figures
        from compare_with_qaa import compare_results
        if summary['finite_estimates']:
            figure_paths = make_figures(estimates_path,folders[7])
        if contexts:
            figure_paths.extend(make_iop_figures(output,args.baseline))
        # 비교는 최종해 저장 이후 수행하며, 비교 기준 QAA 자료는 추정기로 전달하지 않습니다.
        figure_paths.extend(compare_results(output,args.baseline))
        print(f'7/7 Figures saved: {len(figure_paths)} files',flush=True)
    save_json(dict(completed=True,experiment=EXPERIMENT_CONFIG,mode=args.mode,seconds=time.perf_counter()-started,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        source_code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
        summary=summary,iop_method=HOPE_METHOD,iop_conditions=len(deep_inputs),
        iop_successes=len(contexts),iop_failures=len(iop_result['failures']),iop_seconds=iop_seconds,
        baseline_results=str(args.baseline),runtime_WMI_fallback=os.environ.get('HOPE_DISABLE_WMI')=='1',
        figure_paths=[str(p) for p in figure_paths],
        figures_generated=not args.no_figures),output/'run_manifest.json')
    print(json.dumps(summary,ensure_ascii=False,default=json_default),flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} cases failed; inspect 06_final_solution/failures.json')
    return summary


def main():
    """명령줄 인자를 읽고 실행합니다. 기본값은 Settings와 아래 인자에서 확인합니다.

    --mode pilot/full: 36개 검증 또는 전체 계산.
    --data, --bottom: 실제 Rrs 폴더와 순수 바닥 라이브러리 폴더.
    --output: 결과 저장 폴더. --cache: 원본 검사 후 재사용할 선택적 JSON 캐시.
    --limit: 선택된 자료 중 앞에서 지정 수만 실행하는 진단용 옵션.
    --top-k: 12차 v2와 같이 기본값 5이며 초기 9개 중 C4가 작은 후보를 고릅니다.
    --workers: 사례 병렬 작업 수. --no-figures: 추정 결과까지만 저장.
    초기 수심 정밀도나 반복 이동 스텝을 이 인자에서 자동 조절하지 않습니다.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['pilot','full'],default='pilot')
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--bottom',type=Path,default=DEFAULT_BOTTOM)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--top-k',type=int,default=5)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--no-figures',action='store_true')
    parser.add_argument('--baseline',type=Path,
                        default=Path('G:/태평양/11차/iteration1/1번/results'),
                        help='Saved QAA results used only for evaluation after estimation')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.workers < 1:
        parser.error('--workers must be positive')
    run(args)


# Windows에서 병렬 실행할 때 작업 프로세스가 main을 중복 시작하지 않게 하는 구문입니다.
if __name__=='__main__':
    main()
