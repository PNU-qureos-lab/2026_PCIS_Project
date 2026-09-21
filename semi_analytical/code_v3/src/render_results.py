"""이미 저장된 결과에서 그래프만 재생성합니다. 추정은 다시 실행하지 않습니다.

기존 Figure 1/2/3/5/6/7과 HOPE a/bb 산점도, QAA 비교를 같은 출력 위치에 씁니다.
입력 파일은 run_all.py가 만든 결과 폴더입니다. UTF-8의 CSV를 그대로 읽으며
F 범위 밖 사례를 임의로 삭제하거나 축 밖으로 잘라 성능을 좋게 보이게 하지 않습니다.
"""
from pathlib import Path
import argparse
from step07_figures import make_figures
from plot_hope_iops import make_iop_figures
from compare_with_qaa import compare_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('--baseline',type=Path,default=Path('G:/태평양/11차/iteration1/1번/results'))
    args = parser.parse_args()
    paths = make_figures(args.results/'06_final_solution'/'estimates.csv',args.results/'07_figures')
    paths.extend(make_iop_figures(args.results,args.baseline))
    paths.extend(compare_results(args.results,args.baseline))
    print(f'그림 및 확인 표 {len(paths)}개 생성 완료')


if __name__ == '__main__':
    main()
