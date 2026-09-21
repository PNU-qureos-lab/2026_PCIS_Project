"""CSV·JSON 저장용 보조 함수입니다. 추정식이나 결과 선택 규칙은 없습니다.

UTF-8로 저장하며 CSV는 한글 Excel에서 읽기 쉬운 BOM을 포함합니다.
데이터를 별도 정규화·반올림·클리핑하는 목적의 코드가 아닙니다.
"""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd

def json_default(value):
    """NumPy 배열·스칼라와 경로를 JSON에 쓸 수 있는 기본 자료형으로 바꿉니다.
    """
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def save_json(value, path):
    """한글이 유니코드 이스케이프로 바뀌지 않도록 들여쓰기한 JSON으로 저장합니다.
    """
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')


def save_csv(rows, path):
    """행 목록을 CSV로 저장합니다. 소수 저장은 %.17g로 계산 정밀도를 보존합니다.

    엑셀식 행 인덱스 열은 붙이지 않습니다. 빈 행 목록은 빈 표로 저장될 수 있습니다.
    """
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig', float_format='%.17g')


class CsvStream:
    """많은 seed 이동 기록을 한꺼번에 메모리에 모으지 않고 순차 저장합니다.
    """
    def __init__(self, path):
        """출력 파일을 열고 첫 행의 열 이름이 들어올 때까지 기다립니다.

        'w' 모드이므로 같은 경로로 재실행하면 해당 CSV를 덮어씁니다.
        """
        self.file = Path(path).open('w', newline='', encoding='utf-8-sig')
        self.writer = None
    def write(self, rows):
        """처음 받은 행에서 열 순서를 정한 뒤 행들을 이어 쓰고 디스크로 내보냅니다.
        """
        for row in rows:
            if self.writer is None:
                self.writer = csv.DictWriter(self.file, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
        self.file.flush()
    def close(self):
        """출력 파일을 닫습니다. run_all.py의 finally에서 실패 여부와 관계없이 호출합니다.
        """
        self.file.close()
