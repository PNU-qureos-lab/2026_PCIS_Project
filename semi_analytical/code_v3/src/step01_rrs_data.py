"""1단계: 원본 자료에서 추정에 사용할 Rrs와 깊은 물 짝을 정리합니다.

읽는 순서: load_dataset → parse_hydrolight_file → prepare_inputs.
입력: HydroLight 유한수심 4,860개, 깊은 물 27개, 순수 바닥 3종 파일.
출력: observations(얕은 물), deep_inputs(깊은 물), references(평가용 정답).
모든 4밴드 배열 순서는 [443, 490, 560, 665] nm입니다.
바닥 계수 순서는 [sand=모래, brown=갈조, green=녹조]입니다.

주의: 이 파일의 사전 키 'rrs'에는 원본의 수면 위 Rrs가 들어 있습니다.
키 이름이 소문자여도 수면 아래 rrs로 이미 변환했다는 뜻은 아닙니다.
수면 경계 변환은 model.py에서 하며, Lu/Ed로 대체하지 않습니다.
실제 총 IOP와 바닥·수심 정답은 입력 파일 검토와 사후 평가에 보관합니다.
prepare_inputs가 추정기로 넘기는 자료에는 정답 수심·혼합비·총 IOP가 없습니다.
원본은 읽기만 합니다. 선택적 JSON 캐시는 원본 폴더 밖에만 저장합니다.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import time

import numpy as np


WAVELENGTHS = np.array([443.0, 490.0, 560.0, 665.0])
BOTTOM_NAMES = ("sand", "brown", "green")
CACHE_VERSION = 14
WATER_PATH = Path(__file__).resolve().parents[1] / "reference_data" / "H2OabDefaults_SEAwater.txt"
_MATERIAL = {"cs": "sand", "ba": "brown", "ga": "green"}
_FINITE_NAME = re.compile(
    r"M(?P<bottom>(?:ba|cs|ga)\d+(?:(?:ba|cs|ga)\d+)?)"
    r"_Z(?P<z>\d+)_SZ(?P<sz>\d+)_CHL(?P<chl>\d+)_TSM(?P<tsm>\d+)$"
)
_INFINITE_NAME = re.compile(
    r"Minfi_SZ(?P<sz>\d+)_CHL(?P<chl>\d+)_TSM(?P<tsm>\d+)$"
)
_BOTTOM_FILES = {
    "sand": "pure_100_coral_sand.txt",
    "brown": "pure_100_brown_algae.txt",
    "green": "pure_100_green_algae.txt",
}


def _read_text(path):
    # 숫자와 식별자는 원문 그대로 해석합니다. 디코딩 실패 문자를 대체하지 않습니다.
    """텍스트 파일을 UTF-8 또는 CP949로 읽습니다.

    숫자나 파일명이 깨진 상태로 계산에 들어가지 않도록 디코딩에 실패하면
    다음 인코딩을 시도합니다. 손상된 문자를 임의 기호로 바꾸지 않습니다.
    """
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"Unsupported text encoding: {path}")


def _four_rows(lines, start, path, section, return_full=False):
    """표의 시작 위치에서 숫자 행을 읽고 정확한 네 파장을 선택합니다.

    lines는 파일 전체 줄, start는 이 표의 숫자 시작 줄입니다.
    443·490·560·665 nm가 각각 정확히 한 행씩 있어야 합니다.
    관측 자료의 누락 밴드를 주변 파장으로 보간하지 않습니다.
    선택된 네 행과 사용한 네 파장 목록을 반환합니다.
    return_full도 이제 네 행만 돌려줍니다. 다른 관측 파장의 수치 열은 읽지 않습니다.
    """
    rows = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('"'):
            break
        try:
            tokens = stripped.split()
            # 허용 파장을 먼저 검사합니다. 다른 파장의 수치 열은 파싱/검증/저장하지
            # 않으므로 초기식 또는 평가에서 추가 관측 밴드가 들어갈 경로가 없습니다.
            if float(tokens[0]) not in WAVELENGTHS:
                continue
            row = [float(x.replace("D", "E")) for x in tokens]
        except ValueError as exc:
            raise ValueError(f"Malformed {section} row in {path}: {line}") from exc
        rows.append(row)
    if not rows:
        raise ValueError(f"Empty {section} table in {path}")
    try:
        matrix = np.asarray(rows, dtype=float)
    except ValueError as exc:
        raise ValueError(f"Inconsistent {section} columns in {path}") from exc
    if matrix.ndim != 2 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid {section} table in {path}")
    selected = []
    for wavelength in WAVELENGTHS:
        indices = np.flatnonzero(matrix[:, 0] == wavelength)
        if len(indices) != 1:
            raise ValueError(
                f"Expected exactly one {wavelength:g} nm row in {section}: {path}"
            )
        selected.append(matrix[indices[0]])
    if return_full:
        return np.asarray(selected), matrix[:, 0].tolist(), matrix
    return np.asarray(selected), matrix[:, 0].tolist()


def parse_hydrolight_file(path):
    """HydroLight 한 파일에서 네 밴드 Rrs와 평가용 총 a·bb만 읽습니다.

    Rrs라고 표시된 독립 표에서 열 이름을 먼저 확인하고 Rrs 열을 그대로 읽습니다.
    Lw/Ed는 읽은 Rrs와 대체로 같은지 확인하는 용도이며 계산 입력을 교체하지 않습니다.
    이 확인의 rtol=2e-4, atol=1e-10은 출력 반올림을 고려한 기존 파싱 허용차입니다.
    C4 분모 하한이나 역산 종료 기준이 아닙니다.

    총 a·bb는 표층 열을 읽되 다른 수심 열과 같은지 검사합니다.
    현재 모델은 수직으로 균일한 IOP를 전제로 하므로 다르면 오류로 알립니다.
    총 IOP는 a_true·bb_true에 저장합니다. component 1의 물 상수는 사용하지 않습니다.
    aw/bbw는 load_dataset에서 사용자 지정 H2OabDefaults_SEAwater 파일로 공급합니다.
    """
    path = Path(path)
    lines = _read_text(path).splitlines()
    positions = {}
    for index, line in enumerate(lines):
        for name in ("Rrs", "a", "bb"):
            if line.startswith(f'"{name}" '):
                if name in positions:
                    raise ValueError(f"Repeated standalone {name} table: {path}")
                positions[name] = index
    if set(positions) != {"Rrs", "a", "bb"}:
        raise ValueError(f"Missing Rrs/a/bb standalone table in {path}")
    rrs_index = positions["Rrs"]
    rrs_header = re.findall(r'"([^\"]*)"', lines[rrs_index + 2])
    if rrs_header != [" ", "in air", "Rrs", "Ed", "Lw", "Lu"]:
        raise ValueError(f"Unexpected standalone Rrs column labels: {path}")
    rrs_rows, all_waves, rrs_full = _four_rows(lines, rrs_index + 3, path, "Rrs", return_full=True)
    if rrs_rows.shape[1] != 5:
        raise ValueError(f"Expected wavelength/Rrs/Ed/Lw/Lu columns: {path}")
    if np.any(rrs_rows[:, 2] <= 0):
        raise ValueError(f"Nonpositive in-air Ed: {path}")
    # Rrs와 비교할 것은 Lw/Ed입니다. 마지막 Lu 열은 사용하지 않습니다.
    # 이 비교를 통과해도 입력은 원래 Rrs 열을 그대로 유지합니다.
    lw_over_ed = rrs_rows[:, 3] / rrs_rows[:, 2]
    if not np.allclose(rrs_rows[:, 1], lw_over_ed, rtol=2e-4, atol=1e-10):
        raise ValueError(f"Standalone Rrs is inconsistent with rounded Lw/Ed: {path}")
    result = {
        "rrs": rrs_rows[:, 1].copy(),
        "rrs_table_wavelengths_nm": all_waves,
        "initialization_spectrum": dict(wavelengths_nm=WAVELENGTHS.copy(), Rrs=rrs_rows[:, 1].copy()),
        # 아래 tiny는 파싱 진단값의 0 나눗셈 방지에만 사용됩니다.
        # 추정 C4 식에는 이 하한을 전달하지 않습니다.
        "rrs_lw_ed_max_relative_discrepancy": float(
            np.max(np.abs(rrs_rows[:, 1] - lw_over_ed) /
                   np.maximum(np.abs(rrs_rows[:, 1]), np.finfo(float).tiny))
        ),
    }
    for name in ("a", "bb"):
        start = positions[name]
        rows, waves = _four_rows(lines, start + 3, path, name)
        if waves != all_waves:
            raise ValueError(f"Wavelength tables do not match in {path}")
        if not np.all(rows[:, 1:] == rows[:, 1:2]):
            raise ValueError(f"Depth-varying {name} requires an explicit IOP model: {path}")
        if np.any(rows[:, 1] <= 0):
            raise ValueError(f"Nonpositive total {name}: {path}")
        # 원본의 실제 총 IOP는 정답 대조용으로만 보관합니다.
        result[f"{name}_true"] = rows[:, 1].copy()
    return result


def select_cases(dataset, mode):
    """pilot이면 기존 검증용 36개, full이면 전체 유한수심 자료를 선택합니다.

    pilot은 SZ30·CHL03·TSM03 한 조건에서 순수 바닥 3종과 50:50 혼합 3종,
    총 6개 바닥 구성 × 6개 수심입니다. 전체 자료를 대표하는 성능 확정값은 아닙니다.
    파일명 정답은 검증 묶음을 정할 때만 사용하며 seed 선택 기준은 계속 C4입니다.
    """
    cases = dataset['finite']
    if mode == 'pilot':
        selected_bottoms = {'ba100','cs100','ga100','ba50cs50','ba50ga50','cs50ga50'}
        cases = [row for row in cases if row['bottom_code'] in selected_bottoms
                 and row['water_key'] == 'SZ30_CHL03_TSM03']
        if len(cases) != 36:
            raise ValueError(f'Expected 36 pilot cases, found {len(cases)}')
    elif mode != 'full':
        raise ValueError('mode must be pilot or full')
    return sorted(cases, key=lambda row:row['case_id'])


def prepare_inputs(dataset, cases):
    """추정 입력과 정답 표를 나눠 반환합니다.

    observations: case_id, water_key, 태양 천정각, 얕은 물 Rrs.
    deep_inputs: water_key별 깊은 물 Rrs, 순수해수 a·bb, 태양 천정각.
    references: case_id별 실제 수심, 실제 세 바닥 계수, 실제 총 IOP 등 평가 정보.
    Rrs와 순수해수 배열은 복사하여 다음 단계의 작업이 원본 배열을 바꾸지 않게 합니다.
    3~6단계는 observations·HOPE a/bb로 만든 context·바닥 라이브러리만 받습니다.
    """
    # 추정에 필요한 입력만 새 사전에 담습니다. 정답 필드를 통째로 넘기지 않습니다.
    observations = [dict(case_id=row['case_id'], water_key=row['water_key'],
                         solar_zenith=row['solar_zenith'], rrs=np.array(row['rrs'],copy=True))
                    for row in cases]
    deep_inputs = {key:{name:np.array(row[name],copy=True) if name in ('rrs','a_water','bb_water')
                        else row[name] for name in ('rrs','a_water','bb_water','solar_zenith')}
                   for key,row in dataset['infinite'].items()}
    for key, row in dataset['infinite'].items():
        deep_inputs[key]['initialization_spectrum'] = {
            name: np.array(value, copy=True) for name, value in row['initialization_spectrum'].items()}
    truth_fields = ('case_id','bottom_code','true_group','true_z','solar_zenith','chl','tsm',
                    'water_key','true_f_sand','true_f_brown','true_f_green','a_true','bb_true')
    references = {row['case_id']:{key:row[key] for key in truth_fields} for row in cases}
    return observations, deep_inputs, references


def save_prepared_tables(observations, deep_inputs, library, references, audit, output_dir):
    """1단계 자료를 사람이 확인할 수 있는 CSV와 JSON으로 저장합니다.

    observations.csv: 얕은 물 Rrs, 한 사례당 네 행.
    deep_water.csv: 깊은 물 Rrs와 순수해수 상수, 한 수질·태양각 조건당 네 행.
    bottom_library.csv: 순수 모래·갈조·녹조의 네 밴드 바닥 반사도.
    reference_truth.csv: 최종해 선정 후 평가에 결합할 정답 표.
    data_audit.json은 전체 원본 검사 결과, prepared_scope.json은 이번 선택 범위입니다.
    """
    from io_utils import save_csv, save_json
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv([dict(case_id=row['case_id'], water_key=row['water_key'],
                   solar_zenith=row['solar_zenith'], wavelength=int(w), Rrs_obs=float(value))
              for row in observations for w,value in zip(WAVELENGTHS,row['rrs'])],
             output_dir/'observations.csv')
    save_csv([dict(water_key=key, solar_zenith=row['solar_zenith'], wavelength=int(w),
                   Rrs_deep=float(row['rrs'][j]), aw=float(row['a_water'][j]),
                   bbw=float(row['bb_water'][j]))
              for key,row in sorted(deep_inputs.items()) for j,w in enumerate(WAVELENGTHS)],
             output_dir/'deep_water.csv')
    save_csv([dict(water_key=key, wavelength_nm=float(w),
                   Rrs_deep=float(row['initialization_spectrum']['Rrs'][j]),
                   aw=float(row['initialization_spectrum']['aw'][j]))
              for key,row in sorted(deep_inputs.items())
              for j,w in enumerate(row['initialization_spectrum']['wavelengths_nm'])],
             output_dir/'deep_initialization_spectra.csv')
    save_csv([dict(wavelength=int(w), **{name:float(library[name][j]) for name in BOTTOM_NAMES})
              for j,w in enumerate(WAVELENGTHS)], output_dir/'bottom_library.csv')
    save_csv([{**{key:value for key,value in row.items() if key not in ('a_true','bb_true')},
                **{f'{field}_{int(w)}':float(row[field][j])
                   for field in ('a_true','bb_true') for j,w in enumerate(WAVELENGTHS)}}
              for row in references.values()], output_dir/'reference_truth.csv')
    save_json(audit,output_dir/'data_audit.json')
    save_json(dict(selected_cases=len(observations), available_cases=audit['finite_count'],
                   deep_water_conditions=len(deep_inputs),
                   truth_usage='reference_truth.csv is only used after final selection'),
              output_dir/'prepared_scope.json')


def _metadata(path, finite):
    """실험 파일명에 적힌 조건을 해석합니다.

    예: Mba10cs90_Z05_SZ20_CHL01_TSM03은 갈조 10%·모래 90%,
    수심 0.5 m, 태양 천정각 20도, CHL 0.1, TSM 0.3을 뜻합니다.
    water_key는 같은 수질·태양각의 깊은 물 파일을 찾는 식별자입니다.
    true_f_*는 정답이며 추정 F를 제한하는 값이 아닙니다.
    코드 내 비율 합 검사 역시 원본 실험 조건을 검사하는 것이지 추정 범위 제약이 아닙니다.
    """
    match = (_FINITE_NAME if finite else _INFINITE_NAME).fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unexpected HydroLight scenario filename: {path.name}")
    fields = match.groupdict()
    result = {
        "case_id": path.stem[1:],
        "source_path": str(path.resolve()),
        "solar_zenith": int(fields["sz"]),
        "chl": int(fields["chl"]) / 10.0,
        "tsm": int(fields["tsm"]) / 10.0,
        "water_key": f'SZ{fields["sz"]}_CHL{fields["chl"]}_TSM{fields["tsm"]}',
    }
    if finite:
        code = fields["bottom"]
        fractions = dict.fromkeys(BOTTOM_NAMES, 0.0)
        found = re.findall(r"(ba|cs|ga)(\d+)", code)
        material_seen = set()
        for material, percent in found:
            if material in material_seen:
                raise ValueError(f"Repeated material in bottom code: {code}")
            material_seen.add(material)
            fractions[_MATERIAL[material]] = int(percent) / 100.0
        if not np.isclose(sum(fractions.values()), 1.0, rtol=0, atol=1e-12):
            raise ValueError(f"Bottom fractions do not sum to one: {code}")
        active = [name for name in BOTTOM_NAMES if fractions[name] > 0]
        result.update({
            "bottom_code": code,
            "true_group": "_".join(active),
            "true_z": int(fields["z"]) / 10.0,
            **{f"true_f_{name}": value for name, value in fractions.items()},
        })
        if result["true_z"] <= 0 or len(active) not in (1, 2):
            raise ValueError(f"Unexpected depth or bottom fractions: {code}")
    return result


def _load_bottom(path):
    """순수 바닥 표에서 지정된 네 밴드 반사도만 읽습니다.

    여기서 rho는 바닥 반사도이고 수면 위 Rrs가 아닙니다.
    네 파장이 정확히 있어야 하며, 다른 파장을 이용해 누락값을 채우지 않습니다.
    선택된 네 값과 사용한 네 행·원본 경로를 함께 반환합니다.
    """
    lines = _read_text(path).splitlines()
    try:
        start = lines.index(r"\end_header") + 1
    except ValueError as exc:
        raise ValueError(f"Missing bottom spectrum end_header: {path}") from exc
    data_lines = []
    for line in lines[start:]:
        if line.strip() == r"\end_data":
            break
        if line.strip():
            # 바닥 라이브러리에도 네 파장만 남깁니다. 제공 표는 1 nm 간격이어서
            # 네 밴드가 모두 실제로 존재하며 다른 파장 보간이 필요하지 않습니다.
            if float(line.split()[0]) in WAVELENGTHS:
                data_lines.append(line)
    values = np.asarray([[float(x) for x in line.split()]
                         for line in data_lines], dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid bottom reflectance spectrum: {path}")
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError(f"Bottom wavelengths must be unique and increasing: {path}")
    if values[0, 0] > WAVELENGTHS.min() or values[-1, 0] < WAVELENGTHS.max():
        raise ValueError(f"Bottom spectrum does not cover all four bands: {path}")
    exact = all(np.count_nonzero(values[:, 0] == wave) == 1 for wave in WAVELENGTHS)
    if not exact or len(values) != 4:
        raise ValueError('Bottom library must contain all four allowed wavelengths exactly')
    # 정렬된 네 행이 정확히 일치하므로 보간 없이 반사도 열을 복사합니다.
    bands = values[:, 1].copy()
    return bands, {
        "source_path": str(path.resolve()),
        "sampling": "exact wavelength rows" if exact else "in-range linear interpolation",
        "header": "\n".join(lines[:start]),
        "wavelength_nm": values[:, 0],
        "reflectance": values[:, 1],
    }


def _json_default(value):
    """NumPy 배열에 표시를 붙여 JSON 캐시에 저장할 수 있는 형태로 바꿉니다.
    """
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _json_hook(value):
    """캐시를 읽을 때 __ndarray__ 표시가 있는 목록을 NumPy 배열로 복원합니다.
    """
    if set(value) == {"__ndarray__"}:
        return np.asarray(value["__ndarray__"], dtype=float)
    return value


def load_dataset(data_root, bottom_root, cache_path=None):
    """전체 실험을 읽고 유한수심 자료에 대응하는 깊은 물을 연결합니다.

    현재 제공된 실험 전용 구성: 유한/excel/M*.txt 4,860개와 무한/M*.txt 27개.
    같은 결과를 담은 XLSX·S 파일을 추가 관측값으로 중복 계산하지 않습니다.
    파일 수, 중복 ID, 수질 짝, 밴드, 원본에 기록된 IOP 일치 여부를 검사합니다.
    정답 IOP 일치 검사는 실험 파일이 잘 짝지어졌는지 보는 검사이며 IOP 추정이 아닙니다.

    cache_path를 주면 원본 경로·크기·수정 시각이 모두 같은 경우에만 캐시를 씁니다.
    원본이 바뀌면 다시 파싱합니다. 기존 캐시가 손상돼도 원본 읽기로 돌아갑니다.
    반환 사전: finite, infinite, bottoms, bottom_spectra, audit.
    실제 입력 실험의 규모가 바뀌면 파일 수 검증도 검토해야 합니다.
    """
    started = time.perf_counter()
    data_root = Path(data_root).resolve()
    bottom_root = Path(bottom_root).resolve()
    finite_files = sorted((data_root / "유한" / "excel").glob("M*.txt"))
    infinite_files = sorted((data_root / "무한").glob("M*.txt"))
    # 이번 제공 자료의 중복·누락을 잡는 실험 규모 검사입니다. 범용 파일 개수 제한이 아닙니다.
    if len(finite_files) != 4860 or len(infinite_files) != 27:
        raise ValueError(
            f"Expected 4860 finite and 27 infinite files; got "
            f"{len(finite_files)} and {len(infinite_files)}"
        )
    bottom_files = {name: bottom_root / filename for name, filename in _BOTTOM_FILES.items()}
    source_files = finite_files + infinite_files + list(bottom_files.values()) + [WATER_PATH]
    if len({str(path.resolve()).casefold() for path in source_files}) != len(source_files):
        raise ValueError("Duplicate source paths detected")
    manifest = []
    for path in source_files:
        stat = path.stat()
        manifest.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    manifest_hash = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if cache_path is not None:
        cache_path = Path(cache_path).resolve()
        for root in (data_root, bottom_root):
            if cache_path == root or root in cache_path.parents:
                raise ValueError("Cache must be outside the read-only source roots")
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"), object_hook=_json_hook)
                if (cached.get("version") == CACHE_VERSION and
                        cached.get("manifest") == manifest):
                    result = cached["dataset"]
                    result["audit"]["cache_hit"] = True
                    result["audit"]["load_seconds"] = time.perf_counter() - started
                    return result
            except (OSError, ValueError, KeyError, TypeError):
                pass
    from model import load_water
    water_waves, water_aw, water_bbw = load_water(WATER_PATH)
    finite = []
    infinite = {}
    seen_ids = set()
    numeric_groups = defaultdict(list)
    max_rrs_consistency = 0.0
    for finite_flag, paths in ((False, infinite_files), (True, finite_files)):
        for path in paths:
            case = _metadata(path, finite_flag)
            if case["case_id"] in seen_ids:
                raise ValueError(f"Duplicate scenario ID: {case['case_id']}")
            seen_ids.add(case["case_id"])
            case.update(parse_hydrolight_file(path))
            case['a_water'] = water_aw.copy()
            case['bb_water'] = water_bbw.copy()
            case['initialization_spectrum']['aw'] = water_aw.copy()
            max_rrs_consistency = max(max_rrs_consistency,
                case.pop("rrs_lw_ed_max_relative_discrepancy"))
            if finite_flag:
                if case["water_key"] not in infinite:
                    raise ValueError(f"No matching deep-water input for {case['case_id']}")
                deep = infinite[case["water_key"]]
                # 얕은 물·깊은 물 파일이 같은 수질 조건으로 생성됐는지 검사합니다.
                # 이 검사를 통과한 실제 총 IOP로 HOPE 추정 결과를 대체하지 않습니다.
                for field in ("a_true", "bb_true", "a_water", "bb_water"):
                    if not np.array_equal(case[field], deep[field]):
                        raise ValueError(f"Matched finite/infinite IOP mismatch: {case['case_id']}")
                finite.append(case)
                numeric_groups[tuple(case["rrs"])].append(case["case_id"])
            else:
                if case["water_key"] in infinite:
                    raise ValueError(f"Multiple deep-water files for {case['water_key']}")
                infinite[case["water_key"]] = case
    bottoms, bottom_spectra = {}, {}
    for name, path in bottom_files.items():
        bottoms[name], bottom_spectra[name] = _load_bottom(path)
    rrs_stack = np.asarray([case["rrs"] for case in finite])
    audit = {
        "finite_count": len(finite),
        "infinite_count": len(infinite),
        "matched_finite_count": len(finite),
        "wavelengths_nm": WAVELENGTHS.tolist(),
        "bottom_coefficient_order": list(BOTTOM_NAMES),
        "group_counts": dict(sorted(Counter(case["true_group"] for case in finite).items())),
        "bottom_code_counts": dict(sorted(Counter(case["bottom_code"] for case in finite).items())),
        "depth_counts": dict(sorted(Counter(str(case["true_z"]) for case in finite).items())),
        "water_case_counts": dict(sorted(Counter(case["water_key"] for case in finite).items())),
        "rrs_min_by_band": np.min(rrs_stack, axis=0).tolist(),
        "rrs_max_by_band": np.max(rrs_stack, axis=0).tolist(),
        "rrs_zero_count": int(np.count_nonzero(rrs_stack == 0)),
        "rrs_negative_count": int(np.count_nonzero(rrs_stack < 0)),
        "rrs_nonfinite_count": int(np.count_nonzero(~np.isfinite(rrs_stack))),
        "duplicate_source_paths": [],
        "duplicate_case_ids": [],
        # 여기는 입력 스펙트럼이 완전히 같은 사례들의 목록입니다.
        # 6단계의 추정 결과 중복해 검사와는 다른 종류의 중복입니다.
        "exact_four_band_rrs_duplicate_groups": [ids for ids in numeric_groups.values() if len(ids) > 1],
        "rrs_lw_ed_max_relative_discrepancy": max_rrs_consistency,
        "source_manifest_sha256": manifest_hash,
        "source_manifest": manifest,
        "cache_hit": False,
        "load_seconds": time.perf_counter() - started,
        "provenance": {
            "data_root": str(data_root), "bottom_root": str(bottom_root),
            "selected_tables": "four bands ONLY of standalone Rrs and evaluation-only total a/bb; no component-1 water inputs",
            "water_table": str(WATER_PATH),
            "water_table_sha256": hashlib.sha256(WATER_PATH.read_bytes()).hexdigest(),
            "water_aw": water_aw.tolist(), "water_bbw": water_bbw.tolist(),
            "water_backscatter_relation": "bbw=0.49997*bw",
            "water_sampling": "443: linear 440/445 table rows; 490,560,665: exact rows; then four-band arrays only",
            "rrs_column": "standalone Rrs table column 2 (Lw/Ed)",
            "iop_depth": "surface, verified constant across reported depth columns",
            "concentration_units": {"chl": "mg/m^3", "tsm": "g/m^3"},
            "filename_scale": "Z, CHL, TSM integer codes divided by 10; SZ in degrees",
            "source_writes": "none",
        },
    }
    result = {"finite": finite, "infinite": infinite, "bottoms": bottoms,
              "bottom_spectra": bottom_spectra, "audit": audit}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(cache_path.name + ".tmp")
        temporary.write_text(json.dumps(
            {"version": CACHE_VERSION, "manifest": manifest, "dataset": result},
            ensure_ascii=False, default=_json_default, separators=(",", ":")
        ), encoding="utf-8")
        temporary.replace(cache_path)
    return result
