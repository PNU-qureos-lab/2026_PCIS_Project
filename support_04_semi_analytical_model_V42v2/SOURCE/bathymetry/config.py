"""위성·수역 선택, 계수, 해저면 스펙트럼 및 심수역 입력을 검증한다.

화면과 명령줄은 같은 검증 함수를 사용한다. g0·g1은 사전 산정 자료 또는
수역별 공통값을 읽으며, 이 프로그램에서 HydroLight 실행이나 계수 회귀는
수행하지 않는다. 스펙트럼의 파장 단위는 nm, 입력 Rrs의 단위는 sr^-1이다.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import numpy as np

SENSOR_WAVES = {"Sentinel2": (443, 490, 560, 665), "SuperDove": (443, 490, 531, 565, 610, 665)}
WATER_PRESETS = {
    "case1": (0.0949, 0.0794, "Gordon et al. (1988) / Clear"),
    "case2": (0.0840, 0.1700, "Lee et al. (1999) / Turbid"),
    "mixed": (0.0895, 0.1247, "Lee et al. (2002) / Intermediate"),
}
# SuperDove CSV에 허용하는 밴드 파장 표기와 계산용 대표파장의 대응이다.
# 이 대응은 열 이름을 찾기 위한 것이며, 스펙트럼 재합성이나 보간을 수행하지 않는다.
SUPERDOVE_ACTUAL_WAVES = {443: 444, 490: 492, 531: 533, 565: 566, 610: 612, 665: 666}
STANDARD_IDS = {"coral_sand": "sand", "brown_algae": "brown", "green_algae": "green"}


def file_sha256(path, cancelled=None):
    """1 MiB씩 읽어 해시를 계산하며, 각 블록을 읽기 전에 중지 요청을 확인한다.

    cancelled()가 True이면 부분 해시를 반환하지 않고 InterruptedError를 올린다.
    반환값은 파일 전체의 SHA-256을 나타내는 16진수 문자열이다.
    """
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError("사용자가 입력 파일 해시 계산을 중지했습니다.")
            block = stream.read(1024 * 1024)
            if not block:
                break
            result.update(block)
    return result.hexdigest()


def resolve_coefficients(sensor, water_type, mode, app_root):
    """위성·수역·계수 방식에 맞는 g0·g1 배열을 저장된 정밀도로 반환한다.

    수역과 공통계수의 대응은 프로그램의 적용 설정이며, Case 1·2를 탁도 등급과
    동일한 개념으로 정의하지 않는다. 영상의 수역 유형도 자동 판별하지 않는다.
    사전 산정 파장별 계수는 Case 1에만 제공하며, 다른 수역에서는 선택할 수 없다.
    """
    if sensor not in SENSOR_WAVES:
        raise ValueError("지원하지 않는 위성입니다: " + str(sensor))
    if water_type not in WATER_PRESETS:
        raise ValueError("해역 유형은 case1/case2/mixed 중 선택하세요.")
    waves = list(SENSOR_WAVES[sensor])
    if mode == "fitted":
        if water_type != "case1":
            raise ValueError("사전 산정 파장별 계수는 현재 Case 1에서만 사용할 수 있습니다.")
        path = Path(app_root) / "DATA/coefficients" / (sensor + ".json")
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data["knots_nm"] != waves:
            raise ValueError("내장 계수의 파장 순서가 센서와 다릅니다.")
        g0, g1 = data["knot_g0"], data["knot_g1"]
        source = "Case 1 water · precomputed wavelength-specific coefficients (303 conditions)"
        version = "spectral-" + sensor + "-case1-fit303-v1"
    elif mode == "fixed":
        a, b, source = WATER_PRESETS[water_type]
        g0, g1 = [a] * len(waves), [b] * len(waves)
        version = "fixed-" + water_type
    else:
        raise ValueError("계수 방식은 fitted 또는 fixed여야 합니다.")
    if len(g0) != len(waves) or len(g1) != len(waves) or not np.isfinite([g0, g1]).all():
        raise ValueError("계수 수 또는 값이 올바르지 않습니다.")
    if any(x <= 0 for x in g0) or any(x < 0 for x in g1):
        raise ValueError("g0>0, g1>=0인 계수가 필요합니다.")
    return dict(waves=waves, g0=list(g0), g1=list(g1), source=source, id=version)


def read_bottom(path, waves):
    """HydroLight 형식 텍스트 또는 2열 CSV를 읽고 센서 파장에 선형 보간한다.

    제목을 추출하고, end_header 이후의 (파장 nm, 무차원 반사도)를 수치로 읽는다.
    원본 파일의 해시와 전체 수치 스펙트럼도 반환한다. 순서 뒤바뀜·중복 파장·NaN·
    범위 밖 외삽은 오류로 처리한다. 산호사·갈조류·녹조류의 내부 ID는 동점 후보의
    정렬 순서를 일정하게 유지하도록 각각 sand·brown·green으로 고정한다.
    나머지 ID는 파일명이며 밑줄이 있어도 해저면 유형쌍은 별도 배열로 전달한다.
    """
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    header_end = next(
        (i for i, x in enumerate(lines) if x.strip().startswith("\\end_header")), None
    )
    title = next(
        (
            x.strip().removeprefix("Bottom reflectance spectrum for ").strip('"')
            for x in lines
            if "Bottom reflectance spectrum for " in x
        ),
        path.stem,
    )
    values = []
    for line in lines[(header_end + 1 if header_end is not None else 0) :]:
        s = line.strip()
        if s.startswith("\\end_data"):
            break
        if not s or s.startswith(("#", "!", "\\")):
            continue
        parts = re.split(r"[,;\s]+", s)
        if len(parts) != 2:
            raise ValueError("파장·반사도 두 열이 필요합니다: " + path.name)
        try:
            row = [float(x.replace("D", "E").replace("d", "e")) for x in parts]
        except ValueError:
            if (
                header_end is None
                and not values
                and any(x.lower() in s.lower() for x in ("wave", "reflect", "파장", "반사"))
            ):
                continue
            raise ValueError("반사도 파일에 숫자가 아닌 데이터가 있습니다: " + path.name)
        values.append(row)
    data = np.asarray(values, float)
    if len(values) < 2 or not np.isfinite(data).all():
        raise ValueError("유한한 반사도 데이터가 부족합니다: " + path.name)
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("파장은 중복 없이 오름차순이어야 합니다: " + path.name)
    if np.any(data[:, 1] < 0) or np.any(data[:, 1] > 1):
        raise ValueError("해저면 반사도는 0~1의 무차원 값이어야 합니다: " + path.name)
    if min(waves) < data[0, 0] or max(waves) > data[-1, 0]:
        raise ValueError("선택한 위성의 모든 파장을 포함하지 않습니다: " + path.name)
    return dict(
        id=STANDARD_IDS.get(path.stem, path.stem),
        title=title,
        path=str(path),
        waves=list(waves),
        reflectance=np.interp(waves, data[:, 0], data[:, 1]).tolist(),
        spectrum=data.tolist(),
        sha256=file_sha256(path),
        units="dimensionless",
        error=None,
    )


def scan_bottoms(folder, waves):
    """파일 목록이 아닌 스펙트럼만 제시하고, 읽기 오류는 각 항목에 표시한다."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError("해저면 자료 폴더를 찾을 수 없습니다: " + str(folder))
    rows = []
    for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if path.suffix.lower() not in (".txt", ".csv") or path.name.lower() == "filelist.txt":
            continue
        try:
            rows.append(read_bottom(path, waves))
        except (ValueError, OSError, UnicodeError) as exc:
            rows.append(
                dict(
                    id=path.stem,
                    title=path.stem,
                    path=str(path.resolve()),
                    waves=list(waves),
                    reflectance=[],
                    error=str(exc),
                )
            )
    # 산호사·갈조류·녹조류를 먼저 표시하고, 나머지는 제목 순으로 표시한다.
    order = {"sand": 0, "brown": 1, "green": 2}
    return sorted(rows, key=lambda row: (order.get(row["id"], 3), row["title"]))


def load_config(path):
    """저장 설정의 상대경로는 설정 파일 위치 기준으로 해석한다."""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("rrs_files", "bottom_files"):
        if key in data:
            data[key] = [str((path.parent / str(v)).resolve()) for v in data[key]]
    for key in ("deep_csv", "mask", "output_root"):
        if data.get(key):
            data[key] = str((path.parent / str(data[key])).resolve())
    return data


def deep_water_values(config, waves, cancelled=None):
    """심수역의 수면 위 Rrs(sr^-1)와 공기 중 태양 천정각(도)을 읽는다.

    반사도는 영상 픽셀 선택, CSV 또는 직접 입력으로 받는다. CSV는 밴드별 열을
    갖는 1행 형식 또는 wavelength_nm,Rrs 열을 갖는 파장별 행 형식을 지원한다.
    태양 천정각을 설정에 지정하면 CSV의 각도보다 우선한다. 수면 아래 rrs로의
    변환은 IOP 계산 단계에서 수행하므로, 여기서는 수면 위 값을 그대로 반환한다.
    """
    solar = config.get("solar_zenith")
    # 444/492/...는 SuperDove의 CSV 밴드 표기와 계산용 대표파장을 잇는 별칭이다.
    # Sentinel-2 또는 센서를 생략한 입력에 이 대응을 무조건 적용하지 않는다.
    aliases = SUPERDOVE_ACTUAL_WAVES if config.get("sensor") == "SuperDove" else {}
    if config.get("deep_selection") is not None:
        if config.get("deep_csv") or config.get("deep_rrs") is not None:
            raise ValueError("심수역 픽셀 선택, CSV, 직접 입력 중 하나만 지정하세요.")
        from .deepwater import compute_reference

        values = compute_reference(config, config["deep_selection"], cancelled)["rrs"]
    elif config.get("deep_csv"):
        with Path(config["deep_csv"]).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError("깊은 물 CSV에 데이터가 없습니다.")
        if "wavelength_nm" in rows[0] and "Rrs" in rows[0]:
            # 443.9를 int로 잘라 443 nm 관측으로 받아들이면 다른 파장이 섞인다.
            # float의 정확한 값으로 찾으므로 443.0은 443과 같지만 443.9는 다르다.
            mapped = {float(r["wavelength_nm"]): float(r["Rrs"]) for r in rows}
            if not all(math.isfinite(w) and w > 0 for w in mapped):
                raise ValueError("깊은 물 CSV의 파장은 유한한 양수여야 합니다.")
            if len(mapped) != len(rows):
                raise ValueError("깊은 물 CSV의 파장이 중복됩니다.")
            values = []
            for wave in waves:
                key = wave if wave in mapped else aliases.get(wave)
                if key not in mapped:
                    raise ValueError(f"깊은 물 CSV에 {wave} nm Rrs 값이 없습니다.")
                values.append(mapped[key])
            angles = {float(r["solar_zenith_deg"]) for r in rows if r.get("solar_zenith_deg")}
            if len(angles) > 1:
                raise ValueError("깊은 물 CSV의 태양천정각은 한 값이어야 합니다.")
            if solar is None and angles:
                solar = angles.pop()
        else:
            if len(rows) != 1:
                raise ValueError("깊은 물 CSV에는 분석할 장면 1행만 넣어주세요.")
            row = rows[0]
            values = []
            for wave in waves:
                key = f"deep_Rrs_{wave}_sr-1"
                if key not in row and wave in aliases:
                    key = f"deep_Rrs_{aliases[wave]}_sr-1"
                if key not in row:
                    raise ValueError(f"깊은 물 CSV에 {wave} nm Rrs 열이 없습니다.")
                values.append(float(row[key]))
            if solar is None:
                solar = row.get("solar_zenith_deg")
    else:
        values = config.get("deep_rrs", [])
    try:
        values = [float(v) for v in values]
        solar = float(solar)
    except (TypeError, ValueError):
        raise ValueError("깊은 물 Rrs와 태양천정각을 숫자로 입력하세요.")
    if len(values) != len(waves) or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("깊은 물 Rrs는 모든 모델 밴드에 대해 유한한 양수여야 합니다.")
    if not math.isfinite(solar) or not 0 <= solar < 90:
        raise ValueError("공기 중 태양천정각은 0 이상 90도 미만이어야 합니다.")
    return values, solar
