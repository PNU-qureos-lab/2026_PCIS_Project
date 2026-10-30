"""사용자가 선택한 원본 픽셀 10개에서 심수역 기준 반사도를 추출한다.

입력은 수면 위 원격탐사 반사도 Rrs(sr^-1)이며, 수면 아래 심수역 원격탐사 반사도
rrs_dp로의 변환은 역산 엔진이 수행한다.
선택 좌표는 원본 영상의 0부터 시작하는 (행, 열)이다. 화면 축소본의 색상이나
대비 조정값은 계산에 사용하지 않는다. 중복 픽셀은 한 번만 반영하고, 어느
한 밴드라도 NoData·비유한 값·0 이하이면 그 픽셀 전체를 통계에서 제외한다.

수심 분석 마스크는 이 단계에 적용하지 않는다. 심수역은 얕은 수심을 계산할
분석 영역 밖에 있는 경우가 많기 때문이다. 유효값 검사만으로 구름·그림자나
심수역 여부를 판별할 수는 없으며, 해당 영역의 적합성은 사용자가 확인한다.
"""
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
import numpy as np
from rasterio.windows import Window
from rasterio.enums import Resampling
from .config import SENSOR_WAVES, file_sha256
from .raster_input import open_inputs

REQUIRED_SELECTED_PIXELS = 10
MAX_SELECTED_PIXELS = REQUIRED_SELECTED_PIXELS
STATISTIC_LABELS = {"median": "Median", "mean": "Mean"}


def normalize_pixels(pixels, width, height):
    """잘못된 좌표를 반올림하지 않고 거절하며 중복은 동일 픽셀로 취급한다."""
    if not isinstance(pixels, (list, tuple, set)):
        raise ValueError(f"심수역 픽셀은 최대 {MAX_SELECTED_PIXELS:,}개까지 선택할 수 있습니다.")
    unique = set()
    for pair in pixels:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("심수역 픽셀 좌표는 [행, 열]이어야 합니다.")
        if any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in pair
        ):
            raise ValueError("심수역 픽셀의 행·열은 정수여야 합니다.")
        row, col = map(int, pair)
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError("심수역 선택 픽셀이 영상 범위를 벗어났습니다.")
        unique.add((row, col))
        if len(unique) > MAX_SELECTED_PIXELS:
            raise ValueError(f"심수역 픽셀은 최대 {MAX_SELECTED_PIXELS}개까지 선택할 수 있습니다.")
    return sorted(unique)


def aggregate(values, statistic):
    """유효한 동일 픽셀 집합에서 모든 밴드의 평균 또는 중위값을 계산한다.

    각 픽셀에 동일한 가중치를 부여하며, 밴드마다 서로 다른 픽셀을 사용하지 않는다.
    선택 도중에도 통계를 표시할 수 있지만 적용·실행 시에는 유효한 10개를 요구한다.
    반환 필드 rrs는 저장 형식의 이름이며, 값은 수면 위 Rrs(sr^-1)이다.
    std는 유효 픽셀 집합의 모집단 표준편차(ddof=0)로, 추정값의 불확실성이 아니다.
    """
    if statistic not in STATISTIC_LABELS:
        raise ValueError("통계 방식은 mean 또는 median이어야 합니다.")
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] not in (4, 6):
        raise ValueError("심수역 픽셀의 밴드 배열이 올바르지 않습니다.")
    valid = np.all(np.isfinite(values) & (values > 0), axis=1)
    clean = values[valid]
    if not len(clean):
        raise ValueError("선택한 픽셀 중 모든 밴드가 유효한 양수인 픽셀이 없습니다.")
    central = (
        np.mean(clean, axis=0, dtype=np.float64)
        if statistic == "mean"
        else np.median(clean, axis=0)
    )
    return dict(
        statistic=statistic,
        selected_count=len(values),
        valid_count=int(valid.sum()),
        excluded_count=int((~valid).sum()),
        rrs=central.tolist(),
        minimum=clean.min(axis=0).tolist(),
        maximum=clean.max(axis=0).tolist(),
        std=clean.std(axis=0).tolist(),
        valid=valid,
    )


@lru_cache(maxsize=32)
def _cached_digest(path, size, mtime_ns, cancelled=None):
    # 같은 프로세스의 입력 확인/실행 준비에서 대형 파일을 반복 해싱하지 않는다.
    # 파일 크기나 수정 시간이 바뀌면 새 키가 되어 해시를 다시 계산한다.
    return file_sha256(path, cancelled)


def source_identity(files, cancelled=None):
    """밴드 순서대로 원본 파일의 SHA-256을 반환하고 검사 중 변경을 감지한다."""
    result = []
    for value in files:
        if cancelled and cancelled():
            raise InterruptedError("사용자가 심수역 입력 검사를 중지했습니다.")
        path = Path(value).resolve()
        before = path.stat()
        digest = _cached_digest(str(path), before.st_size, before.st_mtime_ns, cancelled)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("반사도 파일이 검사 중 변경됐습니다. 영상을 다시 열어주세요.")
        result.append(digest)
    return result


class RrsSource:
    """분석 마스크와 독립적으로 원본 Rrs 영상을 읽는 접근기.

    파일은 with 문 또는 close()로 해제한다. 통계용 sample은 원본 격자에서 읽고,
    화면용 rgb만 최근접 축소와 대비 신장을 적용한다.
    """

    def __init__(self, config):
        # 심수역 기준은 수심 분석 영역 밖에서도 선택할 수 있으므로 mask를 전달하지 않는다.
        self.config = {"sensor": config["sensor"], "rrs_files": list(config["rrs_files"])}
        self.stack = ExitStack()
        try:
            self.datasets, _ = open_inputs(self.config, self.stack)
            self.first = self.datasets[0]
            self.width = self.first.width
            self.height = self.first.height
            self.waves = list(SENSOR_WAVES[config["sensor"]])
        except BaseException:
            self.stack.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.stack.close()

    def grid(self):
        """선택 좌표의 유효성을 재검사할 영상 크기·좌표계·격자 변환값을 반환한다."""
        return dict(
            width=self.width,
            height=self.height,
            crs=str(self.first.crs),
            transform=list(self.first.transform)[:6],
        )

    def read(self, window, shape=None, indices=None):
        """지정 구간의 밴드를 읽으며 NoData를 NaN으로 바꾼다. indices는 0부터 시작한다."""
        indices = list(range(len(self.waves))) if indices is None else list(indices)
        options = dict(window=window, masked=True)
        if shape:
            options.update(out_shape=(len(indices), *shape), resampling=Resampling.nearest)
        if len(self.datasets) == 1:
            array = self.datasets[0].read([i + 1 for i in indices], **options).astype(float)
        else:
            if shape:
                options["out_shape"] = shape
            array = np.ma.stack(
                [self.datasets[i].read(1, **options).astype(float) for i in indices]
            )
        return array.filled(np.nan)

    def sample(self, pixels, cancelled=None):
        """중복을 제거한 좌표와 원본 Rrs 배열을 같은 순서로 반환한다."""
        pixels = normalize_pixels(pixels, self.width, self.height)
        values = np.empty((len(pixels), len(self.waves)), float)
        # 근처 픽셀은 함께 읽어 클릭 수만큼 디스크를 왕복하지 않는다.
        groups = {}
        for index, (row, col) in enumerate(pixels):
            groups.setdefault((row // 128, col // 128), []).append((index, row, col))
        for (tr, tc), items in groups.items():
            if cancelled and cancelled():
                raise InterruptedError("사용자가 심수역 픽셀 읽기를 중지했습니다.")
            r0, c0 = tr * 128, tc * 128
            block = self.read(Window(c0, r0, min(128, self.width - c0), min(128, self.height - r0)))
            for index, row, col in items:
                values[index] = block[:, row - r0, col - c0]
        return pixels, values

    def rgb_stretch(self):
        """적색·녹색·청색 밴드의 2~98 백분위수로 표시용 대비 범위를 정한다."""
        # 이 대비 범위는 영상 표시에만 적용하며 원본 Rrs와 추출 통계는 바꾸지 않는다.
        indices = [3, 2, 1] if len(self.waves) == 4 else [5, 3, 1]
        scale = min(1, 512 / max(self.width, self.height))
        shape = (max(1, round(self.height * scale)), max(1, round(self.width * scale)))
        rgb = self.read(Window(0, 0, self.width, self.height), shape, indices)
        ranges = []
        for band in rgb:
            valid = band[np.isfinite(band) & (band > 0)]
            lo, hi = np.percentile(valid, [2, 98]) if len(valid) else (0, 1)
            if hi <= lo:
                hi = lo + max(abs(lo) * 0.1, 1e-6)
            ranges.append((float(lo), float(hi)))
        return indices, ranges

    def rgb(self, window, shape, indices, ranges):
        """지정 구간을 화면용 8비트 RGB로 변환한다. 반사도 계산용 배열은 반환하지 않는다."""
        values = self.read(window, shape, indices)
        valid = np.all(np.isfinite(values) & (values > 0), axis=0)
        colors = np.empty((*shape, 3), dtype=np.uint8)
        for i, (lo, hi) in enumerate(ranges):
            scaled = np.nan_to_num((values[i] - lo) / (hi - lo), nan=0, posinf=1, neginf=0)
            colors[:, :, i] = (np.clip(scaled, 0, 1) * 255).astype("uint8")
        colors[~valid] = (32, 37, 43)
        return colors


def create_selection(config, pixels, statistic):
    """유효한 픽셀 10개의 좌표·통계 방식·원본 식별 정보를 저장 형식으로 만든다."""
    before = source_identity(config["rrs_files"])
    with RrsSource(config) as source:
        pixels, values = source.sample(pixels)
        summary = aggregate(values, statistic)
        _require_ten_valid_pixels(summary)
        selection = dict(
            schema_version=1,
            sensor=config["sensor"],
            waves_nm=source.waves,
            grid=source.grid(),
            source_sha256=before,
            pixels=[list(p) for p in pixels],
            statistic=statistic,
            preview={k: v for k, v in summary.items() if k != "valid"},
        )
    if before != source_identity(config["rrs_files"]):
        raise ValueError("선택 도중 영상이 변경됐습니다. 다시 선택하세요.")
    return selection


def _require_ten_valid_pixels(summary):
    """미리보기는 선택 도중에도 허용하지만 저장·실행은 유효한 10개로 제한한다.

    선택 저장과 분석 실행에서 모두 검사하여 무효 픽셀 제외 후 9개 이하로 실행되는 것을 막는다.
    """
    if summary["selected_count"] != REQUIRED_SELECTED_PIXELS:
        raise ValueError(f"심수역 픽셀을 정확히 {REQUIRED_SELECTED_PIXELS}개 선택하세요.")
    if summary["valid_count"] != REQUIRED_SELECTED_PIXELS:
        raise ValueError("사용할 수 없는 픽셀이 포함되어 있습니다. 빨간색 픽셀을 해제하고 다른 픽셀을 선택하세요. 유효한 10개가 필요합니다.")


def compute_reference(config, selection, cancelled=None):
    """파일·격자·밴드 순서를 검증하고 원본 픽셀에서 심수역 기준 Rrs를 다시 산출한다.

    저장된 preview는 화면 표시용이다. 실제 계산은 선택 좌표의 원본 값을 다시 읽어
    수행하므로 설정 파일에 기록된 미리보기 숫자가 분석 기준을 덮어쓰지 않는다.
    """
    if not isinstance(selection, dict) or selection.get("schema_version") != 1:
        raise ValueError("심수역 픽셀 선택 설정의 형식이 올바르지 않습니다.")
    if selection.get("sensor") != config["sensor"] or selection.get("waves_nm") != list(
        SENSOR_WAVES[config["sensor"]]
    ):
        raise ValueError("심수역 픽셀을 선택한 위성과 현재 위성이 다릅니다. 다시 선택하세요.")
    if selection.get("source_sha256") != source_identity(config["rrs_files"], cancelled):
        raise ValueError("심수역 픽셀을 선택한 영상과 현재 영상의 내용 또는 밴드 순서가 다릅니다. 다시 선택하세요.")
    with RrsSource(config) as source:
        if selection.get("grid") != source.grid():
            raise ValueError("심수역 선택 당시와 영상 격자가 다릅니다.")
        pixels, values = source.sample(selection.get("pixels", []), cancelled)
        result = aggregate(values, selection.get("statistic"))
        _require_ten_valid_pixels(result)
        result.update(
            waves_nm=source.waves,
            pixels=pixels,
            samples=values,
            grid=source.grid(),
            source_sha256=selection["source_sha256"],
        )
        return result
