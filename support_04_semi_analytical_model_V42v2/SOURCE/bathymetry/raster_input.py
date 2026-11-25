"""수심 분석과 심수역 픽셀 선택에 공통으로 사용하는 영상 입력 검증.

입력은 보정된 실수형 원격탐사반사도 Rrs(sr^-1) 영상이다. 밴드 수, 자료형,
좌표계와 격자 일치 여부를 확인하며, 재투영·재표본화·반사도 변환은 수행하지 않는다.
"""
from pathlib import Path
import numpy as np
import rasterio
from .config import SENSOR_WAVES


def open_inputs(config, stack):
    """Rrs 영상과 선택적 분석 마스크를 열고 데이터셋 목록·마스크를 반환한다.

    config의 rrs_files는 센서 밴드 순서의 다중밴드 영상 1개 또는 단일밴드
    영상 목록이다. 호출자가 전달한 ExitStack이 모든 열린 파일의 수명을 관리한다.
    mask를 지정한 경우 Rrs와 같은 격자인지 검사한다. 마스크 값의 적용은
    호출부에서 수행하므로, 심수역 선택에서는 mask를 제외한 설정을 전달한다.
    """
    files = config.get("rrs_files", [])
    waves = SENSOR_WAVES[config["sensor"]]
    if len(files) not in (1, len(waves)):
        raise ValueError(f"Rrs는 {len(waves)}밴드 영상 1개 또는 단일밴드 영상 {len(waves)}개가 필요합니다.")
    if len(set(str(Path(x).resolve()) for x in files)) != len(files):
        raise ValueError("같은 Rrs 파일을 중복 선택했습니다.")
    datasets = [stack.enter_context(rasterio.open(p)) for p in files]
    first = datasets[0]
    if first.crs is None:
        raise ValueError("입력 Rrs에 좌표계(CRS)가 없습니다.")
    for ds in datasets:
        if ds.count != (len(waves) if len(files) == 1 else 1):
            raise ValueError("Rrs 영상의 밴드 수가 선택한 입력 방식과 다릅니다: " + ds.name)
        if (ds.width, ds.height, ds.crs) != (
            first.width,
            first.height,
            first.crs,
        ) or not ds.transform.almost_equals(first.transform, precision=1e-9):
            raise ValueError("Rrs 영상끼리 크기·좌표계·픽셀 위치가 다릅니다. 같은 격자로 준비해주세요.")
        if any(abs(s - 1) > 1e-12 for s in ds.scales) or any(abs(o) > 1e-12 for o in ds.offsets):
            raise ValueError("스케일/오프셋이 있는 영상입니다. 실제 Rrs 값(sr^-1)으로 변환한 파일을 사용하세요.")
        if not all(np.issubdtype(np.dtype(t), np.floating) for t in ds.dtypes):
            raise ValueError("정수형 영상은 Rrs로 바로 사용할 수 없습니다. 보정된 실수형 Rrs 파일을 선택하세요.")
    mask = None
    if config.get("mask"):
        mask = stack.enter_context(rasterio.open(config["mask"]))
        if (
            mask.count != 1
            or (mask.width, mask.height, mask.crs) != (first.width, first.height, first.crs)
            or not mask.transform.almost_equals(first.transform, precision=1e-9)
        ):
            raise ValueError("마스크와 Rrs 영상의 격자가 다릅니다. 같은 크기·좌표계·픽셀 위치여야 합니다.")
    return datasets, mask
