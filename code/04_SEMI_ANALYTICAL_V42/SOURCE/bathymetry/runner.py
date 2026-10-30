"""반사도 영상의 검증·분할 처리·결과 기록을 관리한다.

입력은 전처리를 완료한 수면 위 원격탐사 반사도 Rrs(sr^-1)이다. DN이나
지표면 반사도(BOA)를 Rrs로 변환하는 전처리는 포함하지 않는다. 입력 영상의
좌표계·격자를 유지하며, 수치 역산은 engine/iop 모듈에서 수행한다.

영상은 행 단위 블록으로 읽고 완료 결과를 순차 기록한다. 취소·실패 시에도
기록된 픽셀 결과는 보존하며, 미처리 영역과 계산 실패를 별도 상태로 구분한다.
"""
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
import csv
import json
import math
import os
import re
import traceback
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy
from .config import resolve_coefficients, deep_water_values, read_bottom, file_sha256
from .engine import SolverSettings, solve_pixel
from .iop import fit_iop
from .raster_input import open_inputs as _open_inputs
from . import __version__

_WORKER = None


def plain(value):
    """JSON 기록에서 numpy 타입과 비유한 수치를 안전한 표준 값으로 바꾼다."""
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, data):
    """불완전한 JSON을 남기지 않도록 같은 폴더의 임시파일에서 교체한다."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(plain(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


def _windows(first):
    # 너비는 유지하고 최대 128행씩 읽는다. TIFF 내부 타일 크기에 종속되지 않는다.
    for row in range(0, first.height, 128):
        yield Window(0, row, first.width, min(128, first.height - row))


def _read_block(datasets, mask, window):
    """전 밴드가 유효한 양수이고 선택적 분석 마스크가 1인 픽셀만 허용한다.

    래스터의 NoData·내부 유효성 마스크를 먼저 적용하므로 양수로 지정된 NoData도
    제외한다. 분석 마스크가 없으면 영상 자체의 유효성만 검사한다.
    """
    if len(datasets) == 1:
        array = datasets[0].read(window=window, masked=True).astype(float)
    else:
        array = np.ma.stack(
            [ds.read(1, window=window, masked=True).astype(float) for ds in datasets]
        )
    values = array.filled(np.nan)
    valid = np.all(np.isfinite(values) & (values > 0), axis=0)
    if mask is not None:
        m = mask.read(1, window=window, masked=True)
        valid &= (~np.ma.getmaskarray(m)) & (m.filled(0) == 1)
    return values, valid


def _prepare(config, app_root, cancelled=None):
    """계수·심수역 기준·해저면 자료·저장 위치를 검증하고 실행 입력을 구성한다."""
    coeff = resolve_coefficients(
        config.get("sensor"), config.get("water_type"), config.get("coefficient_mode"), app_root
    )
    deep, solar = deep_water_values(config, coeff["waves"], cancelled)
    paths = config.get("bottom_files", [])
    if len(paths) not in (2, 3) or len(set(str(Path(p).resolve()) for p in paths)) != len(paths):
        raise ValueError("Select exactly two or three different bottom-reflectance files.")
    bottoms = [read_bottom(path, coeff["waves"]) for path in paths]
    if len({b["id"] for b in bottoms}) != len(bottoms):
        raise ValueError("Bottom-reflectance file IDs are duplicated; use distinct file names.")
    try:
        workers = int(config.get("workers", 1))
    except (TypeError, ValueError):
        raise ValueError("The worker count must be an integer.")
    if workers < 1 or workers > min(32, os.cpu_count() or 1):
        raise ValueError("Set the worker count from 1 to the available CPU count, with a maximum of 32.")
    if config.get("limit_pixels") is not None and (
        isinstance(config["limit_pixels"], bool)
        or not isinstance(config["limit_pixels"], int)
        or config["limit_pixels"] < 1
    ):
        raise ValueError("The optional limit_pixels test setting must be a positive integer.")
    output = Path(config.get("output_root") or Path(app_root) / "RESULTS").resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("The output location must be a folder.")
    # 실행 결과는 코드·내장 자료와 분리하고 지정 저장 위치 아래에 실행별 폴더를 만든다.
    for forbidden in (Path(app_root) / "DATA", Path(app_root) / "SOURCE"):
        if output == forbidden.resolve() or forbidden.resolve() in output.parents:
            raise ValueError("Select a result folder outside the application DATA and SOURCE folders.")
    return coeff, deep, solar, bottoms, workers, output


def validate_config(config, app_root, cancelled=None):
    """출력 생성이나 역산 없이 실행 가능한 입력인지 검사하고 한글 요약을 반환한다."""
    coeff, deep, solar, bottoms, workers, output = _prepare(config, app_root, cancelled)
    valid_count = 0
    high_rrs = 0
    with ExitStack() as stack:
        datasets, mask = _open_inputs(config, stack)
        first = datasets[0]
        for window in _windows(first):
            if cancelled and cancelled():
                raise InterruptedError
            arr, valid = _read_block(datasets, mask, window)
            valid_count += int(valid.sum())
            high_rrs += int(np.any(arr > 0.3, axis=0)[valid].sum())
        grid = f"{first.width} × {first.height}, {first.crs}, pixel {abs(first.transform.a):g} × {abs(first.transform.e):g}"
    if valid_count == 0:
        raise ValueError("No pixel has positive values in every band after applying the mask and NoData values.")
    notes = [
        "Inherent optical properties estimated from the deep-water reference Rrs are applied across the analysis area.",
        "Bottom fractions are not constrained to 0–1 during optimization; out-of-range values are recorded in quality flags.",
    ]
    if high_rrs:
        notes.append(f"{high_rrs:,} pixels exceed 0.3 sr^-1. Confirm that the input is Rrs rather than BOA reflectance or DN.")
    if config["water_type"] != "case1":
        notes.append("The water-area setting changes the fixed coefficient preset; it is not a separate Case 2 component model or accuracy validation.")
    if config.get("limit_pixels"):
        notes.append(f"Test run: only the first {config['limit_pixels']} valid pixels are processed.")
    if config.get("deep_selection"):
        from .deepwater import compute_reference, STATISTIC_LABELS

        reference = compute_reference(config, config["deep_selection"], cancelled)
        notes.append(
            f"Deep-water reference: {STATISTIC_LABELS[reference['statistic']]} · selected {reference['selected_count']:,} / used {reference['valid_count']:,} / excluded {reference['excluded_count']:,}. The analysis mask is not applied to this selection."
        )
    seeds = 3 if len(bottoms) == 2 else 9
    return {
        "위성": config["sensor"],
        "해역": config["water_type"],
        "계수": coeff["source"],
        "파장_nm": coeff["waves"],
        "g0": coeff["g0"],
        "g1": coeff["g1"],
        "유효_픽셀": valid_count,
        "격자": grid,
        "깊은물_Rrs": deep,
        "태양천정각_도": solar,
        "해저면": [b["title"] for b in bottoms],
        "초기_후보": seeds,
        "탐색_후보": min(5, seeds),
        "병렬_작업": workers,
        "저장_폴더": str(output),
        "주의": notes,
    }


def _init_worker(context, library, settings, cancel_path):
    """작업 프로세스마다 공통 광학 조건과 해저면 자료를 한 번 전달한다."""
    global _WORKER
    _WORKER = (context, library, settings, cancel_path)


def _solve_item(item):
    """프로세스 경계를 넘는 결과는 필요한 최종값만 포함한다."""
    index, obs = item
    context, library, settings, cancel_path = _WORKER
    try:
        result = solve_pixel(
            obs,
            context,
            library,
            settings,
            cancelled=lambda: bool(cancel_path and Path(cancel_path).exists()),
        )
        # 픽셀당 모든 최적화 이동을 전송/보존하면 대형 영상에서 메모리가 급증한다.
        result.pop("candidates", None)
        result.pop("initial_candidates", None)
        return index, result, None
    except InterruptedError:
        return index, None, "CANCELLED"
    except Exception as exc:
        return index, None, type(exc).__name__ + ": " + str(exc)


def _write_csv(path, rows):
    """행 사전의 필드 순서를 유지하고 비유한 수치는 빈 셀로 기록한다."""
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(plain(rows))


def _slug(name):
    # 분석 기록·CSV의 원본 ID는 보존하고 파일명에만 Windows 금지문자를 치환한다.
    result = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name)).strip(" .")
    return result[:100] or "analysis"


def run_analysis(config, app_root, event=None, cancel_path=None):
    """한 실행을 독립 폴더에 저장하고 진행 상태를 event 콜백으로 전달한다.

    심수역 기준에서 CHL·TSM과 IOP를 한 번 추정한 후 모든 대상 픽셀에 공유한다.
    g0·g1은 선택한 내장 계수를 유지한다. cancel_path 파일이 생성되면 작업을
    중지하며, 정상 완료·일부 픽셀 실패·사용자 취소·실행 실패를 구분해 기록한다.
    """
    event = event or (lambda payload: None)
    cancelled = lambda: bool(cancel_path and Path(cancel_path).exists())
    event({"type": "status", "message": "Validating input data."})
    try:
        check = validate_config(config, app_root, cancelled)
    except InterruptedError:
        event({"type": "cancelled", "message": "Processing was stopped during input validation."})
        return {"status": "cancelled", "processed_pixels": 0}
    try:
        coeff, deep, solar, bottoms, workers, parent = _prepare(config, app_root, cancelled)
    except InterruptedError:
        event({"type": "cancelled", "message": "Processing was stopped during setup."})
        return {"status": "cancelled", "processed_pixels": 0}
    parent.mkdir(parents=True, exist_ok=True)
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_"
        + _slug(config.get("run_name") or config["sensor"])
    )
    out = parent / run_id
    out.mkdir(exist_ok=False)
    (out / "MAPS").mkdir()
    (out / "PREVIEWS").mkdir()
    summary = {
        "version": __version__,
        "status": "running",
        "output_dir": str(out),
        "validation": check,
        "processed_pixels": 0,
        "successful_pixels": 0,
        "failed_pixels": 0,
    }
    save_json(out / "configuration.json", config)
    save_json(out / "analysis_record.json", summary)
    counts = dict(processed=0, success=0, failed=0, outside=0, limits=0, alternatives=0)
    depth_min = math.inf
    depth_max = -math.inf
    target = min(check["유효_픽셀"], config.get("limit_pixels") or check["유효_픽셀"])

    def sync_counts():
        # 정상 종료뿐 아니라 예외·취소에도 디스크에 기록한 처리 건수를 보존한다.
        summary.update(
            processed_pixels=counts["processed"],
            successful_pixels=counts["success"],
            failed_pixels=counts["failed"],
            fraction_outside_pixels=counts["outside"],
            evaluation_limit_pixels=counts["limits"],
            alternative_pixels=counts["alternatives"],
            depth_min_m=depth_min,
            depth_max_m=depth_max,
            target_pixels=target,
            scope="sample" if config.get("limit_pixels") else "all_valid_pixels",
        )

    try:
        if cancelled():
            raise InterruptedError
        if config.get("deep_selection"):
            # 화면의 미리보기 값이 아닌 원본 픽셀에서 재산출한 기준과 선택 좌표를 기록한다.
            from .deepwater import compute_reference

            reference = compute_reference(config, config["deep_selection"], cancelled)
            summary["deep_reference"] = {
                k: v for k, v in reference.items() if k not in ("pixels", "samples", "valid")
            }
            reference_row = {
                "solar_zenith_deg": solar,
                "aggregation_method": reference["statistic"],
                "selected_pixel_count": reference["selected_count"],
                "valid_pixel_count": reference["valid_count"],
            }
            reference_row.update({f"deep_Rrs_{w:g}_sr-1": v for w, v in zip(coeff["waves"], deep)})
            _write_csv(out / "deep_water_reference.csv", [reference_row])
            selected_rows = []
            # 저장 좌표는 원본 영상의 0부터 시작하는 행·열이다. 화면의 1 기반 표시와 구분한다.
            for (r, c), values, valid in zip(
                reference["pixels"], reference["samples"], reference["valid"]
            ):
                selected_rows.append(
                    dict(
                        row=r,
                        column=c,
                        used=bool(valid),
                        **{f"Rrs_{w:g}_sr-1": v for w, v in zip(coeff["waves"], values)},
                    )
                )
            _write_csv(out / "deep_water_selected_pixels.csv", selected_rows)
        event(
            {
                "type": "status",
                "message": "Estimating CHL, TSM, and inherent optical properties from the deep-water reference using the packaged g0 and g1 values.",
            }
        )
        source = json.loads((Path(app_root) / "DATA/iop_spectra.json").read_text(encoding="utf-8"))

        def iop_progress(row):
            event(
                dict(
                    type="progress",
                    stage="iop",
                    done=row.get("completed", row.get("done", 0)),
                    total=row.get("total", 98),
                    message=row.get("message", "IOP estimation"),
                )
            )

        fit = fit_iop(
            deep,
            solar,
            coeff["waves"],
            coeff["g0"],
            coeff["g1"],
            source,
            progress=iop_progress,
            cancelled=cancelled,
        )
        _write_csv(out / "IOP_by_wavelength.csv", fit["band_rows"])
        _write_csv(out / "IOP_by_starting_point.csv", fit["multistart"])
        _write_csv(out / "IOP_neighborhood_validation.csv", fit["neighbors"])
        library = {row["id"]: np.asarray(row["reflectance"]) for row in bottoms}
        pairs = list(combinations(library, 2))
        pair_codes = {pair: i + 1 for i, pair in enumerate(pairs)}
        settings = SolverSettings()
        input_paths = list(config["rrs_files"]) + [r["path"] for r in bottoms]
        if config.get("deep_csv"):
            input_paths.append(config["deep_csv"])
        if config.get("mask"):
            input_paths.append(config["mask"])
        summary.update(
            coefficients=coeff,
            bottoms=bottoms,
            solver_settings=asdict(settings),
            iop_parameters=fit["parameters"],
            input_sha256={str(p): file_sha256(p, cancelled) for p in input_paths},
            pairs={str(code): list(pair) for pair, code in pair_codes.items()},
            quality_flags={
                "1": "fraction outside 0 to 1",
                "2": "candidate-evaluation limit reached",
                "4": "bottom reflectance below 0 or above 1",
                "8": "alternative nearby solution exists",
                "16": "calculation failed",
            },
            status_codes={"0": "excluded or not processed", "1": "finite result", "2": "calculation failed"},
            units={
                "Rrs": "sr^-1",
                "depth": "m",
                "fractions": "dimensionless, unbounded",
                "C4": "mean relative squared error",
            },
            geometry="x/y units follow the input CRS; coordinates refer to pixel centers.",
            case_policy="The water-area setting selects a fixed coefficient preset; it is not an automatic water classifier or a Case 2 accuracy validation.",
        )
        float_names = ["depth_m", "C4", "relative_RMS_percent"] + [
            "fraction_" + _slug(k) for k in library
        ]
        if len(set(float_names)) != len(float_names):
            raise ValueError("해저면 이름의 출력 파일명이 겹칩니다.")
        pool = None
        try:
            with ExitStack() as stack:
                datasets, mask = _open_inputs(config, stack)
                first = datasets[0]
                summary["crs"] = str(first.crs)
                profile = dict(
                    driver="GTiff",
                    width=first.width,
                    height=first.height,
                    count=1,
                    crs=first.crs,
                    transform=first.transform,
                    dtype="float32",
                    nodata=np.nan,
                    compress="deflate",
                    BIGTIFF="IF_SAFER",
                )
                outputs = {
                    name: stack.enter_context(
                        rasterio.open(out / "MAPS" / (name + ".tif"), "w", **profile)
                    )
                    for name in float_names
                }
                for name in ("pair_code", "status", "quality_flags"):
                    # 품질값 0은 '추가 품질 표시 없음'을 뜻하므로 NoData는 별도 65535로 둔다.
                    p = dict(
                        profile, dtype="uint16", nodata=65535 if name == "quality_flags" else 0
                    )
                    outputs[name] = stack.enter_context(
                        rasterio.open(out / "MAPS" / (name + ".tif"), "w", **p)
                    )
                # 미처리·제외 영역을 수심 0 m와 구분하도록 실수형 출력은 NaN으로 초기화한다.
                for window in _windows(first):
                    if cancelled():
                        raise InterruptedError
                    shape = (int(window.height), int(window.width))
                    for name, ds in outputs.items():
                        ds.write(
                            np.full(
                                shape,
                                np.nan
                                if name in float_names
                                else 65535
                                if name == "quality_flags"
                                else 0,
                                dtype="float32" if name in float_names else "uint16",
                            ),
                            1,
                            window=window,
                        )
                for name, ds in outputs.items():
                    ds.set_band_description(1, name)
                    ds.update_tags(
                        method="Semi-analytical shallow-water inversion / D2",
                        coefficient_set=coeff["id"],
                        sensor=config["sensor"],
                        water_type=config["water_type"],
                    )
                f = stack.enter_context(
                    (out / "pixel_results.csv").open("w", newline="", encoding="utf-8-sig")
                )
                fields = [
                    "row",
                    "column",
                    "x",
                    "y",
                    "depth_m",
                    "pair_a",
                    "pair_b",
                    "F",
                    "C4",
                    "relative_RMS_percent",
                    "status",
                    "quality_flags",
                ] + ["fraction_" + k for k in library]
                writer = csv.DictWriter(f, fields)
                writer.writeheader()
                errorfile = stack.enter_context(
                    (out / "errors.csv").open("w", newline="", encoding="utf-8-sig")
                )
                errors = csv.writer(errorfile)
                errors.writerow(["row", "column", "reason"])
                _init_worker(
                    fit["context"], library, settings, str(cancel_path) if cancel_path else None
                )
                if workers > 1:
                    pool = ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=_init_worker,
                        initargs=(
                            fit["context"],
                            library,
                            settings,
                            str(cancel_path) if cancel_path else None,
                        ),
                    )
                stop = False
                for window in _windows(first):
                    if cancelled() or counts["processed"] >= target:
                        break
                    data, valid = _read_block(datasets, mask, window)
                    coords = np.argwhere(valid)
                    arrays = {
                        name: np.full(
                            valid.shape,
                            np.nan
                            if name in float_names
                            else 65535
                            if name == "quality_flags"
                            else 0,
                            dtype="float32" if name in float_names else "uint16",
                        )
                        for name in outputs
                    }
                    for start in range(0, len(coords), 64):
                        if cancelled() or counts["processed"] >= target:
                            stop = True
                            break
                        batch = coords[start : start + min(64, target - counts["processed"])]
                        items = [(i, data[:, r, c]) for i, (r, c) in enumerate(batch)]
                        solved = (
                            pool.map(_solve_item, items, chunksize=2)
                            if pool
                            else map(_solve_item, items)
                        )
                        try:
                            for index, result, error in solved:
                                if error == "CANCELLED":
                                    stop = True
                                    break
                                r, c = map(int, batch[index])
                                row = r + int(window.row_off)
                                col = c
                                x, y = xy(first.transform, row, col)
                                counts["processed"] += 1
                                if error:
                                    counts["failed"] += 1
                                    arrays["status"][r, c] = 2
                                    arrays["quality_flags"][r, c] = 16
                                    errors.writerow([row, col, error])
                                    writer.writerow(
                                        dict(
                                            row=row,
                                            column=col,
                                            x=x,
                                            y=y,
                                            status="failed",
                                            quality_flags=16,
                                        )
                                    )
                                    continue
                                counts["success"] += 1
                                z = float(result["depth_m"])
                                depth_min = min(depth_min, z)
                                depth_max = max(depth_max, z)
                                outside = any(v < 0 or v > 1 for v in result["fractions"].values())
                                limits = result["any_evaluation_limit"]
                                alternative = result["has_alternative"]
                                bad_bottom = result.get(
                                    "bottom_reflectance_negative", False
                                ) or result.get("bottom_reflectance_above_one", False)
                                quality = (
                                    int(outside)
                                    + 2 * int(limits)
                                    + 4 * int(bad_bottom)
                                    + 8 * int(alternative)
                                )
                                counts["outside"] += int(outside)
                                counts["limits"] += int(limits)
                                counts["alternatives"] += int(alternative)
                                arrays["depth_m"][r, c] = z
                                arrays["C4"][r, c] = result["c4"]
                                arrays["relative_RMS_percent"][r, c] = result[
                                    "rms_relative_error_pct"
                                ]
                                arrays["pair_code"][r, c] = pair_codes[tuple(result["pair"])]
                                arrays["status"][r, c] = 1
                                arrays["quality_flags"][r, c] = quality
                                for key, v in result["fractions"].items():
                                    arrays["fraction_" + _slug(key)][r, c] = v
                                writer.writerow(
                                    dict(
                                        row=row,
                                        column=col,
                                        x=x,
                                        y=y,
                                        depth_m=z,
                                        pair_a=result["pair"][0],
                                        pair_b=result["pair"][1],
                                        F=result["F"],
                                        C4=result["c4"],
                                        relative_RMS_percent=result["rms_relative_error_pct"],
                                        status=result["status"],
                                        quality_flags=quality,
                                        **{
                                            "fraction_" + k: v
                                            for k, v in result["fractions"].items()
                                        },
                                    )
                                )
                        finally:
                            # 결과 수신 도중 예외가 발생해도 해당 묶음의 완료 픽셀을 기록한다.
                            # GeoTIFF는 ExitStack에서 닫히며 CSV 버퍼는 여기서 디스크에 반영한다.
                            for name, ds in outputs.items():
                                ds.write(arrays[name], 1, window=window)
                            f.flush()
                            errorfile.flush()
                        event(
                            {
                                "type": "progress",
                                "stage": "pixels",
                                "done": counts["processed"],
                                "total": target,
                                "message": f"Depth calculation: {counts['processed']:,} / {target:,} pixels",
                            }
                        )
                        if stop:
                            break
                    if stop:
                        break
        finally:
            if pool:
                pool.shutdown(wait=True, cancel_futures=True)
        sync_counts()
        summary["status"] = (
            "cancelled"
            if cancelled() or counts["processed"] < target
            else "completed_with_failures"
            if counts["failed"]
            else "completed"
        )
        event({"type": "status", "message": "Finalizing result maps and the analysis record."})
        if counts["success"]:
            try:
                from .report import create_report

                create_report(out, summary)
            except Exception as exc:
                # 지도/CSV 계산 완료와 미리보기 생성 실패는 별개로 기록한다.
                summary["report_error"] = str(exc)
                (out / "report_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        save_json(out / "analysis_record.json", summary)
        event(
            {
                "type": "cancelled" if summary["status"] == "cancelled" else "complete",
                "output_dir": str(out),
                "message": "Completed" if summary["status"] == "completed" else summary["status"],
            }
        )
        return summary
    except InterruptedError:
        sync_counts()
        summary["status"] = "cancelled"
        save_json(out / "analysis_record.json", summary)
        event({"type": "cancelled", "output_dir": str(out), "message": "Processing was stopped by the user."})
        return summary
    except Exception as exc:
        sync_counts()
        summary.update(status="failed", error=str(exc))
        save_json(out / "analysis_record.json", summary)
        (out / "error_details.txt").write_text(traceback.format_exc(), encoding="utf-8")
        event({"type": "error", "message": str(exc), "output_dir": str(out)})
        raise
