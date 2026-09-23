#!/usr/bin/env python
"""Run the complete documented 20260808_S2A Tidung worked example.

This is intentionally verbose.  Every numbered block writes a numbered output
folder so a reader can inspect exactly what entered and left each stage.

External scientific products used as inputs:
  * Sentinel-2 L2A BOA reflectance created by Sen2Cor.
  * Glint-ACOLITE B11/B12 choice, rhog_ref and optical transfer factors.
  * A 675-spectrum HydroLight water-only simulation library.

The script does not pretend to be the ACOLITE or HydroLight executables.  It
reproduces the downstream glint correction, validation bookkeeping and product
conversion using the traceable outputs from those programs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
INPUT = PACKAGE_ROOT / "01_INPUT"
OUTPUT = PACKAGE_ROOT / "04_GENERATED_OUTPUT"
SCENE = "20260808_S2A"
OPTICAL_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A")
INPUT_BANDS = (*OPTICAL_BANDS, "B09", "B11", "B12")
EXPORT_BANDS = ("B01", "B02", "B03", "B04")
WAVELENGTH = {"B01": 443, "B02": 492, "B03": 560, "B04": 665,
              "B05": 704, "B06": 740, "B07": 783, "B08": 833, "B8A": 865}


def read_tif(path: Path) -> tuple[np.ndarray, dict]:
    """Read one raster and turn its declared NoData value into NaN."""
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float64)
        if ds.nodata is not None:
            data[data == ds.nodata] = np.nan
        return data, ds.profile.copy()


def write_tif(path: Path, data: np.ndarray, profile: dict, description: str) -> None:
    """Write a compact, georeferenced floating-point output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan,
                       compress="deflate", predictor=3, tiled=True,
                       blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **out_profile) as ds:
        ds.write(data.astype("float32"), 1)
        ds.set_band_description(1, description)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_rgb(path: Path, bands: dict[str, np.ndarray]) -> None:
    """Create one consistent B04/B03/B02 inspection preview."""
    rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1)
    valid = np.all(np.isfinite(rgb), axis=2)
    values = rgb[valid]
    low = np.percentile(values, 2, axis=0)
    high = np.percentile(values, 98, axis=0)
    shown = np.clip((rgb - low) / np.maximum(high - low, 1e-8), 0, 1) ** 0.8
    rgba = np.zeros((*valid.shape, 4), dtype=np.uint8)
    # Transparent pixels contain NaN in the scientific raster. Replace those
    # values only in this display copy so NumPy can safely convert it to bytes.
    rgba[..., :3] = np.round(np.nan_to_num(shown, nan=0.0) * 255).astype(np.uint8)
    rgba[..., 3] = valid.astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(path)


def step_01_verify_inputs() -> tuple[dict[str, np.ndarray], dict]:
    """Step 1: load prepared L2A bands and prove that their grids match."""
    out = OUTPUT / "01_prepared_input"
    bands: dict[str, np.ndarray] = {}
    audit = []
    reference = None
    reference_profile = None
    for band in INPUT_BANDS:
        path = INPUT / "01_l2a_boa_20m" / f"{band}_BOA_20m.tif"
        data, profile = read_tif(path)
        signature = (str(profile["crs"]), tuple(profile["transform"]),
                     profile["width"], profile["height"])
        if reference is None:
            reference, reference_profile = signature, profile
        if signature != reference:
            raise RuntimeError(f"Grid mismatch for {band}: {signature} != {reference}")
        bands[band] = data
        finite = data[np.isfinite(data)]
        audit.append({"band": band, "width": profile["width"], "height": profile["height"],
                      "crs": str(profile["crs"]), "pixel_size_m": abs(profile["transform"].a),
                      "finite_pixels": finite.size, "minimum": float(finite.min()),
                      "median": float(np.median(finite)), "maximum": float(finite.max())})
    write_csv(out / "grid_and_value_audit.csv", audit)
    save_rgb(out / "original_L2A_RGB.png", bands)
    (out / "B01_resolution_note.txt").write_text(
        "B01 is aligned to the 20 m grid but retains native 60 m spatial information.\n",
        encoding="utf-8")
    return bands, reference_profile


def step_02_glint_acolite_cycle0(
    original: dict[str, np.ndarray], profile: dict
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Step 2: reproduce Glint-ACOLITE Cycle 0 for every optical band.

    ACOLITE has already supplied the per-pixel B11/B12 selection, rhog_ref and
    wavelength-transfer factors.  We use those traceable products here.
    """
    out = OUTPUT / "02_glint_acolite_cycle0"
    support = INPUT / "02_acolite_support"
    use_b11, _ = read_tif(support / "SWIR_selection_B11_is_1.tif")
    rhog, _ = read_tif(support / "rhog_ref.tif")
    factors = {row["band"]: row for row in csv_rows(support / "multiband_glint_factors.csv")}
    cycle0: dict[str, np.ndarray] = {}
    estimated: dict[str, np.ndarray] = {}
    summary = []
    for band in OPTICAL_BANDS:
        factor11 = float(factors[band]["factor_from_B11"])
        factor12 = float(factors[band]["factor_from_B12"])
        # The accepted selection raster chooses which transferred estimate is
        # used at each pixel; it is not a LOW/MEDIUM/HIGH classification.
        glint = np.where(use_b11 > 0.5, factor11 * rhog, factor12 * rhog)
        valid = np.isfinite(original[band]) & np.isfinite(glint)
        glint = np.where(valid, glint, np.nan)
        corrected = np.where(valid, original[band] - glint, np.nan)
        estimated[band], cycle0[band] = glint, corrected
        write_tif(out / f"estimated_glint_{band}.tif", glint, profile,
                  f"Glint-ACOLITE estimated component for {band}")
        write_tif(out / f"Cycle0_{band}.tif", corrected, profile,
                  f"Glint-ACOLITE Cycle 0 corrected {band} BOA reflectance")
        summary.append({"band": band, "factor_from_B11": factor11,
                        "factor_from_B12": factor12,
                        "median_glint_removed": float(np.nanmedian(glint)),
                        "negative_percent": float(100 * np.nanmean(corrected < 0))})
    write_csv(out / "cycle0_summary.csv", summary)
    save_rgb(out / "Cycle0_RGB.png", cycle0)
    return cycle0, estimated


def step_03_hydrolight_library() -> list[dict[str, str]]:
    """Step 3: expose all 675 HydroLight simulations and draw every spectrum."""
    out = OUTPUT / "03_hydrolight_simulations"
    out.mkdir(parents=True, exist_ok=True)
    source = INPUT / "03_hydrolight_library" / "model_library_B02_B8A.csv"
    rows = csv_rows(source)
    shutil.copy2(source, out / source.name)
    bands = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A")
    wave = [WAVELENGTH[b] for b in bands]
    values = [[float(row[b]) for b in bands] for row in rows]
    width, height = 1100, 600
    left, right, top, bottom = 80, 30, 60, 65
    ymax = max(max(v) for v in values) * 1.03
    x = lambda i: left + i / (len(wave) - 1) * (width - left - right)
    y = lambda v: height - bottom - v / ymax * (height - top - bottom)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="23" font-weight="700">All {len(rows)} HydroLight water simulations</text>']
    for spectrum in values:
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(spectrum))
        svg.append(f'<polyline points="{points}" fill="none" stroke="#087e73" stroke-opacity=".10"/>')
    for i, wavelength in enumerate(wave):
        svg.append(f'<text x="{x(i):.1f}" y="{height-30}" text-anchor="middle" font-family="Arial" font-size="13">{wavelength}</text>')
    svg += [f'<text x="{width/2}" y="{height-5}" text-anchor="middle" font-family="Arial">Wavelength (nm)</text>', '</svg>']
    (out / "all_675_HydroLight_spectra.svg").write_text("".join(svg), encoding="utf-8")
    platforms = sorted({row["platform"] for row in rows})
    summary = {"simulations": len(rows), "platforms": platforms,
               "solar_zenith_deg": sorted({float(r["sza"]) for r in rows}),
               "chlorophyll_values": sorted({float(r["chl"]) for r in rows}),
               "cdom_values": sorted({float(r["cdom"]) for r in rows}),
               "tsm_values": sorted({float(r["tsm"]) for r in rows}),
               "wind_m_s": sorted({float(r["wind"]) for r in rows})}
    (out / "library_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows


def step_04_deep_water_validation() -> dict:
    """Step 4: preserve the real 30-ROI residual fits and accepted gate."""
    source = INPUT / "04_validation_inputs"
    out = OUTPUT / "04_deep_water_validation"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("selected_30_offshore_rois.csv", "all_30_roi_residual_fits.csv",
                 "scene_residual_gate.json", "deep_water_roi_map.png",
                 "hydrolight_matches.png", "residual_evidence.png"):
        shutil.copy2(source / name, out / name)
    fits = csv_rows(source / "all_30_roi_residual_fits.csv")
    gate = json.loads((source / "scene_residual_gate.json").read_text(encoding="utf-8"))
    detected = sum(row["detected"].lower() == "true" for row in fits)
    summary = {"scene": SCENE, "ROI_count": len(fits), "residual_detections": detected,
               "scene_gate_pass": bool(gate["pass"]),
               "meaning": "Cycle 1 may proceed only when the full gate passes."}
    (out / "deep_water_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return gate


def step_05_cycle1_decision(cycle0: dict[str, np.ndarray], profile: dict) -> dict[str, np.ndarray]:
    """Step 5: apply the recorded Cycle 1 decision.

    This example retained Cycle 0.  Therefore Final is an exact copy of Cycle 0
    and no additional reflectance is removed.
    """
    out = OUTPUT / "05_cycle1_decision"
    source = INPUT / "04_validation_inputs" / "final_qc.json"
    qc = json.loads(source.read_text(encoding="utf-8"))
    retained = qc["retained_product"]
    if retained != "Cycle0":
        raise RuntimeError("This compact example expects the recorded 20260808_S2A Cycle0 decision.")
    final = {band: data.copy() for band, data in cycle0.items()}
    out.mkdir(parents=True, exist_ok=True)
    for band, data in final.items():
        write_tif(out / f"final_{band}_BOA.tif", data, profile,
                  f"Final retained {band} BOA reflectance ({retained})")
    decision = {"scene": SCENE, "deep_water_gate_pass": qc["scene_gate_pass"],
                "retained_product": retained, "cycle1_applied": False,
                "reason": "The deep-water residual gate did not authorize Cycle 1."}
    (out / "final_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    save_rgb(out / "final_RGB.png", final)
    return final


def step_06_shallow_water_check(
    original: dict[str, np.ndarray], cycle0: dict[str, np.ndarray], final: dict[str, np.ndarray]
) -> None:
    """Step 6: sample the same 30 shallow-water 7×7 ROIs at all stages."""
    out = OUTPUT / "06_shallow_water_check"
    source = INPUT / "04_validation_inputs" / "shallow_30_rois.csv"
    roi_rows = csv_rows(source)
    result = []
    stages = {"Original": original, "Cycle0": cycle0, "Final": final}
    for roi in roi_rows:
        row, col = int(float(roi["row"])), int(float(roi["column"]))
        window = np.s_[row-3:row+4, col-3:col+4]
        record = {"roi": roi["roi"], "row": row, "column": col}
        for stage_name, stage_bands in stages.items():
            for band in OPTICAL_BANDS:
                record[f"{stage_name}_{band}"] = float(np.nanmean(stage_bands[band][window]))
        result.append(record)
    write_csv(out / "same_30_shallow_ROIs_before_after.csv", result)
    median_rows = []
    for stage_name in stages:
        for band in OPTICAL_BANDS:
            median_rows.append({"stage": stage_name, "band": band,
                                "wavelength_nm": WAVELENGTH[band],
                                "median_BOA": float(np.median([r[f"{stage_name}_{band}"] for r in result]))})
    write_csv(out / "all_30_ROI_median_before_after.csv", median_rows)
    shutil.copy2(INPUT / "04_validation_inputs" / "shallow_all30_before_after.svg",
                 out / "shallow_all30_before_after.svg")
    shutil.copy2(INPUT / "04_validation_inputs" / "shallow_median_before_after.svg",
                 out / "shallow_median_before_after.svg")


def step_07_export(final: dict[str, np.ndarray], profile: dict) -> None:
    """Step 7: export final BOA, above-surface Rrs and below-surface rrs."""
    out = OUTPUT / "07_final_products"
    qc_rows = []
    for band in EXPORT_BANDS:
        boa = final[band]
        rrs = boa / math.pi
        sub = rrs / (0.52 + 1.7 * rrs)
        write_tif(out / f"{SCENE}_{band}_final_BOA.tif", boa, profile, f"Final {band} BOA reflectance")
        # Do not distinguish these two filenames by letter case alone: Windows
        # filesystems treat Rrs and rrs as the same name. These explicit names
        # preserve both scientifically different products on every platform.
        write_tif(out / f"{SCENE}_{band}_above_surface_Rrs_sr-1.tif", rrs, profile,
                  f"Above-surface Rrs {band}")
        write_tif(out / f"{SCENE}_{band}_below_surface_rrs_sr-1.tif", sub, profile,
                  f"Below-surface rrs {band}")
        finite = np.isfinite(rrs)
        qc_rows.append({"band": band, "valid_pixels": int(finite.sum()),
                        "Rrs_min": float(np.nanmin(rrs)), "Rrs_median": float(np.nanmedian(rrs)),
                        "Rrs_mean": float(np.nanmean(rrs)), "Rrs_max": float(np.nanmax(rrs)),
                        "negative_percent": float(100 * np.mean(rrs[finite] < 0))})
    write_csv(out / "final_product_QC.csv", qc_rows)


def write_manifest() -> None:
    rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file():
            rows.append({"step": path.relative_to(OUTPUT).parts[0],
                         "file": path.relative_to(OUTPUT).as_posix(),
                         "bytes": path.stat().st_size})
    write_csv(OUTPUT / "workflow_output_manifest.csv", rows)


def main() -> None:
    """Run the portable workflow with included or user-supplied inputs."""
    global INPUT, OUTPUT, SCENE
    parser = argparse.ArgumentParser(
        description="Run the complete one-scene Sentinel-2 Glint-ACOLITE example"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=INPUT,
        help="Workflow input directory (default: included input folder)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT,
        help="Directory for regenerated numbered outputs",
    )
    parser.add_argument(
        "--scene", default=SCENE,
        help="Scene identifier used in final output filenames",
    )
    args = parser.parse_args()
    INPUT = args.input_dir.resolve()
    OUTPUT = args.output_dir.resolve()
    SCENE = args.scene

    OUTPUT.mkdir(parents=True, exist_ok=True)
    original, profile = step_01_verify_inputs()
    cycle0, _estimated = step_02_glint_acolite_cycle0(original, profile)
    step_03_hydrolight_library()
    step_04_deep_water_validation()
    final = step_05_cycle1_decision(cycle0, profile)
    step_06_shallow_water_check(original, cycle0, final)
    step_07_export(final, profile)
    write_manifest()
    print(f"Complete worked example written to: {OUTPUT}")


if __name__ == "__main__":
    main()
