"""Validate a generated Tidung standard-Hedley package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from osgeo import gdal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    root = args.package.resolve()
    errors: list[str] = []
    required = [
        root / "README.md",
        root / "Tidung_Planet_Dove_Standard_Hedley_Results.html",
        root / "00_READ_ME_FIRST" / "processing_manifest.json",
        root / "00_READ_ME_FIRST" / "master_scene_qc.csv",
        root / "00_READ_ME_FIRST" / "master_band_qc.csv",
        root / "05_FULL_SOURCE_CODE" / "production_code" / "process_planetscope_standard_hedley.py",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"Missing required file: {path}")
    manifest = json.loads((root / "00_READ_ME_FIRST" / "processing_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("custom_safeguards") is not False:
        errors.append("Manifest does not explicitly disable custom safeguards")
    if manifest.get("negative_clipping") is not False:
        errors.append("Manifest does not explicitly disable negative clipping")
    raster_checks = []
    for stack in sorted((root / "04_GENERATED_OUTPUT").glob("*/04_Corrected_SR/*_Standard_Hedley_SR_8b.tif")):
        dataset = gdal.Open(str(stack))
        if dataset is None or dataset.RasterCount != 8:
            errors.append(f"Invalid corrected stack: {stack}")
            continue
        finite_minima = []
        for index in range(1, 8):
            band = dataset.GetRasterBand(index)
            array = band.ReadAsArray().astype(float)
            nodata = band.GetNoDataValue()
            valid = np.isfinite(array) & (array != nodata)
            finite_minima.append(float(array[valid].min()) if valid.any() else None)
        raster_checks.append({"file": str(stack), "finite_minima_444_707": finite_minima})
        dataset = None
    with (root / "00_READ_ME_FIRST" / "master_scene_qc.csv").open(encoding="utf-8") as handle:
        scenes = list(csv.DictReader(handle))
    if len(raster_checks) != len(scenes):
        errors.append(f"Corrected stack count {len(raster_checks)} does not match scene count {len(scenes)}")
    result = {"package": str(root), "ok": not errors, "errors": errors, "scene_count": len(scenes), "raster_checks": raster_checks}
    output = root / "00_READ_ME_FIRST" / "validation_results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
