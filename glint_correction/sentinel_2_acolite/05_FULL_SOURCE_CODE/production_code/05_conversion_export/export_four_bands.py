"""Create one portable folder of final corrected four-band Rrs and rrs products."""
from __future__ import annotations

# WORKFLOW ROLE — STEP 05B: FOUR-BAND PACKAGE
# Package B01–B04 Rrs/rrs rasters for bathymetry or benthic processing and
# verify that CRS, grid, shape, NoData, and conversion equations agree.

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(r"H:\Sentinel-2\Tidung\Tidung_Glint_Corrected_Products_20260913")
FINAL = ROOT / "FINAL_PRODUCTS"
OUT = ROOT / "FINAL_Rrs_and_rrs_B01_B04_All_Scenes_20m"
BANDS = ("B01", "B02", "B03", "B04")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for scene_dir in sorted(p for p in FINAL.iterdir() if p.is_dir()):
        scene = scene_dir.name
        destination = OUT / scene
        destination.mkdir(exist_ok=True)
        sources = []
        for band in BANDS:
            source = scene_dir / "02_Rrs_Approx" / f"{scene}_{band}_Rrs_approx_sr-1.tif"
            if not source.exists():
                raise FileNotFoundError(source)
            target = destination / f"{scene}_{band}_above_Rrs_approx_sr-1.tif"
            shutil.copy2(source, target)
            sources.append(("Rrs_approx", band, source, target))
            lower_target = destination / f"{scene}_{band}_below_rrs_approx_sr-1.tif"
            with rasterio.open(source) as upper:
                Rrs = upper.read(1).astype(np.float32)
                rrs = np.full(Rrs.shape, np.nan, dtype=np.float32)
                valid = np.isfinite(Rrs)
                rrs[valid] = Rrs[valid] / (0.52 + 1.7 * Rrs[valid])
                profile = upper.profile.copy()
                profile.update(dtype="float32", nodata=np.nan, compress="deflate", predictor=3)
                with rasterio.open(lower_target, "w", **profile) as lower:
                    lower.write(rrs, 1)
                    lower.set_band_description(1, "Below-surface rrs_approx (sr-1), derived from final Rrs_approx")
            sources.append(("rrs_approx", band, source, lower_target))
        ref = None
        max_formula_error = 0.0
        for kind, band, source, target in sources:
            with rasterio.open(target) as ds:
                nodata = ds.nodata
                # NaN is the shared NoData value; NaN != NaN in a direct comparison.
                nodata_key = "NaN" if nodata is not None and np.isnan(nodata) else nodata
                signature = (ds.crs, ds.transform, ds.width, ds.height, nodata_key)
                if ref is None:
                    ref = signature
                elif signature != ref:
                    raise RuntimeError(f"Grid mismatch in {scene}: {target.name}")
        for band in BANDS:
            rrs_path = destination / f"{scene}_{band}_below_rrs_approx_sr-1.tif"
            rrs_upper_path = destination / f"{scene}_{band}_above_Rrs_approx_sr-1.tif"
            with rasterio.open(rrs_upper_path) as upper, rasterio.open(rrs_path) as lower:
                Rrs = upper.read(1).astype(float)
                rrs = lower.read(1).astype(float)
                valid = np.isfinite(Rrs) & np.isfinite(rrs)
                expected = Rrs / (0.52 + 1.7 * Rrs)
                if valid.any():
                    max_formula_error = max(max_formula_error, float(np.max(np.abs(rrs[valid] - expected[valid]))))
        provenance = scene_dir / "05_Provenance" / "B01_processing_and_grid_provenance.json"
        correction = json.loads(provenance.read_text(encoding="utf-8")).get("final_product") if provenance.exists() else "final retained product"
        summary.append({
            "scene": scene,
            "final_correction": correction,
            "bands": "B01,B02,B03,B04",
            "above_surface_Rrs_definition": "Rrs_approx = final glint-corrected Sen2Cor BOA reflectance / pi",
            "below_surface_rrs_definition": "rrs_approx = Rrs_approx / (0.52 + 1.7 * Rrs_approx)",
            "max_rrs_conversion_error": max_formula_error,
            "output_folder": str(destination),
        })
    with (OUT / "all_scene_Rrs_rrs_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUT / "README.txt").write_text(
        "Final retained glint-corrected Sentinel-2 products.\n\n"
        "Each scene folder contains B01 (443 nm), B02 (490 nm), B03 (560 nm), and B04 (665 nm).\n"
        "Rrs_approx is above-surface reflectance in sr-1.\n"
        "rrs_approx is below-surface reflectance in sr-1, calculated as Rrs_approx / (0.52 + 1.7*Rrs_approx).\n"
        "Use Rrs_approx for the revised inversion code unless its complete model explicitly requests rrs.\n",
        encoding="utf-8",
    )
    print(f"Exported {len(summary)} scenes to {OUT}")


if __name__ == "__main__":
    main()
