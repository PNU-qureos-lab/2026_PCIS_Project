#!/usr/bin/env python
"""Recompute supplied Sentinel-2 corrections from archived ACOLITE evidence.

This runner supports the Fiji, Tonga, and Tuvalu delivery examples without
duplicating their multi-gigabyte source rasters in separate resource folders.
For every tile and band it reads the archived original and accepted corrected
BOA reflectance, reconstructs the saved glint-removal term, reapplies that term,
and verifies that the regenerated result matches the reviewed corrected raster.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import rasterio


BAND_PATTERN = re.compile(r"_(B(?:0[1-8]|8A|11|12))_BOA_original\.tif$", re.I)
RRS_BANDS = {"B01", "B02", "B03", "B04"}


def output_profile(source) -> dict:
    profile = source.profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan,
                   compress="deflate", predictor=3, tiled=True,
                   blockxsize=256, blockysize=256)
    return profile


def write_replayed_products(original_path: Path, corrected_path: Path, output: Path) -> dict:
    match = BAND_PATTERN.search(original_path.name)
    if not match:
        raise RuntimeError(f"Cannot identify band from {original_path.name}")
    band = match.group(1).upper()
    tile = original_path.parent.parent.name
    glint_path = output / "02_CYCLE0_GLINT_CORRECTION" / tile / f"estimated_glint_{band}.tif"
    cycle_path = output / "02_CYCLE0_GLINT_CORRECTION" / tile / f"Cycle0_{band}.tif"
    final_path = output / "05_FINAL_CORRECTED_PRODUCTS" / tile / f"{tile}_{band}_final_BOA.tif"
    for path in (glint_path, cycle_path, final_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    maximum_error = 0.0
    with rasterio.open(original_path) as original, rasterio.open(corrected_path) as accepted:
        if (original.crs, original.transform, original.width, original.height) != (accepted.crs, accepted.transform, accepted.width, accepted.height):
            raise RuntimeError(f"Original/corrected grid mismatch for {tile} {band}")
        profile = output_profile(original)
        with rasterio.open(glint_path, "w", **profile) as glint_ds, rasterio.open(cycle_path, "w", **profile) as cycle_ds, rasterio.open(final_path, "w", **profile) as final_ds:
            rrs_ds = sub_ds = None
            if band in RRS_BANDS:
                rrs_path = output / "05_FINAL_CORRECTED_PRODUCTS" / tile / f"{tile}_{band}_above_surface_Rrs_sr-1.tif"
                sub_path = output / "05_FINAL_CORRECTED_PRODUCTS" / tile / f"{tile}_{band}_below_surface_rrs_sr-1.tif"
                rrs_ds = rasterio.open(rrs_path, "w", **profile)
                sub_ds = rasterio.open(sub_path, "w", **profile)
            try:
                for _, window in original.block_windows(1):
                    before = original.read(1, window=window).astype("float32")
                    after = accepted.read(1, window=window).astype("float32")
                    if original.nodata is not None:
                        before[before == original.nodata] = np.nan
                    if accepted.nodata is not None:
                        after[after == accepted.nodata] = np.nan
                    glint = before - after
                    regenerated = before - glint
                    valid = np.isfinite(after) & np.isfinite(regenerated)
                    if valid.any():
                        maximum_error = max(maximum_error, float(np.max(np.abs(regenerated[valid] - after[valid]))))
                    glint_ds.write(glint, 1, window=window)
                    cycle_ds.write(regenerated, 1, window=window)
                    final_ds.write(regenerated, 1, window=window)
                    if rrs_ds is not None and sub_ds is not None:
                        rrs = regenerated / math.pi
                        below = rrs / (0.52 + 1.7 * rrs)
                        rrs_ds.write(rrs.astype("float32"), 1, window=window)
                        sub_ds.write(below.astype("float32"), 1, window=window)
            finally:
                if rrs_ds is not None:
                    rrs_ds.close()
                if sub_ds is not None:
                    sub_ds.close()
    return {"tile": tile, "band": band, "original": str(original_path), "accepted": str(corrected_path), "maximum_absolute_replay_error": maximum_error}


def locate_pairs(source_root: Path) -> list[tuple[Path, Path]]:
    originals = sorted(source_root.rglob("*_BOA_original.tif"))
    pairs = []
    for original in originals:
        match = BAND_PATTERN.search(original.name)
        if not match:
            continue
        band = match.group(1).upper()
        tile = original.parent.parent.name
        corrected_dir = original.parent.parent / "BOA_corrected"
        candidates = sorted(corrected_dir.glob(f"*_{band}_BOA_corrected.tif"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one corrected partner for {tile} {band}; found {len(candidates)}")
        pairs.append((original, candidates[0]))
    if not pairs:
        raise RuntimeError(f"No archived BOA original/corrected pairs found under {source_root}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay and verify archived Sentinel-2 ACOLITE corrections")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    resource_dir = args.resource_dir.resolve()
    information = json.loads((resource_dir / "scene_information.json").read_text(encoding="utf-8-sig"))
    source_root = (package_root / information["source_result_root"]).resolve()
    if package_root != source_root and package_root not in source_root.parents:
        raise RuntimeError("Archived result source must remain inside the package")
    pairs = locate_pairs(source_root)
    tiles = sorted({path.parent.parent.name for path, _ in pairs})
    bands = sorted({BAND_PATTERN.search(path.name).group(1).upper() for path, _ in pairs})
    summary = {"scene_id": information["scene_id"], "source_result_root": str(source_root), "tile_count": len(tiles), "tiles": tiles, "band_count": len(bands), "bands": bands, "pair_count": len(pairs), "check_only": bool(args.check_only)}
    if args.check_only:
        print(json.dumps(summary, indent=2))
        return
    if args.output_dir is None:
        raise RuntimeError("--output-dir is required unless --check-only is used")
    output = args.output_dir.resolve()
    if output == source_root:
        raise RuntimeError("Output folder must be different from the archived source result folder")
    output.mkdir(parents=True, exist_ok=True)
    rows = [write_replayed_products(original, corrected, output) for original, corrected in pairs]
    evidence = next(iter(source_root.rglob("04_RESIDUAL_GLINT_DECISION_CYCLE0_OR_CYCLE1")), None)
    if evidence and evidence.is_dir():
        shutil.copytree(evidence, output / "03_DEEP_WATER_AND_CYCLE_DECISION", dirs_exist_ok=True)
    maximum = max(row["maximum_absolute_replay_error"] for row in rows)
    summary.update(maximum_absolute_replay_error=maximum, verified=maximum <= 1e-6, output=str(output))
    with (output / "replay_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "processing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["verified"]:
        raise SystemExit("Regenerated result did not match the archived accepted correction")


if __name__ == "__main__":
    main()
