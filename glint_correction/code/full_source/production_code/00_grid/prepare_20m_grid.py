#!/usr/bin/env python3
"""Prepare Sentinel-2 BOA reflectance bands on one aligned 20 m grid.

Input rasters must already be BOA reflectance (not Sentinel-2 integer DN).
The output is ready for the B11/B12 glint-correction step.
"""
from __future__ import annotations

# WORKFLOW ROLE — STEP 00: COMMON GRID
# Input: Sen2Cor L2A BOA bands at their native 10 m, 20 m, or 60 m grids.
# Output: all bands aligned to one B11-defined 20 m grid.
# Scientific caution: B01 alignment from 60 m does not create 20 m detail.

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


# Sentinel-2 bands used in the Tidung final products.
BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")


def find_band(input_dir: Path, band: str) -> Path:
    """Find one BOA GeoTIFF whose name includes the requested band."""
    matches = sorted(p for p in input_dir.glob("*.tif") if band in p.stem.split("_"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {band} GeoTIFF in {input_dir}; found: {matches}")
    return matches[0]


def write_20m(source: Path, reference_profile: dict, target: Path) -> None:
    """Resample one BOA raster to the exact reference 20 m grid."""
    with rasterio.open(source) as src:
        source_data = src.read(1).astype(np.float32)
        destination = np.full(
            (reference_profile["height"], reference_profile["width"]),
            np.nan,
            dtype=np.float32,
        )
        reproject(
            source=source_data,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=reference_profile["transform"],
            dst_crs=reference_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    profile = reference_profile.copy()
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(destination, 1)
        dst.set_band_description(1, f"{source.stem}: BOA reflectance on aligned 20 m grid")
        dst.update_tags(
            source_band=source.stem,
            processing="Resampled to common B11 20 m grid using bilinear interpolation",
            spatial_resolution_note="B01 is native 60 m; resampling aligns pixels but creates no 20 m detail.",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an aligned 20 m Sentinel-2 BOA stack")
    parser.add_argument("input_dir", type=Path, help="Folder containing one BOA GeoTIFF for every band")
    parser.add_argument("output_dir", type=Path, help="Folder for aligned 20 m GeoTIFFs")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional existing 20 m GeoTIFF to define the grid; default is B11",
    )
    args = parser.parse_args()
    reference = args.reference or find_band(args.input_dir, "B11")
    with rasterio.open(reference) as ds:
        reference_profile = ds.profile.copy()
        if abs(ds.transform.a) != 20 or abs(ds.transform.e) != 20:
            raise ValueError("The reference raster must have a 20 m pixel size")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for band in BANDS:
        source = find_band(args.input_dir, band)
        target = args.output_dir / f"{band}_BOA_20m.tif"
        write_20m(source, reference_profile, target)
        print(f"Wrote {target.name}")


if __name__ == "__main__":
    main()
