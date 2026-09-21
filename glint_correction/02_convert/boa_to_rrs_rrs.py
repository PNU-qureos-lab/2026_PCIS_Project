#!/usr/bin/env python3
"""Convert final glint-corrected BOA reflectance GeoTIFFs to Rrs and rrs."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio


def convert(source: Path, output_dir: Path) -> tuple[Path, Path]:
    """Write Rrs_approx = BOA/pi and rrs_approx = Rrs/(0.52 + 1.7*Rrs)."""
    with rasterio.open(source) as ds:
        boa = ds.read(1).astype(np.float32)
        profile = ds.profile.copy()
        nodata = ds.nodata
    valid = np.isfinite(boa)
    if nodata is not None and np.isfinite(nodata):
        valid &= boa != nodata
    rrs_above = np.full(boa.shape, np.nan, dtype=np.float32)
    rrs_below = np.full(boa.shape, np.nan, dtype=np.float32)
    rrs_above[valid] = boa[valid] / math.pi
    rrs_below[valid] = rrs_above[valid] / (0.52 + 1.7 * rrs_above[valid])
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    rrs_path = output_dir / f"{stem}_Rrs_approx_sr-1.tif"
    rrs_lower_path = output_dir / f"{stem}_rrs_approx_sr-1.tif"
    for path, array in ((rrs_path, rrs_above), (rrs_lower_path, rrs_below)):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array, 1)
            dst.update_tags(
                quantity=("Rrs_approx" if path == rrs_path else "rrs_approx"),
                units="sr-1",
                formula=("Rrs_approx = final_BOA_reflectance / pi" if path == rrs_path else "rrs_approx = Rrs_approx / (0.52 + 1.7 * Rrs_approx)"),
            )
    return rrs_path, rrs_lower_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Final glint-corrected BOA GeoTIFF")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    upper, lower = convert(args.source, args.output_dir)
    print(f"Wrote {upper}")
    print(f"Wrote {lower}")


if __name__ == "__main__":
    main()
