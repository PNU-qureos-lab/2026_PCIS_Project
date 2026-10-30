"""Create a GEBCO-constrained shallow-water inversion mask.

The semi-analytical model should be inverted over shallow water, not over the
entire open ocean.  This utility reprojects GEBCO elevation to the exact Rrs
grid and keeps pixels from 0 to 30 m below mean sea level.  It then intersects
that condition with the glint workflow's valid-water mask.  The output remains
on the native input grid; no Rrs data are resampled.

GEBCO is much coarser than Planet Dove, so the output is an analysis-domain
mask rather than a source of 3 m bathymetric detail.  Bilinear interpolation is
used only for the GEBCO screening surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


def create_mask(
    reference_raster: Path,
    water_mask: Path,
    gebco: Path,
    output: Path,
    maximum_depth_m: float,
) -> None:
    """Write a uint8 mask where valid water has GEBCO depth 0-max_depth_m."""
    output = output.resolve()
    with rasterio.open(reference_raster) as reference, rasterio.open(water_mask) as water:
        if (
            water.width != reference.width
            or water.height != reference.height
            or water.transform != reference.transform
            or water.crs != reference.crs
        ):
            raise ValueError("The valid-water mask does not match the Rrs grid.")

        profile = reference.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=0, compress="deflate", predictor=2)
        output.parent.mkdir(parents=True, exist_ok=True)
        accepted = 0

        with rasterio.open(gebco) as elevation_source, WarpedVRT(
            elevation_source,
            crs=reference.crs,
            transform=reference.transform,
            width=reference.width,
            height=reference.height,
            resampling=Resampling.bilinear,
            nodata=np.nan,
        ) as elevation, rasterio.open(output, "w", **profile) as target:
            for _, window in reference.block_windows(1):
                z = elevation.read(1, window=window, out_dtype="float32")
                valid_water = water.read(1, window=window) > 0
                shallow = valid_water & np.isfinite(z) & (z < 0) & (z >= -maximum_depth_m)
                target.write(shallow.astype("uint8"), 1, window=window)
                accepted += int(shallow.sum())

            target.set_band_description(1, f"valid_water_and_GEBCO_0_to_{maximum_depth_m:g}m")
            target.update_tags(
                purpose="Semi-analytical shallow-water inversion domain",
                gebco_source=str(gebco.resolve()),
                maximum_depth_m=str(maximum_depth_m),
                values="1=process, 0=exclude",
            )

    print(f"Created {output} | shallow-water pixels={accepted:,}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_raster", type=Path, help="Rrs raster defining the target grid")
    parser.add_argument("water_mask", type=Path, help="Valid-water mask on the same grid")
    parser.add_argument("gebco", type=Path, help="GEBCO elevation NetCDF or raster")
    parser.add_argument("output", type=Path, help="Output uint8 GeoTIFF")
    parser.add_argument("--maximum-depth-m", type=float, default=30.0)
    args = parser.parse_args()
    if args.maximum_depth_m <= 0:
        parser.error("--maximum-depth-m must be positive")
    create_mask(
        args.reference_raster,
        args.water_mask,
        args.gebco,
        args.output,
        args.maximum_depth_m,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
