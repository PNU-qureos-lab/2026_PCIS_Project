"""Prepare a six-band SuperDove Rrs raster for semi-analytical v42.

The Planet Dove glint workflow stores an eight-band Rrs GeoTIFF in the order
444, 492, 533, 566, 612, 666, 707 and 866 nm.  Semi-analytical v42 uses the
six visible bands from 444 through 666 nm.  This utility copies bands 1-6
without resampling, reprojection, scaling, or changing NoData values.  The
native Planet Dove pixel grid is therefore preserved.

Example
-------
python prepare_superdove_v42_input.py input_Rrs_8b.tif output_Rrs_6b.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rasterio


VISIBLE_WAVELENGTHS_NM = (444, 492, 533, 566, 612, 666)


def prepare(input_path: Path, output_path: Path) -> None:
    """Copy visible bands 1-6 from an eight-band Rrs raster."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")

    with rasterio.open(input_path) as source:
        if source.count != 8:
            raise ValueError(f"Expected 8 input bands, found {source.count}: {input_path}")

        profile = source.profile.copy()
        profile.update(count=6, compress="deflate", predictor=3, tiled=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as target:
            # Windowed processing keeps memory use bounded for large Planet scenes.
            for _, window in source.block_windows(1):
                target.write(source.read(indexes=(1, 2, 3, 4, 5, 6), window=window), window=window)

            for band, wavelength in enumerate(VISIBLE_WAVELENGTHS_NM, start=1):
                target.set_band_description(band, f"Rrs_{wavelength}_nm_sr-1")
            target.update_tags(
                preparation="Bands 1-6 selected from the eight-band Planet Dove Rrs product",
                wavelengths_nm=",".join(map(str, VISIBLE_WAVELENGTHS_NM)),
                resampling="none",
            )

    # Re-open the result to catch incomplete writes immediately.
    with rasterio.open(output_path) as check:
        if check.count != 6:
            raise RuntimeError(f"Output validation failed: {output_path}")
        print(
            f"Created {output_path} | bands={check.count} | "
            f"grid={check.width}x{check.height} | resolution={check.res}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Eight-band Planet Dove above-water Rrs GeoTIFF")
    parser.add_argument("output", type=Path, help="Six-band v42 input GeoTIFF")
    args = parser.parse_args()
    prepare(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
