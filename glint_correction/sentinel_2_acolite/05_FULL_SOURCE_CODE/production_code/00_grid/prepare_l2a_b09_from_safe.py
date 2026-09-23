#!/usr/bin/env python3
"""Extract L2A B09 BOA reflectance and align it to the workflow's 20 m grid.

B09 is supplied by Sentinel-2 L2A at 60 m. It is retained for input
completeness and atmospheric/water-vapour reference, but it is not used as a
water-colour correction target because the band lies in strong water-vapour
absorption near 945 nm.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def local_name(tag: str) -> str:
    """Remove an XML namespace from an element name."""
    return tag.split("}")[-1]


def reflectance_scale(safe: Path) -> tuple[float, float]:
    """Return the L2A BOA quantification value and B09 additive offset."""
    tree = ET.parse(safe / "MTD_MSIL2A.xml")
    quantification = 10000.0
    offset = 0.0
    for element in tree.iter():
        name = local_name(element.tag)
        if name == "BOA_QUANTIFICATION_VALUE" and element.text:
            quantification = float(element.text)
        elif name == "BOA_ADD_OFFSET" and element.attrib.get("band_id") == "9":
            offset = float(element.text)
    return quantification, offset


def find_b09(safe: Path) -> Path:
    """Locate the single native-resolution B09 image in an L2A SAFE."""
    matches = list(safe.glob("GRANULE/*/IMG_DATA/R60m/*_B09_60m.jp2"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one B09 60 m image; found {matches}")
    return matches[0]


def prepare(safe: Path, reference: Path, output: Path) -> None:
    """Convert B09 DN to BOA reflectance and align it to the reference grid."""
    quantification, offset = reflectance_scale(safe)
    b09_path = find_b09(safe)
    with rasterio.open(reference) as ref:
        profile = ref.profile.copy()
        destination = np.full((ref.height, ref.width), np.nan, dtype=np.float32)
        destination_transform = ref.transform
        destination_crs = ref.crs

    with rasterio.open(b09_path) as source:
        dn = source.read(1).astype(np.float32)
        # A zero DN is L2A NoData. Valid BOA reflectance follows the Sentinel-2
        # baseline formula: (DN + BOA_ADD_OFFSET) / quantification value.
        boa = np.where(dn != 0, (dn + offset) / quantification, np.nan).astype(np.float32)
        reproject(
            source=boa,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=np.nan,
            dst_transform=destination_transform,
            dst_crs=destination_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan,
                   compress="deflate", predictor=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as target:
        target.write(destination, 1)
        target.set_band_description(1, "Sentinel-2 L2A B09 BOA reflectance aligned to 20 m")
        target.update_tags(
            source=str(b09_path),
            formula=f"BOA=(DN+({offset}))/({quantification})",
            native_resolution="60 m",
            output_grid="20 m alignment; no new 20 m spatial information",
            workflow_role="Input completeness/reference; excluded from glint target bands",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("safe", type=Path, help="Sentinel-2 L2A SAFE directory")
    parser.add_argument("reference", type=Path, help="Workflow 20 m reference GeoTIFF")
    parser.add_argument("output", type=Path, help="Output B09 BOA GeoTIFF")
    args = parser.parse_args()
    prepare(args.safe, args.reference, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
