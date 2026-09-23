"""Shared PlanetScope quality and stable-land masking for the Tidung workflow.

The former per-date NDWI/NIR threshold removed the brightest glinted-water
pixels. This module instead identifies land from its stable multi-date NIR and
NDVI signature, then combines that fixed land mask with each date's UDM2 clear
mask. Bright water can therefore remain available for glint correction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from osgeo import gdal


WAVES = [444, 492, 533, 566, 612, 666, 707, 866]
LAND_NIR_Q25 = 0.08
LAND_MEDIAN_NDVI = 0.02


def _read_band(path: Path, band_number: int) -> np.ndarray:
    dataset = gdal.Open(str(path))
    raw = dataset.GetRasterBand(band_number).ReadAsArray().astype(np.float32)
    dataset = None
    return np.where(raw == 0, np.nan, raw / 10000.0)


def build_stable_land_mask(planet_dir: Path) -> np.ndarray:
    """Return one co-registered land mask shared by all selected Planet dates."""
    paths = sorted(planet_dir.glob("*_AnalyticMS_SR_8b_clip.tif"))
    if not paths:
        raise RuntimeError(f"No Planet SR scenes found in {planet_dir}")
    nir = np.stack([_read_band(path, 8) for path in paths])
    red = np.stack([_read_band(path, 6) for path in paths])
    nir_q25 = np.nanpercentile(nir, 25, axis=0)
    median_ndvi = np.nanmedian((nir - red) / np.maximum(nir + red, 1e-6), axis=0)
    return (nir_q25 > LAND_NIR_Q25) & (median_ndvi > LAND_MEDIAN_NDVI)


def strict_udm2_clear(udm: np.ndarray) -> np.ndarray:
    """Require clear and reject snow, shadow, haze, cloud, and unusable pixels."""
    return ((udm[0] == 1) & (udm[1] == 0) & (udm[2] == 0) &
            (udm[3] == 0) & (udm[4] == 0) & (udm[5] == 0) & (udm[7] == 0))


def corrected_water_mask(
    original: dict[int, np.ndarray], quality: np.ndarray, stable_land: np.ndarray
) -> np.ndarray:
    """Keep all clear finite non-land pixels, including spectrally bright glint."""
    finite = quality.copy()
    for wave in WAVES:
        finite &= np.isfinite(original[wave])
    return finite & ~stable_land


MASK_DESCRIPTION = (
    "UDM2 clear and finite in all eight bands; fixed land excluded using the "
    "four-date 25th-percentile 866 nm (>0.08) and median NDVI (>0.02). Bright "
    "glinted water is retained rather than rejected by a per-date NIR threshold."
)
