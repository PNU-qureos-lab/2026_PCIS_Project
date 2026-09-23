"""Render Planet four-band GeoTIFF products as transparent web-map overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from osgeo import gdal, osr
from PIL import Image


PRODUCTS = {
    "depth_m": ("2025-11-17_Planet_4band_depth_m.tif", "turbo", 0.0, 5.0),
    "sand_fraction": ("2025-11-17_Planet_4band_sand_fraction.tif", "YlGnBu", 0.0, 1.0),
    "seagrass_fraction": ("2025-11-17_Planet_4band_seagrass_fraction.tif", "YlGn", 0.0, 1.0),
    "rrs_rmse": ("2025-11-17_Planet_4band_Rrs_RMSE.tif", "magma", 0.0, None),
    "relative_cost": ("2025-11-17_Planet_4band_relative_cost.tif", "viridis", 0.0, None),
}


def bounds_wgs84(dataset) -> list[float]:
    transform = dataset.GetGeoTransform()
    x0, y0 = transform[0], transform[3]
    x1 = x0 + dataset.RasterXSize * transform[1]
    y1 = y0 + dataset.RasterYSize * transform[5]
    source = osr.SpatialReference()
    source.ImportFromWkt(dataset.GetProjection())
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    convert = osr.CoordinateTransformation(source, target)
    corners = [convert.TransformPoint(x, y) for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def read(path: Path) -> tuple[np.ndarray, object]:
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise RuntimeError(f"Could not open {path}")
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    invalid = ~np.isfinite(data)
    if nodata is not None:
        invalid |= data == nodata
    data[invalid] = np.nan
    return data, dataset


def save_rgba(rgba: np.ndarray, path: Path, categorical: bool = False) -> None:
    image = Image.fromarray(rgba, "RGBA")
    resampling = Image.Resampling.NEAREST if categorical else Image.Resampling.BILINEAR
    image.resize((image.width * 4, image.height * 4), resampling).save(path, optimize=True)


def main(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    layers: dict[str, dict] = {}
    extent = None
    for key, (filename, cmap_name, vmin, configured_vmax) in PRODUCTS.items():
        data, dataset = read(source / filename)
        if extent is None:
            extent = bounds_wgs84(dataset)
        finite = np.isfinite(data)
        vmax = configured_vmax if configured_vmax is not None else float(np.nanpercentile(data, 95))
        scaled = np.clip((data - vmin) / max(vmax - vmin, np.finfo(float).eps), 0, 1)
        rgba = (matplotlib.colormaps[cmap_name](np.nan_to_num(scaled)) * 255).astype("uint8")
        rgba[..., 3] = np.where(finite, 235, 0).astype("uint8")
        if key == "depth_m":
            rgba[finite & (data > vmax)] = [0, 0, 0, 240]
        output_name = f"2025-11-17_{key}.png"
        save_rgba(rgba, destination / output_name)
        layers[key] = {
            "image": f"planet_previews/{output_name}",
            "geotiff": f"semi_analytical/planet_dove_four_band/03_FINAL_PRODUCTS/{filename}",
            "vmin": vmin,
            "vmax": vmax,
        }
        dataset = None

    habitat_name = "2025-11-17_Planet_4band_habitat_exploratory.tif"
    habitat, dataset = read(source / habitat_name)
    rgba = np.zeros((*habitat.shape, 4), dtype="uint8")
    colors = {1: [233, 199, 95, 245], 2: [25, 135, 84, 245], 3: [136, 70, 183, 245], 4: [140, 148, 153, 245]}
    for value, color in colors.items():
        rgba[habitat == value] = color
    output_name = "2025-11-17_habitat.png"
    save_rgba(rgba, destination / output_name, categorical=True)
    layers["habitat"] = {
        "image": f"planet_previews/{output_name}",
        "geotiff": f"semi_analytical/planet_dove_four_band/03_FINAL_PRODUCTS/{habitat_name}",
    }
    dataset = None

    manifest = {
        "id": "2025-11-17_Planet_SuperDove",
        "label": "2025-11-17 · Planet SuperDove",
        "extent": extent,
        "layers": layers,
        "decision": "EXPLORATORY ONLY",
        "note": "Planet deep-water and atmospheric reference remain unvalidated.",
    }
    (destination / "manifest.js").write_text(
        "window.PLANET_RESULT = " + json.dumps(manifest, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    main(arguments.source, arguments.destination)
