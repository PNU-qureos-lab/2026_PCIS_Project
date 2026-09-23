"""Build compact browser-readable pixel grids for the combined result viewer."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parents[1]
SENTINEL_SOURCE = ROOT.parent / "tiding_sentinel_v3" / "results_v3"
PRODUCTS = ("depth_m", "sand_fraction", "seagrass_fraction", "habitat")


def raster_payload(path: Path) -> dict:
    with rasterio.open(path) as src:
        values = src.read(1).astype("<f4", copy=False)
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    return {
        "width": int(values.shape[1]), "height": int(values.shape[0]),
        "extent": list(bounds),
        "data": base64.b64encode(values.tobytes()).decode("ascii"),
    }


def write_js(path: Path, variable: str, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"window.{variable} = {json.dumps(value, separators=(',', ':'))};\n", encoding="utf-8")


def main() -> None:
    sentinel = {}
    for folder in sorted(SENTINEL_SOURCE.iterdir()):
        if not folder.is_dir():
            continue
        layers = {}
        for product in PRODUCTS:
            source = folder / f"{folder.name}_{product}.tif"
            if source.exists():
                layers[product] = raster_payload(source)
        if layers:
            sentinel[folder.name] = layers

    planet_root = ROOT / "semi_analytical" / "planet_dove" / "03_FINAL_PRODUCTS"
    planet_files = {
        "depth_m": "2025-11-17_Planet_4band_depth_m.tif",
        "sand_fraction": "2025-11-17_Planet_4band_sand_fraction.tif",
        "seagrass_fraction": "2025-11-17_Planet_4band_seagrass_fraction.tif",
        "habitat": "2025-11-17_Planet_4band_habitat_exploratory.tif",
        "rrs_rmse": "2025-11-17_Planet_4band_Rrs_RMSE.tif",
        "relative_cost": "2025-11-17_Planet_4band_relative_cost.tif",
    }
    planet = {"2025-11-17_Planet_SuperDove": {
        product: raster_payload(planet_root / filename) for product, filename in planet_files.items()
    }}
    write_js(ROOT / "viewer_data" / "sentinel_values.js", "SENTINEL_VALUES", sentinel)
    write_js(ROOT / "viewer_data" / "planet_values.js", "PLANET_VALUES", planet)
    print(f"Wrote {len(sentinel)} Sentinel scenes and {len(planet)} Planet scene.")


if __name__ == "__main__":
    main()
