"""Create clean single-map PNG layers for the Results viewer.

Titles, dates, resolutions and legends are rendered by HTML. The PNGs contain
only the mapped raster, matching the one-scene/one-product viewer design.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "06_DATA_INVENTORY_AND_VIEWER" / "assets" / "result_maps"
MAX_PIXELS = 1800


SCENES = [
    {
        "sensor": "Sentinel-2", "site": "Tidung", "scene": "20260808_S2A", "resolution": "20 m",
        "base": "05_RESULTS/02_SEMI_ANALYTICAL/01_SENTINEL2_20M/01_INDONESIA/01_TIDUNG/20260808_S2A",
        "tile": "20260808_S2A",
    },
    {
        "sensor": "Sentinel-2", "site": "Fiji Project AOI", "scene": "20240520_S2A · 60KWF", "resolution": "20 m representative tile",
        "base": "05_RESULTS/02_SEMI_ANALYTICAL/01_SENTINEL2_20M/02_FIJI/01_PROJECT_AOI/20240520_S2A",
        "tile": "20240520_S2A_60KWF",
    },
    {
        "sensor": "Sentinel-2", "site": "Tongatapu", "scene": "20200522_S2A · 01KGS", "resolution": "20 m representative tile",
        "base": "05_RESULTS/02_SEMI_ANALYTICAL/01_SENTINEL2_20M/03_TONGA/01_TONGATAPU/20200522_S2A",
        "tile": "20200522_S2A_01KGS",
    },
    {
        "sensor": "Sentinel-2", "site": "Funafuti Atoll", "scene": "20220506_S2B · 60LYR", "resolution": "20 m",
        "base": "05_RESULTS/02_SEMI_ANALYTICAL/01_SENTINEL2_20M/04_TUVALU/01_FUNAFUTI_ATOLL/20220506_S2B",
        "tile": "20220506_S2B_60LYR",
    },
    {
        "sensor": "Planet Dove", "site": "Tidung", "scene": "20251117_8420286", "resolution": "native 3 m",
        "planet_base": "05_RESULTS/02_SEMI_ANALYTICAL/02_PLANET_DOVE/01_INDONESIA/01_TIDUNG/20251117_8420286/05_SCENARIO_1_SAND_SEAGRASS/01_6_BAND_MAIN",
        "prefix": "20251117_8420286_ss_6band",
    },
    {
        "sensor": "Planet Dove", "site": "Tongatapu", "scene": "20260422_8740134", "resolution": "existing 21 m result grid",
        "planet_base": "05_RESULTS/02_SEMI_ANALYTICAL/02_PLANET_DOVE/03_TONGA/01_TONGATAPU/20260422_8740134/02_EXISTING_SEMI_ANALYTICAL_21M/03_4BAND_SAND_SEAGRASS",
        "prefix": "20260422_8740134_4band",
    },
    {
        "sensor": "Planet Dove", "site": "Funafuti Atoll", "scene": "20240616_7377012", "resolution": "existing 21 m result grid",
        "planet_base": "05_RESULTS/02_SEMI_ANALYTICAL/02_PLANET_DOVE/04_TUVALU/01_FUNAFUTI_ATOLL/20240616_7377012/02_EXISTING_SEMI_ANALYTICAL_21M/03_4BAND_SAND_SEAGRASS",
        "prefix": "20240616_7377012_4band",
    },
    {
        "sensor": "Planet Dove", "site": "Savave", "scene": "20240608_7359850", "resolution": "existing 21 m result grid",
        "planet_base": "05_RESULTS/02_SEMI_ANALYTICAL/02_PLANET_DOVE/04_TUVALU/02_SAVAVE/20240608_7359850/02_EXISTING_SEMI_ANALYTICAL_21M/03_4BAND_SAND_SEAGRASS",
        "prefix": "20240608_7359850_4band",
    },
]


PRODUCTS = [
    ("depth", "Estimated bathymetry", "depth", "depth_m.tif"),
    ("sand", "Sand fraction", "sand", "sand_fraction.tif"),
    ("seagrass", "Seagrass fraction", "seagrass", "seagrass_fraction.tif"),
]


def source_path(scene: dict[str, str], suffix: str) -> Path:
    if scene["sensor"] == "Sentinel-2":
        folder = "06_SCENARIO_COMPARISON_AND_FINAL_MAPS" if suffix == "depth_m.tif" else "05_SCENARIO_2_SAND_SEAGRASS_RESULTS"
        filename = "bathymetry_m.tif" if suffix == "depth_m.tif" else suffix
        return ROOT / scene["base"] / folder / scene["tile"] / filename
    return ROOT / scene["planet_base"] / f"{scene['prefix']}_{suffix}"


def read(path: Path) -> np.ma.MaskedArray:
    with rasterio.open(path) as dataset:
        scale = max(dataset.width / MAX_PIXELS, dataset.height / MAX_PIXELS, 1.0)
        shape = (max(1, round(dataset.height / scale)), max(1, round(dataset.width / scale)))
        data = dataset.read(1, out_shape=shape, masked=True, resampling=Resampling.nearest).astype("float32")
    return np.ma.masked_invalid(data)


def continuous_rgba(data: np.ma.MaskedArray, cmap_name: str, low: float, high: float) -> np.ndarray:
    values = np.clip((data.filled(low) - low) / (high - low), 0, 1)
    rgba = (colormaps[cmap_name](values) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.ma.getmaskarray(data), 0, 255).astype(np.uint8)
    return rgba


def class_rgba(data: np.ma.MaskedArray, status: bool) -> np.ndarray:
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    values = data.filled(-999).astype(int)
    colors = ({0: (130, 145, 153, 230), 1: (32, 151, 83, 255), 2: (240, 164, 58, 255)} if status else
              {1: (237, 193, 91, 255), 2: (22, 137, 75, 255), 3: (137, 63, 175, 255)})
    for value, color in colors.items():
        rgba[values == value] = color
    rgba[np.ma.getmaskarray(data)] = (0, 0, 0, 0)
    return rgba


def save_layer(path: Path, output: Path, kind: str) -> None:
    data = read(path)
    if kind == "depth":
        rgba = continuous_rgba(data, "turbo", 0, 15)
    elif kind == "sand":
        rgba = continuous_rgba(data, "YlOrBr", 0, 1)
    elif kind == "seagrass":
        rgba = continuous_rgba(data, "YlGn", 0, 1)
    else:
        rgba = class_rgba(data, status=(kind == "status"))
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(output)


def main() -> None:
    layers: list[dict[str, str]] = []
    for scene in SCENES:
        slug = f"{scene['sensor'].lower().replace(' ', '_')}_{scene['scene'].split(' ')[0]}"
        for key, label, legend, suffix in PRODUCTS:
            source = source_path(scene, suffix)
            output = OUTPUT / f"{slug}_{key}.png"
            save_layer(source, output, legend)
            layers.append({**{k: scene[k] for k in ("sensor", "site", "scene", "resolution")},
                           "product": label, "legend": legend, "image": "assets/result_maps/" + output.name})

        class_suffix = "dominant_benthic_class.tif" if scene["sensor"] == "Sentinel-2" else "status.tif"
        class_path = source_path(scene, class_suffix)
        class_kind = "benthic" if scene["sensor"] == "Sentinel-2" else "status"
        class_label = "Derived benthic class" if scene["sensor"] == "Sentinel-2" else "Model status"
        class_output = OUTPUT / f"{slug}_{class_kind}.png"
        save_layer(class_path, class_output, class_kind)
        layers.append({**{k: scene[k] for k in ("sensor", "site", "scene", "resolution")},
                       "product": class_label, "legend": class_kind, "image": "assets/result_maps/" + class_output.name})

    destination = ROOT / "06_DATA_INVENTORY_AND_VIEWER" / "assets" / "result_layers.js"
    destination.write_text("window.PCIS_RESULT_LAYERS=" + json.dumps(layers, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"Created {len(layers)} clean result layers and {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
