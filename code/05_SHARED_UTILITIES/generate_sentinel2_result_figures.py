"""Generate presentation maps from delivered Sentinel-2 v42 GeoTIFF results.

The scientific GeoTIFFs remain the authoritative outputs. This utility creates
downsampled PNG figures with titles, units, legends and source metadata for the
offline HTML result viewer. It never modifies the source rasters.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import rasterio
from rasterio.enums import Resampling


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PACKAGE_ROOT / "05_RESULTS" / "02_SEMI_ANALYTICAL" / "01_SENTINEL2_20M"
OUTPUT_ROOT = PACKAGE_ROOT / "05_RESULTS" / "03_MAPS" / "01_SENTINEL2_20M"
MAX_DISPLAY_PIXELS = 1800


SCENES = [
    {
        "country": "01_INDONESIA",
        "site": "01_TIDUNG",
        "scene": "20260808_S2A",
        "tile": "20260808_S2A",
        "label": "Indonesia · Tidung · 08 August 2026",
    },
    {
        "country": "02_FIJI",
        "site": "01_PROJECT_AOI",
        "scene": "20240520_S2A",
        "tile": "20240520_S2A_60KWF",
        "label": "Fiji · Project AOI · 20 May 2024 · tile 60KWF",
    },
    {
        "country": "03_TONGA",
        "site": "01_TONGATAPU",
        "scene": "20200522_S2A",
        "tile": "20200522_S2A_01KGS",
        "label": "Tonga · Tongatapu · 22 May 2020 · tile 01KGS",
    },
    {
        "country": "04_TUVALU",
        "site": "01_FUNAFUTI_ATOLL",
        "scene": "20220506_S2B",
        "tile": "20220506_S2B_60LYR",
        "label": "Tuvalu · Funafuti Atoll · 06 May 2022 · tile 60LYR",
    },
]


PRODUCTS = [
    {
        "key": "depth",
        "folder": "06_SCENARIO_COMPARISON_AND_FINAL_MAPS",
        "file": "bathymetry_m.tif",
        "title": "Estimated bathymetry",
        "cmap": "turbo",
        "limits": (0.0, 15.0),
        "units": "Depth (m)",
    },
    {
        "key": "sand_fraction",
        "folder": "05_SCENARIO_2_SAND_SEAGRASS_RESULTS",
        "file": "sand_fraction.tif",
        "title": "Estimated sand fraction",
        "cmap": "YlOrBr",
        "limits": (0.0, 1.0),
        "units": "Fraction (0–1)",
    },
    {
        "key": "seagrass_fraction",
        "folder": "05_SCENARIO_2_SAND_SEAGRASS_RESULTS",
        "file": "seagrass_fraction.tif",
        "title": "Estimated seagrass fraction",
        "cmap": "YlGn",
        "limits": (0.0, 1.0),
        "units": "Fraction (0–1)",
    },
    {
        "key": "benthic_class",
        "folder": "05_SCENARIO_2_SAND_SEAGRASS_RESULTS",
        "file": "dominant_benthic_class.tif",
        "title": "Derived benthic class",
        "categorical": True,
    },
]


def source_path(scene: dict[str, str], product: dict[str, object]) -> Path:
    return (
        SOURCE_ROOT
        / scene["country"]
        / scene["site"]
        / scene["scene"]
        / str(product["folder"])
        / scene["tile"]
        / str(product["file"])
    )


def read_display_array(path: Path) -> tuple[np.ma.MaskedArray, str]:
    """Read one band, reducing only the display copy when a raster is large."""
    with rasterio.open(path) as dataset:
        scale = max(dataset.width / MAX_DISPLAY_PIXELS, dataset.height / MAX_DISPLAY_PIXELS, 1.0)
        width = max(1, round(dataset.width / scale))
        height = max(1, round(dataset.height / scale))
        data = dataset.read(
            1,
            out_shape=(height, width),
            masked=True,
            resampling=Resampling.nearest,
        ).astype("float32")
        crs = dataset.crs.to_string() if dataset.crs else "CRS not recorded"
    data = np.ma.masked_invalid(data)
    return data, crs


def render(scene: dict[str, str], product: dict[str, object]) -> Path:
    src = source_path(scene, product)
    if not src.exists():
        raise FileNotFoundError(src)

    data, crs = read_display_array(src)
    out_dir = OUTPUT_ROOT / scene["country"] / scene["site"] / scene["scene"] / scene["tile"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{scene['tile']}_{product['key']}_map.png"

    fig, ax = plt.subplots(figsize=(12.8, 8.2), dpi=150, facecolor="#f4f8fa")
    ax.set_facecolor("#dfe9ed")

    if product.get("categorical"):
        cmap = ListedColormap(["#edc15b", "#16894b", "#893faf"])
        cmap.set_bad((0.88, 0.92, 0.94, 1.0))
        norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
        ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
        ax.legend(
            handles=[
                Patch(facecolor="#edc15b", label="Sand-dominant"),
                Patch(facecolor="#16894b", label="Seagrass-dominant"),
                Patch(facecolor="#893faf", label="Mixed"),
            ],
            title="Benthic class",
            loc="lower right",
            frameon=True,
        )
    else:
        cmap = plt.get_cmap(str(product["cmap"])).copy()
        cmap.set_bad((0.88, 0.92, 0.94, 1.0))
        vmin, vmax = product["limits"]
        image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.036, pad=0.025)
        colorbar.set_label(str(product["units"]), fontsize=10)

    ax.set_title(f"Sentinel-2 20 m · {product['title']}\n{scene['label']}", fontsize=15, weight="bold", pad=14)
    ax.set_xlabel(f"Scientific source: {src.name} · {crs}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#7f98a5")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main() -> None:
    created: list[Path] = []
    for scene in SCENES:
        for product in PRODUCTS:
            created.append(render(scene, product))
    print(f"Created {len(created)} Sentinel-2 result figures in {OUTPUT_ROOT}")
    for path in created:
        print(path.relative_to(PACKAGE_ROOT))


if __name__ == "__main__":
    main()
