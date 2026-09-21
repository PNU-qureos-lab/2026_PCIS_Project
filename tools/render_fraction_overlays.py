"""Render the satellite-derived fraction maps with distinct out-of-range colors."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from PIL import Image


BELOW_ZERO = [222, 40, 170, 245]  # magenta
ABOVE_ONE = [23, 105, 255, 245]   # vivid blue, distinct from sand orange


def render(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for scene_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        for kind, colormap in (("sand_fraction", "YlOrBr"), ("seagrass_fraction", "YlGn")):
            tif = scene_dir / f"{scene_dir.name}_{kind}.tif"
            if not tif.is_file():
                continue
            with rasterio.open(tif) as dataset:
                data = dataset.read(1)
            scaled = np.clip(data, 0, 1)
            rgba = (matplotlib.colormaps[colormap](np.nan_to_num(scaled)) * 255).astype("uint8")
            rgba[..., 3] = np.where(np.isfinite(data), 235, 0).astype("uint8")
            rgba[np.isfinite(data) & (data < 0)] = BELOW_ZERO
            rgba[np.isfinite(data) & (data > 1)] = ABOVE_ONE
            image = Image.fromarray(rgba, "RGBA")
            image = image.resize((image.width * 8, image.height * 8), Image.Resampling.NEAREST)
            image.save(destination / f"{scene_dir.name}_{kind}.png", optimize=True)
            count += 1
    print(f"Rendered {count} fraction overlays")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Directory containing scene GeoTIFF outputs")
    parser.add_argument("destination", type=Path, help="Directory for web-map PNG overlays")
    args = parser.parse_args()
    render(args.source, args.destination)
