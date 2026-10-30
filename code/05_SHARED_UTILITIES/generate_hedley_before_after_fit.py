"""Generate Planet Dove Hedley regression diagnostics before and after correction."""

from __future__ import annotations

from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.io import MemoryFile


ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = ROOT / "04_DATA/01_RAW/02_PLANET_DOVE/01_INDONESIA/01_TIDUNG/20251117_8420286/Planet_Dove_Indonesia_Tidung_20251117_SR_UDM2.zip"
CORRECTED = ROOT / "05_RESULTS/01_INTERMEDIATE/02_PLANET_DOVE_3M/01_INDONESIA/01_TIDUNG/20251117_8420286/01_GLINT_CORRECTION/05_FINAL_ACCEPTED_PRODUCTS_SR_Rrs_rrs/20251117_8420286_Hedley_corrected_SR_8b.tif"
TRAINING = ROOT / "05_RESULTS/01_INTERMEDIATE/02_PLANET_DOVE_3M/01_INDONESIA/01_TIDUNG/20251117_8420286/01_GLINT_CORRECTION/03_HEDLEY_TRAINING_WATER_SELECTION/20251117_8420286_hedley_training_water.tif"
OUTPUT = ROOT / "05_RESULTS/01_INTERMEDIATE/02_PLANET_DOVE_3M/01_INDONESIA/01_TIDUNG/20251117_8420286/01_GLINT_CORRECTION/06_SHALLOW_WATER_GLINT_SAFETY_QC/20251117_8420286_step2c_before_after_regression_fit.png"


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    denominator = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - np.sum((y - predicted) ** 2) / denominator if denominator > 0 else np.nan
    return float(slope), float(intercept), float(r2)


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as archive:
        member = next(name for name in archive.namelist() if "20251117" in name and "AnalyticMS_SR_8b_clip.tif" in name)
        with MemoryFile(archive.read(member)) as memory, memory.open() as dataset:
            before = dataset.read().astype("float32")
    with rasterio.open(CORRECTED) as dataset:
        after = dataset.read().astype("float32")
    with rasterio.open(TRAINING) as dataset:
        training = dataset.read(1) > 0

    valid = training & np.all(np.isfinite(before), axis=0) & np.all(np.isfinite(after), axis=0)
    indices = np.flatnonzero(valid)
    if indices.size > 15000:
        indices = np.random.default_rng(42).choice(indices, 15000, replace=False)

    nir_before = before[7].ravel()[indices]
    nir_after = after[7].ravel()[indices]
    bands = [(1, "Blue"), (3, "Green"), (5, "Red")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=150)
    for column, (band_index, label) in enumerate(bands):
        for row, (cube, nir, stage) in enumerate(((before, nir_before, "Before correction"), (after, nir_after, "After correction"))):
            values = cube[band_index].ravel()[indices]
            slope, intercept, r2 = fit(nir, values)
            ax = axes[row, column]
            ax.hexbin(nir, values, gridsize=55, mincnt=1, cmap="viridis")
            xline = np.linspace(np.nanmin(nir), np.nanmax(nir), 100)
            ax.plot(xline, slope * xline + intercept, color="#e4552d", linewidth=2)
            ax.set_title(f"{stage} · {label}\nslope={slope:.4f}, R²={r2:.3f}")
            ax.set_xlabel("866 nm surface reflectance")
            ax.set_ylabel(f"{label} surface reflectance")
            ax.grid(alpha=0.2)
    fig.suptitle("Planet Dove Hedley fit check: before and after correction\nTidung · 17 November 2025 · accepted training-water pixels", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
