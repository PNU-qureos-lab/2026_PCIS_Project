"""Render the experimental Sentinel pilot rasters with legends."""
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import rasterio
from rasterio.plot import plotting_extent

SCENE = os.environ.get("SEMI_ANALYTICAL_SCENE", "20260808_S2A")
OUT = Path(os.environ.get("SEMI_ANALYTICAL_OUTPUT", Path(__file__).resolve().parents[1] / "results_step_by_step" / SCENE))

def load(name):
    path = OUT / f"{SCENE}_{name}.tif"
    with rasterio.open(path) as ds:
        return ds.read(1), plotting_extent(ds), ds.crs

depth, extent, crs = load("depth_m")
sand, _, _ = load("sand_fraction")
grass, _, _ = load("seagrass_fraction")
habitat, _, _ = load("habitat")
status, _, _ = load("status")

depth = np.ma.masked_invalid(depth)
sand_display = np.ma.masked_where(~np.isfinite(sand) | (sand < 0) | (sand > 1), sand)
grass_display = np.ma.masked_where(~np.isfinite(grass) | (grass < 0) | (grass > 1), grass)

fig, axes = plt.subplots(2, 2, figsize=(20, 12), layout="constrained")
fig.suptitle(
    f"Tidung KML Sentinel-2 Sand–Seagrass — {SCENE}\n"
    "Four-band semi-analytical HOPE/D2 result",
    fontsize=16, fontweight="bold"
)

depth_cmap = plt.get_cmap("turbo").copy()
depth_cmap.set_over("#111111")
im = axes[0, 0].imshow(depth, extent=extent, origin="upper", cmap=depth_cmap, vmin=0, vmax=5)
axes[0, 0].set_title("Estimated water depth")
cb = fig.colorbar(im, ax=axes[0, 0], shrink=0.88,
                  ticks=[0, 0.5, 1, 2, 3, 4, 5])
cb.set_label("Depth (m); black = >5 m")

fraction_cmap = plt.get_cmap("YlGnBu").copy()
fraction_cmap.set_bad("white")
im = axes[0, 1].imshow(sand_display, extent=extent, origin="upper", cmap=fraction_cmap, vmin=0, vmax=1)
uncertain_overlay = np.ma.masked_where(~((status == 1) & ((sand < 0) | (sand > 1))), np.ones_like(sand))
axes[0, 1].imshow(uncertain_overlay, extent=extent, origin="upper", cmap=ListedColormap(["#bdbdbd"]), vmin=0, vmax=1)
axes[0, 1].set_title("Sand fraction")
cb = fig.colorbar(im, ax=axes[0, 1], shrink=0.88)
cb.set_label("Fraction (0–1); grey = unphysical")

grass_cmap = plt.get_cmap("YlGn").copy()
grass_cmap.set_bad("white")
im = axes[1, 0].imshow(grass_display, extent=extent, origin="upper", cmap=grass_cmap, vmin=0, vmax=1)
axes[1, 0].imshow(uncertain_overlay, extent=extent, origin="upper", cmap=ListedColormap(["#bdbdbd"]), vmin=0, vmax=1)
axes[1, 0].set_title("Seagrass fraction")
cb = fig.colorbar(im, ax=axes[1, 0], shrink=0.88)
cb.set_label("Fraction (0–1); grey = unphysical")

colors = ["#ffffff", "#e8c66a", "#168a45", "#8e44ad", "#9e9e9e"]
habitat_cmap = ListedColormap(colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], habitat_cmap.N)
habitat_display = habitat.copy()
habitat_display[status == 3] = 0
axes[1, 1].imshow(habitat_display, extent=extent, origin="upper", cmap=habitat_cmap, norm=norm)
axes[1, 1].set_title("Habitat classification")
axes[1, 1].legend(handles=[
    Patch(facecolor=colors[1], label="Sand dominant (≥70%)"),
    Patch(facecolor=colors[2], label="Seagrass dominant (≥70%)"),
    Patch(facecolor=colors[3], label="Mixed sand–seagrass"),
    Patch(facecolor=colors[4], label="Uncertain / unphysical fraction"),
], loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True,
   borderaxespad=0.0)

for ax in axes.flat:
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.grid(color="white", alpha=0.18, linewidth=0.5)

fig.text(
    0.5, 0.006,
    "CRS: EPSG:32748 | 20 m pixels | HOPE deep-water relative RMS mismatch: 32.4% | "
    "Semi-analytical HOPE/D2 model output.",
    ha="center", fontsize=9, color="#444444"
)
png = OUT / f"{SCENE}_pilot_maps_with_legends.png"
pdf = OUT / f"{SCENE}_pilot_maps_with_legends.pdf"
fig.savefig(png, dpi=220, facecolor="white")
fig.savefig(pdf, facecolor="white")
print(png)
print(pdf)
