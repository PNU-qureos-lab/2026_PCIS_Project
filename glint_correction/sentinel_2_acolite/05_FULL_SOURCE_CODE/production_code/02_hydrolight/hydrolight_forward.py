#!/usr/bin/env python3
"""TSM retrieval followed by HydroLight forward spectral simulation for Tidung.

The primary retrieval is Nechad-2016 MSI B04.  HydroLight does not re-fit TSM;
it receives the retrieved TSM and predicts a spectrum.  This makes the forward
comparison diagnostically clearer than selecting whichever HydroLight TSM gives
the smallest three-band error.
"""

from __future__ import annotations

# WORKFLOW ROLE — STEP 02A: HYDROLIGHT FORWARD SIMULATION
# Input: water optical parameters, geometry, and Sentinel-2 spectral responses.
# Output: forward-model spectra later assembled into the validation library.
# The external licensed HydroLight executable itself is not redistributed.

import argparse
import base64
import csv
import html
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_xy

import hydrolight_inversion as common


ROOT = Path(__file__).resolve().parents[5]  # glint_correction folder
SOURCE = ROOT / "outputs" / "Tidung_Glint_Comparison_Rerun_20260903"
GEOMETRY_GRID = ROOT / "outputs" / "hydrolight_tidung_reference_grid_20260903"
DEFAULT_OUTPUT = ROOT / "outputs" / "Tidung_TSM_HydroLight_Forward_20260904"
TEMPLATE = ROOT / "outputs" / "hydrolight_pilot_20251101" / "inputs" / "ITIDHL_CLR.txt"
HE5 = Path(r"C:\HE5")
EXECUTABLE = HE5 / "Code" / "mainHL_stnd.exe"
RUN_DIR = HE5 / "run"

BANDS = ("B02", "B03", "B04")
ANCHOR_TSM = (0.10, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0)
NECHAD_A, NECHAD_C = 342.10, 0.19563
NECHAD_MAX_RHO = 0.5 * NECHAD_C
CHL_A = 0.10
CDOM_440 = 0.03
WIND = 5.0
SZA_BASE = 30.0
plt.rcParams.update({"font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15, "legend.fontsize": 11})


def full_input(root_name: str, tsm: float) -> str:
    lines = TEMPLATE.read_text().splitlines()
    lines[1] = f"Tidung retrieved-TSM forward spectrum; TSM {tsm:g} g m-3"
    lines[2] = root_name
    lines[6] = f"0, {CHL_A:.6f}, {CDOM_440:.6f}, {tsm:.6f}"
    sun = [x.strip() for x in lines[34].split(",")]
    sun[2] = f"{SZA_BASE:.6f}"
    lines[34] = ", ".join(sun)
    surface = [x.strip() for x in lines[36].split(",")]
    surface[0] = f"{WIND:.3f}"
    lines[36] = ", ".join(surface)
    return "\n".join(lines) + "\n"


def output_sources(root_name: str) -> dict[str, Path]:
    base = HE5 / "output" / "Hydrolight"
    return {
        "D": base / "digital" / f"D{root_name}.txt",
        "L": base / "digital" / f"L{root_name}.txt",
        "M": base / "excel" / f"M{root_name}.txt",
        "S": base / "excel" / f"S{root_name}.txt",
        "P": base / "printout" / f"P{root_name}.txt",
    }


def run_one(inp: Path, result: Path, root_name: str, resume: bool) -> str:
    expected = {k: result / p.name for k, p in output_sources(root_name).items()}
    if resume and all(p.exists() and p.stat().st_size > 0 for p in expected.values()):
        common.parse_rrs(expected["M"])
        return "SKIPPED_EXISTING"
    result.mkdir(parents=True, exist_ok=True)
    with inp.open("r") as handle:
        proc = subprocess.run([str(EXECUTABLE)], stdin=handle, cwd=RUN_DIR, text=True,
                              capture_output=True, check=False)
    (result / "process_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (result / "process_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"HydroLight failed for {root_name}: {proc.returncode}")
    for key, source in output_sources(root_name).items():
        if not source.exists() or source.stat().st_size == 0:
            raise RuntimeError(f"Missing HydroLight {key} output for {root_name}")
        shutil.copy2(source, expected[key])
    common.parse_rrs(expected["M"])
    return "COMPLETED"


def build_anchor_library(output: Path, resume: bool) -> pd.DataFrame:
    rows = []
    for index, tsm in enumerate(ANCHOR_TSM):
        root_name = f"Z{index:03d}"
        inp = output / "02_HydroLight_Forward_Library" / "inputs" / f"I{root_name}.txt"
        inp.parent.mkdir(parents=True, exist_ok=True)
        inp.write_text(full_input(root_name, tsm), encoding="utf-8")
        result = output / "02_HydroLight_Forward_Library" / "results" / f"TSM_{tsm:g}"
        status = run_one(inp, result, root_name, resume)
        wave, rrs = common.parse_rrs(result / f"M{root_name}.txt")
        rows.extend({"TSM_g_m3": tsm, "wavelength_nm": w, "Rrs_SZA30_sr-1": r,
                     "root_name": root_name, "status": status} for w, r in zip(wave, rrs))
        print(f"{status}: full spectrum TSM={tsm:g} g m-3", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(output / "02_HydroLight_Forward_Library" / "full_spectrum_anchor_library.csv", index=False)
    return df


def load_geometry_spectra() -> dict[float, tuple[np.ndarray, np.ndarray]]:
    manifest = pd.read_csv(GEOMETRY_GRID / "hydrolight_run_manifest.csv")
    out = {}
    subset = manifest[(manifest.case == "CLR") & (manifest.wind_m_s == 5.0)]
    for _, row in subset.iterrows():
        path = Path(row.result_dir) / f"M{row.root_name}.txt"
        out[float(row.sza_deg)] = common.parse_rrs(path)
    return out


class ForwardLibrary:
    def __init__(self, anchors: pd.DataFrame):
        self.tsm = np.asarray(ANCHOR_TSM, float)
        first = anchors[anchors.TSM_g_m3 == ANCHOR_TSM[0]]
        self.wave = first.wavelength_nm.to_numpy(float)
        self.base = np.vstack([
            anchors[anchors.TSM_g_m3 == t].sort_values("wavelength_nm")["Rrs_SZA30_sr-1"].to_numpy(float)
            for t in ANCHOR_TSM
        ])
        geometry = load_geometry_spectra()
        self.szas = np.asarray(sorted(geometry), float)
        self.geom = np.vstack([geometry[s][1] for s in self.szas])
        self.geom_base = self.geom[np.where(self.szas == SZA_BASE)[0][0]]

    def spectrum(self, tsm: float, sza: float) -> np.ndarray:
        t = float(np.clip(tsm, self.tsm.min(), self.tsm.max()))
        base = np.asarray([np.interp(np.log(t), np.log(self.tsm), self.base[:, i])
                           for i in range(self.base.shape[1])])
        s = float(np.clip(sza, self.szas.min(), self.szas.max()))
        geom_at_s = np.asarray([np.interp(s, self.szas, self.geom[:, i]) for i in range(self.geom.shape[1])])
        ratio = np.divide(geom_at_s, self.geom_base, out=np.ones_like(geom_at_s), where=np.abs(self.geom_base) > 1e-12)
        return base * ratio

    def band_table(self, platform: str, sza: float) -> tuple[np.ndarray, np.ndarray]:
        rsr = common.read_rsr(platform)
        values = []
        for tsm in self.tsm:
            spec = self.spectrum(tsm, sza)
            values.append([common.convolve(self.wave, spec, rsr[b.replace("0", "", 1)]) for b in BANDS])
        return self.tsm, np.asarray(values)


def write_raster(path: Path, array: np.ndarray, profile: dict, description: str, units: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", predictor=3)
    with rasterio.open(path, "w", **p) as ds:
        ds.write(array.astype("float32"), 1)
        ds.set_band_description(1, description)
        ds.update_tags(units=units, AREA_OR_POINT="Area")


def retrieve_tsm(red: np.ndarray) -> np.ndarray:
    out = np.full(red.shape, np.nan, np.float32)
    good = np.isfinite(red) & (red > 0) & (red < NECHAD_MAX_RHO)
    out[good] = NECHAD_A * red[good] / (1 - red[good] / NECHAD_C)
    return out


def roi_masks(ds: rasterio.DatasetReader, definitions: list[dict]) -> dict[str, np.ndarray]:
    out = {}
    for item in definitions:
        rid = item.get("roi_id", "")
        if not rid.startswith("C") or rid == "C":
            continue
        xs, ys = transform_xy("EPSG:4326", ds.crs, [item["center_lon"]], [item["center_lat"]])
        row, col = ds.index(xs[0], ys[0])
        mask = np.zeros(ds.shape, bool)
        mask[max(0, row - 3):min(ds.height, row + 4), max(0, col - 3):min(ds.width, col + 4)] = True
        out[rid] = mask
    out["C"] = np.logical_or.reduce(list(out.values()))
    return out


def process(output: Path, library: ForwardLibrary) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inv = pd.read_csv(SOURCE / "00_Scene_Selection" / "scene_inventory.csv")
    qc = pd.read_csv(SOURCE / "09_Final_Comparison" / "scene_physical_QC_flags.csv")
    scenes = qc.loc[qc.final_aggregate_included.astype(str).str.lower() == "true", "scene"].tolist()
    scene_rows, roi_rows, spectra_rows, excluded_rows = [], [], [], []
    for scene in scenes:
        date, platform = scene.split("_")
        meta = inv[(inv.date.astype(str) == date) & (inv.platform == platform) & (inv.decision == "ACCEPT")].iloc[0]
        sza = float(meta.sun_zenith_angle)
        paths = [SOURCE / "06_ACOLITE" / "Corrected" / scene / f"{b}_corrected.tif" for b in BANDS]
        corrected = []
        with rasterio.open(paths[0]) as ds:
            profile = ds.profile
            corrected.append(ds.read(1))
            definitions = json.loads((SOURCE / "03_ROI" / scene / "roi_definitions.json").read_text())
            masks = roi_masks(ds, definitions)
        for path in paths[1:]:
            with rasterio.open(path) as ds:
                corrected.append(ds.read(1))
        with rasterio.open(SOURCE / "01_Masks" / scene / "valid_water_mask.tif") as ds:
            water = ds.read(1) > 0
        with rasterio.open(SOURCE / "03_ROI" / scene / "ROI_A.tif") as ds:
            deep = ds.read(1) > 0
        observed = np.stack(corrected, axis=-1) / np.pi
        tsm = retrieve_tsm(corrected[2])
        tsm[~water] = np.nan
        tgrid, model_grid = library.band_table(platform, sza)
        model_maps = []
        valid_model = deep & np.isfinite(tsm)
        if valid_model.sum() < 100:
            excluded_rows.append({"scene": scene, "reason": "fewer than 100 valid positive B04 pixels in deep-water ROI-A",
                                  "ROI_A_valid_TSM_pixels": int(valid_model.sum())})
            print(f"EXCLUDED {scene}: only {valid_model.sum()} valid TSM pixels in ROI-A", flush=True)
            continue
        write_raster(output / "01_TSM_Nechad2016" / scene / "TSM_approx_Nechad2016_B04.tif", tsm, profile,
                     "TSM approximation from Glint-ACOLITE corrected MSI B04", "g m-3")
        for ib, band in enumerate(BANDS):
            arr = np.full(tsm.shape, np.nan, np.float32)
            arr[valid_model] = np.interp(np.log(np.clip(tsm[valid_model], tgrid.min(), tgrid.max())),
                                         np.log(tgrid), model_grid[:, ib])
            model_maps.append(arr)
            write_raster(output / "04_HydroLight_Forward_Closure" / scene / f"modeled_Rrs_{band}.tif", arr, profile,
                         f"HydroLight forward-modeled Rrs from Nechad TSM, {band}", "sr-1")
            write_raster(output / "04_HydroLight_Forward_Closure" / scene / f"residual_observed_minus_modeled_{band}.tif",
                         observed[..., ib] - arr, profile, f"Observed Rrs_approx minus HydroLight Rrs, {band}", "sr-1")
        model_stack = np.stack(model_maps, axis=-1)
        rmse = np.full(tsm.shape, np.nan, np.float32)
        rmse[valid_model] = np.sqrt(np.mean((observed[valid_model] - model_stack[valid_model]) ** 2, axis=-1))
        write_raster(output / "04_HydroLight_Forward_Closure" / scene / "forward_spectral_RMSE_B02_B04.tif", rmse, profile,
                     "Three-band forward spectral RMSE", "sr-1")

        for rid, mask in masks.items():
            good = mask & np.all(np.isfinite(observed), axis=-1) & np.isfinite(tsm)
            if good.sum() == 0:
                continue
            obs = np.nanmedian(observed[good], axis=0)
            rt = float(np.nanmedian(tsm[good]))
            spectrum = library.spectrum(rt, sza)
            rsr = common.read_rsr(platform)
            bands_model = np.asarray([common.convolve(library.wave, spectrum, rsr[b.replace("0", "", 1)]) for b in BANDS])
            residual = obs - bands_model
            den = np.linalg.norm(obs) * np.linalg.norm(bands_model)
            angle = float(np.degrees(np.arccos(np.clip(np.dot(obs, bands_model) / den, -1, 1)))) if den else np.nan
            row = {"scene": scene, "ROI": rid, "valid_pixels": int(good.sum()), "scene_SZA_deg": sza,
                   "retrieved_TSM_median_g_m3": rt, "TSM_clipped_for_HydroLight_g_m3": float(np.clip(rt, min(ANCHOR_TSM), max(ANCHOR_TSM))),
                   "fixed_Chl_a_mg_m3": CHL_A, "fixed_a_CDOM_440_m-1": CDOM_440,
                   "three_band_RMSE_sr-1": float(np.sqrt(np.mean(residual ** 2))), "spectral_angle_deg": angle}
            for i, band in enumerate(BANDS):
                row[f"observed_Rrs_approx_{band}_sr-1"] = float(obs[i])
                row[f"modeled_Rrs_{band}_sr-1"] = float(bands_model[i])
                row[f"residual_{band}_sr-1"] = float(residual[i])
            roi_rows.append(row)
            for wavelength, value in zip(library.wave, spectrum):
                spectra_rows.append({"scene": scene, "ROI": rid, "retrieved_TSM_g_m3": rt,
                                     "scene_SZA_deg": sza, "wavelength_nm": wavelength,
                                     "HydroLight_forward_Rrs_sr-1": value})

        values = tsm[deep & np.isfinite(tsm)]
        rv = rmse[np.isfinite(rmse)]
        scene_rows.append({"scene": scene, "platform": platform, "SZA_deg": sza, "deep_water_pixels": int(values.size),
                           "TSM_P05_g_m3": float(np.percentile(values, 5)), "TSM_median_g_m3": float(np.median(values)),
                           "TSM_P95_g_m3": float(np.percentile(values, 95)),
                           "TSM_above_old_0.10_percent": float(100 * np.mean(values > .1)),
                           "TSM_above_10_percent": float(100 * np.mean(values > 10)),
                           "forward_RMSE_median_sr-1": float(np.median(rv)), "forward_RMSE_P95_sr-1": float(np.percentile(rv, 95)),
                           "TSM_outside_0.1_20_percent": float(100 * np.mean((values < .1) | (values > 20)))})
        print(f"SCENE {scene}: TSM median {np.median(values):.2f} g m-3; forward RMSE {np.median(rv):.5f} sr-1", flush=True)
    scene_df, roi_df, spectra_df = pd.DataFrame(scene_rows), pd.DataFrame(roi_rows), pd.DataFrame(spectra_rows)
    scene_df.to_csv(output / "05_QC" / "scene_TSM_and_forward_closure.csv", index=False)
    pd.DataFrame(excluded_rows).to_csv(output / "05_QC" / "excluded_after_deep_water_QC.csv", index=False)
    roi_df.to_csv(output / "03_ROI_Results" / "fixed_deep_water_ROI_results.csv", index=False)
    spectra_df.to_csv(output / "03_ROI_Results" / "fixed_deep_water_HydroLight_spectra.csv", index=False)
    return scene_df, roi_df, spectra_df


def figures(output: Path, scene_df: pd.DataFrame, roi_df: pd.DataFrame, spectra_df: pd.DataFrame) -> None:
    folder = output / "06_Figures"
    folder.mkdir(parents=True, exist_ok=True)
    anchors = pd.read_csv(output / "02_HydroLight_Forward_Library" / "full_spectrum_anchor_library.csv")
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for i, tsm_value in enumerate(ANCHOR_TSM):
        group = anchors[anchors.TSM_g_m3 == tsm_value].sort_values("wavelength_nm")
        ax.plot(group.wavelength_nm, group["Rrs_SZA30_sr-1"], lw=2.2,
                color=cmap(i / (len(ANCHOR_TSM) - 1)), label=f"{tsm_value:g} g m⁻³")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("HydroLight water-leaving Rrs (sr⁻¹)")
    ax.set_title("HydroLight full-spectrum response to the expanded TSM range")
    ax.grid(alpha=.25)
    ax.legend(title="TSM input", frameon=False, ncol=2)
    fig.savefig(folder / "00_HydroLight_TSM_anchor_spectra.png", dpi=180)
    plt.close(fig)

    d = scene_df.sort_values("TSM_median_g_m3")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    ax.errorbar(x, d.TSM_median_g_m3,
                yerr=[d.TSM_median_g_m3 - d.TSM_P05_g_m3, d.TSM_P95_g_m3 - d.TSM_median_g_m3],
                fmt="o", color="#087f6a", capsize=3)
    ax.axhspan(10, 17, color="#f59e0b", alpha=.16, label="Published Kepulauan Seribu regional range: 10–17 mg L⁻¹")
    ax.axhline(.1, color="#dc2626", ls="--", label="Old HydroLight upper TSM: 0.10 g m⁻³")
    ax.set_xticks(x, d.scene, rotation=70, ha="right")
    ax.set_xlabel("Scene date and Sentinel-2 platform")
    ax.set_ylabel("Nechad-2016 TSM_approx (g m⁻³ = mg L⁻¹)")
    ax.set_title("Deep-water ROI-A TSM approximation across retained Tidung scenes")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False)
    fig.savefig(folder / "01_all_scene_TSM.png", dpi=180)
    plt.close(fig)

    primary = "20241025_S2A"
    tpath = output / "01_TSM_Nechad2016" / primary / "TSM_approx_Nechad2016_B04.tif"
    rpath = output / "04_HydroLight_Forward_Closure" / primary / "forward_spectral_RMSE_B02_B04.tif"
    with rasterio.open(tpath) as ds: tsm = ds.read(1)
    with rasterio.open(rpath) as ds: rmse = ds.read(1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    im = axes[0].imshow(tsm, vmin=0, vmax=min(30, np.nanpercentile(tsm, 98)), cmap="viridis")
    axes[0].set_title("Nechad-2016 TSM_approx"); axes[0].set_xlabel("Image column (20 m pixel)"); axes[0].set_ylabel("Image row (20 m pixel)")
    c = fig.colorbar(im, ax=axes[0]); c.set_label("g m⁻³ = mg L⁻¹")
    im = axes[1].imshow(rmse, vmin=0, vmax=np.nanpercentile(rmse, 98), cmap="magma")
    axes[1].set_title("HydroLight forward mismatch"); axes[1].set_xlabel("Image column (20 m pixel)"); axes[1].set_ylabel("Image row (20 m pixel)")
    c = fig.colorbar(im, ax=axes[1]); c.set_label("Three-band RMSE (sr⁻¹)")
    fig.suptitle(f"{primary} · retrieval and forward-closure maps", fontsize=19)
    fig.savefig(folder / "02_primary_scene_maps.png", dpi=180)
    plt.close(fig)

    subset = roi_df[(roi_df.scene == primary) & (roi_df.ROI != "C")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for ax, (_, row) in zip(axes.flat, subset.iterrows()):
        spec = spectra_df[(spectra_df.scene == primary) & (spectra_df.ROI == row.ROI)]
        ax.plot(spec.wavelength_nm, spec["HydroLight_forward_Rrs_sr-1"], lw=2.6, color="#d97706", label="HydroLight forward spectrum")
        obs = [row[f"observed_Rrs_approx_{b}_sr-1"] for b in BANDS]
        ax.scatter([492, 560, 665], obs, s=55, color="#1d4ed8", zorder=3, label="Glint-ACOLITE Rrs_approx")
        ax.set_title(f"ROI-{row.ROI} · retrieved TSM {row.retrieved_TSM_median_g_m3:.2f} g m⁻³")
        ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Rrs (sr⁻¹)"); ax.grid(alpha=.25)
    axes.flat[-1].axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, l, loc="center", frameon=False, fontsize=14)
    fig.suptitle(f"{primary} · TSM-driven HydroLight spectra at five fixed deep-water ROIs", fontsize=19)
    fig.savefig(folder / "03_primary_scene_five_ROI_spectra.png", dpi=180)
    plt.close(fig)

    old = pd.read_csv(SOURCE / "08_HydroLight" / "HydroLight_Sentinel2_convolution.csv")
    old = old[(old.scenario == "CLR") & (old.wind_m_s == 5.0)]
    comparisons = []
    for _, row in roi_df[roi_df.ROI == "C"].iterrows():
        platform = row.scene.split("_")[1]
        old_sza = min(old.simulation_SZA_deg.unique(), key=lambda value: abs(value - row.scene_SZA_deg))
        old_rows = old[(old.platform == platform) & (old.simulation_SZA_deg == old_sza)].set_index("band")
        observed = np.asarray([row[f"observed_Rrs_approx_{band}_sr-1"] for band in BANDS], float)
        old_values = np.asarray([old_rows.loc[band, "Rrs_sr-1"] for band in BANDS], float)
        new_values = np.asarray([row[f"modeled_Rrs_{band}_sr-1"] for band in BANDS], float)
        def angle(a, b):
            return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
        comparisons.append({
            "scene": row.scene,
            "old_CLR_RMSE_sr-1": float(np.sqrt(np.mean((observed - old_values) ** 2))),
            "new_TSM_driven_RMSE_sr-1": float(np.sqrt(np.mean((observed - new_values) ** 2))),
            "old_CLR_spectral_angle_deg": angle(observed, old_values),
            "new_TSM_driven_spectral_angle_deg": angle(observed, new_values),
            "observed_band_order": ">".join(np.asarray(BANDS)[np.argsort(observed)[::-1]]),
            "old_CLR_band_order": ">".join(np.asarray(BANDS)[np.argsort(old_values)[::-1]]),
            "new_TSM_driven_band_order": ">".join(np.asarray(BANDS)[np.argsort(new_values)[::-1]]),
        })
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(output / "05_QC" / "old_CLR_vs_new_TSM_HydroLight_shape.csv", index=False)

    row = roi_df[(roi_df.scene == primary) & (roi_df.ROI == "C")].iloc[0]
    old_sza = min(old.simulation_SZA_deg.unique(), key=lambda value: abs(value - row.scene_SZA_deg))
    old_rows = old[(old.platform == "S2A") & (old.simulation_SZA_deg == old_sza)].set_index("band")
    observed = np.asarray([row[f"observed_Rrs_approx_{band}_sr-1"] for band in BANDS], float)
    old_values = np.asarray([old_rows.loc[band, "Rrs_sr-1"] for band in BANDS], float)
    new_values = np.asarray([row[f"modeled_Rrs_{band}_sr-1"] for band in BANDS], float)
    wavelengths = np.asarray([492, 560, 665])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    series = [("Glint-ACOLITE Rrs_approx", observed, "#1d4ed8", "o-"),
              ("Old HydroLight CLR · TSM 0.05", old_values, "#64748b", "s--"),
              ("New HydroLight · retrieved TSM 3.86", new_values, "#d97706", "^--")]
    for label, values, color, style in series:
        axes[0].plot(wavelengths, values, style, lw=2.6, ms=7, color=color, label=label)
        axes[1].plot(wavelengths, values / values.max(), style, lw=2.6, ms=7, color=color, label=label)
    axes[0].set_title("Absolute magnitude")
    axes[0].set_xlabel("Sentinel-2 band-centre wavelength (nm)")
    axes[0].set_ylabel("Rrs / Rrs_approx (sr⁻¹)")
    axes[1].set_title("Spectral shape normalized to each curve maximum")
    axes[1].set_xlabel("Sentinel-2 band-centre wavelength (nm)")
    axes[1].set_ylabel("Normalized spectral value (unitless)")
    for ax in axes:
        ax.grid(alpha=.25); ax.set_xticks(wavelengths)
    axes[0].legend(frameon=False)
    fig.suptitle("2024-10-25 ROI-C · old clear-water reference versus retrieved-TSM HydroLight", fontsize=18)
    fig.savefig(folder / "04_old_CLR_vs_new_TSM_HydroLight.png", dpi=180)
    plt.close(fig)


def build_html(output: Path, scene_df: pd.DataFrame, roi_df: pd.DataFrame) -> None:
    primary = roi_df[(roi_df.scene == "20241025_S2A") & (roi_df.ROI != "C")]
    roi_rows = "".join(f"<tr><td>{r.ROI}</td><td>{r.valid_pixels}</td><td>{r.retrieved_TSM_median_g_m3:.2f}</td>"
                       f"<td>{r.three_band_RMSE_sr_1:.5f}</td><td>{r.spectral_angle_deg:.2f}°</td></tr>"
                       for _, r in primary.rename(columns={"three_band_RMSE_sr-1": "three_band_RMSE_sr_1"}).iterrows())
    all_rows = "".join(f"<tr><td>{r.scene}</td><td>{r.TSM_P05_g_m3:.2f}</td><td>{r.TSM_median_g_m3:.2f}</td>"
                       f"<td>{r.TSM_P95_g_m3:.2f}</td><td>{r.TSM_above_10_percent:.1f}%</td><td>{r.forward_RMSE_median_sr_1:.5f}</td></tr>"
                       for _, r in scene_df.sort_values("scene", ascending=False).rename(columns={"forward_RMSE_median_sr-1": "forward_RMSE_median_sr_1"}).iterrows())
    comparison = pd.read_csv(output / "05_QC" / "old_CLR_vs_new_TSM_HydroLight_shape.csv")
    comparison_primary = comparison[comparison.scene == "20241025_S2A"].iloc[0]
    new_rmse_better = int((comparison["new_TSM_driven_RMSE_sr-1"] < comparison["old_CLR_RMSE_sr-1"]).sum())
    new_shape_better = int((comparison.new_TSM_driven_spectral_angle_deg < comparison.old_CLR_spectral_angle_deg).sum())
    page = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Tidung TSM → HydroLight</title><style>
body{{margin:0;background:#f4f7fa;color:#152238;font:16px/1.58 Arial,sans-serif}}header{{padding:30px 5vw;background:#073b4c;color:white}}main{{max-width:1200px;margin:auto;padding:22px}}section{{background:white;border:1px solid #d6e0e9;border-radius:14px;padding:24px;margin:18px 0}}h1,h2,h3{{line-height:1.22}}.flow{{display:flex;gap:9px;overflow:auto;align-items:center}}.step{{min-width:175px;background:#e8f5f2;border:1px solid #8acabb;border-radius:10px;padding:13px}}.arrow{{font-size:28px}}img{{max-width:100%;border:1px solid #d8e1e9;border-radius:10px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#edf3f7;position:sticky;top:0}}.scroll{{max-height:500px;overflow:auto}}.warn{{background:#fff6dc;border-left:5px solid #d97706;padding:14px}}.good{{background:#eaf7f1;border-left:5px solid #07806a;padding:14px}}.lang button{{padding:8px 14px;margin-right:6px}}[data-lang=ko]{{display:none}}body.ko [data-lang=en]{{display:none}}body.ko [data-lang=ko]{{display:block}}</style></head><body>
<header><div class='lang'><button onclick="document.body.className=''">English</button><button onclick="document.body.className='ko'">한국어</button></div><div data-lang='en'><h1>Tidung TSM inversion and HydroLight forward spectra</h1><p>Corrected Sentinel-2 spectrum → TSM_approx → physically declared HydroLight spectrum</p></div><div data-lang='ko'><h1>티둥 TSM 역산 및 HydroLight 전방 스펙트럼</h1><p>보정 Sentinel-2 스펙트럼 → TSM 근사값 → 명시된 HydroLight 스펙트럼</p></div></header><main>
<section data-lang='en'><h2>1 · Why the HydroLight experiment was rerun</h2><p>The previous four clear-water cases stopped at TSM = 0.10 g m⁻³. In the retained Tidung scenes, the corrected B04 data produce much larger Nechad-2016 estimates. Regional research for Kepulauan Seribu also reports 10–17 mg L⁻¹. Because 1 mg L⁻¹ is numerically 1 g m⁻³, the new full-spectrum HydroLight library spans 0.10–20 g m⁻³.</p><p class='warn'><b>This does not mean Tidung is forced to be turbid.</b> The wide range prevents clipping across the observed deep-water medians. Values above 20 g m⁻³ remain in the TSM GeoTIFF but are explicitly outside the current forward-spectral library.</p></section>
<section data-lang='ko'><h2>1 · HydroLight 재실행 이유</h2><p>기존 시나리오는 TSM 0.10 g m⁻³에서 끝났지만, 보정 B04로 계산한 값은 더 넓습니다. 새 전체 스펙트럼 라이브러리는 0.10–20 g m⁻³를 사용하며 20 g m⁻³ 초과값은 범위 밖으로 명시합니다.</p></section>
<section><h2>2 · Processing order</h2><div class='flow'><div class='step'><b>Stage 1</b><br>Glint-ACOLITE corrected B02/B03/B04</div><div class='arrow'>→</div><div class='step'><b>Stage 2</b><br>Rrs_approx = corrected BOA / π</div><div class='arrow'>→</div><div class='step'><b>Stage 3</b><br>Nechad-2016 B04 TSM_approx GeoTIFF</div><div class='arrow'>→</div><div class='step'><b>Stage 4</b><br>Insert ROI TSM into HydroLight</div><div class='arrow'>→</div><div class='step'><b>Stage 5</b><br>Forward spectrum and residual QC</div></div></section>
<section data-lang='en'><h2>3 · Equations, variables, and units</h2><p><b>TSM = Aρ<sub>w</sub>/(1 − ρ<sub>w</sub>/C)</b>, where A = 342.10 g m⁻³, C = 0.19563, and ρ<sub>w</sub> is the dimensionless corrected Sentinel-2 MSI B04 reflectance proxy. TSM output is g m⁻³ (= mg L⁻¹).</p><p>HydroLight input for every forward spectrum: retrieved TSM; Chl-a = {CHL_A:.2f} mg m⁻³; aCDOM(440) = {CDOM_440:.2f} m⁻¹; SZA interpolated to the scene metadata; wind = {WIND:.0f} m s⁻¹; infinite deep water; no bottom. X-axis = wavelength (nm). Y-axis = above-water water-leaving Rrs (sr⁻¹), linear scale.</p><p class='warn'><b>Interpretation limit:</b> only TSM is retrieved here. Chl-a and CDOM are fixed assumptions. The satellite source is corrected Sen2Cor L2A BOA, so the products remain TSM_approx/Rrs_approx rather than field-validated aquatic products.</p></section>
<section data-lang='ko'><h2>3 · 식과 단위</h2><p><b>TSM = Aρ<sub>w</sub>/(1 − ρ<sub>w</sub>/C)</b>. A=342.10 g m⁻³, C=0.19563이며 출력 단위는 g m⁻³(=mg L⁻¹)입니다. HydroLight Rrs 단위는 sr⁻¹이고 모든 그래프는 선형 축입니다.</p></section>
<section><h2>4 · HydroLight TSM sensitivity</h2><img src='../06_Figures/00_HydroLight_TSM_anchor_spectra.png' alt='HydroLight TSM anchor spectra'><p><b>How to read:</b> X is wavelength in nm and Y is modeled above-water water-leaving Rrs in sr⁻¹. Each line changes TSM only; Chl-a, CDOM, SZA, wind, and depth remain fixed. The graph shows why TSM changes both spectral magnitude and shape, especially from green to red wavelengths.</p></section>
<section><h2>5 · All retained scenes</h2><img src='../06_Figures/01_all_scene_TSM.png' alt='All scene TSM'><p><b>How to read:</b> each point is the median over the same deep-water ROI-A; vertical bars are P05–P95. The red dashed line is the old model's maximum, not a quality threshold. The orange band is a regional comparison range, not field truth for these exact pixels and dates.</p><div class='scroll'><table><thead><tr><th>Scene</th><th>P05<br>g m⁻³</th><th>Median<br>g m⁻³</th><th>P95<br>g m⁻³</th><th>Pixels &gt;10</th><th>Forward RMSE<br>sr⁻¹</th></tr></thead><tbody>{all_rows}</tbody></table></div></section>
<section><h2>6 · Primary scene: 2024-10-25 S2A</h2><img src='../06_Figures/02_primary_scene_maps.png' alt='TSM and residual maps'><p><b>Left:</b> TSM_approx derived pixel-by-pixel from corrected B04. <b>Right:</b> mismatch between observed B02–B04 Rrs_approx and the TSM-driven HydroLight spectrum. Dark right-panel pixels have smaller model mismatch. This map diagnoses the declared water model; it is not itself a TSM map.</p><img src='../06_Figures/03_primary_scene_five_ROI_spectra.png' alt='Five ROI spectra'><p><b>Blue points:</b> observed Glint-ACOLITE Rrs_approx at the three available visible bands. <b>Orange line:</b> full 400–900 nm HydroLight spectrum after inserting that ROI's retrieved TSM. Agreement should repeat across C1–C5. A mismatch indicates that glint/atmosphere, fixed Chl-a/CDOM, mineral IOPs, or ROI depth may need revision.</p><table><thead><tr><th>ROI</th><th>Pixels</th><th>Retrieved TSM<br>g m⁻³</th><th>3-band RMSE<br>sr⁻¹</th><th>Spectral angle</th></tr></thead><tbody>{roi_rows}</tbody></table></section>
<section><h2>7 · Does the past Glint-ACOLITE result match the new spectrum?</h2><img src='../06_Figures/04_old_CLR_vs_new_TSM_HydroLight.png' alt='Old and new HydroLight comparison'><p><b>No—not for the primary ROI-C.</b> The observed Glint-ACOLITE spectrum decreases B02 &gt; B03 &gt; B04. The old low-TSM CLR reference has the same ordering but is too low in magnitude. The new TSM-driven model peaks at B03 and is too high in magnitude.</p><ul><li>Old CLR: RMSE {comparison_primary['old_CLR_RMSE_sr-1']:.5f} sr⁻¹; spectral angle {comparison_primary.old_CLR_spectral_angle_deg:.2f}°.</li><li>New retrieved-TSM HydroLight: RMSE {comparison_primary['new_TSM_driven_RMSE_sr-1']:.5f} sr⁻¹; spectral angle {comparison_primary.new_TSM_driven_spectral_angle_deg:.2f}°.</li><li>Across 32 ROI-C scene means, the new model lowers RMSE in only {new_rmse_better}/32 scenes and improves spectral angle in only {new_shape_better}/32 scenes.</li></ul><p class='warn'>The old curve's closer shape does not prove the old TSM = 0.05 g m⁻³ was correct. It means the new run exposes an inconsistency: B04-derived TSM combined with the current HydroLight mineral IOP, Chl-a and CDOM assumptions does not reproduce the corrected three-band spectrum.</p></section>
<section><h2>8 · What is saved</h2><ul><li><b>01_TSM_Nechad2016:</b> reusable georeferenced TSM_approx GeoTIFF for every retained scene.</li><li><b>02_HydroLight_Forward_Library:</b> inputs plus raw D/L/M/S/P outputs for seven full-spectrum TSM anchors.</li><li><b>03_ROI_Results:</b> per-ROI observed values, retrieved TSM, full model spectra, residuals, RMSE, and spectral angle.</li><li><b>04_HydroLight_Forward_Closure:</b> modeled Rrs and observed-minus-modeled residual GeoTIFFs per band.</li><li><b>05_QC:</b> one row per scene with percentiles and old-versus-new shape metrics.</li></ul><p class='good'>These files allow later analysis without extracting values from figures.</p></section>
<section><h2>9 · Scientific conclusion supported by this run</h2><p>The old 0–0.10 g m⁻³ HydroLight TSM range was not suitable for this archive. The new workflow retrieves TSM from each scene first and then uses that value in HydroLight. For the primary ROI-C, retrieved TSM is 3.86 g m⁻³, but the TSM-driven spectrum remains substantially above the satellite points. Therefore increasing TSM fixed the unrealistic input range but did <b>not</b> validate the current Glint-ACOLITE Rrs_approx. The next sensitivity test must vary Chl-a, CDOM and mineral-specific IOPs, and the atmospheric-correction source should be revisited before absolute TSM use.</p></section>
</main></body></html>"""
    dash = output / "dashboard"; dash.mkdir(parents=True, exist_ok=True)
    (dash / "index.html").write_text(page, encoding="utf-8")
    standalone = page
    for image_name in ("00_HydroLight_TSM_anchor_spectra.png", "01_all_scene_TSM.png",
                       "02_primary_scene_maps.png", "03_primary_scene_five_ROI_spectra.png",
                       "04_old_CLR_vs_new_TSM_HydroLight.png"):
        data = base64.b64encode((output / "06_Figures" / image_name).read_bytes()).decode("ascii")
        standalone = standalone.replace(f"../06_Figures/{image_name}", f"data:image/png;base64,{data}")
    (dash / "Tidung_TSM_HydroLight_Forward_Standalone_20260904.html").write_text(standalone, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    for name in ("00_Method", "01_TSM_Nechad2016", "02_HydroLight_Forward_Library", "03_ROI_Results",
                 "04_HydroLight_Forward_Closure", "05_QC", "06_Figures", "dashboard"):
        (output / name).mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": str(SOURCE),
        "input": "Glint-ACOLITE corrected Sen2Cor L2A BOA B02/B03/B04",
        "retrieval": {"algorithm": "Nechad-2016 MSI B04 SPM/TSM", "A": NECHAD_A, "C": NECHAD_C,
                      "output": "TSM_approx g m-3"},
        "forward_model": {"software": "HydroLight 5.1.4", "TSM_anchor_g_m3": list(ANCHOR_TSM),
                          "Chl_a_mg_m3": CHL_A, "a_CDOM_440_m-1": CDOM_440, "wind_m_s": WIND,
                          "depth": "infinite/no bottom", "SZA": "scene metadata via interpolation of 20-40 degree grid"},
        "status": "retrieval plus optical closure; not independent field validation",
    }
    (output / "00_Method" / "method_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    anchors = build_anchor_library(output, args.resume)
    library = ForwardLibrary(anchors)
    scene_df, roi_df, spectra_df = process(output, library)
    figures(output, scene_df, roi_df, spectra_df)
    build_html(output, scene_df, roi_df)
    print(f"DONE: {output}", flush=True)


if __name__ == "__main__":
    main()
