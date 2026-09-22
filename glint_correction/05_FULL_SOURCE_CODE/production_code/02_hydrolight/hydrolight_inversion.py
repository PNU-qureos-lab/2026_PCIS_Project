#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retrieve TSM proxies from the Tidung glint-corrected archive and run HydroLight closure.

This is deliberately a two-estimate workflow:

1. Nechad-2016 SPM/TSM from corrected Sentinel-2 B04 water reflectance.
2. A discrete HydroLight lookup-table fit to B02/B03/B04 Rrs_approx.

The second estimate is an optical closure/sensitivity experiment, not independent
validation.  The satellite input is the existing Sen2Cor L2A BOA product after the
project's Glint-ACOLITE subtraction; therefore outputs are named TSM_approx.
"""

from __future__ import annotations

# WORKFLOW ROLE — STEP 02B: HYDROLIGHT INVERSION/MODEL MATCHING
# This file defines the HydroLight input/output parsing and spectral matching
# used to compare measured water spectra with physically simulated spectra.

import argparse
import csv
import html
import json
import math
import re
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


ROOT = Path(__file__).resolve().parents[4]  # ACOLITE-Glint Correction workspace
SOURCE = ROOT / "outputs" / "Tidung_Glint_Comparison_Rerun_20260903"
DEFAULT_OUTPUT = ROOT / "outputs" / "Tidung_TSM_HydroLight_Inversion_20260904"
PILOT_TEMPLATE = ROOT / "outputs" / "hydrolight_pilot_20251101" / "inputs" / "ITIDHL_CLR.txt"
HE5 = Path(r"C:\HE5")
EXECUTABLE = HE5 / "Code" / "mainHL_stnd.exe"
RUN_DIR = HE5 / "run"

BANDS = ("B02", "B03", "B04")
BAND_LABEL = {"B02": "blue", "B03": "green", "B04": "red"}
SZA_GRID = (20.0, 25.0, 30.0, 35.0, 40.0)
TSM_GRID = (0.10, 0.25, 0.50, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0)
BACKGROUNDS = {
    "OLI": {"name": "Oligotrophic-clear", "chl": 0.05, "cdom": 0.01},
    "CLR": {"name": "Clear-coastal", "chl": 0.10, "cdom": 0.03},
    "PRD": {"name": "Productive-coastal", "chl": 0.50, "cdom": 0.06},
}
WIND_M_S = 5.0

# ACOLITE's distributed Nechad 2016 MSI B4 calibration.
NECHAD_A = 342.10
NECHAD_C = 0.19563
NECHAD_VALID_MAX = 0.5 * NECHAD_C

COLORS = {"observed": "#27364d", "model": "#d97706", "nechad": "#087f6a", "hydro": "#6d28d9"}
plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 11})


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def parse_rrs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith('"Rrs"'))
    shape = re.search(r"\s(\d+)\s+(\d+)\s*$", lines[start])
    expected = int(shape.group(1)) if shape else 101
    rows = []
    for line in lines[start + 3 :]:
        values = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)
        if len(values) < 2:
            if rows:
                break
            continue
        rows.append((float(values[0]), float(values[1])))
        if len(rows) >= expected:
            break
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} HydroLight rows in {path}, found {len(rows)}")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def output_sources(root_name: str) -> dict[str, Path]:
    base = HE5 / "output" / "Hydrolight"
    return {
        "D": base / "digital" / f"D{root_name}.txt",
        "L": base / "digital" / f"L{root_name}.txt",
        "M": base / "excel" / f"M{root_name}.txt",
        "S": base / "excel" / f"S{root_name}.txt",
        "P": base / "printout" / f"P{root_name}.txt",
    }


def hydro_input(root_name: str, sza: float, background: str, tsm: float) -> str:
    lines = PILOT_TEMPLATE.read_text().splitlines()
    bg = BACKGROUNDS[background]
    lines[1] = f"Tidung TSM LUT; {background}; TSM {tsm:g} g m-3; SZA {sza:g}"
    lines[2] = root_name
    # Case-2 component concentrations: [model flag, Chl-a, aCDOM(440), mineral/TSM].
    lines[6] = f"0, {bg['chl']:.6f}, {bg['cdom']:.6f}, {tsm:.6f}"
    sun = [x.strip() for x in lines[34].split(",")]
    sun[2] = f"{sza:.6f}"
    lines[34] = ", ".join(sun)
    surface = [x.strip() for x in lines[36].split(",")]
    surface[0] = f"{WIND_M_S:.3f}"
    lines[36] = ", ".join(surface)
    # HydroLight 5.1.4 rejects the shortened three-wavelength output record on
    # this installation. Keep the validated full-spectrum template, then
    # convolve the modeled spectrum to Sentinel-2 B02/B03/B04 afterwards.
    return "\n".join(lines) + "\n"


def run_hydrolight(input_path: Path, result_dir: Path, root_name: str, resume: bool) -> str:
    expected = {key: result_dir / src.name for key, src in output_sources(root_name).items()}
    if resume and all(p.exists() and p.stat().st_size > 0 for p in expected.values()):
        return "SKIPPED_EXISTING"
    result_dir.mkdir(parents=True, exist_ok=True)
    with input_path.open("r") as handle:
        proc = subprocess.run([str(EXECUTABLE)], stdin=handle, cwd=RUN_DIR, text=True, capture_output=True, check=False)
    (result_dir / "process_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (result_dir / "process_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"HydroLight failed for {root_name} with code {proc.returncode}")
    for key, src in output_sources(root_name).items():
        if not src.exists() or src.stat().st_size == 0:
            raise RuntimeError(f"Missing HydroLight output {key}: {src}")
        shutil.copy2(src, expected[key])
    if "fatal error" in expected["P"].read_text(errors="replace").lower():
        raise RuntimeError(f"HydroLight printout reports fatal error for {root_name}")
    parse_rrs(expected["M"])
    return "COMPLETED"


def read_rsr(platform: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    path = ROOT / "acolite-source" / "data" / "RSR" / f"{platform}_MSI_V4.txt"
    out: dict[str, list[tuple[float, float]]] = {}
    band = None
    for line in path.read_text().splitlines():
        m = re.match(r"#\s+\S+_MSI Band (\S+)", line)
        if m:
            band = f"B{m.group(1)}"
            out.setdefault(band, [])
        elif band and line and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 2:
                out[band].append((float(parts[0]), float(parts[1])))
    return {b: (np.asarray(v)[:, 0], np.asarray(v)[:, 1]) for b, v in out.items() if v}


def convolve(wave: np.ndarray, values: np.ndarray, rsr: tuple[np.ndarray, np.ndarray]) -> float:
    rw, response = rsr
    interp = np.interp(rw, wave, values, left=np.nan, right=np.nan)
    good = np.isfinite(interp) & np.isfinite(response) & (response > 0)
    return float(np.trapz(interp[good] * response[good], rw[good]) / np.trapz(response[good], rw[good]))


def build_lut(output: Path, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = output / "02_HydroLight_LUT" / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    runs, spectra = [], []
    for si, sza in enumerate(SZA_GRID):
        for bi, background in enumerate(BACKGROUNDS):
            for ti, tsm in enumerate(TSM_GRID):
                root_name = f"Q{si}{bi}{ti:02d}"
                inp = inputs / f"I{root_name}.txt"
                inp.write_text(hydro_input(root_name, sza, background, tsm), encoding="utf-8")
                result = output / "02_HydroLight_LUT" / "results" / f"SZA_{int(sza):02d}" / background / f"TSM_{tsm:g}"
                status = run_hydrolight(inp, result, root_name, resume)
                mfile = result / f"M{root_name}.txt"
                wave, rrs = parse_rrs(mfile)
                runs.append({
                    "root_name": root_name, "SZA_deg": sza, "wind_m_s": WIND_M_S,
                    "background": background, "background_name": BACKGROUNDS[background]["name"],
                    "Chl_a_mg_m3": BACKGROUNDS[background]["chl"],
                    "a_CDOM_440_m-1": BACKGROUNDS[background]["cdom"], "TSM_g_m3": tsm,
                    "status": status, "input_file": str(inp), "result_dir": str(result),
                })
                spectra.extend({"root_name": root_name, "SZA_deg": sza, "background": background,
                                "TSM_g_m3": tsm, "wavelength_nm": w, "Rrs_sr-1": r}
                               for w, r in zip(wave, rrs))
                print(f"{status}: {root_name} SZA={sza:g} {background} TSM={tsm:g}", flush=True)
    run_df, spectrum_df = pd.DataFrame(runs), pd.DataFrame(spectra)
    run_df.to_csv(output / "02_HydroLight_LUT" / "hydrolight_run_manifest.csv", index=False)
    spectrum_df.to_csv(output / "02_HydroLight_LUT" / "hydrolight_spectra_full.csv", index=False)

    band_rows = []
    for root_name, group in spectrum_df.groupby("root_name", sort=False):
        info = run_df.loc[run_df.root_name == root_name].iloc[0]
        wave, vals = group.wavelength_nm.to_numpy(), group["Rrs_sr-1"].to_numpy()
        for platform in ("S2A", "S2B", "S2C"):
            rsr = read_rsr(platform)
            row = {"root_name": root_name, "platform": platform, "SZA_deg": info.SZA_deg,
                   "background": info.background, "TSM_g_m3": info.TSM_g_m3}
            # Integrate the full HydroLight spectrum through each MSI band response.
            # Do not assign the first three 5-nm samples (400/405/410 nm) to
            # B02/B03/B04: those are spectral-grid rows, not Sentinel-2 bands.
            row.update({band: convolve(wave, vals, rsr[band.replace("0", "", 1)])
                        for band in BANDS})
            band_rows.append(row)
    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(output / "02_HydroLight_LUT" / "hydrolight_sentinel2_convolution.csv", index=False)
    return run_df, band_df


def write_raster(path: Path, array: np.ndarray, profile: dict, description: str, units: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", predictor=3)
    with rasterio.open(path, "w", **p) as ds:
        ds.write(array.astype("float32"), 1)
        ds.set_band_description(1, description)
        ds.update_tags(AREA_OR_POINT="Area", units=units)


def nechad_tsm(rho_red: np.ndarray) -> np.ndarray:
    valid = np.isfinite(rho_red) & (rho_red > 0) & (rho_red < NECHAD_VALID_MAX)
    out = np.full(rho_red.shape, np.nan, dtype=np.float32)
    out[valid] = NECHAD_A * rho_red[valid] / (1.0 - rho_red[valid] / NECHAD_C)
    return out


def spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / den, -1, 1))))


def fixed_roi_masks(ds: rasterio.DatasetReader, definition: list[dict]) -> dict[str, np.ndarray]:
    masks = {}
    for item in definition:
        rid = item.get("roi_id")
        if not rid or not re.fullmatch(r"C\d+", rid):
            continue
        xs, ys = transform_xy("EPSG:4326", ds.crs, [item["center_lon"]], [item["center_lat"]])
        row, col = ds.index(xs[0], ys[0])
        mask = np.zeros(ds.shape, dtype=bool)
        mask[max(0, row - 3): min(ds.height, row + 4), max(0, col - 3): min(ds.width, col + 4)] = True
        masks[rid] = mask
    masks["C"] = np.logical_or.reduce(list(masks.values()))
    return masks


def percentile_text(values: np.ndarray) -> str:
    v = values[np.isfinite(values)]
    if not v.size:
        return "n/a"
    q = np.percentile(v, [5, 50, 95])
    return f"P05 {q[0]:.2f}; median {q[1]:.2f}; P95 {q[2]:.2f} g m⁻³"


def save_scene_figures(output: Path, scene: str, red: np.ndarray, tsm_n: np.ndarray,
                       tsm_h: np.ndarray, rmse: np.ndarray, roi_rows: list[dict]) -> None:
    folder = output / "06_Figures" / scene
    folder.mkdir(parents=True, exist_ok=True)
    panels = [
        (red, "Corrected B04 input", "BOA reflectance", 0, np.nanpercentile(red[np.isfinite(red)], 98), "magma"),
        (tsm_n, "Nechad-2016 TSM_approx", "g m⁻³", 0, min(50, np.nanpercentile(tsm_n[np.isfinite(tsm_n)], 98)), "viridis"),
        (tsm_h, "HydroLight LUT TSM_approx · deep-water mask", "g m⁻³", 0, 50, "viridis"),
        (rmse, "HydroLight spectral-fit residual", "RMSE (sr⁻¹)", 0, np.nanpercentile(rmse[np.isfinite(rmse)], 98), "inferno"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, (arr, title, unit, vmin, vmax, cmap) in zip(axes.flat, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Image column (20 m pixel)")
        ax.set_ylabel("Image row (20 m pixel)")
        cb = fig.colorbar(im, ax=ax, shrink=.82)
        cb.set_label(unit)
    fig.suptitle(f"{scene} · TSM inversion products", fontsize=19)
    fig.savefig(folder / "01_tsm_maps.png", dpi=160)
    plt.close(fig)

    rows = [r for r in roi_rows if r["scene"] == scene and r["ROI"] != "C"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, row in zip(axes.flat, rows):
        x = [492, 560, 665]
        obs = [row[f"observed_Rrs_{b}_sr-1"] for b in BANDS]
        mod = [row[f"modeled_Rrs_{b}_sr-1"] for b in BANDS]
        ax.plot(x, obs, "o-", lw=2.5, color=COLORS["observed"], label="Glint-ACOLITE Rrs_approx")
        ax.plot(x, mod, "s--", lw=2.5, color=COLORS["model"], label="Best HydroLight LUT")
        ax.set_title(f"ROI-{row['ROI']} · TSM {row['HydroLight_TSM_g_m3']:.2f} g m⁻³")
        ax.set_xlabel("Sentinel-2 band-centre wavelength (nm)")
        ax.set_ylabel("Rrs (sr⁻¹)")
        ax.grid(alpha=.25)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"{scene} · fixed deep-water ROI spectra and HydroLight closure", fontsize=18)
    fig.savefig(folder / "02_roi_spectral_closure.png", dpi=160)
    plt.close(fig)


def process_scenes(output: Path, band_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(SOURCE / "00_Scene_Selection" / "scene_inventory.csv")
    qc = pd.read_csv(SOURCE / "09_Final_Comparison" / "scene_physical_QC_flags.csv")
    scenes = qc.loc[qc.final_aggregate_included.astype(str).str.lower() == "true", "scene"].tolist()
    scene_rows, roi_rows = [], []
    for scene in scenes:
        date, platform = scene.split("_")
        meta = inventory[(inventory.date.astype(str) == date) & (inventory.platform == platform) & (inventory.decision == "ACCEPT")]
        if meta.empty:
            raise RuntimeError(f"No accepted inventory record for {scene}")
        meta = meta.iloc[0]
        sza = float(min(SZA_GRID, key=lambda x: abs(x - float(meta.sun_zenith_angle))))
        models = band_df[(band_df.platform == platform) & (band_df.SZA_deg == sza)].reset_index(drop=True)
        model_values = models[list(BANDS)].to_numpy(dtype=np.float32)

        paths = [SOURCE / "06_ACOLITE" / "Corrected" / scene / f"{b}_corrected.tif" for b in BANDS]
        with rasterio.open(paths[0]) as ds:
            profile = ds.profile
            corrected = [ds.read(1)]
            roi_defs = json.loads((SOURCE / "03_ROI" / scene / "roi_definitions.json").read_text())
            fixed_masks = fixed_roi_masks(ds, roi_defs)
        for p in paths[1:]:
            with rasterio.open(p) as ds:
                corrected.append(ds.read(1))
        with rasterio.open(SOURCE / "03_ROI" / scene / "ROI_A.tif") as ds:
            deep = ds.read(1) > 0
        with rasterio.open(SOURCE / "01_Masks" / scene / "valid_water_mask.tif") as ds:
            water = ds.read(1) > 0

        red = corrected[2]
        tsm_n = nechad_tsm(red)
        tsm_n[~water] = np.nan
        n_dir = output / "01_TSM_Nechad2016" / scene
        write_raster(n_dir / "TSM_approx_Nechad2016_B04.tif", tsm_n, profile,
                     "TSM approximation from corrected MSI B04 using Nechad 2016", "g m-3")

        obs = np.stack(corrected, axis=-1).astype(np.float32) / np.pi
        valid = deep & np.all(np.isfinite(obs), axis=-1) & np.all(obs > 0, axis=-1)
        flat = obs[valid]
        best_idx = np.empty(flat.shape[0], dtype=np.int16)
        best_rmse = np.empty(flat.shape[0], dtype=np.float32)
        chunk = 25000
        for start in range(0, flat.shape[0], chunk):
            part = flat[start:start + chunk]
            err = np.sqrt(np.mean((part[:, None, :] - model_values[None, :, :]) ** 2, axis=2))
            idx = np.argmin(err, axis=1)
            best_idx[start:start + chunk] = idx
            best_rmse[start:start + chunk] = err[np.arange(len(idx)), idx]
        tsm_h = np.full(deep.shape, np.nan, np.float32)
        rmse = np.full(deep.shape, np.nan, np.float32)
        bg_code = np.full(deep.shape, np.nan, np.float32)
        modeled = [np.full(deep.shape, np.nan, np.float32) for _ in BANDS]
        if flat.size:
            chosen = models.iloc[best_idx]
            tsm_h[valid] = chosen.TSM_g_m3.to_numpy(np.float32)
            rmse[valid] = best_rmse
            bg_map = {k: i + 1 for i, k in enumerate(BACKGROUNDS)}
            bg_code[valid] = chosen.background.map(bg_map).to_numpy(np.float32)
            for i, band in enumerate(BANDS):
                modeled[i][valid] = chosen[band].to_numpy(np.float32)
        h_dir = output / "04_HydroLight_Closure" / scene
        write_raster(h_dir / "TSM_approx_HydroLight_LUT.tif", tsm_h, profile,
                     "Best-fit HydroLight LUT TSM on deep-water ROI-A", "g m-3")
        write_raster(h_dir / "spectral_fit_RMSE.tif", rmse, profile,
                     "B02-B04 Rrs spectral fit RMSE", "sr-1")
        write_raster(h_dir / "best_background_code.tif", bg_code, profile,
                     "HydroLight background code: 1 OLI, 2 CLR, 3 PRD", "class")
        for band, arr in zip(BANDS, modeled):
            write_raster(h_dir / f"modeled_Rrs_{band}.tif", arr, profile,
                         f"Best-fit HydroLight modeled Rrs {band}", "sr-1")
            write_raster(h_dir / f"residual_Rrs_{band}.tif", obs[..., BANDS.index(band)] - arr, profile,
                         f"Observed Rrs_approx minus modeled HydroLight Rrs {band}", "sr-1")

        if not np.any(valid):
            scene_rows.append({
                "scene": scene, "platform": platform,
                "scene_SZA_deg": float(meta.sun_zenith_angle), "LUT_SZA_deg": sza,
                "deep_water_pixels": 0,
                "Nechad_TSM_P05_g_m3": np.nan,
                "Nechad_TSM_median_g_m3": np.nan,
                "Nechad_TSM_P95_g_m3": np.nan,
                "HydroLight_TSM_P05_g_m3": np.nan,
                "HydroLight_TSM_median_g_m3": np.nan,
                "HydroLight_TSM_P95_g_m3": np.nan,
                "HydroLight_fit_RMSE_median_sr-1": np.nan,
                "HydroLight_fit_RMSE_P95_sr-1": np.nan,
                "LUT_edge_selection_percent": np.nan,
                "interpretation": "excluded_no_valid_positive_deep_water_spectrum",
            })
            print(f"SCENE {scene}: excluded (no valid positive deep-water spectrum)", flush=True)
            continue

        local_roi_rows = []
        for rid, mask in fixed_masks.items():
            good = mask & np.all(np.isfinite(obs), axis=-1) & np.all(obs > 0, axis=-1)
            if not np.any(good):
                print(f"SCENE {scene} ROI {rid}: excluded (no valid positive spectrum)", flush=True)
                continue
            observed = np.nanmedian(obs[good], axis=0)
            errors = np.sqrt(np.mean((model_values - observed[None, :]) ** 2, axis=1))
            idx = int(np.argmin(errors))
            mod = models.iloc[idx]
            row = {
                "scene": scene, "ROI": rid, "valid_pixels": int(good.sum()),
                "scene_SZA_deg": float(meta.sun_zenith_angle), "LUT_SZA_deg": sza,
                "background": mod.background, "background_name": BACKGROUNDS[mod.background]["name"],
                "Chl_a_mg_m3": BACKGROUNDS[mod.background]["chl"],
                "a_CDOM_440_m-1": BACKGROUNDS[mod.background]["cdom"],
                "Nechad_TSM_median_g_m3": float(np.nanmedian(tsm_n[mask])),
                "HydroLight_TSM_g_m3": float(mod.TSM_g_m3),
                "spectral_RMSE_sr-1": float(errors[idx]),
                "spectral_angle_deg": spectral_angle(observed, mod[list(BANDS)].to_numpy(float)),
            }
            for i, band in enumerate(BANDS):
                row[f"observed_Rrs_{band}_sr-1"] = float(observed[i])
                row[f"modeled_Rrs_{band}_sr-1"] = float(mod[band])
                row[f"residual_{band}_sr-1"] = float(observed[i] - mod[band])
            roi_rows.append(row)
            local_roi_rows.append(row)

        nvals, hvals, rvals = tsm_n[deep], tsm_h[deep], rmse[deep]
        nvals, hvals, rvals = nvals[np.isfinite(nvals)], hvals[np.isfinite(hvals)], rvals[np.isfinite(rvals)]
        edge = np.isclose(hvals, min(TSM_GRID)) | np.isclose(hvals, max(TSM_GRID))
        row = {
            "scene": scene, "platform": platform, "scene_SZA_deg": float(meta.sun_zenith_angle), "LUT_SZA_deg": sza,
            "deep_water_pixels": int(valid.sum()),
            "Nechad_TSM_P05_g_m3": float(np.percentile(nvals, 5)),
            "Nechad_TSM_median_g_m3": float(np.median(nvals)),
            "Nechad_TSM_P95_g_m3": float(np.percentile(nvals, 95)),
            "HydroLight_TSM_P05_g_m3": float(np.percentile(hvals, 5)),
            "HydroLight_TSM_median_g_m3": float(np.median(hvals)),
            "HydroLight_TSM_P95_g_m3": float(np.percentile(hvals, 95)),
            "HydroLight_fit_RMSE_median_sr-1": float(np.median(rvals)),
            "HydroLight_fit_RMSE_P95_sr-1": float(np.percentile(rvals, 95)),
            "LUT_edge_selection_percent": float(100 * np.mean(edge)),
            "interpretation": "retrieval_and_model_closure_not_independent_validation",
        }
        scene_rows.append(row)
        save_scene_figures(output, scene, red, tsm_n, tsm_h, rmse, local_roi_rows)
        print(f"SCENE {scene}: Nechad median={row['Nechad_TSM_median_g_m3']:.2f}; HydroLight median={row['HydroLight_TSM_median_g_m3']:.2f} g m-3", flush=True)

    scene_df, roi_df = pd.DataFrame(scene_rows), pd.DataFrame(roi_rows)
    scene_df.to_csv(output / "05_QC" / "scene_TSM_summary.csv", index=False)
    roi_df.to_csv(output / "03_ROI_Inversion" / "fixed_deep_water_ROI_results.csv", index=False)
    return scene_df, roi_df


def summary_figures(output: Path, scene_df: pd.DataFrame, roi_df: pd.DataFrame) -> None:
    folder = output / "06_Figures" / "summary"
    folder.mkdir(parents=True, exist_ok=True)
    d = scene_df.sort_values("Nechad_TSM_median_g_m3")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)
    ax.errorbar(x, d.Nechad_TSM_median_g_m3,
                yerr=[d.Nechad_TSM_median_g_m3 - d.Nechad_TSM_P05_g_m3,
                      d.Nechad_TSM_P95_g_m3 - d.Nechad_TSM_median_g_m3],
                fmt="o", capsize=3, color=COLORS["nechad"], label="Nechad-2016: median and P05–P95")
    ax.scatter(x, d.HydroLight_TSM_median_g_m3, marker="s", color=COLORS["hydro"], label="HydroLight LUT median")
    ax.set_xticks(x, d.scene, rotation=70, ha="right")
    ax.set_ylabel("TSM approximation (g m⁻³ = mg L⁻¹)")
    ax.set_xlabel("Scene date and Sentinel-2 platform")
    ax.set_title("Deep-water ROI-A TSM estimates across 33 retained scenes")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(folder / "01_scene_tsm_summary.png", dpi=170)
    plt.close(fig)

    c = roi_df[roi_df.ROI == "C"].copy()
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.scatter(c.Nechad_TSM_median_g_m3, c.HydroLight_TSM_g_m3, s=55, alpha=.8, color="#2563eb")
    lim = max(20, float(np.nanmax([c.Nechad_TSM_median_g_m3.max(), c.HydroLight_TSM_g_m3.max()])))
    ax.plot([0, lim], [0, lim], "--", color="#64748b", label="1:1 guide")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Nechad-2016 ROI-C median TSM_approx (g m⁻³)")
    ax.set_ylabel("Best HydroLight LUT ROI-C TSM (g m⁻³)")
    ax.set_title("Two TSM estimates are diagnostic, not interchangeable")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.savefig(folder / "02_nechad_vs_hydrolight_roi_c.png", dpi=170)
    plt.close(fig)


def build_html(output: Path, scene_df: pd.DataFrame, roi_df: pd.DataFrame) -> None:
    rows = []
    for _, r in scene_df.sort_values("scene", ascending=False).iterrows():
        rows.append(f"<tr><td>{html.escape(r.scene)}</td><td>{r.Nechad_TSM_median_g_m3:.2f}</td>"
                    f"<td>{r.HydroLight_TSM_median_g_m3:.2f}</td><td>{r['HydroLight_fit_RMSE_median_sr-1']:.5f}</td>"
                    f"<td>{r.LUT_edge_selection_percent:.1f}%</td></tr>")
    roi_primary = roi_df[(roi_df.scene == "20241025_S2A") & (roi_df.ROI != "C")]
    roi_table = "".join(
        f"<tr><td>C{r.ROI[1:]}</td><td>{r.Nechad_TSM_median_g_m3:.2f}</td><td>{r.HydroLight_TSM_g_m3:.2f}</td>"
        f"<td>{html.escape(r.background_name)}</td><td>{r.spectral_RMSE_sr_1:.5f}</td></tr>"
        for _, r in roi_primary.rename(columns={"spectral_RMSE_sr-1": "spectral_RMSE_sr_1"}).iterrows()
    )
    page = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Tidung TSM inversion and HydroLight closure</title><style>
body{{font-family:Arial,sans-serif;margin:0;color:#172033;background:#f5f7fb;line-height:1.55}}header{{background:#083344;color:white;padding:28px max(24px,5vw)}}main{{max-width:1200px;margin:auto;padding:24px}}section{{background:white;border:1px solid #d9e1eb;border-radius:14px;padding:24px;margin:18px 0}}h1,h2,h3{{line-height:1.2}}.flow{{display:flex;gap:10px;align-items:stretch;overflow:auto}}.step{{min-width:170px;padding:14px;border-radius:10px;background:#e8f5f2;border:1px solid #85c9ba}}.arrow{{font-size:28px;align-self:center}}img{{max-width:100%;height:auto;border:1px solid #dbe3ec;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{padding:9px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#edf2f7;position:sticky;top:0}}.scroll{{overflow:auto;max-height:520px}}.note{{padding:14px;background:#fff7df;border-left:5px solid #d97706}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}.unit{{color:#476075}}a{{color:#075e8c}}</style></head><body>
<header><h1>Tidung TSM inversion → HydroLight spectral closure</h1><p>Expanded Indonesian-water TSM search range · 33 physically retained Sentinel-2 L2A scenes</p></header><main>
<section><h2>1. What was changed, and why</h2><p>The old HydroLight clear-water cases used only 0–0.10 g m⁻³ TSM. The corrected red-band data indicate a much wider range, and published work for Kepulauan Seribu reports 10–17 mg L⁻¹. Because <b>1 mg L⁻¹ = 1 g m⁻³</b> for this concentration unit, the new lookup table spans <b>0.10–80 g m⁻³</b>. This is a search range, not an assumption that every Tidung pixel is turbid.</p>
<div class='flow'><div class='step'><b>1 · Input</b><br>Glint-ACOLITE corrected B02/B03/B04 BOA reflectance</div><div class='arrow'>→</div><div class='step'><b>2 · Rrs approximation</b><br>Corrected BOA ÷ π, sr⁻¹</div><div class='arrow'>→</div><div class='step'><b>3 · Nechad TSM</b><br>B04 single-band TSM_approx map</div><div class='arrow'>→</div><div class='step'><b>4 · HydroLight LUT</b><br>TSM × water background × SZA</div><div class='arrow'>→</div><div class='step'><b>5 · Spectral fit</b><br>Compare B02/B03/B04 and save residuals</div></div></section>
<section><h2>2. Equations and units</h2><h3>Nechad-2016 MSI B04</h3><p><i>TSM</i> = Aρ<sub>w</sub> / (1 − ρ<sub>w</sub>/C), with A = 342.10 g m⁻³ and C = 0.19563. Here ρ<sub>w</sub> is the corrected B04 BOA reflectance proxy (dimensionless). Output: g m⁻³, numerically equivalent to mg L⁻¹.</p><h3>HydroLight closure</h3><p>For each observed vector [Rrs(B02), Rrs(B03), Rrs(B04)], the report selects the lookup-table spectrum with the smallest three-band RMSE. X is wavelength (nm); Y is Rrs (sr⁻¹), always linear.</p><div class='note'><b>Scientific limit:</b> the current inputs are corrected Sen2Cor L2A BOA reflectance, not ACOLITE aquatic L2R. Therefore this report uses <b>Rrs_approx</b> and <b>TSM_approx</b>. Fitting HydroLight to the same spectrum used for retrieval is optical closure, not independent validation.</div></section>
<section><h2>3. HydroLight experiment design</h2><p>TSM grid: {', '.join(f'{x:g}' for x in TSM_GRID)} g m⁻³. Each value is simulated at SZA 20°, 25°, 30°, 35° and 40°, wind 5 m s⁻¹, infinite depth/no bottom, using three declared optical backgrounds:</p><ul><li>OLI · oligotrophic-clear: Chl-a 0.05 mg m⁻³; aCDOM(440) 0.01 m⁻¹.</li><li>CLR · clear-coastal: Chl-a 0.10 mg m⁻³; aCDOM(440) 0.03 m⁻¹.</li><li>PRD · productive-coastal: Chl-a 0.50 mg m⁻³; aCDOM(440) 0.06 m⁻¹.</li></ul><p>These backgrounds expose Chl-a/CDOM ambiguity instead of pretending three visible bands identify all constituents uniquely.</p></section>
<section><h2>4. Cross-scene result</h2><img src='../06_Figures/summary/01_scene_tsm_summary.png' alt='Scene TSM summary'><p><b>How to read:</b> green circles are Nechad B04 medians with P05–P95 spread within deep-water ROI-A; purple squares are best-fit HydroLight LUT medians. Large separation means the single-band and full three-band model interpret the spectrum differently and requires further checking.</p></section>
<section><h2>5. Primary scene example · 2024-10-25 S2A</h2><img src='../06_Figures/20241025_S2A/01_tsm_maps.png' alt='TSM inversion maps'><p><b>Panel meanings:</b> corrected B04 is the inversion input; Nechad converts it pixel-by-pixel to TSM_approx; HydroLight TSM is fitted only inside the deep-water mask; fit RMSE shows the remaining spectral mismatch in sr⁻¹. Low RMSE is model agreement, not proof of field accuracy.</p><img src='../06_Figures/20241025_S2A/02_roi_spectral_closure.png' alt='ROI spectral closure'><p><b>How to read:</b> each panel is one independently fixed 7×7-pixel ROI. Solid lines are Glint-ACOLITE Rrs_approx; dashed lines are the selected HydroLight spectrum. A credible closure should be repeatable across all five ROIs, not only one hand-picked box.</p><div class='scroll'><table><thead><tr><th>ROI</th><th>Nechad TSM<br>g m⁻³</th><th>HydroLight TSM<br>g m⁻³</th><th>Selected background</th><th>RMSE<br>sr⁻¹</th></tr></thead><tbody>{roi_table}</tbody></table></div></section>
<section><h2>6. All retained scenes</h2><div class='scroll'><table><thead><tr><th>Scene</th><th>Nechad median<br>g m⁻³</th><th>HydroLight median<br>g m⁻³</th><th>Median fit RMSE<br>sr⁻¹</th><th>LUT-edge pixels</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p><b>LUT-edge pixels</b> are pixels selecting either 0.10 or 80 g m⁻³. A large percentage means the search grid or water model may still be inadequate; it is not automatically a high/low TSM finding.</p></section>
<section><h2>7. Reusable outputs</h2><ul><li>GeoTIFF Nechad TSM maps: <code>01_TSM_Nechad2016/&lt;scene&gt;/</code></li><li>HydroLight raw runs and Sentinel-2 convolution: <code>02_HydroLight_LUT/</code></li><li>Five fixed-ROI inversion table: <code>03_ROI_Inversion/fixed_deep_water_ROI_results.csv</code></li><li>GeoTIFF best-fit TSM, modeled Rrs and residuals: <code>04_HydroLight_Closure/&lt;scene&gt;/</code></li><li>Scene QC summary: <code>05_QC/scene_TSM_summary.csv</code></li></ul></section>
</main></body></html>"""
    dash = output / "dashboard"
    dash.mkdir(parents=True, exist_ok=True)
    (dash / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    for folder in ("00_Input_Audit", "01_TSM_Nechad2016", "02_HydroLight_LUT", "03_ROI_Inversion",
                   "04_HydroLight_Closure", "05_QC", "06_Figures", "dashboard"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    if not EXECUTABLE.exists():
        raise FileNotFoundError(EXECUTABLE)
    if not PILOT_TEMPLATE.exists():
        raise FileNotFoundError(PILOT_TEMPLATE)

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive": r"H:\Sentinel-2\Tidung\L2A",
        "source_workflow": str(SOURCE),
        "input_quantity": "Glint-ACOLITE-corrected Sen2Cor L2A BOA reflectance",
        "retrieval_output": "TSM_approx in g m-3 (numerically mg L-1)",
        "nechad": {"parameter": "SPM/TSM", "sensor": "Sentinel-2 MSI", "band": "B04",
                    "A_g_m-3": NECHAD_A, "C_reflectance": NECHAD_C,
                    "source_file": str(ROOT / "acolite-source/data/Shared/algorithms/Nechad/Nechad_calibration_201609.txt")},
        "HydroLight_TSM_grid_g_m3": list(TSM_GRID), "HydroLight_backgrounds": BACKGROUNDS,
        "HydroLight_SZA_grid_deg": list(SZA_GRID), "wind_m_s": WIND_M_S,
        "interpretation_limit": "Retrieval plus optical closure; not independent field validation.",
    }
    (output / "00_Input_Audit" / "method_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    _, band_df = build_lut(output, args.resume)
    scene_df, roi_df = process_scenes(output, band_df)
    summary_figures(output, scene_df, roi_df)
    build_html(output, scene_df, roi_df)
    print(f"DONE: {output}", flush=True)


if __name__ == "__main__":
    main()
