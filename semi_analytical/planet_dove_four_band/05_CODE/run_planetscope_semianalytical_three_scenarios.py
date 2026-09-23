#!/usr/bin/env python
"""PlanetScope exploratory semi-analytical inversion for one Tidung scene.

This script adapts the Lee-D2 shallow-water formulation used by the project's
Sentinel-2 ``semi_analytical`` workflow to generic Planet SuperDove band sets.
It intentionally runs one scene and three band scenarios on the same pixels:

* S4: 444, 492, 566, 666 nm
* S6: S4 plus 533 and 612 nm
* S7: S6 plus 707 nm (sensitivity scenario)

The input called Rrs is the project's Hedley-corrected Planet surface
reflectance divided by pi.  It is therefore approximate Rrs, not a dedicated
aquatic atmospheric-correction product.  Outputs are exploratory and must not
be treated as field-validated bathymetry or habitat ground truth.
"""
from __future__ import annotations

import csv
import html
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from scipy.ndimage import binary_closing, binary_opening, distance_transform_edt, label
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

gdal.UseExceptions()

SCENE = "2025-11-17"
SOURCE = Path(r"H:\PlanetDove\Tidung_Standard_Hedley_Final_20260923")
OUTPUT = Path(r"H:\PlanetDove\Tidung_Planet_Semi_Analytical_20251117_20260923")
PROJECT = Path(__file__).resolve().parents[1]
REFERENCE = PROJECT / "Pacific_Island_Project" / "semi_analytical" / "code_v3" / "reference_data"
REMOTE_COMMIT = "a3f06c6d097af2c81f0281568189c48d37bb56d7"
ALL_WAVES = np.array([444, 492, 533, 566, 612, 666, 707], dtype=float)
SCENARIOS = {
    "S4_core": np.array([444, 492, 566, 666], dtype=float),
    "S6_extended": np.array([444, 492, 533, 566, 612, 666], dtype=float),
    "S7_with_707_sensitivity": ALL_WAVES.copy(),
}
SOLAR_ZENITH_DEG = 90.0 - 70.04813
G0, G1 = 0.089, 0.1245
NODATA = -9999.0

# Published/project four-band HOPE coefficient samples.  Additional Planet
# wavelengths are linearly interpolated; 707 nm is linearly extrapolated and
# therefore kept as a sensitivity scenario rather than the primary result.
COEF_WAVES = np.array([443, 490, 560, 665], dtype=float)
COEF_A0 = np.array([0.98902, 0.7558, 0.3433, 0.70225], dtype=float)
COEF_A1 = np.array([0.0018, 0.0256, 0.0659, 0.13615], dtype=float)


def sample_linear_extrap(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = np.interp(target, x, y)
    left = target < x[0]
    right = target > x[-1]
    if np.any(left):
        out[left] = y[0] + (target[left] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    if np.any(right):
        out[right] = y[-1] + (target[right] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return out


def read_envi_table(path: Path, columns: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == r"\end_header") + 1
    rows = []
    for line in lines[start:]:
        value = line.strip()
        if value == r"\end_data":
            break
        if value and not value.startswith(("!", "#")):
            parts = value.replace("D", "E").replace("d", "e").split()
            if len(parts) >= columns:
                rows.append([float(v) for v in parts[:columns]])
    return np.asarray(rows, dtype=float)


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    ds = gdal.Open(str(path))
    array = ds.ReadAsArray().astype("float64")
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        array[array == nodata] = np.nan
    meta = {"geotransform": ds.GetGeoTransform(), "projection": ds.GetProjection(),
            "xsize": ds.RasterXSize, "ysize": ds.RasterYSize}
    return array, meta


def write_raster(path: Path, array: np.ndarray, meta: dict, dtype=gdal.GDT_Float32,
                 nodata=NODATA, description="") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    options = ["COMPRESS=DEFLATE", "TILED=YES", "PREDICTOR=2"] if dtype != gdal.GDT_Byte else ["COMPRESS=DEFLATE", "TILED=YES"]
    ds = gdal.GetDriverByName("GTiff").Create(str(path), meta["xsize"], meta["ysize"], 1, dtype, options=options)
    ds.SetGeoTransform(meta["geotransform"]); ds.SetProjection(meta["projection"])
    band = ds.GetRasterBand(1); band.SetNoDataValue(nodata)
    out = np.asarray(array).copy()
    out[~np.isfinite(out)] = nodata
    band.WriteArray(out); band.SetDescription(description); band.FlushCache(); ds = None


def rrs_from_Rrs(Rrs: np.ndarray) -> np.ndarray:
    return Rrs / (0.52 + 1.7 * Rrs)


def Rrs_from_rrs(rrs: np.ndarray) -> np.ndarray:
    return 0.52 * rrs / (1.0 - 1.7 * rrs)


def spectral_inputs(waves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    water = read_envi_table(REFERENCE / "H2OabDefaults_SEAwater.txt", 3)
    sand = read_envi_table(REFERENCE / "coral_sand.txt", 2)
    grass = read_envi_table(REFERENCE / "avg_seagrass.txt", 2)
    aw = np.interp(waves, water[:, 0], water[:, 1])
    bbw = 0.49997 * np.interp(waves, water[:, 0], water[:, 2])
    rho_sand = np.interp(waves, sand[:, 0], sand[:, 1])
    rho_grass = np.interp(waves, grass[:, 0], grass[:, 1])
    return aw, bbw, rho_sand, rho_grass


def pgx_spectrum(pgx: np.ndarray, waves: np.ndarray, aw: np.ndarray, bbw: np.ndarray) -> dict:
    p, g, x = pgx
    a0 = sample_linear_extrap(COEF_WAVES, COEF_A0, waves)
    a1 = sample_linear_extrap(COEF_WAVES, COEF_A1, waves)
    a_phi = (a0 + a1 * np.log(max(p, 1e-12))) * p
    a_g = g * np.exp(-0.015 * (waves - 440.0))
    bbp = x * (550.0 / waves) ** 0.5
    a = aw + a_phi + a_g
    bb = bbw + bbp
    u = bb / (a + bb)
    modeled = Rrs_from_rrs((G0 + G1 * u) * u)
    return {"a": a, "bb": bb, "modeled_Rrs": modeled,
            "a_phi": a_phi, "a_g": a_g, "bbp": bbp}


def fit_deep(deep: np.ndarray, waves: np.ndarray, aw: np.ndarray, bbw: np.ndarray) -> tuple[np.ndarray, dict]:
    def residual(log_params):
        pgx = np.exp(log_params)
        modeled = pgx_spectrum(pgx, waves, aw, bbw)["modeled_Rrs"]
        return (modeled - deep) / np.maximum(deep, 1e-6)
    starts = [np.array([0.02, 0.02, 0.003]), np.array([0.05, 0.05, 0.01]), np.array([0.10, 0.02, 0.02])]
    fits = [least_squares(residual, np.log(s), max_nfev=2000) for s in starts]
    best = min(fits, key=lambda x: np.mean(x.fun ** 2))
    pgx = np.exp(best.x)
    spec = pgx_spectrum(pgx, waves, aw, bbw)
    return pgx, {**spec, "relative_rmse": float(np.sqrt(np.mean(best.fun ** 2))),
                 "success": bool(best.success), "message": best.message}


def make_candidate_mask(water: np.ndarray, training: np.ndarray, stack: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
    valid = water & np.all(np.isfinite(stack), axis=0) & np.all(stack > 0, axis=0)
    labs, count = label(~water)
    components = []
    h, w = water.shape
    for index in range(1, count + 1):
        rr, cc = np.where(labs == index)
        if len(rr) == 0:
            continue
        touches = (np.any(rr == 0) or np.any(rr == h - 1) or np.any(cc == 0) or np.any(cc == w - 1))
        if not touches:
            components.append((len(rr), index))
    if not components:
        raise RuntimeError("No enclosed island component found in water mask")
    island_index = max(components)[1]
    island = labs == island_index
    distance_m = distance_transform_edt(~island) * abs(float(META["geotransform"][1]))
    fit_pixels = training & valid
    median = np.median(stack[:, fit_pixels], axis=1)
    mad = 1.4826 * np.median(np.abs(stack[:, fit_pixels] - median[:, None]), axis=1)
    mad = np.maximum(mad, 1e-5)
    score = np.sqrt(np.mean(((stack - median[:, None, None]) / mad[:, None, None]) ** 2, axis=0))
    threshold = float(np.percentile(score[fit_pixels], 95))
    candidate = valid & (distance_m >= 9.0) & (distance_m <= 450.0) & (score > threshold)
    candidate = binary_opening(candidate, structure=np.ones((2, 2), bool))
    candidate = binary_closing(candidate, structure=np.ones((3, 3), bool))
    # Morphological closing may bridge across an invalid pixel.  Reapply every
    # scientific eligibility condition so no non-water or nonpositive spectrum
    # can enter the inversion.
    candidate &= valid & (distance_m >= 9.0) & (distance_m <= 450.0) & (score > threshold)
    metadata = {
        "main_island_pixels": int(island.sum()), "distance_from_island_m": [9, 450],
        "deep_training_score_percentile": 95, "score_threshold": threshold,
        "candidate_pixels": int(candidate.sum()), "valid_water_pixels": int(valid.sum()),
        "definition": "Positive seven-band water within 9-450 m of the main island and spectrally beyond the 95th percentile of the fixed offshore Hedley-training water.",
        "warning": "This is an image-derived candidate benthic mask, not field-validated bottom visibility."
    }
    return candidate, metadata, score


def build_lut(waves: np.ndarray, deep: np.ndarray) -> tuple[dict, dict]:
    aw, bbw, sand, grass = spectral_inputs(waves)
    pgx, fit = fit_deep(deep, waves, aw, bbw)
    a, bb = fit["a"], fit["bb"]
    kappa = a + bb; u = bb / kappa
    mu_s = math.sqrt(1.0 - (math.sin(math.radians(SOLAR_ZENITH_DEG)) / 1.34) ** 2)
    xc = (1.0 / mu_s + 1.03 * np.sqrt(1.0 + 2.4 * u)) * kappa
    xb = (1.0 / mu_s + 1.04 * np.sqrt(1.0 + 5.4 * u)) * kappa
    deep_below = rrs_from_Rrs(deep)
    depths = np.round(np.arange(0.0, 20.0001, 0.1), 3)
    fractions = np.round(np.arange(0.0, 1.0001, 0.01), 3)
    zz, ff = np.meshgrid(depths, fractions, indexing="ij")
    z = zz.ravel()[:, None]; f = ff.ravel()[:, None]
    bottom_rho = f * sand[None, :] + (1.0 - f) * grass[None, :]
    below = deep_below[None, :] * (-np.expm1(-xc[None, :] * z)) + (bottom_rho / np.pi) * np.exp(-xb[None, :] * z)
    predicted = Rrs_from_rrs(below)
    good = np.all(np.isfinite(predicted), axis=1) & np.all(predicted > 0, axis=1)
    lut = {"predicted": predicted[good], "depth": zz.ravel()[good], "sand": ff.ravel()[good]}
    metadata = {"PGX": pgx.tolist(), "deep_fit_relative_rmse": fit["relative_rmse"],
                "deep_observed_Rrs": deep.tolist(), "deep_modeled_Rrs": fit["modeled_Rrs"].tolist(),
                "aw": aw.tolist(), "bbw": bbw.tolist(), "sand_reflectance": sand.tolist(),
                "seagrass_reflectance": grass.tolist(), "lut_entries": int(good.sum()),
                "depth_grid_m": [0.0, 20.0, 0.1], "sand_fraction_grid": [0.0, 1.0, 0.01]}
    return lut, metadata


def invert_scenario(name: str, waves: np.ndarray, stack7: np.ndarray, candidate: np.ndarray,
                    training: np.ndarray, meta: dict) -> dict:
    out = OUTPUT / {"S4_core": "03_SCENARIO_4_BANDS", "S6_extended": "04_SCENARIO_6_BANDS",
                    "S7_with_707_sensitivity": "05_SCENARIO_7_BANDS"}[name]
    out.mkdir(parents=True, exist_ok=True)
    band_indices = [int(np.where(ALL_WAVES == wave)[0][0]) for wave in waves]
    stack = stack7[band_indices]
    fit_mask = training & np.all(np.isfinite(stack), axis=0) & np.all(stack > 0, axis=0)
    deep = np.median(stack[:, fit_mask], axis=1)
    lut, lut_meta = build_lut(waves, deep)
    obs = stack[:, candidate].T
    scale = np.maximum(deep, 0.002)
    tree = cKDTree(lut["predicted"] / scale[None, :])
    _, neighbors = tree.query(obs / scale[None, :], k=16, workers=-1)
    pred = lut["predicted"][neighbors]
    rel = (pred - obs[:, None, :]) / np.maximum(obs[:, None, :], 1e-6)
    costs = np.mean(rel * rel, axis=2)
    choice_col = np.argmin(costs, axis=1)
    choice = neighbors[np.arange(len(obs)), choice_col]
    best_cost = costs[np.arange(len(obs)), choice_col]
    rows, cols = candidate.shape
    depth = np.full((rows, cols), np.nan, "float32")
    sand = np.full((rows, cols), np.nan, "float32")
    cost = np.full((rows, cols), np.nan, "float32")
    rmse = np.full((rows, cols), np.nan, "float32")
    depth[candidate] = lut["depth"][choice]
    sand[candidate] = lut["sand"][choice]
    cost[candidate] = best_cost
    rmse[candidate] = np.sqrt(np.mean((lut["predicted"][choice] - obs) ** 2, axis=1))
    grass = np.where(np.isfinite(sand), 1.0 - sand, np.nan)
    boundary_percent = float(100.0 * np.mean((lut["depth"][choice] == 0.0) | (lut["depth"][choice] == 20.0)))
    pgx_unstable = bool(np.max(np.asarray(lut_meta["PGX"])) > 10.0)
    if boundary_percent > 5.0 or pgx_unstable:
        decision = "REJECT — inversion collapsed to bounds or deep-water fit is unstable"
    elif lut_meta["deep_fit_relative_rmse"] > 0.25:
        decision = "EXPLORATORY ONLY — spatial solution exists, but deep-water optical fit is poor"
    else:
        decision = "CANDIDATE — still requires field validation"
    status = np.zeros(candidate.shape, "uint8"); status[candidate] = 1
    write_raster(out / f"{SCENE}_{name}_depth_m.tif", depth, meta, description="Lee-D2 candidate depth, metres")
    write_raster(out / f"{SCENE}_{name}_sand_fraction.tif", sand, meta, description="Candidate coral-sand fraction")
    write_raster(out / f"{SCENE}_{name}_seagrass_fraction.tif", grass, meta, description="Candidate seagrass fraction")
    write_raster(out / f"{SCENE}_{name}_relative_cost.tif", cost, meta, description="Mean squared relative spectral residual")
    write_raster(out / f"{SCENE}_{name}_Rrs_RMSE.tif", rmse, meta, description="Absolute Rrs spectral RMSE, sr-1")
    write_raster(out / f"{SCENE}_{name}_status.tif", status, meta, dtype=gdal.GDT_Byte, nodata=0, description="1=inverted candidate pixel; 0=not processed")
    summary = {"scenario": name, "bands_nm": waves.astype(int).tolist(), "pixels": int(candidate.sum()),
               "median_depth_m": float(np.nanmedian(depth)), "p05_depth_m": float(np.nanpercentile(depth, 5)),
               "p95_depth_m": float(np.nanpercentile(depth, 95)), "median_sand_fraction": float(np.nanmedian(sand)),
               "median_Rrs_RMSE_sr-1": float(np.nanmedian(rmse)), "p95_Rrs_RMSE_sr-1": float(np.nanpercentile(rmse, 95)),
               "median_relative_cost": float(np.nanmedian(cost)), "depth_boundary_percent": boundary_percent,
               "deep_fit_relative_rmse": lut_meta["deep_fit_relative_rmse"], "decision": decision, "lut": lut_meta,
               "coefficient_policy": "A0/A1 linearly sampled from the project four-band coefficients; 707 nm is extrapolated and sensitivity-only."}
    (out / "scenario_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "depth": depth, "sand": sand, "rmse": rmse, "folder": out}


def save_figures(stack: np.ndarray, candidate: np.ndarray, score: np.ndarray, results: dict) -> None:
    figdir = OUTPUT / "06_COMPARISON" / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    rr, cc = np.where(candidate)
    pad = 35
    rs = slice(max(0, int(rr.min()) - pad), min(candidate.shape[0], int(rr.max()) + pad + 1))
    cs = slice(max(0, int(cc.min()) - pad), min(candidate.shape[1], int(cc.max()) + pad + 1))
    rgb = np.moveaxis(stack[[5, 3, 1]], 0, -1) * np.pi
    lo, hi = np.nanpercentile(rgb[np.all(np.isfinite(rgb), axis=2)], [1, 99])
    shown = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.8
    fig, ax = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    ax[0].imshow(shown[rs, cs]); ax[0].set_title("Hedley-corrected Planet RGB\n2025-11-17")
    ax[1].imshow(candidate[rs, cs], cmap="Blues", vmin=0, vmax=1); ax[1].set_title(f"Common candidate mask\n{candidate.sum():,} pixels")
    im = ax[2].imshow(score[rs, cs], cmap="magma", vmin=0, vmax=np.nanpercentile(score[candidate], 99)); ax[2].contour(candidate[rs, cs], [.5], colors="cyan", linewidths=.5); ax[2].set_title("Spectral distance from offshore training water"); fig.colorbar(im, ax=ax[2], shrink=.75)
    for a in ax: a.axis("off")
    fig.suptitle("Input and common processing mask", fontsize=18, weight="bold")
    fig.savefig(figdir / "01_input_and_candidate_mask.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14), constrained_layout=True)
    for row, (name, result) in enumerate(results.items()):
        items = [(result["depth"], "Depth (m)", "viridis", (0, 12)), (result["sand"], "Sand fraction", "YlOrBr", (0, 1)), (result["rmse"], "Rrs RMSE (sr⁻¹)", "magma", (0, np.nanpercentile(result["rmse"], 95)))]
        for col, (arr, title, cmap, limits) in enumerate(items):
            im = axes[row, col].imshow(arr[rs, cs], cmap=cmap, vmin=limits[0], vmax=limits[1]); axes[row, col].set_title(f"{name}\n{title}"); axes[row, col].axis("off"); fig.colorbar(im, ax=axes[row, col], shrink=.70)
    fig.suptitle("Same scene and pixels: three semi-analytical band scenarios", fontsize=18, weight="bold")
    fig.savefig(figdir / "02_three_scenario_maps.png", dpi=170); plt.close(fig)
    names = list(results); labels = ["4 bands", "6 bands", "7 bands\n(+707 sensitivity)"]
    metrics = [[results[n]["summary"][key] for n in names] for key in ("median_depth_m", "median_sand_fraction", "median_Rrs_RMSE_sr-1")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    titles = ["Median retrieved depth (m)", "Median sand fraction", "Median spectral RMSE (sr⁻¹)"]
    colors = ["#2457d6", "#008b78", "#d97706"]
    for ax, vals, title in zip(axes, metrics, titles):
        bars = ax.bar(labels, vals, color=colors); ax.set_title(title, weight="bold"); ax.grid(axis="y", alpha=.25)
        for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.4g}", ha="center", va="bottom")
    fig.savefig(figdir / "03_scenario_summary.png", dpi=180); plt.close(fig)


def build_report(mask_meta: dict, results: dict) -> None:
    summaries = [r["summary"] for r in results.values()]
    rows = "".join(f"<tr><td>{html.escape(s['scenario'])}</td><td>{', '.join(map(str,s['bands_nm']))}</td><td>{s['median_depth_m']:.2f}</td><td>{s['median_sand_fraction']:.3f}</td><td>{s['median_Rrs_RMSE_sr-1']:.5f}</td><td>{s['depth_boundary_percent']:.1f}%</td><td>{html.escape(s['decision'])}</td></tr>" for s in summaries)
    report = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PlanetScope Tidung semi-analytical scenarios</title><style>
body{{font-family:Arial,sans-serif;margin:0;background:#eef4f3;color:#132f36;font-size:18px;line-height:1.55}}main{{max-width:1250px;margin:auto;padding:28px}}header,.card{{background:white;border:1px solid #c8d8d5;border-radius:16px;padding:26px;margin:18px 0}}h1{{font-size:36px}}h2{{font-size:27px;border-bottom:3px solid #13a594;padding-bottom:8px}}.warn{{background:#fff3d6;border-left:6px solid #d97706;padding:18px}}.good{{background:#e4f7f2;border-left:6px solid #078772;padding:18px}}img{{width:100%;height:auto;border-radius:10px}}table{{width:100%;border-collapse:collapse;font-size:16px}}th,td{{padding:11px;border-bottom:1px solid #ccd9d7;text-align:left}}th{{background:#174d58;color:white}}code{{background:#edf2f1;padding:2px 5px}}a{{color:#075ec7}}li{{margin:8px 0}}</style></head><body><main>
<header><h1>Tidung PlanetScope semi-analytical inversion</h1><p><b>One data scene:</b> {SCENE}. <b>Three band scenarios:</b> 4, 6, and 7 bands. All scenarios use the same {mask_meta['candidate_pixels']:,} candidate pixels so the comparison is fair.</p></header>
<section class="card"><h2>Easy concept</h2><ul><li>Hedley first reduces surface glint.</li><li>The semi-analytical model then asks which combination of water depth and sand–seagrass bottom best reproduces each observed spectrum.</li><li>The 4-band result is the direct counterpart of the existing Sentinel workflow.</li><li>The 6-band result tests whether Planet's extra visible bands add useful constraints.</li><li>The 7-band result adds 707 nm only as a sensitivity test.</li></ul><div class="warn"><b>Important:</b> input is approximate Rrs = corrected Planet surface reflectance / π. It is not a dedicated aquatic atmospheric-correction product and is not field validated. Depth and bottom fractions are exploratory.</div></section>
<section class="card"><h2>Step 1 — Input and processing mask</h2><img src="06_COMPARISON/figures/01_input_and_candidate_mask.png"><p>{html.escape(mask_meta['definition'])}</p></section>
<section class="card"><h2>Step 2 — Fit the deep-water optical background</h2><p>The fixed offshore training pixels provide one median spectrum for each scenario. The workflow fits the project HOPE P/G/X water model to that spectrum. A large relative fit error or extreme P/G/X values is a warning that approximate Planet Rrs does not satisfy the model assumptions.</p></section>
<section class="card"><h2>Step 3 — Build the shallow-water library</h2><p>For every scenario, the workflow calculates 20,301 spectra: depth 0–20 m in 0.1 m steps, crossed with sand fraction 0–1 in 0.01 steps. Seagrass fraction is <code>1 − sand fraction</code>.</p><p><code>rrs = rrs_deep × (1 − exp(−Kc × depth)) + bottom/π × exp(−Kb × depth)</code></p></section>
<section class="card"><h2>Step 4 — Match every candidate pixel</h2><p>Each observed Planet spectrum is compared with the library. The stored depth and mixture minimize the mean squared relative spectral residual among the nearest library candidates.</p></section>
<section class="card"><h2>Step 5 — Results and failure screen</h2><img src="06_COMPARISON/figures/02_three_scenario_maps.png"><table><thead><tr><th>Scenario</th><th>Bands (nm)</th><th>Median depth</th><th>Median sand</th><th>Median RMSE</th><th>Depth at 0/20 m</th><th>QC decision</th></tr></thead><tbody>{rows}</tbody></table><img src="06_COMPARISON/figures/03_scenario_summary.png"><div class="warn"><b>Boundary rule:</b> if more than 5% of pixels select exactly 0 or 20 m, the scenario is rejected because the optimizer is asking to leave the tested depth range. A low-looking spectral RMSE does not override this failure.</div></section>
<section class="card"><h2>How to use the outputs</h2><ul><li>Use <b>S4_core</b> as the most comparable result to the existing four-band Sentinel workflow.</li><li>Use <b>S6_extended</b> as the preferred Planet sensitivity comparison because it adds 533 and 612 nm without relying on 707 nm.</li><li>Use <b>S7_with_707_sensitivity</b> only to see how the red-edge-like 707 nm constraint changes the solution.</li><li>Do not choose a scenario only because its RMSE is lower: more bands and model mismatch change the objective.</li><li>Do not call the sand/seagrass maps habitat classifications until field spectra and depth observations validate them.</li></ul></section>
<section class="card"><h2>Reproducibility</h2><p>The exact executable code, copied input rasters, reference spectra, metadata, and run command are included in this folder. The adaptation is based on the project semi-analytical workflow at commit <code>{REMOTE_COMMIT}</code>, using the Lee shallow-water model and the project's sand–seagrass library.</p><p><a href="07_FULL_SOURCE_CODE/run_planetscope_semianalytical_three_scenarios.py">Open the full code</a> · <a href="06_COMPARISON/scenario_comparison.csv">Open comparison CSV</a> · <a href="00_READ_ME_FIRST/README.md">Read limitations</a></p></section>
</main></body></html>'''
    (OUTPUT / "Tidung_Planet_Semi_Analytical_Three_Scenarios.html").write_text(report, encoding="utf-8")


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Output folder already exists; refusing to overwrite: {OUTPUT}")
    for folder in ["00_READ_ME_FIRST", "01_INPUT", "02_INTERMEDIATE", "03_SCENARIO_4_BANDS", "04_SCENARIO_6_BANDS", "05_SCENARIO_7_BANDS", "06_COMPARISON", "07_FULL_SOURCE_CODE"]:
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
    arrays = []; global META
    for wave in ALL_WAVES.astype(int):
        path = SOURCE / "04_GENERATED_OUTPUT" / SCENE / "05_Rrs_Approx" / f"{SCENE}_{wave}nm_Rrs_approx.tif"
        arr, meta = read_raster(path); arrays.append(arr); META = meta
        shutil.copy2(path, OUTPUT / "01_INPUT" / path.name)
    stack = np.stack(arrays)
    water_path = SOURCE / "04_GENERATED_OUTPUT" / SCENE / "01_Masks" / f"{SCENE}_water_mask.tif"
    training_path = SOURCE / "04_GENERATED_OUTPUT" / SCENE / "01_Masks" / f"{SCENE}_hedley_training_water.tif"
    water, _ = read_raster(water_path); training, _ = read_raster(training_path)
    water = water > 0; training = training > 0
    shutil.copy2(water_path, OUTPUT / "01_INPUT" / water_path.name); shutil.copy2(training_path, OUTPUT / "01_INPUT" / training_path.name)
    for ref in ["H2OabDefaults_SEAwater.txt", "coral_sand.txt", "avg_seagrass.txt"]:
        shutil.copy2(REFERENCE / ref, OUTPUT / "01_INPUT" / ref)
    candidate, mask_meta, score = make_candidate_mask(water, training, stack)
    write_raster(OUTPUT / "02_INTERMEDIATE" / f"{SCENE}_candidate_shallow_water_mask.tif", candidate.astype("uint8"), META, dtype=gdal.GDT_Byte, nodata=0, description="1=common candidate pixel for all three scenarios")
    write_raster(OUTPUT / "02_INTERMEDIATE" / f"{SCENE}_deep_training_distance_score.tif", score, META, description="Robust spectral distance from fixed offshore training-water median")
    (OUTPUT / "02_INTERMEDIATE" / "candidate_mask_metadata.json").write_text(json.dumps(mask_meta, indent=2), encoding="utf-8")
    results = {}
    for name, waves in SCENARIOS.items():
        print(f"Running {name}: {waves.astype(int).tolist()}", flush=True)
        results[name] = invert_scenario(name, waves, stack, candidate, training, META)
    save_figures(stack, candidate, score, results)
    comparison_path = OUTPUT / "06_COMPARISON" / "scenario_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["scenario", "bands_nm", "pixels", "median_depth_m", "p05_depth_m", "p95_depth_m", "median_sand_fraction", "median_Rrs_RMSE_sr-1", "p95_Rrs_RMSE_sr-1", "median_relative_cost", "depth_boundary_percent", "deep_fit_relative_rmse", "decision"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for result in results.values():
            row = {k: result["summary"][k] for k in fields}; row["bands_nm"] = " ".join(map(str, row["bands_nm"])); writer.writerow(row)
    shutil.copy2(Path(__file__), OUTPUT / "07_FULL_SOURCE_CODE" / Path(__file__).name)
    (OUTPUT / "07_FULL_SOURCE_CODE" / "run_command.txt").write_text(f'"C:\\Users\\phili\\anaconda3\\envs\\Dr_doom\\python.exe" "{OUTPUT}\\07_FULL_SOURCE_CODE\\{Path(__file__).name}"\n', encoding="utf-8")
    readme = f'''# Tidung PlanetScope semi-analytical three-scenario package

## What was run

- One scene: **{SCENE}** (selected because it was the only CAUTIOUS standard-Hedley scene in the four-scene QC).
- S4: 444, 492, 566, 666 nm.
- S6: S4 plus 533 and 612 nm.
- S7: S6 plus 707 nm, explicitly sensitivity-only.
- Common candidate pixels: **{mask_meta['candidate_pixels']:,}**.

## What the model estimates

For each candidate pixel, a Lee-D2 shallow-water forward model searches depth from 0–20 m (0.1 m spacing) and a two-endmember sand–seagrass mixture (0.01 fraction spacing). A nearest-neighbour search retrieves the best LUT candidates and the reported solution minimizes relative spectral residual among the 16 nearest candidates.

## Scientific limits

1. Input is approximate Rrs = Hedley-corrected Planet SR / pi; it is not aquatic atmospheric correction.
2. The shallow-water mask is image-derived, not field bathymetry.
3. Sand and seagrass spectra are generic library endmembers, not measured at Tidung.
4. The model's four-band phytoplankton coefficient samples are interpolated for 533/612 nm and extrapolated to 707 nm. Therefore S7 is sensitivity-only.
5. No field depth, habitat, or matched in-water Rrs was available for validation.
6. Outputs are hypotheses for comparison and survey planning, not final habitat truth.

## Provenance

- Project workflow: https://github.com/PNU-qureos-lab/2026_PCIS_Project/tree/main/semi_analytical
- Inspected remote commit: `{REMOTE_COMMIT}`
- Lee et al. shallow-water formulation as implemented in the project workflow.
- Created UTC: {datetime.now(timezone.utc).isoformat()}
'''
    (OUTPUT / "00_READ_ME_FIRST" / "README.md").write_text(readme, encoding="utf-8")
    build_report(mask_meta, results)
    manifest = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file(): manifest.append({"path": str(path.relative_to(OUTPUT)), "bytes": path.stat().st_size})
    (OUTPUT / "00_READ_ME_FIRST" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "files": len(manifest), "summaries": [r["summary"] for r in results.values()]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
