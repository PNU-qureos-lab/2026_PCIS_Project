#!/usr/bin/env python
"""Multi-band joint HydroLight-water and residual-glint experiment for Tidung.

The script extends the existing L2A Glint-ACOLITE products from three visible
bands to the complete usable MSI water spectrum (B02--B8A), reconvolves the
existing full-spectrum HydroLight runs, performs held-out known-answer tests,
and permits one residual correction only when the simulation and real-scene
gates pass.  B11/B12 remain surface-contamination constraints.
"""

from __future__ import annotations

# WORKFLOW ROLE — STEP 02D: JOINT MULTIBAND TEST
# Input: Cycle-0 B02–B8A spectra and the HydroLight simulation library.
# Output: spectral fits, residual evidence, safety metrics, and gate inputs.
# B11/B12 remain surface-glint indicators and are not treated as water spectra.

import base64
import html
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import binary_opening
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[5]  # glint_correction folder
FULL_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FULL_CODE / "01_acolite_cycle0"))
import acolite_helpers as archive
import hydrolight_inversion as hydro

SOURCE = ROOT / "outputs" / "Tidung_Glint_Comparison_Rerun_20260903"
PREVIOUS = ROOT / "outputs" / "Tidung_Clean_Glint_29Scene_Experiment_20260910"
HL_LUT = ROOT / "outputs" / "Tidung_TSM_HydroLight_Inversion_20260904" / "02_HydroLight_LUT"
HL_TRUTH = ROOT / "outputs" / "Tidung_Clean_Glint_Selected_Scenes_Experiment_20260910" / "01_HydroLight_Independent_Cases"
OUTPUT = ROOT / "outputs" / "Tidung_Joint_HydroLight_Glint_Multiband_Experiment_20260911"
SETUP = ROOT / "Tidung_HydroLight_Joint_Water_Glint_Experiment_Setup.md"

WATER_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A")
CORE_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07")
SWIR_BANDS = ("B11", "B12")
ALL_BANDS = WATER_BANDS + SWIR_BANDS
WAVELENGTHS = np.array([492., 560., 665., 704., 740., 783., 833., 865.])
WEIGHTS = np.array([1., 1., 1., 1., 1., 1., .5, .5])
NOISE_SIGMAS = (0.00030, 0.00080)
ALPHA_TRUTH = (-0.002, 0.0, 0.0005, 0.0010, 0.0020, 0.0040)
REPEATS = 10
RNG = np.random.default_rng(20260911)

archive.S2_BAND_ID.update({"B05": 4, "B06": 5, "B07": 6, "B8A": 8})
plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
                     "legend.fontsize": 9, "figure.dpi": 120})


def mkdirs() -> None:
    for name in (
        "00_Experiment_Setup", "01_Scene_Audit", "02_Multiband_Inputs",
        "03_Glint_ACOLITE_Cycle0", "04_Hedley_Comparison", "05_B12_Comparison",
        "06_HydroLight_Library", "07_Synthetic_Validation", "08_ROI_Joint_Inversion",
        "09_Residual_Glint_Test", "10_Candidate_Cycle1", "11_Final_QC",
        "12_Method_Comparison", "13_Figures", "14_Report"
    ):
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(SETUP, OUTPUT / "00_Experiment_Setup" / SETUP.name)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def read_tif(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(1).astype(float), src.profile.copy()


def read_raw_to_profile(path: Path, profile: dict, method: str) -> np.ndarray:
    """Read a Sentinel-2 JP2 directly with rasterio and align it to the saved 20 m crop."""
    out = np.zeros((profile["height"], profile["width"]), dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=profile["transform"], dst_crs=profile["crs"],
            src_nodata=0, dst_nodata=0,
            resampling={"average": Resampling.average, "bilinear": Resampling.bilinear,
                        "nearest": Resampling.nearest}[method],
        )
    return out


def write_tif(path: Path, arr: np.ndarray, profile: dict, description: str, unit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", predictor=3)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype("float32"), 1)
        dst.set_band_description(1, description)
        dst.update_tags(units=unit, AREA_OR_POINT="Area")


def weighted_rmse(diff: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(WEIGHTS * diff * diff, axis=-1) / WEIGHTS.sum())


def spectral_similarity(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 0 else np.nan


def convolve_case(wave: np.ndarray, rrs: np.ndarray, platform: str) -> dict:
    rsr = hydro.read_rsr(platform)
    return {band: hydro.convolve(wave, rrs, rsr[band.replace("0", "", 1)])
            for band in WATER_BANDS}


def build_hydrolight_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconvolve existing full spectra to B02--B8A; no interpolation from 3 bands."""
    rows = []
    manifest = pd.read_csv(HL_LUT / "hydrolight_run_manifest.csv")
    for _, run in manifest.iterrows():
        mfile = Path(run.result_dir) / f"M{run.root_name}.txt"
        wave, rrs = hydro.parse_rrs(mfile)
        for platform in ("S2A", "S2B", "S2C"):
            rows.append({
                "case_id": f"LUT_{run.root_name}_{platform}", "platform": platform,
                "sza": float(run.SZA_deg), "background": str(run.background),
                "chl": float(run.Chl_a_mg_m3), "cdom": float(run["a_CDOM_440_m-1"]),
                "tsm": float(run.TSM_g_m3), "wind": float(run.wind_m_s),
                "source_mfile": str(mfile), **convolve_case(wave, rrs, platform)
            })
    library = pd.DataFrame(rows)
    library.to_csv(OUTPUT / "06_HydroLight_Library" / "model_library_B02_B8A.csv", index=False)

    truths = []
    pattern = re.compile(r"SZA(?P<sza>[\d.]+)_C(?P<chl>[\d.]+)_D(?P<cdom>[\d.]+)_T(?P<tsm>[\d.]+)")
    for folder in sorted((HL_TRUTH / "results").iterdir()):
        match = pattern.fullmatch(folder.name)
        mfiles = list(folder.glob("M*.txt"))
        if not match or not mfiles:
            continue
        values = {k: float(v) for k, v in match.groupdict().items()}
        wave, rrs = hydro.parse_rrs(mfiles[0])
        for platform in ("S2A", "S2B", "S2C"):
            truths.append({
                "truth_id": f"{folder.name}_{platform}", "platform": platform,
                **values, "wind": 5.0, "source_mfile": str(mfiles[0]),
                **convolve_case(wave, rrs, platform)
            })
    truth = pd.DataFrame(truths)
    truth.to_csv(OUTPUT / "06_HydroLight_Library" / "withheld_truth_cases_B02_B8A.csv", index=False)
    manifest_out = {
        "library_cases": int(len(library)), "unique_hydrolight_runs": int(len(manifest)),
        "withheld_truth_cases": int(len(truth)),
        "bands": list(WATER_BANDS),
        "note": "Values were reconvolved directly from stored 400-900 nm HydroLight output using platform-specific MSI response functions.",
    }
    write_json(OUTPUT / "06_HydroLight_Library" / "library_manifest.json", manifest_out)
    return library, truth


def optical_terms_all(meta: dict, aot: float) -> tuple[pd.DataFrame, dict]:
    sensor, rsrd = archive.sensor_info(meta["platform"])
    lutdw = archive.ac.aerlut.import_luts(
        pressures=[1013], base_luts=["ACOLITE-LUT-202110-MOD2"], sensor=sensor,
        lut_par=["ttot"], return_lut_array=True, add_rsky=False)
    lut = list(lutdw)[0]
    raa, vza, sza = meta["relative_azimuth_angle"], meta["view_zenith_angle"], meta["sun_zenith_angle"]
    xi = [1013., raa, vza, sza]
    sza_r, vza_r, raa_r = np.radians([sza, vza, raa])
    omega = np.arccos(np.cos(sza_r) * np.cos(vza_r) +
                      np.sin(sza_r) * np.sin(vza_r) * np.cos(raa_r)) / 2
    refri = archive.ac.shared.wopp.refri()
    refri_sen = archive.ac.shared.rsr_convolute_dict(refri["wave"] / 1000, refri["n"], rsrd["rsr"])
    band_map = {"B02": "2", "B03": "3", "B04": "4", "B05": "5", "B06": "6",
                "B07": "7", "B08": "8", "B8A": "8A", "B11": "11", "B12": "12"}
    mus, muv = np.cos(sza_r), np.cos(vza_r)
    records = []
    for name, token in band_map.items():
        ttot = float(lutdw[lut]["rgi"][token]((xi[0], lutdw[lut]["ipd"]["ttot"], xi[1], xi[2], xi[3], aot)))
        trans = float(np.exp(-ttot / muv) * np.exp(-ttot / mus))
        fresnel = float(archive.ac.ac.sky_refl(omega, n_w=refri_sen[token]))
        records.append({"band": name, "wavelength_nm": float(rsrd["wave_nm"][token]),
                        "two_way_transmittance": trans, "fresnel_reflectance": fresnel,
                        "AOT_550": aot})
    table = pd.DataFrame(records)
    lookup = table.set_index("band")
    factors = {}
    for band in WATER_BANDS:
        for swir in SWIR_BANDS:
            factors[(band, swir)] = float(
                lookup.loc[band, "two_way_transmittance"] / lookup.loc[swir, "two_way_transmittance"] *
                lookup.loc[band, "fresnel_reflectance"] / lookup.loc[swir, "fresnel_reflectance"])
    return table, factors


def prepare_scene(scene: str, safe: Path, roi_positions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    meta = archive.read_metadata(safe)
    valid, profile = read_tif(SOURCE / "01_Masks" / scene / "valid_water_mask.tif")
    land, _ = read_tif(SOURCE / "01_Masks" / scene / "land_nonwater_mask.tif")
    valid = valid > 0
    bands = {}
    for band in ALL_BANDS:
        resolution = 10 if band == "B08" else 20
        method = "average" if band == "B08" else "bilinear"
        raw = read_raw_to_profile(archive.find_band(safe, band, resolution), profile, method)
        bands[band] = archive.reflectance(raw, meta, band)
        write_tif(OUTPUT / "02_Multiband_Inputs" / scene / f"{band}_BOA_20m.tif",
                  np.where(valid, bands[band], np.nan), profile,
                  f"Masked Sentinel-2 L2A {band} BOA reflectance", "dimensionless")

    use_b11, _ = read_tif(SOURCE / "06_ACOLITE" / "SWIR_Selection" / scene / "SWIR_selection_B11_is_1.tif")
    rhog, _ = read_tif(SOURCE / "06_ACOLITE" / "Rhog_Reference" / scene / "rhog_ref.tif")
    aot_info = json.loads((SOURCE / "06_ACOLITE" / "AOT" / scene / "AOT_statistics.json").read_text())
    terms, factors = optical_terms_all(meta, float(aot_info["water_median_used"]))
    cycle0_dir = OUTPUT / "03_Glint_ACOLITE_Cycle0" / scene
    cycle0_dir.mkdir(parents=True, exist_ok=True)
    terms.to_csv(cycle0_dir / "multiband_optical_terms.csv", index=False)
    pd.DataFrame([{"band": b, "factor_from_B11": factors[(b, "B11")],
                   "factor_from_B12": factors[(b, "B12")]} for b in WATER_BANDS]).to_csv(
        cycle0_dir / "multiband_glint_factors.csv", index=False)

    before = np.stack([bands[b] for b in WATER_BANDS], axis=-1)
    selected_b11 = use_b11 > 0.5
    glint = np.stack([np.where(selected_b11, factors[(b, "B11")] * rhog,
                              factors[(b, "B12")] * rhog) for b in WATER_BANDS], axis=-1)
    cycle0 = before - glint
    good = valid & np.all(np.isfinite(before), axis=-1) & np.all(np.isfinite(glint), axis=-1)
    before[~good] = np.nan; glint[~good] = np.nan; cycle0[~good] = np.nan
    for i, band in enumerate(WATER_BANDS):
        write_tif(OUTPUT / "03_Glint_ACOLITE_Cycle0" / scene / f"estimated_glint_{band}.tif",
                  glint[..., i], profile, f"Estimated Glint-ACOLITE component {band}", "dimensionless")
        write_tif(OUTPUT / "03_Glint_ACOLITE_Cycle0" / scene / f"Cycle0_{band}.tif",
                  cycle0[..., i], profile, f"Cycle-0 corrected BOA reflectance {band}", "dimensionless")

    audit = []
    for i, band in enumerate(("B02", "B03", "B04")):
        old, _ = read_tif(SOURCE / "06_ACOLITE" / "Corrected" / scene / f"{band}_corrected.tif")
        d = np.abs(old - cycle0[..., i]); values = d[np.isfinite(d)]
        audit.append({"band": band, "maximum_abs_difference": float(values.max()),
                      "median_abs_difference": float(np.median(values))})
    pd.DataFrame(audit).to_csv(OUTPUT / "03_Glint_ACOLITE_Cycle0" / scene / "visible_reproduction_audit.csv", index=False)

    roi_rows = []
    for _, r in roi_positions.iterrows():
        row, col = int(r.row), int(r.col)
        mask = np.zeros(valid.shape, bool)
        mask[row-3:row+4, col-3:col+4] = True
        ok = mask & good
        if ok.sum() != 49:
            continue
        entry = {"scene": scene, "roi": r.roi, "stratum": r.stratum, "row": row, "col": col,
                 "valid_pixels": int(ok.sum()), "rhog_ref_mean": float(np.mean(rhog[ok])),
                 "B11_mean": float(np.mean(bands["B11"][ok])), "B12_mean": float(np.mean(bands["B12"][ok]))}
        for i, band in enumerate(WATER_BANDS):
            entry[f"before_{band}"] = float(np.mean(before[..., i][ok]))
            entry[f"cycle0_{band}"] = float(np.mean(cycle0[..., i][ok]))
            entry[f"glint0_{band}"] = float(np.mean(glint[..., i][ok]))
        roi_rows.append(entry)

    scene_shape = np.average(np.vstack([
        np.array([factors[(b, "B11")] for b in WATER_BANDS]),
        np.array([factors[(b, "B12")] for b in WATER_BANDS])]), axis=0,
        weights=[float(np.mean(selected_b11[good])), float(np.mean(~selected_b11[good]))])
    scene_shape /= np.max(scene_shape)
    data = {"scene": scene, "meta": meta, "profile": profile, "valid": good, "land": land > 0,
            "bands": bands, "before": before, "cycle0": cycle0, "glint0": glint,
            "rhog": rhog, "gshape": scene_shape}
    return data, pd.DataFrame(roi_rows)


def fit_joint(obs: np.ndarray, g: np.ndarray, candidates: pd.DataFrame) -> dict:
    models = np.pi * candidates[list(WATER_BANDS)].to_numpy(float)
    d0 = obs[None, :] - models
    rmse0 = weighted_rmse(d0)
    denom = float(np.sum(WEIGHTS * g * g))
    alpha = np.maximum(0., np.sum(WEIGHTS * g * d0, axis=1) / denom)
    d1 = d0 - alpha[:, None] * g[None, :]
    rmse1 = weighted_rmse(d1)
    i0, i1 = int(np.argmin(rmse0)), int(np.argmin(rmse1))
    residual = obs - models[i1]
    improvement = 100 * (rmse0[i0] - rmse1[i1]) / max(rmse0[i0], 1e-12)
    order = np.argsort(rmse1)[:5]
    return {"i0": i0, "i1": i1, "rmse0": float(rmse0[i0]), "rmse1": float(rmse1[i1]),
            "alpha": float(alpha[i1]), "similarity": spectral_similarity(residual, g),
            "improvement": float(improvement), "model": models[i1],
            "prediction": models[i1] + alpha[i1] * g, "residual": residual,
            "top5": order.tolist()}


def synthetic_validation(library: pd.DataFrame, truth: pd.DataFrame, templates: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    for ti, t in truth.iterrows():
        candidates = library[(library.platform == t.platform) & (library.sza == t.sza)].reset_index(drop=True)
        water = np.pi * t[list(WATER_BANDS)].to_numpy(float)
        available = [x for x in templates.values() if x[0] == t.platform and abs(x[1] - t.sza) <= 7.5]
        g = (available[ti % len(available)][2] if available else np.median([x[2] for x in templates.values()], axis=0))
        for alpha_true in ALPHA_TRUTH:
            for sigma in NOISE_SIGMAS:
                band_sigma = sigma * np.array([1., .9, .9, 1., 1.1, 1.2, 1.3, 1.3])
                for repeat in range(REPEATS):
                    obs = water + alpha_true * g + RNG.normal(0., band_sigma)
                    fit = fit_joint(obs, g, candidates)
                    selected = candidates.iloc[fit["i1"]]
                    rows.append({"truth_id": t.truth_id, "platform": t.platform, "sza": t.sza,
                                 "alpha_true": alpha_true, "noise_sigma": sigma, "repeat": repeat,
                                 "alpha_est": fit["alpha"], "rmse_water": fit["rmse0"],
                                 "rmse_joint": fit["rmse1"], "improvement_percent": fit["improvement"],
                                 "similarity": fit["similarity"], "truth_tsm": t.tsm,
                                 "selected_tsm": selected.tsm, "selected_background": selected.background})
    frame = pd.DataFrame(rows)
    validation = frame[frame.alpha_true == 0]
    lod = float(np.quantile(validation.alpha_est, .95))
    improvement_limit = float(np.quantile(validation.improvement_percent, .95))
    positive_cal = frame[frame.alpha_true >= .001]
    similarity_limit = float(max(0.5, np.quantile(positive_cal.similarity.dropna(), .05)))
    frame["detected"] = ((frame.alpha_est > lod) &
                         (frame.improvement_percent > improvement_limit) &
                         (frame.similarity > similarity_limit))
    fp = float(frame[frame.alpha_true == 0].detected.mean())
    over_fp = float(frame[frame.alpha_true < 0].detected.mean())
    sensitivity = float(frame[frame.alpha_true >= .001].detected.mean())
    error = float(np.median(np.abs(frame[frame.alpha_true >= 0].alpha_est -
                                   frame[frame.alpha_true >= 0].alpha_true)))
    metrics = {"component_detection_limit_reflectance": lod,
               "minimum_improvement_percent": improvement_limit,
               "minimum_shape_similarity": similarity_limit,
               "no_glint_false_positive_rate": fp,
               "overremoval_false_positive_rate": over_fp,
               "sensitivity_alpha_ge_0.001": sensitivity,
               "median_alpha_abs_error": error,
               "pass_limits": {"false_positive_max": .05, "overremoval_false_positive_max": .05,
                               "sensitivity_min": .80, "median_alpha_abs_error_max": .0005}}
    metrics["phase_A_pass"] = bool(fp <= .05 and over_fp <= .05 and sensitivity >= .80 and error <= .0005)
    return frame, metrics


def fit_real_rois(rois: pd.DataFrame, library: pd.DataFrame, scene_data: dict, thresholds: dict) -> pd.DataFrame:
    rows = []
    for _, r in rois.iterrows():
        data = scene_data[r.scene]
        g = data["gshape"]
        sza_grid = np.sort(library[library.platform == data["meta"]["platform"]].sza.unique())
        sza = float(sza_grid[np.argmin(abs(sza_grid - data["meta"]["sun_zenith_angle"]))])
        candidates = library[(library.platform == data["meta"]["platform"]) & (library.sza == sza)].reset_index(drop=True)
        obs = r[[f"cycle0_{b}" for b in WATER_BANDS]].to_numpy(float)
        fit = fit_joint(obs, g, candidates)
        selected = candidates.iloc[fit["i1"]]
        row = {"scene": r.scene, "roi": r.roi, "stratum": r.stratum,
               "actual_sza": data["meta"]["sun_zenith_angle"], "tested_sza": sza,
               "background": selected.background, "chl": selected.chl, "cdom": selected.cdom,
               "tsm": selected.tsm, "water_only_rmse": fit["rmse0"], "joint_rmse": fit["rmse1"],
               "improvement_percent": fit["improvement"], "alpha": fit["alpha"],
               "shape_similarity": fit["similarity"], "rhog_ref_mean": r.rhog_ref_mean}
        row["detected"] = bool(fit["alpha"] > thresholds["component_detection_limit_reflectance"] and
                               fit["improvement"] > thresholds["minimum_improvement_percent"] and
                               fit["similarity"] > thresholds["minimum_shape_similarity"])
        for i, band in enumerate(WATER_BANDS):
            row[f"cycle0_{band}"] = obs[i]
            row[f"model_{band}"] = fit["model"][i]
            row[f"joint_{band}"] = fit["prediction"][i]
            row[f"residual_{band}"] = fit["residual"][i]
        for rank, idx in enumerate(fit["top5"], start=1):
            c = candidates.iloc[idx]
            row[f"rank{rank}_case"] = c.case_id
            row[f"rank{rank}_background"] = c.background
            row[f"rank{rank}_tsm"] = c.tsm
        rows.append(row)
    return pd.DataFrame(rows)


def scene_decisions(real: pd.DataFrame, phase_pass: bool) -> pd.DataFrame:
    rows = []
    for scene, g in real.groupby("scene"):
        low, high = g[g.stratum == "LOW"], g[g.stratum == "HIGH"]
        corr = spearmanr(g.alpha, g.rhog_ref_mean).statistic if len(g) > 2 else np.nan
        high_detected = int(high.detected.sum()); low_detected = int(low.detected.sum())
        separation = float(high.alpha.median() - low.alpha.median())
        real_gate = bool(high_detected >= 5 and low_detected <= 2 and np.isfinite(corr) and corr >= .30 and separation > 0)
        apply = bool(phase_pass and real_gate)
        rows.append({"scene": scene, "phase_A_pass": phase_pass,
                     "high_ROIs_detected": high_detected, "low_ROIs_detected": low_detected,
                     "median_alpha_high_minus_low": separation,
                     "spearman_alpha_vs_rhog": float(corr), "real_scene_gate": real_gate,
                     "cycle1_authorized": apply,
                     "decision": "CANDIDATE_CYCLE1" if apply else "RETAIN_CYCLE0"})
    return pd.DataFrame(rows)


def pixel_joint(data: dict, library: pd.DataFrame, threshold: dict, authorized: bool) -> dict:
    cycle0, valid, g = data["cycle0"], data["valid"], data["gshape"]
    alpha_map = np.full(valid.shape, np.nan, np.float32)
    detected = np.zeros(valid.shape, bool)
    if not authorized:
        return {"alpha": alpha_map, "detected": detected, "candidate": cycle0.copy()}
    sza_grid = np.sort(library[library.platform == data["meta"]["platform"]].sza.unique())
    sza = float(sza_grid[np.argmin(abs(sza_grid - data["meta"]["sun_zenith_angle"]))])
    candidates = library[(library.platform == data["meta"]["platform"]) & (library.sza == sza)].reset_index(drop=True)
    models = np.pi * candidates[list(WATER_BANDS)].to_numpy(float)
    coords = np.argwhere(valid)
    denom = float(np.sum(WEIGHTS * g * g))
    for start in range(0, len(coords), 4000):
        xy = coords[start:start+4000]
        obs = cycle0[xy[:, 0], xy[:, 1]]
        d0 = obs[:, None, :] - models[None, :, :]
        rm0_all = weighted_rmse(d0)
        i0 = np.argmin(rm0_all, axis=1); rm0 = rm0_all[np.arange(len(xy)), i0]
        alpha_all = np.maximum(0., np.sum(WEIGHTS * g * d0, axis=2) / denom)
        d1 = d0 - alpha_all[..., None] * g
        rm1_all = weighted_rmse(d1)
        i1 = np.argmin(rm1_all, axis=1); rm1 = rm1_all[np.arange(len(xy)), i1]
        a = alpha_all[np.arange(len(xy)), i1]
        residual = obs - models[i1]
        sim = np.sum(residual * g, axis=1) / np.maximum(np.linalg.norm(residual, axis=1) * np.linalg.norm(g), 1e-12)
        imp = 100 * (rm0 - rm1) / np.maximum(rm0, 1e-12)
        keep = ((a > threshold["component_detection_limit_reflectance"]) &
                (imp > threshold["minimum_improvement_percent"]) &
                (sim > threshold["minimum_shape_similarity"]))
        alpha_map[xy[:, 0], xy[:, 1]] = a
        detected[xy[:, 0], xy[:, 1]] = keep
    detected = binary_opening(detected, structure=np.ones((3, 3), bool))
    applied = np.where(detected, alpha_map, 0.)
    candidate = cycle0 - applied[..., None] * g
    candidate[~valid] = np.nan
    return {"alpha": alpha_map, "detected": detected, "candidate": candidate}


def rgb(stack: np.ndarray) -> np.ndarray:
    out = np.stack([stack[..., 2], stack[..., 1], stack[..., 0]], axis=-1)
    return np.power(np.clip((np.nan_to_num(out) + .01) / .16, 0, 1), .72)


def scene_figures(scene: str, data: dict, rois: pd.DataFrame, real: pd.DataFrame,
                  pixel: dict, decision: dict) -> None:
    folder = OUTPUT / "13_Figures" / scene; folder.mkdir(parents=True, exist_ok=True)
    final = pixel["candidate"] if decision["final_decision"] == "RETAIN_CYCLE1" else data["cycle0"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, arr, title in zip(axes, (data["before"], data["cycle0"], final),
                              ("Before correction", "Glint-ACOLITE Cycle 0", "Final retained image")):
        ax.imshow(rgb(arr)); ax.set_title(title); ax.set_xlabel("Image column (20 m pixel)"); ax.set_ylabel("Image row (20 m pixel)")
    fig.suptitle(f"{scene} · identical crop and RGB stretch")
    fig.savefig(folder / "01_before_cycle0_final.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.imshow(rgb(data["before"]))
    colors = {"LOW": "#2563eb", "MEDIUM": "#f59e0b", "HIGH": "#dc2626"}
    for _, r in rois.iterrows():
        ax.add_patch(plt.Rectangle((r.col-3, r.row-3), 7, 7, fill=False, lw=1.5, ec=colors[r.stratum]))
        ax.text(r.col+4, r.row, r.roi, color="white", fontsize=7,
                bbox=dict(facecolor=colors[r.stratum], edgecolor="none", pad=1))
    ax.set(title=f"{scene} · 30 fully contained offshore ROIs", xlabel="Image column (20 m pixel)", ylabel="Image row (20 m pixel)")
    handles = [plt.Line2D([0], [0], color=c, lw=3, label=f"{s}: 10 ROIs") for s, c in colors.items()]
    ax.legend(handles=handles, loc="lower left")
    fig.savefig(folder / "02_roi_map.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True, sharey=True)
    for ax, stratum in zip(axes, ("LOW", "MEDIUM", "HIGH")):
        subset = rois[rois.stratum == stratum]
        for _, r in subset.iterrows():
            ax.plot(WAVELENGTHS, r[[f"before_{b}" for b in WATER_BANDS]], color="#64748b", alpha=.28, lw=1)
            ax.plot(WAVELENGTHS, r[[f"cycle0_{b}" for b in WATER_BANDS]], color="#0f766e", alpha=.35, lw=1)
        ax.plot(WAVELENGTHS, subset[[f"before_{b}" for b in WATER_BANDS]].median(), "o-", color="#334155", lw=2.4, label="Before median")
        ax.plot(WAVELENGTHS, subset[[f"cycle0_{b}" for b in WATER_BANDS]].median(), "o-", color="#0f766e", lw=2.4, label="Cycle-0 median")
        ax.set(title=f"{stratum} apparent glint · 10 ROIs", xlabel="Wavelength (nm)", ylabel="BOA reflectance (dimensionless)")
        ax.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle(f"{scene} · all 30 multi-band spectra before and after Glint-ACOLITE")
    fig.savefig(folder / "03_all_roi_before_cycle0.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True, sharey=True)
    for ax, stratum in zip(axes, ("LOW", "MEDIUM", "HIGH")):
        subset = real[real.stratum == stratum]
        for _, r in subset.iterrows():
            ax.plot(WAVELENGTHS, r[[f"cycle0_{b}" for b in WATER_BANDS]], color="#0f766e", alpha=.25)
            ax.plot(WAVELENGTHS, r[[f"model_{b}" for b in WATER_BANDS]], color="#d97706", alpha=.25, ls="--")
        ax.plot(WAVELENGTHS, subset[[f"cycle0_{b}" for b in WATER_BANDS]].median(), "o-", color="#0f766e", lw=2.4, label="Cycle-0 median")
        ax.plot(WAVELENGTHS, subset[[f"model_{b}" for b in WATER_BANDS]].median(), "s--", color="#d97706", lw=2.4, label="HydroLight median")
        ax.set(title=f"{stratum} · satellite versus water model", xlabel="Wavelength (nm)", ylabel="BOA reflectance (dimensionless)"); ax.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle(f"{scene} · all ROI-specific HydroLight matches")
    fig.savefig(folder / "04_all_roi_hydrolight_matches.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for _, r in real.iterrows():
        axes[0].plot(WAVELENGTHS, r[[f"residual_{b}" for b in WATER_BANDS]], alpha=.28,
                     color={"LOW":"#2563eb", "MEDIUM":"#f59e0b", "HIGH":"#dc2626"}[r.stratum])
    axes[0].axhline(0, color="black", lw=1); axes[0].set(title="Cycle 0 minus selected HydroLight water", xlabel="Wavelength (nm)", ylabel="Model residual (dimensionless)")
    axes[0].grid(alpha=.25)
    for stratum, color in colors.items():
        q = real[real.stratum == stratum]
        axes[1].scatter(q.rhog_ref_mean, q.alpha, color=color, label=stratum)
    axes[1].axhline(0, color="black", lw=1); axes[1].set(title="Fitted residual-glint magnitude", xlabel="Initial B11/B12 glint reference (dimensionless)", ylabel="Fitted alpha (dimensionless)")
    axes[1].grid(alpha=.25); axes[1].legend()
    fig.suptitle(f"{scene} · residual evidence; difference alone is not proof of glint")
    fig.savefig(folder / "05_residual_evidence.png", dpi=170); plt.close(fig)


def method_comparison(scene: str, data: dict, rois: pd.DataFrame) -> pd.DataFrame:
    methods = {"Before": [], "Glint-ACOLITE": []}
    for band in ("B02", "B03", "B04"):
        methods["Before"].append(data["before"][..., WATER_BANDS.index(band)])
        methods["Glint-ACOLITE"].append(data["cycle0"][..., WATER_BANDS.index(band)])
    for label, folder in (("Hedley", "Hedley"), ("B12 subtraction", "B12_SWIR")):
        methods[label] = []
        for band in ("B02", "B03", "B04"):
            base = SOURCE / "10_Rrs_Approx" / folder / scene
            candidates = (base / f"Rrs_approx_{band}.tif", base / f"{band}_Rrs_approx_sr-1.tif")
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                raise FileNotFoundError(f"No retained {label} Rrs product for {scene} {band}")
            arr, _ = read_tif(path)
            methods[label].append(arr * np.pi)
    rows = []
    for method, arrays in methods.items():
        stack = np.stack(arrays, axis=-1)
        good = data["valid"] & np.all(np.isfinite(stack), axis=-1)
        neg = float(100 * np.mean(np.any(stack[good] < 0, axis=1)))
        corr = float(np.corrcoef(data["before"][..., :3][good].ravel(), stack[good].ravel())[0, 1])
        for stratum in ("LOW", "MEDIUM", "HIGH"):
            vals = []
            for _, r in rois[rois.stratum == stratum].iterrows():
                vals.append(stack[int(r.row)-3:int(r.row)+4, int(r.col)-3:int(r.col)+4].mean(axis=(0,1)))
            rows.append({"scene": scene, "method": method, "stratum": stratum,
                         "mean_B02_B04": float(np.mean(vals)), "negative_pixel_percent": neg,
                         "spatial_correlation_with_before": corr})
    return pd.DataFrame(rows)


def validation_figure(frame: pd.DataFrame, metrics: dict) -> None:
    summary = frame.groupby(["alpha_true", "noise_sigma"]).agg(
        detection=("detected", "mean"), alpha_est=("alpha_est", "median"),
        alpha_p05=("alpha_est", lambda x: np.quantile(x,.05)),
        alpha_p95=("alpha_est", lambda x: np.quantile(x,.95))).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for sigma, g in summary.groupby("noise_sigma"):
        axes[0].plot(g.alpha_true, 100*g.detection, "o-", label=f"noise sigma={sigma:.5f}")
        axes[1].errorbar(g.alpha_true, g.alpha_est,
                         yerr=[g.alpha_est-g.alpha_p05, g.alpha_p95-g.alpha_est], marker="o", capsize=3, label=f"noise sigma={sigma:.5f}")
    axes[0].axhline(80, color="#dc2626", ls="--", label="required sensitivity 80%")
    axes[0].set(xlabel="Known added glint alpha (dimensionless)", ylabel="Detection rate (%)", title="Known-glint detection")
    axes[1].plot([min(ALPHA_TRUTH), max(ALPHA_TRUTH)], [min(ALPHA_TRUTH), max(ALPHA_TRUTH)], "k--", label="1:1 recovery")
    axes[1].set(xlabel="Known alpha (dimensionless)", ylabel="Recovered alpha (dimensionless)", title="Glint-magnitude recovery")
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    status = "PASS" if metrics["phase_A_pass"] else "FAIL"
    fig.suptitle(f"Phase A {status} · multi-band held-out HydroLight validation")
    fig.savefig(OUTPUT / "13_Figures" / "00_synthetic_validation.png", dpi=180); plt.close(fig)


def build_html(summary: dict, audit: pd.DataFrame, decisions: pd.DataFrame) -> Path:
    options, panels = [], []
    for idx, scene in enumerate(summary["included_scenes"]):
        d = decisions[decisions.scene == scene].iloc[0]
        figdir = Path("../13_Figures") / scene
        detector_note = (
            f"The candidate detector flagged {int(d.high_ROIs_detected)}/10 high-apparent-glint ROIs "
            f"and {int(d.low_ROIs_detected)}/10 low-control ROIs. "
            "Because the controls were also flagged, this is not specific evidence of residual glint."
        )
        options.append(f"<option value='{scene}'>{scene}</option>")
        images = "".join(f"<h3>{title}</h3><a href='{(figdir/name).as_posix()}' target='_blank'><img src='{(figdir/name).as_posix()}' alt='{title}'></a>" for title,name in [
          ("1. Before, Cycle 0 and final", "01_before_cycle0_final.png"),
          ("2. Thirty offshore ROI locations", "02_roi_map.png"),
          ("3. All 30 spectra before and after", "03_all_roi_before_cycle0.png"),
          ("4. All 30 HydroLight matches", "04_all_roi_hydrolight_matches.png"),
          ("5. Residual-glint evidence", "05_residual_evidence.png")])
        panels.append(
            f"<section class='scene' id='s-{scene}' style='display:{'block' if idx==0 else 'none'}'>"
            f"<h2>{scene}</h2><p class='decision'><b>Final decision:</b> {html.escape(str(d.final_decision))}. "
            f"Phase A {'passed' if d.phase_A_pass else 'failed'}. {detector_note}</p>"
            f"{images}</section>"
        )
    aud = "".join(f"<tr><td>{r.scene}</td><td>{r.status}</td><td>{html.escape(str(r.reason))}</td></tr>" for _,r in audit.iterrows())
    metrics = summary["synthetic_validation"]
    validation_class = "decision" if metrics["phase_A_pass"] else "decision fail"
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>Tidung joint HydroLight-glint experiment</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f3f6f8;color:#102a43;font:16px/1.55 Arial,sans-serif}}
    main,header{{max-width:1180px;margin:auto}}header{{padding:28px 24px}}main{{background:white;padding:25px 32px 60px;border:1px solid #ccd9e2}}
    h1{{margin:0;font-size:31px}}h2{{border-bottom:1px solid #ccd9e2;padding-bottom:7px;margin-top:35px}}h3{{font-size:19px;margin-top:25px}}
    .lead{{font-size:18px;color:#496579}}.flow{{max-width:760px;margin:20px auto;text-align:center}}.flow div{{border:2px solid #1769e0;padding:11px;margin:0;background:#f7fbff}}.flow b,.flow span{{display:block}}.flow i{{display:block;font-style:normal;color:#1769e0;font-size:24px}}
    .decision{{padding:12px;border-left:5px solid #087f6a;background:#eef9f6}}.fail{{border-left-color:#b42318;background:#fff2f0}}select{{padding:10px;font:inherit;font-weight:bold;min-width:320px}}
    img{{width:100%;height:auto;border:1px solid #ccd9e2}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #ccd9e2;padding:7px;text-align:left}}th{{background:#15334e;color:white}}
    </style></head><body><header><h1>Tidung residual-glint experiment</h1><p class='lead'>Multi-band Sentinel-2 L2A · HydroLight water model · validation-gated Cycle-1 test</p></header><main>
    <h2>1. Goal and model</h2><p>Separate physically plausible water reflectance from a possible residual surface-glint component. A HydroLight mismatch is not automatically called glint.</p>
    <pre>Cycle 0 BOA = pi × HydroLight Rrs(water variables) + alpha × glint shape + remaining error</pre>
    <h2>2. Processing flow</h2><div class='flow'><div><b>Input</b><span>L2A B02-B8A; B11/B12; masks; geometry</span></div><i>↓</i><div><b>Initial corrections</b><span>Glint-ACOLITE, Hedley and B12 comparison</span></div><i>↓</i><div><b>Known-answer validation</b><span>HydroLight water + known glint + noise → retrieve water and glint</span></div><i>↓</i><div><b>Real joint inversion</b><span>30 deep-water ROIs per accepted scene</span></div><i>↓</i><div><b>Decision</b><span>Apply Cycle 1 only if simulation, spectral, spatial and preservation gates pass</span></div></div>
    <h2>3. Synthetic validation</h2><p class='{validation_class}'><b>Phase A: {'PASS' if metrics['phase_A_pass'] else 'FAIL'}.</b> No-glint false positives {100*metrics['no_glint_false_positive_rate']:.1f}%; over-removal false positives {100*metrics['overremoval_false_positive_rate']:.1f}%; sensitivity {100*metrics['sensitivity_alpha_ge_0.001']:.1f}%; median alpha error {metrics['median_alpha_abs_error']:.6f}.</p>
    <a href='../13_Figures/00_synthetic_validation.png' target='_blank'><img src='../13_Figures/00_synthetic_validation.png' alt='Synthetic validation'></a>
    <h2>4. Scene results</h2><label for='scene'>Select date</label><br><select id='scene'>{''.join(options)}</select>{''.join(panels)}
    <h2>5. Scene eligibility</h2><table><thead><tr><th>Scene</th><th>Status</th><th>Reason</th></tr></thead><tbody>{aud}</tbody></table>
    <h2>6. Scientific conclusion</h2><p>{html.escape(summary['scientific_conclusion'])}</p>
    </main><script>const q=document.getElementById('scene');q.onchange=()=>{{document.querySelectorAll('.scene').forEach(x=>x.style.display='none');document.getElementById('s-'+q.value).style.display='block';}};</script></body></html>"""
    out = OUTPUT / "14_Report" / "Tidung_Joint_HydroLight_Glint_Multiband_Experiment.html"
    out.write_text(doc, encoding="utf-8")
    return out


def main() -> None:
    mkdirs()
    inventory = pd.read_csv(SOURCE / "00_Scene_Selection" / "scene_inventory.csv", dtype={"date": str})
    previous_audit = pd.read_csv(PREVIOUS / "00_scene_eligibility_audit.csv")
    included = previous_audit.loc[previous_audit.status == "INCLUDED", "scene"].tolist()
    roi_positions = pd.read_csv(PREVIOUS / "00_roi_inputs.csv")
    previous_audit.to_csv(OUTPUT / "01_Scene_Audit" / "scene_eligibility_audit.csv", index=False)

    library, truth = build_hydrolight_tables()
    scene_data, roi_frames, reproduction = {}, [], []
    for number, scene in enumerate(included, 1):
        print(f"PREPARE {number}/{len(included)} {scene}", flush=True)
        date, platform = scene.split("_")
        record = inventory[(inventory.date == date) & (inventory.platform == platform)].iloc[0]
        data, roi = prepare_scene(scene, Path(record.safe), roi_positions[roi_positions.scene == scene])
        scene_data[scene] = data; roi_frames.append(roi)
        check = pd.read_csv(OUTPUT / "03_Glint_ACOLITE_Cycle0" / scene / "visible_reproduction_audit.csv")
        reproduction.append({"scene": scene, "maximum_difference": float(check.maximum_abs_difference.max())})
    rois = pd.concat(roi_frames, ignore_index=True)
    rois.to_csv(OUTPUT / "08_ROI_Joint_Inversion" / "all_30ROI_multiband_inputs.csv", index=False)
    pd.DataFrame(reproduction).to_csv(OUTPUT / "03_Glint_ACOLITE_Cycle0" / "visible_reproduction_audit_all_scenes.csv", index=False)

    templates = {scene: (d["meta"]["platform"], float(d["meta"]["sun_zenith_angle"]), d["gshape"])
                 for scene, d in scene_data.items()}
    synthetic, validation = synthetic_validation(library, truth, templates)
    synthetic.to_csv(OUTPUT / "07_Synthetic_Validation" / "all_known_answer_trials.csv", index=False)
    write_json(OUTPUT / "07_Synthetic_Validation" / "validation_metrics_and_gate.json", validation)
    validation_figure(synthetic, validation)

    real = fit_real_rois(rois, library, scene_data, validation)
    real.to_csv(OUTPUT / "08_ROI_Joint_Inversion" / "all_scene_all_ROI_joint_fits.csv", index=False)
    decisions = scene_decisions(real, validation["phase_A_pass"])
    decisions.to_csv(OUTPUT / "09_Residual_Glint_Test" / "scene_residual_glint_decisions.csv", index=False)

    final_rows, comparisons = [], []
    for number, scene in enumerate(included, 1):
        print(f"FINALIZE {number}/{len(included)} {scene}", flush=True)
        data = scene_data[scene]; d = decisions[decisions.scene == scene].iloc[0].to_dict()
        pix = pixel_joint(data, library, validation, bool(d["cycle1_authorized"]))
        valid = data["valid"]
        cycle0_neg = 100 * np.mean(np.any(data["cycle0"][valid] < 0, axis=1))
        candidate_neg = 100 * np.mean(np.any(pix["candidate"][valid] < 0, axis=1))
        corrected_fraction = 100 * np.mean(pix["detected"][valid])
        corr = np.corrcoef(data["cycle0"][valid].ravel(), pix["candidate"][valid].ravel())[0,1]
        final_pass = bool(d["cycle1_authorized"] and candidate_neg-cycle0_neg <= .10 and corr >= .98 and corrected_fraction > 0)
        final_decision = "RETAIN_CYCLE1" if final_pass else "RETAIN_CYCLE0"
        decisions.loc[decisions.scene == scene, "final_decision"] = final_decision
        final = pix["candidate"] if final_pass else data["cycle0"]
        for i, band in enumerate(WATER_BANDS):
            write_tif(OUTPUT / "10_Candidate_Cycle1" / scene / f"candidate_Cycle1_{band}.tif",
                      pix["candidate"][..., i], data["profile"], f"Candidate Cycle-1 {band}", "dimensionless")
            write_tif(OUTPUT / "11_Final_QC" / scene / f"final_{band}.tif", final[..., i], data["profile"],
                      f"Final retained corrected BOA reflectance {band}", "dimensionless")
        write_tif(OUTPUT / "10_Candidate_Cycle1" / scene / "fitted_alpha.tif", pix["alpha"], data["profile"],
                  "Fitted residual-glint magnitude alpha", "dimensionless")
        write_tif(OUTPUT / "10_Candidate_Cycle1" / scene / "accepted_cycle1_mask.tif",
                  np.where(valid, pix["detected"].astype(float), np.nan), data["profile"],
                  "Accepted residual correction mask", "class: 0 no correction; 1 correction")
        final_rows.append({"scene": scene, "cycle0_negative_percent": cycle0_neg,
                           "candidate_negative_percent": candidate_neg, "corrected_water_percent": corrected_fraction,
                           "cycle0_candidate_spatial_r": float(corr), "final_qc_pass": final_pass,
                           "final_decision": final_decision})
        comparisons.append(method_comparison(scene, data, rois[rois.scene == scene]))
        scene_figures(scene, data, rois[rois.scene == scene], real[real.scene == scene], pix,
                      {**d, "final_decision": final_decision})
    decisions.to_csv(OUTPUT / "09_Residual_Glint_Test" / "scene_residual_glint_decisions.csv", index=False)
    final_qc = pd.DataFrame(final_rows)
    final_qc.to_csv(OUTPUT / "11_Final_QC" / "all_scene_final_QC.csv", index=False)
    comparison = pd.concat(comparisons, ignore_index=True)
    comparison.to_csv(OUTPUT / "12_Method_Comparison" / "all_scene_method_metrics.csv", index=False)

    accepted = int((final_qc.final_decision == "RETAIN_CYCLE1").sum())
    conclusion = (f"The multi-band known-answer validation passed and {accepted} of {len(included)} scenes retained a validated Cycle-1 residual correction."
                  if validation["phase_A_pass"] else
                  f"The multi-band known-answer validation failed its fixed acceptance requirements. Therefore no Cycle-1 subtraction was accepted; Cycle 0 remains final for all {len(included)} analyzed scenes. The HydroLight mismatches remain diagnostic and are not labelled glint.")
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "output_folder": str(OUTPUT), "source_L2A": r"H:\Sentinel-2\Tidung\L2A",
               "included_scene_count": len(included), "included_scenes": included,
               "ROI_count": int(len(rois)), "bands_used_for_joint_fit": list(WATER_BANDS),
               "bands_used_as_surface_constraints": list(SWIR_BANDS),
               "hydrolight_library_cases": int(len(library)), "withheld_truth_cases": int(len(truth)),
               "synthetic_validation": validation, "cycle1_retained_scene_count": accepted,
               "scientific_conclusion": conclusion}
    write_json(OUTPUT / "experiment_summary.json", summary)
    report = build_html(summary, previous_audit, decisions)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"REPORT: {report}", flush=True)


if __name__ == "__main__":
    main()
