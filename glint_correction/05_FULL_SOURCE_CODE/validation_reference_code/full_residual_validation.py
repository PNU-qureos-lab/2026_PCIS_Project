#!/usr/bin/env python3
"""Full, validation-gated residual-glint experiment for one Tidung L2A scene.

No HTML is produced.  The script uses the already validated controlled
HydroLight library, injects known glint into real low-apparent-glint pixels,
freezes empirical detector thresholds, tests 30 offshore ROI windows, and
creates a Cycle-1 candidate only when every preceding gate passes.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_erosion, binary_opening, distance_transform_edt, uniform_filter
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_tidung_controlled_hydrolight_glint_validation as controlled


SCENE = "20240925_S2A"
SOURCE = ROOT / "outputs" / "Tidung_Glint_Comparison_Rerun_20260903"
JOINT = ROOT / "outputs" / "Tidung_Joint_HydroLight_Glint_Multiband_Experiment_20260911"
CONTROL = ROOT / "outputs" / "Tidung_Residual_Glint_V3_Experiment_20260911"
OUT = ROOT / "outputs" / "Tidung_Full_Residual_Glint_Experiment_20260911"

BANDS = list(controlled.BANDS)
WAVE = controlled.WAVE
WEIGHTS = np.array([1., 1., 1., 1., 1., 1., .5, .5])
CORE = np.array([True, True, True, True, True, True, False, False])
K_VALUES = (0., .10, .25, .50, 1.00)
ROI_HALF = 3
RNG = np.random.default_rng(20260911)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def read_tif(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(1).astype(float), src.profile.copy()


def read_stack(paths: list[Path]) -> tuple[np.ndarray, dict]:
    arrays, profile = [], None
    for path in paths:
        arr, profile = read_tif(path)
        arrays.append(arr)
    return np.stack(arrays, axis=-1), profile


def write_tif(path: Path, arr: np.ndarray, profile: dict, description: str, units: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", predictor=3)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype("float32"), 1)
        dst.set_band_description(1, description)
        dst.update_tags(units=units, AREA_OR_POINT="Area")


def weighted_rmse(diff: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(WEIGHTS * diff * diff, axis=-1) / WEIGHTS.sum())


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide(np.sum(a * b, axis=-1), den,
                     out=np.full(a.shape[:-1], np.nan), where=den > 0)


def dense_water_library() -> tuple[np.ndarray, pd.DataFrame]:
    train = pd.read_csv(CONTROL / "05_HydroLight_Water_Library" / "training_cases.csv")
    model = controlled.WaterModel(train)
    rows, spectra = [], []
    for chl in np.linspace(controlled.TRAIN_CHL.min(), controlled.TRAIN_CHL.max(), 7):
        for cdom in np.linspace(controlled.TRAIN_CDOM.min(), controlled.TRAIN_CDOM.max(), 7):
            for nap in np.linspace(controlled.TRAIN_TSM.min(), controlled.TRAIN_TSM.max(), 12):
                rows.append({"chl_mg_m3": chl, "cdom440_m_1": cdom, "nap_tsm_g_m3": nap})
                spectra.append(model.spectrum(np.array([chl, cdom, nap])))
    return np.asarray(spectra), pd.DataFrame(rows)


def load_scene() -> dict:
    before, profile = read_stack([
        JOINT / "02_Multiband_Inputs" / SCENE / f"{b}_BOA_20m.tif" for b in BANDS])
    cycle0, _ = read_stack([
        JOINT / "03_Glint_ACOLITE_Cycle0" / SCENE / f"Cycle0_{b}.tif" for b in BANDS])
    glint0, _ = read_stack([
        JOINT / "03_Glint_ACOLITE_Cycle0" / SCENE / f"estimated_glint_{b}.tif" for b in BANDS])
    valid, _ = read_tif(SOURCE / "01_Masks" / SCENE / "valid_water_mask.tif")
    land, _ = read_tif(SOURCE / "01_Masks" / SCENE / "land_nonwater_mask.tif")
    rhog, _ = read_tif(SOURCE / "06_ACOLITE" / "Rhog_Reference" / SCENE / "rhog_ref.tif")
    good = ((valid > 0) & ~(land > 0) & np.all(np.isfinite(before), axis=-1) &
            np.all(np.isfinite(cycle0), axis=-1))
    before[~good] = np.nan
    cycle0[~good] = np.nan
    glint0[~good] = np.nan
    return {"before": before, "cycle0": cycle0, "glint0": glint0,
            "valid": good, "land": land > 0, "rhog": rhog, "profile": profile}


def farthest_points(coords: np.ndarray, count: int, prior: list[np.ndarray]) -> list[np.ndarray]:
    pool = coords.copy(); selected = []
    while len(selected) < count and len(pool):
        refs = prior + selected
        if refs:
            ref = np.vstack(refs)
            d2 = np.min(np.sum((pool[:, None, :]-ref[None, :, :])**2, axis=2), axis=1)
            idx = int(np.argmax(d2))
            if np.sqrt(d2[idx]) < 8:
                break
        else:
            centre = pool.mean(axis=0); idx = int(np.argmax(np.sum((pool-centre)**2, axis=1)))
        point = pool[idx]; selected.append(point)
        pool = pool[np.sum((pool-point)**2, axis=1) >= 64]
    return selected


def load_rois(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    valid = data["valid"] & np.all(data["cycle0"] > 0, axis=-1)
    centres = binary_erosion(valid, np.ones((7, 7), bool), border_value=0)
    offshore = distance_transform_edt(~data["land"]) >= 50  # 1 km at 20 m
    edge = np.zeros(valid.shape, bool); edge[25:-25, 25:-25] = True
    b04 = np.nan_to_num(data["cycle0"][..., 2])
    mean = uniform_filter(b04, 7, mode="nearest")
    sq = uniform_filter(b04*b04, 7, mode="nearest")
    cv = np.sqrt(np.maximum(0., sq-mean*mean))/np.maximum(np.abs(mean), 1e-8)
    rhog_valid = np.where(valid & np.isfinite(data["rhog"]), data["rhog"], 0.0)
    rhog_mean = uniform_filter(rhog_valid, 7, mode="nearest")
    eligible = centres & offshore & edge & np.isfinite(rhog_mean) & (cv <= .15)
    if eligible.sum() < 30:
        raise RuntimeError(f"Only {eligible.sum()} eligible candidate-deep-water ROI centres")
    q1, q2 = np.quantile(rhog_mean[eligible], [1/3, 2/3])
    masks = {"LOW": eligible & (rhog_mean <= q1),
             "MEDIUM": eligible & (rhog_mean > q1) & (rhog_mean <= q2),
             "HIGH": eligible & (rhog_mean > q2)}
    rows, chosen = [], []
    for stratum, mask in masks.items():
        points = farthest_points(np.argwhere(mask), 10, chosen)
        if len(points) != 10:
            raise RuntimeError(f"{stratum} supplied only {len(points)} separated ROI windows")
        for point in points:
            chosen.append(point); rr, cc = map(int, point)
            sl = (slice(rr-3,rr+4),slice(cc-3,cc+4))
            before = data["before"][sl].mean(axis=(0,1)); cycle0 = data["cycle0"][sl].mean(axis=(0,1)); glint0 = data["glint0"][sl].mean(axis=(0,1))
            row = {"scene":SCENE,"roi":f"R{len(rows)+1:02d}","stratum":stratum,
                   "row":rr,"col":cc,"valid_pixels":49,
                   "distance_to_land_m":float(distance_transform_edt(~data["land"])[rr,cc]*20),
                   "local_B04_cv":float(cv[rr,cc]),"rhog_ref_mean":float(rhog_mean[rr,cc])}
            for i,b in enumerate(BANDS):
                row[f"before_{b}"]=before[i];row[f"cycle0_{b}"]=cycle0[i];row[f"glint0_{b}"]=glint0[i]
            rows.append(row)
    rois = pd.DataFrame(rows)
    pixel_rows, spectra, glint_spectra = [], [], []
    for _, r in rois.iterrows():
        rr, cc = int(r.row), int(r.col)
        for y in range(rr - ROI_HALF, rr + ROI_HALF + 1):
            for x in range(cc - ROI_HALF, cc + ROI_HALF + 1):
                if not data["valid"][y, x]:
                    raise RuntimeError(f"Invalid pixel inside {r.roi}")
                pixel_rows.append({"roi": r.roi, "stratum": r.stratum, "row": y, "col": x,
                                   "rhog_ref": float(data["rhog"][y, x])})
                spectra.append(data["cycle0"][y, x])
                glint_spectra.append(data["glint0"][y, x])
    return rois, pd.DataFrame(pixel_rows), np.asarray(spectra), np.asarray(glint_spectra)


def glint_shapes() -> dict[str, np.ndarray]:
    factors = pd.read_csv(JOINT / "03_Glint_ACOLITE_Cycle0" / SCENE /
                          "multiband_glint_factors.csv").set_index("band").loc[BANDS]
    b11 = factors.factor_from_B11.to_numpy(float)
    b12 = factors.factor_from_B12.to_numpy(float)
    b11 /= b11.max(); b12 /= b12.max()
    mix = .5 * (b11 + b12); mix /= mix.max()
    return {"B11-derived": b11, "B11/B12 mixture": mix, "B12-derived": b12}


def fit_one(obs: np.ndarray, models: np.ndarray, g: np.ndarray, bias: np.ndarray) -> dict:
    residual = obs[None, :] - bias[None, :] - models
    rmse0 = weighted_rmse(residual)
    denom = float(np.sum(WEIGHTS * g * g))
    alpha = np.maximum(0., np.sum(WEIGHTS * residual * g[None, :], axis=1) / denom)
    after = residual - alpha[:, None] * g[None, :]
    rmse1 = weighted_rmse(after)
    i0, i1 = int(np.argmin(rmse0)), int(np.argmin(rmse1))
    chosen_residual = residual[i1]
    return {"water_i": i1, "water": models[i1], "raw_alpha": float(alpha[i1]),
            "rmse0": float(rmse0[i0]), "rmse1": float(rmse1[i1]),
            "improvement": float(100 * (rmse0[i0] - rmse1[i1]) / max(rmse0[i0], 1e-12)),
            "similarity": float(cosine_rows(chosen_residual[None, :], g[None, :])[0])}


def calibrate_bias(low_spectra: np.ndarray, models: np.ndarray) -> np.ndarray:
    bias = np.zeros(len(BANDS))
    for _ in range(4):
        residuals = []
        for obs in low_spectra:
            d = obs[None, :] - bias[None, :] - models
            idx = int(np.argmin(weighted_rmse(d)))
            residuals.append(obs - models[idx])
        updated = np.median(residuals, axis=0)
        # A scene-level mismatch is allowed, but sharp band-to-band corrections
        # are not: use a three-band running smoother with protected endpoints.
        smooth = updated.copy()
        smooth[1:-1] = .25 * updated[:-2] + .5 * updated[1:-1] + .25 * updated[2:]
        if np.max(np.abs(smooth - bias)) < 1e-7:
            bias = smooth
            break
        bias = smooth
    return bias


def real_injection_validation(pixel_meta: pd.DataFrame, spectra: np.ndarray,
                              glint_spectra: np.ndarray, models: np.ndarray,
                              bias: np.ndarray) -> tuple[pd.DataFrame, dict]:
    # The residual decision is made for 7x7 ROI means, not isolated pixels.
    # Bootstrap each low ROI to retain measured within-ROI noise while testing
    # the same spatial support used by the real-scene decision.
    base_rows, glint_rows, meta_rows = [], [], []
    low_meta = pixel_meta[pixel_meta.stratum.eq("LOW")]
    for roi in sorted(low_meta.roi.unique()):
        idx = np.where(pixel_meta.roi.to_numpy() == roi)[0]
        for repeat in range(50):
            draw = RNG.choice(idx, size=len(idx), replace=True)
            base_rows.append(spectra[draw].mean(axis=0))
            glint_rows.append(glint_spectra[draw].mean(axis=0))
            meta_rows.append({"roi": roi, "bootstrap_repeat": repeat, "stratum": "LOW",
                              "row": float(pixel_meta.iloc[idx].row.mean()),
                              "col": float(pixel_meta.iloc[idx].col.mean()),
                              "rhog_ref": float(pixel_meta.iloc[draw].rhog_ref.mean())})
    base = np.asarray(base_rows)
    glint_basis = np.asarray(glint_rows)
    meta = pd.DataFrame(meta_rows)
    raw0 = np.array([fit_one(obs, models, g, bias)["raw_alpha"] for obs, g in zip(base, glint_basis)])
    k_offset = float(np.median(raw0))
    roi_reference = {}
    for roi in sorted(meta.roi.unique()):
        use = meta.roi.eq(roi).to_numpy()
        ref_obs = base[use].mean(axis=0)
        ref_g = glint_basis[use].mean(axis=0)
        roi_reference[roi] = fit_one(ref_obs, models, ref_g, bias)
    rows = []
    for i, (obs0, g) in enumerate(zip(base, glint_basis)):
        reference = roi_reference[meta.iloc[i].roi]
        for k_true in K_VALUES:
            obs = obs0 + k_true * g
            fit = fit_one(obs, models, g, bias)
            # Paired known-answer recovery: remove the same ROI's zero-injection
            # baseline. This measures recovery of the component we added, while
            # the separate natural-scene gate still uses only low-control data.
            k_est = max(0., fit["raw_alpha"] - reference["raw_alpha"])
            corrected = obs - k_est * g
            rows.append({**meta.iloc[i].to_dict(), "k_true": k_true, "k_est": k_est,
                         "k_absolute_error": abs(k_est-k_true),
                         "true_component_max_reflectance": float(np.max(k_true*g)),
                         "estimated_component_max_reflectance": float(np.max(k_est*g)),
                         "component_max_absolute_error": float(np.max(np.abs((k_est-k_true)*g))),
                         "corrected_to_base_rmse": float(np.sqrt(np.mean((corrected-obs0)**2))),
                         "water_only_rmse": fit["rmse0"], "joint_rmse": fit["rmse1"],
                         "absolute_rmse_improvement": fit["rmse0"]-fit["rmse1"],
                         "improvement_percent": fit["improvement"],
                         "improvement_gain_percent": fit["improvement"]-reference["improvement"],
                         "shape_similarity": fit["similarity"]})
    frame = pd.DataFrame(rows)
    null = frame[frame.k_true == 0]
    component_lod = float(null.estimated_component_max_reflectance.quantile(.95, interpolation="higher"))
    imp_limit = float(null.improvement_gain_percent.quantile(.95, interpolation="higher"))
    real_rmse_limit = float(null.absolute_rmse_improvement.quantile(.95, interpolation="higher"))
    positive = frame[frame.true_component_max_reflectance >= .001]
    similarity_limit = float(max(.50, positive.shape_similarity.quantile(.05)))
    frame["detected"] = ((frame.estimated_component_max_reflectance > component_lod) &
                         (frame.improvement_gain_percent > imp_limit) &
                         (frame.shape_similarity >= similarity_limit))
    null = frame[frame.k_true == 0]
    positive = frame[frame.true_component_max_reflectance >= .001]
    metrics = {
        "real_low_control_ROIs": int(low_meta.roi.nunique()),
        "bootstrap_ROI_means": int(len(base)),
        "trials": int(len(frame)),
        "k_baseline_offset": k_offset,
        "detection_limit_reflectance": component_lod,
        "minimum_injection_improvement_gain_percent": imp_limit,
        "minimum_real_rmse_improvement_reflectance": real_rmse_limit,
        "minimum_shape_similarity": similarity_limit,
        "zero_glint_false_positive_rate": float(null.detected.mean()),
        "sensitivity_component_ge_0_001": float(positive.detected.mean()),
        "median_k_relative_error_component_ge_0_001": float(
            np.median(positive.k_absolute_error / positive.k_true)),
        "median_component_absolute_error": float(positive.component_max_absolute_error.median()),
        "median_corrected_to_base_rmse": float(positive.corrected_to_base_rmse.median())}
    metrics["pass"] = bool(metrics["zero_glint_false_positive_rate"] <= .05 and
                           metrics["sensitivity_component_ge_0_001"] >= .90 and
                           metrics["median_k_relative_error_component_ge_0_001"] <= .20 and
                           metrics["median_corrected_to_base_rmse"] <= .00030)
    return frame, metrics


def fit_rois(rois: pd.DataFrame, models: np.ndarray, params: pd.DataFrame,
             bias: np.ndarray, thresholds: dict) -> pd.DataFrame:
    rows = []
    for _, r in rois.iterrows():
        obs = r[[f"cycle0_{b}" for b in BANDS]].to_numpy(float)
        g = r[[f"glint0_{b}" for b in BANDS]].to_numpy(float)
        fit = fit_one(obs, models, g, bias)
        k = float(np.clip(fit["raw_alpha"] - thresholds["k_baseline_offset"], 0., 1.))
        component = k*g
        component_max = float(np.max(component))
        water = fit["water"]
        water_reference = water + bias
        p = params.iloc[fit["water_i"]]
        row = {"scene": SCENE, "roi": r.roi, "stratum": r.stratum,
               "row": int(r.row), "col": int(r.col), "rhog_ref_mean": r.rhog_ref_mean,
               "k": k, "component_max_reflectance": component_max,
               "detected": False, "water_only_rmse": fit["rmse0"],
               "joint_rmse": fit["rmse1"],
               "absolute_rmse_improvement": fit["rmse0"]-fit["rmse1"],
               "improvement_percent": fit["improvement"],
               "shape_similarity": fit["similarity"], "nap_tsm_g_m3": p.nap_tsm_g_m3,
               "chl_diagnostic_only": p.chl_mg_m3, "cdom_diagnostic_only": p.cdom440_m_1}
        for j, band in enumerate(BANDS):
            row[f"cycle0_{band}"] = obs[j]
            row[f"water_{band}"] = water[j] + bias[j]
            row[f"candidate_{band}"] = obs[j] - component[j]
        row["cycle0_hydrolight_rmse"] = float(np.sqrt(np.mean((obs-water_reference)**2)))
        row["candidate_hydrolight_rmse"] = float(np.sqrt(np.mean((obs-component-water_reference)**2)))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["detected"] = ((frame.component_max_reflectance > thresholds["detection_limit_reflectance"]) &
                         (frame.absolute_rmse_improvement > thresholds["minimum_real_rmse_improvement_reflectance"]) &
                         (frame.shape_similarity >= thresholds["minimum_shape_similarity"]))
    for band in BANDS:
        frame.loc[~frame.detected, f"candidate_{band}"] = frame.loc[~frame.detected, f"cycle0_{band}"]
    cycle0_matrix=frame[[f"cycle0_{b}" for b in BANDS]].to_numpy(float)
    candidate_matrix=frame[[f"candidate_{b}" for b in BANDS]].to_numpy(float)
    water_matrix=frame[[f"water_{b}" for b in BANDS]].to_numpy(float)
    frame["cycle0_hydrolight_rmse"]=np.sqrt(np.mean((cycle0_matrix-water_matrix)**2,axis=1))
    frame["candidate_hydrolight_rmse"]=np.sqrt(np.mean((candidate_matrix-water_matrix)**2,axis=1))
    return frame


def roi_scene_gate(fits: pd.DataFrame, injection_pass: bool, lod: float) -> dict:
    low, medium, high = (fits[fits.stratum.eq(s)] for s in ("LOW", "MEDIUM", "HIGH"))
    rho, rho_p = spearmanr(fits.rhog_ref_mean, fits.component_max_reflectance)
    mw = mannwhitneyu(high.component_max_reflectance, low.component_max_reflectance, alternative="greater")
    result = {
        "injection_validation_pass": bool(injection_pass),
        "low_detected": int(low.detected.sum()), "medium_detected": int(medium.detected.sum()),
        "high_detected": int(high.detected.sum()),
        "median_component_low": float(low.component_max_reflectance.median()),
        "median_component_medium": float(medium.component_max_reflectance.median()),
        "median_component_high": float(high.component_max_reflectance.median()),
        "high_minus_low_median_component": float(high.component_max_reflectance.median()-low.component_max_reflectance.median()),
        "spearman_alpha_vs_rhog": float(rho), "spearman_p": float(rho_p),
        "high_greater_low_mannwhitney_p": float(mw.pvalue),
        "candidate_deep_water_only": True,
        "bathymetry_available": False}
    result["pass"] = bool(injection_pass and result["low_detected"] <= 1 and
                          result["high_detected"] >= 5 and
                          result["high_minus_low_median_component"] > lod and
                          np.isfinite(rho) and rho >= .30 and rho_p <= .10 and mw.pvalue <= .05)
    return result


def pixel_candidate(data: dict, models: np.ndarray, bias: np.ndarray,
                    thresholds: dict, authorized: bool) -> dict:
    valid, obs = data["valid"], data["cycle0"]
    alpha = np.full(valid.shape, np.nan, np.float32)
    improvement = np.full(valid.shape, np.nan, np.float32)
    similarity = np.full(valid.shape, np.nan, np.float32)
    detected = np.zeros(valid.shape, bool)
    coords = np.argwhere(valid)
    for start in range(0, len(coords), 350):
        xy = coords[start:start+350]
        spectra = obs[xy[:, 0], xy[:, 1]]
        gg = data["glint0"][xy[:, 0], xy[:, 1]]
        residual = spectra[:, None, :] - bias[None, None, :] - models[None, :, :]
        rm0all = weighted_rmse(residual)
        rm0 = rm0all.min(axis=1)
        denom = np.sum(WEIGHTS[None, :] * gg * gg, axis=1)
        k_all = np.maximum(0., np.sum(WEIGHTS[None, None, :] * residual * gg[:, None, :], axis=2) /
                           np.maximum(denom[:, None], 1e-14))
        rm1all = weighted_rmse(residual - k_all[..., None] * gg[:, None, :])
        idx = np.argmin(rm1all, axis=1)
        raw = k_all[np.arange(len(xy)), idx]
        k = np.clip(raw-thresholds["k_baseline_offset"], 0., 1.)
        rm1 = rm1all[np.arange(len(xy)), idx]
        res = residual[np.arange(len(xy)), idx]
        sim = cosine_rows(res, gg)
        delta_rmse = rm0-rm1
        imp = 100 * delta_rmse / np.maximum(rm0, 1e-12)
        component = k[:, None]*gg
        component_max = np.max(component, axis=1)
        keep = ((component_max > thresholds["detection_limit_reflectance"]) &
                (delta_rmse > thresholds["minimum_real_rmse_improvement_reflectance"]) &
                (sim >= thresholds["minimum_shape_similarity"]))
        yy, xx = xy[:, 0], xy[:, 1]
        alpha[yy, xx] = k; improvement[yy, xx] = imp; similarity[yy, xx] = sim
        detected[yy, xx] = keep
    if authorized:
        support = uniform_filter(detected.astype(float), 5, mode="constant") >= .48
        detected &= binary_opening(support, structure=np.ones((3, 3), bool))
    else:
        detected[:] = False
    candidate = obs.copy()
    candidate[detected] -= alpha[detected, None] * data["glint0"][detected]
    candidate[~valid] = np.nan
    return {"alpha": alpha, "improvement": improvement, "similarity": similarity,
            "mask": detected, "candidate": candidate}


def final_qc(data: dict, candidate: dict, rois: pd.DataFrame, scene_gate: bool) -> dict:
    valid = data["valid"]
    base, cand = data["cycle0"], candidate["candidate"]
    neg = float(100*np.mean(np.any(cand[valid][:, CORE] < 0, axis=1)))
    neg_all = float(100*np.mean(np.any(cand[valid] < 0, axis=1)))
    corr = float(np.corrcoef(base[valid].ravel(), cand[valid].ravel())[0, 1])
    corrected_fraction = float(candidate["mask"][valid].mean())
    low_changes, high_changes = [], []
    for _, r in rois.iterrows():
        rr, cc = int(r.row), int(r.col)
        a = base[rr-3:rr+4, cc-3:cc+4]
        b = cand[rr-3:rr+4, cc-3:cc+4]
        change = float(np.mean(np.abs(a-b)))
        (low_changes if r.stratum == "LOW" else high_changes if r.stratum == "HIGH" else []).append(change)
    low_change = float(np.median(low_changes))
    high_change = float(np.median(high_changes))
    result = {"scene_gate_pass": bool(scene_gate), "negative_visible_pixel_percent": neg,
              "negative_all_eight_band_pixel_percent": neg_all,
              "spatial_preservation_r": corr, "corrected_valid_water_fraction": corrected_fraction,
              "median_low_ROI_absolute_change": low_change,
              "median_high_ROI_absolute_change": high_change}
    result["pass"] = bool(scene_gate and neg <= 1.0 and neg_all <= 1.0 and corr >= .95 and
                          low_change <= .00030 and corrected_fraction > 0 and
                          high_change > low_change)
    result["retained_product"] = "Cycle1" if result["pass"] else "Cycle0"
    return result


def apply_conservative_scale(data: dict, raw_candidate: dict, scale: float) -> dict:
    """Scale only the accepted residual term; keep the detection decision fixed."""
    result = {k: v for k, v in raw_candidate.items() if k != "candidate"}
    result["alpha"] = raw_candidate["alpha"] * scale
    corrected = data["cycle0"].copy()
    mask = raw_candidate["mask"]
    corrected[mask] -= result["alpha"][mask, None] * data["glint0"][mask]
    corrected[~data["valid"]] = np.nan
    result["candidate"] = corrected
    result["correction_strength"] = float(scale)
    return result


def select_safe_candidate(data: dict, raw_candidate: dict, rois: pd.DataFrame,
                          scene_gate: bool) -> tuple[dict, dict, pd.DataFrame]:
    """Use the strongest predeclared correction that satisfies every final-QC gate."""
    rows = []
    chosen_candidate = apply_conservative_scale(data, raw_candidate, 0.0)
    chosen_qc = final_qc(data, chosen_candidate, rois, False)
    for scale in np.round(np.arange(1.0, 0.0, -0.05), 2):
        candidate = apply_conservative_scale(data, raw_candidate, float(scale))
        qc = final_qc(data, candidate, rois, scene_gate)
        rows.append({"correction_strength": float(scale), **qc})
        if qc["pass"]:
            chosen_candidate, chosen_qc = candidate, qc
            break
    chosen_qc["selected_correction_strength"] = float(chosen_candidate["correction_strength"])
    chosen_qc["strength_selection_rule"] = (
        "Largest value in the predeclared sequence 1.00, 0.95, ..., 0.05 that passes all final-QC gates")
    return chosen_candidate, chosen_qc, pd.DataFrame(rows)


def method_metrics(data: dict, final: np.ndarray, rois: pd.DataFrame) -> pd.DataFrame:
    methods = {"Before correction": data["before"][..., :3],
               "Glint-ACOLITE Cycle 0": data["cycle0"][..., :3],
               "Residual Cycle 1": final[..., :3]}
    for label, folder in (("Hedley", "Hedley"), ("B12 subtraction", "B12_SWIR")):
        arrs = []
        for b in ("B02", "B03", "B04"):
            path = SOURCE / "04_Hedley" / "Corrected" / SCENE / f"{b}_corrected.tif" if label == "Hedley" else \
                   SOURCE / "05_NIR_SWIR" / "Corrected" / SCENE / f"{b}_corrected.tif"
            arrs.append(read_tif(path)[0])
        methods[label] = np.stack(arrs, axis=-1)
    rows = []
    valid = data["valid"]
    for name, stack in methods.items():
        good = valid & np.all(np.isfinite(stack), axis=-1)
        neg = float(100*np.mean(np.any(stack[good] < 0, axis=1)))
        corr = float(np.corrcoef(data["before"][..., :3][good].ravel(), stack[good].ravel())[0, 1])
        for stratum in ("LOW", "MEDIUM", "HIGH"):
            vals = []
            for _, r in rois[rois.stratum.eq(stratum)].iterrows():
                rr, cc = int(r.row), int(r.col)
                vals.append(stack[rr-3:rr+4, cc-3:cc+4].mean(axis=(0, 1)))
            med = np.median(vals, axis=0)
            rows.append({"method": name, "stratum": stratum,
                         "negative_valid_water_percent": neg,
                         "spatial_correlation_with_before": corr,
                         **{f"median_{b}": med[i] for i, b in enumerate(("B02", "B03", "B04"))}})
    return pd.DataFrame(rows)


def rgb(stack: np.ndarray, stretch: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    x = np.stack([stack[..., 2], stack[..., 1], stack[..., 0]], axis=-1)
    if stretch is None:
        finite = x[np.all(np.isfinite(x), axis=-1)]
        lo = np.quantile(finite, .01, axis=0); hi = np.quantile(finite, .98, axis=0)
    else:
        lo, hi = stretch
    return np.power(np.clip((np.nan_to_num(x, nan=lo)-lo)/(hi-lo+1e-9), 0, 1), .85)


def make_figures(data: dict, rois: pd.DataFrame, injection: pd.DataFrame,
                 fits: pd.DataFrame, candidate: dict, qc: dict,
                 methods: pd.DataFrame, shapes: dict[str, np.ndarray], bias: np.ndarray) -> None:
    folder = OUT / "13_Figures"; folder.mkdir(parents=True, exist_ok=True)
    # Academic flowchart.
    fig, ax = plt.subplots(figsize=(8.5, 12), constrained_layout=True); ax.axis("off")
    boxes = [
        ("INPUT", "Sentinel-2 L2A valid water\nB02–B8A, B11/B12, masks, geometry"),
        ("CYCLE 0", "Glint-ACOLITE correction\nHedley and B12 retained as comparisons"),
        ("KNOWN-ANSWER TEST", "Inject known glint into bootstrapped 7×7 low-glint ROI means\nRecover k and freeze thresholds"),
        ("DECISION 1", "Injection validation passes?"),
        ("REAL RESIDUAL TEST", "30 offshore candidate-deep ROIs\n10 low + 10 medium + 10 high"),
        ("DECISION 2", "Residual magnitude follows B11/B12\nand is stronger in high than low ROIs?"),
        ("CYCLE 1 CANDIDATE", "Subtract only detected, spatially coherent component"),
        ("FINAL QC", "Negative pixels, low-control change,\nspatial preservation and method comparison"),
        ("OUTPUT", "Retain Cycle 1 only if every gate passes;\notherwise retain Cycle 0")]
    ys = np.linspace(.94, .06, len(boxes))
    for i, ((head, body), y) in enumerate(zip(boxes, ys)):
        fc = "#e0f2fe" if "DECISION" not in head else "#fef3c7"
        box = FancyBboxPatch((.16, y-.035), .68, .07, boxstyle="round,pad=.012",
                             fc=fc, ec="#164e63", lw=1.5, transform=ax.transAxes)
        ax.add_patch(box); ax.text(.5, y+.012, head, ha="center", va="center", weight="bold", transform=ax.transAxes)
        ax.text(.5, y-.014, body, ha="center", va="center", fontsize=9, transform=ax.transAxes)
        if i < len(boxes)-1:
            ax.annotate("", xy=(.5, ys[i+1]+.042), xytext=(.5, y-.042), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", lw=1.6, color="#164e63"))
    ax.set_title("Tidung residual-glint experiment flow", fontsize=17, pad=12)
    fig.savefig(folder / "01_complete_experiment_flowchart.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True); ax.imshow(rgb(data["before"]))
    colors = {"LOW":"#2563eb", "MEDIUM":"#f59e0b", "HIGH":"#dc2626"}
    for _, r in rois.iterrows():
        ax.add_patch(Rectangle((r.col-3, r.row-3), 7, 7, fill=False, ec=colors[r.stratum], lw=1.5))
        ax.text(r.col+4, r.row, r.roi, color="white", fontsize=7,
                bbox=dict(facecolor=colors[r.stratum], edgecolor="none", pad=1))
    ax.legend(handles=[plt.Line2D([0],[0], color=c, lw=3, label=f"{s}: 10 ROIs") for s,c in colors.items()])
    ax.set(title=f"{SCENE} · 30 offshore candidate-deep-water ROIs",
           xlabel="Image column (20 m pixel)", ylabel="Image row (20 m pixel)")
    fig.savefig(folder / "02_roi_map.png", dpi=180); plt.close(fig)

    summary = injection.groupby("k_true").k_est.agg(
        median="median", p05=lambda x: x.quantile(.05), p95=lambda x: x.quantile(.95)).reset_index()
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.plot(summary.k_true, summary["median"], "o-", color="#0f766e", label="Median recovered k")
    ax.fill_between(summary.k_true, summary.p05, summary.p95, color="#0f766e", alpha=.16, label="P05–P95")
    lim=max(injection.k_true.max(), injection.k_est.quantile(.995)); ax.plot([0,lim],[0,lim],"k--",label="Perfect recovery")
    ax.set(xlabel="Known injected multiplier k (unitless)", ylabel="Recovered multiplier k (unitless)",
           title="Known k × ACOLITE glint injected into real low-glint ROI means"); ax.grid(alpha=.25); ax.legend()
    fig.savefig(folder / "03_real_pixel_injection_recovery.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    for ax, stratum in zip(axes, ("LOW", "MEDIUM", "HIGH")):
        q = rois[rois.stratum.eq(stratum)]
        for _, r in q.iterrows():
            ax.plot(WAVE, r[[f"before_{b}" for b in BANDS]], color="#64748b", alpha=.25)
            ax.plot(WAVE, r[[f"cycle0_{b}" for b in BANDS]], color="#0f766e", alpha=.30)
        ax.plot(WAVE, q[[f"before_{b}" for b in BANDS]].median(), "o-", color="#334155", lw=2, label="Before")
        ax.plot(WAVE, q[[f"cycle0_{b}" for b in BANDS]].median(), "o-", color="#0f766e", lw=2, label="Cycle 0")
        ax.set(title=f"{stratum}: all 10 ROIs", xlabel="Wavelength (nm)", ylabel="BOA reflectance (dimensionless)"); ax.grid(alpha=.2)
    axes[0].legend(); fig.suptitle("All 30 ROI spectra before and after Glint-ACOLITE")
    fig.savefig(folder / "04_all_roi_before_cycle0.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for s,c in colors.items():
        q=fits[fits.stratum.eq(s)]; axes[0].scatter(q.rhog_ref_mean,q.component_max_reflectance,label=s,color=c,s=45)
    axes[0].axhline(0,color="black",lw=1); axes[0].set(xlabel="Cycle-0 B11/B12 surface-reference magnitude",ylabel="Fitted residual-glint component (reflectance)",title="Spatial glint evidence")
    axes[0].grid(alpha=.2); axes[0].legend()
    axes[1].boxplot([fits[fits.stratum.eq(s)].component_max_reflectance for s in ("LOW","MEDIUM","HIGH")],tick_labels=["LOW","MEDIUM","HIGH"])
    axes[1].set(xlabel="Apparent-glint stratum",ylabel="Fitted residual-glint component (reflectance)",title="Low-to-high separation"); axes[1].grid(axis="y",alpha=.2)
    fig.savefig(folder / "05_real_roi_residual_evidence.png", dpi=180); plt.close(fig)

    final = candidate["candidate"] if qc["retained_product"] == "Cycle1" else data["cycle0"]
    ref_rgb=np.stack([data["before"][...,2],data["before"][...,1],data["before"][...,0]],axis=-1)
    ref_finite=ref_rgb[np.all(np.isfinite(ref_rgb),axis=-1)]
    common_stretch=(np.quantile(ref_finite,.01,axis=0),np.quantile(ref_finite,.98,axis=0))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, arr, title in zip(axes, (data["before"],data["cycle0"],final),
                              ("Before correction","Glint-ACOLITE Cycle 0",f"Final retained: {qc['retained_product']}")):
        ax.imshow(rgb(arr,common_stretch)); ax.set(title=title,xlabel="Image column (20 m pixel)",ylabel="Image row (20 m pixel)")
    fig.savefig(folder / "06_before_cycle0_final_rgb.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    for ax, stratum in zip(axes,("LOW","MEDIUM","HIGH")):
        q=fits[fits.stratum.eq(stratum)]
        for _,r in q.iterrows():
            ax.plot(WAVE,r[[f"cycle0_{b}" for b in BANDS]],color="#0f766e",alpha=.20)
            ax.plot(WAVE,r[[f"candidate_{b}" for b in BANDS]],color="#2563eb",alpha=.25)
            ax.plot(WAVE,r[[f"water_{b}" for b in BANDS]],color="#d97706",alpha=.20,ls="--")
        ax.plot(WAVE,q[[f"cycle0_{b}" for b in BANDS]].median(),"o-",color="#0f766e",lw=2,label="Cycle 0")
        ax.plot(WAVE,q[[f"candidate_{b}" for b in BANDS]].median(),"o-",color="#2563eb",lw=2,label="Cycle 1 candidate")
        ax.plot(WAVE,q[[f"water_{b}" for b in BANDS]].median(),"s--",color="#d97706",lw=2,label="HydroLight + scene bias")
        ax.set(title=f"{stratum}: all 10 ROIs",xlabel="Wavelength (nm)",ylabel="BOA reflectance (dimensionless)");ax.grid(alpha=.2)
    axes[0].legend(); fig.suptitle("Residual correction and HydroLight comparison for every ROI")
    fig.savefig(folder / "07_all_roi_cycle0_cycle1_hydrolight.png",dpi=180);plt.close(fig)

    fig, ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    palette={"Before correction":"#475569","Glint-ACOLITE Cycle 0":"#0f766e","Residual Cycle 1":"#2563eb","Hedley":"#7c3aed","B12 subtraction":"#d97706"}
    for name,gp in methods.groupby("method"):
        high=gp[gp.stratum.eq("HIGH")].iloc[0]
        ax.plot(WAVE[:3],[high[f"median_{b}"] for b in ("B02","B03","B04")],"o-",lw=2,label=name,color=palette[name])
    ax.set(xlabel="Wavelength (nm)",ylabel="Median high-ROI BOA reflectance (dimensionless)",title="Same high-glint ROIs: all correction methods");ax.grid(alpha=.2);ax.legend()
    fig.savefig(folder / "08_method_comparison_high_rois.png",dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    im=axes[0].imshow(candidate["alpha"],cmap="magma");fig.colorbar(im,ax=axes[0],label="Residual-glint magnitude (reflectance)")
    axes[0].set_title("Estimated magnitude before gating")
    axes[1].imshow(candidate["mask"],cmap="gray",vmin=0,vmax=1);axes[1].set_title("Accepted spatial correction mask")
    valid=data["valid"]
    diff=np.full(valid.shape,np.nan,dtype=float)
    diff[valid]=np.mean((data["cycle0"]-candidate["candidate"])[valid],axis=-1)
    im2=axes[2].imshow(diff,cmap="viridis");fig.colorbar(im2,ax=axes[2],label="Mean removed reflectance")
    axes[2].set_title("Applied Cycle-1 removal")
    for ax in axes: ax.set(xlabel="Image column (20 m pixel)",ylabel="Image row (20 m pixel)")
    fig.savefig(folder / "09_cycle1_spatial_products.png",dpi=180);plt.close(fig)

    pd.DataFrame({"band":BANDS,"wavelength_nm":WAVE,"scene_bias_reflectance":bias,
                  **{k:v for k,v in shapes.items()}}).to_csv(OUT/"03_Scene_Calibration"/"scene_bias_and_glint_shapes.csv",index=False)


def main() -> None:
    for d in ("00_Setup","01_Inputs","02_Controlled_Validation","03_Scene_Calibration",
              "04_Real_Injection_Validation","05_ROI_Residual_Test","06_Candidate_Cycle1",
              "07_Final_QC","08_Method_Comparison","13_Figures","16_Code"):
        (OUT/d).mkdir(parents=True,exist_ok=True)
    control_summary=json.loads((CONTROL/"controlled_validation_summary.json").read_text())
    if not control_summary["known_glint"]["pass"]:
        raise RuntimeError("Controlled known-glint validation has not passed")
    data=load_scene(); rois,pixel_meta,pixel_spectra,pixel_glint=load_rois(data)
    models,params=dense_water_library(); shapes=glint_shapes(); g=shapes["B11/B12 mixture"]
    low_spectra=pixel_spectra[pixel_meta.stratum.to_numpy()=="LOW"]
    bias=calibrate_bias(low_spectra,models)
    injection,injection_metrics=real_injection_validation(pixel_meta,pixel_spectra,pixel_glint,models,bias)
    injection.to_csv(OUT/"04_Real_Injection_Validation"/"real_low_pixel_known_glint_trials.csv",index=False)
    write_json(OUT/"04_Real_Injection_Validation"/"injection_metrics.json",injection_metrics)
    fits=fit_rois(rois,models,params,bias,injection_metrics)
    fits.to_csv(OUT/"05_ROI_Residual_Test"/"all_30_roi_residual_fits.csv",index=False)
    scene_gate=roi_scene_gate(fits,injection_metrics["pass"],injection_metrics["detection_limit_reflectance"])
    scene_gate["minimum_real_rmse_improvement_reflectance"] = injection_metrics["minimum_real_rmse_improvement_reflectance"]
    write_json(OUT/"05_ROI_Residual_Test"/"scene_residual_gate.json",scene_gate)
    raw_candidate=pixel_candidate(data,models,bias,injection_metrics,scene_gate["pass"])
    candidate,qc,strength_trace=select_safe_candidate(data,raw_candidate,rois,scene_gate["pass"])
    strength_trace.to_csv(OUT/"07_Final_QC"/"correction_strength_safety_trace.csv",index=False)
    final=candidate["candidate"] if qc["pass"] else data["cycle0"]
    for i,b in enumerate(BANDS):
        write_tif(OUT/"06_Candidate_Cycle1"/f"candidate_Cycle1_{b}.tif",candidate["candidate"][...,i],data["profile"],f"Experimental residual-glint Cycle-1 {b}","BOA reflectance, dimensionless")
        write_tif(OUT/"07_Final_QC"/f"final_retained_{b}.tif",final[...,i],data["profile"],f"Final retained {qc['retained_product']} {b}","BOA reflectance, dimensionless")
    write_tif(OUT/"06_Candidate_Cycle1"/"estimated_residual_glint_magnitude.tif",candidate["alpha"],data["profile"],"Estimated residual-glint magnitude before spatial gating","reflectance")
    write_tif(OUT/"06_Candidate_Cycle1"/"accepted_cycle1_mask.tif",candidate["mask"].astype(float),data["profile"],"Accepted Cycle-1 correction mask","0 no correction; 1 corrected")
    write_json(OUT/"07_Final_QC"/"final_qc.json",qc)
    methods=method_metrics(data,final,rois);methods.to_csv(OUT/"08_Method_Comparison"/"method_metrics.csv",index=False)
    rois.to_csv(OUT/"01_Inputs"/"selected_30_offshore_rois.csv",index=False)
    make_figures(data,rois,injection,fits,candidate,qc,methods,shapes,bias)
    hydro_context={
        "median_cycle0_rmse_all_rois":float(fits.cycle0_hydrolight_rmse.median()),
        "median_cycle1_candidate_rmse_all_rois":float(fits.candidate_hydrolight_rmse.median()),
        "by_stratum":{s:{
            "cycle0_median_rmse":float(fits.loc[fits.stratum.eq(s),"cycle0_hydrolight_rmse"].median()),
            "cycle1_candidate_median_rmse":float(fits.loc[fits.stratum.eq(s),"candidate_hydrolight_rmse"].median())}
            for s in ("LOW","MEDIUM","HIGH")}}
    decision=("RETAIN_CYCLE1" if qc["pass"] else
              "SCENE_REJECTED" if qc["negative_visible_pixel_percent"] > 1.0 else
              "RETAIN_CYCLE0")
    summary={"created_utc":datetime.now(timezone.utc).isoformat(),"scene":SCENE,
             "input_product":"Sentinel-2 Level-2A BOA","controlled_known_glint_pass":True,
             "real_low_pixel_injection":injection_metrics,"real_roi_residual_gate":scene_gate,
             "hydrolight_context":hydro_context,"final_qc":qc,
             "decision":decision,
             "limitations":["ROIs are offshore candidate-deep water; no bathymetry was available.",
                            "The glint-shape family is B11/B12 ACOLITE-derived; direct-sun and diffuse-sky HydroLight components were not separately available.",
                            "Chl-a and CDOM fits are diagnostic only because controlled identifiability failed.",
                            "This is one-scene preflight, not field validation."]}
    write_json(OUT/"experiment_summary.json",summary)
    shutil.copy2(Path(__file__),OUT/"16_Code"/Path(__file__).name)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
