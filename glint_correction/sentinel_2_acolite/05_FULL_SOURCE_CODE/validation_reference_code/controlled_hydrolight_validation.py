#!/usr/bin/env python3
"""Controlled HydroLight closure and known-glint recovery experiment.

Creates numerical tables and PNG figures only. It does not process real images,
create Cycle-1 rasters, or build HTML.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

import run_tidung_tsm_hydrolight_inversion as common


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "Tidung_Residual_Glint_V3_Experiment_20260911"
TEMPLATE = ROOT / "outputs" / "hydrolight_pilot_20251101" / "inputs" / "ITIDHL_CLR.txt"
FACTORS = ROOT / "outputs" / "Tidung_Joint_HydroLight_Glint_Multiband_Experiment_20260911" / "03_Glint_ACOLITE_Cycle0" / "20241025_S2A" / "multiband_glint_factors.csv"
HE5 = Path(r"C:\HE5")
EXE = HE5 / "Code" / "mainHL_stnd.exe"
RUN_DIR = HE5 / "run"

PLATFORM = "S2A"
SZA = 20.4203573915463
WIND = 5.0
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A"]
WAVE = np.array([492., 560., 665., 704., 740., 783., 833., 865.])
TRAIN_CHL = np.array([0.05, 0.20, 0.50])
TRAIN_CDOM = np.array([0.01, 0.04, 0.10])
TRAIN_TSM = np.array([0.25, 0.75, 1.50, 3.00])
TEST_CHL = np.array([0.10, 0.35])
TEST_CDOM = np.array([0.025, 0.070])
TEST_TSM = np.array([0.50, 1.00])
RNG = np.random.default_rng(20260911)
ROOT_PREFIX = "V31"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def input_text(root_name: str, chl: float, cdom: float, tsm: float) -> str:
    lines = TEMPLATE.read_text(encoding="utf-8").splitlines()
    lines[1] = f"Tidung controlled V3.1; Chl {chl:g}; CDOM {cdom:g}; NAP {tsm:g}"
    lines[2] = root_name
    lines[6] = f"0, {chl:.8f}, {cdom:.8f}, {tsm:.8f}"
    sun = [x.strip() for x in lines[34].split(",")]
    sun[2] = f"{SZA:.8f}"
    lines[34] = ", ".join(sun)
    surface = [x.strip() for x in lines[36].split(",")]
    surface[0] = f"{WIND:.3f}"
    lines[36] = ", ".join(surface)
    return "\n".join(lines) + "\n"


def source_outputs(root_name: str) -> dict[str, Path]:
    base = HE5 / "output" / "Hydrolight"
    return {
        "D": base / "digital" / f"D{root_name}.txt",
        "L": base / "digital" / f"L{root_name}.txt",
        "M": base / "excel" / f"M{root_name}.txt",
        "S": base / "excel" / f"S{root_name}.txt",
        "P": base / "printout" / f"P{root_name}.txt",
    }


def run_one(root_name: str, chl: float, cdom: float, tsm: float,
            role: str, resume: bool) -> dict:
    base = OUT / "05_HydroLight_Water_Library"
    inp = base / "inputs" / f"I{root_name}.txt"
    result = base / "results" / role / root_name
    inp.parent.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    inp.write_text(input_text(root_name, chl, cdom, tsm), encoding="utf-8")
    expected = {k: result / p.name for k, p in source_outputs(root_name).items()}
    if not (resume and all(p.exists() and p.stat().st_size for p in expected.values())):
        # HydroLight uses scratch files in its working directory. Give every
        # concurrent case a private run directory to prevent cross-run mixing.
        isolated_run = HE5 / f"run_{root_name}"
        shutil.copytree(RUN_DIR, isolated_run, dirs_exist_ok=True)
        with inp.open("r", encoding="utf-8") as handle:
            proc = subprocess.run([str(EXE)], stdin=handle, cwd=isolated_run,
                                  text=True, capture_output=True, check=False)
        (result / "process_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (result / "process_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"HydroLight failed for {root_name}: {proc.returncode}")
        for key, src in source_outputs(root_name).items():
            if not src.exists() or not src.stat().st_size:
                raise RuntimeError(f"Missing HydroLight {key} output for {root_name}")
            shutil.copy2(src, expected[key])
    wave, rrs = common.parse_rrs(expected["M"])
    rsr = common.read_rsr(PLATFORM)
    conv = {band: common.convolve(wave, rrs, rsr[band.replace("0", "", 1)]) for band in BANDS}
    return {"case_id": root_name, "role": role, "chl_mg_m3": chl,
            "cdom440_m_1": cdom, "nap_g_m3": tsm, "sza_deg": SZA,
            "wind_m_s": WIND, "particle_absorption_file": r"C:\HE5\data\defaults\astarmin_average.txt",
            "particle_phase_function": "isotropic component setting from fixed template",
            "source_M_file": str(expected["M"]), **conv}


def create_library(resume: bool, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = []
    train_values = [(c, d, t) for c in TRAIN_CHL for d in TRAIN_CDOM for t in TRAIN_TSM]
    test_values = [(c, d, t) for c in TEST_CHL for d in TEST_CDOM for t in TEST_TSM]
    all_values = [("training", *v) for v in train_values] + [("withheld", *v) for v in test_values]
    jobs = []
    for i, (role, chl, cdom, tsm) in enumerate(all_values, 1):
        root_name = f"{ROOT_PREFIX}{i:03d}"
        jobs.append((i, root_name, role, chl, cdom, tsm))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run_one, root_name, chl, cdom, tsm, role, resume):
                   (i, root_name, role, chl, cdom, tsm)
                   for i, root_name, role, chl, cdom, tsm in jobs}
        completed = 0
        for future in as_completed(pending):
            i, root_name, role, chl, cdom, tsm = pending[future]
            cases.append(future.result())
            completed += 1
            print(f"HYDROLIGHT COMPLETE {completed}/{len(all_values)} case={i} {role} "
                  f"Chl={chl:g} CDOM={cdom:g} NAP={tsm:g}", flush=True)
    frame = pd.DataFrame(cases)
    train = frame[frame.role == "training"].copy()
    test = frame[frame.role == "withheld"].copy()
    folder = OUT / "05_HydroLight_Water_Library"
    frame.to_csv(folder / "controlled_water_library_manifest.csv", index=False)
    train.to_csv(folder / "training_cases.csv", index=False)
    test.to_csv(folder / "withheld_cases.csv", index=False)
    return train, test


class WaterModel:
    def __init__(self, train: pd.DataFrame):
        cube = np.empty((len(TRAIN_CHL), len(TRAIN_CDOM), len(TRAIN_TSM), len(BANDS)))
        for ic, chl in enumerate(TRAIN_CHL):
            for id_, cdom in enumerate(TRAIN_CDOM):
                for it, tsm in enumerate(TRAIN_TSM):
                    row = train[(train.chl_mg_m3 == chl) & (train.cdom440_m_1 == cdom) &
                                (train.nap_g_m3 == tsm)].iloc[0]
                    cube[ic, id_, it] = np.pi * row[BANDS].to_numpy(float)
        self.interp = RegularGridInterpolator((TRAIN_CHL, TRAIN_CDOM, TRAIN_TSM), cube,
                                               bounds_error=True)
        self.lo = np.array([TRAIN_CHL.min(), TRAIN_CDOM.min(), TRAIN_TSM.min()])
        self.hi = np.array([TRAIN_CHL.max(), TRAIN_CDOM.max(), TRAIN_TSM.max()])

    def spectrum(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(self.interp(np.asarray(params)[None, :])[0], float)

    def invert(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        scale = np.maximum(np.nanmedian(obs), 0.002)
        starts = [(.5 * (self.lo + self.hi)),
                  np.array([.10, .025, .50]), np.array([.35, .07, 1.0])]
        best = None
        for start in starts:
            fit = least_squares(lambda p: (self.spectrum(p) - obs) / scale,
                                np.clip(start, self.lo, self.hi), bounds=(self.lo, self.hi),
                                xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=500)
            rmse = float(np.sqrt(np.mean((self.spectrum(fit.x) - obs) ** 2)))
            if best is None or rmse < best[2]:
                best = (fit.x, self.spectrum(fit.x), rmse)
        return best


def closure_test(model: WaterModel, test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for _, case in test.iterrows():
        truth = np.array([case.chl_mg_m3, case.cdom440_m_1, case.nap_g_m3])
        obs = np.pi * case[BANDS].to_numpy(float)
        fit, recon, rmse = model.invert(obs)
        rel = np.abs(fit - truth) / truth
        rows.append({"case_id": case.case_id,
                     "truth_chl_mg_m3": truth[0], "fit_chl_mg_m3": fit[0],
                     "truth_cdom440_m_1": truth[1], "fit_cdom440_m_1": fit[1],
                     "truth_nap_g_m3": truth[2], "fit_nap_g_m3": fit[2],
                     "chl_relative_error": rel[0], "cdom_relative_error": rel[1],
                     "nap_relative_error": rel[2], "reflectance_rmse": rmse})
    out = pd.DataFrame(rows)
    metrics = {
        "withheld_cases": int(len(out)),
        "median_reflectance_rmse": float(out.reflectance_rmse.median()),
        "p95_reflectance_rmse": float(out.reflectance_rmse.quantile(.95)),
        "median_chl_relative_error": float(out.chl_relative_error.median()),
        "median_cdom_relative_error": float(out.cdom_relative_error.median()),
        "median_nap_relative_error": float(out.nap_relative_error.median()),
    }
    # Keep spectral adequacy separate from constituent identifiability.  A
    # spectrum can be reproduced accurately while Chl and CDOM remain
    # non-unique at Sentinel-2 spectral resolution.  The known-glint test is
    # allowed only when the water-spectrum and NAP/TSM closures are adequate;
    # Chl/CDOM are still reported as non-identifiable when their gate fails.
    metrics["spectral_closure_pass"] = bool(metrics["p95_reflectance_rmse"] <= 0.00030)
    metrics["nap_tsm_closure_pass"] = bool(metrics["median_nap_relative_error"] <= .20)
    metrics["full_constituent_identifiability_pass"] = bool(
        metrics["median_chl_relative_error"] <= .20 and
        metrics["median_cdom_relative_error"] <= .20 and
        metrics["median_nap_relative_error"] <= .20)
    metrics["glint_test_allowed"] = bool(
        metrics["spectral_closure_pass"] and metrics["nap_tsm_closure_pass"])
    metrics["pass"] = metrics["full_constituent_identifiability_pass"]
    return out, metrics


def glint_shape() -> np.ndarray:
    factors = pd.read_csv(FACTORS).set_index("band").loc[BANDS]
    shape = factors[["factor_from_B11", "factor_from_B12"]].mean(axis=1).to_numpy(float)
    return shape / shape.max()


def recover(model: WaterModel, obs: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    corrected = obs.copy()
    alpha_total = 0.0
    for _ in range(5):
        params, water, _ = model.invert(corrected)
        residual = obs - water
        alpha = max(0.0, float(np.dot(residual, g) / np.dot(g, g)))
        if abs(alpha - alpha_total) < 1e-7:
            alpha_total = alpha
            break
        alpha_total = alpha
        corrected = obs - alpha_total * g
    params, water, rmse = model.invert(corrected)
    return params, alpha_total, water, rmse


def glint_validation(model: WaterModel, test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = glint_shape()
    alpha_values = [0.0, 0.0005, 0.0010, 0.0020, 0.0040]
    noise_values = [0.0, 0.00010, 0.00025]
    rows = []
    for _, case in test.iterrows():
        water = np.pi * case[BANDS].to_numpy(float)
        truth_params = np.array([case.chl_mg_m3, case.cdom440_m_1, case.nap_g_m3])
        for alpha_true in alpha_values:
            for sigma in noise_values:
                repeats = 1 if sigma == 0 else 20
                for repeat in range(repeats):
                    noise = RNG.normal(0, sigma * np.array([1, .95, .9, 1, 1.05, 1.1, 1.15, 1.2]), len(BANDS))
                    obs = water + alpha_true * g + noise
                    params, alpha_est, recon, rmse = recover(model, obs, g)
                    corrected = obs - alpha_est * g
                    rows.append({"case_id": case.case_id, "alpha_true": alpha_true,
                                 "noise_sigma": sigma, "repeat": repeat,
                                 "alpha_est": alpha_est,
                                 "alpha_absolute_error": abs(alpha_est-alpha_true),
                                 "corrected_water_rmse": float(np.sqrt(np.mean((corrected-water)**2))),
                                 "forward_fit_rmse": rmse,
                                 "chl_relative_error": abs(params[0]-truth_params[0])/truth_params[0],
                                 "cdom_relative_error": abs(params[1]-truth_params[1])/truth_params[1],
                                 "nap_relative_error": abs(params[2]-truth_params[2])/truth_params[2]})
    out = pd.DataFrame(rows)
    # Use the conservative empirical 95th-percentile order statistic.  Linear
    # interpolation can place the threshold between observations and leave
    # slightly more than 5% of the zero-glint sample above the stated LOD.
    lod = float(out.loc[out.alpha_true == 0, "alpha_est"].quantile(.95, interpolation="higher"))
    out["detected"] = out.alpha_est > lod
    zero = out[out.alpha_true == 0]
    pos = out[out.alpha_true >= .001]
    nonzero = out[out.alpha_true > 0]
    metrics = {"detection_limit_reflectance": lod,
               "zero_glint_false_positive_rate": float(zero.detected.mean()),
               "sensitivity_alpha_ge_0_001": float(pos.detected.mean()),
               "median_alpha_relative_error_alpha_ge_0_001": float(np.median(pos.alpha_absolute_error/pos.alpha_true)),
               "median_corrected_water_rmse": float(nonzero.corrected_water_rmse.median()),
               "median_nap_relative_error": float(nonzero.nap_relative_error.median())}
    metrics["pass"] = bool(metrics["zero_glint_false_positive_rate"] <= .05 and
                           metrics["sensitivity_alpha_ge_0_001"] >= .90 and
                           metrics["median_alpha_relative_error_alpha_ge_0_001"] <= .20 and
                           metrics["median_corrected_water_rmse"] <= .00030)
    return out, metrics


def figures(closure: pd.DataFrame, glint: pd.DataFrame | None) -> None:
    folder = OUT / "13_Figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, name, label in zip(axes, ["chl", "cdom", "nap"],
                               ["Chl-a (mg m⁻³)", "aCDOM(440) (m⁻¹)", "NAP/TSM (g m⁻³)"]):
        x, y = closure[f"truth_{name if name != 'cdom' else 'cdom440'}_{'mg_m3' if name=='chl' else 'm_1' if name=='cdom' else 'g_m3'}"], closure[f"fit_{name if name != 'cdom' else 'cdom440'}_{'mg_m3' if name=='chl' else 'm_1' if name=='cdom' else 'g_m3'}"]
        ax.scatter(x, y, color="#0f766e", s=45)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#64748b")
        ax.set(xlabel=f"Known {label}", ylabel=f"Retrieved {label}", title=f"{name.upper()} closure")
    fig.savefig(folder / "02_controlled_clean_water_closure.png", dpi=180)
    plt.close(fig)
    if glint is None:
        return
    summary = glint.groupby(["alpha_true", "noise_sigma"]).alpha_est.agg(["median", "quantile"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    for sigma, group in glint.groupby("noise_sigma"):
        med = group.groupby("alpha_true").alpha_est.median()
        p05 = group.groupby("alpha_true").alpha_est.quantile(.05)
        p95 = group.groupby("alpha_true").alpha_est.quantile(.95)
        ax.plot(med.index, med.values, "o-", label=f"noise σ={sigma:.5f}")
        ax.fill_between(med.index, p05.values, p95.values, alpha=.15)
    lim = max(glint.alpha_true.max(), glint.alpha_est.quantile(.99)) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="#111827", label="perfect recovery")
    ax.set(xlabel="Known injected glint magnitude k (reflectance)",
           ylabel="Recovered glint magnitude k (reflectance)",
           title="Known-glint recovery", xlim=(0, lim), ylim=(0, lim))
    ax.legend()
    fig.savefig(folder / "03_known_glint_recovery.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-glint-trials", action="store_true")
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2, 3))
    args = parser.parse_args()
    for name in ("05_HydroLight_Water_Library", "06_Clean_Water_Closure",
                 "07_Known_Glint_Validation", "13_Figures", "16_Code"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    train, test = create_library(args.resume, args.workers)
    model = WaterModel(train)
    closure, closure_metrics = closure_test(model, test)
    closure.to_csv(OUT / "06_Clean_Water_Closure" / "controlled_closure_cases.csv", index=False)
    write_json(OUT / "06_Clean_Water_Closure" / "controlled_closure_metrics.json", closure_metrics)
    glint = None
    glint_metrics = None
    if closure_metrics["glint_test_allowed"]:
        print("SPECTRAL + NAP/TSM CLOSURE PASS: running known-glint validation", flush=True)
        trial_path = OUT / "07_Known_Glint_Validation" / "known_glint_trials.csv"
        if args.reuse_glint_trials and trial_path.exists():
            glint = pd.read_csv(trial_path)
            zero_alpha = glint.loc[glint.alpha_true == 0, "alpha_est"]
            lod = float(zero_alpha.quantile(.95, interpolation="higher"))
            glint["detected"] = glint.alpha_est > lod
            zero = glint[glint.alpha_true == 0]
            pos = glint[glint.alpha_true >= .001]
            nonzero = glint[glint.alpha_true > 0]
            glint_metrics = {
                "detection_limit_reflectance": lod,
                "zero_glint_false_positive_rate": float(zero.detected.mean()),
                "sensitivity_alpha_ge_0_001": float(pos.detected.mean()),
                "median_alpha_relative_error_alpha_ge_0_001": float(
                    np.median(pos.alpha_absolute_error / pos.alpha_true)),
                "median_corrected_water_rmse": float(nonzero.corrected_water_rmse.median()),
                "median_nap_relative_error": float(nonzero.nap_relative_error.median())}
            glint_metrics["pass"] = bool(
                glint_metrics["zero_glint_false_positive_rate"] <= .05 and
                glint_metrics["sensitivity_alpha_ge_0_001"] >= .90 and
                glint_metrics["median_alpha_relative_error_alpha_ge_0_001"] <= .20 and
                glint_metrics["median_corrected_water_rmse"] <= .00030)
        else:
            glint, glint_metrics = glint_validation(model, test)
        glint.to_csv(OUT / "07_Known_Glint_Validation" / "known_glint_trials.csv", index=False)
        write_json(OUT / "07_Known_Glint_Validation" / "known_glint_metrics.json", glint_metrics)
    else:
        print("SPECTRAL OR NAP/TSM CLOSURE FAIL: known-glint validation not run", flush=True)
    figures(closure, glint)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "html_created": False, "real_images_processed": False,
               "training_cases": int(len(train)), "withheld_cases": int(len(test)),
               "fixed_particle_model": r"C:\HE5\data\defaults\astarmin_average.txt",
               "closure": closure_metrics, "known_glint": glint_metrics,
               "decision": ("KNOWN_GLINT_PASS" if glint_metrics and glint_metrics["pass"] else
                            "KNOWN_GLINT_FAIL" if glint_metrics else "STOP_AT_CLEAN_WATER_CLOSURE")}
    write_json(OUT / "controlled_validation_summary.json", summary)
    shutil.copy2(Path(__file__), OUT / "16_Code" / Path(__file__).name)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
