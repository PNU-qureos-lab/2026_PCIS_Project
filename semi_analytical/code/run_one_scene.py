"""Run the original four-band HOPE/D2 inversion on one masked Sentinel-2 scene.

This pilot keeps the supplied forward model and optimizer, changes the bottom
library to sand + seagrass, and constructs the original 440/550/640-nm HOPE
initialization values by linear interpolation/extrapolation from B01-B04.
"""
from __future__ import annotations

import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

from model import WAVES, load_water, make_context
from pipeline import solve
from settings import Settings
from step02_hope_pgx import fit_deep_iops


CODE_DIR = Path(__file__).resolve().parent
SCENE = os.environ.get("SEMI_ANALYTICAL_SCENE", "20260808_S2A")
SCENE_DIR = CODE_DIR / "input_values" / f"scene_{SCENE}"
DEEP_DIR = SCENE_DIR
REFERENCE_DIR = CODE_DIR / "input_values" / "reference_data"
WATER_FILE = REFERENCE_DIR / "H2OabDefaults_SEAwater.txt"
SAND_FILE = REFERENCE_DIR / "coral_sand.txt"
SEAGRASS_FILE = REFERENCE_DIR / "avg_seagrass.txt"
OUTPUT = Path(os.environ.get("SEMI_ANALYTICAL_OUTPUT", CODE_DIR.parent / "results_step_by_step" / SCENE))
BANDS = ("B01", "B02", "B03", "B04")

_CTX = None
_LIB = None
_SETTINGS = None


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == r"\end_header") + 1
    rows = []
    for line in lines[start:]:
        if line.strip() == r"\end_data":
            break
        if line.strip():
            rows.append([float(x) for x in line.split()])
    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1]


def linear_sample(waves: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    out = np.interp(targets, waves, values)
    left = targets < waves[0]
    if np.any(left):
        slope = (values[1] - values[0]) / (waves[1] - waves[0])
        out[left] = values[0] + slope * (targets[left] - waves[0])
    right = targets > waves[-1]
    if np.any(right):
        slope = (values[-1] - values[-2]) / (waves[-1] - waves[-2])
        out[right] = values[-1] + slope * (targets[right] - waves[-1])
    return out


def worker_init(context, library, settings):
    global _CTX, _LIB, _SETTINGS
    _CTX, _LIB, _SETTINGS = context, library, settings


def invert_one(item):
    index, observed = item
    try:
        result, _, _, _ = solve(observed, _CTX, _LIB, _SETTINGS)
        return index, True, float(result["z_hat"]), float(result["f_sand"]), float(result["f_seagrass"]), float(result["c4"]), str(result["status"])
    except Exception as exc:
        return index, False, math.nan, math.nan, math.nan, math.nan, type(exc).__name__ + ": " + str(exc)


def write_raster(path: Path, array: np.ndarray, profile: dict, dtype: str, nodata):
    p = profile.copy()
    p.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="deflate", predictor=2)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array.astype(dtype), 1)


def main(limit: int | None = None, workers: int | None = None) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    water_w, water_a, water_bb = load_water(WATER_FILE)
    aw4 = np.interp(WAVES, water_w, water_a)
    bbw4 = np.interp(WAVES, water_w, water_bb)

    with (DEEP_DIR / f"{SCENE}_deep_water_Rrs.csv").open(encoding="utf-8-sig", newline="") as handle:
        deep = next(csv.DictReader(handle))
    deep_rrs = np.array([float(deep[f"deep_Rrs_{int(w)}_sr-1"]) for w in WAVES])
    solar = float(deep["solar_zenith_deg"])

    fitted = fit_deep_iops(deep_rrs, aw4, bbw4, {
        "wavelengths_nm": WAVES.copy(), "Rrs": deep_rrs.copy(), "aw": aw4.copy()
    })
    if not fitted["parameters"]["success"]:
        raise RuntimeError("Deep-water HOPE fit failed: " + fitted["parameters"]["failure_reason"])
    spec = fitted["spectrum"]
    context = make_context(spec["a"], spec["bb"], solar, deep_rrs)

    sw, sr = read_spectrum(SAND_FILE)
    gw, gr = read_spectrum(SEAGRASS_FILE)
    library = {"sand": np.interp(WAVES, sw, sr), "seagrass": np.interp(WAVES, gw, gr)}
    settings = Settings(top_k=3)

    arrays = []
    profile = None
    for band in BANDS:
        path = next(SCENE_DIR.glob(f"{SCENE}_{band}_above_Rrs_approx_sr-1.tif"))
        with rasterio.open(path) as ds:
            arrays.append(ds.read(1).astype("float64"))
            profile = ds.profile
    stack = np.stack(arrays)
    mask_path = SCENE_DIR / f"{SCENE}_shallow_water_mask.tif"
    with rasterio.open(mask_path) as ds:
        shallow = ds.read(1) > 0
    finite = shallow & np.all(np.isfinite(stack), axis=0)
    valid = finite & np.all(stack > 0, axis=0)
    invalid_input = shallow & ~valid
    flat_indices = np.flatnonzero(valid)
    if limit is not None:
        flat_indices = flat_indices[:limit]
    observations = stack.reshape(4, -1)[:, flat_indices].T

    workers = workers or max(1, min(4, (os.cpu_count() or 2) - 1))
    items = ((int(i), obs) for i, obs in zip(flat_indices, observations))
    if workers == 1:
        worker_init(context, library, settings)
        results = map(invert_one, items)
    else:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=worker_init,
                                   initargs=(context, library, settings))
        results = pool.map(invert_one, items, chunksize=16)

    shape = shallow.shape
    depth = np.full(shape, np.nan, dtype="float32")
    sand = np.full(shape, np.nan, dtype="float32")
    grass = np.full(shape, np.nan, dtype="float32")
    cost = np.full(shape, np.nan, dtype="float32")
    status = np.zeros(shape, dtype="uint8")
    status[invalid_input] = 3
    failures = []
    ok = 0
    for n, row in enumerate(results, 1):
        idx, success, z, fs, fg, c4, message = row
        rr, cc = np.unravel_index(idx, shape)
        if success:
            depth[rr, cc], sand[rr, cc], grass[rr, cc], cost[rr, cc] = z, fs, fg, c4
            status[rr, cc] = 1
            ok += 1
        else:
            status[rr, cc] = 2
            failures.append({"flat_index": idx, "row": int(rr), "col": int(cc), "error": message})
        if n == 1 or n % 250 == 0 or n == len(flat_indices):
            print(json.dumps({"processed": n, "total": len(flat_indices), "success": ok, "failed": len(failures)}), flush=True)
    if workers != 1:
        pool.shutdown()

    habitat = np.zeros(shape, dtype="uint8")
    physical = (status == 1) & np.isfinite(sand) & (sand >= 0) & (sand <= 1)
    habitat[physical & (sand >= 0.7)] = 1
    habitat[physical & (grass >= 0.7)] = 2
    habitat[physical & (sand < 0.7) & (grass < 0.7)] = 3
    habitat[(status == 2) | (status == 3) | ((status == 1) & ~physical)] = 4

    write_raster(OUTPUT / f"{SCENE}_depth_m.tif", depth, profile, "float32", np.nan)
    write_raster(OUTPUT / f"{SCENE}_sand_fraction.tif", sand, profile, "float32", np.nan)
    write_raster(OUTPUT / f"{SCENE}_seagrass_fraction.tif", grass, profile, "float32", np.nan)
    write_raster(OUTPUT / f"{SCENE}_C4.tif", cost, profile, "float32", np.nan)
    write_raster(OUTPUT / f"{SCENE}_status.tif", status, profile, "uint8", 0)
    write_raster(OUTPUT / f"{SCENE}_habitat.tif", habitat, profile, "uint8", 0)
    metadata = {
        "scene": SCENE, "experimental": True, "processed_pixels": len(flat_indices),
        "KML_mask_pixels": int(np.count_nonzero(shallow)),
        "positive_four_band_pixels": int(np.count_nonzero(valid)),
        "invalid_or_nonpositive_input_pixels": int(np.count_nonzero(invalid_input)),
        "successful_pixels": ok, "failed_pixels": len(failures), "solar_zenith_deg": solar,
        "deep_Rrs": deep_rrs.tolist(), "aw4": aw4.tolist(), "bbw4": bbw4.tolist(),
        "PGX": {k: fitted["parameters"][k] for k in ("P", "G", "X", "C_IOP")},
        "initialization": "Final code v3 four-band initialization from Sentinel B01-B04",
        "habitat_codes": {"0": "NoData", "1": "sand", "2": "seagrass", "3": "mixed", "4": "uncertain"},
        "fraction_policy": "Raw unbounded optimization retained; habitat assigned only when both fractions are physical",
    }
    (OUTPUT / "pilot_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUTPUT / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    main(args.limit, args.workers)
