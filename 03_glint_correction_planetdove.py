#!/usr/bin/env python
"""03 - Planet Dove (SuperDove, 8 bands, 3 m) sun-glint correction (Hedley et al. 2005) -> 6-band Rrs for the semi-analytical model.

INPUT   one Planet delivery (.zip or folder with PSScene/*_AnalyticMS_SR_8b_clip.tif + *_udm2_clip.tif + *_metadata.json)
        and the date to process (YYYYMMDD). All dates in the delivery are used only for the stable land mask.
OUTPUT  <name>_step5_Rrs_above_surface_6band_443-665nm_sr-1.tif  (3 m, native Planet grid)  -> input of script 04
        <name>_step5_valid_Rrs_mask.tif, <name>_step1_scene_metadata.json, <name>_step5_QC_decision.json

STEPS
  step1  read 8-band surface reflectance (SR) and the UDM2 quality layer of the chosen date
  step2  masks: strict UDM2 clear, stable land (all dates), valid water
  step3  Hedley training water (open water >= 300 m from land) and one least-squares slope per band vs NIR 866 nm
  step4  correction  rho'(band) = rho(band) - slope(band) x (rho866 - min rho866)  + decision (ACCEPT / CAUTIOUS / NO CORRECTION / REJECT)
  step5  Rrs = recommended SR / pi for the 6 visible bands 443-665 nm

Example:
  python 03_glint_correction_planetdove.py --planet input_example/planetdove_SR_UDM2_delivery/PlanetDove_Tidung_20251117_SR_UDM2_delivery.zip \
         --date 20251117 --name PlanetDove_Tidung_20251117 --out output_example/PlanetDove_Tidung_20251117/03_glint_correction_planetdove
"""
import argparse, json, math, sys, zipfile
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

# ============================== RULES AND THRESHOLDS (all in one place) ==============================
WAVES = [444, 492, 533, 566, 612, 666, 707, 866]   # SuperDove band centres (nm), band order of the SR file
TARGETS = WAVES[:-1]                                # bands corrected (866 nm is the NIR reference, unchanged)
SR_SCALE = 10000.0                                  # Planet SR stored as integer x 10000 (Planet product specification)
MOSAIC_RULE = ("all strips of the date on one 3 m grid; largest strip first; a later strip fills pixels with no data or with "
               "UDM2 cloud/shadow/heavy haze (same rule as the PCIS outline code planet_prep.py)")
# UDM2 strict clear (Planet UDM2 bands): 1 clear=1, 2 snow=0, 3 shadow=0, 4 light haze=0, 5 heavy haze=0, 6 cloud=0, 8 unusable=0
LAND_NIR_Q25 = 0.08          # stable land: 25th percentile of NIR 866 nm over all dates > 0.08 ...
LAND_MEDIAN_NDVI = 0.02      # ... AND median NDVI over all dates > 0.02 (project rule; bright glint water is not called land)
LANDLIKE_NDVI = 0.12         # land-like pixel on the date itself: NDVI > 0.12 AND NIR > 0.08 (used only for the distance below)
LANDLIKE_NIR = 0.08
TRAIN_DISTANCE_M = 300.0     # Hedley training water must be >= 300 m from land-like pixels (project rule; email answer)
DEEP_BLUE_OVER_GREEN = 0.90  # training spectral screen: 492 >= 0.90 x 566 ...
DEEP_RED_OVER_GREEN = 0.92   # ... and 666 <= 0.92 x 566  (blue-dominated, deep-looking water)
MIN_TRAIN_SCREENED = 5000    # if fewer screened pixels: use all open water >= 300 m
MIN_TRAIN = 1000             # if fewer than 1000: stop (no reliable regression)
# Decision (project QC rules, not part of the Hedley paper):
REJECT_NEG_PERCENT = 1.0     # REJECT the correction if > 1 % of water pixels become negative in any corrected band
                             # -> the ORIGINAL SR is used (scene not skipped), as in the delivered PCIS Planet results
ACCEPT_R2, ACCEPT_GREEN = 0.45, 5.0     # ACCEPT WITH QC: mean R2 >= 0.45 and median removed at 566 nm >= 5 % of median 566 nm
CAUTIOUS_R2, CAUTIOUS_GREEN = 0.25, 2.0 # CAUTIOUS: mean R2 >= 0.25 and >= 2 %;  otherwise NO CORRECTION NEEDED
USE_CORRECTED = ("ACCEPT WITH QC", "CAUTIOUS")   # recommended product = corrected SR; NO CORRECTION NEEDED or REJECT -> original SR
STAB_ROWS, STAB_COLS = 2, 4  # slope-stability audit: training pixels split into 2 x 4 zones (quantiles of row / column)
STAB_MAX_CV = 25.0           # QC flag if the slope varies by more than 25 % (CV) between zones at 492/533/566 nm, or any zone slope < 0
                             # (project QC used for the delivered PCIS Planet products; not part of the Hedley paper)
RRS_BANDS = [444, 492, 533, 566, 612, 666]       # 6 visible bands used by the model (engine wavelengths 443,490,531,565,610,665)
NODATA = -9999.0
# ======================================================================================================


STEP_DIRS = {"step1": "step1_read_PlanetDove_SR_3m", "step2": "step2_cloud_land_water_masks", "step3": "step3_Hedley_glint_slopes",
             "step4": "step4_glint_corrected_reflectance_and_RGB", "step5": "step5_Rrs_input_for_semi_analytical"}


def step_dir(out, step):
    d = out / STEP_DIRS[step]; d.mkdir(parents=True, exist_ok=True); return d


def log(m):
    print(m, flush=True)


def planet_files(src: Path, work: Path):
    """Return the PSScene folder (extract the zip into <out>/_input_extracted once)."""
    if src.suffix.lower() == ".zip":
        dest = work / "extracted_PlanetDove_delivery"
        if not list(dest.rglob("*_AnalyticMS_SR_8b_clip.tif")):
            with zipfile.ZipFile(src) as z:
                for m in z.infolist():                      # refuse unsafe paths
                    if ".." in Path(m.filename).parts: raise SystemExit(f"unsafe zip member {m.filename}")
                z.extractall(dest)
        src = dest
    folders = sorted({p.parent for p in src.rglob("*_AnalyticMS_SR_8b_clip.tif")})
    if not folders: raise SystemExit(f"no *_AnalyticMS_SR_8b_clip.tif under {src}")
    return folders[0]


class Grid:
    """minimal raster description used by write() for mosaics"""
    def __init__(self, profile):
        self.profile = profile; self.width, self.height = profile["width"], profile["height"]
        self.transform, self.crs = profile["transform"], profile["crs"]


def mosaic_strips(paths, grid=None):
    """Join all strips of one date on one 3 m grid (union of the strips, or the given grid).
    Largest strip first; a later strip fills pixels with no data or with UDM2 cloud / shadow / heavy haze."""
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling
    srcs = [rasterio.open(p) for p in paths]
    if grid is None:
        px = abs(srcs[0].transform.a); L = min(s.bounds.left for s in srcs); T = max(s.bounds.top for s in srcs)
        W = int(round((max(s.bounds.right for s in srcs) - L) / px)); H = int(round((T - min(s.bounds.bottom for s in srcs)) / px))
        prof = srcs[0].profile.copy(); prof.update(width=W, height=H, transform=from_origin(L, T, px, px)); grid = Grid(prof)
    sr = np.zeros((8, grid.height, grid.width), "uint16"); udm = np.zeros((8, grid.height, grid.width), "uint8"); bad = np.zeros((grid.height, grid.width), bool)
    px = abs(grid.transform.a); L, T = grid.transform.c, grid.transform.f
    for s, p in sorted(zip(srcs, paths), key=lambda x: -x[0].width * x[0].height):
        # strips of one delivery share the 3 m pixel alignment: place by row/column offset (no resampling)
        r0, c0 = int(round((T - s.bounds.top) / px)), int(round((s.bounds.left - L) / px))
        rs, cs = max(0, -r0), max(0, -c0); re_, ce = min(s.height, grid.height - r0), min(s.width, grid.width - c0)
        if re_ <= rs or ce <= cs: continue                      # strip outside the grid
        win = rasterio.windows.Window(cs, rs, ce - cs, re_ - rs)
        a = s.read(window=win)
        with rasterio.open(Path(str(p).replace("AnalyticMS_SR_8b_clip", "udm2_clip"))) as u:
            d = u.read(window=win)
        sl = np.s_[r0 + rs:r0 + re_, c0 + cs:c0 + ce]
        have = a.max(0) > 0; badn = (d[5] == 1) | (d[2] == 1) | (d[4] == 1)
        cur, cu, cb = sr[(slice(None),) + sl], udm[(slice(None),) + sl], bad[sl]
        take = have & ((cur.max(0) == 0) | (cb & ~badn))
        cur[:, take] = a[:, take]; cu[:, take] = d[:, take]; cb[take] = badn[take]
    return sr, udm, grid


def read_sr(path):
    with rasterio.open(path) as s:
        raw = s.read().astype(np.float32)
    return {w: np.where(raw[i] == 0, np.nan, raw[i] / SR_SCALE) for i, w in enumerate(WAVES)}


def ols(x, y):
    k = np.isfinite(x) & np.isfinite(y); x, y = x[k].astype(np.float64), y[k].astype(np.float64)
    xm, ym = x.mean(), y.mean(); slope = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    pred = ym + slope * (x - xm); r2 = 1 - float(((y - pred) ** 2).sum() / ((y - ym) ** 2).sum())
    return slope, float(ym - slope * xm), r2, int(x.size)


def write(path, arrays, ref, names=None, nodata=NODATA, dtype="float32", tags=None):
    arrays = arrays if isinstance(arrays, list) else [arrays]
    prof = ref.profile.copy(); [prof.pop(k, None) for k in ("blockxsize", "blockysize", "photometric")]
    prof.update(driver="GTiff", count=len(arrays), dtype=dtype, nodata=nodata, compress="deflate", tiled=ref.width >= 256 and ref.height >= 256)
    if prof["tiled"]: prof.update(blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as d:
        for i, a in enumerate(arrays, 1):
            d.write(np.where(np.isfinite(a), a, nodata).astype(dtype) if dtype == "float32" else a.astype(dtype), i)
            if names: d.set_band_description(i, names[i - 1])
        if tags: d.update_tags(**tags)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planet", required=True, type=Path, help="Planet delivery .zip or folder")
    ap.add_argument("--date", required=True, help="date to process, YYYYMMDD")
    ap.add_argument("--name", required=True); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--training-zone", type=Path, help="optional reviewed Hedley training-water mask (1 = training), same grid as the SR file")
    a = ap.parse_args(); out = a.out; out.mkdir(parents=True, exist_ok=True); N = a.name
    f = lambda step, what: step_dir(out, step) / f"{N}_{step}_{what}"
    qc = {"scene": N, "decision": "PASS", "reasons": [], "warnings": []}

    # ---------------- step1: read SR + UDM2 of the date
    log("step1  read Planet 8-band surface reflectance and UDM2 of ALL strips of the chosen date -> one 3 m mosaic")
    folder = planet_files(a.planet, step_dir(out, "step1"))
    srs = sorted(folder.glob(f"{a.date}_*_AnalyticMS_SR_8b_clip.tif"))
    if not srs: raise SystemExit(f"no scene of {a.date} in {folder}")
    raw, udm, ref = mosaic_strips(srs)
    rho = {w: np.where(raw[i] == 0, np.nan, raw[i] / SR_SCALE) for i, w in enumerate(WAVES)}
    mds = [json.loads((folder / f"{p.name.split('_3B_')[0]}_metadata.json").read_text())["properties"] for p in srs]
    meta = {"date": a.date, "strips": [p.name.split("_3B_")[0] for p in srs], "acquired": [m.get("acquired") for m in mds],
            "solar_zenith_deg": float(np.mean([90.0 - float(m["sun_elevation"]) for m in mds])),
            "sun_azimuth_deg": float(np.mean([m.get("sun_azimuth", np.nan) for m in mds])), "pixel_m": abs(ref.transform.a),
            "grid": f"{ref.width} x {ref.height} px, {ref.crs}", "mosaic_rule": MOSAIC_RULE}
    f("step1", "scene_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write(f("step1", "surface_reflectance_8band_444-866nm.tif"), [rho[w] for w in WAVES], ref, [f"SR {w} nm" for w in WAVES],
          tags={"strips": ",".join(meta["strips"]), "mosaic_rule": MOSAIC_RULE})
    write(f("step1", "UDM2_quality_8band.tif"), list(udm), ref, nodata=None, dtype="uint8")
    if len(srs) > 1: log(f"       {len(srs)} strips joined: {', '.join(meta['strips'])}")

    # ---------------- step2: masks
    log("step2  masks: UDM2 strict clear, stable land (all dates), valid water")
    clear = (udm[0] == 1) & (udm[1] == 0) & (udm[2] == 0) & (udm[3] == 0) & (udm[4] == 0) & (udm[5] == 0) & (udm[7] == 0)
    all_dates = sorted({p.name[:8] for p in folder.glob("*_AnalyticMS_SR_8b_clip.tif")})
    nir, red = [], []
    for d in all_dates:     # every date: its strips mosaicked and put on the grid of the processed date
        r8, _, _ = mosaic_strips(sorted(folder.glob(f"{d}_*_AnalyticMS_SR_8b_clip.tif")), grid=ref)
        nir.append(np.where(r8[7] == 0, np.nan, r8[7] / SR_SCALE)); red.append(np.where(r8[5] == 0, np.nan, r8[5] / SR_SCALE))
    nir, red = np.stack(nir), np.stack(red)
    land = (np.nanpercentile(nir, 25, axis=0) > LAND_NIR_Q25) & (np.nanmedian((nir - red) / np.maximum(nir + red, 1e-6), axis=0) > LAND_MEDIAN_NDVI)
    finite = clear & np.logical_and.reduce([np.isfinite(rho[w]) for w in WAVES])
    water = finite & ~land
    write(f("step2", "UDM2_clear_mask.tif"), clear.astype("uint8"), ref, nodata=0, dtype="uint8")
    write(f("step2", "stable_land_mask_all_dates.tif"), land.astype("uint8"), ref, nodata=0, dtype="uint8",
          tags={"rule": f"NIR q25 > {LAND_NIR_Q25} and median NDVI > {LAND_MEDIAN_NDVI} over {len(all_dates)} dates"})
    write(f("step2", "valid_water_mask.tif"), water.astype("uint8"), ref, nodata=0, dtype="uint8")
    qc["water_pixels"] = int(water.sum())

    # ---------------- step3: training water + regression
    log("step3  Hedley training water (>= 300 m from land) and least-squares slope per band against NIR 866 nm")
    ndvi = (rho[866] - rho[666]) / np.maximum(rho[866] + rho[666], 1e-6)
    landlike = finite & (ndvi > LANDLIKE_NDVI) & (rho[866] > LANDLIKE_NIR)
    dist = distance_transform_edt(finite & ~landlike) * abs(ref.transform.a)
    open_water = water & (dist >= TRAIN_DISTANCE_M)
    train = open_water & (rho[492] >= DEEP_BLUE_OVER_GREEN * rho[566]) & (rho[666] <= DEEP_RED_OVER_GREEN * rho[566])
    if train.sum() < MIN_TRAIN_SCREENED: train = open_water; qc["warnings"].append("spectral training screen relaxed (too few pixels)")
    if a.training_zone:      # optional reviewed training zone (reproduces a hand-refined product)
        with rasterio.open(a.training_zone) as t:
            train = water & (t.read(1) == 1); qc["warnings"].append(f"training water from reviewed mask {a.training_zone.name}")
    if train.sum() < MIN_TRAIN:
        qc.update(decision="SKIP", reasons=[f"only {int(train.sum())} training pixels (< {MIN_TRAIN})"])
        f("step3", "QC_decision.json").write_text(json.dumps(qc, indent=2)); log("SKIP: " + qc["reasons"][0]); return 2
    nir_min = float(np.min(rho[866][train])); delta = rho[866] - nir_min
    fits, corr, removed = [], {w: rho[w].copy() for w in WAVES}, {}
    for w in TARGETS:
        s, icpt, r2, n = ols(rho[866][train], rho[w][train])
        removed[w] = np.where(water, s * delta, np.nan); corr[w] = np.where(water, rho[w] - removed[w], np.nan)
        fits.append({"band_nm": w, "slope": s, "intercept": icpt, "r2": r2, "training_pixels": n,
                     "median_SR_before": float(np.nanmedian(rho[w][water])), "median_removed": float(np.nanmedian(removed[w][water])),
                     "negative_percent_after": float(100 * np.mean(corr[w][water] < 0))})
    corr[866] = np.where(water, rho[866], np.nan)
    # slope-stability audit: same regression in 2 x 4 zones of the training area
    rr, cc = np.where(train); rq = np.quantile(rr, np.linspace(0, 1, STAB_ROWS + 1)); cq = np.quantile(cc, np.linspace(0, 1, STAB_COLS + 1))
    zone_slopes = {w: [] for w in (492, 533, 566)}
    for i in range(STAB_ROWS):
        for j in range(STAB_COLS):
            k = (rr >= rq[i]) & (rr <= rq[i + 1]) & (cc >= cq[j]) & (cc <= cq[j + 1])
            if k.sum() < 100: continue
            for w in zone_slopes: zone_slopes[w].append(ols(rho[866][rr[k], cc[k]], rho[w][rr[k], cc[k]])[0])
    cv = {w: float(100 * np.std(v) / abs(np.mean(v))) if len(v) > 1 else float("nan") for w, v in zone_slopes.items()}
    stable = all(c <= STAB_MAX_CV for c in cv.values()) and min(min(v) for v in zone_slopes.values()) >= 0
    qc["slope_stability"] = {"zones": len(zone_slopes[566]), "cv_percent": {str(k): round(v, 1) for k, v in cv.items()}, "stable": stable}
    if not stable: qc["warnings"].append(f"Hedley slopes not stable between zones (CV {max(cv.values()):.0f} % > {STAB_MAX_CV} %): use result with caution or give --training-zone")
    write(f("step3", "Hedley_training_water_mask.tif"), train.astype("uint8"), ref, nodata=0, dtype="uint8")
    with open(f("step3", "Hedley_glint_slopes.csv"), "w", encoding="utf-8") as h:
        h.write(",".join(fits[0]) + "\n"); [h.write(",".join(str(v) for v in r.values()) + "\n") for r in fits]

    # ---------------- step4: correction + decision
    log("step4  correction rho' = rho - slope x (rho866 - min rho866) and decision")
    neg_any = water & np.logical_or.reduce([corr[w] < 0 for w in TARGETS])
    neg_pct = float(100 * neg_any.sum() / max(water.sum(), 1))
    mean_r2 = float(np.mean([r["r2"] for r in fits])); g = next(r for r in fits if r["band_nm"] == 566)
    green_pct = float(100 * g["median_removed"] / max(abs(g["median_SR_before"]), 1e-9))
    if neg_pct > REJECT_NEG_PERCENT: decision = "REJECT"
    elif mean_r2 >= ACCEPT_R2 and green_pct >= ACCEPT_GREEN: decision = "ACCEPT WITH QC"
    elif mean_r2 >= CAUTIOUS_R2 and green_pct >= CAUTIOUS_GREEN: decision = "CAUTIOUS"
    else: decision = "NO CORRECTION NEEDED"
    recommended = corr if decision in USE_CORRECTED else {w: np.where(water, rho[w], np.nan) for w in WAVES}
    write(f("step4", "glint_corrected_surface_reflectance_8band_444-866nm.tif"), [corr[w] for w in WAVES], ref, [f"corrected SR {w} nm" for w in WAVES])
    write(f("step4", "glint_removed_7band_444-707nm.tif"), [removed[w] for w in TARGETS], ref, [f"removed {w} nm" for w in TARGETS])
    write(f("step4", "RGB_before_glint_correction.tif"), [np.where(water, rho[w], np.nan) for w in (666, 566, 492)], ref, ["red 666", "green 566", "blue 492"])
    write(f("step4", "RGB_after_glint_correction.tif"), [recommended[w] for w in (666, 566, 492)], ref, ["red 666", "green 566", "blue 492"])
    dec = {"decision": decision, "use_for_model": "corrected SR" if decision in USE_CORRECTED else "original SR",
           "mean_R2": mean_r2, "green_566_removed_percent_of_median": green_pct, "negative_any_band_percent": neg_pct,
           "nir_training_minimum": nir_min, "training_pixels": int(train.sum()),
           "rules": f"REJECT (use original SR) if negative > {REJECT_NEG_PERCENT} %; ACCEPT WITH QC if R2 >= {ACCEPT_R2} and green >= {ACCEPT_GREEN} %; "
                    f"CAUTIOUS if R2 >= {CAUTIOUS_R2} and green >= {CAUTIOUS_GREEN} %; else NO CORRECTION NEEDED"}
    f("step4", "glint_decision.json").write_text(json.dumps(dec, indent=2), encoding="utf-8")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    b0 = np.dstack([rho[666], rho[566], rho[492]]); b1 = np.dstack([recommended[666], recommended[566], recommended[492]])
    lo, hi = np.nanpercentile(b0[water], 1, 0), np.nanpercentile(b0[water], 99.5, 0)
    fig, ax = plt.subplots(2, 1, figsize=(14, 12))
    for x, im, t in zip(ax, (np.where(water[..., None], b0, np.nan), b1), ("before (original SR)", f"after ({dec['use_for_model']})")):
        x.imshow(np.clip((np.nan_to_num(im) - lo) / np.maximum(hi - lo, 1e-6), 0, 1) ** 0.85); x.set_title(t); x.axis("off")
    fig.suptitle(f"{N} - Planet glint correction (Hedley) - decision {decision}", fontweight="bold")
    fig.savefig(f("step4", "RGB_before_after_glint_correction.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---------------- step5: Rrs 6 bands
    log("step5  Rrs = recommended SR / pi for 6 bands 443-665 nm (native 3 m grid)")
    rrs = [recommended[w] / math.pi for w in RRS_BANDS]
    valid = np.logical_and.reduce([np.isfinite(r) for r in rrs])
    write(f("step5", "Rrs_above_surface_6band_443-665nm_sr-1.tif"), rrs, ref, [f"Rrs {w} nm sr-1" for w in RRS_BANDS],
          tags={"product": f"Planet SR ({dec['use_for_model']}) / pi", "wavelengths_nm": ",".join(map(str, RRS_BANDS))})
    write(f("step5", "valid_Rrs_mask.tif"), valid.astype("uint8"), ref, nodata=0, dtype="uint8")
    qc.update(glint_decision=decision, mean_R2=round(mean_r2, 3), green_removed_percent=round(green_pct, 2),
              negative_any_band_percent=round(neg_pct, 3), valid_Rrs_pixels=int(valid.sum()))
    if decision == "REJECT": qc["warnings"].append(f"Hedley correction rejected ({neg_pct:.2f} % of water negative > {REJECT_NEG_PERCENT} %): original SR used for the model")
    qc["decision"] = "SKIP" if qc["reasons"] else "PASS"
    f("step5", "QC_decision.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    log(f"done   glint decision {decision} | QC {qc['decision']} | valid Rrs pixels {valid.sum():,}")
    return 0 if qc["decision"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
