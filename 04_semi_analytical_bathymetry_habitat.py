#!/usr/bin/env python
"""04 - Semi-analytical model (Version 42 v2): water depth, bottom-cover fractions and habitat class maps.

INPUT   the Rrs file of script 02 (Sentinel-2, 4 bands) or script 03 (Planet, 6 bands), its valid-Rrs mask and scene metadata (sun angle),
        + ONE way to choose the shallow-water analysis area:
          --shallow-outline  reviewed shallow-water outline raster (1 = shallow)   [used for all delivered PCIS results]
          --gebco            GEBCO elevation raster: water 0 to --max-depth-m deep
          --auto-shallow     image rule: bottom visible where green reflectance exceeds the deep-water level by > 0.010
OUTPUT  <name>_step3_2classes_*.tif, <name>_step4_3classes_*.tif (depth, fractions), <name>_step5_*habitat*.tif,
        <name>_step6_map_from_step3_..., _step4_..., _step5_....png (one PNG per map)

STEPS
  step1  shallow-water analysis mask on the Rrs grid
  step2  deep-water Rrs reference, AUTOMATIC (no GUI): median of >= 3 clean 7 x 7 deep-water windows
  step3  model with 2 bottom types (sand + seagrass)           -> depth, sand, seagrass
  step4  model with 3 bottom types (sand + seagrass + coral)   -> depth, sand, seagrass, coral
  step5  habitat classes: 70 % rule and 50 % rule (largest fraction)
  step6  report-style PNG for every map (coordinates, place name, legend, scale bar, north arrow)

Example (Sentinel-2):
  python 04_semi_analytical_bathymetry_habitat.py --sensor Sentinel2 --glint-dir output_example/Sentinel2_Tidung_20260808/02_glint_correction_sentinel2 \
         --name Sentinel2_Tidung_20260808 --shallow-outline input_example/shallow_water_outline/Sentinel2_Tidung_20260808_shallow_water_outline.tif \
         --place "Tidung Island, Kepulauan Seribu, Indonesia" --out output_example/Sentinel2_Tidung_20260808/04_semi_analytical_bathymetry_habitat
"""
import argparse, csv, glob, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt, uniform_filter

HERE = Path(__file__).resolve().parent
ENGINE = HERE / "support_04_semi_analytical_model_V42v2" / "SOURCE" / "main.py"           # Version 42 v2 engine, unchanged
BOTTOMS = HERE / "support_04_semi_analytical_model_V42v2" / "DATA" / "bottom_reflectances"

# ============================== RULES AND THRESHOLDS (all in one place) ==============================
ENGINE_WAVES = {"Sentinel2": (443, 490, 560, 665), "PlanetDove": (443, 490, 531, 565, 610, 665)}   # fixed by the V42 engine
ENGINE_SENSOR = {"Sentinel2": "Sentinel2", "PlanetDove": "SuperDove"}   # sensor name inside the V42 engine (Planet Dove = SuperDove)
GREEN_INDEX = {"Sentinel2": 2, "PlanetDove": 3}                 # band used for the bottom-visible rule (560 / 565 nm)
OUTLINE_MIN_SHARE = 0.5      # a model pixel is in the mask when >= 50 % of its area lies inside the shallow-water outline
GEBCO_MAX_DEPTH_M = 30.0     # GEBCO option: water 0-30 m deep (PCIS utility create_gebco_shallow_water_mask.py)
AUTO_GREEN_EXCESS = 0.010    # auto option: green REFLECTANCE (Rrs x pi) above the deep-water level by > 0.010 = bottom visible
                             # (threshold T of the PCIS shallow-water outline criteria)
WATER_TYPE = "case1"         # engine water type (clear oceanic water coefficients) - used for all PCIS runs
COEFFICIENT_MODE = "fitted"  # engine coefficients: wavelength-specific fitted set (303 HydroLight conditions)
BOTTOM_2 = ["coral_sand.txt", "avg_seagrass.txt"]                  # 2 classes: sand + seagrass
BOTTOM_3 = ["coral_sand.txt", "avg_seagrass.txt", "avg_coral.txt"] # 3 classes: sand + seagrass + coral
# deep-water reference (step2) - replaces the GUI pixel selection
DEEP_WIN = 7                 # window 7 x 7 pixels
DEEP_MIN_WINDOWS = 3         # at least 3 windows, else SKIP the scene
DEEP_MAX_WINDOWS = 5
DEEP_DISTANCES_M = (2000, 1500, 1000, 600, 300)   # windows must be this far from shallow water / land (first distance with >= 50 candidates)
DEEP_MAX_CV = 0.05           # window homogeneity: coefficient of variation of blue Rrs <= 5 %
DEEP_SPACING_M = (1200, 600, 300)                 # windows at least this far apart (relaxed only if < 3 windows)
# ranking of candidate windows (all-band mean Rrs > 0): Sentinel-2 -> smallest SWIR glint reference (rhog) first
# (reproduces the official Tidung 20260808 reference within 1 %); Planet (no SWIR) -> most homogeneous window (lowest blue CV) first
HABITAT_THRESHOLD = 0.70     # 70 % rule: a type wins only if its fraction >= 0.70, else "mixed"; 50 % rule: the largest fraction wins
CODES = {"sand": 1, "seagrass": 2, "coral": 4}    # habitat codes; 3 mixed, 9 fraction outside 0-1 (unclassified), 0 no data
# ======================================================================================================


STEP_DIRS = {"step1": "step1_shallow_water_mask", "step2": "step2_deep_water_Rrs_reference", "step3": "step3_model_2classes_sand_seagrass",
             "step4": "step4_model_3classes_sand_seagrass_coral", "step5": "step5_habitat_classes", "step6": "step6_PNG_maps_and_QC"}


def step_dir(out, step):
    d = out / STEP_DIRS[step]; d.mkdir(parents=True, exist_ok=True); return d


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


def read(path, band=1):
    with rasterio.open(path) as r:
        a = r.read(band).astype("float64")
        if r.nodata is not None and not np.isnan(r.nodata): a[a == r.nodata] = np.nan
        return a, r.profile


def write(path, arr, prof, nodata=np.nan, dtype="float32", tags=None):
    p = prof.copy(); [p.pop(k, None) for k in ("blockxsize", "blockysize", "photometric")]
    p.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **p) as d:
        d.write(arr.astype(dtype), 1)
        if tags: d.update_tags(**tags)


def onto_grid(src, prof, resampling=Resampling.average, transform_value=None):
    with rasterio.open(src) as s:
        a = s.read(1).astype("float32")
        if transform_value: a = transform_value(a, s.nodata)
        out = np.full((prof["height"], prof["width"]), np.nan, "float32")
        reproject(a, out, src_transform=s.transform, src_crs=s.crs, dst_transform=prof["transform"], dst_crs=prof["crs"],
                  resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
    return out


# ---------------------------------------------------------------- step2 helper
def deep_windows(rrs, valid, notdeep, px_m, rhog=None):
    ok = valid & np.all(np.isfinite(rrs), 0)
    dist = distance_transform_edt(~notdeep) * px_m
    full = uniform_filter(ok.astype(float), DEEP_WIN, mode="constant") > 0.999
    pos = np.all([uniform_filter(np.where(ok, b, 0), DEEP_WIN, mode="constant") > 0 for b in rrs], 0)
    blue = np.where(ok, rrs[1], 0); m = uniform_filter(blue, DEEP_WIN, mode="constant")
    cv = np.sqrt(np.maximum(uniform_filter(blue ** 2, DEEP_WIN, mode="constant") - m ** 2, 0)) / np.maximum(m, 1e-6)
    for dmin in DEEP_DISTANCES_M:
        cand = full & pos & (dist >= dmin) & (cv <= DEEP_MAX_CV)
        if cand.sum() >= 50: break
    rank = cv if rhog is None else uniform_filter(np.nan_to_num(rhog), DEEP_WIN, mode="constant")
    score = np.where(cand, rank, np.inf); order = np.argsort(score, axis=None)
    for spacing in DEEP_SPACING_M:
        sp = spacing / px_m; picked = []
        for idx in order[:500000]:
            if not np.isfinite(score.flat[idx]): break
            r, c = divmod(int(idx), score.shape[1])
            if all((r - a) ** 2 + (c - b) ** 2 >= sp ** 2 for a, b in picked): picked.append((r, c))
            if len(picked) >= DEEP_MAX_WINDOWS: break
        if len(picked) >= DEEP_MIN_WINDOWS: break
    return picked, dmin, spacing, cv, dist


# ---------------------------------------------------------------- step6 helpers (report PNG)
WGS = CRS.from_epsg(4326); SEA = "#DDE7EF"
CLS_COL = {1: "#E3C16F", 2: "#2E8B57", 4: "#E4572E", 3: "#8C6BB1", 9: "#9E9E9E"}
CLS_NAME = {1: "Sand-dominant", 2: "Seagrass-dominant", 4: "Coral-dominant", 3: "Mixed", 9: "Unclassified (fraction outside 0–1)"}


def dms(v, p, n):
    h = p if v >= 0 else n; v = abs(v); d = int(v); m = (v - d) * 60
    if abs(m - 60) < 1e-6: d, m = d + 1, 0
    return f"{d}°{m:04.1f}′{h}" if abs(m - round(m)) > 0.05 else f"{d}°{int(round(m)):02d}′{h}"


def scalebar_north(ax, ext):
    """Scale bar (lower right) and north arrow (upper right) drawn with a fixed size in points."""
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, TextArea, VPacker, HPacker
    from matplotlib.patches import FancyArrow, Rectangle
    x0, x1, y0, y1 = ext; W = x1 - x0; kmd = 111.32 * math.cos(math.radians((y0 + y1) / 2))
    L = [l for l in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if l <= W * kmd / 5][-1]
    fig = ax.figure; fig.canvas.draw(); pts = (L / kmd) / W * ax.get_window_extent().width * 72 / fig.dpi
    bar = DrawingArea(pts, 6)
    bar.add_artist(Rectangle((0, 0), pts / 2, 5, fc="black", ec="black", lw=0.5)); bar.add_artist(Rectangle((pts / 2, 0), pts / 2, 5, fc="white", ec="black", lw=0.5))
    lab = HPacker(children=[TextArea("0"), DrawingArea(max(pts - 40, 4), 1), TextArea(f"{L:g} km")], align="baseline", pad=0, sep=0)
    arrow = DrawingArea(16, 26); arrow.add_artist(FancyArrow(8, 0, 0, 25, width=1.4, head_width=9, head_length=10, length_includes_head=True, color="black"))
    for child, loc in ((VPacker(children=[lab, bar], align="center", pad=0, sep=2), "lower right"),
                       (VPacker(children=[TextArea("N", textprops=dict(fontsize=12, fontweight="bold")), arrow], align="center", pad=0, sep=1), "upper right")):
        box = AnchoredOffsetbox(loc=loc, child=child, pad=0.4, borderpad=0.6, frameon=True)
        box.patch.set(facecolor="white", edgecolor="#666666", linewidth=0.4, alpha=0.9); box.set_zorder(8); ax.add_artist(box)


def png_map(tif, png, kind, title, place, sensor_date, ext, dmax=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap; from matplotlib.patches import Patch, Rectangle
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11})
    with rasterio.open(tif) as s:
        res = abs(s.res[0]) / 111320.0; sc = min(1.0, 2400 / max((ext[1] - ext[0]) / res, (ext[3] - ext[2]) / res))
        w, h = max(200, int((ext[1] - ext[0]) / res * sc)), max(200, int((ext[3] - ext[2]) / res * sc))
        tf = from_bounds(ext[0], ext[2], ext[1], ext[3], w, h); src = s.read(1).astype("float32")
        if s.nodata is not None and not np.isnan(s.nodata): src[src == s.nodata] = np.nan
        if kind == "hab": src[src == 0] = np.nan
        a = np.full((h, w), np.nan, "float32")
        reproject(src, a, src_transform=s.transform, src_crs=s.crs, dst_transform=tf, dst_crs=WGS,
                  resampling=Resampling.nearest if kind == "hab" else Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    lat0 = (ext[2] + ext[3]) / 2; asp = (ext[3] - ext[2]) / ((ext[1] - ext[0]) * math.cos(math.radians(lat0)))
    fig, ax = plt.subplots(figsize=(10.2, min(max(8 * asp, 3.5), 11) + 1.4)); ax.set_facecolor(SEA); e = ext; im = None
    if kind == "depth":
        im = ax.imshow(a, extent=e, cmap="jet", vmin=0, vmax=dmax, interpolation="nearest")
    elif kind == "frac":
        cm = {"Sand": "YlOrBr", "Seagrass": "Greens", "Coral": "Reds"}[title.split()[0]]
        im = ax.imshow(np.where((a >= 0) & (a <= 1), a, np.nan), extent=e, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
        ax.imshow(np.where(a < 0, 1, np.nan), extent=e, cmap=ListedColormap(["#FF00FF"]), interpolation="nearest")
        ax.imshow(np.where(a > 1, 1, np.nan), extent=e, cmap=ListedColormap(["#00D8FF"]), interpolation="nearest")
        ax.legend(handles=[Patch(color="#FF00FF", label="Fraction below 0"), Patch(color="#00D8FF", label="Fraction above 1")], loc="upper left", fontsize=10)
    else:
        x = np.nan_to_num(a).astype(int); rgba = np.zeros(x.shape + (4,)); pres = [c for c in CLS_COL if (x == c).any()]
        for c in pres: rgba[x == c] = list(matplotlib.colors.to_rgb(CLS_COL[c])) + [1]
        ax.imshow(rgba, extent=e, interpolation="nearest")
        pct = lambda c: (x == c).sum() / max((x > 0).sum(), 1) * 100
        ax.legend(handles=[Patch(color=CLS_COL[c], label=f"{CLS_NAME[c]} ({pct(c):.0f} %)" if pct(c) >= 1 else f"{CLS_NAME[c]} (< 1 %)") for c in pres],
                  loc="upper left", bbox_to_anchor=(1.01, 1), title="Habitat class", fontsize=10)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect(1 / math.cos(math.radians(lat0)))
    for axis, (v0, v1), lab in ((ax.xaxis, ext[:2], ("E", "W")), (ax.yaxis, ext[2:], ("N", "S"))):
        st = next((s for s in (0.5 / 60, 1 / 60, 2 / 60, 5 / 60, 10 / 60, 15 / 60, 0.5, 1.0) if (v1 - v0) / s <= 4), 1.0)
        t = np.arange(math.ceil(v0 / st) * st, v1, st); axis.set_ticks(t); axis.set_ticklabels([dms(v, *lab) for v in t])
    ax.grid(color="#6f6f6f", lw=0.35, ls=(0, (2, 2))); ax.tick_params(direction="in", top=True, right=True)
    if im is not None:   # colour bar exactly as tall as the map
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="3%", pad=0.12)).set_label(
            "Depth (m)" if kind == "depth" else f"{title.split(',')[0]} (0–1)")
    scalebar_north(ax, ext)      # fixed size in points -> same look on wide, tall or square maps
    ax.set_title(f"{place}\n{sensor_date} — {title}", fontsize=13, fontweight="bold", loc="left")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor", required=True, choices=["Sentinel2", "PlanetDove"])
    ap.add_argument("--glint-dir", required=True, type=Path, help="output folder of script 02 (Sentinel2) or 03 (PlanetDove)")
    ap.add_argument("--name", required=True, help="scene name used by script 02/03 (file prefix)")
    ap.add_argument("--out", required=True, type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--shallow-outline", type=Path); g.add_argument("--gebco", type=Path); g.add_argument("--auto-shallow", action="store_true")
    ap.add_argument("--max-depth-m", type=float, default=GEBCO_MAX_DEPTH_M)
    ap.add_argument("--place", default="", help="place name written on the PNG maps")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(); N = a.name
    a.out, a.glint_dir = a.out.resolve(), a.glint_dir.resolve()          # the engine runs in its own folder: use absolute paths
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    f = lambda step, what: step_dir(out, step) / f"{N}_{step}_{what}"
    rrs_path = next(a.glint_dir.rglob(f"{N}_step5_Rrs_above_surface_*.tif"))
    meta = json.loads(next(a.glint_dir.rglob(f"{N}_step1_scene_metadata.json")).read_text())
    qcin = json.loads(next(a.glint_dir.rglob(f"{N}_step5_QC_decision.json")).read_text())
    if qcin["decision"] != "PASS": raise SystemExit(f"glint QC says SKIP: {qcin['reasons']}")
    with rasterio.open(rrs_path) as r:
        prof = r.profile; rrs = r.read().astype("float64"); px = abs(r.transform.a)
        if r.nodata is not None and not np.isnan(r.nodata): rrs[rrs == r.nodata] = np.nan
    valid = read(next(a.glint_dir.rglob(f"{N}_step5_valid_Rrs_mask.tif")))[0] == 1
    qc = {"scene": N, "decision": "PASS", "reasons": []}

    # ---------------- step1: shallow-water analysis mask
    log("step1  shallow-water analysis mask")
    if a.shallow_outline:
        share = onto_grid(a.shallow_outline, prof, Resampling.average, lambda x, nd: (x == 1).astype("float32"))
        shallow = (share >= OUTLINE_MIN_SHARE); how = f"reviewed outline {a.shallow_outline.name}, >= {OUTLINE_MIN_SHARE:.0%} of pixel inside"
    elif a.gebco:
        z = onto_grid(a.gebco, prof, Resampling.bilinear)
        shallow = (z <= 0) & (z >= -a.max_depth_m); how = f"GEBCO 0 to {a.max_depth_m} m"
    else:
        shallow = None; how = f"auto: green reflectance > deep-water level + {AUTO_GREEN_EXCESS}"

    # ---------------- step2: deep-water reference (automatic)
    log("step2  deep-water Rrs reference (automatic, >= 3 windows of 7 x 7)")
    notdeep = ~valid | (shallow if shallow is not None else np.zeros_like(valid))
    rh = list(a.glint_dir.rglob(f"{N}_step4_SWIR_glint_reference_rhog.tif"))
    picked, dmin, spacing, cv, dist = deep_windows(list(rrs), valid, notdeep, px, read(rh[0])[0] if rh else None)
    if len(picked) < DEEP_MIN_WINDOWS:
        qc.update(decision="SKIP", reasons=[f"only {len(picked)} deep-water windows (need {DEEP_MIN_WINDOWS})"])
        f("step2", "QC_decision.json").write_text(json.dumps(qc, indent=2)); log("SKIP " + qc["reasons"][0]); return 2
    waves = ENGINE_WAVES[a.sensor]; rows = []
    for i, (r, c) in enumerate(picked, 1):
        win = np.s_[r - 3:r + 4, c - 3:c + 4]; x, y = rasterio.transform.xy(prof["transform"], r, c)
        rows.append({"window": i, "row": r, "column": c, "x": x, "y": y, "distance_from_shallow_or_land_m": float(dist[r, c]),
                     "blue_CV": float(cv[r, c]), **{f"Rrs_{w}": float(np.median(rrs[k][win])) for k, w in enumerate(waves)}})
    with open(f("step2", "deep_water_windows_7x7.csv"), "w", newline="") as h:
        wr = csv.DictWriter(h, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    ref = {"scene": N, **{f"deep_Rrs_{w}_sr-1": float(np.median([rw[f"Rrs_{w}"] for rw in rows])) for w in waves},
           "accepted_ROI_count": len(rows), "solar_zenith_deg": meta["solar_zenith_deg"],
           "aggregation_method": f"median of {len(rows)} window medians; windows >= {dmin} m from shallow/land, >= {spacing} m apart, blue CV <= {DEEP_MAX_CV}"}
    deep_csv = f("step2", "deep_water_Rrs_reference.csv")
    with open(deep_csv, "w", newline="") as h:
        wr = csv.DictWriter(h, fieldnames=list(ref)); wr.writeheader(); wr.writerow(ref)
    log("       deep Rrs = " + ", ".join(f"{ref[f'deep_Rrs_{w}_sr-1']:.4f}" for w in waves) + f"  ({len(rows)} windows, >= {dmin} m)")

    if shallow is None:   # auto rule needs the deep-water level
        green = rrs[GREEN_INDEX[a.sensor]] * math.pi; deep_green = ref[f"deep_Rrs_{waves[GREEN_INDEX[a.sensor]]}_sr-1"] * math.pi
        shallow = green > deep_green + AUTO_GREEN_EXCESS
    mask = shallow & valid & np.all(np.isfinite(rrs), 0)
    write(f("step1", "shallow_water_analysis_mask.tif"), mask.astype("uint8"), prof, nodata=0, dtype="uint8", tags={"rule": how})
    qc.update(mask_rule=how, analysis_pixels=int(mask.sum()))
    if mask.sum() == 0:
        qc.update(decision="SKIP", reasons=["empty shallow-water mask"]); f("step1", "QC_decision.json").write_text(json.dumps(qc)); return 2

    # ---------------- step3 + step4: engine runs
    maps = {}
    for step, k, bottoms in (("step3", "2classes", BOTTOM_2), ("step4", "3classes", BOTTOM_3)):
        log(f"{step}  semi-analytical model, {k} ({', '.join(b[:-4] for b in bottoms)}) on {mask.sum():,} pixels")
        run_root = step_dir(out, step) / "model_run_files"
        if run_root.exists(): shutil.rmtree(run_root)
        run_root.mkdir()
        cfg = {"sensor": ENGINE_SENSOR[a.sensor], "water_type": WATER_TYPE, "coefficient_mode": COEFFICIENT_MODE, "rrs_files": [str(rrs_path)],
               "deep_csv": str(deep_csv), "mask": str(f("step1", "shallow_water_analysis_mask.tif")),
               "bottom_files": [str(BOTTOMS / b) for b in bottoms], "output_root": str(run_root), "workers": a.workers, "run_name": k}
        cp = run_root / "engine_configuration.json"; cp.write_text(json.dumps(cfg, indent=2))
        p = subprocess.run([sys.executable, str(ENGINE), "--run", str(cp)], capture_output=True, text=True, cwd=str(ENGINE.parent))
        done = [d for d in run_root.iterdir() if d.is_dir() and (d / "MAPS" / "depth_m.tif").exists()]
        if not done: raise SystemExit(f"engine failed ({k}):\n" + (p.stdout[-1500:] + p.stderr[-1500:]))
        mp = done[0] / "MAPS"
        for src, nm in (("depth_m", "depth_m"), ("fraction_sand", "fraction_sand"), ("fraction_avg_seagrass", "fraction_seagrass"),
                        ("fraction_avg_coral", "fraction_coral"), ("relative_RMS_percent", "fit_error_relative_RMS_percent"), ("quality_flags", "quality_flags")):
            if (mp / f"{src}.tif").exists():
                dst = f(step, f"{k}_{nm}.tif"); shutil.copy2(mp / f"{src}.tif", dst); maps[(k, nm)] = dst

    # ---------------- step5: habitat classes
    log("step5  habitat classes (70 % rule and 50 % rule)")
    for k in ("2classes", "3classes"):
        names = ["sand", "seagrass"] + (["coral"] if k == "3classes" else [])
        st = np.stack([read(maps[(k, f"fraction_{n}")])[0] for n in names]); fin = np.all(np.isfinite(st), 0)
        phys = fin & np.all((st >= 0) & (st <= 1), 0)
        for tag in ("70pct", "50pct"):
            cls = np.zeros(st.shape[1:], "uint8"); cls[fin & ~phys] = 9
            if tag == "50pct":
                win = np.argmax(st, 0)
                for i, n in enumerate(names): cls[phys & (win == i)] = CODES[n]
            else:
                cls[phys] = 3
                for i, n in enumerate(names): cls[phys & (st[i] >= HABITAT_THRESHOLD)] = CODES[n]
            dst = f("step5", f"{k}_habitat_class_{tag}_rule.tif")
            write(dst, cls, prof, nodata=0, dtype="uint8", tags={"codes": "1 sand, 2 seagrass, 4 coral, 3 mixed, 9 unclassified (fraction <0 or >1), 0 no data",
                                                                  "rule": "largest fraction wins" if tag == "50pct" else f"fraction >= {HABITAT_THRESHOLD} wins, else mixed"})
            maps[(k, f"habitat_{tag}")] = dst

    # ---------------- step6: report PNG
    log("step6  report-style PNG maps")
    png_name = lambda tif: tif.stem.replace(f"{N}_step", f"{N}_step6_map_from_step", 1) + ".png"   # e.g. <name>_step6_map_from_step3_2classes_depth_m.png
    pd_ = step_dir(out, "step6")
    r_, c_ = np.where(mask); x0, y1 = prof["transform"] * (c_.min(), r_.min()); x1, y0 = prof["transform"] * (c_.max() + 1, r_.max() + 1)
    l, b, rr, t = transform_bounds(prof["crs"], WGS, x0, y0, x1, y1); dx, dy = (rr - l) * 0.08, (t - b) * 0.08; ext = (l - dx, rr + dx, b - dy, t + dy)
    d = meta.get("date", ""); sensor_date = ("Sentinel-2 (20 m)" if a.sensor == "Sentinel2" else f"PlanetScope Dove ({px:g} m)") + (f" · {d[6:8]}.{d[4:6]}.{d[:4]}" if d else "")
    for (k, nm), tif in maps.items():
        if nm in ("quality_flags",): continue
        cls_txt = "2 classes (sand, seagrass)" if k == "2classes" else "3 classes (sand, seagrass, coral)"
        if nm == "depth_m":
            v = read(tif)[0]; v = v[np.isfinite(v)]; dmax = float(np.ceil(np.percentile(v, 98))) if v.size else 10
            png_map(tif, pd_ / png_name(tif), "depth", f"Water depth, {cls_txt}", a.place, sensor_date, ext, dmax)
        elif nm.startswith("fraction_") :
            png_map(tif, pd_ / png_name(tif), "frac", f"{nm.split('_')[1].capitalize()} fraction, {cls_txt}", a.place, sensor_date, ext)
        elif nm.startswith("habitat"):
            png_map(tif, pd_ / png_name(tif), "hab", ("Habitat class, 70 % rule" if "70" in nm else "Habitat class, 50 % rule (largest fraction)") + f", {cls_txt}", a.place, sensor_date, ext)
    f("step6", "QC_decision.json").write_text(json.dumps({**qc, "deep_water_reference": ref, "workers": a.workers,
                                                            "outputs": sorted(str(p.relative_to(out)) for p in out.rglob(f"{N}_step*.tif"))}, indent=2))
    log(f"done   {len(maps)} result maps + PNG in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
