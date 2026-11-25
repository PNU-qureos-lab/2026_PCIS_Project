#!/usr/bin/env python
"""02 - Sentinel-2 sun-glint correction (Glint-ACOLITE, Cycle 0) -> above-surface Rrs B01-B04 for the semi-analytical model.

INPUT   one Sentinel-2 Level-2A product (.SAFE folder or .zip) + an area (lon/lat box)
OUTPUT  <name>_step5_Rrs_above_surface_4band_443-665nm_sr-1.tif  (4 bands, 443/490/560/665 nm)  -> input of script 04
        <name>_step5_valid_Rrs_mask.tif, <name>_step1_scene_metadata.json, <name>_step5_QC_decision.json

STEPS (each step writes files whose names start with  <name>_stepN_ ):
  step1  read L2A bands, crop to the area, put all bands on one 20 m grid
  step2  masks: cloud/shadow (+60 m buffer) and valid water
  step3  glint transfer factors from ACOLITE (atmosphere look-up table + Fresnel reflectance)
  step4  glint correction:  corrected(band) = L2A(band) - factor(band) x rhog_SWIR
  step5  Rrs = corrected BOA / pi  + QC decision (PASS / SKIP)

Example:
  python 02_glint_correction_sentinel2.py --safe input_example/sentinel2_L2A_scene/S2A_MSIL2A_20260808T030151_N0512_R032_T48MXU_20260808T123808.SAFE \
         --aoi 106.455 -5.835 106.555 -5.770 --name Sentinel2_Tidung_20260808 --out output_example/Sentinel2_Tidung_20260808/02_glint_correction_sentinel2
"""
import argparse, json, math, sys, xml.etree.ElementTree as ET, zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.ndimage import binary_dilation, distance_transform_edt

HERE = Path(__file__).resolve().parent
ACOLITE_DIR = HERE / "support_02_ACOLITE_code_and_lookup_tables"        # ACOLITE source (GPL-3), only the 3 Sentinel-2 LUTs are included

# ============================== RULES AND THRESHOLDS (all in one place) ==============================
GRID_M = 20                        # analysis grid (m). Project choice: 20 m = native B11/B12 (SWIR glint bands) resolution
BANDS_OUT = ("B01", "B02", "B03", "B04")            # 443, 490, 560, 665 nm: the 4 bands used by the semi-analytical model
WAVE_NM = {"B01": 443, "B02": 490, "B03": 560, "B04": 665}
SCL_WATER = 6                      # Sen2Cor scene class 6 = water (ESA Sen2Cor definition)
SCL_CLOUD = [3, 8, 9, 10, 11]      # Sen2Cor: 3 cloud shadow, 8 cloud medium, 9 cloud high, 10 thin cirrus, 11 snow/ice
SCL_INVALID = [0, 1, 3, 8, 9, 10, 11]   # + 0 no data, 1 saturated/defective
CLOUD_BUFFER_PX = 3                # 3 x 20 m = 60 m buffer around cloud/shadow. Project rule (cloud edges/adjacency excluded)
SWIR_MAX_BOA = 0.05                # B11 BOA above 0.05 = not a pure-glint water pixel (foam, land, cloud edge). Project rule
AEROSOL_MODEL = "MOD2"             # ACOLITE maritime aerosol model (LUT ACOLITE-LUT-202110-MOD2). Project choice for open ocean islands
PRESSURE_HPA = 1013                # standard sea-level pressure for the LUT
MAX_CLOUD_PERCENT = 20.0           # QC gate: SKIP scene if > 20 % of the water/area pixels are cloud/shadow. Project rule for automation
DEEP_DIST_M = 2000                 # "open deep water" = more than 2 km from any non-water pixel (used only for the QC gate below)
DEEP_NEG_LIMIT_BOA = -0.002        # QC gate: SKIP if the ORIGINAL L2A median over open deep water is below -0.002 in B03 or B04
                                   # (Sen2Cor removed too much atmosphere; no valid deep-water reference possible - e.g. Tonga 2025-08-19)
NEG_WARN_PERCENT = 5.0             # QC warning if > 5 % of corrected water pixels are negative in B02
# ======================================================================================================


STEP_DIRS = {"step1": "step1_read_Sentinel2_L2A_20m", "step2": "step2_cloud_land_water_masks", "step3": "step3_ACOLITE_glint_factors",
             "step4": "step4_glint_corrected_reflectance_and_RGB", "step5": "step5_Rrs_input_for_semi_analytical"}


def step_dir(out, step):
    d = out / STEP_DIRS[step]; d.mkdir(parents=True, exist_ok=True); return d


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- reading the .SAFE (folder or zip, no extraction needed)
class Safe:
    def __init__(self, path: Path):
        self.path = path
        if path.suffix.lower() == ".zip":
            self.names = zipfile.ZipFile(path).namelist()
            self.root = lambda inner: f"/vsizip/{path.as_posix()}/{inner}"
            self.read_text = lambda inner: zipfile.ZipFile(path).read(inner).decode("utf-8")
        else:
            self.names = [p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()]
            self.root = lambda inner: str(path / inner)
            self.read_text = lambda inner: (path / inner).read_text(encoding="utf-8")

    def find(self, pattern_end):
        hits = [n for n in self.names if n.endswith(pattern_end)]
        if not hits:
            raise SystemExit(f"File ending with {pattern_end} not found in {self.path}")
        return hits[0]

    def band(self, token, res):
        return self.root(self.find(f"_{token}_{res}m.jp2"))


def ln(tag):
    return tag.split("}")[-1]


def read_metadata(safe: Safe):
    prod = ET.fromstring(safe.read_text(safe.find("MTD_MSIL2A.xml")))
    tile = ET.fromstring(safe.read_text(safe.find("MTD_TL.xml")))
    offsets = {int(e.attrib["band_id"]): float(e.text) for e in prod.iter() if ln(e.tag) == "BOA_ADD_OFFSET"}
    quant = float(next((e.text for e in prod.iter() if ln(e.tag) == "BOA_QUANTIFICATION_VALUE"), "10000"))
    uri = next((e.text for e in prod.iter() if ln(e.tag) == "PRODUCT_URI"), safe.path.name)
    m = {"product": uri, "platform": uri[:3], "date": uri.split("_")[2][:8], "boa_offsets": offsets, "boa_quantification": quant}
    for p in tile.iter():
        if ln(p.tag) == "Mean_Sun_Angle":
            for c in p:
                m["sun_zenith_deg" if ln(c.tag) == "ZENITH_ANGLE" else "sun_azimuth_deg"] = float(c.text)
        elif ln(p.tag) == "Mean_Viewing_Incidence_Angle_List":
            z, a = [], []
            for ang in p:
                for c in ang:
                    if ln(c.tag) == "ZENITH_ANGLE": z.append(float(c.text))
                    elif ln(c.tag) == "AZIMUTH_ANGLE": a.append(float(c.text))
            m["view_zenith_deg"], m["view_azimuth_deg"] = float(np.mean(z)), float(np.mean(a))
    raa = abs(m["sun_azimuth_deg"] - m["view_azimuth_deg"])
    m["relative_azimuth_deg"] = min(raa, 360 - raa)
    m["solar_zenith_deg"] = m["sun_zenith_deg"]            # name used by script 04
    return m


BAND_ID = {"B01": 0, "B02": 1, "B03": 2, "B04": 3, "B11": 11, "B12": 12}


def make_grid(safe: Safe, aoi):
    """20 m grid covering the lon/lat box, aligned to the Sentinel-2 tile pixels."""
    with rasterio.open(safe.band("B11", 20)) as r:
        crs, x0, y0 = r.crs, r.transform.c, r.transform.f
    l, b, rr, t = transform_bounds("EPSG:4326", crs, *aoi)
    l = x0 + math.floor((l - x0) / GRID_M) * GRID_M
    t = y0 - math.floor((y0 - t) / GRID_M) * GRID_M
    w, h = int(math.ceil((rr - l) / GRID_M)), int(math.ceil((t - b) / GRID_M))
    return {"crs": crs, "transform": from_origin(l, t, GRID_M, GRID_M), "width": w, "height": h}


def to_grid(src, g, resampling, dtype="float32"):
    out = np.zeros((g["height"], g["width"]), dtype)
    with rasterio.open(src) as s:
        reproject(rasterio.band(s, 1), out, dst_transform=g["transform"], dst_crs=g["crs"], resampling=resampling, dst_nodata=0)
    return out


def write(path, arrays, g, names=None, nodata=np.nan, dtype="float32", tags=None):
    arrays = arrays if isinstance(arrays, list) else [arrays]
    prof = dict(driver="GTiff", width=g["width"], height=g["height"], count=len(arrays), dtype=dtype, crs=g["crs"],
                transform=g["transform"], nodata=nodata, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as d:
        for i, a in enumerate(arrays, 1):
            d.write(a.astype(dtype), i)
            if names: d.set_band_description(i, names[i - 1])
        if tags: d.update_tags(**tags)


def rgb_png(path, before, after, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def stack(b): return np.dstack([b["B04"], b["B03"], b["B02"]])
    a0, a1 = stack(before), stack(after); ok = np.all(np.isfinite(a1), -1)
    lo, hi = np.nanpercentile(a0[ok], 1, 0), np.nanpercentile(a0[ok], 99.5, 0)          # same stretch for both
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    for x, a, t in zip(ax, (a0, a1), ("before glint correction (L2A)", "after glint correction (Cycle 0)")):
        x.imshow(np.clip((np.nan_to_num(a) - lo) / np.maximum(hi - lo, 1e-6), 0, 1) ** 0.85); x.set_title(t); x.axis("off")
    fig.suptitle(title, fontweight="bold"); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def acolite_factors(meta, aot):
    """ACOLITE optical transfer: factor(band <- SWIR) = [T2(band)/T2(SWIR)] x [Fresnel(band)/Fresnel(SWIR)].
    T2 = two-way direct transmittance exp(-ttot/cos(vza)) * exp(-ttot/cos(sza)) from the ACOLITE LUT at the scene AOT(550);
    Fresnel = sky reflectance of the water surface at the glint geometry. (Same as acolite_helpers.optical_terms of the PCIS package.)"""
    sys.path.insert(0, str(ACOLITE_DIR))
    import acolite as ac
    sensor = {"S2A": "S2A_MSI", "S2B": "S2B_MSI", "S2C": "S2C_MSI_V4"}[meta["platform"]]
    rsrd = ac.shared.rsr_dict(sensor=sensor)[sensor]
    lutdw = ac.aerlut.import_luts(pressures=[PRESSURE_HPA], base_luts=[f"ACOLITE-LUT-202110-{AEROSOL_MODEL}"], sensor=sensor,
                                  lut_par=["ttot"], return_lut_array=True, add_rsky=False)
    lut = list(lutdw)[0]
    sza, vza, raa = meta["sun_zenith_deg"], meta["view_zenith_deg"], meta["relative_azimuth_deg"]
    s, v, r = map(math.radians, (sza, vza, raa))
    omega = math.acos(math.cos(s) * math.cos(v) + math.sin(s) * math.sin(v) * math.cos(r)) / 2
    refri = ac.shared.wopp.refri(); n_w = ac.shared.rsr_convolute_dict(refri["wave"] / 1000, refri["n"], rsrd["rsr"])
    terms = {}
    for name, b in {"B01": "1", "B02": "2", "B03": "3", "B04": "4", "B11": "11", "B12": "12"}.items():
        ttot = float(lutdw[lut]["rgi"][b]((float(PRESSURE_HPA), lutdw[lut]["ipd"]["ttot"], raa, vza, sza, aot)))
        terms[name] = {"wavelength_nm": float(rsrd["wave_nm"][b]), "ttot": ttot,
                       "two_way_transmittance": math.exp(-ttot / math.cos(v)) * math.exp(-ttot / math.cos(s)),
                       "fresnel_reflectance": float(ac.ac.sky_refl(omega, n_w=n_w[b]))}
    fac = {f"{bd}_from_{sw}": terms[bd]["two_way_transmittance"] / terms[sw]["two_way_transmittance"]
           * terms[bd]["fresnel_reflectance"] / terms[sw]["fresnel_reflectance"] for bd in BANDS_OUT for sw in ("B11", "B12")}
    return fac, terms, sensor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safe", required=True, type=Path, help="Sentinel-2 L2A .SAFE folder or .zip")
    ap.add_argument("--aoi", required=True, type=float, nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--name", required=True, help="short scene name used in every output file, e.g. Sentinel2_Tidung_20260808")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-cloud-percent", type=float, default=MAX_CLOUD_PERCENT)
    a = ap.parse_args()
    out = a.out; out.mkdir(parents=True, exist_ok=True); N = a.name
    f = lambda step, what: step_dir(out, step) / f"{N}_{step}_{what}"
    qc = {"scene": N, "decision": "PASS", "reasons": [], "warnings": []}

    # ---------------- step1: read L2A, crop, one 20 m grid
    log("step1  read Sentinel-2 L2A and put B01-B04, B11, B12, SCL, AOT on one 20 m grid")
    safe = Safe(a.safe); meta = read_metadata(safe); g = make_grid(safe, a.aoi)
    boa = {}
    for b in (*BANDS_OUT, "B11", "B12"):
        dn = to_grid(safe.band(b, 60 if b == "B01" else 20), g, Resampling.bilinear)   # B01 is native 60 m, aligned to 20 m
        x = (dn + meta["boa_offsets"].get(BAND_ID[b], 0.0)) / meta["boa_quantification"]  # BOA = (DN + offset) / 10000
        x[dn <= 0] = np.nan; boa[b] = x.astype("float32")
    scl = to_grid(safe.band("SCL", 20), g, Resampling.nearest, "uint8")
    aot = to_grid(safe.band("AOT", 20), g, Resampling.bilinear) / 1000.0                # AOT(550) stored x 1000
    names6 = ["B01 443 nm", "B02 490 nm", "B03 560 nm", "B04 665 nm", "B11 1610 nm", "B12 2190 nm"]
    write(f("step1", "surface_reflectance_6band_443-2190nm.tif"), [boa[b] for b in (*BANDS_OUT, "B11", "B12")], g, names6)
    write(f("step1", "SCL_scene_classification.tif"), scl, g, nodata=0, dtype="uint8")
    write(f("step1", "AOT550.tif"), aot.astype("float32"), g)
    meta.update(grid=f"{g['width']} x {g['height']} px, {GRID_M} m, {g['crs']}", aoi_lonlat=a.aoi)
    f("step1", "scene_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---------------- step2: masks
    log("step2  cloud/shadow mask (+60 m) and valid-water mask")
    cloud = np.isin(scl, SCL_CLOUD); cloud_buf = binary_dilation(cloud, iterations=CLOUD_BUFFER_PX)
    water = (scl == SCL_WATER) & ~np.isin(scl, SCL_INVALID) & ~cloud_buf
    valid = water & np.isfinite(boa["B11"]) & (boa["B11"] <= SWIR_MAX_BOA)
    write(f("step2", "cloud_shadow_mask_60m_buffer.tif"), cloud_buf.astype("uint8"), g, nodata=0, dtype="uint8")
    write(f("step2", "valid_water_mask.tif"), valid.astype("uint8"), g, nodata=0, dtype="uint8",
          tags={"rule": f"SCL=6 water, not cloud/shadow (+{CLOUD_BUFFER_PX * GRID_M} m), B11 <= {SWIR_MAX_BOA}"})
    area = scl > 0
    cloud_pct = float(100 * (cloud & area).sum() / max(area.sum(), 1))
    qc["cloud_shadow_percent"] = round(cloud_pct, 2)
    if cloud_pct > a.max_cloud_percent:
        qc["reasons"].append(f"cloud/shadow {cloud_pct:.1f} % > {a.max_cloud_percent} %")
    dist = distance_transform_edt(scl == SCL_WATER) * GRID_M
    deep = (dist > DEEP_DIST_M) & valid
    if deep.sum() > 100:
        med = {b: float(np.nanmedian(boa[b][deep])) for b in ("B03", "B04")}
        qc["L2A_open_deep_water_median_BOA"] = med
        if min(med.values()) < DEEP_NEG_LIMIT_BOA:
            qc["reasons"].append(f"L2A already negative over open deep water ({med}); Sen2Cor over-correction - no valid deep-water reference")
    else:
        qc["warnings"].append(f"no open water > {DEEP_DIST_M} m from land/cloud: deep-water L2A check not possible")

    # ---------------- step3: ACOLITE glint transfer factors
    aot_used = float(np.nanmedian(aot[valid & (aot > 0)])) if (valid & (aot > 0)).any() else 0.1
    log(f"step3  ACOLITE glint transfer factors (AOT550 = {aot_used:.3f}, {AEROSOL_MODEL} maritime)")
    fac, terms, sensor = acolite_factors(meta, aot_used)
    f("step3", "ACOLITE_glint_transfer_factors.json").write_text(json.dumps(
        {"sensor": sensor, "aerosol_model": AEROSOL_MODEL, "AOT550_median_valid_water": aot_used,
         "formula": "factor(band<-SWIR) = T2(band)/T2(SWIR) x Fresnel(band)/Fresnel(SWIR)", "factors": fac, "terms": terms}, indent=2), encoding="utf-8")

    # ---------------- step4: glint correction (Cycle 0)
    log("step4  glint correction: corrected = L2A - factor x rhog_SWIR (B11 or B12, whichever gives the smaller glint)")
    sw11, sw12 = np.maximum(boa["B11"], 0), np.maximum(boa["B12"], 0)
    use_b11 = fac["B02_from_B11"] * sw11 < fac["B02_from_B12"] * sw12        # per pixel: the more conservative (smaller) estimate
    rhog = np.where(use_b11, sw11, sw12)
    corr, removed = {}, {}
    for b in BANDS_OUT:
        gl = np.where(use_b11, fac[f"{b}_from_B11"] * rhog, fac[f"{b}_from_B12"] * rhog)
        ok = valid & np.isfinite(boa[b])
        corr[b] = np.where(ok, boa[b] - gl, np.nan).astype("float32"); removed[b] = np.where(ok, gl, np.nan).astype("float32")
    names4 = [f"{b} {WAVE_NM[b]} nm" for b in BANDS_OUT]
    write(f("step4", "glint_corrected_surface_reflectance_4band_443-665nm.tif"), [corr[b] for b in BANDS_OUT], g, names4)
    write(f("step4", "glint_removed_4band_443-665nm.tif"), [removed[b] for b in BANDS_OUT], g, names4)
    write(f("step4", "SWIR_glint_reference_rhog.tif"), np.where(valid, rhog, np.nan).astype("float32"), g)
    write(f("step4", "RGB_before_glint_correction.tif"), [boa["B04"], boa["B03"], boa["B02"]], g, ["red B04", "green B03", "blue B02"])
    write(f("step4", "RGB_after_glint_correction.tif"), [corr["B04"], corr["B03"], corr["B02"]], g, ["red B04", "green B03", "blue B02"])
    rgb_png(f("step4", "RGB_before_after_glint_correction.png"), boa, corr, f"{N} - Sentinel-2 glint correction (same colour stretch)")

    # ---------------- step5: Rrs + QC decision
    log("step5  Rrs = corrected BOA / pi  (above-surface remote-sensing reflectance, sr-1)")
    rrs = [corr[b] / math.pi for b in BANDS_OUT]
    final = np.all([np.isfinite(r) for r in rrs], 0)
    write(f("step5", "Rrs_above_surface_4band_443-665nm_sr-1.tif"), rrs, g, [f"Rrs {WAVE_NM[b]} nm sr-1" for b in BANDS_OUT],
          tags={"wavelengths_nm": "443,490,560,665", "product": "Glint-ACOLITE Cycle 0, Rrs = BOA/pi"})
    write(f("step5", "valid_Rrs_mask.tif"), final.astype("uint8"), g, nodata=0, dtype="uint8")
    neg = float(100 * np.mean(corr["B02"][final] < 0)) if final.any() else 100.0
    qc.update(valid_Rrs_pixels=int(final.sum()), negative_B02_percent=round(neg, 3),
              median_glint_removed_BOA={b: float(np.nanmedian(removed[b])) for b in BANDS_OUT},
              SWIR_reference_B11_percent=round(float(100 * use_b11[valid].mean()), 1) if valid.any() else None)
    if neg > NEG_WARN_PERCENT: qc["warnings"].append(f"{neg:.1f} % negative corrected B02 water pixels")
    if not final.any(): qc["reasons"].append("no valid water pixel")
    qc["decision"] = "SKIP" if qc["reasons"] else "PASS"
    f("step5", "QC_decision.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    log(f"done   QC decision: {qc['decision']} {qc['reasons'] or ''} | valid Rrs pixels {final.sum():,}")
    return 0 if qc["decision"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
