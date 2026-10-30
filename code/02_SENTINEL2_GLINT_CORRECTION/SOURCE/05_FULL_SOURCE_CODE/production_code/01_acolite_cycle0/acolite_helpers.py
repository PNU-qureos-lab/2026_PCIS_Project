#!/usr/bin/env python
"""Batch the complete Tidung Sentinel-2 L2A glint-comparison workflow.

The script screens every SAFE product, keeps rejected scenes in the scene-selection
inventory, and processes only scenes that pass the documented local cloud/water
gate. All analysis rasters use one 20 m grid. The three implemented corrections are:

* Hedley et al. style VIS--NIR regression using B8;
* the published simple B12 subtraction in Niroumand-Jadidi et al. (2022), Eq. 2;
* ACOLITE separated residual-glint spectral transfer, adapted to Sen2Cor L2A BOA.

HydroLight is compared against a five-angle SZA reference grid covering the archive.
The simulations remain synthetic plausibility scenarios, not field truth.
"""

from __future__ import annotations

# WORKFLOW ROLE — ACOLITE SUPPORT AND ARCHIVE HELPERS
# These functions read Sentinel metadata, call ACOLITE LUT utilities, construct
# masks, and preserve intermediate products required by the Cycle-0 workflow.

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

# Directly launching a Conda Python on Windows does not always populate PATH
# with the environment DLL directory. Register it before NumPy/Matplotlib load.
_conda_dll_dir = Path(sys.prefix) / "Library" / "bin"
if os.name == "nt" and _conda_dll_dir.exists():
    os.add_dll_directory(str(_conda_dll_dir))
    os.environ["PATH"] = f"{_conda_dll_dir}{os.pathsep}{os.environ.get('PATH', '')}"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, osr
from scipy.ndimage import binary_dilation, distance_transform_edt


ROOT = Path(__file__).resolve().parents[4]  # ACOLITE-Glint Correction workspace
ACOLITE_SOURCE = ROOT / "acolite-source"
sys.path.insert(0, str(ACOLITE_SOURCE))

_plugin_dir = Path(sys.prefix) / "Library" / "lib" / "gdalplugins"
if _plugin_dir.exists():
    os.environ.setdefault("GDAL_DRIVER_PATH", str(_plugin_dir))

import acolite as ac  # noqa: E402


gdal.UseExceptions()
plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 10})

ARCHIVE = Path(r"H:\Sentinel-2\Tidung\L2A")
OUTPUT = ROOT / "outputs" / "Tidung_Glint_Comparison"
HYDRO_GRID_ROOT = ROOT / "outputs" / "hydrolight_tidung_reference_grid_20260903"
BOUNDS = (106.465, -5.825, 106.540, -5.775)  # west, south, east, north
HYDRO_SZA_GRID = (20.0, 25.0, 30.0, 35.0, 40.0)
HYDRO_SZA_TOLERANCE = 2.51
HYDRO_TARGET_VZA = 3.46
HYDRO_TARGET_RAA = 6.73
HYDRO_POLAR_CAP_MAX_VZA = 5.0
HYDRO_CORE_WIND = 5.0
SCL_CLOUD = [3, 8, 9, 10, 11]
SCL_INVALID = [0, 1, 3, 8, 9, 10, 11]
S2_BAND_ID = {"B02": 1, "B03": 2, "B04": 3, "B08": 7, "B11": 11, "B12": 12}
METHODS = ["Original", "Hedley", "NIR_SWIR_B12", "Adapted_ACOLITE"]
FIXED_CLEAR_ROIS = [
    ("C1", 106.52679443359375, -5.7814717292785645),
    ("C2", 106.50865173339844, -5.783599853515625),
    ("C3", 106.48995208740234, -5.7811174392700195),
    ("C4", 106.47344207763672, -5.820680141448975),
    ("C5", 106.49854278564453, -5.820161819458008),
]
HYDRO_CASES = ("PW", "VC", "CLR", "LC")
PUBLICATION = {
    "method": "Simple Sentinel-2 B12 subtraction",
    "citation": "Niroumand-Jadidi et al. (2022), ISPRS Annals V-3-2022, Eq. 2",
    "doi": "10.5194/isprs-annals-V-3-2022-339-2022",
    "equation": "rho_corrected(lambda) = rho(lambda) - rho(B12, ~2200 nm)",
    "original_quantity": "Rrs after ACOLITE DSF",
    "adaptation_here": "Applied algebraically to Sen2Cor L2A BOA surface reflectance; therefore a sensitivity method, not an exact reproduction of the original processing chain.",
}


def local_name(tag: str) -> str:
    return tag.split("}")[-1]


def nearest_hydrolight_sza(sza: float) -> float:
    return float(min(HYDRO_SZA_GRID, key=lambda value: abs(value - sza)))


def find_text(tree: ET.ElementTree, name: str, default=None):
    for element in tree.iter():
        if local_name(element.tag) == name and element.text:
            return element.text.strip()
    return default


def read_metadata(safe: Path) -> dict:
    product = ET.parse(safe / "MTD_MSIL2A.xml")
    tile = ET.parse(next(safe.glob("GRANULE/*/MTD_TL.xml")))
    offsets = {}
    for element in product.iter():
        if local_name(element.tag) == "BOA_ADD_OFFSET":
            offsets[int(element.attrib["band_id"])] = float(element.text)
    meta = {
        "safe": str(safe),
        "scene_id": safe.name,
        "platform": safe.name[:3],
        "date": safe.name.split("_")[2][:8],
        "processing_level": find_text(product, "PROCESSING_LEVEL"),
        "product_start_time": find_text(product, "PRODUCT_START_TIME"),
        "processing_baseline": safe.name.split("_")[3],
        "boa_quantification_value": float(find_text(product, "BOA_QUANTIFICATION_VALUE", "10000")),
        "boa_offsets": offsets,
        "granule_mean_aot_550": float(find_text(product, "GRANULE_MEAN_AOT", "nan")),
        "granule_mean_water_vapour_cm": float(find_text(product, "GRANULE_MEAN_WV", "nan")),
    }
    for parent in tile.iter():
        if local_name(parent.tag) == "Mean_Sun_Angle":
            for child in parent:
                meta[f"sun_{local_name(child.tag).lower()}"] = float(child.text)
        elif local_name(parent.tag) == "Mean_Viewing_Incidence_Angle_List":
            zenith, azimuth = [], []
            for angle in parent:
                for child in angle:
                    if local_name(child.tag) == "ZENITH_ANGLE":
                        zenith.append(float(child.text))
                    elif local_name(child.tag) == "AZIMUTH_ANGLE":
                        azimuth.append(float(child.text))
            meta["view_zenith_angle"] = float(np.mean(zenith))
            meta["view_azimuth_angle"] = float(np.mean(azimuth))
    raa = abs(meta["sun_azimuth_angle"] - meta["view_azimuth_angle"])
    meta["relative_azimuth_angle"] = min(raa, 360 - raa)
    meta["glint_angle"] = float(ac.shared.glint_angle(
        meta["sun_zenith_angle"], meta["view_zenith_angle"], meta["relative_azimuth_angle"]
    ))
    return meta


def find_band(safe: Path, token: str, resolution: int) -> Path:
    matches = list(safe.glob(f"GRANULE/*/IMG_DATA/R{resolution}m/*_{token}_{resolution}m.jp2"))
    if not matches:
        raise FileNotFoundError(f"{token}_{resolution}m missing in {safe.name}")
    return matches[0]


def bounds_in_projection(projection: str) -> tuple[float, float, float, float]:
    west, south, east, north = BOUNDS
    src = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    dst = osr.SpatialReference()
    dst.ImportFromWkt(projection)
    if hasattr(src, "SetAxisMappingStrategy"):
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src, dst)
    points = [transform.TransformPoint(x, y) for x, y in [(west, south), (west, north), (east, south), (east, north)]]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def read_warp(path: Path, reference: dict | None = None, resample: str = "near") -> tuple[np.ndarray, dict]:
    src = gdal.Open(str(path))
    if src is None:
        raise RuntimeError(f"Could not open {path}")
    if reference is None:
        proj = src.GetProjection()
        bounds = bounds_in_projection(proj)
        opts = dict(outputBounds=bounds, xRes=20, yRes=20, targetAlignedPixels=True, dstSRS=proj)
    else:
        gt = reference["geotransform"]
        width, height = reference["width"], reference["height"]
        bounds = (gt[0], gt[3] + height * gt[5], gt[0] + width * gt[1], gt[3])
        opts = dict(outputBounds=bounds, width=width, height=height, dstSRS=reference["projection"])
    warped = gdal.Warp("", src, format="MEM", resampleAlg=resample, **opts)
    data = np.asarray(warped.ReadAsArray())
    ref = {
        "geotransform": warped.GetGeoTransform(),
        "projection": warped.GetProjection(),
        "width": warped.RasterXSize,
        "height": warped.RasterYSize,
    }
    warped = None
    src = None
    return data, ref


def write_tif(path: Path, data: np.ndarray, ref: dict, nodata=None, dtype=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if dtype is None:
        dtype = gdal.GDT_Float32 if np.issubdtype(data.dtype, np.floating) else gdal.GDT_Byte
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), ref["width"], ref["height"], 1, dtype,
                       options=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2"])
    ds.SetGeoTransform(ref["geotransform"])
    ds.SetProjection(ref["projection"])
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    band.WriteArray(data)
    band.FlushCache()
    ds = None


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def publish_method_names(root: Path):
    """Replace the legacy internal method token in human-facing text products.

    Calculations retain the historical dictionary key so this publication-only
    change cannot alter any numerical result.
    """
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        revised = content.replace("Adapted_ACOLITE", "Glint_ACOLITE").replace("Adapted ACOLITE", "Glint-ACOLITE")
        if revised != content:
            path.write_text(revised, encoding="utf-8")


def json_dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def reflectance(raw: np.ndarray, meta: dict, band: str) -> np.ndarray:
    offset = meta["boa_offsets"].get(S2_BAND_ID[band], 0.0)
    out = (raw.astype(np.float32) + offset) / meta["boa_quantification_value"]
    out[raw == 0] = np.nan
    return out


def stretch_rgb(bands: dict[str, np.ndarray], corrected=None) -> np.ndarray:
    source = corrected or bands
    rgb = np.stack([source["B04"], source["B03"], source["B02"]], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb = np.clip((rgb + 0.01) / 0.16, 0, 1)
    return np.power(rgb, 0.72)


def scene_key(meta: dict) -> str:
    return f"{meta['date']}_{meta['platform']}"


def plot_scene_screen(path: Path, rgb: np.ndarray, scl: np.ndarray, aot: np.ndarray, meta: dict, stats: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), constrained_layout=True)
    axes[0].imshow(rgb); axes[0].set_title("L2A RGB")
    axes[1].imshow(scl, vmin=0, vmax=11, cmap="tab20"); axes[1].set_title("SCL classes")
    axes[2].imshow(scl == 6, vmin=0, vmax=1, cmap="Blues"); axes[2].set_title("SCL water mask")
    im = axes[3].imshow(aot, vmin=0, vmax=max(0.5, np.nanpercentile(aot, 99)), cmap="magma")
    axes[3].set_title("AOT 550 nm"); fig.colorbar(im, ax=axes[3], fraction=0.046)
    for ax in axes: ax.axis("off")
    fig.suptitle(f"{scene_key(meta)} | {stats['decision']} | water {stats['local_water_percent']:.1f}% | cloud/shadow {stats['local_cloud_percent']:.1f}%")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def screen_scene(safe: Path, out: Path) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    meta = read_metadata(safe)
    scl, ref = read_warp(find_band(safe, "SCL", 20), resample="near")
    aot_raw, _ = read_warp(find_band(safe, "AOT", 20), reference=ref, resample="bilinear")
    aot = aot_raw.astype(np.float32) / 1000.0
    total = scl.size
    cloud = np.isin(scl, SCL_CLOUD)
    water = scl == 6
    valid = ~np.isin(scl, SCL_INVALID)
    reference_sza = nearest_hydrolight_sza(meta["sun_zenith_angle"])
    sza_difference = float(abs(meta["sun_zenith_angle"] - reference_sza))
    vza_in_polar_cap = bool(meta["view_zenith_angle"] <= HYDRO_POLAR_CAP_MAX_VZA)
    stats = {
        **{k: v for k, v in meta.items() if k != "boa_offsets"},
        "local_pixels": int(total),
        "local_water_pixels": int(water.sum()),
        "local_water_percent": float(100 * water.mean()),
        "local_cloud_percent": float(100 * cloud.mean()),
        "local_invalid_percent": float(100 * (~valid).mean()),
        "hydrolight_reference_sza_deg": reference_sza,
        "hydrolight_sza_difference_deg": sza_difference,
        "hydrolight_target_vza_deg": HYDRO_TARGET_VZA,
        "hydrolight_target_relative_azimuth_deg": HYDRO_TARGET_RAA,
        "hydrolight_polar_cap_max_vza_deg": HYDRO_POLAR_CAP_MAX_VZA,
        "hydrolight_vza_in_polar_cap": vza_in_polar_cap,
    }
    passed = stats["local_water_percent"] >= 95.0 and stats["local_cloud_percent"] <= 2.0
    stats["decision"] = "ACCEPT" if passed else "REJECT"
    stats["decision_reason"] = "passes local >=95% water and <=2% cloud/shadow/snow gate" if passed else "fails local >=95% water and <=2% cloud/shadow/snow gate"
    stats["hydrolight_geometry_eligible"] = sza_difference <= HYDRO_SZA_TOLERANCE and vza_in_polar_cap
    stats["hydrolight_direction_note"] = "M-file Rrs is the theta=0 near-nadir 0-5 degree polar-cap quad average; relative azimuth is not resolved inside that cap"
    b2_raw, _ = read_warp(find_band(safe, "B02", 20), reference=ref, resample="bilinear")
    b3_raw, _ = read_warp(find_band(safe, "B03", 20), reference=ref, resample="bilinear")
    b4_raw, _ = read_warp(find_band(safe, "B04", 20), reference=ref, resample="bilinear")
    rgb_bands = {"B02": reflectance(b2_raw, meta, "B02"), "B03": reflectance(b3_raw, meta, "B03"), "B04": reflectance(b4_raw, meta, "B04")}
    plot_scene_screen(out / "figures" / f"{scene_key(meta)}_scene_screen.png", stretch_rgb(rgb_bands), scl, aot, meta, stats)
    return meta, ref, scl, aot, stats


def band_stats(scene: str, name: str, arr: np.ndarray, mask: np.ndarray) -> dict:
    values = arr[mask & np.isfinite(arr)]
    if values.size == 0:
        return {"scene": scene, "band": name, "n": 0}
    return {
        "scene": scene, "band": name, "n": int(values.size), "min": float(values.min()),
        "max": float(values.max()), "mean": float(values.mean()), "median": float(np.median(values)),
        "sd": float(values.std()), "negative_percent": float(100 * np.mean(values < 0)),
        "greater_than_1_percent": float(100 * np.mean(values > 1)),
    }


def load_bands(safe: Path, meta: dict, ref: dict) -> dict[str, np.ndarray]:
    out = {}
    for band, resolution, resample in [
        ("B02", 20, "bilinear"), ("B03", 20, "bilinear"), ("B04", 20, "bilinear"),
        ("B08", 10, "average"), ("B11", 20, "bilinear"), ("B12", 20, "bilinear"),
    ]:
        raw, _ = read_warp(find_band(safe, band, resolution), reference=ref, resample=resample)
        out[band] = reflectance(raw, meta, band)
    return out


def pixel_lonlat(ref: dict) -> tuple[np.ndarray, np.ndarray]:
    gt = ref["geotransform"]
    cols = np.arange(ref["width"]) + 0.5
    rows = np.arange(ref["height"]) + 0.5
    x = gt[0] + cols * gt[1]
    y = gt[3] + rows * gt[5]
    xx, yy = np.meshgrid(x, y)
    src = osr.SpatialReference(); src.ImportFromWkt(ref["projection"])
    dst = osr.SpatialReference(); dst.ImportFromEPSG(4326)
    if hasattr(src, "SetAxisMappingStrategy"):
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src, dst)
    flat = np.column_stack([xx.ravel(), yy.ravel()])
    coords = np.asarray(transform.TransformPoints(flat))
    return coords[:, 0].reshape(xx.shape), coords[:, 1].reshape(yy.shape)


def nearest_pixel(lon: np.ndarray, lat: np.ndarray, x: float, y: float) -> tuple[int, int]:
    return tuple(int(v) for v in np.unravel_index(np.nanargmin((lon - x) ** 2 + (lat - y) ** 2), lon.shape))


def build_rois(valid_water: np.ndarray, bands: dict, ref: dict, rhog_proxy: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], list[dict]]:
    distance = distance_transform_edt(valid_water) * 20.0
    roi_a = valid_water & (distance >= 500) & np.isfinite(bands["B08"]) & np.isfinite(bands["B11"])
    roi_a &= bands["B11"] < 0.03
    lon, lat = pixel_lonlat(ref)
    roi_c = np.zeros_like(valid_water, dtype=bool)
    definitions = []
    for roi_id, x, y in FIXED_CLEAR_ROIS:
        row, col = nearest_pixel(lon, lat, x, y)
        box = np.zeros_like(valid_water, dtype=bool)
        box[max(0, row - 3):row + 4, max(0, col - 3):col + 4] = True
        box &= valid_water
        roi_c |= box
        definitions.append({"roi_id": roi_id, "purpose": "ROI-C clear-water HydroLight comparison", "center_lon": x, "center_lat": y, "valid_pixels": int(box.sum())})
    proxy = rhog_proxy if rhog_proxy is not None else np.maximum(bands["B12"], 0)
    candidate_b = roi_a & np.isfinite(proxy)
    threshold = float(np.nanpercentile(proxy[candidate_b], 85)) if candidate_b.any() else float("nan")
    low_threshold = float(np.nanpercentile(proxy[candidate_b], 15)) if candidate_b.any() else float("nan")
    roi_b = candidate_b & (proxy >= threshold)
    roi_e = candidate_b & (proxy <= low_threshold)
    roi_d = valid_water & (distance >= 40) & (distance <= 300)
    definitions.extend([
        {"roi_id": "A", "purpose": "deep-water Hedley regression", "valid_pixels": int(roi_a.sum()), "selection": "valid SCL water; >=500 m from non-water; B11 < 0.03"},
        {"roi_id": "B", "purpose": "glint-affected evaluation", "valid_pixels": int(roi_b.sum()), "selection": f"top 15% glint proxy within ROI-A; threshold={threshold:.8f}"},
        {"roi_id": "C", "purpose": "fixed clear-water HydroLight comparison", "valid_pixels": int(roi_c.sum()), "selection": "five fixed geographic boxes defined independently from correction results"},
        {"roi_id": "D", "purpose": "shallow/benthic preservation", "valid_pixels": int(roi_d.sum()), "selection": "valid water 40-300 m from non-water"},
        {"roi_id": "E", "purpose": "held-out low-glint negative-control evaluation", "valid_pixels": int(roi_e.sum()), "selection": f"bottom 15% glint proxy within ROI-A; threshold={low_threshold:.8f}; never used to fit correction"},
    ])
    return {"A": roi_a, "B": roi_b, "C": roi_c, "D": roi_d, "E": roi_e}, definitions


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 3 or np.nanstd(x) == 0:
        return float("nan"), float("nan"), float("nan"), int(x.size)
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    denom = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - pred) ** 2) / denom if denom > 0 else np.nan
    return float(slope), float(intercept), float(r2), int(x.size)


def run_hedley(bands: dict, roi_a: np.ndarray, valid_water: np.ndarray) -> tuple[dict, dict, list[dict], float, np.ndarray]:
    x_all = bands["B08"]
    x = x_all[roi_a]
    lo, hi = np.nanpercentile(x, [5, 95])
    training = roi_a & (x_all >= lo) & (x_all <= hi)
    nir_min = float(np.nanpercentile(x_all[training], 1))
    corrected, removed, stats = {}, {}, []
    for band in ["B02", "B03", "B04"]:
        slope, intercept, r2, n = linear_fit(x_all[training], bands[band][training])
        delta = slope * (x_all - nir_min)
        # The comparison product exists only in the common valid-water domain.
        # Never retain uncorrected land or atmospheric-contamination pixels.
        out = np.full_like(bands[band], np.nan)
        out[valid_water] = bands[band][valid_water] - delta[valid_water]
        corrected[band], removed[band] = out, np.where(valid_water, delta, np.nan)
        stats.append({"band": band, "reference": "B08", "slope": slope, "intercept": intercept, "R2": r2, "N": n, "nir_min": nir_min})
    return corrected, removed, stats, nir_min, training


_LUT_CACHE = {}


def sensor_info(platform: str):
    sensor = {"S2A": "S2A_MSI", "S2B": "S2B_MSI", "S2C": "S2C_MSI_V4"}[platform]
    rsrd = ac.shared.rsr_dict(sensor=sensor)[sensor]
    return sensor, rsrd


def optical_terms(meta: dict, aot_used: float, model: str) -> tuple[list[dict], dict[str, float]]:
    sensor, rsrd = sensor_info(meta["platform"])
    model_lut = {"C": "ACOLITE-LUT-202110-MOD1", "M": "ACOLITE-LUT-202110-MOD2"}[model]
    key = (sensor, model)
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = ac.aerlut.import_luts(
            pressures=[1013], base_luts=[model_lut], sensor=sensor,
            lut_par=["ttot"], return_lut_array=True, add_rsky=False,
        )
    lutdw = _LUT_CACHE[key]
    lut = list(lutdw)[0]
    raa, vza, sza = meta["relative_azimuth_angle"], meta["view_zenith_angle"], meta["sun_zenith_angle"]
    xi = [1013.0, raa, vza, sza]
    sza_r, vza_r, raa_r = map(np.radians, [sza, vza, raa])
    omega = np.arccos(np.cos(sza_r) * np.cos(vza_r) + np.sin(sza_r) * np.sin(vza_r) * np.cos(raa_r)) / 2
    refri = ac.shared.wopp.refri()
    refri_sen = ac.shared.rsr_convolute_dict(refri["wave"] / 1000, refri["n"], rsrd["rsr"])
    band_map = {"B02": "2", "B03": "3", "B04": "4", "B08": "8", "B11": "11", "B12": "12"}
    rows, factors = [], {}
    mus, muv = np.cos(sza_r), np.cos(vza_r)
    for name, b in band_map.items():
        ttot = float(lutdw[lut]["rgi"][b]((xi[0], lutdw[lut]["ipd"]["ttot"], xi[1], xi[2], xi[3], aot_used)))
        trans = float(np.exp(-ttot / muv) * np.exp(-ttot / mus))
        fresnel = float(ac.ac.sky_refl(omega, n_w=refri_sen[b]))
        rows.append({"band": name, "wavelength_nm": float(rsrd["wave_nm"][b]), "aerosol_model": {"M": "maritime", "C": "continental"}[model], "AOT_550_used": aot_used, "ttot": ttot, "two_way_direct_transmittance": trans, "water_refractive_index": float(refri_sen[b]), "fresnel_reflectance": fresnel})
    lookup = {r["band"]: r for r in rows}
    for name in band_map:
        factors[f"{name}_from_B11"] = lookup[name]["two_way_direct_transmittance"] / lookup["B11"]["two_way_direct_transmittance"] * lookup[name]["fresnel_reflectance"] / lookup["B11"]["fresnel_reflectance"]
        factors[f"{name}_from_B12"] = lookup[name]["two_way_direct_transmittance"] / lookup["B12"]["two_way_direct_transmittance"] * lookup[name]["fresnel_reflectance"] / lookup["B12"]["fresnel_reflectance"]
    return rows, factors


def run_acolite(bands: dict, valid_water: np.ndarray, terms: list[dict], factors: dict) -> dict:
    mask = valid_water & np.isfinite(bands["B11"]) & (bands["B11"] <= 0.05)
    swir1 = np.maximum(bands["B11"], 0)
    swir2 = np.maximum(bands["B12"], 0)
    candidate_b11 = factors["B02_from_B11"] * swir1
    candidate_b12 = factors["B02_from_B12"] * swir2
    use_b11 = candidate_b11 < candidate_b12
    rhog_ref = np.where(use_b11, swir1, swir2)
    corrected, removed, scaling = {}, {}, []
    for band in ["B02", "B03", "B04"]:
        f11, f12 = factors[f"{band}_from_B11"], factors[f"{band}_from_B12"]
        glint = np.where(use_b11, f11 * rhog_ref, f12 * rhog_ref)
        glint = np.where(mask, glint, np.nan)
        # Retain only the pixels that pass both the shared valid-water mask and
        # the ACOLITE SWIR-reference validity check. All other pixels are NoData.
        out = np.full_like(bands[band], np.nan)
        out[mask] = bands[band][mask] - glint[mask]
        corrected[band], removed[band] = out, glint
        scaling.append({"band": band, "factor_from_B11": f11, "factor_from_B12": f12})
    return {"mask": mask, "use_b11": use_b11 & mask, "candidate_B11_blue": np.where(mask, candidate_b11, np.nan), "candidate_B12_blue": np.where(mask, candidate_b12, np.nan), "rhog_ref": np.where(mask, rhog_ref, np.nan), "corrected": corrected, "removed": removed, "scaling": scaling}


def parse_hydrolight(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith('"Rrs"'))
    header_shape = re.search(r"\s(\d+)\s+(\d+)\s*$", lines[start])
    expected_rows = int(header_shape.group(1)) if header_shape else 101
    wave, rrs = [], []
    for line in lines[start + 3:]:
        values = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)
        if len(values) < 2:
            if wave: break
            continue
        wave.append(float(values[0])); rrs.append(float(values[1]))
        if len(wave) >= expected_rows:
            break
    return np.asarray(wave), np.asarray(rrs)


def hydrolight_grid_runs() -> list[dict]:
    manifest_path = HYDRO_GRID_ROOT / "hydrolight_reference_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"HydroLight reference grid manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text())["runs"]


def convolve_hydrolight_grid(platform: str, runs: list[dict]) -> dict[tuple[float, float, str], dict[str, float]]:
    _, rsrd = sensor_info(platform)
    output = {}
    for run in runs:
        case = run["case"]
        path = Path(run["result_dir"]) / f"M{run['root_name']}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed HydroLight result: {path}")
        wave, rrs = parse_hydrolight(path)
        conv = ac.shared.rsr_convolute_dict(wave / 1000, rrs, rsrd["rsr"])
        key = (float(run["sza_deg"]), float(run["wind_m_s"]), case)
        output[key] = {f"B{band.zfill(2)}" if band.isdigit() else f"B{band}": float(value) for band, value in conv.items()}
    return output


def core_hydrolight_references(convolved: dict[tuple[float, float, str], dict[str, float]]) -> dict[float, dict[str, dict[str, float]]]:
    return {
        sza: {case: convolved[(sza, HYDRO_CORE_WIND, case)] for case in HYDRO_CASES}
        for sza in HYDRO_SZA_GRID
    }


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or np.std(a[good]) == 0 or np.std(b[good]) == 0:
        return float("nan")
    return float(np.corrcoef(a[good], b[good])[0, 1])


def qc_metrics(scene: str, method: str, source: dict, corrected: dict, removed: dict, rois: dict, hydro: dict, hydro_ok: bool) -> list[dict]:
    rows = []
    for band in ["B02", "B03", "B04"]:
        valid = rois["B"] & np.isfinite(corrected[band])
        vals = corrected[band][valid]
        slope, _, r2, n = linear_fit(source["B08"][rois["A"]], corrected[band][rois["A"]])
        shallow = rois["D"]
        no_glint = rois["E"] & np.isfinite(source[band]) & np.isfinite(corrected[band])
        no_glint_values = corrected[band][no_glint]
        row = {
            "scene": scene, "method": method, "band": band, "n_eval": int(vals.size),
            "residual_VIS_B8_slope": slope, "residual_VIS_B8_R2": r2,
            "negative_percent": float(100 * np.mean(vals < 0)) if vals.size else np.nan,
            "suspicious_high_percent": float(100 * np.mean(vals > 0.2)) if vals.size else np.nan,
            "mean_signal_removed": float(np.nanmean(removed[band][valid])) if valid.any() else np.nan,
            "shallow_preservation_correlation": correlation(source[band][shallow], corrected[band][shallow]),
            "no_glint_control_mean_abs_change": float(np.nanmean(np.abs(corrected[band][no_glint] - source[band][no_glint]))) if no_glint.any() else np.nan,
            "no_glint_control_negative_percent": float(100 * np.mean(no_glint_values < 0)) if no_glint_values.size else np.nan,
            "hydrolight_geometry_eligible": hydro_ok,
        }
        roi_c_vals = corrected[band][rois["C"]] / np.pi
        row["ROI_C_mean_Rrs_approx"] = float(np.nanmean(roi_c_vals)) if np.isfinite(roi_c_vals).any() else np.nan
        for case in HYDRO_CASES:
            ref = hydro[case][band]
            row[f"HydroLight_{case}_Rrs"] = ref
            row[f"HydroLight_{case}_difference"] = row["ROI_C_mean_Rrs_approx"] - ref if hydro_ok else np.nan
        rows.append(row)
    return rows


def rmse_summary(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["scene"], row["method"]), []).append(row)
    output = []
    for (scene, method), items in groups.items():
        result = {"scene": scene, "method": method, "hydrolight_geometry_eligible": items[0]["hydrolight_geometry_eligible"]}
        result["mean_residual_slope_abs"] = float(np.nanmean([abs(x["residual_VIS_B8_slope"]) for x in items]))
        result["mean_residual_R2"] = float(np.nanmean([x["residual_VIS_B8_R2"] for x in items]))
        result["mean_negative_percent"] = float(np.nanmean([x["negative_percent"] for x in items]))
        result["mean_signal_removed"] = float(np.nanmean([x["mean_signal_removed"] for x in items]))
        result["mean_shallow_preservation_correlation"] = float(np.nanmean([x["shallow_preservation_correlation"] for x in items]))
        result["mean_no_glint_control_abs_change"] = float(np.nanmean([x["no_glint_control_mean_abs_change"] for x in items]))
        result["mean_no_glint_control_negative_percent"] = float(np.nanmean([x["no_glint_control_negative_percent"] for x in items]))
        for case in HYDRO_CASES:
            diffs = np.asarray([x[f"HydroLight_{case}_difference"] for x in items], dtype=float)
            result[f"HydroLight_{case}_RMSE"] = float(np.sqrt(np.nanmean(diffs ** 2))) if np.isfinite(diffs).any() else np.nan
            result[f"HydroLight_{case}_MAE"] = float(np.nanmean(np.abs(diffs))) if np.isfinite(diffs).any() else np.nan
            result[f"HydroLight_{case}_bias"] = float(np.nanmean(diffs)) if np.isfinite(diffs).any() else np.nan
        output.append(result)
    return output


def plot_three(path: Path, title: str, arrays: dict[str, np.ndarray], cmap="viridis", symmetric=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.concatenate([a[np.isfinite(a)] for a in arrays.values() if np.isfinite(a).any()])
    if symmetric:
        lim = np.nanpercentile(np.abs(values), 99)
        vmin, vmax = -lim, lim
    else:
        vmin, vmax = np.nanpercentile(values, [1, 99]) if values.size else (0, 1)
    fig, axes = plt.subplots(1, len(arrays), figsize=(4.4 * len(arrays), 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, (name, arr) in zip(axes, arrays.items()):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax); ax.set_title(name); ax.axis("off")
    if im is not None: fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle(title)
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_glint_transects(output: Path, scene: str, source: dict, corrected: dict,
                         rhog: np.ndarray, valid_water: np.ndarray, ref: dict):
    """Save one reproducible across-glint transect and one large plot per VIS band."""
    usable = valid_water & np.isfinite(rhog)
    if not usable.any():
        return
    cutoff = float(np.nanpercentile(rhog[usable], 99))
    hot_rows = np.where(np.any(usable & (rhog >= cutoff), axis=1))[0]
    row = int(np.median(hot_rows)) if hot_rows.size else int(np.nanargmax(np.nanmax(np.where(usable, rhog, np.nan), axis=1)))
    valid_cols = np.where(usable[row])[0]
    if valid_cols.size < 2:
        return
    cols = np.arange(valid_cols.min(), valid_cols.max() + 1)
    line_valid = usable[row, cols]

    line_mask = np.zeros_like(valid_water, dtype=np.uint8)
    line_mask[row, valid_cols] = 1
    qc_dir = output / "07_Quality_Control" / scene
    write_tif(qc_dir / "glint_transect_line.tif", line_mask, ref, nodata=0)
    distance_km = (cols - cols.min()) * 20.0 / 1000.0
    records = []
    for band in ["B02", "B03", "B04"]:
        before = np.where(line_valid, source[band][row, cols] / np.pi, np.nan)
        after = np.where(line_valid, corrected[band][row, cols] / np.pi, np.nan)
        for distance, column, original_value, corrected_value, is_valid in zip(distance_km, cols, before, after, line_valid):
            if not is_valid:
                continue
            records.append({
                "scene": scene, "band": band, "row": row, "column": int(column),
                "distance_km": float(distance), "original_Rrs_approx_sr-1": float(original_value),
                "Glint_ACOLITE_Rrs_approx_sr-1": float(corrected_value),
            })
        fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
        ax.plot(distance_km, before, color="#65738a", linewidth=2.0, label="Original L2A")
        ax.plot(distance_km, after, color="#087f75", linewidth=2.2, label="Glint-ACOLITE")
        ax.axhline(0, color="#b13a48", linewidth=1.2, linestyle="--", label="Zero / negative threshold")
        ax.set_xlabel("Distance along transect (km)", fontsize=16)
        ax.set_ylabel(r"$R_{rs,approx}$ (sr$^{-1}$)", fontsize=16)
        ax.set_title(f"{scene} · {band} across-glint transect", fontsize=18, weight="bold")
        ax.tick_params(labelsize=14)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=14, ncol=3)
        fig.savefig(qc_dir / f"transect_{band}_before_after.png", dpi=180)
        plt.close(fig)
    write_csv(qc_dir / "glint_transect_values.csv", records)


def plot_rgb_compare(path: Path, title: str, methods: dict[str, dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(methods), figsize=(4.5 * len(methods), 4.2), constrained_layout=True)
    for ax, (name, bands) in zip(np.atleast_1d(axes), methods.items()):
        ax.imshow(stretch_rgb(bands)); ax.set_title(name); ax.axis("off")
    fig.suptitle(title); fig.savefig(path, dpi=150); plt.close(fig)


def plot_hedley(path: Path, bands: dict, training: np.ndarray, stats: list[dict], scene: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    for band, stat in zip(["B02", "B03", "B04"], stats):
        fig, ax = plt.subplots(figsize=(7.6, 6.2), constrained_layout=True)
        x, y = bands["B08"][training], bands[band][training]
        ax.scatter(x, y, s=2, alpha=0.25)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xx, stat["intercept"] + stat["slope"] * xx, color="black")
        ax.set_xlabel("B08 BOA reflectance (dimensionless)", fontsize=15)
        ax.set_ylabel(f"{band} BOA reflectance (dimensionless)", fontsize=15)
        ax.set_title(f"{scene} · Hedley {band}\nslope={stat['slope']:.3f}, R²={stat['R2']:.3f}, N={stat['N']}", fontsize=17, weight="bold")
        ax.tick_params(labelsize=13); ax.grid(alpha=0.2)
        fig.savefig(path.parent / f"{band}_VIS_NIR_regression.png", dpi=180)
        plt.close(fig)


def plot_terms(path: Path, terms: list[dict], scene: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    wave = [r["wavelength_nm"] for r in terms]
    plots = [
        ("two_way_direct_transmittance", "Two-way direct transmittance (dimensionless)", "Atmospheric transmittance", "transmittance_by_wavelength.png"),
        ("fresnel_reflectance", "Fresnel reflectance (dimensionless)", "Water-surface Fresnel term", "fresnel_by_wavelength.png"),
    ]
    for key, ylabel, subtitle, filename in plots:
        fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
        ax.plot(wave, [r[key] for r in terms], marker="o", linewidth=2.2, color="#087f75")
        ax.set_xlabel("Wavelength (nm)", fontsize=15); ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(f"{scene} · Glint-ACOLITE\n{subtitle}", fontsize=17, weight="bold")
        ax.tick_params(labelsize=13); ax.grid(alpha=0.22)
        fig.savefig(path.parent / filename, dpi=180); plt.close(fig)


def export_method(scene: str, stage: str, corrected: dict, removed: dict, ref: dict, output: Path):
    for band in ["B02", "B03", "B04"]:
        write_tif(output / stage / "Estimated_Glint" / scene / f"{band}_estimated_glint.tif", removed[band].astype(np.float32), ref, nodata=np.nan)
        write_tif(output / stage / "Corrected" / scene / f"{band}_corrected.tif", corrected[band].astype(np.float32), ref, nodata=np.nan)
    plot_rgb_compare(output / stage / "Corrected" / scene / "RGB_before_after.png", f"{scene} {stage}", {"Original": {**corrected, **{}}, "Corrected": corrected})


def prepare_tree(output: Path):
    folders = [
        "00_Scene_Selection", "01_Masks", "02_Surface_Reflectance", "03_ROI",
        "04_Hedley/Regression", "04_Hedley/Estimated_Glint", "04_Hedley/Corrected",
        "05_NIR_SWIR/Reference", "05_NIR_SWIR/Estimated_Glint", "05_NIR_SWIR/Corrected",
        "06_ACOLITE/AOT", "06_ACOLITE/Geometry", "06_ACOLITE/Fresnel", "06_ACOLITE/Transmittance",
        "06_ACOLITE/SWIR_Selection", "06_ACOLITE/Rhog_Reference", "06_ACOLITE/Glint_Scaling",
        "06_ACOLITE/Estimated_Glint", "06_ACOLITE/Corrected", "07_Quality_Control",
        "08_HydroLight", "09_Final_Comparison", "10_Rrs_Approx/Original",
        "10_Rrs_Approx/Glint_ACOLITE", "10_Rrs_Approx/Hedley", "10_Rrs_Approx/B12_SWIR",
    ]
    for folder in folders: (output / folder).mkdir(parents=True, exist_ok=True)


def process_scene(safe: Path, meta: dict, ref: dict, scl: np.ndarray, aot: np.ndarray, stats: dict, output: Path, hydro: dict) -> list[dict]:
    scene = scene_key(meta)
    cloud = np.isin(scl, SCL_CLOUD)
    # Keep cloud-edge and shadow adjacency out of all glint estimation and
    # correction. Three 20 m pixels give a documented 60 m project buffer.
    cloud_buffer = binary_dilation(cloud, iterations=3)
    water = scl == 6
    land_nonwater = ~water
    valid_water = water & ~np.isin(scl, SCL_INVALID) & ~cloud_buffer
    bands = load_bands(safe, meta, ref)

    mask_dir = output / "01_Masks" / scene
    write_tif(mask_dir / "water_mask.tif", water.astype(np.uint8), ref, nodata=0)
    write_tif(mask_dir / "land_nonwater_mask.tif", land_nonwater.astype(np.uint8), ref, nodata=0)
    write_tif(mask_dir / "cloud_cirrus_shadow_mask.tif", cloud.astype(np.uint8), ref, nodata=0)
    write_tif(mask_dir / "cloud_buffer_60m_mask.tif", cloud_buffer.astype(np.uint8), ref, nodata=0)
    write_tif(mask_dir / "valid_water_mask.tif", valid_water.astype(np.uint8), ref, nodata=0)
    write_tif(mask_dir / "SCL.tif", scl.astype(np.uint8), ref, nodata=0)
    mask_stats = {**stats, "cloud_buffer_distance_m": 60, "processing_valid_water_pixels": int(valid_water.sum()), "processing_valid_water_percent_of_crop": float(100 * valid_water.mean())}
    json_dump(mask_dir / "mask_statistics.json", mask_stats)
    plot_three(mask_dir / "mask_overview.png", f"{scene} masks", {"SCL water": water.astype(float), "Cloud + 60 m buffer": cloud_buffer.astype(float), "Valid water": valid_water.astype(float)}, cmap="gray")

    surface_dir = output / "02_Surface_Reflectance" / scene
    stats_rows = []
    for name, arr in bands.items():
        write_tif(surface_dir / f"{name}_BOA_20m.tif", arr.astype(np.float32), ref, nodata=np.nan)
        stats_rows.append(band_stats(scene, name, arr, valid_water))
    write_csv(surface_dir / "band_statistics.csv", stats_rows)
    json_dump(surface_dir / "scaling_record.json", {"formula": "rho_BOA=(stored_value+BOA_ADD_OFFSET)/BOA_QUANTIFICATION_VALUE", "quantification_value": meta["boa_quantification_value"], "band_id_offsets": meta["boa_offsets"], "grid_resolution_m": 20, "CRS": ref["projection"], "width": ref["width"], "height": ref["height"]})
    plot_rgb_compare(surface_dir / "surface_reflectance_RGB.png", f"{scene} 20 m BOA surface reflectance", {"Original L2A": bands})

    aot_values = aot[valid_water & np.isfinite(aot)]
    aot_used = float(np.nanmedian(aot_values)) if aot_values.size else meta["granule_mean_aot_550"]
    terms_m, factors_m = optical_terms(meta, aot_used, "M")
    terms_c, factors_c = optical_terms(meta, aot_used, "C")
    acolite = run_acolite(bands, valid_water, terms_m, factors_m)
    rois, roi_defs = build_rois(valid_water, bands, ref, acolite["rhog_ref"])

    roi_dir = output / "03_ROI" / scene
    for name, mask in rois.items(): write_tif(roi_dir / f"ROI_{name}.tif", mask.astype(np.uint8), ref, nodata=0)
    json_dump(roi_dir / "roi_definitions.json", roi_defs)
    plot_three(roi_dir / "ROI_map.png", f"{scene} fixed analysis ROIs", {f"ROI-{k}": v.astype(float) for k, v in rois.items()}, cmap="gray")

    hedley_corr, hedley_removed, hedley_stats, _, training = run_hedley(bands, rois["A"], valid_water)
    hreg = output / "04_Hedley" / "Regression" / scene
    write_csv(hreg / "regression_coefficients.csv", hedley_stats)
    write_tif(hreg / "regression_training_mask.tif", training.astype(np.uint8), ref, nodata=0)
    plot_hedley(hreg / "VIS_NIR_scatterplots.png", bands, training, hedley_stats, scene)
    for band in ["B02", "B03", "B04"]:
        write_tif(output / "04_Hedley" / "Estimated_Glint" / scene / f"{band}_estimated_glint.tif", hedley_removed[band].astype(np.float32), ref, nodata=np.nan)
        write_tif(output / "04_Hedley" / "Corrected" / scene / f"{band}_corrected.tif", hedley_corr[band].astype(np.float32), ref, nodata=np.nan)
    plot_rgb_compare(output / "04_Hedley" / "Corrected" / scene / "RGB_before_after.png", f"{scene} Hedley", {"Original": bands, "Hedley": hedley_corr})

    nir_removed = {b: np.where(valid_water, bands["B12"], np.nan) for b in ["B02", "B03", "B04"]}
    # Apply the identical valid-water domain used by Hedley and Glint-ACOLITE.
    # Land, cloud/shadow/cirrus/snow pixels and the 60 m contamination buffer
    # remain NoData rather than leaking into the B12 comparison product.
    nir_corr = {b: np.where(valid_water, bands[b] - bands["B12"], np.nan) for b in ["B02", "B03", "B04"]}
    nref = output / "05_NIR_SWIR" / "Reference" / scene
    json_dump(nref / "published_method.json", PUBLICATION)
    write_tif(nref / "B12_reference.tif", bands["B12"].astype(np.float32), ref, nodata=np.nan)
    for band in ["B02", "B03", "B04"]:
        write_tif(output / "05_NIR_SWIR" / "Estimated_Glint" / scene / f"{band}_estimated_glint.tif", nir_removed[band].astype(np.float32), ref, nodata=np.nan)
        write_tif(output / "05_NIR_SWIR" / "Corrected" / scene / f"{band}_corrected.tif", nir_corr[band].astype(np.float32), ref, nodata=np.nan)
    plot_rgb_compare(output / "05_NIR_SWIR" / "Corrected" / scene / "RGB_before_after.png", f"{scene} published B12 subtraction", {"Original": bands, "B12 subtraction": nir_corr})

    aot_dir = output / "06_ACOLITE" / "AOT" / scene
    write_tif(aot_dir / "AOT_550.tif", aot.astype(np.float32), ref, nodata=np.nan)
    json_dump(aot_dir / "AOT_statistics.json", {"reference_wavelength_nm": 550, "scaling": "stored AOT / 1000", "water_median_used": aot_used, "water_mean": float(np.nanmean(aot_values)), "water_sd": float(np.nanstd(aot_values)), "water_min": float(np.nanmin(aot_values)), "water_max": float(np.nanmax(aot_values))})
    geom = {k: meta[k] for k in ["sun_zenith_angle", "sun_azimuth_angle", "view_zenith_angle", "view_azimuth_angle", "relative_azimuth_angle", "glint_angle"]}
    json_dump(output / "06_ACOLITE" / "Geometry" / scene / "geometry.json", geom)
    write_csv(output / "06_ACOLITE" / "Transmittance" / scene / "atmospheric_optical_terms_maritime.csv", terms_m)
    write_csv(output / "06_ACOLITE" / "Transmittance" / scene / "atmospheric_optical_terms_continental_sensitivity.csv", terms_c)
    write_csv(output / "06_ACOLITE" / "Fresnel" / scene / "fresnel_terms.csv", [{k: r[k] for k in ["band", "wavelength_nm", "water_refractive_index", "fresnel_reflectance"]} for r in terms_m])
    plot_terms(output / "06_ACOLITE" / "Transmittance" / scene / "transmittance_fresnel_plot.png", terms_m, scene)
    write_tif(output / "06_ACOLITE" / "SWIR_Selection" / scene / "B11_candidate_blue.tif", acolite["candidate_B11_blue"].astype(np.float32), ref, nodata=np.nan)
    write_tif(output / "06_ACOLITE" / "SWIR_Selection" / scene / "B12_candidate_blue.tif", acolite["candidate_B12_blue"].astype(np.float32), ref, nodata=np.nan)
    write_tif(output / "06_ACOLITE" / "SWIR_Selection" / scene / "SWIR_selection_B11_is_1.tif", acolite["use_b11"].astype(np.uint8), ref, nodata=0)
    selection_n = int(acolite["mask"].sum())
    json_dump(output / "06_ACOLITE" / "SWIR_Selection" / scene / "selection_statistics.json", {"processing_pixels": selection_n, "B11_percent": float(100 * acolite["use_b11"].sum() / selection_n) if selection_n else np.nan, "B12_percent": float(100 * (acolite["mask"] & ~acolite["use_b11"]).sum() / selection_n) if selection_n else np.nan, "glint_mask_reference": "B11", "glint_mask_threshold": 0.05})
    write_tif(output / "06_ACOLITE" / "Rhog_Reference" / scene / "rhog_ref.tif", acolite["rhog_ref"].astype(np.float32), ref, nodata=np.nan)
    rhog_values = acolite["rhog_ref"][acolite["mask"]]
    json_dump(output / "06_ACOLITE" / "Rhog_Reference" / scene / "rhog_ref_statistics.json", {"n": int(rhog_values.size), "min": float(np.nanmin(rhog_values)), "mean": float(np.nanmean(rhog_values)), "median": float(np.nanmedian(rhog_values)), "max": float(np.nanmax(rhog_values))})
    write_csv(output / "06_ACOLITE" / "Glint_Scaling" / scene / "glint_scaling_factors.csv", acolite["scaling"])
    json_dump(output / "06_ACOLITE" / "Glint_Scaling" / scene / "model_assumptions.json", {"primary_aerosol_model": "ACOLITE maritime LUT MOD2", "sensitivity_model": "ACOLITE continental LUT MOD1", "pressure_hPa": 1013, "AOT_550_used": aot_used, "source_quantity": "Sen2Cor L2A BOA surface reflectance", "adaptation_warning": "Separated ACOLITE spectral transfer is adapted to L2A; this is not the standard ACOLITE L1 atmospheric-correction chain."})
    for band in ["B02", "B03", "B04"]:
        write_tif(output / "06_ACOLITE" / "Estimated_Glint" / scene / f"rhog_{band}.tif", acolite["removed"][band].astype(np.float32), ref, nodata=np.nan)
        write_tif(output / "06_ACOLITE" / "Corrected" / scene / f"{band}_corrected.tif", acolite["corrected"][band].astype(np.float32), ref, nodata=np.nan)
    plot_rgb_compare(output / "06_ACOLITE" / "Corrected" / scene / "RGB_before_after.png", f"{scene} Glint-ACOLITE", {"Original": bands, "Glint-ACOLITE": acolite["corrected"]})

    zero_removed = {b: np.zeros_like(bands[b]) for b in ["B02", "B03", "B04"]}
    method_data = {
        "Original": (bands, zero_removed), "Hedley": (hedley_corr, hedley_removed),
        "NIR_SWIR_B12": (nir_corr, nir_removed), "Adapted_ACOLITE": (acolite["corrected"], acolite["removed"]),
    }
    rrs_folders = {
        "Original": "Original", "Hedley": "Hedley",
        "NIR_SWIR_B12": "B12_SWIR", "Adapted_ACOLITE": "Glint_ACOLITE",
    }
    product_rows = []
    for method, (corr, _) in method_data.items():
        for band in ["B02", "B03", "B04"]:
            rrs = np.where(valid_water & np.isfinite(corr[band]), corr[band] / np.pi, np.nan).astype(np.float32)
            product_path = output / "10_Rrs_Approx" / rrs_folders[method] / scene / f"{band}_Rrs_approx_sr-1.tif"
            write_tif(product_path, rrs, ref, nodata=np.nan)
            product_rows.append({
                "scene": scene, "method": method, "band": band,
                "quantity": "Rrs_approx = Sen2Cor L2A BOA surface reflectance / pi",
                "unit": "sr-1", "valid_domain": "SCL water excluding cloud/shadow/snow and 60 m cloud buffer",
                "path": str(product_path),
            })
    write_csv(output / "10_Rrs_Approx" / scene / "product_manifest.csv", product_rows)
    plot_glint_transects(output, scene, bands, acolite["corrected"], acolite["rhog_ref"], valid_water, ref)
    qc = []
    for method, (corr, removed) in method_data.items():
        qc.extend(qc_metrics(scene, method, bands, corr, removed, rois, hydro, stats["hydrolight_geometry_eligible"]))
    write_csv(output / "07_Quality_Control" / scene / "method_band_metrics.csv", qc)
    plot_rgb_compare(output / "07_Quality_Control" / scene / "method_RGB_comparison.png", f"{scene} all methods", {"Original": bands, "Hedley": hedley_corr, "B12 subtraction": nir_corr, "Glint-ACOLITE": acolite["corrected"]})
    return qc


def final_plots(output: Path, summaries: list[dict]):
    if not summaries: return
    dates = sorted({r["scene"] for r in summaries})
    plots = [
        ("mean_residual_slope_abs", "Mean |residual VIS–B08 slope| (dimensionless)", "Residual glint diagnostic", "multi_scene_residual_slope.png"),
        ("mean_negative_percent", "Negative corrected water pixels (%)", "Over-correction diagnostic", "multi_scene_negative_pixels.png"),
        ("mean_shallow_preservation_correlation", "Shallow-water Pearson r (−1 to 1)", "Spatial-pattern preservation", "multi_scene_shallow_preservation.png"),
    ]
    for key, ylabel, title, filename in plots:
        fig, ax = plt.subplots(figsize=(13, 6.7), constrained_layout=True)
        for method in METHODS:
            rows = {r["scene"]: r for r in summaries if r["method"] == method}
            ax.plot(dates, [rows[d][key] for d in dates], marker="o", linewidth=2,
                    label=method.replace("Adapted_ACOLITE", "Glint-ACOLITE"))
        ax.set_xlabel("Accepted Sentinel-2 scene", fontsize=15); ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(f"Tidung multi-scene QC · {title}", fontsize=18, weight="bold")
        ax.tick_params(labelsize=12); ax.grid(alpha=0.25); ax.legend(ncol=4, fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.savefig(output / "09_Final_Comparison" / filename, dpi=180)
        plt.close(fig)


def hydro_plot(output: Path, hydro_rows: list[dict]):
    rows = [r for r in hydro_rows if r["platform"] == "S2A" and float(r["simulation_SZA_deg"]) == 30.0 and float(r["wind_m_s"]) == HYDRO_CORE_WIND]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = np.arange(3)
    for case in HYDRO_CASES:
        subset = {r["band"]: r["Rrs_sr-1"] for r in rows if r["scenario"] == case}
        ax.plot(x, [subset[b] for b in ["B02", "B03", "B04"]], marker="o", label=case)
    ax.set_xticks(x, ["B2 blue", "B3 green", "B4 red"])
    ax.set_xlabel("Sentinel-2A MSI band")
    ax.set_ylabel("HydroLight-convolved Rrs (sr⁻¹)")
    ax.set_title("HydroLight deep-water scenarios · SZA 30° · wind 5 m s⁻¹")
    ax.grid(alpha=0.25); ax.legend(title="Scenario")
    fig.savefig(output / "08_HydroLight" / "HydroLight_Sentinel2A_scenarios.png", dpi=160)
    plt.close(fig)


def aggregate_results(output: Path, summaries: list[dict]) -> tuple[list[dict], list[dict]]:
    scene_flags = []
    for scene in sorted({r["scene"] for r in summaries}):
        corrected = [r for r in summaries if r["scene"] == scene and r["method"] != "Original"]
        maximum_negative = max(float(r["mean_negative_percent"]) for r in corrected)
        status = "FAIL" if maximum_negative > 10 else "PASS"
        scene_flags.append({
            "scene": scene, "physical_QC_status": status,
            "maximum_corrected_negative_percent": maximum_negative,
            "criterion": "FAIL if any corrected-method mean B2/B3/B4 negative rate exceeds 10%",
            "final_aggregate_included": status == "PASS",
        })
    passed = {r["scene"] for r in scene_flags if r["physical_QC_status"] == "PASS"}
    aggregate = []
    for method in METHODS:
        rows_all = [r for r in summaries if r["method"] == method]
        rows = [r for r in rows_all if r["scene"] in passed]
        hydro_values = [float(r["HydroLight_CLR_RMSE"]) for r in rows_all if np.isfinite(float(r["HydroLight_CLR_RMSE"]))]
        aggregate.append({
            "method": method,
            "accepted_scenes": len(rows_all),
            "physical_QC_pass_scenes_used": len(rows),
            "mean_abs_residual_VIS_B8_slope": float(np.nanmean([r["mean_residual_slope_abs"] for r in rows])),
            "median_abs_residual_VIS_B8_slope": float(np.nanmedian([r["mean_residual_slope_abs"] for r in rows])),
            "mean_negative_percent": float(np.nanmean([r["mean_negative_percent"] for r in rows])),
            "mean_shallow_preservation_correlation": float(np.nanmean([r["mean_shallow_preservation_correlation"] for r in rows])),
            "mean_signal_removed": float(np.nanmean([r["mean_signal_removed"] for r in rows])),
            "mean_no_glint_control_abs_change": float(np.nanmean([r["mean_no_glint_control_abs_change"] for r in rows])),
            "mean_no_glint_control_negative_percent": float(np.nanmean([r["mean_no_glint_control_negative_percent"] for r in rows])),
            "mean_HydroLight_CLR_RMSE_geometry_matched": float(np.nanmean(hydro_values)) if hydro_values else np.nan,
        })
    write_csv(output / "09_Final_Comparison" / "scene_physical_QC_flags.csv", scene_flags)
    write_csv(output / "09_Final_Comparison" / "aggregate_method_comparison.csv", aggregate)
    return scene_flags, aggregate


def select_primary_scenes(output: Path, inventory: list[dict], summaries: list[dict], count: int = 5) -> list[dict]:
    """Rank clear, glint-supported, numerically stable Glint-ACOLITE showcase scenes."""
    accepted = {f"{r['date']}_{r['platform']}": r for r in inventory if r.get("decision") == "ACCEPT"}
    acolite = {r["scene"]: r for r in summaries if r["method"] == "Adapted_ACOLITE"}
    candidates = []
    for scene, row in acolite.items():
        inv = accepted.get(scene)
        if not inv:
            continue
        angle = float(inv["glint_angle"])
        cloud = float(inv["local_cloud_percent"])
        aot = float(inv["granule_mean_aot_550"])
        negative = float(row["mean_negative_percent"])
        no_glint_negative = float(row["mean_no_glint_control_negative_percent"])
        shallow = float(row["mean_shallow_preservation_correlation"])
        negative = negative if np.isfinite(negative) else 100.0
        no_glint_negative = no_glint_negative if np.isfinite(no_glint_negative) else 100.0
        shallow = shallow if np.isfinite(shallow) else 0.0
        try:
            roi_a, _ = read_warp(output / "03_ROI" / scene / "ROI_A.tif")
            valid_water, _ = read_warp(output / "01_Masks" / scene / "valid_water_mask.tif")
            b12, _ = read_warp(output / "02_Surface_Reflectance" / scene / "B12_BOA_20m.tif")
            mask_a = roi_a > 0
            vis_b12 = []
            for band in ["B02", "B03", "B04"]:
                vis, _ = read_warp(output / "02_Surface_Reflectance" / scene / f"{band}_BOA_20m.tif")
                vis_b12.append(correlation(vis[mask_a], b12[mask_a]))
            spectral_correlation = float(np.nanmean(vis_b12))
            rhog, _ = read_warp(output / "06_ACOLITE" / "Rhog_Reference" / scene / "rhog_ref.tif")
            rhog_values = rhog[(valid_water > 0) & np.isfinite(rhog)]
            rhog_p90 = float(np.nanpercentile(rhog_values, 90))
            rhog_spread = float(np.nanpercentile(rhog_values, 90) - np.nanpercentile(rhog_values, 10))
        except Exception:
            spectral_correlation, rhog_p90, rhog_spread = float("nan"), float("nan"), float("nan")

        geometry_support = angle <= 25.0
        atmosphere_support = cloud <= 0.1 and aot <= 0.45
        spectral_support = np.isfinite(spectral_correlation) and spectral_correlation >= 0.45
        spatial_support = np.isfinite(rhog_p90) and rhog_p90 > 0.008 and rhog_spread > 0.004
        support_count = sum([geometry_support, atmosphere_support, spectral_support, spatial_support])
        if spectral_support and support_count >= 3:
            glint_class = "GLINT_CONFIRMED"
        elif spectral_support and support_count >= 2:
            glint_class = "GLINT_PROBABLE"
        elif np.isfinite(spectral_correlation) and spectral_correlation < 0.20:
            glint_class = "NO_DETECTABLE_GLINT"
        else:
            glint_class = "MIXED_GLINT_ATMOSPHERE"

        geometry_score = float(np.clip((35.0 - angle) / 20.0, 0, 1))
        atmosphere_score = float(np.clip(1.0 - max(cloud / 2.0, aot / 0.60), 0, 1))
        spectral_score = float(np.clip((spectral_correlation - 0.20) / 0.60, 0, 1)) if np.isfinite(spectral_correlation) else 0.0
        spatial_score = 1.0 if spatial_support else 0.0
        stability_score = float(np.clip(1.0 - negative / 10.0, 0, 1))
        control_score = float(np.clip(1.0 - no_glint_negative / 5.0, 0, 1))
        shallow_score = float(np.clip(shallow, 0, 1))
        score = (0.20 * geometry_score + 0.10 * atmosphere_score + 0.20 * spectral_score +
                 0.10 * spatial_score + 0.15 * stability_score + 0.10 * control_score + 0.15 * shallow_score)
        eligible = (glint_class in {"GLINT_CONFIRMED", "GLINT_PROBABLE"} and angle <= 35.0 and
                    cloud <= 2.0 and aot <= 0.60 and negative <= 10.0 and no_glint_negative <= 5.0)
        candidates.append({
            "scene": scene, "eligible_for_showcase": eligible, "showcase_score": score,
            "glint_evidence_class": glint_class, "independent_support_count": support_count,
            "mean_VIS_B12_correlation_ROI_A": spectral_correlation,
            "rhog_P90": rhog_p90, "rhog_P90_minus_P10": rhog_spread,
            "glint_angle_deg": angle, "local_cloud_percent": cloud, "AOT_550": aot,
            "Glint_ACOLITE_mean_negative_percent": negative,
            "low_glint_control_negative_percent": no_glint_negative,
            "shallow_preservation_correlation": shallow,
            "basis": "Eligibility requires GLINT_CONFIRMED/PROBABLE from independent geometry, atmosphere, VIS-B12, and rhog evidence; score = 20% geometry + 10% atmosphere + 20% spectral + 10% spatial + 15% corrected stability + 10% held-out stability + 15% shallow preservation",
        })
    candidates.sort(key=lambda r: (not r["eligible_for_showcase"], -r["showcase_score"], r["local_cloud_percent"], r["glint_angle_deg"]))
    selected = [r for r in candidates if r["eligible_for_showcase"]][:count]
    write_csv(output / "09_Final_Comparison" / "primary_scene_ranking.csv", candidates)
    json_dump(output / "09_Final_Comparison" / "primary_scene_selection.json", {
        "primary_scenes": [r["scene"] for r in selected],
        "selection_count": len(selected),
        "selection_rule": "Only GLINT_CONFIRMED or GLINT_PROBABLE scenes with angle <=35 deg, cloud <=2%, AOT <=0.60, Glint-ACOLITE mean negatives <=10%, and held-out low-glint-control negatives <=5%; ranked by documented composite score.",
        "warning": "Showcase rank is for presentation order, not proof of Rrs accuracy.",
        "selected_details": selected,
    })
    return selected


def write_final_interpretation(output: Path, aggregate: list[dict], scene_flags: list[dict]):
    lookup = {r["method"]: r for r in aggregate}
    failed = [r["scene"] for r in scene_flags if r["physical_QC_status"] == "FAIL"]
    text = f"""# Final multi-scene interpretation

## Scope

The aggregate uses {aggregate[0]['physical_QC_pass_scenes_used']} of {aggregate[0]['accepted_scenes']} corrected scenes after physical QC.
The excluded QC-failure scene(s) are: {', '.join(failed) if failed else 'none'}.

## Main trade-offs

- Hedley gives the smallest cross-scene residual VIS–B8 slope ({lookup['Hedley']['mean_abs_residual_VIS_B8_slope']:.3f}) and very few negative pixels ({lookup['Hedley']['mean_negative_percent']:.3f}%), but has the weakest shallow-water preservation correlation ({lookup['Hedley']['mean_shallow_preservation_correlation']:.3f}).
- Published B12 subtraction preserves shallow spatial structure better than Hedley ({lookup['NIR_SWIR_B12']['mean_shallow_preservation_correlation']:.3f}) but produces more negative pixels ({lookup['NIR_SWIR_B12']['mean_negative_percent']:.3f}%).
- Glint-ACOLITE has corrected-method shallow-water preservation correlation {lookup['Adapted_ACOLITE']['mean_shallow_preservation_correlation']:.3f}, a mean negative rate of {lookup['Adapted_ACOLITE']['mean_negative_percent']:.3f}%, and mean low-glint-control absolute change {lookup['Adapted_ACOLITE']['mean_no_glint_control_abs_change']:.6f} in dimensionless BOA reflectance.

Across geometry-matched scenes, the mean CLR-scenario spectral RMSE is
{lookup['NIR_SWIR_B12']['mean_HydroLight_CLR_RMSE_geometry_matched']:.6f} sr⁻¹ for B12 subtraction,
{lookup['Adapted_ACOLITE']['mean_HydroLight_CLR_RMSE_geometry_matched']:.6f} sr⁻¹ for Glint-ACOLITE, and
{lookup['Hedley']['mean_HydroLight_CLR_RMSE_geometry_matched']:.6f} sr⁻¹ for Hedley.

No method is declared accurate from pattern correlation or HydroLight closeness alone.
The final usability decision requires acceptable negatives, low change in the held-out
low-glint control, residual-NIR/SWIR checks, preservation of shallow-water structure,
and plausible agreement with the modeled scenario envelope. HydroLight is not field truth.
"""
    (output / "09_Final_Comparison" / "FINAL_INTERPRETATION.md").write_text(text, encoding="utf-8")


def write_readme(output: Path, inventory: list[dict], summaries: list[dict]):
    accepted = [r for r in inventory if r["decision"] == "ACCEPT"]
    rejected = [r for r in inventory if r["decision"] == "REJECT"]
    hydro = sorted({r["scene"] for r in summaries if r["hydrolight_geometry_eligible"]})
    text = f"""# Tidung Glint Comparison batch result

- Source archive: `{ARCHIVE}`
- Total scenes screened: {len(inventory)}
- Scenes accepted for correction: {len(accepted)}
- Scenes rejected at cloud/water gate: {len(rejected)}
- Common analysis grid: 20 m
- HydroLight SZA grid: {', '.join(f'{x:.0f}' for x in HYDRO_SZA_GRID)} degrees; nearest-grid tolerance: {HYDRO_SZA_TOLERANCE:.2f} degrees
- HydroLight direction: theta=0 polar-cap quad average representing VZA 0-5 degrees
- Geometry-matched HydroLight scenes: {', '.join(hydro) if hydro else 'none'}

All {len(inventory)} scenes remain represented in `00_Scene_Selection`. Rejected scenes are not
forced through correction, consistent with the methodological gate. HydroLight
metrics are blank only if a scene falls outside the SZA grid or VZA polar cap. `Rrs = rho/pi`
is retained only as an explicitly approximate comparison because the source is
Sen2Cor BOA reflectance rather than a dedicated aquatic atmospheric correction.

No UAV-footprint vector was found in the supplied archive or project workspace,
so scene context figures show Pulau Tidung and the fixed AOI but not a UAV polygon.
Add the survey footprint and rerun the context-map stage if that vector becomes
available.

The `05_NIR_SWIR` method follows Niroumand-Jadidi et al. (2022), Eq. 2, by
subtracting Sentinel-2 B12 from each visible band. The `06_ACOLITE` branch uses
the separated ACOLITE transmittance/Fresnel transfer with pixel-wise B11/B12
selection, adapted to the same L2A BOA input.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true", help="Skip a scene if its QC CSV already exists")
    args = parser.parse_args()
    prepare_tree(args.output)
    json_dump(args.output / "05_NIR_SWIR" / "published_method.json", PUBLICATION)
    (args.output / "00_Scene_Selection" / "UAV_FOOTPRINT_MISSING.md").write_text(
        "# UAV footprint unavailable\n\nNo UAV survey-footprint vector was present in the supplied Sentinel-2 archive or project workspace. Pulau Tidung and the fixed AOI are shown, but a UAV polygon could not be overlaid.\n",
        encoding="utf-8",
    )

    hydro_runs = hydrolight_grid_runs()
    hydro_grid_by_platform = {p: convolve_hydrolight_grid(p, hydro_runs) for p in ["S2A", "S2B", "S2C"]}
    hydro_by_platform = {p: core_hydrolight_references(grid) for p, grid in hydro_grid_by_platform.items()}
    hydro_rows = []
    run_index = {(float(r["sza_deg"]), float(r["wind_m_s"]), r["case"]): r for r in hydro_runs}
    for platform, grid in hydro_grid_by_platform.items():
        for (sza, wind, case), values in grid.items():
            for band in ["B02", "B03", "B04"]:
                hydro_rows.append({"platform": platform, "scenario": case, "simulation_SZA_deg": sza, "wind_m_s": wind, "run_role": run_index[(sza, wind, case)]["run_role"], "band": band, "Rrs_sr-1": values[band]})
    write_csv(args.output / "08_HydroLight" / "HydroLight_Sentinel2_convolution.csv", hydro_rows)
    hydro_plot(args.output, hydro_rows)
    hydro_copy = args.output / "08_HydroLight" / "reference_grid"
    hydro_copy.mkdir(parents=True, exist_ok=True)
    for name in ["hydrolight_reference_manifest.json", "hydrolight_run_manifest.csv", "hydrolight_spectra_full.csv", "HYDROLIGHT_REFERENCE_DESIGN.md"]:
        source = HYDRO_GRID_ROOT / name
        if source.exists():
            shutil.copy2(source, hydro_copy / name)
    if (HYDRO_GRID_ROOT / "figures").exists():
        shutil.copytree(HYDRO_GRID_ROOT / "figures", hydro_copy / "figures", dirs_exist_ok=True)
    json_dump(args.output / "08_HydroLight" / "HydroLight_assumptions.json", {
        "simulation_SZA_grid_deg": HYDRO_SZA_GRID,
        "nearest_grid_max_difference_deg": HYDRO_SZA_TOLERANCE,
        "core_wind_m_s": HYDRO_CORE_WIND,
        "wind_sensitivity_m_s": [2.0, 8.0],
        "cases": list(HYDRO_CASES),
        "quantity": "above-water water-leaving Rrs=Lw/Ed in sr-1; Lw excludes reflected-sky/glint radiance",
        "direction": "theta=0 near-nadir polar-cap quad average representing VZA 0-5 degrees",
        "archive_view_geometry": "all current mean VZA values 3.436-3.606 degrees fall inside the polar cap; relative azimuth is not resolved at theta=0",
        "spectral_convolution": "full 400-900 nm HydroLight spectrum weighted by platform-specific Sentinel-2 RSR",
        "warning": "synthetic scenario/plausibility envelope, not field truth or an atmospheric-correction validation matchup",
        "reference_grid": str(HYDRO_GRID_ROOT),
    })

    records = []
    screened = []
    safes = sorted(args.archive.glob("*.SAFE"))
    for index, safe in enumerate(safes, 1):
        print(f"[{index}/{len(safes)}] Screening {safe.name}", flush=True)
        try:
            meta, ref, scl, aot, stats = screen_scene(safe, args.output / "00_Scene_Selection")
            screened.append((safe, meta, ref, scl, aot, stats))
            records.append(stats)
        except Exception as exc:
            records.append({"scene_id": safe.name, "platform": safe.name[:3], "date": safe.name.split("_")[2][:8], "decision": "ERROR", "decision_reason": repr(exc)})
            print(f"  ERROR screening: {exc!r}", flush=True)
    write_csv(args.output / "00_Scene_Selection" / "scene_inventory.csv", records)
    json_dump(args.output / "00_Scene_Selection" / "scene_inventory.json", records)

    all_qc = []
    for index, (safe, meta, ref, scl, aot, stats) in enumerate(screened, 1):
        if stats["decision"] != "ACCEPT":
            print(f"[{index}/{len(screened)}] Rejecting {scene_key(meta)} at gate", flush=True)
            continue
        scene = scene_key(meta)
        qc_path = args.output / "07_Quality_Control" / scene / "method_band_metrics.csv"
        if args.resume and qc_path.exists():
            print(f"[{index}/{len(screened)}] Reusing {scene}", flush=True)
            with qc_path.open(newline="", encoding="utf-8") as handle:
                reused = list(csv.DictReader(handle))
            # Published tables use Glint_ACOLITE; calculations use the stable
            # legacy key internally so resume mode remains backward-compatible.
            for row in reused:
                if row.get("method") == "Glint_ACOLITE":
                    row["method"] = "Adapted_ACOLITE"
            all_qc.extend(reused)
            continue
        print(f"[{index}/{len(screened)}] Processing accepted scene {scene}", flush=True)
        try:
            selected_sza = float(stats["hydrolight_reference_sza_deg"])
            all_qc.extend(process_scene(safe, meta, ref, scl, aot, stats, args.output, hydro_by_platform[meta["platform"]][selected_sza]))
        except Exception as exc:
            print(f"  ERROR processing {scene}: {exc!r}", flush=True)
            stats["decision"] = "PROCESSING_ERROR"
            stats["decision_reason"] = repr(exc)
            json_dump(args.output / "07_Quality_Control" / scene / "processing_error.json", stats)

    # Normalize values reloaded through CSV in resume mode.
    normalized = []
    for row in all_qc:
        item = dict(row)
        for key, value in list(item.items()):
            if key in ["scene", "method", "band"]: continue
            if isinstance(value, str):
                if value in ["True", "False"]: item[key] = value == "True"
                else:
                    try: item[key] = float(value)
                    except ValueError: pass
        normalized.append(item)
    all_qc = normalized
    write_csv(args.output / "07_Quality_Control" / "all_scene_method_band_metrics.csv", all_qc)
    summaries = rmse_summary(all_qc)
    write_csv(args.output / "09_Final_Comparison" / "final_method_comparison.csv", summaries)
    final_plots(args.output, summaries)
    scene_flags, aggregate = aggregate_results(args.output, summaries)
    select_primary_scenes(args.output, records, summaries)
    write_final_interpretation(args.output, aggregate, scene_flags)
    write_readme(args.output, records, summaries)
    json_dump(args.output / "workflow_manifest.json", {"archive": str(args.archive), "output": str(args.output), "scene_count": len(safes), "accepted_count": sum(r.get("decision") == "ACCEPT" for r in records), "bounds_lonlat": BOUNDS, "grid_resolution_m": 20, "methods": METHODS, "published_method": PUBLICATION, "hydrolight_SZA_grid_deg": HYDRO_SZA_GRID, "hydrolight_reference_grid": str(HYDRO_GRID_ROOT)})
    publish_method_names(args.output)
    print(json.dumps({"status": "complete", "output": str(args.output), "screened": len(records), "accepted": sum(r.get("decision") == "ACCEPT" for r in records), "method_summary_rows": len(summaries)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
