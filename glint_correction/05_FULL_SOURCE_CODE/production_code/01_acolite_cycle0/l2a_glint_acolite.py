#!/usr/bin/env python
"""Reproducible Sentinel-2 L2A glint-only workflow for Tidung Island.

Atmospheric correction is intentionally not run. Sentinel-2 BOA reflectance is
converted with ACOLITE, copied to ACOLITE-style ``rhos_*`` datasets, and then
passed to ACOLITE's residual glint routine. The Sen2Cor mean AOT and scene
geometry are supplied only for the wavelength transfer of the glint estimate.
"""

from __future__ import annotations

# WORKFLOW ROLE — STEP 01: GLINT-ACOLITE CYCLE 0
# Input: aligned L2A BOA, Sentinel geometry/AOT, water mask, B11 and B12.
# Output: corrected Cycle-0 bands plus the per-pixel SWIR choice and QC.
# ACOLITE is used for glint-transfer physics; Sen2Cor already supplied BOA.

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

import numpy as np


ROOT = Path(__file__).resolve().parents[4]  # ACOLITE-Glint Correction workspace
ACOLITE_SOURCE = ROOT / "acolite-source"
sys.path.insert(0, str(ACOLITE_SOURCE))

# Conda installs optional GDAL drivers here on Windows.
_plugin_dir = Path(sys.prefix) / "Library" / "lib" / "gdalplugins"
if _plugin_dir.exists():
    os.environ.setdefault("GDAL_DRIVER_PATH", str(_plugin_dir))

import acolite as ac  # noqa: E402
from netCDF4 import Dataset  # noqa: E402
from osgeo import gdal  # noqa: E402
from scipy.ndimage import distance_transform_edt, uniform_filter  # noqa: E402


BANDS = [443, 492, 560, 665, 704, 740, 783, 833, 865, 1614, 2202]
VISIBLE_BANDS = [443, 492, 560, 665, 704, 740, 783, 833, 865]
MODEL_NAMES = {"C": "continental", "M": "maritime"}
SCL_INVALID = [0, 1, 3, 8, 9, 10, 11]


def variable_bands(nc: Dataset, prefix: str) -> list[int]:
    bands = []
    for name in nc.variables:
        if name.startswith(prefix):
            try:
                bands.append(int(name.removeprefix(prefix)))
            except ValueError:
                pass
    return sorted(bands)


def closest_band(available: list[int], target: int) -> int:
    return min(available, key=lambda band: abs(band - target))


def local_name(tag: str) -> str:
    return tag.split("}")[-1]


def find_text(tree: ET.ElementTree, name: str, default=None):
    for element in tree.iter():
        if local_name(element.tag) == name and element.text:
            return element.text.strip()
    return default


def read_scene_metadata(safe: Path) -> dict:
    product_tree = ET.parse(safe / "MTD_MSIL2A.xml")
    tile_xml = next(safe.glob("GRANULE/*/MTD_TL.xml"))
    tile_tree = ET.parse(tile_xml)

    metadata = {
        "safe": str(safe),
        "scene_id": safe.name,
        "processing_level": find_text(product_tree, "PROCESSING_LEVEL"),
        "product_start_time": find_text(product_tree, "PRODUCT_START_TIME"),
        "cloudy_pixel_percentage": float(find_text(product_tree, "CLOUDY_PIXEL_PERCENTAGE", "nan")),
        "granule_mean_aot_550": float(find_text(product_tree, "GRANULE_MEAN_AOT", "0.1")),
        "granule_mean_water_vapour_cm": float(find_text(product_tree, "GRANULE_MEAN_WV", "nan")),
        "ozone_du": float(find_text(product_tree, "OZONE_VALUE", "nan")),
        "boa_quantification_value": float(find_text(product_tree, "BOA_QUANTIFICATION_VALUE", "10000")),
    }

    for parent in tile_tree.iter():
        if local_name(parent.tag) == "Mean_Sun_Angle":
            for child in parent:
                metadata[f"sun_{local_name(child.tag).lower()}"] = float(child.text)
        elif local_name(parent.tag) == "Mean_Viewing_Incidence_Angle_List":
            zenith, azimuth = [], []
            for angle in parent:
                for child in angle:
                    if local_name(child.tag) == "ZENITH_ANGLE":
                        zenith.append(float(child.text))
                    elif local_name(child.tag) == "AZIMUTH_ANGLE":
                        azimuth.append(float(child.text))
            metadata["view_zenith_angle"] = float(np.mean(zenith))
            metadata["view_azimuth_angle"] = float(np.mean(azimuth))

    metadata["relative_azimuth_angle"] = abs(
        metadata["sun_azimuth_angle"] - metadata["view_azimuth_angle"]
    )
    while metadata["relative_azimuth_angle"] > 180:
        metadata["relative_azimuth_angle"] = abs(360 - metadata["relative_azimuth_angle"])
    metadata["glint_angle"] = float(
        ac.shared.glint_angle(
            metadata["sun_zenith_angle"],
            metadata["view_zenith_angle"],
            metadata["relative_azimuth_angle"],
        )
    )
    return metadata


def find_band(safe: Path, token: str, resolution: int) -> Path:
    matches = list(safe.glob(f"GRANULE/*/IMG_DATA/R{resolution}m/*_{token}_{resolution}m.jp2"))
    if not matches:
        raise FileNotFoundError(f"Could not find {token}_{resolution}m in {safe}")
    return matches[0]


def converted_path(output_dir: Path) -> Path | None:
    matches = sorted(output_dir.glob("*_L2A.nc"))
    return matches[0] if matches else None


def convert_l2a(safe: Path, output_dir: Path, limit: list[float]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = converted_path(output_dir)
    if existing:
        return existing
    settings = {
        "output": str(output_dir),
        "limit": limit,
        "s2_target_res": 10,
        "output_geolocation": True,
        "netcdf_projection": True,
        "verbosity": 3,
    }
    files, _ = ac.sentinel2.l2_convert(str(safe), output=str(output_dir), settings=settings)
    if not files:
        raise RuntimeError("ACOLITE did not create an L2A NetCDF")
    return Path(files[0])


def warp_auxiliary(source: Path, nc: Dataset, resample: str = "near") -> np.ndarray:
    min_x, max_x = [float(v) for v in nc.getncattr("xrange")]
    max_y, min_y = [float(v) for v in nc.getncattr("yrange")]
    width = len(nc.dimensions["x"])
    height = len(nc.dimensions["y"])
    warped = gdal.Warp(
        "",
        str(source),
        format="MEM",
        outputBounds=(min_x, min_y, max_x, max_y),
        width=width,
        height=height,
        dstSRS=nc.getncattr("proj4_string"),
        resampleAlg=resample,
    )
    if warped is None:
        raise RuntimeError(f"GDAL could not warp {source}")
    array = warped.ReadAsArray()
    warped = None
    return array


def copy_variable(nc: Dataset, source_name: str, target_name: str, data=None, **attrs):
    if target_name in nc.variables:
        var = nc.variables[target_name]
    else:
        source = nc.variables[source_name]
        var = nc.createVariable(
            target_name,
            "f4",
            source.dimensions,
            fill_value=np.float32(np.nan),
            zlib=True,
            complevel=4,
        )
        for key in source.ncattrs():
            if key != "_FillValue":
                var.setncattr(key, source.getncattr(key))
    var[:] = nc.variables[source_name][:] if data is None else data
    for key, value in attrs.items():
        var.setncattr(key, value)


def add_qc_layers(nc_path: Path, safe: Path, metadata: dict):
    with Dataset(nc_path, "a") as nc:
        scl = warp_auxiliary(find_band(safe, "SCL", 20), nc, "near").astype(np.uint8)
        aot = warp_auxiliary(find_band(safe, "AOT", 20), nc, "bilinear").astype(np.float32) / 1000.0
        wvp = warp_auxiliary(find_band(safe, "WVP", 20), nc, "bilinear").astype(np.float32) / 1000.0

        for name, data, dtype, long_name, units in [
            ("qc_scl", scl, "u1", "Sen2Cor scene classification", "1"),
            ("qc_aot_550", aot, "f4", "Sen2Cor aerosol optical thickness at 550 nm", "1"),
            ("qc_water_vapour", wvp, "f4", "Sen2Cor water vapour", "cm"),
        ]:
            if name not in nc.variables:
                var = nc.createVariable(name, dtype, ("y", "x"), zlib=True, complevel=4)
                var.long_name = long_name
                var.units = units
            nc.variables[name][:] = data

        clear = ~np.isin(scl, SCL_INVALID)
        water = scl == 6
        for name, data, long_name in [
            ("qc_clear_sky_mask", clear, "Clear-sky/non-shadow mask from Sen2Cor SCL"),
            ("qc_scl_water_mask", water, "Water mask from Sen2Cor SCL class 6"),
        ]:
            if name not in nc.variables:
                var = nc.createVariable(name, "u1", ("y", "x"), zlib=True, complevel=4)
                var.long_name = long_name
                var.flag_values = np.array([0, 1], dtype=np.uint8)
            nc.variables[name][:] = data.astype(np.uint8)

        nc.setncattr("workflow_source_level", "Sentinel-2 Level-2A BOA surface reflectance")
        nc.setncattr("workflow_atmospheric_correction_run", "False")
        nc.setncattr("workflow_glint_method", "ACOLITE residual glint correction adapted to Sen2Cor L2A")
        nc.setncattr("workflow_sen2cor_mean_aot_550", metadata["granule_mean_aot_550"])


def prepare_model_file(converted: Path, safe: Path, output_dir: Path, metadata: dict, model: str) -> Path:
    target = output_dir / f"02_L2A_GLINT_{MODEL_NAMES[model].upper()}.nc"
    shutil.copy2(converted, target)
    with Dataset(target, "a") as nc:
        nc.setncattr("sza", metadata["sun_zenith_angle"])
        nc.setncattr("vza", metadata["view_zenith_angle"])
        nc.setncattr("raa", metadata["relative_azimuth_angle"])
        nc.setncattr("saa", metadata["sun_azimuth_angle"])
        nc.setncattr("vaa", metadata["view_azimuth_angle"])
        nc.setncattr("ac_aot_550", metadata["granule_mean_aot_550"])
        nc.setncattr("ac_model", model)
        nc.setncattr("pressure", 1013.0)
        nc.setncattr("acolite_file_type", "L2R_GC_FROM_L2A")
        nc.setncattr("workflow_aerosol_model_for_glint_transfer", MODEL_NAMES[model])
        source_bands = variable_bands(nc, "rhos_l2a_")
        for band in source_bands:
            src = f"rhos_l2a_{band}"
            if src in nc.variables:
                copy_variable(
                    nc,
                    src,
                    f"rhos_{band}",
                    long_name="Working copy of Sen2Cor L2A BOA surface reflectance for ACOLITE glint correction",
                )
    add_qc_layers(target, safe, metadata)
    return target


def run_glint(nc_path: Path, model: str):
    settings = {
        "dsf_interface_reflectance": False,
        "glint_mask_rhos_wave": 1600,
        "glint_mask_rhos_threshold": 0.05,
        "glint_force_band": None,
        "glint_write_rhog_ref": True,
        "glint_write_rhog_all": True,
        "pressure_default": 1013.0,
        "wind_default": 2.0,
        "uoz_default": 0.3,
        "uwv_default": 1.5,
        "gas_transmittance": True,
        "rsr_version": None,
        "add_band_name": False,
        "add_detector_name": False,
        "verbosity": 3,
    }
    # Glint transfer only needs ttot at the scene pressure. Loading ACOLITE's
    # four default pressure levels would add hundreds of MB without changing
    # this sea-level Tidung calculation.
    model_lut = {"C": "ACOLITE-LUT-202110-MOD1", "M": "ACOLITE-LUT-202110-MOD2"}[model]
    with Dataset(nc_path) as nc:
        sensor = str(getattr(nc, "sensor", "S2A_MSI"))
    lutdw = ac.aerlut.import_luts(
        pressures=[1013],
        base_luts=[model_lut],
        sensor=sensor,
        lut_par=["ttot"],
        return_lut_array=True,
        add_rsky=False,
    )
    result = ac.glint.default(str(nc_path), settings=settings, lutdw=lutdw, write=True, new_file=False)
    if result is None and not nc_path.exists():
        raise RuntimeError("ACOLITE glint correction failed")


def add_rrs_products(nc_path: Path):
    with Dataset(nc_path, "a") as nc:
        clear = nc.variables["qc_clear_sky_mask"][:].astype(bool)
        water = nc.variables["qc_scl_water_mask"][:].astype(bool)
        final_mask = clear & water
        if "qc_rrs_water_mask" not in nc.variables:
            mask_var = nc.createVariable("qc_rrs_water_mask", "u1", ("y", "x"), zlib=True, complevel=4)
            mask_var.long_name = "Conservative clear-sky water mask for Rrs QC"
        nc.variables["qc_rrs_water_mask"][:] = final_mask.astype(np.uint8)

        source_bands = variable_bands(nc, "rhos_l2a_")
        for band in source_bands:
            original_name = f"rhos_l2a_{band}"
            corrected_name = f"rhos_{band}"
            if original_name not in nc.variables or corrected_name not in nc.variables:
                continue
            original = np.asarray(nc.variables[original_name][:], dtype=np.float32)
            corrected = np.asarray(nc.variables[corrected_name][:], dtype=np.float32)
            nc.variables[original_name].long_name = "Input Sen2Cor L2A bottom-of-atmosphere surface reflectance before glint correction"
            nc.variables[original_name].units = "1"
            nc.variables[corrected_name].long_name = "Output glint-corrected Sen2Cor L2A bottom-of-atmosphere surface reflectance"
            nc.variables[corrected_name].units = "1"
            rhog_name = f"rhog_{band}"
            if rhog_name in nc.variables:
                nc.variables[rhog_name].long_name = "ACOLITE-estimated sunglint reflectance removed from the L2A surface reflectance"
                nc.variables[rhog_name].units = "1"
            for name, data, long_name in [
                (f"Rrs_before_{band}", original / np.pi, "Approximate Rrs before glint correction"),
                (f"Rrs_gc_{band}", corrected / np.pi, "Approximate Rrs after ACOLITE glint correction"),
                (
                    f"Rrs_gc_masked_{band}",
                    np.where(final_mask, corrected / np.pi, np.nan),
                    "Approximate glint-corrected Rrs masked to clear-sky Sen2Cor water",
                ),
            ]:
                if name not in nc.variables:
                    var = nc.createVariable(
                        name, "f4", ("y", "x"), fill_value=np.float32(np.nan), zlib=True, complevel=4
                    )
                    var.units = "sr-1"
                    var.wavelength = float(band)
                    var.long_name = long_name
                nc.variables[name][:] = data.astype(np.float32)
        if "rhog_ref" in nc.variables:
            nc.variables["rhog_ref"].long_name = "ACOLITE reference sunglint reflectance used for spectral glint transfer"
            nc.variables["rhog_ref"].units = "1"
        nc.setncattr("atmospheric_correction_run_in_this_workflow", "no; Sentinel-2 Sen2Cor L2A BOA reflectance was used directly")
        nc.setncattr("input_quantity", "rhos_l2a: Sen2Cor L2A BOA surface reflectance, dimensionless")
        nc.setncattr("glint_quantity", "rhog: ACOLITE-estimated sunglint reflectance removed, dimensionless")
        nc.setncattr("corrected_quantity", "rhos: glint-corrected L2A BOA surface reflectance, dimensionless")
        nc.setncattr("reported_water_quantity", "Rrs_gc = rhos/pi: approximate remote-sensing reflectance, sr-1")
        nc.setncattr("Rrs_conversion", "Rrs = rhos / pi; approximate because source is Sen2Cor BOA reflectance")


def select_deep_water_roi(nc_path: Path, size: int = 21) -> dict:
    with Dataset(nc_path) as nc:
        water = nc.variables["qc_rrs_water_mask"][:].astype(bool)
        lon = np.asarray(nc.variables["lon"][:])
        lat = np.asarray(nc.variables["lat"][:])
        distance_m = distance_transform_edt(water) * 10.0
        available = variable_bands(nc, "Rrs_gc_")
        selected = [closest_band(available, target) for target in [443, 492, 560, 665, 833, 1614]]
        arrays = [np.asarray(nc.variables[f"Rrs_gc_{b}"][:]) for b in selected]

    candidate = water & (distance_m >= 500)
    candidate &= np.isfinite(arrays[0])
    candidate &= (arrays[0] > 0) & (arrays[1] > 0) & (arrays[2] > 0)
    candidate &= (arrays[4] < 0.01) & (arrays[5] < 0.005)
    half = size // 2
    edge = half + 2
    candidate[:edge, :] = False
    candidate[-edge:, :] = False
    candidate[:, :edge] = False
    candidate[:, -edge:] = False

    score = np.zeros_like(arrays[0], dtype=np.float64)
    for data in arrays[:4]:
        mean = uniform_filter(np.nan_to_num(data, nan=0.0), size=size, mode="nearest")
        mean2 = uniform_filter(np.nan_to_num(data * data, nan=0.0), size=size, mode="nearest")
        std = np.sqrt(np.maximum(mean2 - mean * mean, 0))
        score += std / np.maximum(np.abs(mean), 1e-5)
    score += np.maximum(arrays[4], 0) * 100
    score[~candidate] = np.inf
    if not np.isfinite(score).any():
        raise RuntimeError("No conservative deep-water ROI candidate was found")
    row, col = np.unravel_index(np.argmin(score), score.shape)
    rows = slice(row - half, row + half + 1)
    cols = slice(col - half, col + half + 1)
    mask = np.zeros_like(candidate, dtype=bool)
    mask[rows, cols] = True
    mask &= water
    return {
        "row": int(row),
        "col": int(col),
        "size_pixels": int(size),
        "pixel_size_m": 10,
        "center_lon": float(lon[row, col]),
        "center_lat": float(lat[row, col]),
        "min_lon": float(np.nanmin(lon[mask])),
        "max_lon": float(np.nanmax(lon[mask])),
        "min_lat": float(np.nanmin(lat[mask])),
        "max_lat": float(np.nanmax(lat[mask])),
        "valid_pixels": int(mask.sum()),
        "selection": "lowest local spectral CV; SCL water; >=500 m from non-water; low NIR/SWIR",
    }


def roi_mask(shape: tuple[int, int], roi: dict) -> np.ndarray:
    half = roi["size_pixels"] // 2
    mask = np.zeros(shape, dtype=bool)
    mask[
        roi["row"] - half : roi["row"] + half + 1,
        roi["col"] - half : roi["col"] + half + 1,
    ] = True
    return mask


def add_roi_layer(nc_path: Path, roi: dict):
    with Dataset(nc_path, "a") as nc:
        mask = roi_mask((len(nc.dimensions["y"]), len(nc.dimensions["x"])), roi)
        mask &= nc.variables["qc_rrs_water_mask"][:].astype(bool)
        if "qc_deep_water_roi" not in nc.variables:
            var = nc.createVariable("qc_deep_water_roi", "u1", ("y", "x"), zlib=True, complevel=4)
            var.long_name = "Automatically selected homogeneous deep-water validation ROI"
        nc.variables["qc_deep_water_roi"][:] = mask.astype(np.uint8)
        nc.setncattr("deep_water_roi_center_lon", roi["center_lon"])
        nc.setncattr("deep_water_roi_center_lat", roi["center_lat"])


def parse_hydrolight_rrs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"Rrs"'))
    wave, rrs = [], []
    for line in lines[start + 3 :]:
        values = re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?", line)
        if len(values) < 2:
            if wave:
                break
            continue
        wave.append(float(values[0]))
        rrs.append(float(values[1]))
    return np.asarray(wave), np.asarray(rrs)


def hydrolight_s2(path: Path) -> dict[int, float]:
    wave, rrs = parse_hydrolight_rrs(path)
    rsrd = ac.shared.rsr_dict(sensor="S2A_MSI")["S2A_MSI"]
    convolved = ac.shared.rsr_convolute_dict(wave / 1000.0, rrs, rsrd["rsr"])
    return {int(round(rsrd["wave_nm"][band])): float(convolved[band]) for band in convolved}


def write_roi_outputs(model_files: dict[str, Path], roi: dict, hydrolight: Path, output_dir: Path):
    spectra_rows = []
    sensitivity_rows = []
    hydro = hydrolight_s2(hydrolight)
    for model, path in model_files.items():
        with Dataset(path) as nc:
            mask = nc.variables["qc_deep_water_roi"][:].astype(bool)
            available = variable_bands(nc, "Rrs_gc_")
            for band in available:
                before = np.asarray(nc.variables[f"Rrs_before_{band}"][:])[mask]
                after = np.asarray(nc.variables[f"Rrs_gc_{band}"][:])[mask]
                rhog_name = f"rhog_{band}"
                rhog = np.asarray(nc.variables[rhog_name][:])[mask] / np.pi if rhog_name in nc.variables else np.full(mask.sum(), np.nan)
                row = {
                    "model": MODEL_NAMES[model],
                    "band_nm": band,
                    "n": int(np.isfinite(after).sum()),
                    "Rrs_before_mean": float(np.nanmean(before)),
                    "Rrs_before_std": float(np.nanstd(before)),
                    "Rrs_after_mean": float(np.nanmean(after)),
                    "Rrs_after_std": float(np.nanstd(after)),
                    "Rrs_after_median": float(np.nanmedian(after)),
                    "Rrs_glint_removed_mean": float(np.nanmean(rhog)),
                    "negative_fraction_after": float(np.mean(after < 0)),
                    "HydroLight_Rrs": hydro.get(closest_band(list(hydro), band), np.nan),
                }
                spectra_rows.append(row)

    all_bands = sorted({r["band_nm"] for r in spectra_rows})
    for band in all_bands:
        rows = [r for r in spectra_rows if r["band_nm"] == band]
        values = [r["Rrs_after_mean"] for r in rows]
        sensitivity_rows.append(
            {
                "band_nm": band,
                "min_model_Rrs": min(values),
                "max_model_Rrs": max(values),
                "range_Rrs": max(values) - min(values),
                "relative_range_percent": 100 * (max(values) - min(values)) / max(abs(np.mean(values)), 1e-8),
            }
        )

    def write_csv(path: Path, rows: list[dict]):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "deep_water_roi_spectra.csv", spectra_rows)
    write_csv(output_dir / "aerosol_model_sensitivity.csv", sensitivity_rows)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {k: v for k, v in roi.items() if k not in {"row", "col"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [roi["min_lon"], roi["min_lat"]],
                        [roi["max_lon"], roi["min_lat"]],
                        [roi["max_lon"], roi["max_lat"]],
                        [roi["min_lon"], roi["max_lat"]],
                        [roi["min_lon"], roi["min_lat"]],
                    ]],
                },
            }
        ],
    }
    (output_dir / "deep_water_roi.geojson").write_text(json.dumps(geojson, indent=2), encoding="utf-8")


def make_plots(model_files: dict[str, Path], roi: dict, hydrolight: Path, output_dir: Path):
    preferred = model_files["M"]
    with Dataset(preferred) as nc:
        mask = nc.variables["qc_deep_water_roi"][:].astype(bool)
        visible_bands = [b for b in variable_bands(nc, "Rrs_gc_") if b <= 900]
        before = [float(np.nanmean(np.asarray(nc.variables[f"Rrs_before_{b}"][:])[mask])) for b in visible_bands]
        after_m = [float(np.nanmean(np.asarray(nc.variables[f"Rrs_gc_{b}"][:])[mask])) for b in visible_bands]
    hydro = hydrolight_s2(hydrolight)
    hydro_values = [hydro.get(closest_band(list(hydro), b), np.nan) for b in visible_bands]

    # Dependency-free SVG avoids binary backend problems in new Python builds.
    width, height = 900, 560
    left, right, top, bottom = 90, 35, 65, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = min(visible_bands), max(visible_bands)
    y_max = max(max(before), max(after_m), max(hydro_values)) * 1.12

    def xy(wave, value):
        x = left + (wave - x_min) / (x_max - x_min) * plot_w
        y = top + plot_h - value / y_max * plot_h
        return x, y

    def polyline(values, colour, dash=""):
        points = " ".join(f"{xy(w, v)[0]:.1f},{xy(w, v)[1]:.1f}" for w, v in zip(visible_bands, values))
        circles = "".join(
            f'<circle cx="{xy(w, v)[0]:.1f}" cy="{xy(w, v)[1]:.1f}" r="4" fill="{colour}"/>'
            for w, v in zip(visible_bands, values)
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2.5"{dash_attr}/>{circles}'

    grid = []
    for i in range(6):
        value = y_max * i / 5
        y = top + plot_h - i / 5 * plot_h
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#d8dee8"/>')
        grid.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="13">{value:.3f}</text>')
    for wave in visible_bands:
        x, _ = xy(wave, 0)
        grid.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12">{wave}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="32" font-family="Arial" font-size="22" font-weight="700">Deep-water ROI Rrs spectrum</text>
<g font-family="Arial" fill="#1f2937">{''.join(grid)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>
{polyline(before, '#64748b', '7 5')}
{polyline(after_m, '#0f766e')}
{polyline(hydro_values, '#b45309')}
<text x="{left + plot_w / 2}" y="{height - 38}" text-anchor="middle" font-size="15">Wavelength (nm)</text>
<text x="25" y="{top + plot_h / 2}" transform="rotate(-90 25 {top + plot_h / 2})" text-anchor="middle" font-size="15">Rrs (sr⁻¹)</text>
<g font-size="13">
<line x1="{left + 385}" y1="48" x2="{left + 415}" y2="48" stroke="#64748b" stroke-width="2.5" stroke-dasharray="7 5"/><text x="{left + 423}" y="53">Sen2Cor L2A before glint</text>
<line x1="{left + 385}" y1="68" x2="{left + 415}" y2="68" stroke="#0f766e" stroke-width="2.5"/><text x="{left + 423}" y="73">ACOLITE glint-corrected (maritime)</text>
<line x1="{left + 385}" y1="88" x2="{left + 415}" y2="88" stroke="#b45309" stroke-width="2.5"/><text x="{left + 423}" y="93">HydroLight reference scenario</text>
</g>
<text x="{left}" y="{height - 12}" font-size="11" fill="#64748b">Source: Sentinel-2A L2A, 19 Jun 2026; 210 m deep-water ROI. HydroLight is a scenario comparison, not field validation.</text>
</g></svg>'''
    (output_dir / "qc_deep_water_rrs_spectrum.svg").write_text(svg, encoding="utf-8")


def write_manifest(output_dir: Path, converted: Path, model_files: dict[str, Path], metadata: dict):
    products = [
        {
            "path": str(converted),
            "role": "01 untouched ACOLITE conversion of Sentinel-2 L2A BOA reflectance",
        }
    ]
    products.extend(
        {"path": str(path), "role": f"02 ACOLITE glint sensitivity run: {MODEL_NAMES[model]} aerosol"}
        for model, path in model_files.items()
    )
    for path, role in [
        (output_dir / "deep_water_roi.geojson", "Deep clear-water ROI definition"),
        (output_dir / "deep_water_roi_spectra.csv", "Before/after/glint/HydroLight ROI statistics"),
        (output_dir / "aerosol_model_sensitivity.csv", "Glint-model sensitivity statistics"),
        (output_dir / "qc_deep_water_rrs_spectrum.svg", "ROI spectral QC plot"),
        (output_dir / "geotiff_byproducts", "Per-band before/glint/after/Rrs GeoTIFF exports and RGB previews"),
    ]:
        products.append({"path": str(path), "role": role})
    manifest = {
        "workflow": "Sentinel-2 L2A to approximate Rrs with ACOLITE glint correction only",
        "atmospheric_correction_run": False,
        "source_reflectance": "Sen2Cor BOA surface reflectance",
        "rrs_formula": "Rrs = rhos / pi",
        "warning": "Deep-water ROI is an internal QC check, not independent field validation.",
        "metadata": metadata,
        "products": products,
    }
    (output_dir / "workflow_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe", required=True, type=Path)
    parser.add_argument("--hydrolight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--limit",
        nargs=4,
        type=float,
        default=[-5.86, 106.45, -5.75, 106.57],
        metavar=("S", "W", "N", "E"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metadata = read_scene_metadata(args.safe)
    (args.output / "scene_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    converted = convert_l2a(args.safe, args.output, args.limit)
    canonical = args.output / "01_L2A_ACOLITE_CONVERTED.nc"
    if converted != canonical:
        if not canonical.exists():
            shutil.copy2(converted, canonical)
        converted = canonical

    model_files = {}
    for model in ["C", "M"]:
        print(f"Preparing {MODEL_NAMES[model]} glint sensitivity run")
        target = prepare_model_file(converted, args.safe, args.output, metadata, model)
        run_glint(target, model)
        add_rrs_products(target)
        model_files[model] = target

    roi = select_deep_water_roi(model_files["M"])
    for path in model_files.values():
        add_roi_layer(path, roi)
    write_roi_outputs(model_files, roi, args.hydrolight, args.output)
    make_plots(model_files, roi, args.hydrolight, args.output)
    write_manifest(args.output, converted, model_files, metadata)
    print(json.dumps({"status": "complete", "output": str(args.output), "roi": roi}, indent=2))


if __name__ == "__main__":
    main()
