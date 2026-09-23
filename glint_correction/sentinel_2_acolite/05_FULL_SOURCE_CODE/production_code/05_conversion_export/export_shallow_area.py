"""Export every valid final water pixel inside the user-supplied Tidung KML.

This intentionally does not use the previous conservative candidate-shallow-water
test.  The KML is the processing boundary; only land, cloud/shadow and invalid
observations are excluded inside that boundary.
"""
from __future__ import annotations

# WORKFLOW ROLE — STEP 05C: SHALLOW-WATER EXPORT
# Use the supplied KML only as the outer processing boundary. Inside it, retain
# all valid water pixels and exclude land, cloud, shadow, snow, and invalid data.

import csv
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from rasterio.windows import Window, transform as window_transform

ROOT = Path(r"H:\Sentinel-2\Tidung\Tidung_Glint_Corrected_Products_20260913")
L2A = Path(r"H:\Sentinel-2\Tidung\L2A")
SOURCE = ROOT / "FINAL_PRODUCTS"
KML = Path(r"C:\Users\phili\Downloads\tidung-shallow.kml")
OUT = ROOT / "FINAL_Rrs_and_rrs_ALL_OPTICAL_BANDS_TIDUNG_KML_SHALLOW_WATER_20m"
BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A")
INVALID_SCL = (0, 1, 3, 8, 9, 10, 11)  # no-data, saturated, shadow, cloud/cirrus/snow
LAND_SCL = (4, 5)


def kml_polygon() -> dict:
    root = ET.parse(KML).getroot()
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    node = root.find(".//k:Polygon/k:outerBoundaryIs/k:LinearRing/k:coordinates", ns)
    if node is None or not node.text:
        raise ValueError("No polygon coordinates found in tidung-shallow.kml")
    coords = []
    for item in node.text.split():
        lon, lat, *_ = item.split(",")
        coords.append((float(lon), float(lat)))
    if len(coords) < 4:
        raise ValueError("KML polygon has fewer than four coordinates")
    return {"type": "Polygon", "coordinates": [coords]}


def scene_scl(scene: str, profile: dict) -> np.ndarray:
    platform = scene.split("_")[1]
    matches = list(L2A.glob(f"{platform}_MSIL2A_{scene[:8]}*.SAFE/GRANULE/*/IMG_DATA/R20m/*_SCL_20m.jp2"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one 20 m SCL file for {scene}; found {len(matches)}")
    with rasterio.open(matches[0]) as ds:
        arr = ds.read(1)
        if ds.crs != profile["crs"] or ds.transform != profile["transform"]:
            from rasterio.warp import reproject, Resampling
            aligned = np.zeros((profile["height"], profile["width"]), dtype=np.uint8)
            reproject(arr, aligned, src_transform=ds.transform, src_crs=ds.crs,
                      dst_transform=profile["transform"], dst_crs=profile["crs"],
                      resampling=Resampling.nearest)
            return aligned
        return arr


def crop_window(mask: np.ndarray) -> Window:
    rows, cols = np.where(mask)
    if not rows.size:
        raise ValueError("No valid KML shallow-water pixels in this scene")
    return Window(int(cols.min()), int(rows.min()), int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1))


def main() -> None:
    if not KML.exists():
        raise FileNotFoundError(KML)
    OUT.mkdir(parents=True, exist_ok=True)
    polygon_ll = kml_polygon()
    summary = []
    for scene_dir in sorted(p for p in SOURCE.iterdir() if p.is_dir()):
        scene = scene_dir.name
        sources = [scene_dir / "01_Final_BOA_Reflectance" / f"{scene}_{band}_final_BOA_reflectance.tif" for band in BANDS]
        if not all(p.exists() for p in sources):
            raise FileNotFoundError(f"Missing final optical Rrs input for {scene}")
        with rasterio.open(sources[0]) as ds:
            profile = ds.profile.copy()
            polygon = transform_geom("EPSG:4326", ds.crs, polygon_ll)
            inside_kml = geometry_mask([polygon], out_shape=(ds.height, ds.width), transform=ds.transform, invert=True)
        arrays = []
        for source in sources:
            with rasterio.open(source) as ds:
                if (ds.crs, ds.transform, ds.width, ds.height) != (profile["crs"], profile["transform"], profile["width"], profile["height"]):
                    raise RuntimeError(f"Grid mismatch: {source}")
                arrays.append(ds.read(1).astype(np.float32))
        # Existing exported Rrs TIFFs can be empty for some scenes.  Derive
        # traceable Rrs_approx directly from the retained final BOA product.
        stack = np.stack(arrays) / np.pi
        finite_all = np.all(np.isfinite(stack), axis=0)
        scl = scene_scl(scene, profile)
        excluded_scl = np.isin(scl, INVALID_SCL + LAND_SCL)
        selected = inside_kml & finite_all & ~excluded_scl
        window = crop_window(inside_kml)
        selected_crop = selected[window.toslices()]
        inside_crop = inside_kml[window.toslices()]
        invalid_crop = (~finite_all | excluded_scl)[window.toslices()]
        mask = np.zeros(selected_crop.shape, dtype=np.uint8)
        mask[~inside_crop | invalid_crop] = 255
        mask[selected_crop] = 1
        out_profile = profile.copy()
        out_profile.update(driver="GTiff", height=int(window.height), width=int(window.width),
                           transform=window_transform(window, profile["transform"]), dtype="uint8",
                           nodata=255, compress="deflate", predictor=2)
        destination = OUT / scene
        destination.mkdir(parents=True, exist_ok=True)
        mask_path = destination / f"{scene}_Tidung_KML_shallow_water_mask.tif"
        with rasterio.open(mask_path, "w", **out_profile) as ds:
            ds.write(mask, 1)
            ds.set_band_description(1, "1=valid KML shallow-water pixel; 255=land/cloud/shadow/invalid/outside KML")
            ds.update_tags(processing_boundary="tidung-shallow.kml", mask_definition="KML interior AND finite final B01-B8A AND not SCL land/cloud/shadow")
        data_profile = out_profile.copy()
        data_profile.update(dtype="float32", nodata=np.nan, predictor=3)
        for band, image in zip(BANDS, stack):
            values = image[window.toslices()]
            rrs_upper = np.full(values.shape, np.nan, dtype=np.float32)
            rrs_upper[selected_crop] = values[selected_crop]
            upper = destination / f"{scene}_{band}_above_Rrs_approx_sr-1.tif"
            with rasterio.open(upper, "w", **data_profile) as ds:
                ds.write(rrs_upper, 1)
                ds.set_band_description(1, "Above-surface final glint-corrected Rrs_approx (sr-1), KML shallow water")
            rrs_lower = np.full(values.shape, np.nan, dtype=np.float32)
            rrs_lower[selected_crop] = rrs_upper[selected_crop] / (0.52 + 1.7 * rrs_upper[selected_crop])
            lower = destination / f"{scene}_{band}_below_rrs_approx_sr-1.tif"
            with rasterio.open(lower, "w", **data_profile) as ds:
                ds.write(rrs_lower, 1)
                ds.set_band_description(1, "Below-surface rrs_approx (sr-1), KML shallow water")
        shutil.copy2(KML, destination / "tidung-shallow.kml")
        meta = {"scene": scene, "boundary": str(KML), "mask": str(mask_path),
                "grid": {"crs": str(profile["crs"]), "pixel_size_m": 20, "full_width": profile["width"], "full_height": profile["height"], "crop_width": int(window.width), "crop_height": int(window.height)},
                "selection_rule": "inside user-supplied KML; final BOA B01-B8A converted to Rrs_approx = BOA/pi; excludes SCL land, cloud, cirrus, cloud shadow, snow, saturated and invalid pixels; no conservative depth/bottom/distance screening",
                "counts": {"KML_pixels": int(inside_kml.sum()), "selected_valid_water_pixels": int(selected.sum()), "excluded_land_cloud_shadow_or_invalid_inside_KML": int((inside_kml & ~selected).sum())}}
        (destination / f"{scene}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        summary.append({"scene": scene, **meta["grid"], **meta["counts"]})
        print(f"{scene}: {selected.sum():,} valid KML shallow-water pixels", flush=True)
    shutil.copy2(KML, OUT / "tidung-shallow.kml")
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)
    with (OUT / "all_scene_KML_shallow_water_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    (OUT / "README.txt").write_text(
        "All final retained glint-corrected B01-B8A products, clipped to the user-supplied tidung-shallow.kml boundary.\n"
        "Every finite pixel inside the KML is retained except Sen2Cor SCL land, cloud, cirrus, cloud shadow, snow/saturated and invalid pixels.\n"
        "No previous conservative shallow-water, distance-to-land, deep-water, bottom-influence or local-homogeneity filter is applied.\n"
        "All rasters are aligned to the original 20 m final-product grid. B01 was resampled from native 60 m to this 20 m grid and does not gain true 20 m detail.\n"
        "Rrs_approx is above surface; rrs_approx = Rrs_approx/(0.52 + 1.7*Rrs_approx).\n", encoding="utf-8")
    print(f"Completed {len(summary)} scenes: {OUT}")


if __name__ == "__main__":
    main()
