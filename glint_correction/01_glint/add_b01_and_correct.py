#!/usr/bin/env python
"""Add traceable, glint-corrected Sentinel-2 B01 to every retained Tidung scene.

B01 is resampled from native 60 m to the exact existing 20 m product grid. The
resampling aligns pixels but does not create true 20 m spatial information.
Cycle 0 uses the existing per-pixel B11/B12 choice and rhog reference with a
scene-specific B01 transfer factor from the same ACOLITE LUT physics. For the
single retained Cycle-1 scene, the already accepted pixel multiplier is
reconstructed from B02 and applied to the B01 Cycle-0 glint component.
"""
from __future__ import annotations

import csv, json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

PROJECT = Path(__file__).resolve().parents[1]

ROOT = Path(r"H:\Sentinel-2\Tidung\Tidung_Glint_Corrected_Products_20260913")
L2A = Path(r"H:\Sentinel-2\Tidung\L2A")
JOINT = PROJECT / "outputs" / "Tidung_Joint_HydroLight_Glint_Multiband_Experiment_20260911"
FULL = PROJECT / "outputs" / "Tidung_Full_Glint_All_Scenes_Experiment_20260911"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def read_tif(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds: return ds.read(1).astype(float),ds.profile.copy()

def resample_b01(path: Path, profile: dict) -> np.ndarray:
    out=np.zeros((profile["height"],profile["width"]),np.float32)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src,1),out,src_transform=src.transform,src_crs=src.crs,dst_transform=profile["transform"],dst_crs=profile["crs"],src_nodata=0,dst_nodata=0,resampling=Resampling.bilinear)
    return out


def write_tif(path: Path, arr: np.ndarray, profile: dict, description: str, unit: str) -> None:
    p=profile.copy(); p.update(driver="GTiff",count=1,dtype="float32",nodata=np.nan,
                               compress="deflate",predictor=3,tiled=True,blockxsize=256,blockysize=256)
    path.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(path,"w",**p) as dst:
        dst.write(arr.astype("float32"),1); dst.set_band_description(1,description)
        dst.update_tags(AREA_OR_POINT="Area",unit=unit,
            source="Sentinel-2 Sen2Cor L2A B01; native 60 m resampled to aligned 20 m grid",
            spatial_resolution_note="20 m grid alignment only; native B01 information remains 60 m")


def qc(scene: str, boa: np.ndarray, rrs: np.ndarray, sub: np.ndarray) -> dict:
    valid=np.isfinite(boa); a,b,c=boa[valid],rrs[valid],sub[valid]
    reconstructed=0.52*c/(1-1.7*c)
    return {"scene":scene,"band":"B01","valid_pixels":int(valid.sum()),
        "BOA_p01":float(np.percentile(a,1)),"BOA_median":float(np.median(a)),"BOA_p99":float(np.percentile(a,99)),
        "above_surface_Rrs_p01_sr-1":float(np.percentile(b,1)),"above_surface_Rrs_median_sr-1":float(np.median(b)),"above_surface_Rrs_p99_sr-1":float(np.percentile(b,99)),
        "below_surface_rrs_p01_sr-1":float(np.percentile(c,1)),"below_surface_rrs_median_sr-1":float(np.median(c)),"below_surface_rrs_p99_sr-1":float(np.percentile(c,99)),
        "negative_percent":float(100*np.mean(b<0)),"nonfinite_conversion_pixels":int(np.sum(~np.isfinite(c))),
        "Rrs_rrs_roundtrip_max_abs_error":float(np.max(np.abs(reconstructed-b))),"numeric_conversion_pass":bool(np.max(np.abs(reconstructed-b))<1e-6)}


def main() -> None:
    statuses=csv_rows(FULL/"all_33_scene_final_status.csv")
    retained=[r for r in statuses if r["decision"].startswith("RETAIN_")]
    factors_all=json.loads((ROOT/"00_MASTER_QC"/"B01_ACOLITE_factors.json").read_text())
    master_qc=csv_rows(ROOT/"00_MASTER_QC"/"all_retained_band_value_QC.csv")
    master_qc=[r for r in master_qc if r["band"]!="B01"]
    manifest=csv_rows(ROOT/"00_MASTER_QC"/"product_manifest.csv")
    manifest=[r for r in manifest if r["band"]!="B01"]
    audit=[]
    for status in retained:
        scene=status["scene"]; info=factors_all[scene]
        b02_path=ROOT/"FINAL_PRODUCTS"/scene/"01_Final_BOA_Reflectance"/f"{scene}_B02_final_BOA_reflectance.tif"
        with rasterio.open(b02_path) as ds: profile=ds.profile.copy(); final_b02=ds.read(1).astype(float)
        raw=resample_b01(Path(info["native_B01"]),profile); original=(raw+float(info["boa_offset_B01"]))/float(info["boa_quantification_value"]); original[raw==0]=np.nan
        use_b11=read_tif(JOINT/".."/"Tidung_Glint_Comparison_Rerun_20260903"/"06_ACOLITE"/"SWIR_Selection"/scene/"SWIR_selection_B11_is_1.tif")[0]>.5
        rhog=read_tif(JOINT/".."/"Tidung_Glint_Comparison_Rerun_20260903"/"06_ACOLITE"/"Rhog_Reference"/scene/"rhog_ref.tif")[0]
        terms,f11,f12=info["optical_terms"],float(info["factor_from_B11"]),float(info["factor_from_B12"])
        glint0=np.where(use_b11,f11*rhog,f12*rhog)
        valid=np.isfinite(final_b02)&np.isfinite(original)&np.isfinite(glint0)
        cycle0=np.where(valid,original-glint0,np.nan)
        final=cycle0.copy(); residual_multiplier=np.zeros_like(final)
        if status["decision"]=="RETAIN_CYCLE1":
            cyc_b02=read_tif(JOINT/"03_Glint_ACOLITE_Cycle0"/scene/"Cycle0_B02.tif")[0]
            glint_b02=read_tif(JOINT/"03_Glint_ACOLITE_Cycle0"/scene/"estimated_glint_B02.tif")[0]
            good=valid&np.isfinite(cyc_b02)&np.isfinite(glint_b02)&(np.abs(glint_b02)>1e-12)
            residual_multiplier[good]=np.clip((cyc_b02[good]-final_b02[good])/glint_b02[good],0,1)
            final[good]=cycle0[good]-residual_multiplier[good]*glint0[good]
        final[~np.isfinite(final_b02)]=np.nan
        write_tif(JOINT/"02_Multiband_Inputs"/scene/"B01_BOA_20m.tif",np.where(valid,original,np.nan),profile,"B01 BOA resampled 60 m to aligned 20 m grid","dimensionless")
        write_tif(JOINT/"03_Glint_ACOLITE_Cycle0"/scene/"estimated_glint_B01.tif",np.where(valid,glint0,np.nan),profile,"Estimated Glint-ACOLITE component B01","dimensionless")
        write_tif(JOINT/"03_Glint_ACOLITE_Cycle0"/scene/"Cycle0_B01.tif",cycle0,profile,"Cycle-0 corrected B01 BOA reflectance; native information 60 m","dimensionless")
        products=ROOT/"FINAL_PRODUCTS"/scene
        boa_path=products/"01_Final_BOA_Reflectance"/f"{scene}_B01_final_BOA_reflectance.tif"
        rrs_path=products/"02_Rrs_Approx"/f"{scene}_B01_Rrs_approx_sr-1.tif"
        sub_path=products/"03_rrs_Approx"/f"{scene}_B01_rrs_approx_sr-1.tif"
        rrs=final/math.pi; sub=rrs/(.52+1.7*rrs)
        write_tif(boa_path,final,profile,"Final retained B01 BOA reflectance","dimensionless")
        write_tif(rrs_path,rrs,profile,"Approximate above-water Rrs B01","sr-1")
        write_tif(sub_path,sub,profile,"Approximate below-surface rrs B01","sr-1")
        row=qc(scene,final,rrs,sub); master_qc.append(row)
        scene_qc=csv_rows(products/"04_QC"/"band_value_and_conversion_QC.csv"); scene_qc=[r for r in scene_qc if r["band"]!="B01"]+[row]
        write_csv(products/"04_QC"/"band_value_and_conversion_QC.csv",scene_qc)
        for quantity,path in (("BOA_reflectance",boa_path),("above_surface_Rrs_approx",rrs_path),("below_surface_rrs_approx",sub_path)):
            manifest.append({"scene":scene,"band":"B01","quantity":quantity,"path":str(path)})
        provenance={"scene":scene,"band":"B01","native_resolution_m":60,"output_grid_resolution_m":20,
            "resolution_warning":"Resampling aligns B01 to the 20 m grid but does not create true 20 m spatial information.",
            "resampling":"bilinear","grid_reference":str(b02_path),"valid_mask":"finite mask of final retained B02",
            "cycle0_method":"Existing pixel-level B11/B12 selection and rhog_ref with scene-specific ACOLITE B01 transfer factors",
            "factor_from_B11":f11,"factor_from_B12":f12,"optical_terms":terms,
            "cycle1_method":"Existing accepted B02 residual multiplier applied to B01 glint0" if status["decision"]=="RETAIN_CYCLE1" else "Not applicable; final is Cycle 0",
            "Rrs_formula":"Rrs_approx = final BOA reflectance / pi","solar_zenith_deg":info["solar_zenith_deg"],
            "solar_zenith_source":"Mean_Sun_Angle/ZENITH_ANGLE in GRANULE MTD_TL.xml (scene/tile mean)","created_utc":datetime.now(timezone.utc).isoformat()}
        p=products/"05_Provenance"/"B01_processing_and_grid_provenance.json"; p.write_text(json.dumps(provenance,indent=2),encoding="utf-8")
        factors_path=JOINT/"03_Glint_ACOLITE_Cycle0"/scene/"multiband_glint_factors.csv"; factors=csv_rows(factors_path); factors=[r for r in factors if r["band"]!="B01"]
        write_csv(factors_path,[{"band":"B01","factor_from_B11":f11,"factor_from_B12":f12}]+factors)
        audit.append({"scene":scene,"native_B01":info["native_B01"],"output_grid":"20 m aligned to final B02",
                      "retained_product":status["decision"],"factor_B11":f11,"factor_B12":f12,"valid_pixels":row["valid_pixels"],"negative_percent":row["negative_percent"]})
        print(f"{scene}: B01 complete; {row['valid_pixels']:,} valid pixels",flush=True)
    write_csv(ROOT/"00_MASTER_QC"/"all_retained_band_value_QC.csv",master_qc)
    write_csv(ROOT/"00_MASTER_QC"/"product_manifest.csv",manifest)
    write_csv(ROOT/"00_MASTER_QC"/"B01_all_scene_processing_audit.csv",audit)
    summary_path=ROOT/"00_MASTER_QC"/"package_summary.json"; summary=json.loads(summary_path.read_text())
    summary["bands"]=["B01","B02","B03","B04","B05","B06","B07","B08","B8A"]
    summary["B01_resolution_note"]="B01 is on the exact 20 m grid but retains native 60 m spatial information"
    summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"Completed B01 for {len(audit)} retained scenes")


if __name__=="__main__": main()


