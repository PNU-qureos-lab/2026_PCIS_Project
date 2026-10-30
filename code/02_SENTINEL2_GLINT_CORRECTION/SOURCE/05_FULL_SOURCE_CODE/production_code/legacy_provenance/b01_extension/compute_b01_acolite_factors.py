#!/usr/bin/env python
"""Compute scene-specific B01 ACOLITE transfer factors (run in ACOLITE env)."""
from __future__ import annotations

# WORKFLOW ROLE — STEP 04A: B01 TRANSFER FACTORS
# Run in the ACOLITE environment to compute scene-specific B11→B01 and
# B12→B01 glint-transfer factors from the same LUT physics as other bands.
import csv, json, sys
from pathlib import Path
import numpy as np

PROJECT=Path(__file__).resolve().parents[5]
FULL_CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(FULL_CODE/"01_acolite_cycle0"))
import acolite_helpers as archive

L2A=Path(r"H:\Sentinel-2\Tidung\L2A")
FULL=PROJECT/"outputs"/"Tidung_Full_Glint_All_Scenes_Experiment_20260911"
SOURCE=PROJECT/"outputs"/"Tidung_Glint_Comparison_Rerun_20260903"
OUT=Path(r"H:\Sentinel-2\Tidung\Tidung_Glint_Corrected_Products_20260913\00_MASTER_QC\B01_ACOLITE_factors.json")
archive.S2_BAND_ID["B01"]=0

def safe_for(scene):
    date,platform=scene.split("_"); found=sorted(L2A.glob(f"{platform}_MSIL2A_{date}*.SAFE"))
    if len(found)!=1: raise RuntimeError(f"Expected one SAFE for {scene}, found {len(found)}")
    return found[0]

def factors(meta,aot):
    sensor,rsrd=archive.sensor_info(meta["platform"])
    lutdw=archive.ac.aerlut.import_luts(pressures=[1013],base_luts=["ACOLITE-LUT-202110-MOD2"],sensor=sensor,lut_par=["ttot"],return_lut_array=True,add_rsky=False)
    lut=list(lutdw)[0]; raa,vza,sza=meta["relative_azimuth_angle"],meta["view_zenith_angle"],meta["sun_zenith_angle"]
    xi=[1013.,raa,vza,sza]; sr,vr,rr=np.radians([sza,vza,raa]); omega=np.arccos(np.cos(sr)*np.cos(vr)+np.sin(sr)*np.sin(vr)*np.cos(rr))/2
    refri=archive.ac.shared.wopp.refri(); n=archive.ac.shared.rsr_convolute_dict(refri["wave"]/1000,refri["n"],rsrd["rsr"]); mus,muv=np.cos(sr),np.cos(vr); terms={}
    for name,token in {"B01":"1","B11":"11","B12":"12"}.items():
        t=float(lutdw[lut]["rgi"][token]((xi[0],lutdw[lut]["ipd"]["ttot"],xi[1],xi[2],xi[3],aot)))
        terms[name]={"wavelength_nm":float(rsrd["wave_nm"][token]),"ttot":t,"two_way_transmittance":float(np.exp(-t/muv)*np.exp(-t/mus)),"fresnel_reflectance":float(archive.ac.ac.sky_refl(omega,n_w=n[token]))}
    def f(ref): return terms["B01"]["two_way_transmittance"]/terms[ref]["two_way_transmittance"]*terms["B01"]["fresnel_reflectance"]/terms[ref]["fresnel_reflectance"]
    return terms,float(f("B11")),float(f("B12"))

with (FULL/"all_33_scene_final_status.csv").open(newline="",encoding="utf-8-sig") as f:
    scenes=[r["scene"] for r in csv.DictReader(f) if r["decision"].startswith("RETAIN_")]
records={}
for scene in scenes:
    safe=safe_for(scene); meta=archive.read_metadata(safe)
    aot=float(json.loads((SOURCE/"06_ACOLITE"/"AOT"/scene/"AOT_statistics.json").read_text())["water_median_used"])
    terms,f11,f12=factors(meta,aot)
    records[scene]={"safe":str(safe),"native_B01":str(archive.find_band(safe,"B01",60)),"boa_quantification_value":meta["boa_quantification_value"],"boa_offset_B01":meta["boa_offsets"].get(0,0.0),"solar_zenith_deg":meta["sun_zenith_angle"],"AOT_550":aot,"factor_from_B11":f11,"factor_from_B12":f12,"optical_terms":terms}
    print(scene,f11,f12,flush=True)
OUT.write_text(json.dumps(records,indent=2),encoding="utf-8")
print(f"Wrote {OUT}")
