#!/usr/bin/env python
"""Create a Sentinel-comparable four-band Planet HOPE/D2 package.

The Planet 2025-11-17 Hedley-corrected approximate-Rrs bands are averaged to
the exact 20 m grid of the Sentinel 2026-08-08 candidate shallow-water mask.
The original four-band project model is then run on the common footprint.
No depth clipping or post-hoc spatial smoothing is applied to the rasters.
"""
from __future__ import annotations
import csv, html, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from osgeo import gdal
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_planetscope_semianalytical_three_scenarios as core

SCENE = "2025-11-17"
WAVES = np.array([444, 492, 566, 666], dtype=float)
SOURCE = Path(r"H:\PlanetDove\Tidung_Standard_Hedley_Final_20260923")
SENTINEL_MASK = Path(r"H:\Sentinel-2\Tidung\Tidung_Glint_Corrected_Products_20260913\SEMIANALYTICAL_INPUT\20260808_S2A\20260808_S2A_candidate_shallow_water_mask.tif")
OUTPUT = Path(r"H:\PlanetDove\Tidung_Planet_4Band_Sentinel_Comparable_20260923")
NODATA = -9999.0


def target_grid():
    ds = gdal.Open(str(SENTINEL_MASK)); arr = ds.ReadAsArray()
    meta = {"geotransform": ds.GetGeoTransform(), "projection": ds.GetProjection(),
            "xsize": ds.RasterXSize, "ysize": ds.RasterYSize}
    return arr, meta


def warp_to_grid(path: Path, meta: dict, algorithm: str, src_nodata=None):
    gt = meta["geotransform"]
    bounds = [gt[0], gt[3] + meta["ysize"] * gt[5],
              gt[0] + meta["xsize"] * gt[1], gt[3]]
    options = dict(format="MEM", dstSRS=meta["projection"], outputBounds=bounds,
                   width=meta["xsize"], height=meta["ysize"], resampleAlg=algorithm,
                   dstNodata=NODATA)
    if src_nodata is not None: options["srcNodata"] = src_nodata
    ds = gdal.Warp("", str(path), **options)
    out = ds.ReadAsArray().astype(float); out[out == NODATA] = np.nan
    return out


def main():
    if OUTPUT.exists(): raise RuntimeError(f"Refusing to overwrite existing folder: {OUTPUT}")
    for name in ["00_READ_ME_FIRST","01_INPUT_20M","02_INTERMEDIATE","03_FINAL_PRODUCTS","04_QC","05_CODE"]:
        (OUTPUT/name).mkdir(parents=True, exist_ok=True)
    sentinel_mask, meta = target_grid()
    stacks=[]; input_paths=[]
    for wave in WAVES.astype(int):
        src = SOURCE/"04_GENERATED_OUTPUT"/SCENE/"05_Rrs_Approx"/f"{SCENE}_{wave}nm_Rrs_approx.tif"
        arr = warp_to_grid(src, meta, "average", NODATA); stacks.append(arr); input_paths.append(str(src))
        core.write_raster(OUTPUT/"01_INPUT_20M"/f"{SCENE}_{wave}nm_Rrs_approx_20m_mean.tif", arr, meta,
                          description=f"Planet approximate Rrs {wave} nm averaged to Sentinel 20 m grid")
    stack=np.stack(stacks)
    train_src=SOURCE/"04_GENERATED_OUTPUT"/SCENE/"01_Masks"/f"{SCENE}_hedley_training_water.tif"
    train=warp_to_grid(train_src,meta,"near",0)>0
    candidate=(sentinel_mask==1)&np.all(np.isfinite(stack),axis=0)&np.all(stack>0,axis=0)
    fit=train&np.all(np.isfinite(stack),axis=0)&np.all(stack>0,axis=0)
    if fit.sum()<100: raise RuntimeError(f"Insufficient Planet training-water overlap: {fit.sum()}")
    deep=np.median(stack[:,fit],axis=1)
    lut,lut_meta=core.build_lut(WAVES,deep)
    obs=stack[:,candidate].T; scale=np.maximum(deep,.002)
    tree=cKDTree(lut["predicted"]/scale[None,:]); _,neighbors=tree.query(obs/scale[None,:],k=16,workers=-1)
    pred=lut["predicted"][neighbors]; rel=(pred-obs[:,None,:])/np.maximum(obs[:,None,:],1e-6)
    costs=np.mean(rel*rel,axis=2); cols=np.argmin(costs,axis=1); choice=neighbors[np.arange(len(obs)),cols]
    shape=candidate.shape
    depth=np.full(shape,np.nan,"float32");sand=np.full(shape,np.nan,"float32");cost=np.full(shape,np.nan,"float32");rmse=np.full(shape,np.nan,"float32")
    depth[candidate]=lut["depth"][choice];sand[candidate]=lut["sand"][choice];cost[candidate]=costs[np.arange(len(obs)),cols];rmse[candidate]=np.sqrt(np.mean((lut["predicted"][choice]-obs)**2,axis=1))
    grass=np.where(np.isfinite(sand),1-sand,np.nan)
    boundary=100*float(np.mean((depth[candidate]==0)|(depth[candidate]==20)))
    rmse95=float(np.nanpercentile(rmse,95)); uncertain=candidate&((depth>5)|(rmse>rmse95))
    habitat=np.zeros(shape,np.uint8);habitat[candidate&(sand>=.7)]=1;habitat[candidate&(grass>=.7)]=2;habitat[candidate&(sand<.7)&(grass<.7)]=3;habitat[uncertain]=4
    mask_out=np.zeros(shape,np.uint8);mask_out[candidate]=1
    for name,arr,desc in [("depth_m",depth,"Lee-D2 estimated depth in metres"),("sand_fraction",sand,"Sand fraction 0-1"),("seagrass_fraction",grass,"Seagrass fraction 0-1"),("relative_cost",cost,"Mean squared relative spectral residual"),("Rrs_RMSE",rmse,"Spectral Rrs RMSE sr-1")]:
        core.write_raster(OUTPUT/"03_FINAL_PRODUCTS"/f"{SCENE}_Planet_4band_{name}.tif",arr,meta,description=desc)
    core.write_raster(OUTPUT/"02_INTERMEDIATE"/f"{SCENE}_common_Sentinel_shallow_mask.tif",mask_out,meta,dtype=gdal.GDT_Byte,nodata=0,description="1=Sentinel-defined shallow candidate with valid Planet four-band input")
    core.write_raster(OUTPUT/"03_FINAL_PRODUCTS"/f"{SCENE}_Planet_4band_habitat_exploratory.tif",habitat,meta,dtype=gdal.GDT_Byte,nodata=0,description="1=sand 2=seagrass 3=mixed 4=uncertain")
    v=depth[candidate]
    summary={"scene":SCENE,"bands_nm":WAVES.astype(int).tolist(),"grid":"Sentinel 20 m grid","spatial_resampling":"mean of valid Planet 3 m pixels","candidate_pixels":int(candidate.sum()),"training_pixels":int(fit.sum()),"deep_Rrs":deep.tolist(),"median_depth_m":float(np.median(v)),"p05_depth_m":float(np.percentile(v,5)),"p95_depth_m":float(np.percentile(v,95)),"percent_depth_gt5m":100*float(np.mean(v>5)),"depth_boundary_percent":boundary,"median_sand_fraction":float(np.nanmedian(sand)),"median_Rrs_RMSE_sr-1":float(np.nanmedian(rmse)),"p95_Rrs_RMSE_sr-1":rmse95,"deep_fit_relative_rmse":lut_meta["deep_fit_relative_rmse"],"decision":"EXPLORATORY ONLY — Sentinel-comparable geometry and scale; Planet deep-water/atmospheric reference remains unvalidated","no_cheating_checks":["No depth clipping in GeoTIFF","No spatial smoothing of inversion outputs","All four bands inverted jointly","Pixels >5 m retained; only displayed as black","Uncertain classification is explicit, not removed"],"lut":lut_meta,"created_utc":datetime.now(timezone.utc).isoformat()}
    (OUTPUT/"04_QC"/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (OUTPUT/"04_QC"/"summary.csv").open("w",newline="",encoding="utf-8-sig") as h:
        flat={k:(" ".join(map(str,val)) if isinstance(val,list) and k=="bands_nm" else val) for k,val in summary.items() if not isinstance(val,(dict,list))};w=csv.DictWriter(h,fieldnames=flat.keys());w.writeheader();w.writerow(flat)
    # Four-panel figure matched to the Sentinel presentation.
    rr,cc=np.where(candidate);pad=4;r0,r1=max(0,rr.min()-pad),min(shape[0],rr.max()+pad+1);c0,c1=max(0,cc.min()-pad),min(shape[1],cc.max()+pad+1);sl=np.s_[r0:r1,c0:c1]
    gt=meta["geotransform"];extent=[gt[0]+c0*gt[1],gt[0]+c1*gt[1],gt[3]+r1*gt[5],gt[3]+r0*gt[5]]
    fig,ax=plt.subplots(2,2,figsize=(22,13),constrained_layout=True)
    dc=plt.get_cmap("turbo").copy();dc.set_bad("white");dc.set_over("black");im=ax[0,0].imshow(depth[sl],extent=extent,origin="upper",cmap=dc,vmin=0,vmax=5);ax[0,0].set_title("Estimated water depth",fontsize=18);cb=fig.colorbar(im,ax=ax[0,0],extend="max",shrink=.82);cb.set_label("Depth (m); black = >5 m")
    fc=plt.get_cmap("YlGnBu").copy();fc.set_bad("white");im=ax[0,1].imshow(sand[sl],extent=extent,origin="upper",cmap=fc,vmin=0,vmax=1);ax[0,1].set_title("Sand fraction",fontsize=18);fig.colorbar(im,ax=ax[0,1],shrink=.82,label="Fraction (0–1)")
    gc=plt.get_cmap("YlGn").copy();gc.set_bad("white");im=ax[1,0].imshow(grass[sl],extent=extent,origin="upper",cmap=gc,vmin=0,vmax=1);ax[1,0].set_title("Seagrass fraction",fontsize=18);fig.colorbar(im,ax=ax[1,0],shrink=.82,label="Fraction (0–1)")
    hc=ListedColormap(["white","#e9c75f","#198754","#8846b7","#999999"]);hn=BoundaryNorm([-.5,.5,1.5,2.5,3.5,4.5],hc.N);ax[1,1].imshow(habitat[sl],extent=extent,origin="upper",cmap=hc,norm=hn);ax[1,1].set_title("Exploratory bottom classification",fontsize=18);ax[1,1].legend(handles=[Patch(color="#e9c75f",label="Sand dominant (≥70%)"),Patch(color="#198754",label="Seagrass dominant (≥70%)"),Patch(color="#8846b7",label="Mixed"),Patch(color="#999999",label="Uncertain: >5 m or high error")],loc="center left",bbox_to_anchor=(1.01,.5))
    for a in ax.ravel():a.set_xlabel("Easting (m)");a.set_ylabel("Northing (m)")
    fig.suptitle("Tidung Planet SuperDove — 2025-11-17\nFour-band HOPE/D2 on Sentinel-comparable 20 m grid",fontsize=24,weight="bold")
    fig.text(.5,.006,f"Same Sentinel shallow mask | 20 m mean Planet input | median depth {summary['median_depth_m']:.2f} m | >5 m {summary['percent_depth_gt5m']:.1f}% | deep-fit mismatch {summary['deep_fit_relative_rmse']*100:.1f}% | exploratory, not field validated",ha="center",fontsize=12)
    fig.savefig(OUTPUT/"04_QC"/"Planet_4band_Sentinel_comparable_four_panel.png",dpi=180,facecolor="white");plt.close(fig)
    readme=f'''# Planet four-band Sentinel-comparable result\n\nThis package processes one Planet scene ({SCENE}) using 444, 492, 566 and 666 nm. Planet approximate Rrs was averaged to the Sentinel 20 m grid, and only pixels inside the Sentinel 2026-08-08 candidate shallow-water mask were inverted.\n\n## Integrity\n\nNo depths were clipped or smoothed in the GeoTIFFs. Values above 5 m are retained and shown as black in the figure. The product remains exploratory because the Planet crop does not overlap the reviewed Sentinel offshore deep-water ROIs, and Planet SR/pi is not dedicated aquatic atmospheric correction.\n\n## Result\n\n- Candidate pixels: {summary['candidate_pixels']:,}\n- Median depth: {summary['median_depth_m']:.2f} m\n- 5th–95th percentiles: {summary['p05_depth_m']:.2f}–{summary['p95_depth_m']:.2f} m\n- Pixels >5 m: {summary['percent_depth_gt5m']:.2f}%\n- Depth grid boundary: {summary['depth_boundary_percent']:.3f}%\n- Deep-water relative model mismatch: {summary['deep_fit_relative_rmse']*100:.1f}%\n\nDecision: {summary['decision']}\n'''
    (OUTPUT/"00_READ_ME_FIRST"/"README.md").write_text(readme,encoding="utf-8")
    shutil.copy2(Path(__file__),OUTPUT/"05_CODE"/Path(__file__).name);shutil.copy2(core.__file__,OUTPUT/"05_CODE"/Path(core.__file__).name)
    (OUTPUT/"05_CODE"/"run_command.txt").write_text(f'"C:\\Users\\phili\\anaconda3\\envs\\Dr_doom\\python.exe" "{OUTPUT}\\05_CODE\\{Path(__file__).name}"\n',encoding="utf-8")
    report=f'''<!doctype html><html><head><meta charset="utf-8"><title>Planet four-band Sentinel-comparable result</title><style>body{{font:18px/1.55 Arial;background:#edf4f3;color:#15343a;margin:0}}main{{max-width:1300px;margin:auto;padding:28px}}section,header{{background:white;border:1px solid #cbdad7;border-radius:14px;padding:24px;margin:18px 0}}h1{{font-size:36px}}h2{{font-size:27px}}img{{width:100%}}.warn{{background:#fff3d8;border-left:6px solid #d77a00;padding:16px}}</style></head><body><main><header><h1>Tidung Planet SuperDove four-band result</h1><p>One scene ({SCENE}), four bands, Sentinel-comparable 20 m grid and shallow-water mask.</p></header><section><h2>Final four-panel result</h2><img src="04_QC/Planet_4band_Sentinel_comparable_four_panel.png"><p>Median depth {summary['median_depth_m']:.2f} m; 5th–95th percentiles {summary['p05_depth_m']:.2f}–{summary['p95_depth_m']:.2f} m; {summary['percent_depth_gt5m']:.1f}% deeper than 5 m.</p></section><section><h2>Integrity and limitation</h2><div class="warn">No clipping or smoothing was applied to output rasters. This is exploratory: deep-water model mismatch is {summary['deep_fit_relative_rmse']*100:.1f}%, and the Planet crop does not contain the reviewed Sentinel offshore reference areas.</div></section></main></body></html>'''
    (OUTPUT/"Tidung_Planet_4Band_Sentinel_Comparable.html").write_text(report,encoding="utf-8")
    manifest=[{"path":str(p.relative_to(OUTPUT)),"bytes":p.stat().st_size} for p in sorted(OUTPUT.rglob("*")) if p.is_file()]
    (OUTPUT/"00_READ_ME_FIRST"/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2));print(OUTPUT)

if __name__=="__main__":main()
