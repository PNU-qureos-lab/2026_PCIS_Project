"""
Sentinel-2 catalogue search, scene selection, and download utility.

HANDOFF GUIDE — WHAT THE NEXT USER MUST PREPARE
================================================

1. Computer and account
   - Run this script with Python 3.9 or newer on Linux or WSL.
   - Create a free Copernicus Data Space Ecosystem (CDSE) account:
     https://dataspace.copernicus.eu/
   - Make sure the computer has internet access and enough free disk space.
     Search first without --download; the resulting CSV reports the estimated
     download size in the `size_gb` column and at the end of the run.

2. Python environment
   Recommended setup:

       python3 -m venv .venv
       source .venv/bin/activate
       python -m pip install --upgrade pip
       python -m pip install requests pandas tqdm

3. Folders that must be checked before running
   - BASE_DIR: working directory; CSV search results are written below it.
   - OUTPUT_DIR: downloaded and extracted Sentinel-2 .SAFE products go here.
   - The current OUTPUT_DIR (/mnt/h/Sentinel-2) assumes a Windows H: drive
     mounted in WSL. Change it if that drive/path does not exist.
   - The user must have permission to create and write both directories.

4. Credentials (required only when using --download)
   Do not type a real password into this file. Set environment variables in
   the same terminal before running the script:

       export CDSE_USER='your-CDSE-email@example.com'
       export CDSE_PASS='your-CDSE-password'

   Treat terminal history and shared shell profiles carefully. For a shared
   computer, use a protected secret-management method where possible.

5. Project settings to review
   - DATE_RANGES controls the normal search period for each site.
   - SURVEY_WINDOWS contains important survey/bleaching dates.
   - SITE_AOIS contains bounding boxes in (south, west, north, east) order.
   - SITE_TILES contains fallback Sentinel-2 MGRS tile IDs.
   - DEFAULT_MAX_CLOUD is the preferred maximum cloud percentage.
   Confirm these scientific inputs with the project owner before changing them.

HOW TO USE IT SAFELY
====================

Show all command-line options:

    python sentinel2_search_download_commented.py --help

Start with a small diagnostic search (no files are downloaded):

    python sentinel2_search_download_commented.py --site Tidung --level L2A \
        --diagnose --year 2025

Select the best yearly scenes and write a CSV (still no download):

    python sentinel2_search_download_commented.py --site Fiji --level L2A \
        --yearly --year 2024 --max-cloud 20

After reviewing the generated CSV and estimated size, repeat with --download:

    python sentinel2_search_download_commented.py --site Fiji --level L2A \
        --yearly --year 2024 --max-cloud 20 --download

Reuse an existing selection CSV instead of searching again:

    python sentinel2_search_download_commented.py --site Fiji --level L2A \
        --yearly --year 2024 --skip-search --download

MODE SUMMARY
============
   --diagnose    Count available products by year and processing level.
   --survey      Select scenes near dates in SURVEY_WINDOWS.
   --monthly     Select the lowest-cloud scene for every tile and month.
   --yearly      Select the lowest-cloud scene for every tile and year.
                 This is the default when no mode flag is supplied.
   --all-scenes  Select every matching scene; this can require large storage.

IMPORTANT BEHAVIOUR
===================
   - Search and CSV creation do not require CDSE credentials; downloading does.
   - If nothing meets --max-cloud, survey/monthly/yearly selection may keep the
     best available cloudy image and mark it `best_available` in the CSV.
   - Downloads resume a partial ZIP when the server supports byte ranges.
   - A successful ZIP is extracted into a .SAFE folder and then deleted.
   - Existing .SAFE folders are skipped, so rerunning is normally safe.
   - Do not interrupt extraction or manually rename partial ZIP/.SAFE folders.
   - `--all-scenes --download` can download a very large amount of data; always
     run it once without --download and inspect the CSV/size first.

OUTPUTS
=======
   Search CSVs: BASE_DIR/search_results/
   Image data:  OUTPUT_DIR/<site>/<L1C-or-L2A>/<product>.SAFE/

Common problems:
   - "Set CDSE_USER and CDSE_PASS first": credentials are missing.
   - Permission/path error: fix BASE_DIR or OUTPUT_DIR and folder permissions.
   - HTTP 401/403: check the CDSE account and credentials.
   - HTTP 429: the service is rate-limiting; the script waits and retries.
   - "Offline": that catalogue product is not currently downloadable.
"""

import os, re, time, argparse, zipfile
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import requests, pandas as pd
from tqdm import tqdm

# ---- PORTABLE PATH CONFIGURATION -------------------------------------------
# Default paths are derived from the package root, so moving the complete
# folder does not require editing this source file.
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PACKAGE_ROOT
OUTPUT_DIR = PACKAGE_ROOT / "04_DATA" / "01_RAW" / "01_SENTINEL2_L2A"
RESULT_DIR = PACKAGE_ROOT / "05_RESULTS" / "00_SENTINEL2_DOWNLOAD_LOGS"
CDSE_USER = os.environ.get("CDSE_USER", "your_email@example.com")
CDSE_PASS = os.environ.get("CDSE_PASS", "your_password")
CDSE_ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_DL_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value"
CHUNK_SIZE = 8 * 1024 * 1024
PRODUCT_TYPES = {"L1C": "S2MSI1C", "L2A": "S2MSI2A"}
# ---- PROJECT CONFIGURATION: validate dates, AOIs, and tiles with the owner. ----
DATE_RANGES = {"Fiji": ("2020-01-01","2026-12-31"), "Tonga": ("2020-01-01","2026-12-31"), "Tuvalu": ("2020-01-01","2026-12-31"), "Tidung": ("2020-01-01","2026-12-31")}
SURVEY_BUFFER_DAYS = 15
SURVEY_WINDOWS = {
    "Tidung": [("2025-10-28","Field survey Oct 2025"),("2026-06-23","Field survey Jun 2026")],
    "Tonga": [("2024-01-22","Before bleaching Jan 2024"),("2024-06-20","Peak bleaching Jun 2024"),("2026-01-19","Before bleaching Jan 2026"),("2026-05-04","Peak bleaching May 2026")],
    "Tuvalu": [("2023-12-18","Before bleaching Dec 2023"),("2024-06-03","Peak bleaching Jun 2024")],
    "Fiji": [("2023-01-09","Before bleaching Jan 2023"),("2023-05-22","Peak bleaching May 2023")],
}
SITE_TILES = {
    "Fiji": ["60KVF","60KWD","60KWE","60KWF","60KWG","60KXD","60KXE","60KXF","60KXG","60KYD","60KYE","60KYF","60KYG","01KAU","01KAV","01KAA","01KAB","01LAC","01KBU","01KBV","01KBA","01KBB","01LBC"],
    "Tonga": ["01KFA","01KFS","01KFT","01KFU","01KFV","01KGA","01KGS","01KGT","01KGU","01KGV","01KHA","01KHS","01KHT","01KHU","01KHV","02KKB","02KKC","02KKD","02KKE","02KKF"],
    "Tuvalu": ["01LAH","01LAJ","01LAK","01LAL","01MAM","60LXN","60LXP","60LXQ","60LXR","60LYN","60LYP","60LYQ","60LYR","60MUS","60MUT","60MUU","60MVS","60MVT","60MVU","60MWS","60MWT","60MWU","60MXS","60MYS"],
    "Tidung": ["48MXU"],
}
SITE_AOIS = {
    "Tidung": {"name":"Tidung Kecil Island, Indonesia","bbox":(-5.816,106.495,-5.792,106.548),"parent":"Tidung"},
    "Tonga": {"name":"Tongatapu, Tonga","bbox":(-21.30,-175.38,-20.98,-174.98),"parent":"Tonga"},
    "Tuvalu_Funafuti": {"name":"Funafuti Atoll, Tuvalu","bbox":(-8.64,179.10,-8.42,179.30),"parent":"Tuvalu"},
    "Tuvalu_Savave": {"name":"Savave, Nanumea, Tuvalu","bbox":(-5.718,176.078,-5.637,176.156),"parent":"Tuvalu"},
    "Fiji": {"name":"Viti Levu, Fiji","bbox":(-18.35,177.15,-17.15,178.85),"parent":"Fiji"},
}
DEFAULT_MAX_CLOUD = 20.0

def bbox_to_wkt_polygon(bbox):
    s,w,n,e = bbox
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
def extract_tile_from_name(name):
    m = re.search(r"_T(\d{2}[A-Z]{3})_", str(name))
    return m.group(1) if m else "unknown"
def get_cloud(p):
    for a in p.get("Attributes",[]) or []:
        if a.get("Name")=="cloudCover":
            try: return float(a.get("Value"))
            except: pass
    return 999.0
def get_date(p): return p.get("ContentDate",{}).get("Start","")[:10]
def get_datetime(p): return p.get("ContentDate",{}).get("Start","")
def aoi_keys_for_site(site): return [k for k,v in SITE_AOIS.items() if v.get("parent")==site]
def date_clip_to_today(d):
    today=datetime.today().strftime("%Y-%m-%d")
    return min(d,today)
def safe_size_gb(p):
    try: return round(float(p.get("ContentLength",0))/1e9,3)
    except: return 0.0
def product_to_record(product,site,aoi_key,level,note=""):
    name=product.get("Name",""); cloud=get_cloud(product)
    return {"site":site,"aoi":aoi_key,"tile":extract_tile_from_name(name),"date":get_date(product),"datetime":get_datetime(product),"type":level,"cloud_pct":round(cloud,2) if cloud<999 else None,"cloud_flag":"ok" if cloud<=DEFAULT_MAX_CLOUD else "marginal","selection_note":note,"product_id":product.get("Id",""),"name":name,"online":product.get("Online",True),"size_gb":safe_size_gb(product)}

def search_aoi_window(aoi_key,date_start,date_end,level,session=None):
    if session is None: session=requests.Session()
    aoi=SITE_AOIS[aoi_key]; wkt=bbox_to_wkt_polygon(aoi["bbox"]); ptype=PRODUCT_TYPES[level]
    date_end=date_clip_to_today(date_end)
    if date_start>date_end: return []
    filt=(f"Collection/Name eq 'SENTINEL-2' and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{ptype}') and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and ContentDate/Start ge {date_start}T00:00:00.000Z and ContentDate/Start le {date_end}T23:59:59.999Z")
    products=[]; skip=0
    while True:
        try:
            r=session.get(CDSE_ODATA,params={"$filter":filt,"$orderby":"ContentDate/Start asc","$top":100,"$skip":skip,"$expand":"Attributes"},timeout=90)
            r.raise_for_status(); batch=r.json().get("value",[])
        except Exception as e:
            print(f"    [AOI error] {aoi_key} {level} {date_start}..{date_end}: {e}"); return []
        if not batch: break
        products.extend(batch)
        if len(batch)<100: break
        skip+=100; time.sleep(0.1)
    uniq={}
    for p in products:
        pid=p.get("Id")
        if pid: uniq[pid]=p
    return list(uniq.values())

def search_tile_window(tile_id,date_start,date_end,level,session=None):
    if session is None: session=requests.Session()
    ptype=PRODUCT_TYPES[level]; date_end=date_clip_to_today(date_end)
    if date_start>date_end: return []
    filt=(f"Collection/Name eq 'SENTINEL-2' and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{ptype}') and contains(Name,'_T{tile_id}_') and ContentDate/Start ge {date_start}T00:00:00.000Z and ContentDate/Start le {date_end}T23:59:59.999Z")
    products=[]; skip=0
    while True:
        try:
            r=session.get(CDSE_ODATA,params={"$filter":filt,"$orderby":"ContentDate/Start asc","$top":100,"$skip":skip,"$expand":"Attributes"},timeout=90)
            r.raise_for_status(); batch=r.json().get("value",[])
        except Exception as e:
            print(f"    [Tile error] {tile_id} {level}: {e}"); return []
        if not batch: break
        products.extend(batch)
        if len(batch)<100: break
        skip+=100; time.sleep(0.1)
    return list({p.get("Id"):p for p in products if p.get("Id")}.values())

def search_site_window(site,date_start,date_end,level,tile_fallback=True):
    session=requests.Session(); products={}; aoi_hits=defaultdict(int)
    for aoi_key in aoi_keys_for_site(site):
        found=search_aoi_window(aoi_key,date_start,date_end,level,session)
        aoi_hits[aoi_key]+=len(found)
        for p in found:
            pid=p.get("Id")
            if pid: products[pid]=(p,aoi_key)
    if not products and tile_fallback:
        print(f"    [Fallback] AOI=0 for {site}; trying tiles")
        for tile in SITE_TILES.get(site,[]):
            found=search_tile_window(tile,date_start,date_end,level,session)
            for p in found:
                pid=p.get("Id")
                if pid: products[pid]=(p,f"tile:{tile}")
    return list(products.values()),dict(aoi_hits)

def diagnose_site(site,levels,start_date=None,end_date=None):
    d0,d1=DATE_RANGES[site]
    if start_date: d0=start_date
    if end_date: d1=end_date
    d1=date_clip_to_today(d1); y0,y1=int(d0[:4]),int(d1[:4]); rows=[]
    print(f"\n{'='*76}\nDIAGNOSE: {site}\nAOIs: {aoi_keys_for_site(site)}\nPeriod: {d0} to {d1}\n{'='*76}")
    for year in range(y0,y1+1):
        ys=max(d0,f"{year}-01-01"); ye=min(d1,f"{year}-12-31")
        if ys>ye: continue
        for level in levels:
            found,hits=search_site_window(site,ys,ye,level)
            products=[x[0] for x in found]
            tiles=sorted({extract_tile_from_name(p.get("Name","")) for p in products})
            clouds=[get_cloud(p) for p in products if get_cloud(p)<999]
            mc=min(clouds) if clouds else None
            print(f"  {year} [{level}] products={len(products):3d} tiles={','.join(tiles) if tiles else '-'} min_cloud={mc if mc is not None else '-'}")
            rows.append({"site":site,"year":year,"level":level,"products":len(products),"tiles":",".join(tiles),"min_cloud_pct":round(mc,2) if mc is not None else None})
    df=pd.DataFrame(rows); RESULT_DIR.mkdir(parents=True,exist_ok=True)
    out=RESULT_DIR/f"diagnose_{site}.csv"; df.to_csv(out,index=False); print(f"\nSaved: {out}"); return df

def select_survey(site,levels,max_cloud,buffer_days,survey_date=None):
    windows=SURVEY_WINDOWS.get(site,[])
    if survey_date:
        windows=[(d,l) for d,l in windows if d==survey_date]
        if not windows: windows=[(survey_date,f"User date {survey_date}")]
    if not windows: print(f"[!] No survey windows for {site}"); return []
    records=[]; seen=set()
    for center_date,label in windows:
        center=datetime.strptime(center_date,"%Y-%m-%d")
        d0=(center-timedelta(days=buffer_days)).strftime("%Y-%m-%d")
        d1=(center+timedelta(days=buffer_days)).strftime("%Y-%m-%d")
        print(f"\n  {label}: {d0} to {d1}")
        for level in levels:
            found,_=search_site_window(site,d0,d1,level)
            if not found: print(f"    [{level}] 0 products"); continue
            good=[(p,a) for p,a in found if get_cloud(p)<=max_cloud]
            chosen=good; fb=False
            if not chosen: chosen=[min(found,key=lambda x:get_cloud(x[0]))]; fb=True
            print(f"    [{level}] cat={len(found)}, <={max_cloud:.0f}%={len(good)}, sel={len(chosen)}{' [fallback]' if fb else ''}")
            for p,aoi_key in chosen:
                pid=p.get("Id")
                if not pid or pid in seen: continue
                seen.add(pid)
                rec=product_to_record(p,site,aoi_key,level,note=f"{label}; center={center_date}; +/-{buffer_days}d")
                rec["anchor_date"]=center_date
                rec["date_gap_days"]=abs((datetime.strptime(rec["date"],"%Y-%m-%d")-center).days)
                rec["cloud_flag"]="ok" if (rec["cloud_pct"] is not None and rec["cloud_pct"]<=max_cloud) else "best_available"
                records.append(rec)
    return records

def select_monthly(site,levels,max_cloud,start_date=None,end_date=None):
    import calendar
    d0,d1=DATE_RANGES[site]
    if start_date: d0=start_date
    if end_date: d1=end_date
    d1=date_clip_to_today(d1)
    cur=datetime.strptime(d0[:7]+"-01","%Y-%m-%d"); end_month=datetime.strptime(d1[:7]+"-01","%Y-%m-%d")
    records=[]; seen=set()
    while cur<=end_month:
        y,m=cur.year,cur.month
        ms=f"{y}-{m:02d}-01"; me=f"{y}-{m:02d}-{calendar.monthrange(y,m)[1]:02d}"
        ms=max(ms,d0); me=min(me,d1)
        print(f"\n  {site} {y}-{m:02d}")
        for level in levels:
            found,_=search_site_window(site,ms,me,level)
            if not found: print(f"    [{level}] 0 products"); continue
            by_tile=defaultdict(list)
            for p,aoi_key in found: by_tile[extract_tile_from_name(p.get("Name",""))].append((p,aoi_key))
            for tile,candidates in sorted(by_tile.items()):
                best=min(candidates,key=lambda x:get_cloud(x[0])); p,aoi_key=best
                pid=p.get("Id")
                if not pid or pid in seen: continue
                seen.add(pid); cloud=get_cloud(p)
                note="monthly_best"
                if cloud>max_cloud: note+=f"; best_available_above_{max_cloud:g}%"
                rec=product_to_record(p,site,aoi_key,level,note=note)
                rec["year"]=y; rec["month"]=m
                rec["cloud_flag"]="ok" if cloud<=max_cloud else "best_available"
                records.append(rec)
                print(f"    [{level}] tile={tile} cand={len(candidates):2d} best={rec['date']} cloud={cloud:.2f}%")
        if m==12: cur=datetime(y+1,1,1)
        else: cur=datetime(y,m+1,1)
    return records

def select_yearly(site,levels,max_cloud,start_date=None,end_date=None):
    d0,d1=DATE_RANGES[site]
    if start_date: d0=start_date
    if end_date: d1=end_date
    d1=date_clip_to_today(d1); y0,y1=int(d0[:4]),int(d1[:4])
    records=[]; seen=set()
    for y in range(y0,y1+1):
        ys=max(d0,f"{y}-01-01"); ye=min(d1,f"{y}-12-31")
        if ys>ye: continue
        print(f"\n  {site} {y}")
        for level in levels:
            found,_=search_site_window(site,ys,ye,level)
            if not found: print(f"    [{level}] 0 products"); continue
            by_tile=defaultdict(list)
            for p,aoi_key in found: by_tile[extract_tile_from_name(p.get("Name",""))].append((p,aoi_key))
            for tile,candidates in sorted(by_tile.items()):
                p,aoi_key=min(candidates,key=lambda x:get_cloud(x[0]))
                pid=p.get("Id")
                if not pid or pid in seen: continue
                seen.add(pid); cloud=get_cloud(p)
                rec=product_to_record(p,site,aoi_key,level,note="yearly_best")
                rec["year"]=y; rec["cloud_flag"]="ok" if cloud<=max_cloud else "best_available"
                records.append(rec)
                print(f"    [{level}] tile={tile} cand={len(candidates):3d} best={rec['date']} cloud={cloud:.2f}%")
    return records

def select_all_scenes(site,levels,max_cloud,start_date=None,end_date=None):
    d0,d1=DATE_RANGES[site]
    if start_date: d0=start_date
    if end_date: d1=end_date
    d1=date_clip_to_today(d1); records=[]; seen=set()
    y0,y1=int(d0[:4]),int(d1[:4])
    for y in range(y0,y1+1):
        ys=max(d0,f"{y}-01-01"); ye=min(d1,f"{y}-12-31")
        if ys>ye: continue
        print(f"\n  {site} {y} ALL SCENES")
        for level in levels:
            found,_=search_site_window(site,ys,ye,level)
            print(f"    [{level}] products={len(found)}")
            for p,aoi_key in found:
                pid=p.get("Id")
                if not pid or pid in seen: continue
                seen.add(pid); cloud=get_cloud(p)
                rec=product_to_record(p,site,aoi_key,level,note="all_scenes")
                rec["year"]=y; rec["cloud_flag"]="ok" if cloud<=max_cloud else "cloudy"
                records.append(rec)
    records.sort(key=lambda r:(r["site"],r["type"],r["date"],r["tile"]))
    return records

class CDSEAuth:
    """Request and refresh a CDSE access token using environment credentials."""
    def __init__(self,user,password): self.user=user; self.password=password; self._token=None; self._expiry=0
    def token(self):
        if time.time()>self._expiry-30: self._refresh()
        return self._token
    def _refresh(self):
        r=requests.post(CDSE_TOKEN_URL,data={"client_id":"cdse-public","username":self.user,"password":self.password,"grant_type":"password"},timeout=30)
        r.raise_for_status(); j=r.json(); self._token=j["access_token"]; self._expiry=time.time()+j.get("expires_in",600)
        print("  [Auth] Token refreshed")
    def headers(self): return {"Authorization":f"Bearer {self.token()}"}

def safe_and_zip_paths(out_dir,name):
    name=str(name).strip()
    safe_name=name if name.endswith(".SAFE") else name+".SAFE"
    zip_name=name.replace(".SAFE","")+".zip"
    return out_dir/safe_name, out_dir/zip_name

def download_product(pid,name,out_path,auth):
    url=CDSE_DL_URL.format(id=pid); existing=out_path.stat().st_size if out_path.exists() else 0
    for attempt in range(1,4):
        try:
            headers=auth.headers().copy()
            if existing: headers["Range"]=f"bytes={existing}-"; print(f"  [Resume] {existing/1e6:.1f} MB")
            r=requests.get(url,headers=headers,stream=True,timeout=120)
            if r.status_code==429: wait=int(r.headers.get("Retry-After",60)); print(f"  [Rate limit] {wait}s"); time.sleep(wait); continue
            if r.status_code==503: print("  [Offline]"); return False
            r.raise_for_status(); total=int(r.headers.get("Content-Length",0))+existing
            with open(out_path,"ab" if existing else "wb") as f, tqdm(total=total,initial=existing,unit="B",unit_scale=True,desc=name[:50],ncols=90) as bar:
                for chunk in r.iter_content(CHUNK_SIZE):
                    if chunk: f.write(chunk); bar.update(len(chunk))
            return True
        except Exception as e: print(f"  [Error {attempt}/3] {e}"); time.sleep(10*attempt)
    return False

def extract_zip(zip_path,out_dir):
    try:
        with zipfile.ZipFile(zip_path,"r") as z: z.extractall(out_dir)
        zip_path.unlink(missing_ok=True); return True
    except zipfile.BadZipFile: print(f"  [Bad zip] {zip_path}"); return False

def download_records(records,l1c_first=False):
    """Download selected products, extract .SAFE data, and skip existing data."""
    if not records: print("[!] Nothing to download"); return
    if CDSE_USER=="your_email@example.com" or CDSE_PASS=="your_password":
        raise RuntimeError("Set CDSE_USER and CDSE_PASS first")
    df=pd.DataFrame(records).drop_duplicates("product_id")
    if l1c_first:
        df["_o"]=df["type"].map({"L1C":0,"L2A":1}).fillna(9)
        df=df.sort_values(["_o","site","date"]).drop(columns="_o")
    else: df=df.sort_values(["site","date","type"])
    auth=CDSEAuth(CDSE_USER,CDSE_PASS); done=skipped=failed=0
    for _,row in df.iterrows():
        site=row["site"]; level=row["type"]; name=row["name"]; pid=row["product_id"]
        out_dir=OUTPUT_DIR/site/level; out_dir.mkdir(parents=True,exist_ok=True)
        safe_path,zip_path=safe_and_zip_paths(out_dir,name)
        print(f"\n[{site}] [{level}] {row['date']} cloud={row.get('cloud_pct')}%")
        print(f"  {name}")
        if safe_path.exists(): print("  [Skip] SAFE exists"); skipped+=1; continue
        if zip_path.exists() and zip_path.stat().st_size>1e6:
            print("  [Existing zip]")
            if extract_zip(zip_path,out_dir) and safe_path.exists(): skipped+=1; continue
        if str(row.get("online",True)).lower()=="false": print("  [Offline]"); failed+=1; continue
        ok=download_product(pid,name,zip_path,auth)
        if ok: extract_zip(zip_path,out_dir); done+=1
        else: failed+=1
    print(f"\nDownloaded={done}, skipped={skipped}, failed={failed}")

def main():
    global BASE_DIR, OUTPUT_DIR, RESULT_DIR
    parser=argparse.ArgumentParser(description="Sentinel-2 AOI search/download v4")
    parser.add_argument("--main-root",type=Path,default=PACKAGE_ROOT,
                        help="Main delivery folder; default input/output paths are derived from it")
    parser.add_argument("--download-dir",type=Path,default=None,
                        help="Optional override for downloaded Sentinel-2 SAFE products")
    parser.add_argument("--search-result-dir",type=Path,default=None,
                        help="Optional override for catalogue CSV and diagnostic outputs")
    parser.add_argument("--site",default=None)
    parser.add_argument("--level",default="both",choices=["L1C","L2A","both"])
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--diagnose",action="store_true")
    mode.add_argument("--survey",action="store_true")
    mode.add_argument("--monthly",action="store_true")
    mode.add_argument("--yearly",action="store_true")
    mode.add_argument("--all-scenes",action="store_true")
    parser.add_argument("--max-cloud",type=float,default=DEFAULT_MAX_CLOUD)
    parser.add_argument("--survey-buffer",type=int,default=SURVEY_BUFFER_DAYS)
    parser.add_argument("--survey-date",default=None)
    parser.add_argument("--start",default=None)
    parser.add_argument("--end",default=None)
    parser.add_argument("--year",type=int,default=None)
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--l1c-first",action="store_true")
    parser.add_argument("--skip-search",action="store_true")
    args=parser.parse_args()
    BASE_DIR=args.main_root.resolve()
    OUTPUT_DIR=(args.download_dir or (BASE_DIR/"04_DATA"/"01_RAW"/"01_SENTINEL2_L2A")).resolve()
    RESULT_DIR=(args.search_result_dir or (BASE_DIR/"05_RESULTS"/"00_SENTINEL2_DOWNLOAD_LOGS")).resolve()
    if args.year is not None: args.start=f"{args.year}-01-01"; args.end=f"{args.year}-12-31"
    sites=[args.site] if args.site else list(DATE_RANGES.keys())
    for s in sites:
        if s not in DATE_RANGES: parser.error(f"Unknown site: {s}")
    levels=["L1C","L2A"] if args.level=="both" else [args.level]
    RESULT_DIR.mkdir(parents=True,exist_ok=True); OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    mode_name="yearly"
    if args.diagnose: mode_name="diagnose"
    elif args.survey: mode_name="survey"
    elif args.monthly: mode_name="monthly"
    elif args.yearly: mode_name="yearly"
    elif args.all_scenes: mode_name="all_scenes"
    all_records=[]
    if args.skip_search:
        loaded=[]
        for site in sites:
            p=RESULT_DIR/f"{mode_name}_selection_{site}_v4.csv"
            if p.exists(): loaded.append(pd.read_csv(p)); print(f"Loaded: {p}")
            else: print(f"[Missing] {p}")
        if loaded:
            df=pd.concat(loaded,ignore_index=True)
            if args.year: df=df[df["date"].str[:4]==str(args.year)]; print(f"Filtered year {args.year}: {len(df)}")
            if args.level!="both": df=df[df["type"]==args.level]; print(f"Filtered level {args.level}: {len(df)}")
            all_records=df.drop_duplicates("product_id").to_dict("records")
    elif mode_name=="diagnose":
        for site in sites: diagnose_site(site,levels,args.start,args.end)
        return
    else:
        for site in sites:
            print(f"\n{'#'*80}\nSITE: {site} | MODE: {mode_name} | LEVELS: {levels}\n{'#'*80}")
            if mode_name=="survey": records=select_survey(site,levels,args.max_cloud,args.survey_buffer,args.survey_date)
            elif mode_name=="monthly": records=select_monthly(site,levels,args.max_cloud,args.start,args.end)
            elif mode_name=="all_scenes": records=select_all_scenes(site,levels,args.max_cloud,args.start,args.end)
            else: records=select_yearly(site,levels,args.max_cloud,args.start,args.end)
            if not records: print(f"[!] No records for {site}"); continue
            df_site=pd.DataFrame(records).drop_duplicates("product_id")
            out=RESULT_DIR/f"{mode_name}_selection_{site}_v4.csv"
            df_site.to_csv(out,index=False)
            print(f"\nSaved {len(df_site)} records: {out}")
            print(f"Tiles: {sorted(df_site['tile'].dropna().unique())}")
            print(f"Download: {df_site['size_gb'].sum():.2f} GB")
            all_records.extend(df_site.to_dict("records"))
        if all_records:
            combined=pd.DataFrame(all_records).drop_duplicates("product_id")
            combined_out=RESULT_DIR/f"{mode_name}_selection_all_v4.csv"
            combined.to_csv(combined_out,index=False)
            print(f"\nCombined: {combined_out}")
            print(f"Total: {len(combined)} scenes, {combined['size_gb'].sum():.2f} GB")
    if args.download and all_records:
        download_records(all_records,l1c_first=args.l1c_first)
    print("\nDone.")

if __name__=="__main__": main()
