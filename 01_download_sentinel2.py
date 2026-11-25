#!/usr/bin/env python
"""01 - Search and download Sentinel-2 Level-2A scenes from the Copernicus Data Space Ecosystem (CDSE).

INPUT   an area (lon/lat box), a date range and a maximum cloud percentage
OUTPUT  <out>/<name>_step1_search_results.csv          all matching scenes (date, tile, cloud %, size), clearest first
        <out>/<product>.SAFE                           downloaded and unzipped scene(s) -> input of script 02

STEPS
  step1  search the public CDSE catalogue (no login needed) and write the list, clearest scene first
  step2  download the chosen scene(s) (login needed: environment variables CDSE_USER and CDSE_PASS)

Examples:
  python 01_download_sentinel2.py --aoi 106.46 -5.83 106.55 -5.77 --start 2026-06-01 --end 2026-09-30 --name Tidung --out downloads
  export CDSE_USER=you@example.com; read -s CDSE_PASS; export CDSE_PASS
  python 01_download_sentinel2.py --aoi 106.46 -5.83 106.55 -5.77 --start 2026-06-01 --end 2026-09-30 --name Tidung --out downloads --download 1
"""
import argparse, csv, json, os, sys, time, urllib.parse, zipfile
from pathlib import Path

import requests

# ============================== RULES AND SETTINGS ==============================
CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value"
PRODUCT_TYPE = "S2MSI2A"         # Level-2A (Sen2Cor bottom-of-atmosphere reflectance) - required by script 02
MAX_CLOUD = 10.0                 # default: scene cloud cover <= 10 % (whole tile; cloud over the reef is checked again in script 02)
# =================================================================================


def search(aoi, start, end, max_cloud):
    w, s, e, n = aoi
    poly = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    flt = (f"Collection/Name eq 'SENTINEL-2' and Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and a/OData.CSC.StringAttribute/Value eq '{PRODUCT_TYPE}') "
           f"and OData.CSC.Intersects(area=geography'SRID=4326;{poly}') "
           f"and Attributes/OData.CSC.DoubleAttribute/any(a:a/Name eq 'cloudCover' and a/OData.CSC.DoubleAttribute/Value le {max_cloud}) "
           f"and ContentDate/Start ge {start}T00:00:00.000Z and ContentDate/Start le {end}T23:59:59.999Z")
    url = CATALOGUE + "?" + urllib.parse.urlencode({"$filter": flt, "$top": "1000", "$expand": "Attributes"})
    rows = []
    for p in requests.get(url, timeout=120).json().get("value", []):
        att = {a["Name"]: a.get("Value") for a in p.get("Attributes", [])}
        rows.append({"date": p["ContentDate"]["Start"][:10], "tile": p["Name"].split("_")[5], "cloud_percent": round(float(att.get("cloudCover", 0)), 3),
                     "size_GB": round(p.get("ContentLength", 0) / 1e9, 2), "online": p.get("Online"), "product": p["Name"], "id": p["Id"]})
    return sorted(rows, key=lambda r: (r["cloud_percent"], r["date"]))


def token():
    u, pw = os.environ.get("CDSE_USER"), os.environ.get("CDSE_PASS")
    if not u or not pw: raise SystemExit("Set CDSE_USER and CDSE_PASS (your Copernicus Data Space login) before --download.")
    r = requests.post(TOKEN_URL, data={"client_id": "cdse-public", "username": u, "password": pw, "grant_type": "password"}, timeout=60)
    if r.status_code != 200: raise SystemExit(f"CDSE login refused: {r.text[:300]}")
    return r.json()["access_token"]


def download(row, out):
    safe = out / row["product"]
    if safe.exists(): print("already downloaded:", safe.name); return safe
    zp = out / (row["product"].replace(".SAFE", "") + ".zip"); done = zp.stat().st_size if zp.exists() else 0
    hdr = {"Authorization": f"Bearer {token()}"}
    if done: hdr["Range"] = f"bytes={done}-"                      # resume a partial download
    with requests.get(DOWNLOAD.format(id=row["id"]), headers=hdr, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(zp, "ab" if done else "wb") as h:
            for chunk in r.iter_content(8 << 20): h.write(chunk)
    with zipfile.ZipFile(zp) as z: z.extractall(out)
    zp.unlink(); print("downloaded:", safe.name); return safe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", required=True, type=float, nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--start", required=True, help="YYYY-MM-DD"); ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--max-cloud", type=float, default=MAX_CLOUD)
    ap.add_argument("--name", required=True); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--download", type=int, default=0, help="download the N clearest scenes (0 = only search)")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    print("step1  search CDSE catalogue")
    rows = search(a.aoi, a.start, a.end, a.max_cloud)
    with open(a.out / f"{a.name}_step1_search_results.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["date", "tile", "cloud_percent", "size_GB", "online", "product", "id"]); w.writeheader(); w.writerows(rows)
    print(f"       {len(rows)} scenes with cloud <= {a.max_cloud} % -> {a.name}_step1_search_results.csv")
    for r in rows[:5]: print(f"       {r['date']} {r['tile']} cloud {r['cloud_percent']} % {r['size_GB']} GB")
    if a.download:
        print(f"step2  download the {a.download} clearest scene(s)")
        for r in rows[:a.download]: download(r, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
