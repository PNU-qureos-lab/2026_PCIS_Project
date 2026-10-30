"""Generate the portable JSON/JavaScript inventory used by the HTML viewer."""
from pathlib import Path
import json

# script -> 05_SHARED_UTILITIES -> 03_CODE -> package root
ROOT=Path(__file__).resolve().parents[2]
VIEWER=ROOT/"06_DATA_INVENTORY_AND_VIEWER"
SKIP={".venv","copy_logs","__pycache__"}

def metadata(relative:Path):
    parts=relative.parts
    sensor="Sentinel-2" if any("SENTINEL2" in p.upper() for p in parts) else ("Planet Dove" if any("PLANET" in p.upper() for p in parts) else "")
    country=next((p for p in parts if p[:3] in {"01_","02_","03_","04_"} and p.upper() in {"01_INDONESIA","02_FIJI","03_TONGA","04_TUVALU"}),"")
    island=next((p for p in parts if any(x in p.upper() for x in ("TIDUNG","TONGATAPU","FUNAFUTI","SAVAVE","PROJECT_AOI"))),"")
    date=next((p for p in parts if len(p)>=8 and p[:8].isdigit()),"")
    category=parts[0] if parts else ""
    return category,sensor,country,island,date

rows=[]
for path in ROOT.rglob("*"):
    if not path.is_file() or any(part in SKIP for part in path.parts):continue
    rel=path.relative_to(ROOT); category,sensor,country,island,date=metadata(rel)
    rows.append({"category":category,"sensor":sensor,"country":country,"island":island,"date":date,"path":str(rel),"bytes":path.stat().st_size})
rows.sort(key=lambda r:r["path"].lower())
VIEWER.mkdir(parents=True,exist_ok=True);(VIEWER/"assets").mkdir(exist_ok=True)
(VIEWER/"data_inventory.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding="utf-8")
(VIEWER/"assets"/"data_inventory.js").write_text("window.PCIS_INVENTORY="+json.dumps(rows,ensure_ascii=False)+";\n",encoding="utf-8")
print(f"Inventoried {len(rows)} files")
