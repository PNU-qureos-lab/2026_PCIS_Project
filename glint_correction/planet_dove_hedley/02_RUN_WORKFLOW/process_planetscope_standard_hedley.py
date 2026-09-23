"""Build a self-contained, standard-equation Hedley package for Tidung SuperDove.

The correction follows Hedley, Harborne & Mumby (2005):

    rho_corrected(lambda) = rho(lambda) - b(lambda) * (rho_NIR - min(rho_NIR))

where b(lambda) is the ordinary least-squares slope fitted in a documented
open-water training region.  This implementation does not replace the minimum
with a percentile, truncate the NIR excess, scale the fitted correction, clip
negative reflectance, or hide negative pixels.  Land, cloud, shadow, haze and
invalid pixels are NoData; negative corrected water values remain in the files
and are reported by QC.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from scipy.ndimage import distance_transform_edt

from planetscope_temporal_mask import (
    MASK_DESCRIPTION,
    build_stable_land_mask,
    corrected_water_mask,
    strict_udm2_clear,
)


WAVES = [444, 492, 533, 566, 612, 666, 707, 866]
TARGETS = WAVES[:-1]
COLORS = ["#0067c5", "#00a6ca", "#00a878", "#73b21a", "#d4a600", "#df5b20", "#a63d9d"]
NODATA = -9999.0


def safe_extract(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)


def locate_planet_dir(root: Path) -> Path:
    for path in sorted(root.rglob("PSScene")):
        if list(path.glob("*_AnalyticMS_SR_8b_clip.tif")):
            return path
    raise RuntimeError(f"Planet Surface Reflectance files were not found under {root}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corr(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    a = x[keep] - np.mean(x[keep])
    b = y[keep] - np.mean(y[keep])
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / den) if den else float("nan")


def ordinary_least_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep].astype(np.float64)
    y = y[keep].astype(np.float64)
    if x.size < 3:
        raise RuntimeError("Too few pixels for Hedley regression")
    xm, ym = float(x.mean()), float(y.mean())
    denominator = float(np.sum((x - xm) ** 2))
    if denominator <= 0:
        raise RuntimeError("NIR training pixels have no variance")
    slope = float(np.sum((x - xm) * (y - ym)) / denominator)
    intercept = ym - slope * xm
    prediction = intercept + slope * x
    total = float(np.sum((y - ym) ** 2))
    r2 = 1.0 - float(np.sum((y - prediction) ** 2)) / total if total > 0 else float("nan")
    return slope, intercept, r2, int(x.size)


def create_like(path: Path, source, bands: int, dtype, nodata: float | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    output = gdal.GetDriverByName("GTiff").Create(
        str(path), source.RasterXSize, source.RasterYSize, bands, dtype,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    output.SetGeoTransform(source.GetGeoTransform())
    output.SetProjection(source.GetProjection())
    if nodata is not None:
        for index in range(1, bands + 1):
            output.GetRasterBand(index).SetNoDataValue(nodata)
    return output


def write_mask(path: Path, array: np.ndarray, source, description: str) -> None:
    output = create_like(path, source, 1, gdal.GDT_Byte, 0)
    band = output.GetRasterBand(1)
    band.WriteArray(array.astype(np.uint8))
    band.SetDescription(description)
    output.SetMetadataItem("VALUES", "0=excluded, 1=included")
    output.FlushCache()
    output = None


def write_float(path: Path, array: np.ndarray, source, mask: np.ndarray, description: str, unit: str) -> None:
    output = create_like(path, source, 1, gdal.GDT_Float32, NODATA)
    band = output.GetRasterBand(1)
    band.WriteArray(np.where(mask & np.isfinite(array), array, NODATA).astype(np.float32))
    band.SetDescription(description)
    band.SetUnitType(unit)
    output.SetMetadataItem("METHOD", "Hedley et al. (2005) standard equation")
    output.SetMetadataItem("NEGATIVE_POLICY", "Preserved for QC; never clipped")
    output.SetMetadataItem("MASK", MASK_DESCRIPTION)
    output.FlushCache()
    output = None


def write_stack(path: Path, arrays: dict[int, np.ndarray], source, mask: np.ndarray, prefix: str) -> None:
    output = create_like(path, source, len(WAVES), gdal.GDT_Float32, NODATA)
    for index, wave in enumerate(WAVES, 1):
        band = output.GetRasterBand(index)
        band.WriteArray(np.where(mask & np.isfinite(arrays[wave]), arrays[wave], NODATA).astype(np.float32))
        band.SetDescription(f"{prefix}_{wave}_nm")
        band.SetUnitType("unitless surface reflectance")
    output.SetMetadataItem("METHOD", "Hedley et al. (2005) standard equation")
    output.SetMetadataItem("CORRECTED_BANDS_NM", ",".join(map(str, TARGETS)))
    output.SetMetadataItem("REFERENCE_BAND_NM", "866 (unchanged)")
    output.SetMetadataItem("NEGATIVE_POLICY", "Preserved for QC; never clipped")
    output.FlushCache()
    output = None


def write_target_stack(path: Path, arrays: dict[int, np.ndarray], source, mask: np.ndarray, prefix: str) -> None:
    output = create_like(path, source, len(TARGETS), gdal.GDT_Float32, NODATA)
    for index, wave in enumerate(TARGETS, 1):
        band = output.GetRasterBand(index)
        band.WriteArray(np.where(mask & np.isfinite(arrays[wave]), arrays[wave], NODATA).astype(np.float32))
        band.SetDescription(f"{prefix}_{wave}_nm")
        band.SetUnitType("unitless surface reflectance")
    output.SetMetadataItem("METHOD", "Hedley et al. (2005) standard equation")
    output.SetMetadataItem("BANDS_NM", ",".join(map(str, TARGETS)))
    output.FlushCache()
    output = None


def rgb_array(data: dict[int, np.ndarray], mask: np.ndarray, limits=None):
    cube = np.dstack([data[666], data[566], data[492]])
    if limits is None:
        lo = np.nanpercentile(cube[mask], 2, axis=0)
        hi = np.nanpercentile(cube[mask], 98, axis=0)
    else:
        lo, hi = limits
    rgb = np.clip((cube - lo) / np.maximum(hi - lo, 1e-8), 0, 1)
    rgb = np.power(rgb, 0.85)
    return np.where(mask[..., None], rgb, 1.0), (lo, hi)


def save_masks_figure(path: Path, date: str, original, clear, water, training) -> None:
    rgb_valid = np.logical_and.reduce([np.isfinite(original[wave]) for wave in WAVES])
    rgb, _ = rgb_array(original, rgb_valid)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    panels = [(rgb, "1A · Original Planet SR RGB"), (clear, f"1B · Strict UDM2 clear · {clear.sum():,}"),
              (water, f"1C · Water retained · {water.sum():,}"), (training, f"1D · Hedley training water · {training.sum():,}")]
    cmaps = [None, "Greens", "Blues", "Oranges"]
    for ax, (image, title), cmap in zip(axes.ravel(), panels, cmaps):
        ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.axis("off")
    fig.suptitle(f"{date} · Step 1: input and pixels used", fontsize=18, weight="bold")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def save_fit_figure(path: Path, date: str, x: np.ndarray, original, bands) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    step = max(1, x.size // 12000)
    xs = x[::step]
    for ax, metric, color in zip(axes.ravel(), bands, COLORS):
        wave = metric["wave"]
        ys = original[wave][metric["training_mask"]][::step]
        ax.scatter(xs, ys, s=2, alpha=0.18, color=color, rasterized=True)
        line_x = np.array([float(np.min(x)), float(np.max(x))])
        line_y = metric["intercept"] + metric["slope"] * line_x
        ax.plot(line_x, line_y, color="#111827", linewidth=2)
        ax.set_title(f"{wave} nm · slope {metric['slope']:.3f} · R² {metric['r2']:.3f}", fontsize=11, weight="bold")
        ax.set_xlabel("866 nm SR")
        ax.set_ylabel(f"{wave} nm SR")
        ax.grid(alpha=0.2)
    axes.ravel()[-1].axis("off")
    fig.suptitle(f"{date} · Step 2: ordinary least-squares Hedley fits", fontsize=18, weight="bold")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def save_removed_figure(path: Path, date: str, removal, water) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    for ax, wave in zip(axes.ravel(), TARGETS):
        values = removal[wave]
        limit = max(abs(float(np.nanpercentile(values[water], 1))), abs(float(np.nanpercentile(values[water], 99))), 1e-6)
        image = ax.imshow(np.where(water, values, np.nan), cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(f"{wave} nm removed term", fontsize=11, weight="bold")
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.042, pad=0.02)
    axes.ravel()[-1].axis("off")
    fig.suptitle(f"{date} · Step 3: band-specific term subtracted", fontsize=18, weight="bold")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def save_comparison(path: Path, date: str, original, corrected, water) -> None:
    before, limits = rgb_array(original, water)
    after, _ = rgb_array(corrected, water, limits)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    for ax, image, title in zip(axes, [before, after], ["Before · original Planet SR", "After · standard Hedley"]):
        ax.imshow(image)
        ax.set_title(title, fontsize=15, weight="bold")
        ax.axis("off")
    fig.suptitle(f"{date} · Step 4: before and after with identical RGB stretch", fontsize=18, weight="bold")
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def save_spectra(path: Path, date: str, bands: list[dict]) -> None:
    before = [item["median_before"] for item in bands]
    after = [item["median_after"] for item in bands]
    fig, ax = plt.subplots(figsize=(12, 6.8), constrained_layout=True)
    ax.plot(TARGETS, before, "o--", linewidth=3, markersize=7, color="#334155", label="Before")
    ax.plot(TARGETS, after, "o-", linewidth=3, markersize=7, color="#00897b", label="After standard Hedley")
    ax.axhline(0, color="#dc2626", linewidth=1.5)
    ax.set_title(f"{date} · Step 5: median valid-water spectrum", fontsize=17, weight="bold")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Planet surface reflectance")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def save_negative_map(path: Path, date: str, negative: np.ndarray, water: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    display = np.full(water.shape, np.nan)
    display[water] = 0
    display[negative] = 1
    ax.imshow(display, cmap=matplotlib.colors.ListedColormap(["#e5f2ef", "#dc2626"]), vmin=0, vmax=1)
    ax.set_title(f"{date} · Step 6: pixels negative in one or more corrected bands\nred = negative; pale = non-negative water", fontsize=16, weight="bold")
    ax.axis("off")
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def classify(mean_r2: float, green_removed_percent: float, negative_any_percent: float) -> tuple[str, str]:
    if negative_any_percent > 1:
        return "REJECT", "Standard Hedley produces more than 1% negative water pixels."
    if mean_r2 >= 0.45 and green_removed_percent >= 5:
        return "ACCEPT WITH QC", "Visible bands covary with NIR and a measurable component is removed."
    if mean_r2 >= 0.25 and green_removed_percent >= 2:
        return "CAUTIOUS", "Some NIR-linked signal is present, but correction evidence is moderate."
    return "NO CORRECTION NEEDED", "The scene shows weak evidence of a removable NIR-linked component."


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_html(root: Path, scenes: list[dict]) -> Path:
    payload = json.dumps(scenes, separators=(",", ":"))
    rows = "".join(
        f"<tr><td>{s['date']}</td><td><b>{s['decision']}</b></td><td>{s['mean_r2']:.3f}</td>"
        f"<td>{s['green_removed_percent']:.2f}%</td><td>{s['negative_any_percent']:.3f}%</td>"
        f"<td>{s['spatial_preservation_r']:.3f}</td></tr>" for s in scenes
    )
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tidung Planet SuperDove Standard Hedley Results</title><style>
:root{{--ink:#17313a;--muted:#567078;--blue:#125fba;--teal:#00897b;--paper:#f3f7f6;--card:#fff;--line:#cbd9d6;--warn:#fff5dc}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:18px/1.55 Arial,sans-serif}}header{{background:#103f4b;color:white;padding:34px max(5vw,28px)}}header h1{{font-size:40px;margin:0 0 8px}}main{{max-width:1500px;margin:auto;padding:28px}}nav{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:22px}}nav a,.download{{padding:10px 14px;border:1px solid #8fb0cf;border-radius:8px;background:white;color:#064f9b;text-decoration:none;font-weight:700}}section{{background:white;border:1px solid var(--line);border-radius:14px;padding:26px;margin:20px 0}}h2{{font-size:28px;margin-top:0}}h3{{font-size:22px}}.plain{{background:#e9f7f3;border-left:5px solid var(--teal);padding:16px;border-radius:8px}}.warn{{background:var(--warn);border-left:5px solid #d97706;padding:16px;border-radius:8px}}.equation{{font-size:22px;font-weight:700;background:#eef5ff;border:1px solid #9ab8de;border-radius:10px;padding:18px;margin:16px 0}}.flow{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.step,.metric{{border:1px solid var(--line);border-radius:10px;padding:15px;background:#fbfdfc}}.step b,.metric b{{display:block;font-size:21px}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}label{{font-weight:700}}select{{font-size:18px;padding:10px;border-radius:8px;min-width:260px}}img{{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:10px;margin:14px 0}}table{{width:100%;border-collapse:collapse;font-size:16px}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left}}th{{background:#eaf2f0}}code{{background:#edf2f1;padding:2px 5px}}small{{color:var(--muted)}}@media(max-width:700px){{body{{font-size:16px}}header h1{{font-size:30px}}main{{padding:12px}}section{{padding:17px}}}}
</style></head><body><header><h1>Tidung · Planet SuperDove · Standard Hedley</h1><p>Planet Surface Reflectance input · published Hedley equation · all intermediate results and QC</p></header><main>
<nav><a href="#method">Method</a><a href="#workflow">Workflow</a><a href="#summary">All scenes</a><a href="#detail">Step-by-step results</a><a href="#files">Files and code</a></nav>
<section id="method"><h2>What was done</h2><div class="plain"><b>Easy concept:</b> surface glint tends to brighten both a visible band and the 866 nm NIR band. Hedley learns their relationship in selected open water and subtracts the NIR-linked component from each visible band.</div>
<div class="equation">ρ′λ = ρλ − bλ × (ρ866 − min(ρ866))</div>
<ul><li><b>Published part:</b> the equation, the minimum training-water NIR reference, and ordinary least-squares visible–NIR slopes.</li><li><b>Project-specific selection:</b> training water is automatically chosen from UDM2-clear water at least 300 m from land-like pixels, with a simple deep-water spectral screen.</li><li><b>Not used:</b> no percentile reference, no positive-only truncation, no correction-strength scaling, and no clipping of negative reflectance.</li></ul>
<div class="warn"><b>Input limitation:</b> these are Planet 6S Surface Reflectance products, not dedicated aquatic atmospheric-correction products. Corrected SR is the primary output. Rrs = SR/π and subsurface rrs are supplied only as explicitly labelled approximations.</div></section>
<section id="workflow"><h2>Complete workflow</h2><div class="flow"><div class="step"><b>1 · Input</b>Read eight-band Planet SR and UDM2.</div><div class="step"><b>2 · Mask</b>Exclude land, cloud, haze, shadow and invalid pixels.</div><div class="step"><b>3 · Train</b>Select documented offshore open-water pixels.</div><div class="step"><b>4 · Fit</b>Fit one ordinary least-squares slope per 444–707 nm band against 866 nm.</div><div class="step"><b>5 · Correct</b>Apply the standard Hedley equation without scaling.</div><div class="step"><b>6 · Evaluate</b>Report negatives, residual NIR relation and spatial preservation.</div></div></section>
<section id="summary"><h2>All-scene screening result</h2><div style="overflow:auto"><table><thead><tr><th>Date</th><th>Decision</th><th>Mean fit R²</th><th>566 nm removed</th><th>Any negative</th><th>Spatial preservation</th></tr></thead><tbody>{rows}</tbody></table></div><p><small>Decision thresholds are project QC rules, not part of the Hedley publication. They are recorded so the decision is reproducible.</small></p></section>
<section id="detail"><h2>Step-by-step result</h2><label for="scene">Planet scene</label> <select id="scene"></select><div id="decision" class="plain" style="margin-top:15px"></div><div id="metrics" class="metrics" style="margin-top:15px"></div>
<h3>Step 1 · Input and masks</h3><img id="masks" alt="Input and masks"><h3>Step 2 · Visible–NIR regression</h3><img id="fits" alt="Regression plots"><h3>Step 3 · Estimated term removed from every band</h3><img id="removed" alt="Removed term maps"><h3>Step 4 · Before and after</h3><img id="comparison" alt="Before and after RGB"><h3>Step 5 · Median spectrum</h3><img id="spectra" alt="Before and after spectra"><h3>Step 6 · Negative-pixel QC</h3><img id="negative" alt="Negative pixel map"><div style="overflow:auto"><table><thead><tr><th>Band</th><th>Slope</th><th>R²</th><th>Before median</th><th>After median</th><th>Removed median</th><th>Negative</th></tr></thead><tbody id="bands"></tbody></table></div><h3>Downloads</h3><p id="downloads"></p></section>
<section id="files"><h2>Folder organization</h2><ul><li><code>00_READ_ME_FIRST</code>: method statement, input audit and master QC.</li><li><code>01_INPUT</code>: the extracted Planet delivery used by the workflow.</li><li><code>02_RUN_WORKFLOW</code>: one-command runner and requirements.</li><li><code>03_EXAMPLE_OUTPUT</code>: a compact copy of the strongest accepted/cautious example.</li><li><code>04_GENERATED_OUTPUT</code>: every scene, intermediate raster, final raster and QC figure.</li><li><code>05_FULL_SOURCE_CODE</code>: production code plus validation code.</li></ul><p><a class="download" href="02_RUN_WORKFLOW/run_command.txt">Run command</a> <a class="download" href="05_FULL_SOURCE_CODE/production_code/process_planetscope_standard_hedley.py">Full processing code</a> <a class="download" href="00_READ_ME_FIRST/master_scene_qc.csv">Master scene QC</a> <a class="download" href="00_READ_ME_FIRST/master_band_qc.csv">Master band QC</a></p></section>
<script>const scenes={payload};const q=s=>document.querySelector(s),sel=q('#scene');sel.innerHTML=scenes.map((s,i)=>`<option value="${{i}}">${{s.date}} · ${{s.decision}}</option>`).join('');const f=(v,n=3)=>Number(v).toFixed(n);function render(){{const s=scenes[+sel.value||0];q('#decision').innerHTML=`<b>${{s.decision}}</b><br>${{s.decision_explanation}}`;q('#metrics').innerHTML=`<div class="metric"><small>Water pixels</small><b>${{s.water_pixels.toLocaleString()}}</b></div><div class="metric"><small>Training pixels</small><b>${{s.training_pixels.toLocaleString()}}</b></div><div class="metric"><small>Minimum NIR</small><b>${{f(s.nir_minimum,5)}}</b></div><div class="metric"><small>Any negative</small><b>${{f(s.negative_any_percent)}}%</b></div><div class="metric"><small>Spatial preservation</small><b>${{f(s.spatial_preservation_r)}}</b></div>`;for(const id of ['masks','fits','removed','comparison','spectra','negative'])q('#'+id).src=s.figures[id];q('#bands').innerHTML=s.bands.map(b=>`<tr><td>${{b.wave}} nm</td><td>${{f(b.slope,5)}}</td><td>${{f(b.r2)}}</td><td>${{f(b.median_before,5)}}</td><td>${{f(b.median_after,5)}}</td><td>${{f(b.median_removed,5)}}</td><td>${{f(b.negative_percent)}}%</td></tr>`).join('');q('#downloads').innerHTML=Object.entries(s.downloads).map(([k,v])=>`<a class="download" href="${{v}}">${{k}}</a>`).join(' ')}}sel.onchange=render;render();</script></main></body></html>'''
    output = root / "Tidung_Planet_Dove_Standard_Hedley_Results.html"
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--distance-m", default=300.0, type=float)
    args = parser.parse_args()

    root = args.output_dir
    readme_dir = root / "00_READ_ME_FIRST"
    input_dir = root / "01_INPUT" / "Planet_delivery"
    run_dir = root / "02_RUN_WORKFLOW"
    example_dir = root / "03_EXAMPLE_OUTPUT"
    generated = root / "04_GENERATED_OUTPUT"
    source_dir = root / "05_FULL_SOURCE_CODE"
    for directory in [readme_dir, input_dir, run_dir, example_dir, generated, source_dir / "production_code", source_dir / "validation_reference_code"]:
        directory.mkdir(parents=True, exist_ok=True)

    if not list(input_dir.rglob("*_AnalyticMS_SR_8b_clip.tif")):
        print("Extracting Planet delivery...", flush=True)
        safe_extract(args.input_zip, input_dir)
    planet = locate_planet_dir(input_dir)
    sr_files = sorted(planet.glob("*_AnalyticMS_SR_8b_clip.tif"))
    if not sr_files:
        raise RuntimeError("No Planet SR files found")

    inventory = []
    for path in sorted(planet.iterdir()):
        if path.is_file():
            inventory.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    with (readme_dir / "source_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "bytes", "sha256"])
        writer.writeheader(); writer.writerows(inventory)

    stable_land = build_stable_land_mask(planet)
    first = gdal.Open(str(sr_files[0]))
    write_mask(readme_dir / "stable_multi_date_land_mask.tif", stable_land, first, "Stable multi-date land mask")
    first = None

    scenes: list[dict] = []
    master_bands: list[dict] = []
    for tif in sr_files:
        scene_id = tif.name.split("_3B_")[0]
        compact = scene_id[:8]
        date = f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"
        print(f"Processing {date}", flush=True)
        scene_root = generated / date
        masks_dir = scene_root / "01_Masks"
        fit_dir = scene_root / "02_Hedley_Fit"
        removed_dir = scene_root / "03_Removed_Term"
        corrected_dir = scene_root / "04_Corrected_SR"
        rrs_dir = scene_root / "05_Rrs_Approx"
        sub_dir = scene_root / "06_rrs_Approx"
        qc_dir = scene_root / "07_QC"
        provenance_dir = scene_root / "08_Provenance"
        for directory in [masks_dir, fit_dir, removed_dir, corrected_dir, rrs_dir, sub_dir, qc_dir, provenance_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        source = gdal.Open(str(tif))
        raw = source.ReadAsArray().astype(np.float32)
        original = {wave: np.where(raw[index] == 0, np.nan, raw[index] / 10000.0) for index, wave in enumerate(WAVES)}
        udm_path = planet / f"{scene_id}_3B_udm2_clip.tif"
        udm_ds = gdal.Open(str(udm_path)); udm = udm_ds.ReadAsArray(); udm_ds = None
        clear = strict_udm2_clear(udm)
        finite = clear.copy()
        for wave in WAVES:
            finite &= np.isfinite(original[wave])
        water = corrected_water_mask(original, clear, stable_land)

        ndvi = (original[866] - original[666]) / np.maximum(original[866] + original[666], 1e-6)
        land_like = finite & (ndvi > 0.12) & (original[866] > 0.08)
        pixel_size = abs(source.GetGeoTransform()[1])
        distance = distance_transform_edt(finite & ~land_like) * pixel_size
        open_water = water & (distance >= args.distance_m)
        deep_screen = (original[492] >= 0.90 * original[566]) & (original[666] <= 0.92 * original[566])
        training = open_water & deep_screen
        if training.sum() < 5000:
            training = open_water
        if training.sum() < 1000:
            raise RuntimeError(f"Insufficient training pixels for {date}: {training.sum()}")

        nir_training = original[866][training]
        nir_minimum = float(np.min(nir_training))
        delta = original[866] - nir_minimum
        corrected = {wave: original[wave].copy() for wave in WAVES}
        removal: dict[int, np.ndarray] = {}
        metrics: list[dict] = []
        for wave in TARGETS:
            slope, intercept, r2, nfit = ordinary_least_squares(nir_training, original[wave][training])
            removal[wave] = slope * delta
            corrected[wave][water] = original[wave][water] - removal[wave][water]
            before = original[wave][water]
            after = corrected[wave][water]
            metric = {
                "date": date, "wave": wave, "slope": slope, "intercept": intercept, "r2": r2,
                "fit_pixels": nfit, "median_before": float(np.nanmedian(before)),
                "median_after": float(np.nanmedian(after)), "median_removed": float(np.nanmedian(removal[wave][water])),
                "negative_pixels": int(np.sum(after < 0)), "negative_percent": float(100 * np.sum(after < 0) / max(after.size, 1)),
                "residual_abs_corr_nir": abs(corr(corrected[wave][training], original[866][training])),
                "training_mask": training,
            }
            metrics.append(metric)
            master_bands.append({key: value for key, value in metric.items() if key != "training_mask"})
        corrected[866] = original[866].copy()
        negative_any = water & np.logical_or.reduce([corrected[wave] < 0 for wave in TARGETS])
        negative_any_percent = float(100 * negative_any.sum() / max(water.sum(), 1))

        before_intensity = np.mean(np.column_stack([original[492][water], original[566][water], original[666][water]]), axis=1)
        after_intensity = np.mean(np.column_stack([corrected[492][water], corrected[566][water], corrected[666][water]]), axis=1)
        spatial_r = corr(before_intensity, after_intensity)
        mean_r2 = float(np.nanmean([item["r2"] for item in metrics]))
        green = next(item for item in metrics if item["wave"] == 566)
        green_removed_percent = float(100 * green["median_removed"] / max(abs(green["median_before"]), 1e-9))
        decision, explanation = classify(mean_r2, green_removed_percent, negative_any_percent)

        write_mask(masks_dir / f"{date}_UDM2_strict_clear.tif", clear, source, "Strict UDM2 clear mask")
        write_mask(masks_dir / f"{date}_water_mask.tif", water, source, "Clear non-land water mask")
        write_mask(masks_dir / f"{date}_hedley_training_water.tif", training, source, "Open-water Hedley training mask")
        write_mask(qc_dir / f"{date}_any_negative_mask.tif", negative_any, source, "Negative in one or more corrected bands")

        serial_metrics = [{key: value for key, value in item.items() if key != "training_mask"} for item in metrics]
        with (fit_dir / f"{date}_hedley_band_fits.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(serial_metrics[0]))
            writer.writeheader(); writer.writerows(serial_metrics)
        parameters = {
            "date": date, "scene_id": scene_id, "method": "Hedley et al. (2005) standard equation",
            "formula": "rho_corrected(lambda) = rho(lambda) - b(lambda) * (rho866 - min_training_rho866)",
            "nir_reference_nm": 866, "nir_training_minimum": nir_minimum,
            "training_pixels": int(training.sum()), "training_distance_m": args.distance_m,
            "training_selection_note": "Project-specific automatic offshore/deep-water selection; not part of the published equation.",
            "custom_scaling": False, "percentile_reference": False, "positive_excess_truncation": False,
            "negative_clipping": False, "bands": serial_metrics,
        }
        (fit_dir / f"{date}_hedley_parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")

        stack = corrected_dir / f"{date}_Planet_Standard_Hedley_SR_8b.tif"
        removed_stack = removed_dir / f"{date}_Planet_Standard_Hedley_removed_term_7b.tif"
        write_stack(stack, corrected, source, water, "Standard_Hedley_SR")
        write_target_stack(removed_stack, removal, source, water, "Hedley_removed_term")
        for wave in TARGETS:
            write_float(corrected_dir / f"{date}_{wave}nm_Standard_Hedley_corrected_SR.tif", corrected[wave], source, water, f"Standard Hedley corrected SR {wave} nm", "unitless")
            rrs = corrected[wave] / math.pi
            write_float(rrs_dir / f"{date}_{wave}nm_Rrs_approx.tif", rrs, source, water, f"Approximate Rrs {wave} nm = corrected SR/pi", "sr-1")
            subsurface = rrs / (0.52 + 1.7 * rrs)
            write_float(sub_dir / f"{date}_{wave}nm_rrs_approx.tif", subsurface, source, water, f"Approximate subsurface rrs {wave} nm", "sr-1")

        masks_png = qc_dir / f"{date}_step1_input_masks.png"
        fits_png = qc_dir / f"{date}_step2_hedley_fits.png"
        removed_png = qc_dir / f"{date}_step3_removed_terms.png"
        comparison_png = qc_dir / f"{date}_step4_before_after.png"
        spectra_png = qc_dir / f"{date}_step5_spectra.png"
        negative_png = qc_dir / f"{date}_step6_negative_pixels.png"
        save_masks_figure(masks_png, date, original, clear, water, training)
        save_fit_figure(fits_png, date, nir_training, original, metrics)
        save_removed_figure(removed_png, date, removal, water)
        save_comparison(comparison_png, date, original, corrected, water)
        save_spectra(spectra_png, date, serial_metrics)
        save_negative_map(negative_png, date, negative_any, water)

        qc = {
            "date": date, "scene_id": scene_id, "decision": decision, "decision_explanation": explanation,
            "water_pixels": int(water.sum()), "training_pixels": int(training.sum()), "nir_minimum": nir_minimum,
            "mean_r2": mean_r2, "green_removed_percent": green_removed_percent,
            "negative_any_pixels": int(negative_any.sum()), "negative_any_percent": negative_any_percent,
            "spatial_preservation_r": spatial_r, "bands": serial_metrics,
            "negative_policy": "Preserved in corrected rasters; not clipped and not changed to NoData.",
        }
        qc_path = qc_dir / f"{date}_quality_control.json"
        qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
        shutil.copy2(fit_dir / f"{date}_hedley_parameters.json", provenance_dir / f"{date}_hedley_parameters.json")
        shutil.copy2(qc_path, provenance_dir / f"{date}_quality_control.json")

        figures = {"masks": relative(masks_png, root), "fits": relative(fits_png, root), "removed": relative(removed_png, root),
                   "comparison": relative(comparison_png, root), "spectra": relative(spectra_png, root), "negative": relative(negative_png, root)}
        downloads = {"Corrected SR stack": relative(stack, root), "Removed-term stack": relative(removed_stack, root),
                     "Fit parameters": relative(fit_dir / f"{date}_hedley_parameters.json", root), "QC JSON": relative(qc_path, root)}
        scene_payload = {**qc, "figures": figures, "downloads": downloads}
        scenes.append(scene_payload)
        source = None

    with (readme_dir / "master_scene_qc.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["date", "scene_id", "decision", "mean_r2", "green_removed_percent", "negative_any_pixels", "negative_any_percent", "spatial_preservation_r", "water_pixels", "training_pixels", "nir_minimum"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for scene in scenes:
            writer.writerow({field: scene[field] for field in fields})
    with (readme_dir / "master_band_qc.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(master_bands[0])
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(master_bands)

    manifest = {
        "title": "Tidung Planet SuperDove standard Hedley processing",
        "created_by": "process_planetscope_standard_hedley.py",
        "input": str(args.input_zip.resolve()), "input_product": "Planet ortho_analytic_8b_sr",
        "atmospheric_correction": "Planet Surface Reflectance (6S); ACOLITE was not rerun",
        "method": "Hedley et al. (2005) standard equation", "scene_count": len(scenes),
        "dates": [scene["date"] for scene in scenes],
        "formula": "rho_corrected(lambda) = rho(lambda) - b(lambda) * (rho866 - min_training_rho866)",
        "custom_safeguards": False, "negative_clipping": False,
        "corrected_bands_nm": TARGETS, "unchanged_reference_band_nm": 866,
        "mask": MASK_DESCRIPTION,
        "reference": "Hedley JD, Harborne AR, Mumby PJ (2005), DOI 10.1080/01431160500034086",
    }
    (readme_dir / "processing_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    accepted = [scene for scene in scenes if scene["decision"] in {"ACCEPT WITH QC", "CAUTIOUS"}]
    example = max(accepted or scenes, key=lambda scene: scene["mean_r2"])
    example_scene = generated / example["date"]
    for figure in (example_scene / "07_QC").glob("*.png"):
        shutil.copy2(figure, example_dir / figure.name)
    (example_dir / "README.md").write_text(
        f"# Example output\n\nSelected automatically from the processed scenes by the highest mean visible–NIR R² among accepted/cautious results.\n\nDate: **{example['date']}**  \nDecision: **{example['decision']}**\n",
        encoding="utf-8",
    )

    shutil.copy2(Path(__file__), source_dir / "production_code" / Path(__file__).name)
    mask_source = Path(__file__).with_name("planetscope_temporal_mask.py")
    shutil.copy2(mask_source, source_dir / "production_code" / mask_source.name)
    validate_source = Path(__file__).with_name("validate_planetscope_standard_hedley_package.py")
    if validate_source.exists():
        shutil.copy2(validate_source, source_dir / "validation_reference_code" / validate_source.name)
    shutil.copy2(Path(__file__), run_dir / Path(__file__).name)
    shutil.copy2(mask_source, run_dir / mask_source.name)
    if validate_source.exists():
        shutil.copy2(validate_source, run_dir / validate_source.name)
    (run_dir / "requirements.txt").write_text("numpy\nscipy\nmatplotlib\nGDAL\n", encoding="utf-8")
    command = f'python process_planetscope_standard_hedley.py --input-zip "{args.input_zip.resolve()}" --output-dir "{root.resolve()}" --distance-m {args.distance_m:g}'
    (run_dir / "run_command.txt").write_text(command + "\n", encoding="utf-8")
    (run_dir / "RUN_ME_WINDOWS.bat").write_text("@echo off\r\n" + command + "\r\npause\r\n", encoding="utf-8")

    method_text = """# Method and interpretation

## Input

The source TIFFs are Planet `ortho_analytic_8b_sr` Surface Reflectance products. Planet's atmospheric correction has already been applied. ACOLITE atmospheric correction is not applied again.

## Published Hedley calculation

For each visible band, an ordinary least-squares slope against the 866 nm NIR band is fitted in the training-water mask. The correction is:

`corrected = original - slope * (NIR - minimum training-water NIR)`

The minimum is the actual finite minimum, not a percentile. The NIR difference is not truncated. The fitted slope is not scaled. Negative corrected water values are preserved and reported.

## Project-specific choices

The multi-date land mask, UDM2 screening, 300 m offshore distance and spectral deep-water screen are documented implementation choices used to select training pixels. They are not claimed as part of the original publication.

## Interpretation

Corrected surface reflectance is the primary result. `Rrs_Approx` and `rrs_Approx` are convenience transformations and are not field-validated aquatic reflectance products. HydroLight may be used separately for evaluation, but it is not part of the subtraction.
"""
    (readme_dir / "METHOD.md").write_text(method_text, encoding="utf-8")
    analysis_lines = [
        "# Final analysis\n",
        "The standard Hedley equation was calculated for every delivered scene. Corrected rasters are retained for audit even when QC does not support using them as final products.\n",
        "## Scene decisions\n",
    ]
    for scene in scenes:
        analysis_lines.append(
            f"- **{scene['date']} — {scene['decision']}**: mean visible–NIR R² {scene['mean_r2']:.3f}; "
            f"566 nm median removal {scene['green_removed_percent']:.2f}%; "
            f"negative in any corrected band {scene['negative_any_percent']:.3f}%; "
            f"spatial preservation r {scene['spatial_preservation_r']:.3f}. {scene['decision_explanation']}"
        )
    candidates = [scene for scene in scenes if scene["decision"] == "ACCEPT WITH QC"]
    cautious = [scene for scene in scenes if scene["decision"] == "CAUTIOUS"]
    recommended = (candidates or cautious)
    analysis_lines.extend([
        "\n## Recommended use\n",
        (f"- The strongest currently usable candidate is **{max(recommended, key=lambda item: item['mean_r2'])['date']}**, but it remains conditional on independent validation."
         if recommended else "- No scene passes the current acceptance screen; retain original masked Planet SR."),
        "- For scenes labelled **NO CORRECTION NEEDED**, use the original masked Planet SR rather than forcing Hedley.",
        "- Do not use scenes labelled **REJECT** as corrected products.",
        "- Corrected SR is the primary numerical output. Approximate Rrs/rrs products are exploratory and not field-validated.",
        "- The decision thresholds are documented project QC rules; they are not part of Hedley et al. (2005).\n",
    ])
    (readme_dir / "FINAL_ANALYSIS.md").write_text("\n".join(analysis_lines), encoding="utf-8")
    (root / "README.md").write_text(
        "# Tidung Planet SuperDove standard Hedley package\n\nOpen `Tidung_Planet_Dove_Standard_Hedley_Results.html`. The package contains the input audit, complete runnable code, every intermediate product, corrected SR, approximate Rrs/rrs, and scene/band QC. Existing H:\\PlanetDove packages were not modified.\n",
        encoding="utf-8",
    )
    report = build_html(root, scenes)
    print(json.dumps({"output": str(root), "report": str(report), "scenes": len(scenes), "example": example["date"]}, indent=2))


if __name__ == "__main__":
    main()
