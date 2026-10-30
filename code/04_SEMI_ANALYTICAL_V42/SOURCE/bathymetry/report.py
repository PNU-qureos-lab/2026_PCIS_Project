"""계산 결과 GeoTIFF에서 PNG 미리보기와 HTML 결과 요약을 만든다.

미리보기는 최근접 방식으로 축소한다. 표시 색상이나 색상 범위는 원본 계산값과
CSV·GeoTIFF를 변경하지 않는다. 미리보기 생성에는 동봉된 Pillow를 사용하며
외부 지도 서비스에 접속하지 않는다.
"""
from pathlib import Path
import html
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    """설치된 한글 글꼴을 우선 사용하고 없으면 기본 글꼴을 사용한다."""
    for name in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def _gradient(t):
    """수심·오차·혼합계수용 연속 색상 범례를 밝은 청색에서 짙은 청색으로 만든다."""
    colors = np.array([[224, 248, 248], [71, 175, 191], [22, 91, 146], [18, 42, 94]], float)
    q = np.clip(t, 0, 1) * 3
    low = np.minimum(q.astype(int), 2)
    f = (q - low)[..., None]
    return (colors[low] * (1 - f) + colors[low + 1] * f).astype("uint8")


def _preview(path, target, title, kind="continuous", labels=None):
    """원본 값의 유효성 마스크를 유지하여 범례를 포함한 표시용 PNG를 저장한다."""
    with rasterio.open(path) as ds:
        scale = min(1.0, 1000 / ds.width, 700 / ds.height)
        h, w = max(1, round(ds.height * scale)), max(1, round(ds.width * scale))
        data = ds.read(1, out_shape=(h, w), masked=True, resampling=Resampling.nearest)
        crs = str(ds.crs)
    values = (
        data.filled(np.nan).astype(float)
        if np.issubdtype(data.dtype, np.floating)
        else data.astype(float).filled(np.nan)
    )
    valid = np.isfinite(values)
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    legend = []
    if kind == "category":
        palette = [(39, 135, 153), (214, 161, 68), (115, 93, 156), (194, 74, 70)]
        for i, (value, label) in enumerate(labels.items()):
            color = palette[i % len(palette)]
            rgb[valid & (values == value)] = color
            legend.append((color, label))
        scale_note = ""
    elif kind == "quality":
        rgb[valid & (values == 0)] = (53, 145, 125)
        rgb[valid & (values != 0)] = (229, 165, 71)
        rgb[valid & ((np.nan_to_num(values).astype(int) & 16) != 0)] = (184, 64, 64)
        legend = [
            ((53, 145, 125), "No additional quality flag"),
            ((229, 165, 71), "Review required"),
            ((184, 64, 64), "Calculation failed"),
        ]
        scale_note = "See analysis_record.json for quality-flag definitions"
    else:
        if not valid.any():
            return None
        lo, hi = (
            (0.0, 1.0)
            if kind == "fraction"
            else (float(values[valid].min()), float(values[valid].max()))
        )
        denom = hi - lo if hi > lo else 1.0
        t = np.nan_to_num((values - lo) / denom, nan=0, posinf=1, neginf=0)
        rgb[valid] = _gradient(t)[valid]
        if kind == "fraction":
            rgb[valid & (values < 0)] = (226, 137, 41)
            rgb[valid & (values > 1)] = (185, 60, 75)
            legend = [((226, 137, 41), "Below 0"), ((185, 60, 75), "Above 1")]
        unit = (
            " m" if path.stem == "depth_m" else " %" if path.stem == "relative_RMS_percent" else ""
        )
        scale_note = f"{lo:.6g}{unit} → {hi:.6g}{unit}"
    width = max(720, w + 48)
    footer = 110 + len(legend) * 28
    canvas = Image.new("RGB", (width, h + 80 + footer), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), title, fill=(30, 45, 55), font=_font(22))
    draw.text((24, 48), f"{crs} · PNG preview · white = excluded/not processed", fill=(75, 85, 95), font=_font(14))
    canvas.paste(Image.fromarray(rgb), ((width - w) // 2, 80))
    y = h + 94
    if kind not in ("category", "quality"):
        bar = _gradient(np.linspace(0, 1, width - 48)[None, :])
        canvas.paste(Image.fromarray(np.repeat(bar, 16, axis=0)), (24, y))
        y += 22
    if scale_note:
        draw.text((24, y), scale_note, fill=(30, 45, 55), font=_font(16))
        y += 28
    for color, label in legend:
        draw.rectangle((24, y + 3, 40, y + 19), fill=color)
        draw.text((50, y), label, fill=(30, 45, 55), font=_font(15))
        y += 28
    canvas.save(target)
    return target.name


def create_report(out, summary):
    """실제 설정·처리 상태·품질 표시를 지도와 연결한 결과 요약을 저장한다."""
    out = Path(out)
    maps = out / "MAPS"
    preview = out / "PREVIEWS"
    images = []
    titles = {
        "depth_m": "Estimated depth (m)",
        "C4": "Mean relative squared error C4",
        "relative_RMS_percent": "Rrs relative RMS error (%)",
    }
    by_id = {b["id"]: b["title"] for b in summary["bottoms"]}
    for path in sorted(maps.glob("*.tif")):
        name = path.stem
        kind = "continuous"
        labels = None
        if name.startswith("fraction_"):
            # 저장 파일명과 표시 제목은 분석 기록에 보존한 해저면 ID로 연결한다.
            from .runner import _slug

            match = next((k for k in by_id if "fraction_" + _slug(k) == name), name)
            title = "Bottom fraction: " + by_id.get(match, match)
            kind = "fraction"
        elif name == "pair_code":
            title = "Selected bottom-type combination"
            kind = "category"
            labels = {
                int(k): " + ".join(by_id[x] for x in pair) for k, pair in summary["pairs"].items()
            }
        elif name == "status":
            title = "Pixel-processing status"
            kind = "category"
            labels = {1: "Result available", 2: "Calculation failed"}
        elif name == "quality_flags":
            title = "Result quality flags"
            kind = "quality"
        else:
            title = titles.get(name, name)
        target = preview / (name + ".png")
        if _preview(path, target, title, kind, labels):
            images.append((title, target.name))
    esc = lambda x: html.escape(str(x))
    status_labels = {
        "running": "Running",
        "completed": "Completed",
        "completed_with_failures": "Completed with pixel failures",
        "cancelled": "Stopped by user",
        "failed": "Failed",
    }
    rows = [
        ("Status", status_labels.get(summary["status"], summary["status"])),
        ("Processed pixels", summary["processed_pixels"]),
        ("Failed calculations", summary["failed_pixels"]),
        ("Pixels with fractions outside range", summary["fraction_outside_pixels"]),
        ("Pixels reaching evaluation limit", summary["evaluation_limit_pixels"]),
        ("Sensor", summary["validation"]["위성"]),
        ("Water type", summary["validation"]["해역"]),
        ("Coefficients", summary["coefficients"]["source"]),
    ]
    if summary.get("deep_reference"):
        ref = summary["deep_reference"]
        rows.extend(
            [
                ("Deep-water statistic", "Mean" if ref["statistic"] == "mean" else "Median"),
                (
                    "Deep-water selected / used / excluded pixels",
                    f"{ref['selected_count']} / {ref['valid_count']} / {ref['excluded_count']}",
                ),
            ]
        )
    table = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows)
    imgs = "".join(
        f'<section><h2>{esc(t)}</h2><img src="PREVIEWS/{esc(n)}" alt="{esc(t)}"></section>'
        for t, n in images
    )
    notes = "".join("<li>" + esc(x) + "</li>" for x in summary["validation"]["주의"])
    document = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Analysis results</title>
<style>body{font-family:Segoe UI,Malgun Gothic,sans-serif;margin:28px auto;max-width:1120px;padding:0 20px;color:#23333c;background:#fff}h1,h2{font-weight:600}table{border-collapse:collapse;width:100%;max-width:850px}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}th{width:35%}img{max-width:100%;height:auto}section{margin:35px 0}li{margin:8px 0}.note{background:#f0f5f6;padding:16px;border-radius:8px}</style>
<h1>Depth and Bottom Analysis Results</h1><table>"""
        + table
        + """</table><div class="note"><ul>"""
        + notes
        + """</ul>Scientific GeoTIFFs are in MAPS, pixel values are in pixel_results.csv, and settings plus provenance are in analysis_record.json. Nearby-solution ranges are not statistical confidence intervals.</div>"""
        + imgs
        + """</html>"""
    )
    (out / "result_summary.html").write_text(document, encoding="utf-8")
