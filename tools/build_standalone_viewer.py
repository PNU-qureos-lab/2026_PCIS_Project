"""Build a single-file version of the combined satellite results viewer."""
from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "semi_analytical_map.html"
OUTPUT = ROOT / "semi_analytical_map_standalone.html"


def data_url(relative_path: str) -> str:
    path = ROOT / relative_path
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def inline_manifest(relative_path: str) -> str:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    image_paths = set(re.findall(r'(?:(?:semi|planet)_previews/[^"\']+\.png)', text))
    for image_path in image_paths:
        text = text.replace(image_path, data_url(image_path))
    return text


def main() -> None:
    html = SOURCE.read_text(encoding="utf-8-sig")
    css = (ROOT / "vendor" / "leaflet.css").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="vendor/leaflet.css">', f"<style>{css}</style>")

    scripts = {
        "vendor/leaflet.js": (ROOT / "vendor" / "leaflet.js").read_text(encoding="utf-8"),
        "semi_previews/manifest.js": inline_manifest("semi_previews/manifest.js"),
        "planet_previews/manifest.js": inline_manifest("planet_previews/manifest.js"),
        "viewer_data/sentinel_values.js": (ROOT / "viewer_data" / "sentinel_values.js").read_text(encoding="utf-8"),
        "viewer_data/planet_values.js": (ROOT / "viewer_data" / "planet_values.js").read_text(encoding="utf-8"),
        "tools/export_map.js": (ROOT / "tools" / "export_map.js").read_text(encoding="utf-8-sig"),
    }
    for source, script in scripts.items():
        html = html.replace(f'<script src="{source}"></script>', f"<script>{script}</script>")

    html = html.replace("Satellite semi-analytical results viewer</title>", "Satellite semi-analytical results viewer — standalone</title>")
    OUTPUT.write_text(html, encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT.name} ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
