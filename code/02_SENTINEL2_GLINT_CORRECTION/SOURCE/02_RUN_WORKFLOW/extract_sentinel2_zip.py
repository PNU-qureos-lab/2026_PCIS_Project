#!/usr/bin/env python
"""Extract and verify a packaged Sentinel-2 L2A ZIP archive.

This helper is the code-mode equivalent of the GUI ZIP reader. It keeps the
official SAFE contents unchanged, blocks unsafe ZIP paths, and confirms that
the extracted data contain L2A metadata and the required MSI bands.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


REQUIRED_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
        source.extractall(destination)


def verify_scene(destination: Path) -> dict:
    metadata = list(destination.rglob("MTD_MSIL2A.xml"))
    found = {}
    for band in REQUIRED_BANDS:
        candidates = list(destination.rglob(f"*_{band}_*.jp2"))
        found[band] = len(candidates)
    missing = [band for band, count in found.items() if not count]
    report = {
        "working_folder": str(destination.resolve()),
        "l2a_metadata_files": len(metadata),
        "band_file_counts": found,
        "missing_required_bands": missing,
        "ready": bool(metadata) and not missing,
    }
    if not report["ready"]:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and verify a Sentinel-2 L2A ZIP")
    parser.add_argument("archive", type=Path, help="Sentinel-2 L2A ZIP archive")
    parser.add_argument("output", type=Path, help="Short working directory for extracted SAFE data")
    parser.add_argument("--replace", action="store_true", help="Replace an existing output folder")
    args = parser.parse_args()
    if args.output.exists() and args.replace:
        shutil.rmtree(args.output)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty: {args.output}. Use --replace to rebuild it.")
    safe_extract(args.archive, args.output)
    report = verify_scene(args.output)
    report_path = args.output / "sentinel2_zip_verification.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
