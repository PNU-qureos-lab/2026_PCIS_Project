"""Shared entry point for the graphical interface and calculations.

Normal execution opens the Tkinter interface. ``--run`` starts the calculation
mode, and JSONL event files carry progress when no console is available.
"""
from pathlib import Path
import argparse
import json
import multiprocessing
import os
import sys
import traceback

# Pixel work is already distributed across processes; avoid nested BLAS threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True


def app_root():
    """Locate the application folder from either source or a frozen executable."""
    return (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent.parent
    )


def main():
    """Run the requested GUI, validation, analysis, or self-check mode."""
    parser = argparse.ArgumentParser(description="Satellite Rrs depth and bottom analysis")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", type=Path, help="run an analysis from a saved configuration JSON")
    mode.add_argument("--validate", type=Path, help="validate a saved configuration without processing")
    parser.add_argument("--events", type=Path, help="JSONL progress-event file read by the GUI")
    parser.add_argument("--cancel", type=Path, help="stop safely when this file is created")
    mode.add_argument("--self-check", type=Path, help="write the application self-check JSON here")
    args = parser.parse_args()

    def emit(row):
        """Emit the same progress events for GUI and command-line execution."""
        # Keep command-line JSON portable on Windows systems whose console uses
        # a legacy code page such as CP949. Unicode remains preserved through
        # JSON escapes and is decoded normally by JSON readers.
        text = json.dumps(row, ensure_ascii=True, default=str)
        if args.events:
            args.events.parent.mkdir(parents=True, exist_ok=True)
            with args.events.open("a", encoding="utf-8") as f:
                f.write(text + "\n")
        elif sys.stdout is not None:
            print(text, flush=True)

    try:
        root = app_root()
        if args.self_check:
            import numpy, scipy, rasterio, PIL, tkinter
            from bathymetry.config import resolve_coefficients, scan_bottoms, SENSOR_WAVES

            # Initialize Tcl/Tk, the real interface, and packaged data without showing a window.
            from bathymetry.gui import BathymetryApp

            tk = tkinter.Tk()
            tk.withdraw()
            gui = BathymetryApp(tk, root)
            tk.update_idletasks()
            gui_tabs = len(gui.notebook.tabs())
            # Initialize the pixel selector and Pillow/Tk bridge to detect missing dependencies.
            from bathymetry.config import load_config
            from bathymetry.pixel_selector import PixelSelector

            example = load_config(root / "DATA/examples/Sentinel2/config.json")
            picker = PixelSelector(tk, example, lambda value: None, visible=False)
            picker.render()
            pixel_selector_ready = picker.photo is not None
            picker.close()
            tk.destroy()
            report = {
                "root": str(root),
                "frozen": bool(getattr(sys, "frozen", False)),
                "versions": {
                    "python": sys.version,
                    "numpy": numpy.__version__,
                    "scipy": scipy.__version__,
                    "rasterio": rasterio.__version__,
                    "pillow": PIL.__version__,
                },
                "sensors": {
                    s: resolve_coefficients(s, "case1", "fitted", root) for s in SENSOR_WAVES
                },
                "bottom_count": len(
                    scan_bottoms(root / "DATA/bottom_reflectances", SENSOR_WAVES["SuperDove"])
                ),
                "gui_tabs": gui_tabs,
                "pixel_selector_ready": pixel_selector_ready,
                "passed": True,
            }
            args.self_check.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        elif args.run or args.validate:
            from bathymetry.config import load_config
            from bathymetry.runner import run_analysis, validate_config

            config = load_config(args.run or args.validate)
            if args.validate:
                emit({"type": "validated", "summary": validate_config(config, root)})
            else:
                run_analysis(config, root, emit, args.cancel)
        else:
            from bathymetry.gui import launch

            launch(root)
        return 0
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        if args.self_check:
            args.self_check.write_text(
                json.dumps(
                    {"passed": False, "error": str(exc), "traceback": traceback.format_exc()},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        if not any((args.run, args.validate, args.self_check)):
            import tkinter.messagebox

            tkinter.messagebox.showerror("Application error", str(exc))
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
