"""Bathymetry estimation 프로그램의 Windows 사용자 화면.

이 모듈은 화면과 실행 관리를 담당합니다. 과학 계산은 ``runner`` 모듈이,
해저면 자료와 내장 g0·g1의 해석은 ``config`` 모듈이 담당합니다. 사용자가
Selected한 Configuration은 JSON으로 고정한 다음 별도 계산 프로세스에 전달합니다.
따라서 계산 중에도 창이 응답하고, 계산에 사용한 Configuration을 재현할 수 있습니다.

실행 파일은 콘솔이 없는 Windows 프로그램으로 배포할 수 있습니다.
표준출력 파이프 대신 UTF-8 JSONL 이벤트 파일로 진행 상황을 받는 이유도
이 때문입니다. GUI 스레드에서 무거운 입력 검증이나 영상 역산을 하지 않습니다.
"""

from __future__ import annotations

import datetime as _datetime
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .config import SENSOR_WAVES, SUPERDOVE_ACTUAL_WAVES, resolve_coefficients, scan_bottoms
from .runner import validate_config


SENSOR_LABELS = {"Sentinel-2": "Sentinel2", "SuperDove": "SuperDove"}
WATER_LABELS = {"Case 1 water": "case1", "Case 2 water": "case2", "Mixed water": "mixed"}
WATER_NAMES = {value: key for key, value in WATER_LABELS.items()}
FIXED_DISPLAY = {
    "case1": "Clear-water preset: g₀ = 0.0949, g₁ = 0.0794 · Gordon et al. (1988)",
    "case2": "Turbid-water preset: g₀ = 0.0840, g₁ = 0.1700 · Lee et al. (1999)",
    "mixed": "Intermediate preset: g₀ = 0.0895, g₁ = 0.1247 · Lee et al. (2002)",
}
TIFF_TYPES = [("GeoTIFF image", "*.tif *.tiff"), ("All files", "*.*")]


def _wave_label(value: float) -> str:
    """443.0 같은 내부 값을 화면에서는 443 nm로 표시합니다."""
    return f"{float(value):g}"


def _display_number(value: Any) -> str:
    try:
        return f"{float(value):.8g}"
    except (TypeError, ValueError):
        return str(value)


class _ScrollFrame(ttk.Frame):
    """화면이 작은 노트북에서도 모든 입력란을 볼 수 있는 스크롤 영역.

    MouseWheel을 bind_all로 등록하면 다른 탭이나 목록의 스크롤을 가로채기
    쉽습니다. 이 창에 전달되는 이벤트의 포인터 위치를 확인해서 해당 영역
    위에서만 스크롤하며, 해저면 목록과 같은 Treeview는 자체 동작을 유지합니다.
    """

    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas, padding=14)
        self.window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.body.bind("<Configure>", self._body_changed)
        self.canvas.bind("<Configure>", self._canvas_changed)
        self.winfo_toplevel().bind("<MouseWheel>", self._wheel, add="+")

    def _body_changed(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _canvas_changed(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window, width=event.width)

    def _wheel(self, event: tk.Event) -> None:
        target = self.winfo_containing(event.x_root, event.y_root)
        if target is None or not self.winfo_ismapped():
            return
        if isinstance(target, (ttk.Treeview, tk.Text, ttk.Combobox)):
            return
        name = str(target)
        if name == str(self.canvas) or name.startswith(str(self.body) + "."):
            if self.canvas.bbox("all") and self.body.winfo_height() > self.canvas.winfo_height():
                self.canvas.yview_scroll(-int(event.delta / 120), "units")


class BathymetryApp:
    """Tk 객체는 메인 스레드에서만 조작하는 화면 컨트롤러입니다."""

    def __init__(self, root: tk.Tk, app_root: Path):
        self.root = root
        self.app_root = Path(app_root).resolve()
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.process: subprocess.Popen | None = None
        self.runtime_dir: tempfile.TemporaryDirectory | None = None
        self.event_path: Path | None = None
        self.cancel_path: Path | None = None
        self.event_offset = 0
        self.terminal_event: dict[str, Any] | None = None
        self.output_dir: Path | None = None
        self.validating = False
        self.loading = False
        self.close_when_finished = False
        self.bottoms: list[dict[str, Any]] = []
        self.selected_bottoms: set[str] = set()
        self.pending_bottom_paths: list[str] | None = None
        self.bottom_scan_generation = 0
        self.bottom_scan_pending = False
        self.rrs_vars: dict[str, tk.StringVar] = {}
        self.deep_vars: dict[str, tk.StringVar] = {}
        self.last_sensor: str | None = None
        self.deep_selection: dict | None = None
        self.deep_selection_signature = None

        self.sensor = tk.StringVar(value="Sentinel-2")
        self.water = tk.StringVar(value="Case 1 water")
        self.coefficient_mode = tk.StringVar(value="fitted")
        self.rrs_mode = tk.StringVar(value="separate")
        self.stack_file = tk.StringVar()
        self.deep_mode = tk.StringVar(value="pixels")
        self.deep_pixel_note = tk.StringVar(value="Select 10 deep-water reference pixels from the image.")
        self.deep_csv = tk.StringVar()
        self.solar_zenith = tk.StringVar()
        self.mask = tk.StringVar()
        self.output_root = tk.StringVar(
            value=str(self.app_root.parents[1] / "05_RESULTS" / "02_SEMI_ANALYTICAL")
        )
        self.run_name = tk.StringVar()
        self.workers = tk.StringVar(value=str(max(1, min(4, (os.cpu_count() or 2) - 1))))
        self.bottom_folder = tk.StringVar(value=str(self.app_root / "DATA" / "bottom_reflectances"))
        self.bottom_count = tk.IntVar(value=3)
        self.selection_status = tk.StringVar(value="Reading bottom-reflectance data.")
        self.coefficient_note = tk.StringVar()
        self.coefficient_source = tk.StringVar()
        self.overview = tk.StringVar()
        self.status = tk.StringVar(value="Select the settings and specify the input data.")
        self.progress_text = tk.StringVar(value="Ready")
        self.input_hint = tk.StringVar()

        root.title("Semi-Analytical Processing (Version 42)")
        width = min(1120, max(920, root.winfo_screenwidth() - 80))
        height = min(900, max(680, root.winfo_screenheight() - 100))
        root.geometry(f"{width}x{height}")
        root.minsize(880, 620)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._style()
        self._build()
        self.sensor.trace_add("write", self._selection_changed)
        self.water.trace_add("write", self._selection_changed)
        self.coefficient_mode.trace_add("write", self._selection_changed)
        self.rrs_mode.trace_add("write", lambda *_: self._show_rrs_mode())
        self.deep_mode.trace_add("write", lambda *_: self._show_deep_mode())
        self.bottom_count.trace_add("write", lambda *_: self._refresh_bottom_selection())
        self._selection_changed()
        self._scan_bottoms()
        self.root.after(150, self._poll)

    def _style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 5))
        style.configure("TCheckbutton", font=("Segoe UI", 10))
        style.configure("TRadiobutton", font=("Segoe UI", 10))
        style.configure("TNotebook.Tab", font=("Segoe UI", 10), padding=(14, 8))
        style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 11, "bold"))
        style.configure("Hint.TLabel", foreground="#596575", font=("Segoe UI", 9))
        style.configure("Strong.TLabel", foreground="#154d79", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        self.root.option_add("*Font", ("Segoe UI", 10))

    def _build(self) -> None:
        header = ttk.Frame(self.root, padding=(20, 14, 20, 8))
        header.pack(fill="x")
        ttk.Label(header, text="Semi-Analytical Processing (Version 42)", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Satellite remote sensing · reflectance-model analysis", style="Hint.TLabel").pack(
            side="right", pady=(9, 0)
        )
        ttk.Label(
            self.root, textvariable=self.overview, padding=(20, 0, 20, 10), style="Strong.TLabel"
        ).pack(fill="x")

        toolbar = ttk.Frame(self.root, padding=(20, 0, 20, 8))
        toolbar.pack(fill="x")
        self.config_buttons: list[ttk.Button] = []
        for label, command in [
            ("Load configuration", self._load_dialog),
            ("Save configuration", self._save_dialog),
            ("Sentinel-2 example", lambda: self._load_example("Sentinel2")),
            ("SuperDove example", lambda: self._load_example("SuperDove")),
            ("User guide", self._open_manual),
        ]:
            button = ttk.Button(toolbar, text=label, command=command)
            button.pack(side="left", padx=(0, 6))
            self.config_buttons.append(button)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=18)
        self.tabs: list[ttk.Frame] = []
        for title in ["1  Analysis settings", "2  Image · deep water", "3  Bottom selection", "4  Run · results"]:
            pane = _ScrollFrame(self.notebook)
            self.notebook.add(pane, text=title)
            self.tabs.append(pane.body)
        self._build_analysis(self.tabs[0])
        self._build_inputs(self.tabs[1])
        self._build_bottoms(self.tabs[2])
        self._build_results(self.tabs[3])

        footer = ttk.Frame(self.root, padding=(20, 10, 20, 14))
        footer.pack(fill="x")
        buttons = ttk.Frame(footer)
        buttons.pack(side="right", padx=(16, 0))
        self.validate_button = ttk.Button(
            buttons, text="Validate inputs", command=lambda: self._validate(False)
        )
        self.validate_button.pack(side="left", padx=3)
        self.run_button = ttk.Button(buttons, text="Start analysis", command=lambda: self._validate(True))
        self.run_button.pack(side="left", padx=3)
        self.cancel_button = ttk.Button(buttons, text="Stop", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=3)
        progress_frame = ttk.Frame(footer)
        progress_frame.pack(side="left", fill="both", expand=True)
        ttk.Label(progress_frame, textvariable=self.status, wraplength=650).pack(anchor="w")
        self.progress = ttk.Progressbar(progress_frame, maximum=100, mode="determinate")
        self.progress.pack(fill="x", pady=(5, 2))
        ttk.Label(progress_frame, textvariable=self.progress_text, style="Hint.TLabel").pack(
            anchor="w"
        )

    @staticmethod
    def _section(parent: tk.Misc, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=12, style="Section.TLabelframe")
        frame.pack(fill="x", pady=(0, 12))
        return frame

    @staticmethod
    def _hint(parent: tk.Misc, text: str, **kwargs: Any) -> ttk.Label:
        label = ttk.Label(parent, text=text, style="Hint.TLabel", wraplength=880, justify="left")
        label.pack(anchor="w", fill="x", **kwargs)
        return label

    def _build_analysis(self, parent: ttk.Frame) -> None:
        section = self._section(parent, "Sensor and water type")
        row = ttk.Frame(section)
        row.pack(fill="x")
        ttk.Label(row, text="Sensor").grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Combobox(
            row, textvariable=self.sensor, values=list(SENSOR_LABELS), state="readonly", width=20
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(row, text="Water type").grid(row=0, column=2, sticky="w", padx=(32, 12))
        ttk.Combobox(
            row, textvariable=self.water, values=list(WATER_LABELS), state="readonly", width=20
        ).grid(row=0, column=3, sticky="w")
        self._hint(section, "The user selects the water type. The program does not automatically classify water quality pixel by pixel.", pady=(10, 0))

        section = self._section(parent, "g₀ · g₁ coefficient mode")
        self.fitted_radio = ttk.Radiobutton(
            section, text="Precomputed wavelength-specific coefficients (Case 1 only)", variable=self.coefficient_mode, value="fitted"
        )
        self.fitted_radio.pack(anchor="w", pady=3)
        ttk.Radiobutton(
            section, text="Wavelength-independent coefficients — water-type presets", variable=self.coefficient_mode, value="fixed"
        ).pack(anchor="w", pady=3)
        ttk.Label(
            section, textvariable=self.coefficient_note, wraplength=880, style="Strong.TLabel"
        ).pack(anchor="w", fill="x", pady=(9, 4))
        self._hint(section, "Wavelength-specific coefficients are loaded from the supplied sensor tables. HydroLight is not run and coefficients are not recalculated during processing.")
        self._hint(
            section,
            "Case 1/Case 2 and clear/turbid are not equivalent classifications. The preset association is this application's default selection rule.",
            pady=(4, 0),
        )

        section = self._section(parent, "Coefficients used for processing")
        self.coefficient_tree = ttk.Treeview(
            section, columns=("wave", "g0", "g1"), show="headings", height=6, selectmode="none"
        )
        for key, title, width in [("wave", "Wavelength (nm)", 150), ("g0", "g₀", 210), ("g1", "g₁", 210)]:
            self.coefficient_tree.heading(key, text=title)
            self.coefficient_tree.column(key, width=width, anchor="center")
        self.coefficient_tree.pack(fill="x")
        ttk.Label(
            section, textvariable=self.coefficient_source, wraplength=880, style="Hint.TLabel"
        ).pack(anchor="w", fill="x", pady=(8, 0))
        ttk.Button(parent, text="Next: image · deep water →", command=lambda: self.notebook.select(1)).pack(
            anchor="e"
        )

    def _path_row(
        self, parent: tk.Misc, label: str, variable: tk.StringVar, command: Any
    ) -> ttk.Frame:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=17).pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row, text="Browse…", command=command).pack(side="right")
        return row

    def _build_inputs(self, parent: ttk.Frame) -> None:
        section = self._section(parent, "Analysis image · above-water remote-sensing reflectance Rrs (sr⁻¹)")
        self._hint(
            section,
            "Use GeoTIFFs after atmospheric/glint correction and reflectance-to-Rrs conversion. Do not use raw DN, ordinary surface reflectance, or reflectance multiplied by π.",
        )
        ttk.Label(
            section, textvariable=self.input_hint, style="Strong.TLabel", wraplength=880
        ).pack(anchor="w", fill="x", pady=(7, 6))
        radio_row = ttk.Frame(section)
        radio_row.pack(fill="x")
        ttk.Radiobutton(
            radio_row, text="Separate single-band files", variable=self.rrs_mode, value="separate"
        ).pack(side="left", padx=(0, 20))
        ttk.Radiobutton(
            radio_row, text="One multiband file containing all wavelengths", variable=self.rrs_mode, value="stack"
        ).pack(side="left")
        self.separate_frame = ttk.Frame(section)
        self.separate_frame.pack(fill="x", pady=(7, 0))
        self.rrs_rows = ttk.Frame(self.separate_frame)
        self.rrs_rows.pack(fill="x")
        ttk.Button(
            self.separate_frame, text="Select multiple files · assign wavelengths automatically", command=self._pick_rrs_files
        ).pack(anchor="w", pady=(4, 0))
        self._hint(
            self.separate_frame,
            "Files are assigned automatically when wavelengths (nm) appear in their names. Verify each assignment and replace individual files when necessary.",
            pady=(4, 0),
        )
        self.stack_frame = ttk.Frame(section)
        self._path_row(
            self.stack_frame,
            "Multiband image",
            self.stack_file,
            lambda: self._pick_file(self.stack_file, TIFF_TYPES),
        )
        self._hint(
            self.stack_frame, "Band order must match the wavelength order shown above. Verify the order manually when band metadata is unavailable."
        )

        section = self._section(parent, "Deep-water reference reflectance · scene-specific IOP estimation")
        self._hint(section, "Estimate inherent optical properties (IOPs) once from the scene's deep-water reference and apply them to all analysis pixels in that scene.")
        row = ttk.Frame(section)
        row.pack(fill="x", pady=(8, 4))
        ttk.Radiobutton(row, text="Select pixels from image", variable=self.deep_mode, value="pixels").pack(
            side="left", padx=(0, 20)
        )
        ttk.Radiobutton(row, text="Deep-water reference CSV", variable=self.deep_mode, value="csv").pack(
            side="left", padx=(0, 20)
        )
        ttk.Radiobutton(row, text="Enter representative Rrs manually", variable=self.deep_mode, value="manual").pack(
            side="left"
        )
        self.deep_pixel_frame = ttk.Frame(section)
        self.deep_pixel_frame.pack(fill="x")
        ttk.Button(
            self.deep_pixel_frame, text="Select deep-water pixels · choose mean/median", command=self._open_pixel_selector
        ).pack(anchor="w", pady=(3, 5))
        ttk.Label(
            self.deep_pixel_frame,
            textvariable=self.deep_pixel_note,
            wraplength=880,
            style="Strong.TLabel",
        ).pack(anchor="w", fill="x")
        self._hint(
            self.deep_pixel_frame,
            "Select 10 deep-water pixels individually. Median is the default; mean is optional. The reference CSV is saved automatically with the results.",
            pady=(5, 4),
        )
        self.deep_csv_frame = ttk.Frame(section)
        self.deep_csv_frame.pack(fill="x")
        self._path_row(
            self.deep_csv_frame,
            "Deep-water CSV",
            self.deep_csv,
            lambda: self._pick_file(self.deep_csv, [("CSV files", "*.csv"), ("All files", "*.*")]),
        )
        self._hint(
            self.deep_csv_frame,
            "Standard columns: one deep_Rrs wavelength field for each selected-sensor band plus solar_zenith_deg. See the supplied example CSV.",
        )
        self.deep_manual_frame = ttk.Frame(section)
        self.deep_rows = ttk.Frame(self.deep_manual_frame)
        self.deep_rows.pack(fill="x")
        self._hint(self.deep_manual_frame, "Enter above-water Rrs in sr⁻¹, for example 0.0042.")
        row = ttk.Frame(section)
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="Solar zenith angle (°)", width=17).pack(side="left")
        ttk.Entry(row, textvariable=self.solar_zenith, width=14).pack(side="left")
        ttk.Label(
            row, text="  When using CSV, leave blank to use its value · required for pixel selection/manual entry", style="Hint.TLabel"
        ).pack(side="left")

        section = self._section(parent, "Analysis area and output location")
        self._path_row(
            section, "Analysis mask (optional)", self.mask, lambda: self._pick_file(self.mask, TIFF_TYPES)
        )
        self._hint(
            section, "The mask must match the Rrs CRS and grid; only pixels with value 1 are analysed. If omitted, all valid image pixels are analysed."
        )
        self._path_row(
            section, "Results folder", self.output_root, lambda: self._pick_folder(self.output_root)
        )
        row = ttk.Frame(section)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Run name (optional)", width=17).pack(side="left")
        ttk.Entry(row, textvariable=self.run_name).pack(side="left", fill="x", expand=True)
        self._hint(section, "Each run creates a separate result folder. Use a short run name to distinguish outputs.")
        ttk.Button(parent, text="Next: bottom selection →", command=lambda: self.notebook.select(2)).pack(
            anchor="e"
        )

    def _build_bottoms(self, parent: ttk.Frame) -> None:
        section = self._section(parent, "Bottom-reflectance data")
        self._hint(
            section,
            "Input data are representative unmixed reflectance spectra for each bottom type. Premixed 25:75 or 10% interval files are not required.",
            pady=(0, 6),
        )
        self._hint(
            section,
            "The program mixes A and B as F×A + (1−F)×B. Initial searches use 25:75, 50:50 and 75:25, then refine F in steps of 0.01.",
            pady=(0, 8),
        )
        self._path_row(section, "Bottom-data folder", self.bottom_folder, self._pick_bottom_folder)
        ttk.Button(section, text="Rescan folder", command=self._scan_bottoms).pack(
            anchor="w", pady=(3, 7)
        )
        row = ttk.Frame(section)
        row.pack(fill="x")
        ttk.Radiobutton(
            row, text="Select 2 bottom types · estimate one mixture pair", variable=self.bottom_count, value=2
        ).pack(side="left", padx=(0, 20))
        ttk.Radiobutton(
            row, text="Select 3 bottom types · compare the best of three pairs", variable=self.bottom_count, value=3
        ).pack(side="left")
        self._hint(
            section,
            "With three types, A+B, A+C and B+C are compared and the best pair is selected per pixel. All three are not mixed simultaneously.",
            pady=(7, 4),
        )

        section = self._section(parent, "Select bottom types by title")
        self.bottom_tree = ttk.Treeview(
            section,
            columns=("selected", "title", "file", "state"),
            show="headings",
            height=10,
            selectmode="browse",
        )
        for key, title, width, stretch in [
            ("selected", "Selected", 55, False),
            ("title", "Bottom title", 360, True),
            ("file", "Source filename", 240, True),
            ("state", "Data status", 120, False),
        ]:
            self.bottom_tree.heading(key, text=title)
            self.bottom_tree.column(
                key,
                width=width,
                minwidth=40,
                stretch=stretch,
                anchor="center" if key in {"selected", "state"} else "w",
            )
        scrollbar = ttk.Scrollbar(section, orient="vertical", command=self.bottom_tree.yview)
        self.bottom_tree.configure(yscrollcommand=scrollbar.set)
        self.bottom_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.bottom_tree.bind("<ButtonRelease-1>", self._bottom_clicked)
        self.bottom_tree.bind("<space>", self._bottom_space)
        self.bottom_tree.tag_configure("invalid", foreground="#a63a3a")
        ttk.Label(
            parent, textvariable=self.selection_status, wraplength=900, style="Strong.TLabel"
        ).pack(fill="x", pady=(0, 10))

        section = self._section(parent, "Selected-bottom reflectance at sensor wavelengths")
        self.bottom_preview = ttk.Treeview(section, show="headings", height=4, selectmode="none")
        self.bottom_preview.pack(fill="x")
        self._hint(
            section,
            "Values are sampled from the source spectra at the sensor processing wavelengths. Bottom titles and selected files are recorded with the result configuration.",
            pady=(7, 0),
        )
        ttk.Button(parent, text="Next: run · results →", command=lambda: self.notebook.select(3)).pack(
            anchor="e"
        )

    def _build_results(self, parent: ttk.Frame) -> None:
        section = self._section(parent, "Run preparation")
        self._hint(
            section, "Use “Validate inputs” to check image bands, grids, masks, coefficients and bottom data. “Start analysis” performs the same validation first."
        )
        row = ttk.Frame(section)
        row.pack(fill="x", pady=(10, 4))
        ttk.Label(row, text="Parallel workers").pack(side="left", padx=(0, 12))
        ttk.Spinbox(
            row, from_=1, to=max(1, os.cpu_count() or 1), textvariable=self.workers, width=7
        ).pack(side="left")
        ttk.Label(row, text=f"  CPU {os.cpu_count() or 1} logical CPUs · default maximum 4", style="Hint.TLabel").pack(
            side="left"
        )
        self._hint(section, "More workers use more memory. Keep the default when other applications are running.")

        section = self._section(parent, "Input validation results")
        self.review_text = tk.Text(
            section,
            height=11,
            wrap="word",
            relief="flat",
            background="#f3f6f8",
            padx=10,
            pady=8,
            state="disabled",
        )
        self.review_text.pack(fill="x")
        self._set_text(
            self.review_text,
            "Specify the input data and bottom types, then click “Validate inputs”.\n"
            "Built-in coefficients are shown immediately in the “Analysis settings” tab.",
        )

        section = self._section(parent, "Progress log")
        self.log_text = tk.Text(
            section,
            height=10,
            wrap="word",
            relief="flat",
            background="#f3f6f8",
            padx=10,
            pady=8,
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True)
        self.open_result_button = ttk.Button(
            parent, text="Open completed results folder", command=self._open_result, state="disabled"
        )
        self.open_result_button.pack(anchor="e")
        self._hint(
            parent,
            "Stop lets the current operation finish safely before cancellation. Check the result folder for the stopped status and any files already saved.",
            pady=(10, 0),
        )

    def _sensor_key(self) -> str:
        return SENSOR_LABELS[self.sensor.get()]

    def _water_key(self) -> str:
        return WATER_LABELS[self.water.get()]

    def _waves(self) -> list[float]:
        return list(SENSOR_WAVES[self._sensor_key()])

    def _selection_changed(self, *_args: Any) -> None:
        if self.loading:
            return
        sensor = self._sensor_key()
        water = self._water_key()
        # Case 2/Mixed water에는 동봉된 파장별 계수가 없습니다. 앞 화면의 Selected을
        # 그대로 남겨 잘못 적용하는 일을 막기 위해 화면과 값 모두 갱신합니다.
        if water != "case1":
            self.fitted_radio.configure(state="disabled")
            if self.coefficient_mode.get() != "fixed":
                self.coefficient_mode.set("fixed")
                return
        else:
            self.fitted_radio.configure(state="normal")
        if self.coefficient_mode.get() == "fitted":
            note = "Apply the precomputed sensor-specific wavelength coefficients for Case 1 water."
        else:
            note = FIXED_DISPLAY[water]
            if water != "case1":
                note += "\nWavelength-specific coefficients are not available for this water type."
        self.coefficient_note.set(note)
        self.coefficient_tree.delete(*self.coefficient_tree.get_children())
        try:
            coefficients = resolve_coefficients(
                sensor, water, self.coefficient_mode.get(), self.app_root
            )
            for wave, g0, g1 in zip(coefficients["waves"], coefficients["g0"], coefficients["g1"]):
                self.coefficient_tree.insert(
                    "", "end", values=(_wave_label(wave), _display_number(g0), _display_number(g1))
                )
            self.coefficient_source.set("Source · " + str(coefficients.get("source", "precomputed coefficients")))
        except Exception as exc:
            self.coefficient_source.set(f"Check the coefficient data: {exc}")
        mode_label = "Precomputed wavelength-specific coefficients" if self.coefficient_mode.get() == "fitted" else "Water-type preset coefficients"
        self.overview.set(f"{self.sensor.get()}   →   {self.water.get()}   →   {mode_label}")
        hint = (
            "Representative wavelength order used for processing: "
            + " → ".join(_wave_label(w) for w in self._waves())
            + f" nm  ({len(self._waves())} bands)"
        )
        if sensor == "SuperDove":
            hint += "\nSuperDove actual-band mapping: 444 → 492 → 533 → 566 → 612 → 666 nm"
        self.input_hint.set(hint)
        if sensor != self.last_sensor:
            # 두 센서의 파장이 같더라도 파일의 센서가 같다는 뜻은 아닙니다.
            # Sensor을 바꾸면 영상/깊은 물 입력을 비워 다른 센서 자료의 재사용을
            # 방지합니다. Load configuration는 이 초기화 뒤 저장된 값을 채웁니다.
            changed_existing = self.last_sensor is not None
            self.last_sensor = sensor
            self._rebuild_wavelength_inputs()
            if changed_existing:
                self.deep_selection = None
                self.deep_selection_signature = None
                self.deep_pixel_note.set("The sensor changed. Select deep-water pixels again from the corresponding image.")
                self.stack_file.set("")
                self.deep_csv.set("")
                self.solar_zenith.set("")
                self.mask.set("")
                self.status.set("The sensor changed. Specify the image and deep-water data for that sensor.")
                self._scan_bottoms()
        self._show_rrs_mode()
        self._show_deep_mode()

    def _rebuild_wavelength_inputs(self) -> None:
        for holder in [self.rrs_rows, self.deep_rows]:
            for widget in holder.winfo_children():
                widget.destroy()
        self.rrs_vars = {}
        self.deep_vars = {}
        for wave in self._waves():
            key = _wave_label(wave)
            self.rrs_vars[key] = tk.StringVar()
            self.deep_vars[key] = tk.StringVar()
            variable = self.rrs_vars[key]
            self._path_row(
                self.rrs_rows,
                f"{key} nm",
                variable,
                lambda var=variable: self._pick_file(var, TIFF_TYPES),
            )
        for index, wave in enumerate(self._waves()):
            key = _wave_label(wave)
            ttk.Label(self.deep_rows, text=f"{key} nm").grid(
                row=0, column=index, sticky="w", padx=(0, 12)
            )
            ttk.Entry(self.deep_rows, textvariable=self.deep_vars[key], width=12).grid(
                row=1, column=index, sticky="w", padx=(0, 12), pady=(4, 7)
            )

    def _show_rrs_mode(self) -> None:
        self.separate_frame.pack_forget()
        self.stack_frame.pack_forget()
        (self.stack_frame if self.rrs_mode.get() == "stack" else self.separate_frame).pack(
            fill="x", pady=(7, 0)
        )

    def _show_deep_mode(self) -> None:
        self.deep_csv_frame.pack_forget()
        self.deep_manual_frame.pack_forget()
        self.deep_pixel_frame.pack_forget()
        # Solar zenith angle 행 앞에 삽입하여 입력 방식 변경 뒤에도 순서를 유지합니다.
        host = self.deep_csv_frame.master
        solar_row = host.winfo_children()[-1]
        frame = {
            "manual": self.deep_manual_frame,
            "csv": self.deep_csv_frame,
            "pixels": self.deep_pixel_frame,
        }[self.deep_mode.get()]
        frame.pack(fill="x", before=solar_row)

    def _image_config(self) -> dict:
        """심수역 Selected은 해저면·분석 마스크 입력 없이 영상만으로 열 수 있다."""
        files = (
            [self.stack_file.get().strip()]
            if self.rrs_mode.get() == "stack"
            else [self.rrs_vars[_wave_label(w)].get().strip() for w in self._waves()]
        )
        if any(not p for p in files):
            raise ValueError("First specify all Rrs image files used for the analysis.")
        return {
            "sensor": self._sensor_key(),
            "rrs_files": [str(Path(p).expanduser().resolve()) for p in files],
        }

    @staticmethod
    def _image_signature(config):
        return config["sensor"], tuple(config["rrs_files"])

    def _refresh_deep_pixel_note(self):
        selection = self.deep_selection
        if not selection:
            self.deep_pixel_note.set("Select 10 deep-water reference pixels from the image.")
            return
        preview = selection.get("preview", {})
        method = "Mean" if selection["statistic"] == "mean" else "Median"
        values = " · ".join(
            f"{w:g} nm: {_display_number(v)}"
            for w, v in zip(selection["waves_nm"], preview.get("rrs", []))
        )
        self.deep_pixel_note.set(
            f"Selected {len(selection['pixels']):,} · valid {preview.get('valid_count','—')} · excluded {preview.get('excluded_count','—')} · {method}\n{values}"
        )

    def _open_pixel_selector(self):
        if self.validating or self.process is not None:
            return
        try:
            from .pixel_selector import PixelSelector
            from .deepwater import source_identity

            image_config = self._image_config()
            signature = self._image_signature(image_config)
            initial = self.deep_selection if signature == self.deep_selection_signature else None
            if initial and initial.get("source_sha256") != source_identity(
                image_config["rrs_files"]
            ):
                initial = None

            def apply(selection):
                self.deep_selection = selection
                self.deep_selection_signature = signature
                self.deep_mode.set("pixels")
                self._refresh_deep_pixel_note()
                self.status.set("The deep-water reference was applied. Check the solar zenith angle and remaining inputs.")

            PixelSelector(self.root, image_config, apply, initial)
        except Exception as exc:
            messagebox.showerror("Deep-water pixel selection", str(exc), parent=self.root)

    def _pick_file(self, variable: tk.StringVar, types: list[tuple[str, str]]) -> None:
        path = filedialog.askopenfilename(parent=self.root, title="Select input file", filetypes=types)
        if path:
            variable.set(path)

    def _pick_folder(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory(
            parent=self.root, title="Select folder", initialdir=variable.get() or str(self.app_root)
        )
        if path:
            variable.set(path)

    def _pick_bottom_folder(self) -> None:
        before = self.bottom_folder.get()
        self._pick_folder(self.bottom_folder)
        if self.bottom_folder.get() != before:
            self.selected_bottoms.clear()
            self._scan_bottoms()

    def _pick_rrs_files(self) -> None:
        paths = list(
            filedialog.askopenfilenames(
                parent=self.root, title="Select wavelength-specific Rrs GeoTIFFs", filetypes=TIFF_TYPES
            )
        )
        if not paths:
            return
        waves = [_wave_label(w) for w in self._waves()]
        if len(paths) != len(waves):
            messagebox.showwarning(
                "Check file count",
                f"{self.sensor.get()} requires {len(waves)} are required.\nCurrently {len(paths)} files are selected.",
                parent=self.root,
            )
            return
        # Selected 대화상자의 반환 순서나 단순 문자 정렬은 밴드 순서를 보장하지
        # 않습니다. 파일명에서 파장을 유일하게 식별한 경우에만 자동 배치합니다.
        mapping: dict[str, str] = {}
        for path in paths:
            name = Path(path).stem
            found = []
            for wave in waves:
                aliases = {wave}
                if self._sensor_key() == "SuperDove":
                    aliases.add(str(SUPERDOVE_ACTUAL_WAVES[int(wave)]))
                if any(
                    re.search(rf"(?<!\d){re.escape(alias)}(?:\.0)?(?!\d)", name)
                    for alias in aliases
                ):
                    found.append(wave)
            if len(found) == 1 and found[0] not in mapping:
                mapping[found[0]] = path
        if len(mapping) == len(waves):
            for wave in waves:
                self.rrs_vars[wave].set(mapping[wave])
            self.status.set("Files were assigned from wavelengths in their names. Verify each wavelength and filename.")
        else:
            messagebox.showinfo(
                "Wavelength-specific files required",
                "The selected file names do not identify every wavelength unambiguously.\n\nAutomatic placement was cancelled to prevent an incorrect band order. Use ‘Browse…’ beside each wavelength to select the correct file.",
                parent=self.root,
            )

    def _scan_bottoms(self) -> None:
        self.bottom_scan_generation += 1
        self.bottom_scan_pending = True
        generation = self.bottom_scan_generation
        folder = Path(self.bottom_folder.get()).expanduser()
        waves = self._waves()
        self.selection_status.set("Reading bottom-reflectance data…")

        def task() -> None:
            try:
                result = scan_bottoms(folder, waves)
                self.messages.put(("bottoms", (generation, result, None)))
            except Exception as exc:
                self.messages.put(("bottoms", (generation, [], str(exc))))

        threading.Thread(target=task, name="bottom-preview", daemon=True).start()

    def _accept_bottoms(self, payload: tuple[Any, ...]) -> None:
        generation, records, error = payload
        if generation != self.bottom_scan_generation:
            return
        self.bottom_scan_pending = False
        self.bottoms = list(records)
        self.bottom_tree.delete(*self.bottom_tree.get_children())
        available = {
            str(Path(record["path"]).resolve())
            for record in self.bottoms
            if not record.get("error")
        }
        if self.pending_bottom_paths is not None:
            self.selected_bottoms = {
                str(Path(path).resolve()) for path in self.pending_bottom_paths
            }
            missing = self.selected_bottoms - available
            self.pending_bottom_paths = None
            if missing:
                self._log(
                    "The configured bottom file was not found in the current folder: "
                    + ", ".join(Path(path).name for path in sorted(missing))
                )
            self.selected_bottoms &= available
        else:
            self.selected_bottoms &= available
        for index, record in enumerate(self.bottoms):
            path = str(Path(record["path"]).resolve())
            record["path"] = path
            record["title"] = record.get("title") or Path(path).stem
            self.bottom_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    "✓" if path in self.selected_bottoms else "□",
                    record["title"],
                    Path(path).name,
                    "Unavailable" if record.get("error") else "Available",
                ),
                tags=("invalid",) if record.get("error") else (),
            )
        self._refresh_bottom_selection()
        if error:
            self.selection_status.set("Cannot read the bottom-data folder: " + error)
        elif not records:
            self.selection_status.set("No readable bottom-reflectance spectra were found. Check the folder.")

    def _bottom_clicked(self, event: tk.Event) -> None:
        item = self.bottom_tree.identify_row(event.y)
        if item:
            self._toggle_bottom(item)

    def _bottom_space(self, _event: tk.Event) -> str:
        item = self.bottom_tree.focus()
        if item:
            self._toggle_bottom(item)
        return "break"

    def _toggle_bottom(self, item: str) -> None:
        record = self.bottoms[int(item)]
        if record.get("error"):
            messagebox.showwarning("Check bottom data", str(record["error"]), parent=self.root)
            return
        path = record["path"]
        if path in self.selected_bottoms:
            self.selected_bottoms.remove(path)
        elif len(self.selected_bottoms) < self.bottom_count.get():
            self.selected_bottoms.add(path)
        else:
            self.selection_status.set(
                f"Already {self.bottom_count.get()} files are selected. To choose another bottom type, click a selected item again to clear it."
            )
            return
        self._refresh_bottom_selection()

    def _refresh_bottom_selection(self) -> None:
        needed = self.bottom_count.get()
        count = len(self.selected_bottoms)
        pair_text = "One mixture pair will be estimated." if needed == 2 else "Three pairs will be compared to select the best pair per pixel."
        self.selection_status.set(
            f"{needed} of {count} selected · {pair_text}" + (" Select the required number." if count != needed else "")
        )
        for index, record in enumerate(self.bottoms):
            item = str(index)
            if self.bottom_tree.exists(item):
                self.bottom_tree.set(
                    item, "selected", "✓" if record["path"] in self.selected_bottoms else "□"
                )
        waves = [_wave_label(w) for w in self._waves()]
        columns = ["title"] + [f"w{index}" for index in range(len(waves))]
        self.bottom_preview.delete(*self.bottom_preview.get_children())
        self.bottom_preview.configure(columns=columns)
        self.bottom_preview.heading("title", text="Bottom title")
        self.bottom_preview.column("title", width=250, minwidth=150)
        for index, wave in enumerate(waves):
            key = f"w{index}"
            self.bottom_preview.heading(key, text=f"{wave} nm")
            self.bottom_preview.column(key, width=95, minwidth=75, anchor="center")
        for record in self.bottoms:
            if record["path"] in self.selected_bottoms:
                values = record.get("reflectance", [])
                self.bottom_preview.insert(
                    "",
                    "end",
                    values=[record["title"]] + [_display_number(value) for value in values],
                )

    def _collect_config(self) -> dict[str, Any]:
        """화면 값을 실행 Configuration으로 변환하고 단순한 입력 누락을 확인합니다.

        파일 내용·GeoTIFF 격자 등 자세한 검증은 runner가 수행합니다.
        저장 Configuration도 같은 형식이므로 GUI와 명령줄이 같은 값을 사용합니다.
        """
        if self.bottom_scan_pending:
            raise ValueError("Bottom-reflectance data are still being read. Wait for the list to refresh before running.")
        waves = [_wave_label(w) for w in self._waves()]
        rrs = (
            [self.stack_file.get().strip()]
            if self.rrs_mode.get() == "stack"
            else [self.rrs_vars[wave].get().strip() for wave in waves]
        )
        if any(not path for path in rrs):
            raise ValueError("Specify all Rrs files used for the analysis.")
        if len(self.selected_bottoms) != self.bottom_count.get():
            raise ValueError(
                f"Select exactly {self.bottom_count.get()} files. Currently {len(self.selected_bottoms)} files are selected."
            )
        try:
            workers = int(self.workers.get())
        except ValueError as exc:
            raise ValueError("Parallel workers must be an integer of at least 1.") from exc
        if workers < 1 or workers > max(1, os.cpu_count() or 1):
            raise ValueError(f"Parallel workers must be between 1 and {os.cpu_count() or 1}.")
        output = self.output_root.get().strip()
        if not output:
            raise ValueError("Specify the results folder.")
        # 파일의 표시 순서를 보존하면 A+B, A+C 등의 순서도 다시 실행할 때
        # 동일합니다. set 자체를 저장하지 않고 목록 순서로 정렬하여 기록합니다.
        bottoms = [
            record["path"] for record in self.bottoms if record["path"] in self.selected_bottoms
        ]
        config: dict[str, Any] = {
            "sensor": self._sensor_key(),
            "water_type": self._water_key(),
            "coefficient_mode": self.coefficient_mode.get(),
            "rrs_files": [str(Path(path).expanduser().resolve()) for path in rrs],
            "mask": str(Path(self.mask.get().strip()).expanduser().resolve())
            if self.mask.get().strip()
            else None,
            "bottom_files": bottoms,
            "output_root": str(Path(output).expanduser().resolve()),
            "workers": workers,
        }
        if self.run_name.get().strip():
            config["run_name"] = self.run_name.get().strip()
        if self.deep_mode.get() == "pixels":
            if not self.deep_selection:
                raise ValueError("Select deep-water pixels from the image and apply the reference values.")
            if self.deep_selection_signature != self._image_signature(config):
                raise ValueError("The Rrs image or band order changed. Select the deep-water pixels again.")
            config["deep_selection"] = self.deep_selection
            if not self.solar_zenith.get().strip():
                raise ValueError("Enter the solar zenith angle when using deep-water pixel selection.")
        elif self.deep_mode.get() == "manual":
            try:
                config["deep_rrs"] = [float(self.deep_vars[wave].get()) for wave in waves]
            except ValueError as exc:
                raise ValueError("Enter a numeric representative deep-water Rrs for each wavelength.") from exc
            if not self.solar_zenith.get().strip():
                raise ValueError("Enter the solar zenith angle when entering deep-water Rrs manually.")
        else:
            path = self.deep_csv.get().strip()
            if not path:
                raise ValueError("Specify the deep-water reference CSV.")
            config["deep_csv"] = str(Path(path).expanduser().resolve())
        if self.solar_zenith.get().strip():
            try:
                config["solar_zenith"] = float(self.solar_zenith.get())
            except ValueError as exc:
                raise ValueError("Enter the solar zenith angle as a number.") from exc
        return config

    def _validate(self, launch_after: bool) -> None:
        if self.validating or self.process is not None:
            return
        try:
            config = self._collect_config()
        except Exception as exc:
            messagebox.showwarning("Validate inputs", str(exc), parent=self.root)
            return
        self.validating = True
        self._set_busy(True)
        self.notebook.select(3)
        self.status.set("Validating input data and analysis settings…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(14)
        self.progress_text.set("Validating inputs")
        self._log("Input validation started.")

        def task() -> None:
            try:
                summary = validate_config(config, self.app_root)
                self.messages.put(("validated", (config, summary, launch_after, None)))
            except Exception as exc:
                self.messages.put(("validated", (config, None, launch_after, str(exc))))

        threading.Thread(target=task, name="input-validation", daemon=True).start()

    def _validated(self, payload: tuple[Any, ...]) -> None:
        config, summary, launch_after, error = payload
        self.validating = False
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        if error:
            self._set_busy(False)
            self.status.set("Input validation failed. Correct the listed items.")
            self.progress_text.set("Input validation required")
            self._set_text(self.review_text, "Input validation failed\n\n" + error)
            self._log("Input validation failed: " + error)
            messagebox.showerror("Input validation failed", error, parent=self.root)
            if self.close_when_finished:
                self.root.destroy()
            return
        lines = [
            "Input validation complete",
            "",
            f"Sensor: {self.sensor.get()} / water type: {self.water.get()}",
            "Coefficients: " + ("Precomputed wavelength-specific coefficients" if config["coefficient_mode"] == "fitted" else "Water-type preset coefficients"),
            f"Rrs images: {len(config['rrs_files'])} files / {len(self._waves())} processing bands",
            "Bottom types: " + ", ".join(Path(path).name for path in config["bottom_files"]),
            f"Bottom combinations: {1 if len(config['bottom_files']) == 2 else 3} pair comparisons",
            f"Output location: {config['output_root']}",
            f"Parallel workers: {config['workers']}",
        ]
        # 검증기는 배포 버전에 따라 요약 항목을 추가할 수 있습니다. 타입에
        # 의존하지 않고 기본 Configuration 아래에 모든 검증 결과를 읽기 쉽게 표시합니다.
        if summary:
            labels = {
                "width": "Image width",
                "height": "Image height",
                "valid_pixels": "Analysis pixels",
                "total_pixels": "Total pixels",
                "pixel_count": "Analysis pixels",
                "crs": "CRS",
                "warnings": "Checks",
                "message": "Validation message",
                "shape": "Image shape",
                "count": "Target count",
                "solar_zenith": "Solar zenith angle",
            }
            lines += ["", "Data inspection results"]
            if isinstance(summary, dict):
                for key, value in summary.items():
                    if isinstance(value, (dict, list, tuple)):
                        value = json.dumps(value, ensure_ascii=False, default=str)
                    lines.append(f"{labels.get(key, key)}: {value}")
            else:
                lines.append(str(summary))
        self._set_text(self.review_text, "\n".join(lines))
        self._log("Input validation passed.")
        if self.close_when_finished:
            self.root.destroy()
            return
        if launch_after:
            self._start_process(config)
        else:
            self._set_busy(False)
            self.status.set("Input validation complete. Click ‘Start analysis’ to begin processing.")
            self.progress_text.set("Ready to run")

    def _start_process(self, config: dict[str, Any]) -> None:
        try:
            self.runtime_dir = tempfile.TemporaryDirectory(prefix="bathymetry_")
            runtime = Path(self.runtime_dir.name)
            config_path = runtime / "config.json"
            self.event_path = runtime / "events.jsonl"
            self.cancel_path = runtime / "cancel.request"
            config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            executable = (
                [sys.executable]
                if getattr(sys, "frozen", False)
                else [sys.executable, str(self.app_root / "SOURCE" / "main.py")]
            )
            command = executable + [
                "--run",
                str(config_path),
                "--events",
                str(self.event_path),
                "--cancel",
                str(self.cancel_path),
            ]
            options: dict[str, Any] = {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "cwd": str(self.app_root),
            }
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            self.event_offset = 0
            self._pending_event_bytes = b""
            self.terminal_event = None
            self.output_dir = None
            self.open_result_button.configure(state="disabled")
            self.process = subprocess.Popen(command, **options)
            self._set_busy(True)
            self.cancel_button.configure(state="normal")
            self.progress.configure(mode="determinate", value=0)
            self.progress_text.set("Preparing calculation")
            self.status.set("Analysis started.")
            self._log("A separate calculation process started. Final settings and outputs will be saved in the result folder.")
        except Exception as exc:
            self.process = None
            self._cleanup_runtime()
            self._set_busy(False)
            self.status.set("Could not start the analysis process.")
            self._log("Run failed: " + str(exc))
            messagebox.showerror("Run failed", str(exc), parent=self.root)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for button in [self.validate_button, self.run_button, *self.config_buttons]:
            button.configure(state=state)
        for index in range(3):
            self.notebook.tab(index, state="disabled" if busy else "normal")
        if not busy:
            self.cancel_button.configure(state="disabled")

    def _poll(self) -> None:
        """스레드 큐와 이벤트 파일을 짧게 읽고 즉시 Tk 이벤트 루프로 복귀합니다."""
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "bottoms":
                    self._accept_bottoms(payload)
                elif kind == "validated":
                    self._validated(payload)
        except queue.Empty:
            pass
        if self.process is not None:
            self._read_events()
            return_code = self.process.poll()
            if return_code is not None:
                self._read_events(final=True)
                self._finish_process(return_code)
        # 완료 콜백이 창 닫기 요청을 처리했을 수 있으므로 Tk 종료 뒤에는
        # 다음 폴링을 예약하지 않습니다.
        try:
            if self.root.winfo_exists():
                self.root.after(250, self._poll)
        except tk.TclError:
            pass

    def _read_events(self, final: bool = False) -> None:
        if self.event_path is None or not self.event_path.exists():
            return
        try:
            with self.event_path.open("rb") as stream:
                stream.seek(self.event_offset)
                chunk = stream.read()
                self.event_offset = stream.tell()
            # UTF-8 문자가 쓰기 중간에 잘릴 가능성을 피하려면 완성된 줄만
            # 디코딩해야 합니다. JSONL은 ASCII 이스케이프 여부에 무관합니다.
            if not hasattr(self, "_pending_event_bytes"):
                self._pending_event_bytes = b""
            raw = self._pending_event_bytes + chunk
            parts = raw.split(b"\n")
            self._pending_event_bytes = parts.pop()
            if final and self._pending_event_bytes:
                parts.append(self._pending_event_bytes)
                self._pending_event_bytes = b""
            for line in parts:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line.decode("utf-8-sig"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._log("Could not read one progress-log line.")
                    continue
                self._handle_event(event)
        except OSError:
            # 짧은 파일 잠금은 다음 poll에서 다시 읽습니다. 오프셋은 실제
            # 읽기가 성공한 뒤에만 변경되므로 해당 기록을 잃지 않습니다.
            return

    def _handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type", "status")
        message = str(event.get("message", ""))
        if kind == "progress":
            done = int(event.get("done", 0) or 0)
            total = int(event.get("total", 0) or 0)
            stage = str(event.get("stage", "Bathymetry estimation"))
            if total > 0:
                self.progress.configure(value=max(0, min(100, 100 * done / total)))
                self.progress_text.set(
                    f"{stage} · {done:,} / {total:,} ({100 * done / total:.1f}%)"
                )
            else:
                self.progress_text.set(stage)
            if message:
                self.status.set(message)
        elif kind in {"complete", "cancelled", "error"}:
            self.terminal_event = event
            if event.get("output_dir"):
                self.output_dir = Path(event["output_dir"])
            if message:
                self._log(message)
        else:
            if message:
                self.status.set(message)
                self._log(message)

    def _finish_process(self, return_code: int) -> None:
        self.process = None
        self._set_busy(False)
        event = self.terminal_event or {}
        kind = event.get("type")
        if kind == "complete" and return_code == 0:
            self.progress.configure(value=100)
            summary: dict[str, Any] = {}
            if self.output_dir is not None:
                try:
                    summary = json.loads(
                        (self.output_dir / "analysis_record.json").read_text(encoding="utf-8-sig")
                    )
                except (OSError, json.JSONDecodeError):
                    pass
            failed = int(summary.get("failed_pixels", 0) or 0)
            if failed or event.get("message") == "completed_with_failures":
                # 전체 실행의 종료와 모든 픽셀의 성공은 다릅니다. 일부 실패가
                # 있으면 정상 완료와 구분하고 원인 CSV를 바로 안내합니다.
                self.progress_text.set("Analysis complete · some pixels failed")
                self.status.set(f"Analysis completed, but {failed:,} pixel calculations failed. Check the error CSV in the result folder.")
                self._log(f"Successful calculations: {int(summary.get('successful_pixels', 0)):,} / failed: {failed:,}")
            else:
                self.progress_text.set("Analysis complete")
                self.status.set("Analysis completed. Review the maps in the result folder.")
            self._log("Analysis complete: " + str(self.output_dir or "Results folder"))
        elif kind == "cancelled":
            self.progress_text.set("Stopped by user")
            self.status.set("Analysis stopped. Check the stopped status in the result folder.")
            self._log("Analysis stopped at the user's request.")
        else:
            error = str(
                event.get("message") or f"The calculation process ended without a completion record. Exit code: {return_code}"
            )
            self.progress_text.set("Analysis failed")
            self.status.set("An error occurred during analysis. Check the progress log.")
            self._log(error)
            if not self.close_when_finished:
                messagebox.showerror("Analysis failed", error, parent=self.root)
        if self.output_dir and self.output_dir.exists():
            self.open_result_button.configure(state="normal")
        self._cleanup_runtime()
        if self.close_when_finished:
            self.root.destroy()

    def _cleanup_runtime(self) -> None:
        if self.runtime_dir is not None:
            try:
                self.runtime_dir.cleanup()
            except OSError:
                pass
        self.runtime_dir = None
        self.event_path = None
        self.cancel_path = None
        self._pending_event_bytes = b""

    def _cancel(self) -> None:
        if self.process is None or self.cancel_path is None:
            return
        try:
            self.cancel_path.write_text("cancel\n", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Stop request failed", str(exc), parent=self.root)
            return
        self.cancel_button.configure(state="disabled")
        self.status.set("Stop requested. Finishing the current operation safely…")
        self._log("Stop requested.")

    def _close(self) -> None:
        if self.process is not None:
            if messagebox.askyesno(
                "Close during analysis", "Analysis is running. Request a stop, wait for safe shutdown, and then close the window?", parent=self.root
            ):
                self.close_when_finished = True
                self._cancel()
            return
        if self.validating:
            if messagebox.askyesno("Validating inputs", "Close the window after validation finishes?", parent=self.root):
                self.close_when_finished = True
            return
        self._cleanup_runtime()
        self.root.destroy()

    def _save_dialog(self) -> None:
        try:
            config = self._collect_config()
        except Exception as exc:
            messagebox.showwarning(
                "Save configuration", "Complete the required inputs before saving the configuration.\n\n" + str(exc), parent=self.root
            )
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save analysis configuration",
            initialfile="analysis_configuration.json",
            defaultextension=".json",
            filetypes=[("Analysis configuration JSON", "*.json")],
        )
        if path:
            try:
                Path(path).write_text(
                    json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self.status.set("Configuration saved: " + path)
            except OSError as exc:
                messagebox.showerror("Configuration save failed", str(exc), parent=self.root)

    def _load_dialog(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root, title="Load analysis configuration", filetypes=[("Analysis configuration JSON", "*.json")]
        )
        if path:
            self._load_config(Path(path))

    def _load_example(self, sensor: str) -> None:
        self._load_config(self.app_root / "DATA" / "examples" / sensor / "config.json", example=True)

    def _load_config(self, path: Path, example: bool = False) -> None:
        try:
            config = json.loads(path.read_text(encoding="utf-8-sig"))
            if config.get("sensor") not in SENSOR_LABELS.values():
                raise ValueError("sensor must be Sentinel2 or SuperDove.")
            if config.get("water_type", "case1") not in WATER_LABELS.values():
                raise ValueError("water_type must be case1, case2 or mixed.")
            if config.get("coefficient_mode", "fitted") not in {"fitted", "fixed"}:
                raise ValueError("coefficient_mode must be fitted or fixed.")

            def full(value: str | None) -> str:
                if not value:
                    return ""
                candidate = Path(value).expanduser()
                return str(
                    (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
                )

            self.loading = True
            self.sensor.set(
                next(label for label, key in SENSOR_LABELS.items() if key == config["sensor"])
            )
            self.water.set(WATER_NAMES[config.get("water_type", "case1")])
            mode = config.get("coefficient_mode", "fitted")
            unsupported = config.get("water_type", "case1") != "case1" and mode == "fitted"
            self.coefficient_mode.set("fixed" if unsupported else mode)
            self.loading = False
            self._selection_changed()
            files = [full(value) for value in config.get("rrs_files", [])]
            if len(files) == 1:
                self.rrs_mode.set("stack")
                self.stack_file.set(files[0])
                for variable in self.rrs_vars.values():
                    variable.set("")
            elif len(files) == len(self._waves()):
                self.rrs_mode.set("separate")
                self.stack_file.set("")
                for wave, filename in zip(self._waves(), files):
                    self.rrs_vars[_wave_label(wave)].set(filename)
            else:
                raise ValueError(
                    f"The Rrs file count is incorrect. Provide one multiband file or {len(self._waves())} single-band files."
                )
            self.deep_selection = None
            self.deep_selection_signature = None
            if "deep_selection" in config:
                self.deep_selection = config["deep_selection"]
                self.deep_selection_signature = self._image_signature(
                    {"sensor": config["sensor"], "rrs_files": files}
                )
                self.deep_mode.set("pixels")
                self.deep_csv.set("")
            elif "deep_rrs" in config:
                values = config["deep_rrs"]
                if len(values) != len(self._waves()):
                    raise ValueError("The deep-water Rrs count does not match the sensor wavelength count.")
                self.deep_mode.set("manual")
                self.deep_csv.set("")
                for wave, value in zip(self._waves(), values):
                    self.deep_vars[_wave_label(wave)].set(str(value))
            else:
                self.deep_mode.set("csv")
                self.deep_csv.set(full(config.get("deep_csv")))
            self._refresh_deep_pixel_note()
            self.solar_zenith.set(
                str(config["solar_zenith"]) if config.get("solar_zenith") is not None else ""
            )
            self.mask.set(full(config.get("mask")))
            self.output_root.set(
                full(config.get("output_root"))
                or str(self.app_root.parents[1] / "05_RESULTS" / "02_SEMI_ANALYTICAL")
            )
            self.run_name.set(str(config.get("run_name", "")))
            self.workers.set(str(config.get("workers", max(1, min(4, (os.cpu_count() or 2) - 1)))))
            bottoms = [full(value) for value in config.get("bottom_files", [])]
            if len(bottoms) not in {2, 3}:
                raise ValueError("The configuration requires two or three bottom-reflectance files.")
            # 사용자 화면은 하나의 Bottom-data folder를 탐색합니다. 다른 폴더의
            # 파일을 조용히 누락하지 않고 불러오기 단계에서 명확히 알립니다.
            parents = {str(Path(value).parent) for value in bottoms}
            if len(parents) != 1:
                raise ValueError("Place all selected bottom-reflectance files in one folder.")
            self.bottom_folder.set(str(Path(bottoms[0]).parent))
            self.bottom_count.set(len(bottoms))
            self.pending_bottom_paths = bottoms
            self._scan_bottoms()
            self.notebook.select(0)
            self.status.set(("Example" if example else "Configuration") + " loaded. Check the inputs and bottom selection before running.")
            self._set_text(
                self.review_text, "Configuration loaded. Click “Validate inputs” before running.\n\n" + str(path)
            )
            self._log("Load configuration: " + str(path))
            if unsupported:
                messagebox.showinfo(
                    "Coefficient mode changed", "Precomputed wavelength-specific coefficients are unavailable for the selected water type, so preset coefficients were selected.", parent=self.root
                )
        except Exception as exc:
            self.loading = False
            messagebox.showerror("Configuration load failed", str(exc), parent=self.root)

    def _open_manual(self) -> None:
        """동봉된 설명서를 기본 HTML 뷰어로 연다. 설치 위치가 바뀌어도 동작한다."""
        manual = self.app_root / "USER_GUIDE.html"
        try:
            if not manual.is_file():
                raise FileNotFoundError("USER_GUIDE.html must be located beside the source application folder.")
            os.startfile(str(manual))
        except OSError as exc:
            messagebox.showerror("Could not open the user guide", str(exc), parent=self.root)

    def _open_result(self) -> None:
        if self.output_dir is None or not self.output_dir.exists():
            messagebox.showinfo("Results folder", "No completed results folder is available yet.", parent=self.root)
            return
        try:
            os.startfile(str(self.output_dir))
        except OSError as exc:
            messagebox.showerror("Could not open the folder", str(exc), parent=self.root)

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _log(self, text: str) -> None:
        now = _datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{now}] {text}\n")
        # 진행률마다 로그를 쌓지 않으며 상태 기록도 최대 400줄로 제한합니다.
        if int(self.log_text.index("end-1c").split(".")[0]) > 400:
            self.log_text.delete("1.0", "50.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


def launch(app_root: Path) -> None:
    """소스 실행과 Windows 실행 파일이 공통으로 호출하는 진입점."""
    # 고해상도 Windows에서 흐릿한 글자를 줄입니다. 시스템에 해당 API가
    # 없거나 Already Configuration되어 있어도 계산/실행을 중단하지 않습니다.
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    BathymetryApp(root, Path(app_root))
    root.mainloop()
