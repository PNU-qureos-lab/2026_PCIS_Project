"""원본 Rrs 영상에서 심수역 기준 픽셀 10개를 직접 선택하는 창.

큰 영상 전체를 화면 해상도로 펼치지 않고 현재 보이는 범위만 읽는다. 확대·축소는
표시용이며 선택 좌표와 통계는 항상 원본 픽셀 격자를 사용한다. 좌표는 내부에서
0부터, 화면에서는 1부터 표시한다. 평균/중위값 전환은 선택 집합을 변경하지 않는다.
각 픽셀은 클릭으로 선택·해제하며 모든 밴드가 유효한 10개를 선택해야 적용할 수 있다.
"""
import math
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from rasterio.windows import Window
from .deepwater import (
    RrsSource,
    aggregate,
    normalize_pixels,
    create_selection,
    MAX_SELECTED_PIXELS,
    STATISTIC_LABELS,
)


class PixelSelector:
    """선택·표시·통계를 관리하고 적용 시 검증한 설정을 on_apply에 전달한다."""

    def __init__(self, parent, config, on_apply, initial=None, visible=True):
        self.config = config
        self.on_apply = on_apply
        self.source = RrsSource(config)
        self.window = tk.Toplevel(parent)
        self.window.withdraw()
        self.window.title("Select Deep-Water Reference Pixels")
        self.window.geometry(
            f"{min(1180,parent.winfo_screenwidth()-80)}x{min(860,parent.winfo_screenheight()-100)}"
        )
        self.window.minsize(850, 600)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.closed = False
        self.busy = False
        self.pending = None
        self.poll_id = None
        self.drag = None
        self.photo = None
        self.needs_initial_fit = True
        self.messages = queue.Queue()
        self.pixels = set()
        self.previous = None
        self.invalid = set()
        self.zoom = 1.0
        self.statistic = tk.StringVar(value="median")
        self.status = tk.StringVar(value="Selected 0/10 · Click 10 deep-water pixels individually.")
        self.position = tk.StringVar(value="Wheel: zoom · Middle-button drag or scroll bars: move")
        self.summary = None
        try:
            self.indices, self.ranges = self.source.rgb_stretch()
            self._build()
            if (
                initial
                and initial.get("grid") == self.source.grid()
                and initial.get("sensor") == config["sensor"]
            ):
                # 이전 설정이 10개를 넘으면 임의로 일부를 고르지 않고 새로 선택하게 한다.
                if len(initial.get("pixels", [])) <= MAX_SELECTED_PIXELS:
                    self.pixels = set(
                        normalize_pixels(
                            initial.get("pixels", []), self.source.width, self.source.height
                        )
                    )
                self.statistic.set(initial.get("statistic", "median"))
            self.statistic.trace_add("write", lambda *_: self._statistics())
            self._statistics()
            self.window.update_idletasks()
            self.fit()
            if visible:
                self.window.deiconify()
                self.window.grab_set()
            self.poll_id = self.window.after(100, self._poll)
        except BaseException:
            self.close()
            raise

    def _build(self):
        header = ttk.Frame(self.window, padding=(12, 10))
        header.pack(fill="x")
        ttk.Label(header, text="Select Deep-Water Reference Pixels", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(
            header,
            text="Click 10 deep-water pixels where bottom influence is negligible. Avoid clouds, shadows, and surface reflection.",
            wraplength=1050,
        ).pack(anchor="w", pady=(3, 0))
        toolbar = ttk.Frame(self.window, padding=(12, 0, 12, 6))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Click: select / click again: clear").pack(side="left", padx=(0, 10))
        ttk.Button(toolbar, text="Clear selection", command=self.clear).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Fit image", command=self.fit).pack(side="right", padx=3)
        ttk.Button(toolbar, text="100%", command=lambda: self.set_zoom(1.0)).pack(
            side="right", padx=3
        )
        ttk.Button(toolbar, text="−", width=3, command=lambda: self.set_zoom(self.zoom / 2)).pack(
            side="right", padx=2
        )
        ttk.Button(toolbar, text="+", width=3, command=lambda: self.set_zoom(self.zoom * 2)).pack(
            side="right", padx=2
        )
        self.zoom_label = ttk.Label(toolbar, text="100%")
        self.zoom_label.pack(side="right", padx=8)

        view = ttk.Frame(self.window, padding=(12, 0))
        view.pack(fill="both", expand=True)
        view.rowconfigure(0, weight=1)
        view.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(view, background="#20252b", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        horizontal = ttk.Scrollbar(
            view, orient="horizontal", command=lambda *args: self._scroll("x", args)
        )
        vertical = ttk.Scrollbar(
            view, orient="vertical", command=lambda *args: self._scroll("y", args)
        )
        horizontal.grid(row=1, column=0, sticky="ew")
        vertical.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<Configure>", self._resized)
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<Motion>", self.hover)
        self.canvas.bind("<MouseWheel>", self.wheel)
        self.canvas.bind("<ButtonPress-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>", self.pan)
        ttk.Label(self.window, textvariable=self.position, padding=(12, 4)).pack(fill="x")

        panel = ttk.Frame(self.window, padding=(12, 3, 12, 5))
        panel.pack(fill="x")
        ttk.Label(panel, text="Reference statistic").pack(side="left", padx=(0, 14))
        self.stat_radios = []
        for text, value in [("Median (default)", "median"), ("Mean", "mean")]:
            control = ttk.Radiobutton(panel, text=text, variable=self.statistic, value=value)
            control.pack(side="left", padx=(0, 15))
            self.stat_radios.append(control)
        ttk.Label(panel, text="Yellow: valid · Red: replace this pixel", foreground="#665522").pack(side="right")
        self.table = ttk.Treeview(
            self.window,
            columns=("wave", "value", "std", "range"),
            show="headings",
            height=len(self.source.waves),
        )
        for key, title, width in [
            ("wave", "Wavelength (nm)", 130),
            ("value", "Reference Rrs (sr⁻¹)", 220),
            ("std", "Rrs standard deviation (sr⁻¹)", 240),
            ("range", "Rrs minimum to maximum (sr⁻¹)", 330),
        ]:
            self.table.heading(key, text=title)
            self.table.column(key, width=width, anchor="center")
        self.table.pack(fill="x", padx=12)
        ttk.Label(
            self.window,
            text="The statistic gives equal weight to above-water remote-sensing reflectance (Rrs) from every valid selected pixel. Pixels may be selected outside the analysis mask.",
            padding=(12, 5),
            wraplength=1050,
        ).pack(fill="x")
        footer = ttk.Frame(self.window, padding=(12, 4, 12, 12))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status, wraplength=760).pack(
            side="left", fill="x", expand=True
        )
        self.apply_button = ttk.Button(footer, text="Apply the 10 selected pixels", command=self.apply)
        self.apply_button.pack(side="right", padx=(6, 0))
        ttk.Button(footer, text="Close", command=self.close).pack(side="right")

    def _scroll(self, axis, args):
        (self.canvas.xview if axis == "x" else self.canvas.yview)(*args)
        self.schedule_render()

    def _resized(self, event):
        # 표시 전 창은 1×1로 보고될 수 있어 실제 화면 배치 후 전체 보기 배율을 정한다.
        if self.needs_initial_fit and event.width > 1 and event.height > 1:
            self.needs_initial_fit = False
            self.fit()
        else:
            self.schedule_render()

    def pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.schedule_render()

    def fit(self):
        self.set_zoom(
            min(
                1,
                max(1, self.canvas.winfo_width()) / self.source.width,
                max(1, self.canvas.winfo_height()) / self.source.height,
            )
        )
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.schedule_render()

    def set_zoom(self, value, anchor=None):
        if self.closed:
            return
        fit = min(
            1,
            max(1, self.canvas.winfo_width()) / self.source.width,
            max(1, self.canvas.winfo_height()) / self.source.height,
        )
        value = min(16, max(min(fit, 0.01), float(value)))
        x, y = anchor or (self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)
        col = self.canvas.canvasx(x) / self.zoom
        row = self.canvas.canvasy(y) / self.zoom
        self.zoom = value
        self.canvas.configure(
            scrollregion=(0, 0, self.source.width * value, self.source.height * value)
        )
        self.canvas.xview_moveto(max(0, (col * value - x) / (self.source.width * value)))
        self.canvas.yview_moveto(max(0, (row * value - y) / (self.source.height * value)))
        self.zoom_label.configure(text=f"{100*value:.0f}%")
        self.schedule_render()

    def wheel(self, event):
        self.set_zoom(self.zoom * (1.5 if event.delta > 0 else 1 / 1.5), (event.x, event.y))
        return "break"

    def schedule_render(self):
        # 연속 확대·이동 이벤트는 마지막 요청으로 합쳐 같은 화면을 반복해서 읽지 않는다.
        if self.closed:
            return
        if self.pending is not None:
            self.window.after_cancel(self.pending)
        self.pending = self.window.after(45, self.render)

    def render(self):
        """현재 보이는 원본 구간만 읽어 표시하고 선택 픽셀의 유효 상태를 덧그린다."""
        self.pending = None
        if self.closed:
            return
        left = max(0, self.canvas.canvasx(0))
        top = max(0, self.canvas.canvasy(0))
        c0 = min(self.source.width - 1, int(left / self.zoom))
        r0 = min(self.source.height - 1, int(top / self.zoom))
        c1 = min(self.source.width, math.ceil((left + self.canvas.winfo_width()) / self.zoom) + 1)
        r1 = min(self.source.height, math.ceil((top + self.canvas.winfo_height()) / self.zoom) + 1)
        w = max(1, round((c1 - c0) * self.zoom))
        h = max(1, round((r1 - r0) * self.zoom))
        try:
            rgb = self.source.rgb(
                Window(c0, r0, c1 - c0, r1 - r0), (h, w), self.indices, self.ranges
            )
            picture = Image.fromarray(rgb)
            draw = ImageDraw.Draw(picture)
            marks = {}
            for r, c in self.pixels:
                if r0 <= r < r1 and c0 <= c < c1:
                    key = (round((c - c0 + 0.5) * self.zoom), round((r - r0 + 0.5) * self.zoom))
                    bad = (r, c) in self.invalid
                    marks[key] = marks.get(key, False) or bad
            for (x, y), bad in marks.items():
                radius = max(2, self.zoom * 0.45)
                draw.rectangle(
                    (x - radius, y - radius, x + radius, y + radius),
                    outline="#ff5b60" if bad else "#ffff44",
                    width=2,
                )
            self.photo = ImageTk.PhotoImage(picture, master=self.window)
            self.canvas.itemconfigure(self.image_item, image=self.photo)
            self.canvas.coords(self.image_item, c0 * self.zoom, r0 * self.zoom)
        except Exception as exc:
            self.status.set("Image display error: " + str(exc))

    def pixel_at(self, x, y, clamp=False):
        """화면 좌표를 원본 영상의 0부터 시작하는 (행, 열)로 변환한다."""
        col = math.floor(self.canvas.canvasx(x) / self.zoom)
        row = math.floor(self.canvas.canvasy(y) / self.zoom)
        if clamp:
            return (
                min(self.source.height - 1, max(0, row)),
                min(self.source.width - 1, max(0, col)),
            )
        return (
            (row, col) if 0 <= row < self.source.height and 0 <= col < self.source.width else None
        )

    def hover(self, event):
        pixel = self.pixel_at(event.x, event.y)
        if pixel:
            self.position.set(
                f"Row {pixel[0]+1:,} · Column {pixel[1]+1:,} (source pixel)   |   Wheel: zoom · Middle button: move"
            )

    def press(self, event):
        if self.busy:
            return
        self.drag = self.pixel_at(event.x, event.y)

    def release(self, event):
        start = self.drag
        self.drag = None
        if start is None or self.busy:
            return
        end = self.pixel_at(event.x, event.y, True)
        # 클릭 시작과 종료가 같은 픽셀일 때만 토글하여 드래그에 의한 일괄 선택을 막는다.
        if start != end:
            return
        selected = set(self.pixels)
        if start in selected:
            selected.remove(start)
        else:
            selected.add(start)
        self.set_pixels(selected)

    def set_pixels(self, pixels):
        if self.busy:
            return
        if len(pixels) > MAX_SELECTED_PIXELS:
            messagebox.showinfo(
                "Selection limit", f"Select no more than {MAX_SELECTED_PIXELS:,} pixels.", parent=self.window
            )
            return
        self.previous = set(self.pixels)
        self.pixels = set(pixels)
        self._statistics()

    def clear(self):
        self.set_pixels(set())

    def undo(self):
        if self.previous is not None and not self.busy:
            old = self.previous
            self.previous = None
            self.pixels = old
            self._statistics()

    def _statistics(self):
        """선택 도중의 통계를 갱신하고 유효한 픽셀 10개일 때만 적용을 허용한다."""
        if self.closed or self.busy:
            return
        self.table.delete(*self.table.get_children())
        self.summary = None
        self.invalid = set()
        try:
            pixels, values = self.source.sample(self.pixels)
            self.invalid = {
                p for p, v in zip(pixels, values) if not np.all(np.isfinite(v) & (v > 0))
            }
            if not pixels:
                raise ValueError("Click 10 deep-water pixels individually.")
            self.summary = aggregate(values, self.statistic.get())
            for i, wave in enumerate(self.source.waves):
                self.table.insert(
                    "",
                    "end",
                    values=(
                        wave,
                        f"{self.summary['rrs'][i]:.8g}",
                        f"{self.summary['std'][i]:.5g}",
                        f"{self.summary['minimum'][i]:.6g} ~ {self.summary['maximum'][i]:.6g}",
                    ),
                )
            ready = (
                len(pixels) == MAX_SELECTED_PIXELS
                and self.summary["valid_count"] == MAX_SELECTED_PIXELS
            )
            hint = (
                " · Clear each red pixel and choose another pixel."
                if self.invalid
                else ("" if ready else " · Select all 10 pixels.")
            )
            self.status.set(
                f"Selected {len(pixels)}/10 · Valid {self.summary['valid_count']} · {STATISTIC_LABELS[self.statistic.get()]}{hint}"
            )
            self.apply_button.configure(state="normal" if ready else "disabled")
        except Exception as exc:
            self.status.set(f"Selected {len(self.pixels)}/10 · Excluded {len(self.invalid)} · " + str(exc))
            self.apply_button.configure(state="disabled")
        self.schedule_render()

    def apply(self):
        """원본 파일 확인은 별도 스레드에서 수행하여 선택 창의 응답을 유지한다."""
        if self.busy or self.summary is None:
            return
        if (
            len(self.pixels) != MAX_SELECTED_PIXELS
            or self.summary["valid_count"] != MAX_SELECTED_PIXELS
        ):
            return
        self.busy = True
        self.apply_button.configure(state="disabled")
        for control in self.stat_radios:
            control.configure(state="disabled")
        self.status.set("Checking the selected source pixels and reference values…")
        pixels = sorted(self.pixels)
        statistic = self.statistic.get()

        def work():
            try:
                self.messages.put(("ok", create_selection(self.config, pixels, statistic)))
            except Exception as exc:
                self.messages.put(("error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        # 작업 스레드는 큐로 결과만 전달하고 Tk 위젯 갱신은 이 화면 스레드에서 수행한다.
        self.poll_id = None
        if self.closed:
            return
        try:
            kind, payload = self.messages.get_nowait()
            self.busy = False
            for control in self.stat_radios:
                control.configure(state="normal")
            if kind == "ok":
                self.on_apply(payload)
                self.close()
                return
            self._statistics()
            messagebox.showerror("Deep-water reference", payload, parent=self.window)
        except queue.Empty:
            pass
        self.poll_id = self.window.after(100, self._poll)

    def close(self):
        """예약된 화면 갱신을 취소한 뒤 영상 파일과 창을 해제한다."""
        if self.closed:
            return
        self.closed = True
        if self.pending is not None:
            try:
                self.window.after_cancel(self.pending)
            except tk.TclError:
                pass
        if self.poll_id is not None:
            try:
                self.window.after_cancel(self.poll_id)
            except tk.TclError:
                pass
        self.source.close()
        try:
            self.window.grab_release()
            self.window.destroy()
        except tk.TclError:
            pass
