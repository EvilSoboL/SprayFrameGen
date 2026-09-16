"""Полный интерфейс настроек, предпросмотра и экспорта."""

import json
import threading
import time
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .. import __version__
from . import InterfaceError
from ..configuration import (
    Configuration,
    ConfigurationError,
    dumps,
    from_mapping,
    load,
    loads,
    new_configuration,
    save,
    update_configuration,
)
from ..environment import current_environment
from ..generation import generate_frames, prepare
from ..generation.controller import (
    CancellationToken,
    CompletionEvent,
    ExportError,
    ProgressEvent,
    run_generation,
)
from ..render import create_renderer, encode_intensity


RESOLUTIONS = [(640, 480), (1024, 768), (1280, 1024), (1920, 1080)]
FORMAT_BITDEPTH = [("PNG 8 бит", "png", 8), ("PNG 16 бит", "png", 16), ("TIFF 16 бит", "tiff", 16)]
SIZE_DISTRIBUTIONS = [("Логнормальное", "lognormal"), ("Нормальное", "normal")]
CONCENTRATIONS = [("Низкая", "low"), ("Средняя", "medium"), ("Высокая", "high")]
BACKGROUND_TONES = [("Светлый", "light"), ("Тёмный", "dark")]
RENDER_MODES = [("Реалистичный", "realistic"), ("Идеальный", "ideal")]


class SettingsWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"SprayFrameGen {__version__} — Настройки")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        self.config: Optional[Configuration] = None
        self.preview_config: Optional[Configuration] = None
        self.preview_image: Optional[Image.Image] = None
        self.preview_tk: Optional[ImageTk.PhotoImage] = None
        self.preview_overlay: bool = False
        self.generation_thread: Optional[threading.Thread] = None
        self.cancellation_token: Optional[CancellationToken] = None
        self.generation_start_time: float = 0
        self.generation_cancelled: bool = False
        self._alive: bool = True

        self.vars: dict[str, tk.Variable] = {}
        self.widgets: dict[str, tk.Widget] = {}

        self._build_ui()
        self._reset_config("realistic")
        self._update_preview()

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_paned, width=500)
        main_paned.add(left_frame, weight=1)

        right_frame = ttk.Frame(main_paned, width=700)
        main_paned.add(right_frame, weight=2)

        self._build_left_panel(left_frame)
        self._build_right_panel(right_frame)

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=10, pady=5)
        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor="w")

    def _build_left_panel(self, parent: ttk.Frame):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        self.tab_general = ttk.Frame(notebook)
        self.tab_droplets = ttk.Frame(notebook)
        self.tab_motion = ttk.Frame(notebook)
        self.tab_appearance = ttk.Frame(notebook)
        self.tab_background = ttk.Frame(notebook)
        self.tab_random = ttk.Frame(notebook)

        notebook.add(self.tab_general, text="Серия и камера")
        notebook.add(self.tab_droplets, text="Капли")
        notebook.add(self.tab_motion, text="Скорость")
        notebook.add(self.tab_appearance, text="Внешний вид")
        notebook.add(self.tab_background, text="Фон и шум")
        notebook.add(self.tab_random, text="Случайная генерация")

        self._build_tab_general()
        self._build_tab_droplets()
        self._build_tab_motion()
        self._build_tab_appearance()
        self._build_tab_background()
        self._build_tab_random()

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Новая: реалистичный", command=lambda: self._reset_config("realistic")).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="Новая: идеальный", command=lambda: self._reset_config("ideal")).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="Открыть…", command=self._open_file).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="Сохранить…", command=self._save_file).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="Проверить", command=self._check_config).pack(side="left", padx=3)

    def _build_tab_general(self):
        f = self.tab_general
        self._add_combobox(f, "camera.width_px", "Разрешение", 0,
                          [f"{w}×{h}" for w, h in RESOLUTIONS],
                          path2="camera.height_px")
        self._add_int(f, "camera.frame_count", "Число кадров", 1, 2, 1000)
        self._add_float(f, "camera.scale_um_per_px", "Масштаб (мкм/пиксель)", 2, 0.1, 1000.0)
        self._add_float(f, "camera.frame_interval_us", "Межкадровый интервал (мкс)", 3, 0.1, 100000.0)
        self._add_float(f, "camera.exposure_us", "Выдержка (мкс)", 4, 0.1, 100000.0)
        self._add_combobox(f, "camera.format", "Формат и разрядность", 5,
                          [label for label, _, _ in FORMAT_BITDEPTH],
                          lambda v: (FORMAT_BITDEPTH[v.current()][1], FORMAT_BITDEPTH[v.current()][2]),
                          path2="camera.bit_depth")

    def _build_tab_droplets(self):
        f = self.tab_droplets
        self._add_int(f, "droplets.count_per_frame", "Капель на кадр", 0, 1, 10000)
        self._add_combobox(f, "droplets.size_distribution", "Тип распределения размеров", 1,
                          [label for label, _ in SIZE_DISTRIBUTIONS],
                          lambda v: SIZE_DISTRIBUTIONS[v.current()][1])
        self._add_float(f, "droplets.mean_um", "Среднее (мкм, для нормального)", 2, 1.0, 5000.0)
        self._add_float(f, "droplets.std_um", "Стд. отклонение (мкм, для нормального)", 3, 0.0, 5000.0)
        self._add_float(f, "droplets.median_um", "Медиана (мкм, для логнормального)", 4, 1.0, 5000.0)
        self._add_float(f, "droplets.geometric_std", "Геом. стд. отклонение (для логнормального)", 5, 1.0, 10.0)
        self._add_float(f, "droplets.max_diameter_um", "Макс. диаметр (мкм)", 6, 1.0, 5000.0)
        self._add_combobox(f, "droplets.concentration", "Концентрация", 7,
                          [label for label, _ in CONCENTRATIONS],
                          lambda v: CONCENTRATIONS[v.current()][1])
        self._add_combobox(f, "droplets.render_mode", "Режим рендера", 8,
                          [label for label, _ in RENDER_MODES],
                          lambda v: RENDER_MODES[v.current()][1])

    def _build_tab_motion(self):
        f = self.tab_motion
        self._add_float(f, "motion.speed_mean_m_s", "Средняя скорость (м/с)", 0, 0.1, 100.0)
        self._add_float(f, "motion.speed_std_m_s", "Стд. отклонение скорости (м/с)", 1, 0.0, 50.0)
        self._add_float(f, "motion.speed_min_m_s", "Мин. скорость (м/с)", 2, 0.1, 100.0)
        self._add_float(f, "motion.speed_max_m_s", "Макс. скорость (м/с)", 3, 0.1, 100.0)
        self._add_float(f, "motion.direction_std_deg", "Угловой разброс (°)", 4, 0.0, 90.0)
        self._add_checkbox(f, "motion.motion_blur_enabled", "Размытие движения", 5)

    def _build_tab_appearance(self):
        f = self.tab_appearance
        self._add_checkbox(f, "appearance.defocus_enabled", "Гауссова расфокусировка", 0)
        self._add_float(f, "appearance.blur_sigma_min_px", "Мин. сигма (пиксель)", 1, 0.0, 10.0)
        self._add_float(f, "appearance.blur_sigma_max_px", "Макс. сигма (пиксель)", 2, 0.0, 10.0)

    def _build_tab_background(self):
        f = self.tab_background
        self._add_combobox(f, "background.tone", "Тон фона", 0,
                          [label for label, _ in BACKGROUND_TONES],
                          lambda v: BACKGROUND_TONES[v.current()][1])
        self._add_checkbox(f, "background.noise_enabled", "Сенсорный шум", 1)
        self._add_float(f, "background.noise_sigma", "Интенсивность шума (σ)", 2, 0.0, 1.0)

    def _build_tab_random(self):
        f = self.tab_random
        self._add_int(f, "seed", "Seed", 0, 0, 4294967295)
        btn = ttk.Button(f, text="Создать новый seed", command=self._new_seed)
        btn.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

    def _build_right_panel(self, parent: ttk.Frame):
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill="both", expand=True)

        preview_frame = ttk.LabelFrame(top_frame, text="Предпросмотр", padding=5)
        preview_frame.pack(fill="both", expand=True, pady=(0, 5))

        self.preview_canvas = tk.Canvas(preview_frame, bg="gray", highlightthickness=1, highlightbackground="gray")
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.bind("<Configure>", lambda e: self._update_preview())

        overlay_frame = ttk.Frame(preview_frame)
        overlay_frame.pack(fill="x", pady=5)
        self.overlay_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(overlay_frame, text="Показать служебный слой (ID, контуры, векторы, границы)",
                       variable=self.overlay_var, command=self._toggle_overlay).pack(anchor="w")

        export_frame = ttk.LabelFrame(top_frame, text="Экспорт", padding=5)
        export_frame.pack(fill="x", pady=5)

        dir_frame = ttk.Frame(export_frame)
        dir_frame.pack(fill="x", pady=2)
        ttk.Label(dir_frame, text="Папка:").pack(side="left")
        self.export_dir_var = tk.StringVar()
        self.vars["export.output_directory"] = self.export_dir_var
        self.export_dir_entry = ttk.Entry(dir_frame, textvariable=self.export_dir_var)
        self.export_dir_entry.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(dir_frame, text="Обзор…", command=self._choose_export_dir).pack(side="left")

        progress_frame = ttk.Frame(export_frame)
        progress_frame.pack(fill="x", pady=5)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x")

        info_frame = ttk.Frame(progress_frame)
        info_frame.pack(fill="x", pady=2)
        self.progress_label = tk.StringVar(value="Готов к запуску")
        ttk.Label(info_frame, textvariable=self.progress_label).pack(side="left")
        self.time_label = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.time_label).pack(side="right")

        btn_frame = ttk.Frame(export_frame)
        btn_frame.pack(fill="x", pady=5)
        self.start_btn = ttk.Button(btn_frame, text="Запустить генерацию", command=self._start_generation)
        self.start_btn.pack(side="left", padx=3)
        self.cancel_btn = ttk.Button(btn_frame, text="Отмена", command=self._cancel_generation, state="disabled")
        self.cancel_btn.pack(side="left", padx=3)

    def _add_int(self, parent, path, label, row, min_val, max_val):
        var = tk.IntVar()
        self.vars[path] = var
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=3)
        spin = ttk.Spinbox(parent, from_=min_val, to=max_val, textvariable=var, width=20)
        spin.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        var.trace_add("write", lambda *_: self._on_field_change(path))
        parent.columnconfigure(1, weight=1)

    def _add_float(self, parent, path, label, row, min_val, max_val):
        var = tk.DoubleVar()
        self.vars[path] = var
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=3)
        spin = ttk.Spinbox(parent, from_=min_val, to=max_val, textvariable=var, width=20, increment=0.1)
        spin.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        var.trace_add("write", lambda *_: self._on_field_change(path))
        parent.columnconfigure(1, weight=1)

    def _add_combobox(self, parent, path, label, row, values, getter=None, path2=None):
        var = tk.StringVar()
        self.vars[path] = var
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=3)
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=18)
        combo.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        combo.current(0)

        if path == "camera.width_px" and path2 == "camera.height_px":
            combo.bind("<<ComboboxSelected>>", lambda e: self._on_resolution_change())
        elif path == "camera.format" and path2 == "camera.bit_depth":
            combo.bind("<<ComboboxSelected>>", lambda e: self._on_combobox_change(path, path2, getter))
        else:
            combo.bind("<<ComboboxSelected>>", lambda e: self._on_field_change(path))
        parent.columnconfigure(1, weight=1)

    def _add_checkbox(self, parent, path, label, row):
        var = tk.BooleanVar()
        self.vars[path] = var
        ttk.Checkbutton(parent, text=label, variable=var,
                       command=lambda: self._on_field_change(path)).grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=3)

    def _on_field_change(self, path: str):
        try:
            self._update_config_from_vars()
            self._update_preview()
        except Exception:
            pass

    def _on_combobox_change(self, path1: str, path2: str, getter):
        try:
            val = getter(self.widgets.get(path1, None))
            if path2 == "camera.bit_depth":
                self._update_config_from_vars()
            self._update_preview()
        except Exception:
            pass

    def _on_resolution_change(self):
        try:
            val = self.vars["camera.width_px"].get()
            w, h = RESOLUTIONS[[f"{w}×{h}" for w, h in RESOLUTIONS].index(val)]
            self._update_config_from_vars()
            self._update_preview()
        except Exception:
            pass

    def _reset_config(self, render_mode: str):
        self.config = new_configuration(render_mode=render_mode)
        self._vars_from_config()
        self._update_preview()

    def _vars_from_config(self):
        if not self.config:
            return
        data = asdict(self.config)
        for path, var in self.vars.items():
            try:
                value = data
                for part in path.split("."):
                    value = value[part]
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(value))
                elif isinstance(var, tk.IntVar):
                    var.set(int(value))
                elif isinstance(var, tk.DoubleVar):
                    var.set(float(value))
                elif isinstance(var, tk.StringVar):
                    if path == "camera.width_px":
                        w, h = value, data["camera"]["height_px"]
                        var.set(f"{w}×{h}")
                    elif path == "camera.format":
                        fmt, bits = value, data["camera"]["bit_depth"]
                        for label, f, b in FORMAT_BITDEPTH:
                            if f == fmt and b == bits:
                                var.set(label)
                                break
                    elif path == "droplets.size_distribution":
                        for label, v in SIZE_DISTRIBUTIONS:
                            if v == value:
                                var.set(label)
                                break
                    elif path == "droplets.concentration":
                        for label, v in CONCENTRATIONS:
                            if v == value:
                                var.set(label)
                                break
                    elif path == "droplets.render_mode":
                        for label, v in RENDER_MODES:
                            if v == value:
                                var.set(label)
                                break
                    elif path == "background.tone":
                        for label, v in BACKGROUND_TONES:
                            if v == value:
                                var.set(label)
                                break
                    elif path == "motion.motion_blur_enabled":
                        pass
                    else:
                        var.set(str(value))
            except (KeyError, TypeError):
                pass

    def _update_config_from_vars(self):
        if not self.config:
            return
        data = asdict(self.config)
        for path, var in self.vars.items():
            try:
                value = var.get()
                parts = path.split(".")
                target = data
                for part in parts[:-1]:
                    target = target[part]
                if isinstance(var, tk.BooleanVar):
                    target[parts[-1]] = bool(value)
                elif isinstance(var, tk.IntVar):
                    target[parts[-1]] = int(value)
                elif isinstance(var, tk.DoubleVar):
                    target[parts[-1]] = float(value)
                elif isinstance(var, tk.StringVar):
                    if path == "camera.width_px":
                        if value:
                            w, h = map(int, value.split("×"))
                            data["camera"]["width_px"] = w
                            data["camera"]["height_px"] = h
                    elif path == "camera.format":
                        for label, f, b in FORMAT_BITDEPTH:
                            if label == value:
                                data["camera"]["format"] = f
                                data["camera"]["bit_depth"] = b
                                break
                    elif path == "droplets.size_distribution":
                        for label, v in SIZE_DISTRIBUTIONS:
                            if label == value:
                                target[parts[-1]] = v
                                break
                    elif path == "droplets.concentration":
                        for label, v in CONCENTRATIONS:
                            if label == value:
                                target[parts[-1]] = v
                                break
                    elif path == "droplets.render_mode":
                        for label, v in RENDER_MODES:
                            if label == value:
                                target[parts[-1]] = v
                                break
                    elif path == "background.tone":
                        for label, v in BACKGROUND_TONES:
                            if label == value:
                                target[parts[-1]] = v
                                break
                    else:
                        target[parts[-1]] = value
            except (KeyError, TypeError, ValueError):
                pass
        result = from_mapping(data)
        self.config = result.configuration
        for w in result.warnings:
            self.status_var.set(f"Предупреждение: {w}")

    def _new_seed(self):
        import secrets
        new_seed = secrets.randbits(32)
        self.vars["seed"].set(new_seed)
        self._on_field_change("seed")

    def _choose_export_dir(self):
        path = filedialog.askdirectory(parent=self.root)
        if path:
            self.export_dir_var.set(path)
            self._on_field_change("export.output_directory")

    def _open_file(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("Настройки JSON", "*.json")])
        if path:
            try:
                result = load(path)
                self.config = result.configuration
                self._vars_from_config()
                self._update_preview()
                self.status_var.set("Настройки загружены. " + " ".join(result.warnings))
                if result.warnings:
                    messagebox.showwarning("Предупреждения", "\n".join(result.warnings), parent=self.root)
            except (ConfigurationError, OSError, UnicodeError) as exc:
                messagebox.showerror("Ошибка загрузки", str(exc), parent=self.root)

    def _save_file(self):
        try:
            self._update_config_from_vars()
            path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".json",
                                              filetypes=[("Настройки JSON", "*.json")])
            if path:
                save(self.config, path)
                self.status_var.set(f"Настройки сохранены: {path}")
        except (ConfigurationError, OSError, UnicodeError) as exc:
            messagebox.showerror("Ошибка сохранения", str(exc), parent=self.root)

    def _check_config(self):
        try:
            self._update_config_from_vars()
            result = loads(dumps(self.config))
            msg = "Конфигурация корректна."
            if result.warnings:
                msg += "\nПредупреждения:\n" + "\n".join(result.warnings)
            messagebox.showinfo("Проверка", msg, parent=self.root)
            self.status_var.set("Настройки проверены. " + " ".join(result.warnings))
        except (ConfigurationError, OSError, UnicodeError) as exc:
            messagebox.showerror("Ошибка настроек", str(exc), parent=self.root)

    def _toggle_overlay(self):
        self.preview_overlay = self.overlay_var.get()
        self._update_preview()

    def _on_close(self):
        self._alive = False
        self.root.destroy()

    def _update_preview(self):
        if not self.config or not self._alive:
            return
        try:
            if self.root.state() == "withdrawn":
                if self._alive:
                    self.root.after(50, self._update_preview)
                return
            canvas_w = self.preview_canvas.winfo_width()
            canvas_h = self.preview_canvas.winfo_height()
            if canvas_w <= 1 or canvas_h <= 1:
                if self._alive:
                    self.root.after(50, self._update_preview)
                return

            preview_cfg = update_configuration(self.config, {
                "camera.frame_count": 2,
                "droplets.count_per_frame": min(self.config.droplets.count_per_frame, 20),
                "background.noise_enabled": False,
                "motion.motion_blur_enabled": False,
                "appearance.defocus_enabled": False,
            }).configuration

            fresh, _ = prepare(preview_cfg)
            renderer = create_renderer(fresh)
            frame = next(generate_frames(fresh))
            img_array = encode_intensity(renderer.render(frame), fresh.camera.bit_depth)

            if fresh.camera.bit_depth == 16:
                img_array = (img_array.astype(np.float32) / 65535.0 * 255).astype(np.uint8)

            img = Image.fromarray(img_array, mode="L")

            if self.preview_overlay:
                img = self._draw_overlay(img, frame, fresh.camera)

            img.thumbnail((canvas_w - 10, canvas_h - 10), Image.Resampling.LANCZOS)
            self.preview_tk = ImageTk.PhotoImage(img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.preview_tk, anchor="center")

        except Exception as exc:
            self.status_var.set(f"Ошибка предпросмотра: {exc}")

    def _draw_overlay(self, img: Image.Image, frame, camera) -> Image.Image:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        scale_x = img.width / camera.width_px
        scale_y = img.height / camera.height_px

        for state in frame.droplets:
            d = state.droplet
            x = d.center_x_px * scale_x
            y = d.center_y_px * scale_y
            a = state.semi_major_px * scale_x
            b = state.semi_minor_px * scale_y
            angle = d.angle_deg

            if angle != 0:
                bbox = [x - a, y - b, x + a, y + b]
                draw.ellipse(bbox, outline=(255, 255, 0, 200), width=2)
            else:
                draw.ellipse([x - a, y - b, x + a, y + b], outline=(255, 255, 0, 200), width=2)

            draw.text((x + a + 2, y - b - 2), str(d.droplet_id), fill=(255, 255, 0, 255))

            if d.vx_m_s != 0 or d.vy_m_s != 0:
                dx = d.vx_m_s * camera.exposure_us / camera.scale_um_per_px * scale_x
                dy = d.vy_m_s * camera.exposure_us / camera.scale_um_per_px * scale_y
                draw.line([x, y, x + dx, y + dy], fill=(0, 255, 255, 200), width=2, arrow="last")

            if state.border:
                draw.rectangle([x - a - 2, y - b - 2, x + a + 2, y + b + 2], outline=(255, 0, 0, 200), width=2)

        base = img.convert("RGBA")
        return Image.alpha_composite(base, overlay).convert("L")

    def _start_generation(self):
        try:
            self._update_config_from_vars()
            if not self.export_dir_var.get():
                messagebox.showerror("Ошибка", "Выберите папку для экспорта", parent=self.root)
                return

            result = loads(dumps(self.config))
            if result.warnings:
                msg = "Предупреждения:\n" + "\n".join(result.warnings) + "\n\nПродолжить?"
                if not messagebox.askyesno("Предупреждения", msg, parent=self.root):
                    return

            export_dir = self.export_dir_var.get()
            final_config = update_configuration(self.config, {"export.output_directory": export_dir}).configuration

            self.start_btn.config(state="disabled")
            self.cancel_btn.config(state="normal")
            self.progress_var.set(0)
            self.progress_label.set("Подготовка...")
            self.time_label.set("")
            self.generation_start_time = time.time()
            self.generation_cancelled = False

            self.cancellation_token = CancellationToken()

            def progress_cb(event: ProgressEvent):
                self.root.after(0, lambda: self._update_progress(event))

            def completion_cb(event: CompletionEvent):
                self.root.after(0, lambda: self._on_completion(event))

            self.generation_thread = threading.Thread(
                target=lambda: run_generation(final_config, progress_cb, completion_cb, self.cancellation_token),
                daemon=True
            )
            self.generation_thread.start()

        except (ConfigurationError, OSError, UnicodeError) as exc:
            messagebox.showerror("Ошибка запуска", str(exc), parent=self.root)

    def _update_progress(self, event: ProgressEvent):
        if event.total_frames > 0:
            pct = event.completed_frames * 100 / event.total_frames
        else:
            pct = 0
        self.progress_var.set(pct)
        self.progress_label.set(f"Кадр {event.frame_number}/{event.total_frames} ({pct:.0f}%)")
        elapsed = time.time() - self.generation_start_time
        self.time_label.set(f"Время: {elapsed:.1f}с")

    def _on_completion(self, event: CompletionEvent):
        self.start_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")

        if event.status == "completed":
            self.progress_var.set(100)
            self.progress_label.set(f"Готово: {event.completed_frames}/{event.total_frames} кадров")
            elapsed = time.time() - self.generation_start_time
            self.time_label.set(f"Время: {elapsed:.1f}с")
            self.status_var.set(f"Серия сохранена: {event.directory}")
            messagebox.showinfo("Генерация завершена",
                              f"Успешно сохранено {event.completed_frames} кадров.\nПапка: {event.directory}",
                              parent=self.root)
        elif event.status == "cancelled":
            self.progress_label.set(f"Отменено: {event.completed_frames}/{event.total_frames} кадров")
            self.status_var.set(f"Генерация отменена. Частичная серия: {event.directory}")
            messagebox.showinfo("Генерация отменена",
                              f"Сохранено {event.completed_frames} из {event.total_frames} кадров.\nПапка: {event.directory}",
                              parent=self.root)
        elif event.status == "error":
            self.progress_label.set("Ошибка")
            self.status_var.set(f"Ошибка: {event.message}")
            messagebox.showerror("Ошибка генерации", event.message, parent=self.root)

    def _cancel_generation(self):
        if self.cancellation_token:
            self.cancellation_token.cancel()
            self.cancel_btn.config(state="disabled")
            self.progress_label.set("Отмена...")


def run() -> None:
    try:
        root = tk.Tk()
        SettingsWindow(root)
        root.mainloop()
    except tk.TclError as exc:
        raise InterfaceError(
            "Не удалось запустить окно настроек. Проверьте установку Python с Tcl/Tk "
            "и доступ к интерактивному рабочему столу. "
            "Проверить JSON без окна можно командой run.bat --check <файл>."
        ) from exc