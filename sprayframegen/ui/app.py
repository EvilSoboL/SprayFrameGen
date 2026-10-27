"""Полный интерфейс настроек, предпросмотра и экспорта."""

import copy
import json
import math
import threading
import time
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

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
from .parameter_units import (
    format_editable, format_reference, hz_to_interval_us, interval_us_to_hz,
    micrometers_to_pixels, parse_number, pixels_to_micrometers,
    speed_to_pixels_per_frame,
)


DIAMETER_PATHS = (
    "droplets.mean_um", "droplets.std_um", "droplets.median_um", "droplets.max_diameter_um",
)


RESOLUTIONS = [(640, 480), (1024, 768), (1280, 1024), (1920, 1080)]
FORMAT_BITDEPTH = [("PNG 8 бит", "png", 8), ("PNG 16 бит", "png", 16), ("TIFF 16 бит", "tiff", 16)]
SIZE_DISTRIBUTIONS = [("Логнормальное", "lognormal"), ("Нормальное", "normal")]
CONCENTRATIONS = [("Низкая", "low"), ("Средняя", "medium"), ("Высокая", "high")]
BACKGROUND_TONES = [("Светлый", "light"), ("Тёмный", "dark")]
RENDER_MODES = [("Реалистичный", "realistic"), ("Идеальный", "ideal")]


PARAMETER_HELP = {
    "camera.width_px": "Размер кадра в пикселях. Список содержит поддерживаемые сочетания ширины и высоты.",
    "camera.frame_count": "Сколько кадров войдёт в серию. Допустимо от 2 до 1000 кадров.",
    "camera.scale_um_per_px": "Физический размер одного пикселя в мкм/пиксель; должен быть больше нуля. Используется для пересчёта координат, диаметров и скорости. Например, при масштабе 10 диаметр 120 мкм равен 12 пикселям. Смена масштаба сохраняет физические размеры капель.",
    "camera.frame_interval_us": "Время между серединами выдержек соседних кадров. Период и частота должны быть больше нуля; частота (Гц) = 1 000 000 ÷ период (мкс). Например, 100 мкс = 10 000 Гц. Выдержка задаётся отдельно и не превышает период.",
    "camera.exposure_us": "Длительность накопления света в одном кадре, в микросекундах: больше нуля и не больше межкадрового интервала. Определяет длину следа при включённом размытии движения; при выключенном размытии не меняет изображение. Не задаёт время между кадрами.",
    "camera.format": "Формат и разрядность экспортируемых изображений: PNG 8 бит, PNG 16 бит или TIFF 16 бит. Разрядность задаёт число уровней яркости; предпросмотр показывает экранное 8-битное представление.",
    "interval_unit": "Выберите способ отображения периода между кадрами: микросекунды или частота в герцах. В конфигурации хранится период в микросекундах.",
    "droplets.count_per_frame": "Заданное число геометрически присутствующих капель в каждом кадре: целое от 1 до 10 000. Учитываются и полностью перекрытые капли. Чистый предпросмотр сейчас показывает не более 20 капель; экспорт использует заданное количество.",
    "droplets.size_distribution": "Закон распределения исходных эквивалентных диаметров. Для нормального закона используются среднее и стандартное отклонение, для логнормального — медиана и геометрическое стандартное отклонение.",
    "droplets.mean_um": "Среднее исходного нормального распределения диаметров до отбора по допустимому диапазону. Используется при выбранном нормальном законе; при логнормальном сохраняется, но не применяется.",
    "droplets.std_um": "Стандартное отклонение исходного нормального распределения диаметров: не меньше нуля. При нуле используется фиксированный средний диаметр, который должен попадать в допустимый диапазон. При логнормальном законе настройка сохраняется, но не применяется.",
    "droplets.median_um": "Медиана исходного логнормального распределения диаметров: больше нуля. Это параметр до отбора по допустимому диапазону. При нормальном законе сохраняется, но не применяется.",
    "droplets.geometric_std": "Безразмерное геометрическое стандартное отклонение логнормального распределения: не меньше 1. При 1 используется фиксированная медиана, которая должна попадать в допустимый диапазон. При нормальном законе настройка сохраняется, но не применяется.",
    "droplets.max_diameter_um": "Верхняя граница допустимого эквивалентного диаметра капли до размытия, а не длины следа движения. Нижняя граница зависит от масштаба и равна 3 пикселям.",
    "diameter_unit": "Выберите единицы отображения диаметров. При вводе в пикселях значение переводится в микрометры по масштабу; при смене масштаба физический диаметр сохраняется.",
    "droplets.concentration": "Низкая, средняя или высокая концентрация задаёт сгущение центров к середине кадра при начальном заполнении и к середине левой входной стороны при замене вышедших капель. Чем выше концентрация, тем уже распределение центров; количество капель задаётся отдельно.",
    "droplets.render_mode": "Идеальный режим рисует круглые капли, реалистичный — эллиптические с ориентацией и изменением формы по модели. Режим влияет на изображение; настройка расфокусировки задаётся отдельно.",
    "motion.speed_mean_m_s": "Средний модуль скорости исходного нормального распределения, в м/с. Это конечный параметр до отбора по диапазону, а не измеренное среднее по созданным каплям. При нулевом разбросе среднее должно попадать между минимумом и максимумом.",
    "motion.speed_std_m_s": "Стандартное отклонение исходного распределения модуля скорости, в м/с: не меньше нуля. При нуле скорость фиксирована и равна заданному среднему. Фактические скорости ограничиваются минимумом и максимумом.",
    "motion.speed_min_m_s": "Минимальный допустимый модуль скорости капли, в м/с: больше нуля. Выборка выполняется между минимумом и максимумом; обе границы включены.",
    "motion.speed_max_m_s": "Максимальный допустимый модуль скорости капли, в м/с: не меньше минимума. Равные границы допустимы при нулевом разбросе и среднем, равном этой границе.",
    "motion.direction_std_deg": "Стандартное отклонение направления основной группы относительно движения вправо, в градусах: не меньше нуля. Углы основной группы отбираются в пределах ±30°; редкие выбросы задаются моделью отдельно. Это не угол поворота эллипса.",
    "motion.motion_blur_enabled": "Включает усреднение движения капель в течение выдержки при генерации серии. Настройка выдержки остаётся отдельной. Чистый предпросмотр сейчас показывает капли без этого эффекта.",
    "motion.speed_reference": "Пересчёт заданного среднего в пикс/кадр: скорость (м/с) × интервал (мкс) ÷ масштаб (мкм/пиксель). Например, 8 × 100 ÷ 10 = 80 пикс/кадр. Это не статистика созданных капель; индивидуальные стрелки могут иметь другую длину. При ошибке черновика показано «—».",
    "appearance.defocus_enabled": "Включает гауссову расфокусировку в серии в обоих режимах независимо от размытия движения. При выключении границы сигмы сохраняются, но фактическая сигма равна нулю. Геометрия до размытия не меняется; чистый предпросмотр сейчас отключает этот эффект.",
    "appearance.blur_sigma_min_px": "Нижняя граница равномерного распределения сигмы гауссовой расфокусировки, в пикселях: не меньше нуля. Применяется при включённой расфокусировке; равные границы дают фиксированную сигму.",
    "appearance.blur_sigma_max_px": "Верхняя граница равномерного распределения сигмы гауссовой расфокусировки, в пикселях: не меньше нижней границы. Применяется при включённой расфокусировке; при обеих нулевых границах размытия нет.",
    "background.tone": "Задаёт однородный светлый или тёмный фон с нормированной яркостью 0,85 или 0,15 соответственно. Шум включается отдельно; полярность изображения капель подстраивается под фон.",
    "background.noise_enabled": "Включает добавление сенсорного шума к готовым кадрам серии. При выключении заданная интенсивность сохраняется. Чистый предпросмотр сейчас отключает шум.",
    "background.noise_sigma": "Стандартное отклонение сенсорного шума в нормированных единицах яркости, а не в отсчётах PNG: не меньше нуля. Нормированная яркость лежит от 0 до 1, но сигма не ограничена единицей. Нулевая сигма или выключенный шум означают отсутствие шума.",
    "seed": "Начальное состояние генератора случайных чисел: целое от 0 до 4 294 967 295. При одинаковых настройках, версии программы и окружении позволяет повторить результат; кнопка «Создать новый seed» выбирает случайное начальное состояние.",
    "export.output_directory": "Существующая папка с правом записи; требуется абсолютный путь. Каждый запуск создаёт в ней отдельную папку серии с кадрами, CSV и конфигурацией. Для предпросмотра выбирать папку не требуется.",
    "preview.overlay": "Жёлтые ID и контуры показывают геометрию до размытия. Бирюзовая стрелка — перемещение центра за межкадровый интервал (пикс/кадр), а не за выдержку. Красная рамка отмечает каплю, часть геометрической проекции которой вне кадра. Нулевая скорость — без стрелки; короче 1 экранного пикселя — точка; за краем стрелка обрезается.",
}


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
        self.labels: dict[str, ttk.Label] = {}
        self.float_paths: set[str] = set()
        self.float_bounds: dict[str, tuple[float, float]] = {}
        self._loading_vars = False
        self._rendered_text: dict[str, str] = {}
        self._draft_data = None
        self.interval_unit = "us"
        self.diameter_unit = "um"
        self._preview_is_stale = False
        self._tooltip_timer = None
        self._tooltip_window = None

        self._build_ui()
        self._reset_config("realistic")
        self._update_preview()

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.bind("<Destroy>", self._on_root_destroy, add="+")
        root.bind("<Configure>", self._on_root_configure, add="+")

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
        notebook.bind("<<NotebookTabChanged>>", lambda _event: self._hide_parameter_tooltip(), add="+")

        self._build_tab_general()
        self._build_tab_droplets()
        self._build_tab_motion()
        self._build_tab_appearance()
        self._build_tab_background()
        self._build_tab_random()

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Открыть…", command=self._open_file).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="Сохранить…", command=self._save_file).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="Проверить", command=self._check_config).pack(side="left", padx=3)

    def _build_tab_general(self):
        f = self.tab_general
        self._add_combobox(f, "camera.width_px", "Разрешение", 0,
                          [f"{w}×{h}" for w, h in RESOLUTIONS],
                          path2="camera.height_px")
        self._add_int(f, "camera.frame_count", "Число кадров", 1, 2, 1000)
        self.scale_label = ttk.Label(f, text="Масштаб (мкм/пиксель)")
        self.scale_label.grid(row=2, column=0, sticky="w", padx=5, pady=3)
        self._add_float(f, "camera.scale_um_per_px", "", 2, 0.1, 1000.0)
        self._bind_parameter_help("camera.scale_um_per_px", self.scale_label)
        self.interval_label = ttk.Label(f, text="Межкадровый интервал (мкс)")
        self.interval_label.grid(row=3, column=0, sticky="w", padx=5, pady=3)
        self._add_float(f, "camera.frame_interval_us", "", 3, 0.1, 100000.0)
        self._bind_parameter_help("camera.frame_interval_us", self.interval_label)
        self.interval_unit_var = tk.StringVar(value="мкс")
        self.interval_unit_combo = ttk.Combobox(f, textvariable=self.interval_unit_var,
            values=("мкс", "Гц"), state="readonly", width=7)
        self.interval_unit_combo.grid(row=3, column=2, padx=4)
        self.interval_unit_combo.bind("<<ComboboxSelected>>", self._change_interval_unit)
        self._bind_parameter_help("interval_unit", self.interval_unit_combo)
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
        self.diameter_unit_var = tk.StringVar(value="мкм")
        self.diameter_unit_combo = ttk.Combobox(f, textvariable=self.diameter_unit_var,
            values=("мкм", "пикс"), state="readonly", width=7)
        self.diameter_unit_combo.grid(row=9, column=2, padx=4)
        self.diameter_unit_combo.bind("<<ComboboxSelected>>", self._change_diameter_unit)
        self._bind_parameter_help("diameter_unit", self.diameter_unit_combo)
        self.diameter_help = ttk.Label(f, text="Эквивалентный диаметр до размытия. При смене масштаба размеры в мкм сохраняются.", wraplength=460)
        self.diameter_help.grid(row=10, column=0, columnspan=3, sticky="w", padx=5, pady=3)
        self._add_combobox(f, "droplets.concentration", "Концентрация", 7,
                          [label for label, _ in CONCENTRATIONS],
                          lambda v: CONCENTRATIONS[v.current()][1])
        self._add_combobox(f, "droplets.render_mode", "Режим рендера", 8,
                          [label for label, _ in RENDER_MODES],
                          lambda v: RENDER_MODES[v.current()][1])

    def _build_tab_motion(self):
        f = self.tab_motion
        self._add_float(f, "motion.speed_mean_m_s", "Средняя скорость (м/с)", 0, 0.1, 100.0)
        self.speed_pixels_var = tk.StringVar(value="—")
        speed_reference_label = ttk.Label(f, text="Средняя скорость (пикс/кадр)")
        speed_reference_label.grid(row=6, column=0, sticky="w", padx=5, pady=3)
        speed_reference_value = ttk.Label(f, textvariable=self.speed_pixels_var)
        speed_reference_value.grid(row=6, column=1, sticky="w", padx=5, pady=3)
        self._bind_parameter_help("motion.speed_reference", speed_reference_label, speed_reference_value)
        ttk.Label(f, text="Пересчёт заданной средней скорости").grid(row=7, column=0, columnspan=2, sticky="w", padx=5)
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
        self.overlay_checkbutton = ttk.Checkbutton(
            overlay_frame, text="Показать служебный слой (ID, контуры, векторы, границы)",
            variable=self.overlay_var, command=self._toggle_overlay)
        self.overlay_checkbutton.pack(anchor="w")
        self._bind_parameter_help("preview.overlay", self.overlay_checkbutton)

        self.overlay_legend = ttk.Frame(overlay_frame)
        legend_items = (
            ((255, 255, 0), "ID и контур — геометрия капли до размытия"),
            ((0, 255, 255), "Стрелка — перемещение за межкадровый интервал (пикс/кадр)"),
            ((255, 0, 0), "Рамка — граничная капля (часть контура за кадром)"),
        )
        for color, description in legend_items:
            item_frame = ttk.Frame(self.overlay_legend)
            item_frame.pack(fill="x", anchor="w")
            swatch = tk.Canvas(item_frame, width=18, height=14,
                               highlightthickness=0, takefocus=False)
            swatch.pack(side="left", padx=(0, 4), pady=1)
            swatch.create_line(2, 7, 16, 7, fill="#{:02x}{:02x}{:02x}".format(*color), width=3)
            ttk.Label(item_frame, text=description, wraplength=420,
                      justify="left").pack(side="left", fill="x", expand=True)
        ttk.Label(
            self.overlay_legend,
            text="Нулевая скорость — без стрелки; короче 1 экранного пикселя — точка; "
                 "за краем стрелка обрезается. Служебный слой не экспортируется.",
            wraplength=420, justify="left",
        ).pack(anchor="w", pady=(2, 0))

        export_frame = ttk.LabelFrame(top_frame, text="Экспорт", padding=5)
        export_frame.pack(fill="x", pady=5)

        dir_frame = ttk.Frame(export_frame)
        dir_frame.pack(fill="x", pady=2)
        ttk.Label(dir_frame, text="Папка:").pack(side="left")
        self.export_dir_var = tk.StringVar()
        self.vars["export.output_directory"] = self.export_dir_var
        self.export_dir_entry = ttk.Entry(dir_frame, textvariable=self.export_dir_var)
        self.export_dir_entry.pack(side="left", fill="x", expand=True, padx=5)
        self._bind_parameter_help("export.output_directory", self.export_dir_entry)
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
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=5, pady=3)
        spin = ttk.Spinbox(parent, from_=min_val, to=max_val, textvariable=var, width=20)
        spin.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        self._bind_parameter_help(path, label_widget, spin)
        var.trace_add("write", lambda *_: self._on_field_change(path))
        parent.columnconfigure(1, weight=1)

    def _add_float(self, parent, path, label, row, min_val, max_val):
        var = tk.StringVar()
        self.vars[path] = var
        self.float_paths.add(path)
        self.float_bounds[path] = (min_val, max_val)
        label_widget = ttk.Label(parent, text=label)
        if label:
            label_widget.grid(row=row, column=0, sticky="w", padx=5, pady=3)
        spin = ttk.Spinbox(parent, from_=min_val, to=max_val, textvariable=var, width=20, increment=0.1)
        self.widgets[path] = spin
        self.labels[path] = label_widget
        spin.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        self._bind_parameter_help(path, label_widget if label else None, spin)
        var.trace_add("write", lambda *_: self._on_field_change(path))
        parent.columnconfigure(1, weight=1)

    def _add_combobox(self, parent, path, label, row, values, getter=None, path2=None):
        var = tk.StringVar()
        self.vars[path] = var
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=5, pady=3)
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=18)
        combo.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        combo.current(0)
        self._bind_parameter_help(path, label_widget, combo)

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
        checkbutton = ttk.Checkbutton(parent, text=label, variable=var,
                       command=lambda: self._on_field_change(path))
        checkbutton.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=3)
        self._bind_parameter_help(path, checkbutton)

    def _bind_parameter_help(self, key: str, *widgets: tk.Widget | None):
        for widget in widgets:
            if widget is None:
                continue
            widget.bind("<Enter>", lambda event, help_key=key: self._schedule_parameter_tooltip(event, help_key), add="+")
            widget.bind("<Leave>", lambda _event: self._hide_parameter_tooltip(), add="+")

    def _parameter_help_text(self, key: str) -> str:
        text = PARAMETER_HELP.get(key)
        if text is None:
            return ""
        if key == "camera.frame_interval_us":
            unit = "частота кадров в герцах" if self.interval_unit == "hz" else "период в микросекундах"
            text += f" Сейчас поле показывает {unit}."
        elif key == "interval_unit":
            selected = "герцы" if self.interval_unit == "hz" else "микросекунды"
            text += f" Сейчас выбраны {selected}."
        elif key == "diameter_unit":
            selected = "пиксели" if self.diameter_unit == "px" else "микрометры"
            text += f" Сейчас выбраны {selected}."
        elif key in DIAMETER_PATHS:
            unit = "пикселях" if self.diameter_unit == "px" else "микрометрах"
            text += f" Сейчас значение в {unit}; размер относится к эквивалентному диаметру до размытия, не к следу движения."
        if key in ("droplets.mean_um", "droplets.std_um", "droplets.median_um", "droplets.geometric_std") and self._draft_data:
            distribution = self._draft_data["droplets"]["size_distribution"]
            applies = distribution == ("normal" if key in ("droplets.mean_um", "droplets.std_um") else "lognormal")
            text += " Сейчас применяется." if applies else " Сейчас не применяется: выбран другой закон."
        if key == "droplets.render_mode" and self._draft_data:
            mode = self._draft_data["droplets"]["render_mode"]
            text += f" Сейчас выбран режим: {'реалистичный' if mode == 'realistic' else 'идеальный'}."
        elif key == "preview.overlay":
            text += " Слой служит для проверки, не изменяет модель и данные экспорта и не входит в экспортируемые кадры."
        return text

    def _schedule_parameter_tooltip(self, event, key: str):
        self._hide_parameter_tooltip()
        if not PARAMETER_HELP.get(key):
            return
        try:
            self._tooltip_timer = self.root.after(
                450, lambda: self._show_parameter_tooltip(event.widget, key))
        except tk.TclError:
            self._tooltip_timer = None

    def _show_parameter_tooltip(self, widget: tk.Widget, key: str):
        self._tooltip_timer = None
        if not self._alive or not self.root.winfo_exists() or not widget.winfo_exists():
            return
        text = self._parameter_help_text(key)
        if not text:
            return
        try:
            tooltip = tk.Toplevel(self.root)
            tooltip.wm_overrideredirect(True)
            label = tk.Label(tooltip, text=text, justify="left", anchor="w", wraplength=360,
                             background="#fffbe6", foreground="#202020", relief="solid", borderwidth=1,
                             padx=8, pady=6)
            label.pack()
            tooltip.update_idletasks()
            width, height = tooltip.winfo_reqwidth(), tooltip.winfo_reqheight()
            screen_width, screen_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            target_x, target_y = widget.winfo_rootx(), widget.winfo_rooty()
            root_x = self.root.winfo_rootx()
            panel_split = root_x + self.root.winfo_width() // 2
            if target_x < panel_split:
                # Keep field help clear of the adjacent input controls.
                x, y = panel_split + 10, target_y
            else:
                # The right panel has no editable fields beside its labels.
                x, y = target_x, target_y - height - 8
            x = max(0, min(x, screen_width - width))
            y = max(0, min(y, screen_height - height))
            tooltip.geometry(f"+{x}+{y}")
            self._tooltip_window = tooltip
        except tk.TclError:
            self._tooltip_window = None

    def _hide_parameter_tooltip(self):
        if self._tooltip_timer is not None:
            try:
                self.root.after_cancel(self._tooltip_timer)
            except tk.TclError:
                pass
            self._tooltip_timer = None
        if self._tooltip_window is not None:
            try:
                if self._tooltip_window.winfo_exists():
                    self._tooltip_window.destroy()
            except tk.TclError:
                pass
            self._tooltip_window = None

    def _on_root_destroy(self, event):
        if event.widget is self.root:
            self._hide_parameter_tooltip()

    def _on_root_configure(self, event):
        if event.widget is self.root:
            self._hide_parameter_tooltip()

    def _on_field_change(self, path: str):
        if self._loading_vars:
            return
        try:
            self._update_config_from_vars(changed_path=path)
            self._clear_preview_error()
            self._update_preview()
            self._update_reference_speed()
        except (ConfigurationError, ValueError, OverflowError) as exc:
            self._set_draft_error(path, exc)

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
        self._draft_data = asdict(self.config)
        self._vars_from_config()
        self._update_preview()

    def _vars_from_config(self):
        if not self.config:
            return
        self._loading_vars = True
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
                elif path in self.float_paths:
                    display = self._display_float(path, value, data["camera"]["scale_um_per_px"])
                    text = format_editable(display)
                    var.set(text)
                    self._rendered_text[path] = text
                elif isinstance(var, tk.StringVar):
                    if path == "export.output_directory":
                        var.set("" if value is None else str(value))
                    elif path == "camera.width_px":
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
        self._loading_vars = False
        if hasattr(self, "interval_unit_var"):
            self.interval_unit_var.set("Гц" if self.interval_unit == "hz" else "мкс")
            self.diameter_unit_var.set("пикс" if self.diameter_unit == "px" else "мкм")
        self._clear_preview_error()
        self._update_unit_labels()
        self._update_reference_speed()

    def _update_config_from_vars(self, changed_path: str | None = None, *, validate: bool = True):
        if not self.config:
            return
        data = self._draft_data.copy() if self._draft_data else asdict(self.config)
        # Nested structures are copied before edits so a failed candidate is atomic.
        data = copy.deepcopy(data)
        rendered_text = dict(self._rendered_text)
        for path, var in self.vars.items():
            value = var.get()
            parts = path.split(".")
            target = data
            for part in parts[:-1]:
                target = target[part]
            if path in self.float_paths:
                text = str(value)
                if not text.strip():
                    raise ValueError(f"{path}: введите число")
                if text != self._rendered_text.get(path):
                    parsed = parse_number(text, path)
                    if path == "camera.frame_interval_us" and self.interval_unit == "hz":
                        parsed = hz_to_interval_us(parsed)
                    elif path in DIAMETER_PATHS and self.diameter_unit == "px":
                        # The scale text is interpreted after diameter drafts, against the prior scale.
                        scale = data["camera"]["scale_um_per_px"]
                        if changed_path == "camera.scale_um_per_px":
                            scale = self._draft_data["camera"]["scale_um_per_px"]
                        parsed = pixels_to_micrometers(parsed, scale)
                    target[parts[-1]] = parsed
                    rendered_text[path] = text
            elif isinstance(var, tk.BooleanVar):
                target[parts[-1]] = bool(value)
            elif isinstance(var, tk.IntVar):
                target[parts[-1]] = int(value)
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
                    elif path in self.float_paths:
                        pass
                    elif path == "export.output_directory":
                        target[parts[-1]] = value or None
                    else:
                        target[parts[-1]] = value
        result = from_mapping(data) if validate else None
        if result:
            # A valid physical configuration must also fit the selected UI units.
            # Reject overflow before committing or suppressing field callbacks.
            for field in self.float_paths:
                value = data
                for part in field.split("."):
                    value = value[part]
                self._display_float(field, value, data["camera"]["scale_um_per_px"])
        scale_changed = result is not None and (
            result.configuration.camera.scale_um_per_px != self.config.camera.scale_um_per_px
        )
        if result:
            self.config = result.configuration
        self._draft_data = data
        self._rendered_text = rendered_text
        if scale_changed:
            # A different field can make a previously invalid scale acceptable.
            # Refresh its dependent views on every successful commit path.
            self._render_float_draft()
        if result:
            for w in result.warnings:
                self.status_var.set(f"Предупреждение: {w}")

    def _display_float(self, path, value, scale):
        display = float(value)
        if path == "camera.frame_interval_us" and self.interval_unit == "hz":
            return interval_us_to_hz(display)
        if path in DIAMETER_PATHS and self.diameter_unit == "px":
            return micrometers_to_pixels(display, scale)
        return display

    def _render_float_draft(self):
        if not self._draft_data:
            return
        self._loading_vars = True
        for path in self.float_paths:
            value = self._draft_data
            for part in path.split("."):
                value = value[part]
            display = self._display_float(path, value, self._draft_data["camera"]["scale_um_per_px"])
            rendered = format_editable(display)
            self.vars[path].set(rendered)
            self._rendered_text[path] = rendered
        self._loading_vars = False
        self._update_unit_labels()

    def _update_unit_labels(self):
        interval = "Частота кадров (Гц)" if self.interval_unit == "hz" else "Межкадровый интервал (мкс)"
        self.interval_label.configure(text=interval)
        interval_min, interval_max = self.float_bounds["camera.frame_interval_us"]
        if self.interval_unit == "hz":
            self.widgets["camera.frame_interval_us"].configure(
                from_=1_000_000 / interval_max, to=1_000_000 / interval_min)
        else:
            self.widgets["camera.frame_interval_us"].configure(from_=interval_min, to=interval_max)
        unit = "пикс" if self.diameter_unit == "px" else "мкм"
        names = {
            "droplets.mean_um": f"Среднее ({unit}, для нормального)",
            "droplets.std_um": f"Стд. отклонение ({unit}, для нормального)",
            "droplets.median_um": f"Медиана ({unit}, для логнормального)",
            "droplets.max_diameter_um": f"Макс. диаметр ({unit})",
        }
        for path, label in names.items():
            self.labels[path].configure(text=label)
            low, high = self.float_bounds[path]
            if self.diameter_unit == "px":
                scale = self._draft_data["camera"]["scale_um_per_px"] if self._draft_data else self.config.camera.scale_um_per_px
                low, high = low / scale, high / scale
            self.widgets[path].configure(from_=low, to=high)

    def _update_reference_speed(self):
        try:
            if self._draft_data is None or not self.config:
                self.speed_pixels_var.set("—")
                return
            value = speed_to_pixels_per_frame(
                self._draft_data["motion"]["speed_mean_m_s"],
                self._draft_data["camera"]["frame_interval_us"],
                self._draft_data["camera"]["scale_um_per_px"],
            )
            self.speed_pixels_var.set(format_reference(value))
        except (ValueError, OverflowError):
            self.speed_pixels_var.set("—")

    def _set_draft_error(self, path: str, exc: Exception):
        self.speed_pixels_var.set("—")
        message = str(exc)
        matched_path = next((item for item in self.float_paths if item in message), None)
        if matched_path:
            path = matched_path
        field = self.labels[path].cget("text") if path in self.labels else path
        if path == "camera.frame_interval_us":
            field = self.interval_label.cget("text")
        elif path == "camera.scale_um_per_px":
            field = self.scale_label.cget("text")
        elif path == "droplets":
            field = "Параметры капель"
        self.status_var.set(f"Ошибка черновика — {field}: {message}")
        if not self._preview_is_stale:
            self._preview_is_stale = True
            self.preview_canvas.create_text(10, 10, anchor="nw",
                text="Предпросмотр по последним корректным настройкам", fill="yellow", tags="draft-error")

    def _clear_preview_error(self):
        self._preview_is_stale = False
        if hasattr(self, "preview_canvas"):
            self.preview_canvas.delete("draft-error")
        if hasattr(self, "status_var"):
            self.status_var.set("Готово")

    def _change_interval_unit(self, _event=None):
        requested = "hz" if self.interval_unit_var.get() == "Гц" else "us"
        if requested == self.interval_unit:
            return
        previous = self.interval_unit
        try:
            self._update_config_from_vars(validate=False)
            self.interval_unit = requested
            self._render_float_draft()
            try:
                self._update_config_from_vars()
                self._clear_preview_error()
                self._update_preview()
            except (ConfigurationError, ValueError, OverflowError) as exc:
                self._set_draft_error("camera.frame_interval_us", exc)
        except (ConfigurationError, ValueError, OverflowError) as exc:
            changed = self.interval_unit != previous
            self.interval_unit = previous
            self.interval_unit_var.set("Гц" if previous == "hz" else "мкс")
            if changed and self._draft_data:
                self._render_float_draft()
            self._set_draft_error("camera.frame_interval_us", exc)

    def _change_diameter_unit(self, _event=None):
        requested = "px" if self.diameter_unit_var.get() == "пикс" else "um"
        if requested == self.diameter_unit:
            return
        previous = self.diameter_unit
        try:
            self._update_config_from_vars(validate=False)
            self.diameter_unit = requested
            self._render_float_draft()
            try:
                self._update_config_from_vars()
                self._clear_preview_error()
                self._update_preview()
            except (ConfigurationError, ValueError, OverflowError) as exc:
                self._set_draft_error("droplets", exc)
        except (ConfigurationError, ValueError, OverflowError) as exc:
            changed = self.diameter_unit != previous
            self.diameter_unit = previous
            self.diameter_unit_var.set("пикс" if previous == "px" else "мкм")
            if changed and self._draft_data:
                self._render_float_draft()
            self._set_draft_error("droplets", exc)

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
                data = asdict(result.configuration)
                # Check the selected representation before touching any live state.
                # Even valid physical values can overflow in pixels or Hz.
                for field in self.float_paths:
                    value = data
                    for part in field.split("."):
                        value = value[part]
                    self._display_float(field, value, data["camera"]["scale_um_per_px"])
            except (ConfigurationError, OSError, UnicodeError, ValueError, OverflowError) as exc:
                messagebox.showerror("Ошибка загрузки", str(exc), parent=self.root)
                return
            self.config = result.configuration
            self._draft_data = data
            self._vars_from_config()
            self._update_preview()
            self.status_var.set("Настройки загружены. " + " ".join(result.warnings))
            if result.warnings:
                messagebox.showwarning("Предупреждения", "\n".join(result.warnings), parent=self.root)

    def _save_file(self):
        try:
            self._update_config_from_vars()
            path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".json",
                                              filetypes=[("Настройки JSON", "*.json")])
            if path:
                save(self.config, path)
                self.status_var.set(f"Настройки сохранены: {path}")
        except (ConfigurationError, OSError, UnicodeError, ValueError, OverflowError) as exc:
            self._set_draft_error("camera.frame_interval_us", exc)
            messagebox.showerror("Ошибка сохранения", self.status_var.get(), parent=self.root)

    def _check_config(self):
        try:
            self._update_config_from_vars()
            result = loads(dumps(self.config))
            msg = "Конфигурация корректна."
            if result.warnings:
                msg += "\nПредупреждения:\n" + "\n".join(result.warnings)
            messagebox.showinfo("Проверка", msg, parent=self.root)
            self.status_var.set("Настройки проверены. " + " ".join(result.warnings))
        except (ConfigurationError, OSError, UnicodeError, ValueError, OverflowError) as exc:
            self._set_draft_error("camera.frame_interval_us", exc)
            messagebox.showerror("Ошибка настроек", self.status_var.get(), parent=self.root)

    def _toggle_overlay(self):
        self.preview_overlay = self.overlay_var.get()
        if self.preview_overlay:
            self.overlay_legend.pack(fill="x", anchor="w", padx=(18, 0), pady=(2, 0))
        else:
            self.overlay_legend.pack_forget()
        self._update_preview()

    def _on_close(self):
        self._alive = False
        self._hide_parameter_tooltip()
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

            img.thumbnail((canvas_w - 10, canvas_h - 10), Image.Resampling.LANCZOS)
            if self.preview_overlay:
                img = self._draw_overlay(img, frame, fresh.camera)
            self.preview_tk = ImageTk.PhotoImage(img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.preview_tk, anchor="center")

        except Exception as exc:
            self.status_var.set(f"Ошибка предпросмотра: {exc}")

    def _draw_overlay(self, img: Image.Image, frame, camera) -> Image.Image:
        """Добавляет цветные диагностические обозначения только к экранному кадру."""
        result = img.convert("RGB")
        draw = ImageDraw.Draw(result)
        scale_x = img.width / camera.width_px
        scale_y = img.height / camera.height_px
        yellow = (255, 255, 0)
        cyan = (0, 255, 255)
        red = (255, 0, 0)
        font = ImageFont.load_default(size=12)

        for state in frame.droplets:
            d = state.droplet
            cx, cy = d.center_x_px, d.center_y_px
            a, b = state.semi_major_px, state.semi_minor_px
            theta = math.radians(d.angle_deg)
            cosine, sine = math.cos(theta), math.sin(theta)

            # Bound the chord error after independent X/Y preview scaling.
            max_radius = max(a, b) * max(scale_x, scale_y)
            if max_radius <= 0.5:
                segments = 12
            else:
                angle_step = math.acos(max(-1.0, 1.0 - 0.5 / max_radius))
                segments = max(12, math.ceil(math.pi / angle_step))
            contour = []
            for index in range(segments + 1):
                angle = 2 * math.pi * index / segments
                px = cx + a * math.cos(angle) * cosine - b * math.sin(angle) * sine
                py = cy + a * math.cos(angle) * sine + b * math.sin(angle) * cosine
                contour.append((px * scale_x, py * scale_y))
            draw.line(contour, fill=yellow, width=2)

            half_width = math.hypot(a * cosine, b * sine)
            half_height = math.hypot(a * sine, b * cosine)
            left, right = (cx - half_width) * scale_x, (cx + half_width) * scale_x
            top, bottom = (cy - half_height) * scale_y, (cy + half_height) * scale_y

            label = str(d.droplet_id)
            text_box = draw.textbbox((0, 0), label, font=font, stroke_width=2)
            text_width, text_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
            label_x = right + 3
            label_y = top - text_height - 4
            label_x = min(max(0, label_x), max(0, img.width - text_width))
            label_y = min(max(0, label_y), max(0, img.height - text_height))
            draw.text((label_x, label_y), label, font=font, fill=yellow,
                      stroke_width=2, stroke_fill=(0, 0, 0))

            if state.border:
                draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), outline=red, width=2)

            dx = d.vx_m_s * camera.frame_interval_us / camera.scale_um_per_px
            dy = d.vy_m_s * camera.frame_interval_us / camera.scale_um_per_px
            start = (cx * scale_x, cy * scale_y)
            end = ((cx + dx) * scale_x, (cy + dy) * scale_y)
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            if length == 0:
                continue
            if length < 1:
                px, py = end
                draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=cyan)
                continue

            ux = (end[0] - start[0]) / length
            uy = (end[1] - start[1]) / length
            head_length = min(6.0, length / 3)
            half_head_width = min(3.0, length / 6)
            base_x, base_y = end[0] - ux * head_length, end[1] - uy * head_length
            perpendicular = (-uy * half_head_width, ux * half_head_width)
            triangle = (
                end,
                (base_x + perpendicular[0], base_y + perpendicular[1]),
                (base_x - perpendicular[0], base_y - perpendicular[1]),
            )
            draw.line((start, end), fill=cyan, width=2)
            # Pillow clips the triangle to the image; don't move an external tip to the edge.
            if 0 <= end[0] < img.width and 0 <= end[1] < img.height:
                draw.polygon(triangle, fill=cyan)

        return result

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

        except (ConfigurationError, OSError, UnicodeError, ValueError, OverflowError) as exc:
            self._set_draft_error("camera.frame_interval_us", exc)
            messagebox.showerror("Ошибка запуска", self.status_var.get(), parent=self.root)

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
