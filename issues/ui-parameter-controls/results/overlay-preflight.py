r"""P01: диагностика исходного слоя и проверка процедуры передачи, не приёмка T01.

Запуск из корня: .venv\Scripts\python.exe issues/ui-parameter-controls/results/overlay-preflight.py
Ошибки слоя печатаются как наблюдения; нулевой exit code не означает прохождение AC.
"""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import tkinter as tk

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import PIL
from PIL import Image, ImageDraw

from sprayframegen.configuration import Camera, dumps, new_configuration, update_configuration
from sprayframegen.generation.controller import run_generation
from sprayframegen.generation.sequence import Frame, FrameDroplet, generate_frames
from sprayframegen.geometry.projections import Ellipse
from sprayframegen.model import Droplet
from sprayframegen.render import create_renderer, encode_intensity
from sprayframegen.ui.app import SettingsWindow
from sprayframegen.ui.parameter_units import speed_to_pixels_per_frame

sys.stdout.reconfigure(encoding="utf-8")


def emit(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


emit(python=sys.version.split()[0], numpy=np.__version__, pillow=PIL.__version__)
for name in (
    "sprayframegen/ui/app.py", "sprayframegen/model/__init__.py",
    "sprayframegen/generation/sequence.py", "sprayframegen/geometry/projections.py",
    "sprayframegen/export/series.py", "sprayframegen/generation/controller.py",
    "sprayframegen/render/optical.py",
    "sprayframegen/ui/parameter_units.py", "tests/ui_smoke.py",
):
    emit(file=name, sha256=hashlib.sha256(Path(name).read_bytes()).hexdigest())

try:
    ImageDraw.Draw(Image.new("RGB", (10, 10))).line((1, 1, 5, 5), arrow="last")
except TypeError as exc:
    emit(pillow_arrow=str(exc))

camera = Camera()
droplet = Droplet(42, 100, 100, 200, 20, False, 0, 2, 2, 0, 1.35, 45, 0)
ellipse = Ellipse.from_droplet(droplet, camera.width_px)
state = FrameDroplet(droplet, ellipse.a, ellipse.b, True, False, False)
for name, vx, vy in (("nonzero", 2, 0), ("zero", 0, 0)):
    current = replace(state, droplet=replace(droplet, vx_m_s=vx, vy_m_s=vy))
    base = Image.new("L", (640, 480), 128)
    before = base.tobytes()
    try:
        result = SettingsWindow._draw_overlay(None, base, Frame(1, 0, (current,)), camera)
        emit(draw_case=name, mode=result.mode, size=result.size, input_unchanged=base.tobytes() == before)
    except Exception as exc:
        emit(draw_case=name, error=f"{type(exc).__name__}: {exc}", input_unchanged=base.tobytes() == before)

for vx, vy, dt, scale, expected in (
    (2, 0, 100, 10, (20, 0)), (2, -1, 100, 10, (20, -10)),
    (2, 0, 200, 10, (40, 0)), (2, 0, 100, 20, (10, 0)),
    (0, 0, 100, 10, (0, 0)), (0.05, 0, 100, 10, (0.5, 0)),
    (100, 0, 100, 10, (1000, 0)),
):
    actual = tuple(speed_to_pixels_per_frame(v, dt, scale) for v in (vx, vy))
    assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(actual, expected))
    emit(vector_input=[vx, vy, dt, scale], displacement=actual)

emit(ellipse_case="model at x=100, q_left=1.35, r=10, angle=45", a=ellipse.a,
     b=ellipse.b, bounds=ellipse.bounds)

root = tk.Tk()
try:
    emit(tcl_tk=root.tk.call("info", "patchlevel"))
    root.geometry("1000x700")
    app = SettingsWindow(root)
    root.update()
    with TemporaryDirectory(prefix="sprayframe-overlay-p01-") as directory:
        for mode in ("ideal", "realistic"):
            config = update_configuration(new_configuration(seed=42, render_mode=mode), {
                "camera.frame_count": 2, "droplets.count_per_frame": 10,
                "camera.bit_depth": 8, "camera.format": "png",
                "motion.motion_blur_enabled": False, "appearance.defocus_enabled": False,
                "background.noise_enabled": False,
                "export.output_directory": str(Path(directory).resolve()),
            }).configuration
            app.config = config
            app._vars_from_config()
            app.overlay_var.set(False)
            app._toggle_overlay()
            root.update()
            serialized = dumps(app.config)
            model_before = tuple(generate_frames(app.config))
            before = run_generation(app.config)
            observations = []
            for _ in range(3):
                app.status_var.set("Готово")
                app.overlay_checkbutton.invoke()
                root.update()
                observations.append({"enabled": app.preview_overlay, "status": app.status_var.get()})
                assert app.preview_overlay
                app.overlay_checkbutton.invoke()
                root.update()
                assert not app.preview_overlay
            assert dumps(app.config) == serialized
            assert tuple(generate_frames(app.config)) == model_before
            after = run_generation(app.config)
            files_before = sorted((before / "frames").iterdir())
            files_after = sorted((after / "frames").iterdir())
            assert len(files_before) == len(files_after) == 2
            renderer = create_renderer(config)
            for first, second, frame in zip(files_before, files_after, model_before):
                with Image.open(first) as a, Image.open(second) as b:
                    expected = encode_intensity(renderer.render(frame), config.camera.bit_depth)
                    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
                    np.testing.assert_array_equal(np.asarray(a), expected)
                    assert len(a.getbands()) == len(b.getbands()) == 1
            assert (before / "droplets.csv").read_bytes() == (after / "droplets.csv").read_bytes()
            emit(mode=mode, toggles=observations, config_equal=True, frames_equal=True,
                 exported_pixels_equal=True, exports_match_clean_renderer=True, csv_equal=True,
                 note="Изоляция исходного кода при ошибке включения; AC01 и приёмка T01 не подтверждены")
finally:
    root.destroy()
