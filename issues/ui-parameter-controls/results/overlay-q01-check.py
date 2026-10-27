"""Независимые проверки Sol Q01; --ui открывает подготовленное окно для осмотра."""

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from sprayframegen.configuration import Camera, dumps, new_configuration, update_configuration
from sprayframegen.generation import generate_frames
from sprayframegen.generation.controller import run_generation
from sprayframegen.generation.sequence import Frame, FrameDroplet
from sprayframegen.model import Droplet
from sprayframegen.render import create_renderer, encode_intensity
from sprayframegen.ui.app import SettingsWindow


def config(mode, directory, **changes):
    return update_configuration(new_configuration(seed=42, render_mode=mode), {
        "camera.frame_count": 2, "droplets.count_per_frame": 3,
        "motion.speed_mean_m_s": 2, "motion.speed_std_m_s": 0,
        "motion.speed_min_m_s": 1, "motion.speed_max_m_s": 3,
        "motion.motion_blur_enabled": True, "appearance.defocus_enabled": True,
        "background.noise_enabled": True,
        "export.output_directory": str(Path(directory).resolve()), **changes,
    }).configuration


def emit(**data):
    print(json.dumps(data, ensure_ascii=False), flush=True)


if "--ui" in sys.argv:
    directory = tempfile.mkdtemp(prefix="SprayFrameGen-Q01-UI-")
    root = tk.Tk()
    app = SettingsWindow(root)
    root.title("SprayFrameGen — Sol Q01")
    root.geometry("1000x700")
    app.config = config("realistic", directory, **{"droplets.count_per_frame": 10})
    app._draft_data = asdict(app.config)
    app._vars_from_config()
    app._update_preview()
    emit(ui_directory=directory)
    root.mainloop()
    sys.exit(0)

if "--samples" in sys.argv:
    samples = []
    for identifier, x, y, radius, angle, vx, vy in (
        (1,120,130,45,0,2,-1), (2,320,130,45,45,2,-1), (3,520,130,45,90,2,-1),
        (4,5,310,30,45,2,0), (5,170,310,25,0,0,0),
        (6,320,310,25,0,.05,0), (7,440,400,25,0,100,0),
    ):
        d = Droplet(identifier,x,y,radius*20,radius*2,False,0,math.hypot(vx,vy),vx,vy,1.35,angle,0)
        a,b = d.semi_axes_at(x,640)
        samples.append(FrameDroplet(d,a,b,False,identifier==4,False))
    sheet = Image.new("RGB",(1280,480))
    for x,tone in ((0,217),(640,38)):
        base = Image.new("L",(640,480),tone)
        layer = SettingsWindow._draw_overlay(None,base,Frame(1,0,tuple(samples)),Camera())
        canvas = ImageDraw.Draw(layer)
        canvas.text((15,15),"1: 0 deg   2: 45 deg   3: 90 deg   4: border",fill="black" if tone==217 else "white")
        canvas.text((15,35),"5: zero   6: <1 px   7: clipped long vector",fill="black" if tone==217 else "white")
        sheet.paste(layer,(x,0))
    target = Path(tempfile.gettempdir())/"SprayFrameGen-Q01-overlay-cases.png"
    sheet.save(target)
    emit(samples_path=str(target))
    sys.exit(0)

for file in ("sprayframegen/ui/app.py", "tests/test_preview_overlay.py", "tests/ui_smoke.py",
             "issues/ui-parameter-controls/results/overlay-contract.md"):
    emit(file=file, sha256=hashlib.sha256(Path(file).read_bytes()).hexdigest())


def draw(state, camera, size):
    return SettingsWindow._draw_overlay(None, Image.new("L", size, 80), Frame(1, 0, (state,)), camera)


# Проверяем всю ломаную через обратный поворот в систему эллипса,
# а не вычисляем только первую вершину по формуле реализации.
contour_cases = 0
for angle in (0, 30, 45, 90, 135):
    for size in ((640, 480), (320, 120)):
        d = Droplet(42, 100, 100, 200, 20, False, 0, 0, 0, 0, 1.35, angle, 0)
        a, b = d.semi_axes_at(100, 640)
        state = FrameDroplet(d, a, b, False, True, False)
        lines, rectangles = [], []
        line, rectangle = ImageDraw.ImageDraw.line, ImageDraw.ImageDraw.rectangle

        def capture_line(ctx, xy, *args, **kwargs):
            lines.append(xy)
            return line(ctx, xy, *args, **kwargs)

        def capture_rectangle(ctx, xy, *args, **kwargs):
            rectangles.append(xy)
            return rectangle(ctx, xy, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, "line", capture_line), \
             patch.object(ImageDraw.ImageDraw, "rectangle", capture_rectangle):
            output = draw(state, Camera(), size)
        points = np.array(lines[0])
        local = points / np.array([size[0] / 640, size[1] / 480]) - [100, 100]
        theta = math.radians(angle)
        basis = np.array([[math.cos(theta), -math.sin(theta)],
                          [math.sin(theta), math.cos(theta)]])
        normalized = (local @ basis) / [a, b]
        np.testing.assert_allclose(np.sum(normalized ** 2, axis=1), 1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(points[0], points[-1], rtol=0, atol=1e-9)
        # К серединам хорд проводим лучи из центра в локальной системе;
        # аналитическая точка пересечения луча с эллипсом даёт контроль ошибки.
        midpoints = (normalized[:-1] + normalized[1:]) / 2
        exact = midpoints / np.linalg.norm(midpoints, axis=1)[:, None]
        error = ((exact - midpoints) * [a, b]) @ basis.T
        error *= [size[0] / 640, size[1] / 480]
        assert np.linalg.norm(error, axis=1).max() <= .5 + 1e-12
        hx = math.sqrt((a * math.cos(theta)) ** 2 + (b * math.sin(theta)) ** 2)
        hy = math.sqrt((a * math.sin(theta)) ** 2 + (b * math.cos(theta)) ** 2)
        sx, sy = size[0] / 640, size[1] / 480
        np.testing.assert_allclose(rectangles[0],
            [(100-hx)*sx-2, (100-hy)*sy-2, (100+hx)*sx+2, (100+hy)*sy+2],
            rtol=1e-12, atol=1e-9)
        assert state.droplet == d
        contour_cases += 1
emit(contour_cases=contour_cases, all_vertices_and_chord_errors="PASS")

vector_cases = 0
for vx, vy, dt, scale, exposure, size in (
    (2, 0, 100, 10, 20, (640, 480)), (2, -1, 100, 10, 20, (640, 480)),
    (2, 0, 200, 10, 20, (640, 480)), (2, 0, 100, 20, 20, (640, 480)),
    (2, 0, 100, 10, 10, (640, 480)), (0, 2, 100, 10, 20, (640, 480)),
    (2, -1, 100, 10, 20, (320, 120)), (0, 0, 100, 10, 20, (640, 480)),
    (.05, 0, 100, 10, 20, (640, 480)), (100, 0, 100, 10, 20, (640, 480)),
):
    camera = replace(Camera(), frame_interval_us=dt, scale_um_per_px=scale, exposure_us=exposure)
    d = Droplet(7, 100, 100, 200, 20, False, 0, math.hypot(vx, vy), vx, vy, 1, 0, 0)
    state = FrameDroplet(d, 10, 10, False, False, False)
    lines, tips, dots = [], [], []
    line, polygon, ellipse = ImageDraw.ImageDraw.line, ImageDraw.ImageDraw.polygon, ImageDraw.ImageDraw.ellipse

    def capture_vector(ctx, xy, *args, **kwargs):
        if kwargs.get("fill") == (0, 255, 255):
            lines.append(xy)
        return line(ctx, xy, *args, **kwargs)

    def capture_tip(ctx, xy, *args, **kwargs):
        tips.append(xy)
        return polygon(ctx, xy, *args, **kwargs)

    def capture_dot(ctx, xy, *args, **kwargs):
        dots.append(xy)
        return ellipse(ctx, xy, *args, **kwargs)

    with patch.object(ImageDraw.ImageDraw, "line", capture_vector), \
         patch.object(ImageDraw.ImageDraw, "polygon", capture_tip), \
         patch.object(ImageDraw.ImageDraw, "ellipse", capture_dot):
        output = draw(state, camera, size)
    start = np.array([100*size[0]/640, 100*size[1]/480])
    end = start + np.array([vx, vy])*dt/scale*np.array([size[0]/640,size[1]/480])
    length = np.linalg.norm(end-start)
    if length == 0:
        assert not lines and not tips and not dots
    elif length < 1:
        assert not lines and not tips and len(dots) == 1
        np.testing.assert_allclose([(dots[0][0]+dots[0][2])/2,(dots[0][1]+dots[0][3])/2], end)
    else:
        np.testing.assert_allclose(lines[0], [start,end], rtol=1e-12, atol=1e-9)
        if end[0] >= size[0]:
            assert not tips
            assert output.getpixel((size[0]-1,100)) == (0,255,255)
        else:
            np.testing.assert_allclose(tips[0][0], end, rtol=0, atol=1e-9)
    vector_cases += 1
emit(vector_cases=vector_cases, length_direction_zero_short_long_and_exposure_independence="PASS")

root = tk.Tk()
try:
    root.geometry("1000x700")
    app = SettingsWindow(root)
    root.update()
    with tempfile.TemporaryDirectory(prefix="SprayFrameGen-Q01-") as directory:
        for mode in ("ideal", "realistic"):
            for fmt, bits in (("png",8),("png",16),("tiff",16)):
                app.config = config(mode, directory, **{"camera.format":fmt,"camera.bit_depth":bits})
                app._draft_data = asdict(app.config)
                app._vars_from_config()
                app.overlay_var.set(False)
                app._toggle_overlay()
                root.update()
                data = dumps(app.config)
                frames = tuple(generate_frames(app.config))
                before = run_generation(app.config)
                for _ in range(3):
                    for enabled in (True, False):
                        app.overlay_checkbutton.invoke()
                        root.update()
                        assert app.preview_overlay == enabled
                        assert "Ошибка" not in app.status_var.get()
                # Проверяем экспорт, пока слой фактически включён.
                app.overlay_checkbutton.invoke()
                root.update()
                assert app.preview_overlay and app.overlay_legend.winfo_ismapped()
                after = run_generation(app.config)
                assert dumps(app.config) == data
                assert tuple(generate_frames(app.config)) == frames
                assert (before/"droplets.csv").read_bytes() == (after/"droplets.csv").read_bytes()
                renderer = create_renderer(app.config)
                files_a, files_b = sorted((before/"frames").iterdir()), sorted((after/"frames").iterdir())
                assert len(files_a) == len(files_b) == 2
                for pa,pb,frame in zip(files_a,files_b,frames):
                    with Image.open(pa) as ia, Image.open(pb) as ib:
                        assert len(ia.getbands()) == len(ib.getbands()) == 1
                        aa,bb = np.array(ia),np.array(ib)
                        ia.close()
                        ib.close()
                    np.testing.assert_array_equal(aa,bb)
                    np.testing.assert_array_equal(aa,encode_intensity(renderer.render(frame),bits))
                emit(mode=mode,format=fmt,bits=bits,effects="all enabled",
                     real_toggle_and_exported_pixels_and_csv_and_model="PASS")
finally:
    root.destroy()
