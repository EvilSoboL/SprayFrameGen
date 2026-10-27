"""Строгие проверки геометрии и изоляции служебного слоя предпросмотра."""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from sprayframegen.configuration import Camera, dumps, new_configuration, update_configuration
from sprayframegen.generation import generate_frames
from sprayframegen.generation.controller import run_generation
from sprayframegen.generation.sequence import Frame, FrameDroplet
from sprayframegen.model import Droplet
from sprayframegen.render import create_renderer, encode_intensity
from sprayframegen.ui.app import SettingsWindow


def droplet(*, x=100, y=100, vx=0, vy=0, angle=45, droplet_id=7):
    return Droplet(droplet_id, x, y, 200, 20, False, 0, math.hypot(vx, vy),
                   vx, vy, 1.35, angle, 0)


class PreviewOverlayTests(unittest.TestCase):
    def draw(self, image, states, camera=None):
        return SettingsWindow._draw_overlay(None, image, Frame(1, 0, tuple(states)), camera or Camera())

    def test_rotated_contour_arrow_and_border_use_contract_geometry(self):
        camera = replace(Camera(), frame_interval_us=100, scale_um_per_px=10, exposure_us=10)
        d = droplet(vx=2, vy=-1, angle=45)
        a, b = 11.381179640090037, 8.78643542781378
        state = FrameDroplet(d, a, b, False, True, False)
        source = Image.new("L", (640, 480), 70)
        before = source.tobytes()
        lines, rectangles, polygons = [], [], []
        original_line = ImageDraw.ImageDraw.line
        original_rectangle = ImageDraw.ImageDraw.rectangle
        original_polygon = ImageDraw.ImageDraw.polygon

        def line_spy(draw, xy, *args, **kwargs):
            lines.append((xy, kwargs.get("fill")))
            return original_line(draw, xy, *args, **kwargs)

        def rectangle_spy(draw, xy, *args, **kwargs):
            rectangles.append((xy, kwargs.get("outline")))
            return original_rectangle(draw, xy, *args, **kwargs)

        def polygon_spy(draw, xy, *args, **kwargs):
            polygons.append((xy, kwargs.get("fill")))
            return original_polygon(draw, xy, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, "line", line_spy), \
             patch.object(ImageDraw.ImageDraw, "rectangle", rectangle_spy), \
             patch.object(ImageDraw.ImageDraw, "polygon", polygon_spy):
            result = self.draw(source, (state,), camera)

        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, source.size)
        self.assertEqual(source.tobytes(), before)
        self.assertGreaterEqual(len(lines), 2)
        contour = lines[0][0]
        self.assertEqual(lines[0][1], (255, 255, 0))
        theta = math.radians(45)
        expected_first = (100 + a * math.cos(theta), 100 + a * math.sin(theta))
        self.assertAlmostEqual(contour[0][0], expected_first[0], delta=1e-9)
        self.assertAlmostEqual(contour[0][1], expected_first[1], delta=1e-9)
        # Every chord midpoint stays within the contract's 0.5 screen-pixel sagitta.
        segments = len(contour) - 1
        self.assertLessEqual(max(a, b) * (1 - math.cos(math.pi / segments)), 0.5)
        second_angle = 2 * math.pi / segments
        expected_second = (
            100 + a * math.cos(second_angle) * math.cos(theta) - b * math.sin(second_angle) * math.sin(theta),
            100 + a * math.cos(second_angle) * math.sin(theta) + b * math.sin(second_angle) * math.cos(theta),
        )
        np.testing.assert_allclose(contour[1], expected_second, rtol=1e-12, atol=1e-9)

        self.assertEqual(lines[1][1], (0, 255, 255))
        arrow = lines[1][0]
        self.assertAlmostEqual(arrow[0][0], 100)
        self.assertAlmostEqual(arrow[0][1], 100)
        self.assertAlmostEqual(arrow[1][0], 120)
        self.assertAlmostEqual(arrow[1][1], 90)
        self.assertEqual(polygons[0][1], (0, 255, 255))
        ux, uy = 20 / math.sqrt(500), -10 / math.sqrt(500)
        base_x, base_y = 120 - ux * 6, 90 - uy * 6
        self.assertEqual(polygons[0][0][0], (120, 90))
        np.testing.assert_allclose(polygons[0][0][1], (base_x - uy * 3, base_y + ux * 3),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(polygons[0][0][2], (base_x + uy * 3, base_y - ux * 3),
                                   rtol=0, atol=1e-12)
        expected_bounds = (100 - math.hypot(a * math.cos(theta), b * math.sin(theta)) - 2,
                           100 - math.hypot(a * math.sin(theta), b * math.cos(theta)) - 2,
                           100 + math.hypot(a * math.cos(theta), b * math.sin(theta)) + 2,
                           100 + math.hypot(a * math.sin(theta), b * math.cos(theta)) + 2)
        self.assertEqual(rectangles[0][1], (255, 0, 0))
        np.testing.assert_allclose(rectangles[0][0], expected_bounds, rtol=0, atol=1e-12)
        self.assertIn((255, 255, 0), [color for _, color in result.getcolors(maxcolors=640 * 480)])

    def test_zero_short_long_vectors_empty_frame_and_non_mutation(self):
        camera = replace(Camera(), frame_interval_us=100, scale_um_per_px=10)
        base = Image.new("RGB", (640, 480), (20, 30, 40))
        for vx, expected_cyan in ((0, False), (0.05, True), (100, True)):
            with self.subTest(vx=vx):
                state = FrameDroplet(droplet(vx=vx, angle=0), 10, 10, False, False, False)
                result = self.draw(base, (state,), camera)
                colors = {color for _, color in result.getcolors(maxcolors=640 * 480)}
                self.assertEqual(base.getpixel((100, 100)), (20, 30, 40))
                self.assertEqual(result.mode, "RGB")
                self.assertEqual((0, 255, 255) in colors, expected_cyan)
        empty = self.draw(base, ())
        np.testing.assert_array_equal(np.asarray(empty), np.asarray(base))
        partial = self.draw(base, (FrameDroplet(
            droplet(x=-5, vx=2, angle=0), 10, 10, False, True, False),), camera)
        self.assertIn((0, 255, 255), {color for _, color in partial.getcolors(maxcolors=640 * 480)})

    def test_export_matrix_is_independent_of_overlay_and_matches_clean_renderer(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("realistic", "ideal"):
                for fmt, bits in (("png", 8), ("png", 16), ("tiff", 16)):
                    with self.subTest(mode=mode, fmt=fmt, bits=bits):
                        config = update_configuration(new_configuration(seed=42, render_mode=mode), {
                            "camera.frame_count": 2,
                            "camera.format": fmt,
                            "camera.bit_depth": bits,
                            "droplets.count_per_frame": 3,
                            "motion.motion_blur_enabled": False,
                            "appearance.defocus_enabled": False,
                            "background.noise_enabled": False,
                            "export.output_directory": str(Path(tmp).resolve()),
                        }).configuration
                        saved_config = dumps(config)
                        frames = tuple(generate_frames(config))
                        before = run_generation(config)
                        # The preview compositor is deliberately independent of config/model/export.
                        preview = Image.new("L", (320, 240), 90)
                        overlay = self.draw(preview, (), config.camera)
                        after = run_generation(config)
                        self.assertEqual(saved_config, dumps(config))
                        self.assertEqual(frames, tuple(generate_frames(config)))
                        self.assertEqual((before / "droplets.csv").read_bytes(),
                                         (after / "droplets.csv").read_bytes())
                        renderer = create_renderer(config)
                        first_files = sorted((before / "frames").iterdir())
                        second_files = sorted((after / "frames").iterdir())
                        self.assertEqual(len(first_files), len(second_files), 2)
                        for first, second, frame in zip(first_files, second_files, frames):
                            expected = encode_intensity(renderer.render(frame), bits)
                            with Image.open(first) as image_a, Image.open(second) as image_b:
                                pixels_a, pixels_b = np.asarray(image_a).copy(), np.asarray(image_b).copy()
                                self.assertEqual(len(image_a.getbands()), 1)
                                self.assertEqual(len(image_b.getbands()), 1)
                                image_a.close()
                                image_b.close()
                            np.testing.assert_array_equal(pixels_a, pixels_b)
                            np.testing.assert_array_equal(pixels_a, expected)
                        self.assertEqual(overlay.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
