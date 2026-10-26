"""Дополнительная матрица AC01–AC05; запуск из корня проекта."""

import math
import sys
import tkinter as tk
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sprayframegen.configuration import load
from sprayframegen.ui.app import SettingsWindow, DIAMETER_PATHS


class Acceptance(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.addCleanup(self.root.destroy)
        self.app = SettingsWindow(self.root)
        self.root.update()
        rendering = patch.object(self.app, "_update_preview")
        rendering.start()
        self.addCleanup(rendering.stop)

    def hz(self):
        self.app.interval_unit_var.set("Гц")
        self.app._change_interval_unit()

    def pixels(self):
        self.app.diameter_unit_var.set("пикс")
        self.app._change_diameter_unit()

    def test_invalid_frequency_and_action_guards(self):
        self.hz()
        a = self.app
        for text in ("0", "-1", "NaN", "inf", "", "1e", "1e309", "1e-320"):
            with self.subTest(text=text):
                a.vars["camera.frame_interval_us"].set(text)
                self.assertEqual(a.vars["camera.frame_interval_us"].get(), text)
                self.assertEqual(a.speed_pixels_var.get(), "—")
                self.assertTrue(a._preview_is_stale)
                with patch("sprayframegen.ui.app.messagebox.showerror") as error, \
                     patch("sprayframegen.ui.app.filedialog.asksaveasfilename") as save_dialog:
                    a._check_config()
                    a._save_file()
                    a._start_generation()
                    self.assertEqual(error.call_count, 3)
                    save_dialog.assert_not_called()
                    self.assertIsNone(a.generation_thread)
                a.vars["camera.frame_interval_us"].set("10000")
                self.assertEqual(a.speed_pixels_var.get(), "80")
                self.assertFalse(a._preview_is_stale)

    def test_exposure_and_diameter_boundaries(self):
        a = self.app
        self.hz()
        a.vars["camera.frame_interval_us"].set("50000")
        self.assertEqual(a.config.camera.frame_interval_us, 20)
        a.vars["camera.frame_interval_us"].set("100000")
        self.assertEqual(a.speed_pixels_var.get(), "—")
        a.vars["camera.exposure_us"].set("5")
        self.assertEqual(a.config.camera.frame_interval_us, 10)
        self.assertEqual(a.config.camera.exposure_us, 5)
        a.vars["camera.frame_interval_us"].set("10000")
        self.pixels()
        a.vars["droplets.max_diameter_um"].set("2.99")
        self.assertEqual(a.speed_pixels_var.get(), "—")
        a.vars["droplets.max_diameter_um"].set("50")
        a.vars["droplets.size_distribution"].set("Нормальное")
        a._on_field_change("droplets.size_distribution")
        a.vars["droplets.mean_um"].set("3")
        a.vars["droplets.std_um"].set("0")
        a.vars["droplets.max_diameter_um"].set("3")
        self.assertEqual(a.config.droplets.mean_um, 30)
        self.assertEqual(a.config.droplets.max_diameter_um, 30)
        a.vars["droplets.mean_um"].set("2.99")
        self.assertEqual(a.speed_pixels_var.get(), "—")
        a.vars["droplets.max_diameter_um"].set("50")
        a.vars["droplets.std_um"].set("2")
        self.assertAlmostEqual(a.config.droplets.mean_um, 29.9)

    def test_all_diameters_and_unfinished_scale(self):
        a = self.app
        self.pixels()
        for path, pixels in zip(DIAMETER_PATHS, (15, 0, 16, 60)):
            a.vars[path].set(str(pixels))
            self.assertEqual(getattr(a.config.droplets, path.split(".")[1]), pixels * 10)
        self.assertEqual(a.config.droplets.geometric_std, 1.35)
        a.vars["droplets.median_um"].set("1e")
        a.vars["camera.scale_um_per_px"].set("20")
        self.assertEqual(a.config.camera.scale_um_per_px, 10)
        a.diameter_unit_var.set("мкм")
        a._change_diameter_unit()
        self.assertEqual(a.diameter_unit, "px")
        self.assertEqual(a.vars["droplets.median_um"].get(), "1e")
        a.vars["camera.scale_um_per_px"].set("10")
        a.vars["droplets.median_um"].set("16")
        self.assertEqual(a.config.droplets.median_um, 160)

    def test_old_json_ui_roundtrip_and_nontrivial_interval(self):
        a = self.app
        a.vars["camera.frame_interval_us"].set("123.456789")
        self.hz()
        self.assertTrue(math.isclose(float(a.vars["camera.frame_interval_us"].get()),
                                     1e6 / 123.456789, rel_tol=1e-12))
        self.pixels()
        source = Path("docs/examples/config.default.json").resolve()
        with patch("sprayframegen.ui.app.filedialog.askopenfilename", return_value=str(source)):
            a._open_file()
        expected = asdict(load(source).configuration)
        self.assertEqual(a.speed_pixels_var.get(), "80")
        self.assertEqual(a.vars["camera.frame_interval_us"].get(), "10000")
        self.assertEqual(a.vars["droplets.median_um"].get(), "12")
        with TemporaryDirectory() as directory:
            target = Path(directory) / "saved.json"
            with patch("sprayframegen.ui.app.filedialog.asksaveasfilename", return_value=str(target)):
                a._save_file()
            with patch("sprayframegen.ui.app.filedialog.askopenfilename", return_value=str(target)):
                a._open_file()
            self.assertEqual(asdict(a.config), expected)
            self.assertEqual(asdict(load(target).configuration), expected)
            self.assertEqual((a.interval_unit, a.diameter_unit), ("hz", "px"))

    def test_display_overflow_does_not_commit_or_disable_input(self):
        a = self.app
        a.vars["droplets.mean_um"].set("1e308")
        self.pixels()
        previous = a.config
        a.vars["camera.scale_um_per_px"].set("0.1")
        self.assertEqual(a.config, previous)
        self.assertFalse(a._loading_vars)
        self.assertEqual(a.speed_pixels_var.get(), "—")
        a.vars["camera.scale_um_per_px"].set("10")
        self.assertEqual(a.config.camera.scale_um_per_px, 10)
        self.assertEqual(a.speed_pixels_var.get(), "80")
        self.assertFalse(a._preview_is_stale)


if __name__ == "__main__":
    unittest.main(verbosity=2)
