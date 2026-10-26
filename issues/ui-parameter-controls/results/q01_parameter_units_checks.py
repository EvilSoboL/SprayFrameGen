"""Независимые регрессии Q01. Запуск из корня: python <этот файл>."""

import sys
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sprayframegen.ui.app import SettingsWindow


class Q01Checks(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.addCleanup(self.root.destroy)
        self.app = SettingsWindow(self.root)
        self.root.update()
        self.preview = patch.object(self.app, "_update_preview")
        self.preview.start()
        self.addCleanup(self.preview.stop)

    def test_scale_recovery_renders_preserved_diameters(self):
        app = self.app
        app.diameter_unit_var.set("пикс")
        app._change_diameter_unit()
        app.vars["camera.scale_um_per_px"].set("200")
        self.assertEqual(app.speed_pixels_var.get(), "—")
        app.vars["droplets.max_diameter_um"].set("5")
        self.assertEqual(app.config.camera.scale_um_per_px, 200)
        self.assertEqual(app.config.droplets.median_um, 120)
        self.assertAlmostEqual(float(app.vars["droplets.median_um"].get()), 0.6)

    def test_failed_load_preserves_unfinished_input(self):
        app = self.app
        app.vars["camera.frame_interval_us"].set("1e")
        before = {path: var.get() for path, var in app.vars.items()}
        before_config, before_status = app.config, app.status_var.get()
        with patch("sprayframegen.ui.app.filedialog.askopenfilename", return_value="Q01-missing.json"), \
             patch("sprayframegen.ui.app.load", side_effect=OSError("Q01: missing file")), \
             patch("sprayframegen.ui.app.messagebox.showerror") as error:
            app._open_file()
        error.assert_called_once()
        self.assertEqual(app.config, before_config)
        self.assertEqual({path: var.get() for path, var in app.vars.items()}, before)
        self.assertEqual(app.status_var.get(), before_status)
        self.assertEqual(app.speed_pixels_var.get(), "—")


if __name__ == "__main__":
    unittest.main(verbosity=2)
