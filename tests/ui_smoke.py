"""Проверка окна в интерактивном Windows-сеансе, отдельно от headless-тестов."""

import tkinter as tk
import unittest
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from tkinter import ttk
from unittest.mock import patch

import numpy as np
from PIL import ImageTk

from sprayframegen.configuration import dumps, loads
from sprayframegen.configuration import update_configuration
from sprayframegen.ui.app import SettingsWindow


class WindowSmokeTests(unittest.TestCase):
    def test_window_creation_and_basic_widgets(self):
        root = tk.Tk()
        try:
            app = SettingsWindow(root)
            root.update()

            def find_buttons(frame):
                buttons = {}
                for child in frame.winfo_children():
                    if isinstance(child, ttk.Button):
                        buttons[str(child.cget("text"))] = child
                    buttons.update(find_buttons(child))
                return buttons

            all_buttons = find_buttons(root)
            self.assertNotIn("Новая: реалистичный", all_buttons)
            self.assertNotIn("Новая: идеальный", all_buttons)
            self.assertIn("Открыть…", all_buttons)
            self.assertIn("Сохранить…", all_buttons)
            self.assertIn("Проверить", all_buttons)

            app._vars_from_config()
            self.assertEqual(app.vars["camera.width_px"].get(), "640×480")

            render_mode = next(
                child for child in app.tab_droplets.winfo_children()
                if isinstance(child, ttk.Combobox)
                and tuple(child.cget("values")) == ("Реалистичный", "Идеальный")
            )
            render_mode.current(1)
            render_mode.event_generate("<<ComboboxSelected>>")
            root.update()
            self.assertEqual(app.config.droplets.render_mode, "ideal")
            self.assertEqual(app.vars["droplets.render_mode"].get(), "Идеальный")

            render_mode.current(0)
            render_mode.event_generate("<<ComboboxSelected>>")
            root.update()
            self.assertEqual(app.config.droplets.render_mode, "realistic")

            app.vars["camera.frame_count"].set(1)
            from unittest.mock import patch
            with patch("sprayframegen.ui.app.messagebox.showerror") as error:
                all_buttons["Проверить"].invoke()
                root.update()
                error.assert_called_once()
                self.assertIn("2", error.call_args.args[1])
                self.assertIn("1000", error.call_args.args[1])
        finally:
            root.destroy()

    def test_preview_generation(self):
        root = tk.Tk()
        try:
            app = SettingsWindow(root)
            root.update()
            app._update_preview()
            root.update()
            self.assertIsNotNone(app.preview_tk)
        finally:
            root.destroy()

    def test_overlay_toggle_repaints_clean_and_color_previews(self):
        root = tk.Tk()
        try:
            root.geometry("1000x700")
            app = SettingsWindow(root)
            root.update()
            legend_text = " ".join(
                child.cget("text") for child in app.overlay_legend.winfo_children()
                for child in child.winfo_children() if isinstance(child, ttk.Label)
            ) + " " + " ".join(
                child.cget("text") for child in app.overlay_legend.winfo_children()
                if isinstance(child, ttk.Label)
            )
            self.assertIn("межкадровый интервал", legend_text)
            self.assertIn("не экспортируется", legend_text)
            swatch_colors = [
                child.winfo_children()[0].itemcget(1, "fill")
                for child in app.overlay_legend.winfo_children()
                if isinstance(child, ttk.Frame)
            ]
            self.assertEqual(swatch_colors, ["#ffff00", "#00ffff", "#ff0000"])

            for mode in ("realistic", "ideal"):
                for tone in ("light", "dark"):
                    for bit_depth in (8, 16):
                        with self.subTest(mode=mode, tone=tone, bit_depth=bit_depth):
                            app.config = update_configuration(app.config, {
                                "droplets.render_mode": mode,
                                "background.tone": tone,
                                "camera.bit_depth": bit_depth,
                                "camera.frame_count": 2,
                                "droplets.count_per_frame": 10,
                                "seed": 42,
                            }).configuration
                            app._vars_from_config()
                            app.overlay_var.set(False)
                            app._toggle_overlay()
                            root.update()
                            clean = np.asarray(ImageTk.getimage(app.preview_tk).convert("RGB")).copy()
                            baseline_items = app.preview_canvas.find_all()
                            self.assertEqual(len(baseline_items), 1)
                            self.assertEqual(app.overlay_legend.winfo_manager(), "")

                            for _ in range(3):
                                app.status_var.set("Готово")
                                app.overlay_checkbutton.invoke()
                                root.update()
                                self.assertTrue(app.preview_overlay)
                                self.assertNotIn("Ошибка предпросмотра", app.status_var.get())
                                self.assertEqual(app.overlay_legend.winfo_manager(), "pack")
                                self.assertEqual(len(app.preview_canvas.find_all()), 1)
                                overlaid = np.asarray(ImageTk.getimage(app.preview_tk).convert("RGB"))
                                self.assertFalse(np.array_equal(clean, overlaid))

                                app.overlay_checkbutton.invoke()
                                root.update()
                                self.assertFalse(app.preview_overlay)
                                self.assertEqual(app.overlay_legend.winfo_manager(), "")
                                self.assertEqual(len(app.preview_canvas.find_all()), 1)
                                restored = np.asarray(ImageTk.getimage(app.preview_tk).convert("RGB"))
                                np.testing.assert_array_equal(restored, clean)
        finally:
            root.destroy()

    def test_scale_recovery_refreshes_all_pixel_sizes(self):
        root = tk.Tk()
        try:
            app = SettingsWindow(root)
            root.update()
            with patch.object(app, "_update_preview"):
                app.diameter_unit_var.set("пикс")
                app._change_diameter_unit()
                app.vars["camera.scale_um_per_px"].set("200")
                self.assertEqual(app.speed_pixels_var.get(), "—")
                app.vars["droplets.max_diameter_um"].set("5")
                self.assertEqual(app.config.camera.scale_um_per_px, 200)
                for path, physical, pixels in (
                    ("droplets.mean_um", 120, 0.6),
                    ("droplets.std_um", 20, 0.1),
                    ("droplets.median_um", 120, 0.6),
                    ("droplets.max_diameter_um", 1000, 5),
                ):
                    self.assertEqual(getattr(app.config.droplets, path.split(".")[1]), physical)
                    self.assertAlmostEqual(float(app.vars[path].get()), pixels)
                    self.assertEqual(float(app.widgets[path].cget("to")), 25)
                self.assertEqual(float(app.vars["droplets.geometric_std"].get()), 1.35)
                self.assertEqual(app.speed_pixels_var.get(), "4")
                app._update_config_from_vars()
                self.assertEqual(loads(dumps(app.config)).configuration, app.config)
        finally:
            root.destroy()

    def test_failed_json_load_preserves_entire_draft(self):
        root = tk.Tk()
        try:
            app = SettingsWindow(root)
            root.update()
            app.interval_unit_var.set("Гц")
            app._change_interval_unit()
            app.diameter_unit_var.set("пикс")
            app._change_diameter_unit()
            app.vars["camera.frame_interval_us"].set("1e")
            before_values = {path: var.get() for path, var in app.vars.items()}
            before_draft = copy.deepcopy(app._draft_data)
            before_rendered = dict(app._rendered_text)
            before_config, before_status = app.config, app.status_var.get()
            before_overlay = app.preview_canvas.find_withtag("draft-error")
            with TemporaryDirectory() as tmp:
                malformed = Path(tmp) / "malformed.json"
                malformed.write_text("{", encoding="utf-8")
                invalid = Path(tmp) / "invalid.json"
                invalid.write_text("{}", encoding="utf-8")
                overflow = Path(tmp) / "overflow.json"
                data = json.loads(dumps(app.config))
                data["camera"]["scale_um_per_px"] = 0.1
                data["droplets"]["mean_um"] = 1e308
                overflow.write_text(json.dumps(data), encoding="utf-8")
                # Physical configuration is valid, but its pixel representation overflows.
                loads(overflow.read_text(encoding="utf-8"))
                for path in (Path(tmp) / "missing.json", malformed, invalid, overflow):
                    with self.subTest(path=path.name), \
                         patch("sprayframegen.ui.app.filedialog.askopenfilename", return_value=str(path)), \
                         patch("sprayframegen.ui.app.messagebox.showerror") as error:
                        app._open_file()
                        error.assert_called_once()
                        self.assertEqual({key: var.get() for key, var in app.vars.items()}, before_values)
                        self.assertEqual(app.config, before_config)
                        self.assertEqual(app._draft_data, before_draft)
                        self.assertEqual(app._rendered_text, before_rendered)
                        self.assertEqual(app.status_var.get(), before_status)
                        self.assertEqual(app.speed_pixels_var.get(), "—")
                        self.assertEqual(app.preview_canvas.find_withtag("draft-error"), before_overlay)
                        self.assertEqual((app.interval_unit, app.diameter_unit), ("hz", "px"))
                        self.assertFalse(app._loading_vars)
        finally:
            root.destroy()

    def test_units_scale_loading_and_invalid_draft_guards(self):
        root = tk.Tk()
        try:
            app = SettingsWindow(root)
            root.update()

            app.interval_unit_var.set("Гц")
            app._change_interval_unit()
            self.assertEqual(app.interval_label.cget("text"), "Частота кадров (Гц)")
            app.vars["camera.frame_interval_us"].set("5000")
            root.update()
            self.assertAlmostEqual(app.config.camera.frame_interval_us, 200)
            self.assertEqual(app.speed_pixels_var.get(), "160")

            app.diameter_unit_var.set("пикс")
            app._change_diameter_unit()
            self.assertEqual(app.vars["droplets.median_um"].get(), "12")
            app.vars["camera.scale_um_per_px"].set("20")
            root.update()
            self.assertEqual(app.config.droplets.median_um, 120)
            self.assertEqual(app.vars["droplets.median_um"].get(), "6")
            self.assertEqual(app.speed_pixels_var.get(), "80")

            original_interval = app.config.camera.frame_interval_us
            original_diameters = tuple(getattr(app.config.droplets, path.rsplit(".", 1)[1])
                                       for path in ("droplets.mean_um", "droplets.std_um",
                                                    "droplets.median_um", "droplets.max_diameter_um"))
            with patch.object(app, "_update_preview"):
                for i in range(1000):
                    app.interval_unit_var.set("мкс" if i % 2 == 0 else "Гц")
                    app._change_interval_unit()
                    app.diameter_unit_var.set("мкм" if i % 2 == 0 else "пикс")
                    app._change_diameter_unit()
            self.assertEqual(app.config.camera.frame_interval_us, original_interval)
            self.assertEqual(tuple(getattr(app.config.droplets, path.rsplit(".", 1)[1])
                                   for path in ("droplets.mean_um", "droplets.std_um",
                                                "droplets.median_um", "droplets.max_diameter_um")), original_diameters)

            # UI units are transient; the serialized configuration remains physical.
            serialized = dumps(app.config)
            self.assertEqual(loads(serialized).configuration.droplets.median_um, 120)
            self.assertNotIn("unit", serialized)

            with TemporaryDirectory() as tmp:
                target = Path(tmp) / "roundtrip.json"
                with patch("sprayframegen.ui.app.filedialog.asksaveasfilename", return_value=str(target)):
                    app._save_file()
                saved = loads(target.read_text(encoding="utf-8"))
                self.assertEqual(saved.configuration.camera.frame_interval_us, original_interval)
                self.assertEqual(saved.configuration.droplets.median_um, 120)

            app.diameter_unit_var.set("мкм")
            app._change_diameter_unit()
            app.vars["camera.scale_um_per_px"].set("200")
            root.update()
            self.assertEqual(app.config.camera.scale_um_per_px, 20)
            self.assertIn("camera.scale_um_per_px", app.status_var.get())
            app.vars["droplets.max_diameter_um"].set("1000")
            root.update()
            self.assertEqual(app.config.camera.scale_um_per_px, 200)
            self.assertEqual(app.config.droplets.max_diameter_um, 1000)
            app.diameter_unit_var.set("пикс")
            app._change_diameter_unit()

            app.vars["camera.frame_interval_us"].set("1e")
            root.update()
            self.assertEqual(app.vars["camera.frame_interval_us"].get(), "1e")
            self.assertEqual(app.speed_pixels_var.get(), "—")
            self.assertIn("camera.frame_interval_us", app.status_var.get())
            app.interval_unit_var.set("мкс")
            app._change_interval_unit()
            self.assertEqual(app.interval_unit_var.get(), "Гц")
            self.assertEqual(app.vars["camera.frame_interval_us"].get(), "1e")
            with TemporaryDirectory() as tmp:
                target = Path(tmp) / "should-not-exist.json"
                with patch("sprayframegen.ui.app.filedialog.asksaveasfilename", return_value=str(target)) as dialog, \
                     patch("sprayframegen.ui.app.messagebox.showerror") as error:
                    app._save_file()
                    dialog.assert_not_called()
                    error.assert_called_once()
                    self.assertFalse(target.exists())
                app.export_dir_var.set(tmp)
                with patch("sprayframegen.ui.app.messagebox.showerror") as error:
                    app._start_generation()
                    error.assert_called_once()
                    self.assertIsNone(app.generation_thread)

            # Loading an old physical JSON preserves selected units.
            app.vars["camera.frame_interval_us"].set("100")
            root.update()
            old_config = Path("docs/examples/config.default.json").resolve()
            with patch("sprayframegen.ui.app.filedialog.askopenfilename", return_value=str(old_config)):
                app._open_file()
            root.update()
            self.assertEqual(app.interval_unit_var.get(), "Гц")
            self.assertEqual(app.diameter_unit_var.get(), "пикс")
            self.assertEqual(app.vars["camera.frame_interval_us"].get(), "10000")
            self.assertEqual(app.vars["droplets.median_um"].get(), "12")
            self.assertEqual(app.speed_pixels_var.get(), "80")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
