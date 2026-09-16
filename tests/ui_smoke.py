"""Проверка окна в интерактивном Windows-сеансе, отдельно от headless-тестов."""

import tkinter as tk
import unittest
from tkinter import ttk

from sprayframegen.configuration import loads
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
            self.assertIn("Новая: реалистичный", all_buttons)
            self.assertIn("Новая: идеальный", all_buttons)
            self.assertIn("Открыть…", all_buttons)
            self.assertIn("Сохранить…", all_buttons)
            self.assertIn("Проверить", all_buttons)

            app._vars_from_config()
            self.assertEqual(app.vars["camera.width_px"].get(), "640×480")

            all_buttons["Новая: идеальный"].invoke()
            root.update()
            self.assertFalse(app.vars["appearance.defocus_enabled"].get())

            all_buttons["Новая: реалистичный"].invoke()
            root.update()

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


if __name__ == "__main__":
    unittest.main()
