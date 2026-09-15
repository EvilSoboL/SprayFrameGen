"""Проверка окна в интерактивном Windows-сеансе, отдельно от headless-тестов."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from sprayframegen.configuration import load, loads
from sprayframegen.ui.app import create_window


class WindowSmokeTests(unittest.TestCase):
    def test_start_edit_validate_save_load(self):
        root = create_window()
        try:
            root.withdraw()
            root.update()
            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)
            editor = next(widget for widget in descendants(root) if isinstance(widget, ScrolledText))
            toolbar = next(widget for widget in root.winfo_children() if isinstance(widget, ttk.Frame))
            buttons = {str(button.cget("text")): button for button in toolbar.winfo_children()}
            self.assertEqual(loads(editor.get("1.0", "end-1c")).configuration.camera.width_px, 640)
            buttons["Новая: идеальный"].invoke()
            self.assertFalse(loads(editor.get("1.0", "end-1c")).configuration.appearance.defocus_enabled)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "настройки.json"
                with patch("sprayframegen.ui.app.filedialog.asksaveasfilename", return_value=str(path)):
                    buttons["Сохранить…"].invoke()
                expected = load(path).configuration
                buttons["Новая: реалистичный"].invoke()
                with patch("sprayframegen.ui.app.filedialog.askopenfilename", return_value=str(path)):
                    buttons["Открыть…"].invoke()
                self.assertEqual(loads(editor.get("1.0", "end-1c")).configuration, expected)
            editor.delete("1.0", "end")
            editor.insert("1.0", "{}")
            with patch("sprayframegen.ui.app.messagebox.showerror") as error:
                buttons["Проверить"].invoke()
                error.assert_called_once()
                self.assertIn("обязательное поле", error.call_args.args[1])
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
