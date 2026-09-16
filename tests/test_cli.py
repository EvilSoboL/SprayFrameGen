"""Проверки точки входа без интерактивного окна и сетевого доступа."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

from sprayframegen.__main__ import main
from sprayframegen.configuration import load, save
from test_render import ideal_config


class CommandLineTests(unittest.TestCase):
    def invoke(self, *args):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = main(list(args))
        return status, output.getvalue(), errors.getvalue()

    def test_create_and_check_unicode_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "настройки.json"
            status, output, errors = self.invoke("--new", str(path))
            self.assertEqual(status, 0)
            self.assertIn("Настройки сохранены", output)
            self.assertEqual(errors, "")
            self.assertIsNone(load(path).configuration.run)
            status, output, errors = self.invoke("--check", str(path))
            self.assertEqual(status, 0)
            self.assertIn("Конфигурация корректна", output)
            self.assertEqual(errors, "")

    def test_invalid_config_returns_error_and_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{}', encoding="utf-8")
            status, output, errors = self.invoke("--check", str(path))
            self.assertEqual(status, 1)
            self.assertEqual(output, "")
            self.assertIn("schema_version", errors)
            self.assertIn("отсутствует обязательное поле", errors)
            self.assertNotIn("Traceback", errors)

    def test_missing_file_returns_error(self):
        with tempfile.TemporaryDirectory() as directory:
            status, _, errors = self.invoke("--check", str(Path(directory) / "missing.json"))
            self.assertEqual(status, 1)
            self.assertIn("Ошибка:", errors)

    def test_generate_series_from_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "параметры.json"
            save(ideal_config(**{"export.output_directory": str(Path(directory).resolve())}), path)
            status, output, errors = self.invoke("--generate", str(path))
            self.assertEqual(status, 0)
            self.assertIn("Серия сохранена:", output)
            self.assertEqual(errors, "")
            series = next(Path(directory).glob("series_*"))
            self.assertEqual(load(series / "config.json").configuration.run.status, "completed")

    def test_generate_with_optical_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "параметры.json"
            save(ideal_config(**{"motion.motion_blur_enabled": True,
                                "export.output_directory": str(Path(directory).resolve())}), path)
            status, output, errors = self.invoke("--generate", str(path))
            self.assertEqual(status, 0)
            self.assertEqual(errors, "")
            self.assertEqual(len(list(Path(directory).glob("series_*"))), 1)

    def test_unavailable_tk_returns_actionable_error(self):
        with patch("sprayframegen.ui.app.create_window", side_effect=tk.TclError("init.tcl")):
            status, _, errors = self.invoke()
        self.assertEqual(status, 1)
        self.assertIn("Tcl/Tk", errors)
        self.assertIn("run.bat --check", errors)
        self.assertNotIn("Traceback", errors)

    def test_import_failure_returns_error(self):
        with patch("sprayframegen.ui.app.run", side_effect=ImportError("недоступна зависимость")):
            status, _, errors = self.invoke()
        self.assertEqual(status, 1)
        self.assertIn("недоступна зависимость", errors)

    @unittest.skipUnless(os.name == "nt", "Windows launcher")
    def test_launcher_missing_environment_in_unicode_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory) / "папка с пробелами"
            workdir.mkdir()
            launcher = workdir / "run.bat"
            launcher.write_bytes(Path("run.bat").read_bytes())
            result = subprocess.run(
                ["cmd.exe", "/d", "/c", "run.bat"], cwd=workdir,
                capture_output=True, encoding="utf-8", timeout=10,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Локальное окружение не найдено", result.stdout)
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
