"""Тесты контроллера генерации: прогресс, отмена, ошибки, атомарность."""

import csv
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sprayframegen.configuration import update_configuration, load
from sprayframegen.generation.controller import run_generation, run_generation_in_thread, CancellationToken, ExportError, ProgressEvent, CompletionEvent, _estimate_series_size, _check_disk_space, _check_write_permission
from sprayframegen.generation import generate_frames
from tests.test_render import ideal_config


def config(**changes):
    return update_configuration(ideal_config(), changes).configuration


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generation_completes_with_progress_events(self):
        c = config(**{
            "droplets.count_per_frame": 2,
            "camera.frame_count": 3,
            "export.output_directory": str(self.output_dir),
        })
        events = []
        def progress(event):
            events.append(event)

        directory = run_generation(c, progress_callback=progress)

        self.assertTrue(directory.exists())
        self.assertEqual(len(events), 3)
        for i, event in enumerate(events):
            self.assertEqual(event.frame_number, i + 1)
            self.assertEqual(event.total_frames, 3)
            self.assertEqual(event.completed_frames, i + 1)
            self.assertEqual(event.status, "running")
            self.assertGreater(event.elapsed_seconds, 0)

        # Проверяем итоговый config.json
        saved = load(directory / "config.json").configuration
        self.assertEqual(saved.run.status, "completed")
        self.assertEqual(saved.run.completed_frame_count, 3)

        # Проверяем CSV
        rows = list(csv.DictReader((directory / "droplets.csv").read_text().splitlines()))
        self.assertEqual(len(rows), 6)  # 2 капли * 3 кадра

    def test_cancellation_before_first_frame(self):
        c = config(**{
            "droplets.count_per_frame": 2,
            "camera.frame_count": 5,
            "export.output_directory": str(self.output_dir),
        })
        token = CancellationToken()
        token.cancel()

        with self.assertRaises(ExportError) as cm:
            run_generation(c, cancellation_token=token)

        self.assertIn("отменена", str(cm.exception))
        self.assertIsNotNone(cm.exception.directory)
        directory = cm.exception.directory
        self.assertTrue(directory.exists())

        saved = load(directory / "config.json").configuration
        self.assertEqual(saved.run.status, "cancelled")
        self.assertEqual(saved.run.completed_frame_count, 0)

    def test_cancellation_between_frames(self):
        c = config(**{
            "droplets.count_per_frame": 2,
            "camera.frame_count": 5,
            "export.output_directory": str(self.output_dir),
        })

        def progress(event):
            if event.completed_frames == 2:
                token.cancel()

        token = CancellationToken()
        with self.assertRaises(ExportError) as cm:
            run_generation(c, progress_callback=progress, cancellation_token=token)

        self.assertIn("отменена", str(cm.exception))
        self.assertIsNotNone(cm.exception.directory)
        directory = cm.exception.directory

        self.assertTrue(directory.exists())
        saved = load(directory / "config.json").configuration
        self.assertEqual(saved.run.status, "cancelled")
        self.assertEqual(saved.run.completed_frame_count, 2)

        # Проверяем что CSV содержит только 2 кадра
        rows = list(csv.DictReader((directory / "droplets.csv").read_text().splitlines()))
        self.assertEqual(len(rows), 4)  # 2 капли * 2 кадра

        # Изображения только для 2 кадров
        images = list((directory / "frames").iterdir())
        self.assertEqual(len(images), 2)

    def test_cancellation_does_not_corrupt_completed_frames(self):
        """При отмене уже готовые кадры и CSV не повреждаются."""
        c = config(**{
            "droplets.count_per_frame": 3,
            "camera.frame_count": 4,
            "export.output_directory": str(self.output_dir),
        })

        def progress(event):
            if event.completed_frames == 2:
                token.cancel()

        token = CancellationToken()
        with self.assertRaises(ExportError) as cm:
            run_generation(c, progress_callback=progress, cancellation_token=token)

        self.assertIn("отменена", str(cm.exception))
        directory = cm.exception.directory

        # Проверяем что первые 2 кадра читаются корректно
        for i in range(1, 3):
            img_path = directory / "frames" / f"frame_{i:06d}.png"
            self.assertTrue(img_path.exists())
            # Проверяем что файл не пустой и читается
            from PIL import Image
            with Image.open(img_path) as im:
                arr = np.array(im)
                self.assertEqual(arr.shape, (480, 640))

    def test_error_during_write_sets_error_status(self):
        """Ошибка записи изображения устанавливает status=error."""
        c = config(**{
            "droplets.count_per_frame": 1,
            "camera.frame_count": 2,
            "export.output_directory": str(self.output_dir),
        })

        # Мокаем сохранение изображения, чтобы выбросить ошибку на 2-м кадре
        original_save = None
        call_count = [0]

        def failing_save(self, path, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise OSError("Диск полон")
            return original_save(self, path, *args, **kwargs)

        from PIL import Image
        original_save = Image.Image.save

        with patch.object(Image.Image, "save", failing_save):
            with self.assertRaises(ExportError) as cm:
                run_generation(c)

        self.assertIn("Не удалось завершить серию", str(cm.exception))
        directory = cm.exception.directory
        self.assertIsNotNone(directory)

        saved = load(directory / "config.json").configuration
        self.assertEqual(saved.run.status, "error")
        self.assertEqual(saved.run.completed_frame_count, 1)

        # Первый кадр должен быть целым
        rows = list(csv.DictReader((directory / "droplets.csv").read_text().splitlines()))
        self.assertEqual(len(rows), 1)

    def test_disk_space_check_fails_when_insufficient(self):
        c = config(**{"export.output_directory": str(self.output_dir)})

        # Мокаем shutil.disk_usage чтобы вернуть очень мало места
        with patch("shutil.disk_usage") as mock_disk:
            mock_disk.return_value = type("obj", (object,), {"free": 100})()
            with self.assertRaises(Exception) as cm:
                _check_disk_space(self.output_dir, 1000000)
            self.assertIn("недостаточно свободного места", str(cm.exception).lower())

    def test_write_permission_check_fails_when_no_permission(self):
        c = config(**{"export.output_directory": str(self.output_dir)})

        # Мокаем запись тестового файла чтобы выбросить ошибку
        with patch("pathlib.Path.write_text", side_effect=OSError("Permission denied")):
            with self.assertRaises(Exception) as cm:
                _check_write_permission(self.output_dir)
            self.assertIn("нет прав записи", str(cm.exception).lower())

    def test_progress_callback_receives_correct_elapsed_time(self):
        c = config(**{
            "droplets.count_per_frame": 1,
            "camera.frame_count": 2,
            "export.output_directory": str(self.output_dir),
        })
        events = []

        def progress(event):
            events.append(event)

        run_generation(c, progress_callback=progress)

        self.assertEqual(len(events), 2)
        self.assertGreater(events[1].elapsed_seconds, events[0].elapsed_seconds)

    def test_run_generation_in_thread_returns_thread_and_token(self):
        c = config(**{
            "droplets.count_per_frame": 1,
            "camera.frame_count": 2,
            "export.output_directory": str(self.output_dir),
        })

        thread, token = run_generation_in_thread(c)
        self.assertIsInstance(thread, threading.Thread)
        self.assertIsInstance(token, CancellationToken)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())

    def test_cancellation_via_token_in_thread(self):
        c = config(**{
            "droplets.count_per_frame": 10,
            "camera.frame_count": 10,
            "export.output_directory": str(self.output_dir),
        })

        completed = []

        def progress(event):
            completed.append(event.completed_frames)
            if event.completed_frames >= 2:
                token.cancel()

        token = CancellationToken()
        thread, _ = run_generation_in_thread(c, progress_callback=progress, cancellation_token=token)
        thread.join(timeout=10)

        # Должно завершиться с cancelled
        # Находим папку серии
        series_dirs = list(self.output_dir.glob("series_*"))
        self.assertEqual(len(series_dirs), 1)
        saved = load(series_dirs[0] / "config.json").configuration
        self.assertEqual(saved.run.status, "cancelled")
        self.assertEqual(saved.run.completed_frame_count, 2)

    def test_estimate_series_size_returns_positive(self):
        c = config(**{
            "camera.width_px": 640,
            "camera.height_px": 480,
            "camera.frame_count": 10,
            "camera.bit_depth": 8,
            "camera.format": "png",
            "droplets.count_per_frame": 100,
        })
        size = _estimate_series_size(c)
        self.assertGreater(size, 0)

    def test_csv_rollback_on_image_write_failure(self):
        """При ошибке записи изображения CSV откатывается к предыдущему состоянию."""
        c = config(**{
            "droplets.count_per_frame": 2,
            "camera.frame_count": 3,
            "export.output_directory": str(self.output_dir),
        })

        call_count = [0]
        original_save = None

        def failing_save(self, path, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # ошибка на 2-м кадре
                raise OSError("Write error")
            return original_save(self, path, *args, **kwargs)

        from PIL import Image
        original_save = Image.Image.save

        with patch.object(Image.Image, "save", failing_save):
            with self.assertRaises(ExportError):
                run_generation(c)

        directory = list(self.output_dir.glob("series_*"))[0]
        rows = list(csv.DictReader((directory / "droplets.csv").read_text().splitlines()))
        # Должен остаться только 1 кадр (2 капли)
        self.assertEqual(len(rows), 2)


class ControllerIntegrationTests(unittest.TestCase):
    """Интеграционные тесты с обоими режимами рендера."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_ideal_mode_with_effects_completes(self):
        c = config(**{
            "droplets.render_mode": "ideal",
            "droplets.count_per_frame": 2,
            "camera.frame_count": 3,
            "motion.motion_blur_enabled": True,
            "appearance.defocus_enabled": True,
            "background.noise_enabled": True,
            "export.output_directory": str(self.output_dir),
        })
        directory = run_generation(c)
        self.assertTrue(directory.exists())
        saved = load(directory / "config.json").configuration
        self.assertEqual(saved.run.status, "completed")

    def test_realistic_mode_completes(self):
        c = config(**{
            "droplets.render_mode": "realistic",
            "droplets.count_per_frame": 2,
            "camera.frame_count": 3,
            "export.output_directory": str(self.output_dir),
        })
        directory = run_generation(c)
        self.assertTrue(directory.exists())
        saved = load(directory / "config.json").configuration
        self.assertEqual(saved.run.status, "completed")


if __name__ == "__main__":
    unittest.main()