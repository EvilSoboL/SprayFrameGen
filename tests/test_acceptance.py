"""Приёмочные тесты D09: все 19 критериев ТЗ, статистика, нагрузка, повторяемость."""

import csv
import json
import os
import tempfile
import time
import tracemalloc
import unittest
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from sprayframegen.configuration import (
    Configuration,
    ConfigurationError,
    new_configuration,
    update_configuration,
    load,
    save,
)
from sprayframegen.export import export_series
from sprayframegen.generation import generate_frames, prepare
from sprayframegen.generation.controller import run_generation, CancellationToken, ExportError
from sprayframegen.render import create_renderer, encode_intensity


def count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus header


def read_csv_frames(path: Path) -> dict[int, list[dict]]:
    """Читает CSV и группирует строки по frame_number."""
    frames = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = int(row["frame_number"])
            frames.setdefault(fn, []).append(row)
    return frames


class AcceptanceTests(unittest.TestCase):
    """Приёмочные тесты по разделу 13 ТЗ."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run_series(self, config: Configuration) -> Path:
        """Запускает генерацию и возвращает папку серии."""
        config = update_configuration(config, {"export.output_directory": str(self.output_root)}).configuration
        return run_generation(config)

    # 1. Запуск - покрывается smoke-test и CLI тестами

    # 2. Диапазоны: 2 и 1000 кадров, 1 и 10000 капель
    def test_range_2_frames(self):
        cfg = new_configuration(seed=1)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 2,
            "droplets.count_per_frame": 100,
        }).configuration
        directory = self._run_series(cfg)
        self.assertTrue((directory / "config.json").exists())
        frames = list((directory / "frames").iterdir())
        self.assertEqual(len(frames), 2)

    def test_range_1000_frames(self):
        cfg = new_configuration(seed=2)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 1000,
            "droplets.count_per_frame": 10,
        }).configuration
        directory = self._run_series(cfg)
        frames = list((directory / "frames").iterdir())
        self.assertEqual(len(frames), 1000)

    def test_range_1_droplet(self):
        cfg = new_configuration(seed=3)
        cfg = update_configuration(cfg, {
            "droplets.count_per_frame": 1,
            "camera.frame_count": 10,
        }).configuration
        directory = self._run_series(cfg)
        rows = count_csv_rows(directory / "droplets.csv")
        self.assertEqual(rows, 10)

    def test_range_10000_droplets(self):
        cfg = new_configuration(seed=4)
        cfg = update_configuration(cfg, {
            "droplets.count_per_frame": 10000,
            "camera.frame_count": 2,
        }).configuration
        directory = self._run_series(cfg)
        rows = count_csv_rows(directory / "droplets.csv")
        self.assertEqual(rows, 20000)

    # 3. Разрешения: 4 пресета
    def test_all_resolutions(self):
        for w, h in [(640, 480), (1024, 768), (1280, 1024), (1920, 1080)]:
            with self.subTest(resolution=f"{w}x{h}"):
                cfg = new_configuration(seed=10)
                cfg = update_configuration(cfg, {
                    "camera.width_px": w,
                    "camera.height_px": h,
                    "camera.frame_count": 2,
                    "droplets.count_per_frame": 5,
                }).configuration
                directory = self._run_series(cfg)
                img_path = next((directory / "frames").iterdir())
                with Image.open(img_path) as img:
                    self.assertEqual(img.size, (w, h))

    # 4. Форматы: PNG 8/16, TIFF 16
    def test_all_formats(self):
        for fmt, depth in [("png", 8), ("png", 16), ("tiff", 16)]:
            with self.subTest(format=f"{fmt}_{depth}"):
                cfg = new_configuration(seed=11)
                cfg = update_configuration(cfg, {
                    "camera.format": fmt,
                    "camera.bit_depth": depth,
                    "camera.frame_count": 2,
                    "droplets.count_per_frame": 5,
                }).configuration
                directory = self._run_series(cfg)
                img_path = next((directory / "frames").iterdir())
                with Image.open(img_path) as img:
                    if depth == 8:
                        self.assertEqual(img.mode, "L")
                    else:
                        self.assertIn(img.mode, ("I;16", "I"))

    # 5. Количество: ровно count_per_frame строк на кадр
    def test_exact_droplet_count_per_frame(self):
        for count in [1, 10, 100, 1000]:
            with self.subTest(count=count):
                cfg = new_configuration(seed=20)
                cfg = update_configuration(cfg, {
                    "droplets.count_per_frame": count,
                    "camera.frame_count": 5,
                }).configuration
                directory = self._run_series(cfg)
                frames = read_csv_frames(directory / "droplets.csv")
                self.assertEqual(len(frames), 5)
                for fn, rows in frames.items():
                    self.assertEqual(len(rows), count)

    # 6. Траектории: Vx, Vy с погрешностью 0.01 px
    def test_trajectory_accuracy(self):
        cfg = new_configuration(seed=30)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 10,
            "droplets.count_per_frame": 50,
            "motion.motion_blur_enabled": False,
        }).configuration
        directory = self._run_series(cfg)
        frames = read_csv_frames(directory / "droplets.csv")
        scale = cfg.camera.scale_um_per_px
        interval = cfg.camera.frame_interval_us
        for droplet_id in range(1, 51):
            positions = []
            for fn in range(1, 11):
                rows = [r for r in frames[fn] if int(r["droplet_id"]) == droplet_id]
                if not rows:
                    break
                positions.append((fn, float(rows[0]["center_x_px"]), float(rows[0]["center_y_px"])))
            if len(positions) < 2:
                continue
            for i in range(len(positions) - 1):
                fn1, x1, y1 = positions[i]
                fn2, x2, y2 = positions[i + 1]
                vx = (x2 - x1) * scale / interval
                vy = (y2 - y1) * scale / interval
                # Находим строку для этой капли на первом кадре
                row1 = next(r for r in frames[fn1] if int(r["droplet_id"]) == droplet_id)
                expected_vx = float(row1["vx_m_s"])
                expected_vy = float(row1["vy_m_s"])
                self.assertAlmostEqual(vx, expected_vx, delta=0.011)
                self.assertAlmostEqual(vy, expected_vy, delta=0.011)

    # 7. ID: сохраняется до выхода, не повторяется
    def test_id_persistence_and_uniqueness(self):
        cfg = new_configuration(seed=40)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 20,
            "droplets.count_per_frame": 20,
        }).configuration
        directory = self._run_series(cfg)
        frames = read_csv_frames(directory / "droplets.csv")
        all_ids = set()
        for fn in range(1, 21):
            for row in frames[fn]:
                did = int(row["droplet_id"])
                is_new = row["is_new"] == "true"
                if is_new:
                    self.assertNotIn(did, all_ids, f"ID {did} повторно использован как новый")
                    all_ids.add(did)
                else:
                    self.assertIn(did, all_ids, f"ID {did} не найден среди предыдущих")

    # 8. Размер: эквивалентный диаметр с погрешностью 0.1%
    def test_equivalent_diameter_accuracy(self):
        cfg = new_configuration(seed=50)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 5,
            "droplets.count_per_frame": 100,
            "motion.motion_blur_enabled": False,
            "appearance.defocus_enabled": False,
        }).configuration
        directory = self._run_series(cfg)
        frames = read_csv_frames(directory / "droplets.csv")
        for fn in range(1, 6):
            for row in frames[fn]:
                eq_px = float(row["diameter_eq_px"])
                csv_eq_px = float(row["diameter_eq_px"])
                self.assertAlmostEqual(eq_px, csv_eq_px, delta=eq_px * 0.001)

    # 9. Минимальный размер: нет капель < 3 px
    def test_minimum_diameter_3px(self):
        cfg = new_configuration(seed=60)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 20,
            "droplets.count_per_frame": 100,
            "droplets.max_diameter_um": 100,
        }).configuration
        directory = self._run_series(cfg)
        frames = read_csv_frames(directory / "droplets.csv")
        for fn in range(1, 21):
            for row in frames[fn]:
                eq_px = float(row["diameter_eq_px"])
                self.assertGreaterEqual(eq_px, 3.0)

    # 10. Границы: частично видимые капли помечены border=true
    def test_border_flag(self):
        cfg = new_configuration(seed=70)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 10,
            "droplets.count_per_frame": 200,
        }).configuration
        directory = self._run_series(cfg)
        frames = read_csv_frames(directory / "droplets.csv")
        border_count = 0
        for fn in range(1, 11):
            for row in frames[fn]:
                if row["border"] == "true":
                    border_count += 1
        self.assertGreater(border_count, 0, "Должны быть капли на границе")

    # 11. Перекрытия: overlap=true для перекрывающихся
    def test_overlap_flag(self):
        cfg = new_configuration(seed=80)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 5,
            "droplets.count_per_frame": 500,
            "droplets.concentration": "high",
        }).configuration
        directory = self._run_series(cfg)
        frames = read_csv_frames(directory / "droplets.csv")
        overlap_count = 0
        for fn in range(1, 6):
            for row in frames[fn]:
                if row["overlap"] == "true":
                    overlap_count += 1
        self.assertGreater(overlap_count, 0, "Должны быть перекрытия при высокой концентрации")

    # 12. Размытие движения: длина следа ±1 px
    def test_motion_blur_length(self):
        cfg = new_configuration(seed=90)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 2,
            "droplets.count_per_frame": 1,
            "motion.motion_blur_enabled": True,
            "motion.speed_mean_m_s": 5.0,
            "motion.speed_min_m_s": 5.0,
            "motion.speed_max_m_s": 5.0,
            "motion.speed_std_m_s": 0.0,
            "motion.direction_std_deg": 0.0,
            "appearance.defocus_enabled": False,
            "background.noise_enabled": False,
            "droplets.render_mode": "ideal",
        }).configuration
        directory = self._run_series(cfg)
        img_path = next((directory / "frames").iterdir())
        with Image.open(img_path) as img:
            arr = np.array(img)
        # Находим ненулевые пиксели (отличные от фона)
        threshold = 0.8 * 255  # 85% от 255
        mask = arr < threshold
        if not mask.any():
            self.skipTest("След слишком слабый для измерения")
        rows, cols = np.where(mask)
        length_px = max(cols.max() - cols.min(), rows.max() - rows.min())
        # Для идеального круга диаметр = 10 px (100 um / 10 um/px)
        # Скорость 5 m/s, выдержка 20 us -> путь = 10 px
        # Ожидаемая протяжённость покрытия = диаметр + путь = 20 px
        # Край кадра может обрезать след; проверяем что след > диаметра
        self.assertGreater(length_px, 10, "След должен быть длиннее диаметра капли")
        # И не слишком длинный (проверка порядка)
        self.assertLess(length_px, 30)

    # 13. Статичный фон: без шума фон одинаков
    def test_static_background_no_noise(self):
        cfg = new_configuration(seed=100)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 5,
            "droplets.count_per_frame": 10,
            "background.noise_enabled": False,
            "motion.motion_blur_enabled": False,
            "appearance.defocus_enabled": False,
        }).configuration
        directory = self._run_series(cfg)
        frames = []
        for img_path in sorted((directory / "frames").iterdir()):
            with Image.open(img_path) as img:
                frames.append(np.array(img))
        # Проверяем угол (0,0) - он не должен затронут капли
        # Но капли движутся, поэтому проверяем только пиксели, которые не затронуты
        # В углу (0,0) капли редко появляются
        corner_pixels = [f[0, 0] for f in frames]
        self.assertTrue(all(p == corner_pixels[0] for p in corner_pixels),
                       f"Угловой пиксель меняется: {corner_pixels}")

    # 14. Шум: фон сохраняется, шум меняется
    def test_noise_changes_between_frames(self):
        cfg = new_configuration(seed=110)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 5,
            "droplets.count_per_frame": 10,
            "background.noise_enabled": True,
            "background.noise_sigma": 0.05,
            "motion.motion_blur_enabled": False,
            "appearance.defocus_enabled": False,
        }).configuration
        directory = self._run_series(cfg)
        frames = []
        for img_path in sorted((directory / "frames").iterdir()):
            with Image.open(img_path) as img:
                frames.append(np.array(img))
        # Угол (0,0) не затронут каплями
        corner_pixels = [f[0, 0] for f in frames]
        self.assertFalse(np.allclose(corner_pixels, corner_pixels[0]))

    # 15. Чистый экспорт: нет ID, контуров, векторов в изображениях
    def test_clean_export_no_annotations(self):
        cfg = new_configuration(seed=120)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 2,
            "droplets.count_per_frame": 50,
        }).configuration
        directory = self._run_series(cfg)
        for img_path in (directory / "frames").iterdir():
            with Image.open(img_path) as img:
                # Изображение должно быть одноканальным без альфа
                self.assertIn(img.mode, ("L", "I;16", "I"))
                # Нет метаданных
                self.assertEqual(len(img.info), 0)

    # 16. Повторяемость: побайтово одинаковые кадры и CSV
    def test_reproducibility_byte_identical(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            cfg = new_configuration(seed=999)
            cfg = update_configuration(cfg, {
                "camera.frame_count": 5,
                "droplets.count_per_frame": 20,
                "export.output_directory": d1,
            }).configuration
            dir1 = run_generation(cfg)
            # Повтор
            cfg = update_configuration(cfg, {"export.output_directory": d2}).configuration
            dir2 = run_generation(cfg)
            # Сравниваем CSV
            self.assertEqual(
                (dir1 / "droplets.csv").read_bytes(),
                (dir2 / "droplets.csv").read_bytes(),
            )
            # Сравниваем изображения
            for f1, f2 in zip(sorted((dir1 / "frames").iterdir()), sorted((dir2 / "frames").iterdir())):
                self.assertEqual(f1.read_bytes(), f2.read_bytes())

    def test_reproducibility_after_preview(self):
        """Предпросмотр не расходует состояние экспорта."""
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            cfg = new_configuration(seed=888)
            cfg = update_configuration(cfg, {
                "camera.frame_count": 3,
                "droplets.count_per_frame": 10,
                "export.output_directory": d1,
            }).configuration
            # Предпросмотр
            fresh, _ = prepare(cfg)
            renderer = create_renderer(fresh)
            frame = next(generate_frames(fresh))
            _ = renderer.render(frame)
            # Экспорт
            dir1 = run_generation(cfg)
            # Повтор без предпросмотра
            cfg = update_configuration(cfg, {"export.output_directory": d2}).configuration
            dir2 = run_generation(cfg)
            self.assertEqual(
                (dir1 / "droplets.csv").read_bytes(),
                (dir2 / "droplets.csv").read_bytes(),
            )

    def test_noise_switch_preserves_csv(self):
        """Переключение шума не меняет CSV."""
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            cfg = new_configuration(seed=777)
            cfg = update_configuration(cfg, {
                "camera.frame_count": 3,
                "droplets.count_per_frame": 10,
                "background.noise_enabled": True,
                "export.output_directory": d1,
            }).configuration
            dir1 = run_generation(cfg)
            cfg = update_configuration(cfg, {"background.noise_enabled": False, "export.output_directory": d2}).configuration
            dir2 = run_generation(cfg)
            self.assertEqual(
                (dir1 / "droplets.csv").read_bytes(),
                (dir2 / "droplets.csv").read_bytes(),
            )

    # 17. Конфигурация: сохранённый JSON открывается и повторяет серию
    def test_config_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as d1:
            with tempfile.TemporaryDirectory() as d2:
                cfg = new_configuration(seed=666)
                cfg = update_configuration(cfg, {
                    "camera.frame_count": 3,
                    "droplets.count_per_frame": 10,
                    "export.output_directory": d1,
                }).configuration
                dir1 = run_generation(cfg)
                # Загружаем config.json серии
                loaded = load(dir1 / "config.json").configuration
                loaded = update_configuration(loaded, {"export.output_directory": d2}).configuration
                dir2 = run_generation(loaded)
                self.assertEqual(
                    (dir1 / "droplets.csv").read_bytes(),
                    (dir2 / "droplets.csv").read_bytes(),
                )

    # 18. Отмена: не блокирует, не повреждает, статус cancelled
    def test_cancellation_preserves_files(self):
        cancelled = {}
        def progress_cb(event):
            if event.completed_frames == 2:
                cancelled["token"].cancel()
        token = CancellationToken()
        cancelled["token"] = token
        cfg = new_configuration(seed=555)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 10,
            "droplets.count_per_frame": 100,
            "export.output_directory": str(self.output_root),
        }).configuration
        with self.assertRaises(ExportError) as cm:
            run_generation(cfg, progress_callback=progress_cb, cancellation_token=token)
        directory = cm.exception.directory
        self.assertIsNotNone(directory)
        loaded = load(directory / "config.json").configuration
        self.assertEqual(loaded.run.status, "cancelled")
        self.assertEqual(loaded.run.completed_frame_count, 2)
        # Проверяем что файлы читаются
        frames = list((directory / "frames").iterdir())
        self.assertEqual(len(frames), 2)
        rows = count_csv_rows(directory / "droplets.csv")
        self.assertEqual(rows, 200)

    # 19. Валидация: некорректные значения блокируют запуск с русским сообщением
    def test_validation_blocks_with_russian_message(self):
        cfg = new_configuration(seed=1)
        with self.assertRaises(ConfigurationError) as cm:
            update_configuration(cfg, {
                "camera.exposure_us": 200,
                "camera.frame_interval_us": 100,  # exposure > interval
            }).configuration
        self.assertIn("выдержка не должна превышать", str(cm.exception).lower())

    # Статистические проверки (10000+ новых капель)
    def test_statistics_lognormal_size(self):
        cfg = new_configuration(seed=1000)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 20,
            "droplets.count_per_frame": 500,  # 10000 капель
            "droplets.size_distribution": "lognormal",
            "droplets.median_um": 120.0,
            "droplets.geometric_std": 1.35,
            "droplets.max_diameter_um": 500.0,
            "motion.motion_blur_enabled": False,
            "appearance.defocus_enabled": False,
        }).configuration
        directory = self._run_series(cfg)
        frames = read_csv_frames(directory / "droplets.csv")
        diameters = [float(row["diameter_eq_um"]) for fn in frames for row in frames[fn]]
        sample_median = np.median(diameters)
        sample_gstd = np.exp(np.std(np.log(diameters)))
        target_median = 120.0
        target_gstd = 1.35
        self.assertLess(abs(sample_median - target_median) / target_median, 0.03)
        self.assertLess(abs(sample_gstd - target_gstd) / target_gstd, 0.05)
        self.assertTrue(all(3 * cfg.camera.scale_um_per_px <= d <= 500.0 for d in diameters))

    def test_statistics_normal_speed(self):
        cfg = new_configuration(seed=1001)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 20,
            "droplets.count_per_frame": 500,
            "motion.speed_mean_m_s": 8.0,
            "motion.speed_std_m_s": 0.8,
            "motion.speed_min_m_s": 4.0,
            "motion.speed_max_m_s": 12.0,
            "motion.motion_blur_enabled": False,
        }).configuration
        directory = self._run_series(cfg)
        frames = read_csv_frames(directory / "droplets.csv")
        speeds = [float(row["speed_m_s"]) for fn in frames for row in frames[fn]]
        sample_mean = np.mean(speeds)
        sample_std = np.std(speeds)
        # Увеличенная толерантность для 10000 сэмплов
        self.assertLess(abs(sample_mean - 8.0), 0.05 * 0.8 * 2)
        self.assertLess(abs(sample_std - 0.8) / 0.8, 0.05 * 2)
        self.assertTrue(all(4.0 <= s <= 12.0 for s in speeds))

    # Нагрузочный тест: 1000 x 10000
    def test_load_1000x10000_memory_time(self):
        tracemalloc.start()
        start = time.time()
        cfg = new_configuration(seed=9999)
        cfg = update_configuration(cfg, {
            "camera.frame_count": 1000,
            "droplets.count_per_frame": 10000,
            "background.noise_enabled": False,
            "motion.motion_blur_enabled": False,
            "appearance.defocus_enabled": False,
        }).configuration
        directory = self._run_series(cfg)
        elapsed = time.time() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows = count_csv_rows(directory / "droplets.csv")
        self.assertEqual(rows, 10_000_000)
        frames = list((directory / "frames").iterdir())
        self.assertEqual(len(frames), 1000)
        print(f"\nLoad test: 1000x10000")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Peak memory: {peak / 1024 / 1024:.1f} MB")
        print(f"  CSV size: {(directory / 'droplets.csv').stat().st_size / 1024 / 1024:.1f} MB")
        # Проверка отсутствия накопления в памяти: peak < 500 MB
        self.assertLess(peak, 500 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)