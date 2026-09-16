from dataclasses import replace
from datetime import datetime, timezone
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from sprayframegen.configuration import load, ConfigurationError
from sprayframegen.environment import current_environment
from sprayframegen.export import export_series, ExportError
from sprayframegen.export.csv_data import COLUMNS, number, row
from sprayframegen.generation import generate_frames
from sprayframegen.generation.sequence import Frame
from test_render import ideal_config, state_at


class ExportTests(unittest.TestCase):
    def config(self, directory, **changes):
        return ideal_config(**{"export.output_directory": str(Path(directory).resolve()), **changes})

    def test_all_resolutions_formats_and_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            for width, height in ((640, 480), (1024, 768), (1280, 1024), (1920, 1080)):
                for fmt, bits in (("png", 8), ("png", 16), ("tiff", 16)):
                    with self.subTest(size=(width, height), fmt=fmt, bits=bits):
                        config = self.config(directory, **{
                            "camera.width_px": width, "camera.height_px": height,
                            "camera.format": fmt, "camera.bit_depth": bits,
                        })
                        result = export_series(config)
                        files = sorted((result / "frames").iterdir())
                        extension = "png" if fmt == "png" else "tif"
                        self.assertEqual([p.name for p in files],
                                         [f"frame_000001.{extension}", f"frame_000002.{extension}"])
                        for path in files:
                            with Image.open(path) as image:
                                self.assertEqual(image.size, (width, height))
                                self.assertEqual(len(image.getbands()), 1)
                                values = np.array(image, copy=True)
                                self.assertEqual(values.dtype, np.uint8 if bits == 8 else np.uint16)
                                self.assertEqual(int(values[0, 0]), int(np.floor(.85 * ((1 << bits) - 1) + .5)))
                                self.assertNotIn("gamma", image.info)
                                self.assertNotIn("transparency", image.info)
                                if fmt == "tiff":
                                    self.assertEqual(image.tag_v2[259], 1)  # Compression=None
                                    self.assertNotIn(306, image.tag_v2)  # DateTime
                                image.close()  # TIFF может держать mmap после выхода из контекста.
                        with (result / "droplets.csv").open(encoding="utf-8", newline="") as stream:
                            rows = list(csv.reader(stream))
                        self.assertEqual(rows[0], list(COLUMNS))
                        self.assertEqual(len(rows), 21)
                        self.assertTrue(all(len(r) == 19 for r in rows))

    def test_csv_matches_every_model_field_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            destination = export_series(config)
            data = (destination / "droplets.csv").read_bytes()
            self.assertFalse(data.startswith(b"\xef\xbb\xbf"))
            self.assertNotIn(b"\r", data)
            self.assertTrue(data.endswith(b"\n"))
            rows = list(csv.DictReader(data.decode("utf-8").splitlines()))
            for frame in generate_frames(config):
                for state in frame.droplets:
                    entry = rows.pop(0)
                    d = state.droplet
                    self.assertEqual(int(entry["frame_number"]), frame.frame_number)
                    self.assertEqual(float(entry["time_us"]), frame.time_us)
                    self.assertEqual(int(entry["droplet_id"]), d.droplet_id)
                    for key in ("center_x_px", "center_y_px", "diameter_eq_px", "diameter_eq_um",
                                "angle_deg", "vx_m_s", "vy_m_s", "speed_m_s", "blur_sigma_px"):
                        self.assertEqual(float(entry[key]), getattr(d, key))
                    for key in ("semi_major_px", "semi_minor_px"):
                        self.assertEqual(float(entry[key]), getattr(state, key))
                    for key in ("is_new", "border", "overlap"):
                        self.assertEqual(entry[key], "true" if getattr(state, key) else "false")
                    self.assertEqual(float(entry["center_x_mm"]), d.center_x_px * 10 / 1000)
                    self.assertEqual(float(entry["center_y_mm"]), d.center_y_px * 10 / 1000)
            self.assertFalse(rows)
            saved = load(destination / "config.json")
            self.assertFalse(saved.warnings)
            self.assertEqual(replace(saved.configuration, run=None), config)
            run = saved.configuration.run
            self.assertEqual((run.status, run.completed_frame_count), ("completed", 2))
            self.assertEqual(run.output_directory, str(destination))
            self.assertIsNotNone(datetime.fromisoformat(run.started_at).utcoffset())
            self.assertEqual(run.environment, current_environment())

    def test_reference_row_and_negative_zero(self):
        state = state_at()
        values = row(Frame(1, 0, (state,)), state, 10)
        self.assertEqual(",".join(values), "1,0,1,true,100,100,1,1,10,100,5,5,0,2,0,2,0,false,false")
        self.assertEqual(number(-0.0), "0")
        self.assertEqual(float(number(1.2345678901234567)), 1.2345678901234567)

    def test_fully_hidden_droplet_has_its_own_csv_row(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, **{"droplets.count_per_frame": 2})
            first = replace(state_at(), overlap=True)
            second = replace(state_at(identifier=2), overlap=True)
            frames = (Frame(1, 0, (first, second)), Frame(2, 100, (
                replace(first, is_new=False), replace(second, is_new=False))))
            with patch("sprayframegen.export.series.generate_frames", return_value=iter(frames)):
                destination = export_series(config)
            rows = list(csv.DictReader((destination / "droplets.csv").read_text().splitlines()))
            self.assertEqual([r["droplet_id"] for r in rows], ["1", "2", "1", "2"])
            self.assertTrue(all(r["overlap"] == "true" for r in rows))

    def test_replay_json_is_byte_identical_for_all_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            for fmt, bits in (("png", 8), ("png", 16), ("tiff", 16)):
                config = self.config(directory, **{"camera.format": fmt, "camera.bit_depth": bits})
                first = export_series(config)
                restored = load(first / "config.json").configuration
                next(generate_frames(restored))  # partial preview before independent export
                second = export_series(restored)
                self.assertNotEqual(first, second)
                for file in (first / "frames").iterdir():
                    self.assertEqual(file.read_bytes(), (second / "frames" / file.name).read_bytes())
                self.assertEqual((first / "droplets.csv").read_bytes(), (second / "droplets.csv").read_bytes())

    def test_collision_uses_first_free_suffix_without_overwrite(self):
        fixed = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc).astimezone()
        with tempfile.TemporaryDirectory() as directory, patch("sprayframegen.export.series.datetime") as clock:
            clock.now.return_value = fixed
            config = self.config(directory)
            first = export_series(config)
            self.assertEqual(first.name, f"series_{fixed:%Y%m%d_%H%M%S}_123")
            occupied = first.with_name(first.name + "_001")
            occupied.write_text("сохранить", encoding="utf-8")
            second = export_series(config)
            self.assertEqual(second.name, first.name + "_002")
            occupied.unlink()
            third = export_series(config)
            self.assertEqual(third.name, first.name + "_001")
            self.assertTrue((first / "config.json").exists())

    def test_invalid_settings_do_not_create_series(self):
        with tempfile.TemporaryDirectory() as directory:
            for changes in ({"export.output_directory": None},
                            {"export.output_directory": str(Path(directory) / "missing")}):
                with self.assertRaises(ConfigurationError):
                    export_series(self.config(directory, **changes))
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_image_write_error_keeps_only_completed_frame(self):
        original = Image.Image.save
        count = 0
        def fail_second(image, *args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("имитация ошибки записи")
            return original(image, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Image.Image, "save", fail_second), self.assertRaises(ExportError):
                export_series(self.config(directory))
            result = next(Path(directory).iterdir())
            config = load(result / "config.json").configuration
            self.assertEqual((config.run.status, config.run.completed_frame_count), ("error", 1))
            self.assertEqual([p.name for p in (result / "frames").iterdir()], ["frame_000001.png"])
            self.assertEqual(len((result / "droplets.csv").read_text().splitlines()), 11)

    def test_export_commits_each_frame_before_requesting_next(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            original = generate_frames(config)
            def checked_frames(_):
                yield next(original)
                result = next(Path(directory).iterdir())
                self.assertTrue((result / "frames" / "frame_000001.png").is_file())
                self.assertEqual(load(result / "config.json").configuration.run.completed_frame_count, 1)
                self.assertEqual(len((result / "droplets.csv").read_text().splitlines()), 11)
                yield next(original)
            with patch("sprayframegen.export.series.generate_frames", checked_frames):
                export_series(config)


if __name__ == "__main__":
    unittest.main()
