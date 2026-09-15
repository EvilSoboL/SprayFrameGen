from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sprayframegen import __version__
from sprayframegen.configuration import (
    ConfigurationError, dumps, from_mapping, load, loads, new_configuration, save,
    update_configuration,
)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = new_configuration(seed=123456789)

    def changed(self, **changes):
        return update_configuration(self.config, changes).configuration

    def test_defaults_match_contract(self):
        example = json.loads(Path("docs/examples/config.default.json").read_text())
        self.assertEqual(asdict(self.config), example)

    def test_seed_generated_only_for_new_configuration(self):
        with patch("sprayframegen.configuration.secrets.randbits", return_value=42) as rng:
            self.assertEqual(new_configuration().seed, 42)
            self.assertEqual(new_configuration(seed=0).seed, 0)
            loads(dumps(self.config))
            rng.assert_called_once_with(32)

    def test_roundtrip_all_parameters_and_inactive_settings(self):
        config = self.changed(**{
            "seed": 4294967295, "camera.format": "tiff", "camera.bit_depth": 16,
            "droplets.size_distribution": "normal", "droplets.mean_um": 130,
            "droplets.median_um": 140, "appearance.defocus_enabled": False,
            "appearance.blur_sigma_min_px": 1, "appearance.blur_sigma_max_px": 3,
            "motion.motion_blur_enabled": False, "background.noise_enabled": False,
            "background.noise_sigma": 0.3, "export.output_directory": "C:\\Серии",
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "настройки.json"
            save(config, path)
            self.assertEqual(load(path).configuration, config)
            self.assertNotIn(b"\r\n", path.read_bytes())

    def test_missing_every_required_field(self):
        data = asdict(self.config)
        for key, value in data.items():
            with self.subTest(key=key):
                copy = dict(data)
                del copy[key]
                with self.assertRaisesRegex(ConfigurationError, key):
                    from_mapping(copy)
            if isinstance(value, dict):
                for nested in value:
                    with self.subTest(key=f"{key}.{nested}"):
                        copy = asdict(self.config)
                        del copy[key][nested]
                        with self.assertRaisesRegex(ConfigurationError, f"{key}.{nested}"):
                            from_mapping(copy)

    def test_unknown_fields_warn_and_are_removed(self):
        data = asdict(self.config)
        data["future"] = {"anything": 1}
        data["camera"]["future"] = 1
        result = from_mapping(data)
        self.assertEqual(len(result.warnings), 2)
        self.assertEqual(result.configuration, self.config)

    def test_schema_and_profiles(self):
        for path, value in [("schema_version", "2.0"), ("schema_version", "garbage"),
                            ("appearance.profile", "unknown")]:
            with self.subTest(path=path, value=value), self.assertRaises(ConfigurationError):
                self.changed(**{path: value})
        result = update_configuration(self.config, {"schema_version": "1.2"})
        self.assertTrue(result.warnings)

    def test_saving_writes_actual_version(self):
        config = self.changed(program_version="old-build")
        self.assertTrue(from_mapping(asdict(config)).warnings)
        self.assertEqual(json.loads(dumps(config))["program_version"], __version__)

    def test_numeric_types_and_finiteness(self):
        for group in ("camera", "droplets", "motion", "appearance", "background"):
            for key, original in asdict(self.config)[group].items():
                if type(original) not in (int, float):
                    continue
                for bad in (True, "1", float("nan"), float("inf"), -float("inf"), 10 ** 1000):
                    with self.subTest(path=f"{group}.{key}", bad=str(bad)[:20]):
                        with self.assertRaises(ConfigurationError):
                            self.changed(**{f"{group}.{key}": bad})

    def test_seed_and_integer_bounds(self):
        for seed in (0, 4294967295):
            self.assertEqual(self.changed(seed=seed).seed, seed)
        for seed in (-1, 4294967296, "1", True, 1.0, 1.5):
            with self.subTest(seed=seed), self.assertRaisesRegex(ConfigurationError, "seed"):
                self.changed(seed=seed)
        for path, value in [("camera.frame_count", 1), ("camera.frame_count", 1001),
                            ("droplets.count_per_frame", 0), ("droplets.count_per_frame", 10001),
                            ("camera.frame_count", 20.0)]:
            with self.assertRaises(ConfigurationError):
                self.changed(**{path: value})

    def test_resolutions_and_formats(self):
        for width, height in [(640, 480), (1024, 768), (1280, 1024), (1920, 1080)]:
            self.changed(**{"camera.width_px": width, "camera.height_px": height})
        for fmt, depth in [("png", 8), ("png", 16), ("tiff", 16)]:
            self.changed(**{"camera.format": fmt, "camera.bit_depth": depth})
        for change in [{"camera.width_px": 1024}, {"camera.format": "tiff"},
                       {"camera.bit_depth": 32}, {"camera.format": "jpeg"}]:
            with self.assertRaises(ConfigurationError):
                self.changed(**change)

    def test_invalid_ranges(self):
        for path, value in [("camera.exposure_us", 101), ("camera.exposure_us", 0),
                            ("camera.scale_um_per_px", 0), ("camera.frame_interval_us", -1),
                            ("droplets.std_um", -1), ("droplets.geometric_std", 0.9),
                            ("droplets.median_um", 0), ("droplets.max_diameter_um", 29),
                            ("motion.speed_min_m_s", 0), ("motion.speed_std_m_s", -1),
                            ("motion.direction_std_deg", -1), ("motion.speed_max_m_s", 3),
                            ("appearance.blur_sigma_min_px", -1),
                            ("appearance.blur_sigma_max_px", -1), ("background.noise_sigma", -1)]:
            with self.subTest(path=path), self.assertRaises(ConfigurationError):
                self.changed(**{path: value})

    def test_degenerate_distributions(self):
        cases = [
            {"droplets.size_distribution": "normal", "droplets.mean_um": 30,
             "droplets.std_um": 0, "droplets.max_diameter_um": 30},
            {"droplets.median_um": 30, "droplets.geometric_std": 1,
             "droplets.max_diameter_um": 30},
            {"motion.speed_min_m_s": 2, "motion.speed_max_m_s": 2,
             "motion.speed_mean_m_s": 2, "motion.speed_std_m_s": 0},
        ]
        for case in cases:
            self.changed(**case)
        for changes in [{"droplets.std_um": 0, "droplets.mean_um": 1,
                         "droplets.size_distribution": "normal"},
                        {"droplets.geometric_std": 1, "droplets.median_um": 1},
                        {"droplets.max_diameter_um": 30},
                        {"motion.speed_std_m_s": 0, "motion.speed_mean_m_s": 1},
                        {"motion.speed_min_m_s": 8, "motion.speed_max_m_s": 8}]:
            with self.subTest(changes=changes), self.assertRaises(ConfigurationError):
                self.changed(**changes)
        # Nonzero spread outside the interval is not a validation failure.
        self.changed(**{"droplets.mean_um": -500, "droplets.median_um": 1000,
                        "motion.speed_mean_m_s": -10})
        self.changed(**{"droplets.std_um": 0, "droplets.mean_um": -1})  # inactive normal law

    def test_derived_overflow(self):
        for changes in [{"camera.frame_interval_us": 1e308},
                        {"camera.scale_um_per_px": 1e-308},
                        {"motion.speed_max_m_s": 1e308},
                        {"appearance.blur_sigma_max_px": 1e308}]:
            with self.subTest(changes=changes), self.assertRaises(ConfigurationError):
                self.changed(**changes)

    def test_defocus_mode_switch_and_equal_bounds(self):
        self.assertFalse(new_configuration(render_mode="ideal").appearance.defocus_enabled)
        config = self.changed(**{"droplets.render_mode": "ideal"})
        self.assertTrue(loads(dumps(config)).configuration.appearance.defocus_enabled)
        self.changed(**{"appearance.blur_sigma_min_px": 2, "appearance.blur_sigma_max_px": 2})
        self.changed(**{"appearance.blur_sigma_max_px": 0})
        self.assertIsNone(self.config.export.output_directory)

    def test_paths_and_json_errors(self):
        for value in ("", "relative", "C:relative", "C:\\bad\x00name"):
            with self.assertRaises(ConfigurationError):
                self.changed(**{"export.output_directory": value})
        for text in ("{", "[]", "null", "true"):
            with self.assertRaises(ConfigurationError):
                loads(text)


if __name__ == "__main__":
    unittest.main()
