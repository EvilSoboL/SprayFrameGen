from dataclasses import asdict
import io
import unittest

import numpy as np
from PIL import Image

from sprayframegen.configuration import ConfigurationError, from_mapping, loads, dumps, new_configuration
from sprayframegen.environment import current_environment
from sprayframegen.generation import prepare
from sprayframegen.generation.random_streams import RandomStreams


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.data = asdict(new_configuration(seed=123))
        self.data["run"] = {
            "status": "cancelled", "started_at": "2026-09-14T20:00:00+07:00",
            "completed_frame_count": 3, "output_directory": "C:\\Серии\\series_123",
            "environment": asdict(current_environment()),
        }

    def test_metadata_roundtrip_and_restart(self):
        result = from_mapping(self.data)
        self.assertFalse(result.warnings)
        config = loads(dumps(result.configuration)).configuration
        self.assertEqual(config.run, result.configuration.run)
        fresh, streams = prepare(config)
        self.assertIsNone(fresh.run)
        self.assertEqual(fresh.seed, 123)
        self.assertEqual(streams.model.random(), RandomStreams.from_seed(123).model.random())

    def test_environment_comparison(self):
        for key in ("build_id", "os", "processor", "architecture"):
            original = self.data["run"]["environment"][key]
            self.data["run"]["environment"][key] = "different"
            self.assertTrue(from_mapping(self.data).warnings)
            self.data["run"]["environment"][key] = original
        self.data["run"]["environment"]["dependencies"]["numpy"] = "different"
        self.assertTrue(from_mapping(self.data).warnings)

    def test_invalid_metadata(self):
        for path, value in [
            (("status",), "unknown"), (("started_at",), "2026-09-14"),
            (("completed_frame_count",), -1), (("completed_frame_count",), True),
            (("completed_frame_count",), 21), (("output_directory",), "relative"),
            (("environment", "numerical_profile"), "unknown"),
            (("environment", "build_id"), ""), (("environment", "dependencies"), {}),
        ]:
            with self.subTest(path=path):
                target = self.data["run"]
                for key in path[:-1]:
                    target = target[key]
                previous = target[path[-1]]
                target[path[-1]] = value
                with self.assertRaises(ConfigurationError):
                    from_mapping(self.data)
                target[path[-1]] = previous
        self.data["run"]["status"] = "completed"
        with self.assertRaises(ConfigurationError):
            from_mapping(self.data)
        self.data["run"]["completed_frame_count"] = 20
        from_mapping(self.data)

    def test_missing_metadata_fields(self):
        def check(data):
            for key, value in list(data.items()):
                del data[key]
                with self.subTest(key=key), self.assertRaises(ConfigurationError):
                    from_mapping(self.data)
                data[key] = value
                if isinstance(value, dict) and key != "dependencies":
                    check(value)
        check(self.data["run"])

    def test_model_and_noise_are_independent(self):
        first = RandomStreams.from_seed(123)
        other = RandomStreams.from_seed(123)
        first.noise.normal(size=1000)
        np.testing.assert_array_equal(first.model.random(100), other.model.random(100))
        self.assertFalse(np.array_equal(other.model.random(100), other.noise.random(100)))

    def test_preview_does_not_consume_export(self):
        config = new_configuration(seed=123)
        _, first = prepare(config)
        expected = first.model.normal(size=100)
        _, preview = prepare(config)
        preview.model.normal(size=10000)
        preview.noise.normal(size=10000)
        _, exported = prepare(config)
        np.testing.assert_array_equal(expected, exported.model.normal(size=100))

    def test_selected_codecs_preserve_single_channel_values(self):
        for fmt, dtype, options in [("PNG", np.uint8, {"compress_level": 6}),
                                   ("PNG", np.uint16, {"compress_level": 6}),
                                   ("TIFF", np.uint16, {"compression": "raw"})]:
            with self.subTest(format=fmt, dtype=dtype):
                array = np.array([[0, 1, 127], [128, 255, np.iinfo(dtype).max]], dtype=dtype)
                buffer = io.BytesIO()
                Image.fromarray(array).save(buffer, format=fmt, **options)
                buffer.seek(0)
                with Image.open(buffer) as image:
                    self.assertEqual(len(image.getbands()), 1)
                    np.testing.assert_array_equal(np.asarray(image), array)


if __name__ == "__main__":
    unittest.main()
