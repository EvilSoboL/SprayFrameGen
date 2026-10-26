import math
import unittest

from sprayframegen.ui.parameter_units import (
    format_editable,
    format_reference,
    hz_to_interval_us,
    interval_us_to_hz,
    micrometers_to_pixels,
    parse_number,
    pixels_to_micrometers,
    speed_to_pixels_per_frame,
)


class ParameterUnitTests(unittest.TestCase):
    def test_interval_roundtrip_and_precision(self):
        self.assertEqual(interval_us_to_hz(100), 10000)
        self.assertEqual(hz_to_interval_us(10000), 100)
        value = 123.456789
        self.assertTrue(math.isclose(hz_to_interval_us(interval_us_to_hz(value)), value,
                                     rel_tol=1e-12, abs_tol=1e-12))
        original = value
        for _ in range(1000):
            value = hz_to_interval_us(interval_us_to_hz(value))
        self.assertEqual(value, original)

    def test_size_roundtrip_and_speed_examples(self):
        self.assertEqual(micrometers_to_pixels(120, 10), 12)
        self.assertEqual(pixels_to_micrometers(12, 10), 120)
        self.assertEqual(speed_to_pixels_per_frame(8, 100, 10), 80)
        self.assertEqual(speed_to_pixels_per_frame(8, 200, 10), 160)
        self.assertEqual(speed_to_pixels_per_frame(8, 100, 20), 40)
        self.assertEqual(format_reference(80), "80")

    def test_invalid_and_underflow_values_are_rejected(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                interval_us_to_hz(value)
        with self.assertRaises(ValueError):
            hz_to_interval_us(1e-320)
        with self.assertRaises(ValueError):
            parse_number("1e", "Частота")
        with self.assertRaisesRegex(ValueError, "точку"):
            parse_number("1,5", "Частота")
        self.assertEqual(parse_number(" 1e4 ", "Частота"), 10000)
        self.assertEqual(float(format_editable(1 / 3)), 1 / 3)


if __name__ == "__main__":
    unittest.main()
