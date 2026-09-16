from dataclasses import replace
import unittest

import numpy as np

from sprayframegen.configuration import new_configuration, update_configuration, ConfigurationError
from sprayframegen.generation.sequence import Frame, FrameDroplet
from sprayframegen.model import Droplet
from sprayframegen.render import IdealRenderer, encode_intensity


def ideal_config(**changes):
    base = new_configuration(seed=123, render_mode="ideal")
    return update_configuration(base, {
        "motion.motion_blur_enabled": False, "background.noise_enabled": False,
        "camera.frame_count": 2, "droplets.count_per_frame": 10, **changes,
    }).configuration


def state_at(x=100, y=100, radius=5, identifier=1):
    droplet = Droplet(identifier, x, y, radius * 20, radius * 2, False, 0,
                      2, 2, -0.0, 1, 0, 0)
    return FrameDroplet(droplet, radius, radius, True, x < radius, False)


class IdealRenderTests(unittest.TestCase):
    def test_brightness_polarity_and_no_annotations(self):
        frame = Frame(1, 0, (state_at(),))
        light = IdealRenderer(ideal_config()).render(frame)
        dark = IdealRenderer(ideal_config(**{"background.tone": "dark"})).render(frame)
        self.assertEqual(light.dtype, np.float64)
        self.assertEqual(light.shape, (480, 640))
        self.assertEqual(light[100, 100], .1)
        self.assertEqual(light[0, 0], .85)
        self.assertEqual(dark[100, 100], .9)
        self.assertEqual(dark[0, 0], .15)
        np.testing.assert_allclose(light + dark, 1, atol=2e-16, rtol=0)
        altered = replace(frame, droplets=(replace(frame.droplets[0], is_new=False,
                           border=True, overlap=True, droplet=replace(frame.droplets[0].droplet,
                           droplet_id=999, vx_m_s=100, vy_m_s=30)),))
        np.testing.assert_array_equal(light, IdealRenderer(ideal_config()).render(altered))

    def test_4x4_against_scalar_subpixel_reference_and_clipping(self):
        for x, y in ((7.37, 8.19), (0, 0), (-1, 5.4), (639.6, 479.8)):
            radius = 2.3
            frame = Frame(1, 0, (state_at(x, y, radius),))
            actual = IdealRenderer(ideal_config()).render(frame)
            expected = np.full((480, 640), .85, dtype=np.float64)
            for i in range(max(0, int(y) - 4), min(480, int(y) + 5)):
                for j in range(max(0, int(x) - 4), min(640, int(x) + 5)):
                    count = sum((j + (kx + .5) / 4 - x) ** 2 +
                                (i + (ky + .5) / 4 - y) ** 2 <= radius ** 2
                                for kx in range(4) for ky in range(4))
                    expected[i, j] = count / 16 * .1 + (1 - count / 16) * .85
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(np.any((actual > .1) & (actual < .85)))

    def test_composition_uses_averaged_layer_alpha(self):
        first, second = state_at(100.4, 100.2, 2.3), state_at(101.1, 100.7, 2.3, 2)
        renderer = IdealRenderer(ideal_config())
        a = renderer.render(Frame(1, 0, (first,)))
        b = renderer.render(Frame(1, 0, (second,)))
        combined = renderer.render(Frame(1, 0, (second, first)))
        alpha_b = (.85 - b) / .75
        np.testing.assert_allclose(combined, alpha_b * .1 + (1 - alpha_b) * a, atol=2e-16, rtol=0)

    def test_quantization_clipping_and_half_up(self):
        for bits, dtype in ((8, np.uint8), (16, np.uint16)):
            limit = (1 << bits) - 1
            values = np.array([[-1, 0, .5 / limit, 1.5 / limit, .5, 1, 2]], dtype=np.float64)
            result = encode_intensity(values, bits)
            self.assertEqual(result.dtype, dtype)
            np.testing.assert_array_equal(result, [[0, 0, 1, 2, (limit + 1) // 2, limit, limit]])
        with self.assertRaises(ValueError):
            encode_intensity(np.array([[np.nan]]), 8)

    def test_ideal_renderer_requires_ideal_mode(self):
        with self.assertRaisesRegex(ConfigurationError, "create_renderer"):
            IdealRenderer(ideal_config(**{"droplets.render_mode": "realistic"}))
        IdealRenderer(ideal_config(**{"background.noise_enabled": True, "background.noise_sigma": 0,
                                     "appearance.defocus_enabled": True, "appearance.blur_sigma_max_px": 0}))

    def test_exposure_disabled_and_huge_radius(self):
        frame = Frame(1, 0, (state_at(),))
        np.testing.assert_array_equal(IdealRenderer(ideal_config()).render(frame),
            IdealRenderer(ideal_config(**{"camera.exposure_us": 100})).render(frame))
        image = IdealRenderer(ideal_config()).render(Frame(1, 0, (state_at(radius=1e150),)))
        self.assertTrue(np.all(image == .1))


if __name__ == "__main__":
    unittest.main()
