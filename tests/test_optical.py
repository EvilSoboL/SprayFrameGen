from dataclasses import replace
import csv
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sprayframegen.configuration import update_configuration, load
from sprayframegen.export import export_series
from sprayframegen.generation import generate_frames
from sprayframegen.generation.sequence import Frame
from sprayframegen.render import create_renderer, composite
from sprayframegen.render.optical import optical_layer, temporal_sampling, gaussian_blur, gaussian_kernel
from tests.test_render import ideal_config, state_at


def config(**changes):
    return update_configuration(ideal_config(), changes).configuration


class OpticalTests(unittest.TestCase):
    def test_profile_and_opposite_polarities_before_quantization(self):
        frame = Frame(1, 0, (state_at(radius=20),))
        for tone, inside, rim, background in (("light", .885, .1, .85), ("dark", .115, .9, .15)):
            image = create_renderer(config(**{"droplets.render_mode": "realistic", "background.tone": tone})).render(frame)
            self.assertAlmostEqual(image[100, 100], inside)
            self.assertAlmostEqual(image[100, 118], rim)
            self.assertEqual(image[0, 0], background)

    def test_id_order_with_transparent_interior_and_opaque_rim(self):
        first, second = state_at(radius=20), state_at(x=118, radius=20, identifier=2)
        renderer = create_renderer(config(**{"droplets.render_mode": "realistic"}))
        image = renderer.render(Frame(1, 0, (second, first)))
        self.assertAlmostEqual(image[100, 100], .1)
        first = replace(first, droplet=replace(first.droplet, droplet_id=3))
        reordered = renderer.render(Frame(1, 0, (first, second)))
        self.assertAlmostEqual(reordered[100, 100], .35 * .95 + .65 * .1)

    def test_rotated_profile_against_scalar_samples(self):
        state = state_at(x=100.2, y=100.7, radius=6)
        state = replace(state, droplet=replace(state.droplet, angle_deg=37, q_left=1.35))
        c = config(**{"droplets.render_mode": "realistic"})
        actual = create_renderer(c).render(Frame(1, 0, (state,)))
        a, b = state.droplet.semi_axes_at(100.2, 640)
        cosine, sine = math.cos(math.radians(37)), math.sin(math.radians(37))
        for y in range(92, 109):
            for x in range(92, 109):
                p, alpha = 0, 0
                for ky in range(4):
                    for kx in range(4):
                        dx, dy = x + (kx + .5) / 4 - 100.2, y + (ky + .5) / 4 - 100.7
                        rho = math.hypot((dx * cosine + dy * sine) / a,
                                         (-dx * sine + dy * cosine) / b)
                        if rho <= .8:
                            p += .35 * .95 / 16
                            alpha += .35 / 16
                        elif rho <= 1:
                            p += .1 / 16
                            alpha += 1 / 16
                self.assertAlmostEqual(actual[y, x], p + (1 - alpha) * .85)

    def test_gaussian_kernel_and_separable_impulse(self):
        for sigma in (0, .01, .5, 2, 4):
            kernel = gaussian_kernel(sigma)
            r = math.ceil(4 * sigma)
            self.assertEqual(len(kernel), 2 * r + 1)
            self.assertAlmostEqual(kernel.sum(), 1)
            np.testing.assert_array_equal(kernel, kernel[::-1])
            impulse = np.zeros((2 * r + 3, 2 * r + 3))
            impulse[r + 1, r + 1] = 1
            blurred = gaussian_blur(impulse, kernel)
            expected = np.zeros_like(impulse)
            expected[1:-1, 1:-1] = np.outer(kernel, kernel)
            np.testing.assert_allclose(blurred, expected, atol=1e-16)
        # Ядро шире самого слоя; нулевая граница не перенормируется.
        kernel = gaussian_kernel(2)
        self.assertAlmostEqual(gaussian_blur(np.ones((1, 1)), kernel)[0, 0], kernel[len(kernel)//2] ** 2)

    def test_defocus_uses_geometry_outside_image(self):
        c = config(**{"appearance.defocus_enabled": True})
        edge = state_at(x=0, y=100, radius=5)
        edge = replace(edge, droplet=replace(edge.droplet, blur_sigma_px=2))
        shifted = replace(edge, droplet=replace(edge.droplet, center_x_px=100))
        renderer = create_renderer(c)
        at_edge = renderer.render(Frame(1, 0, (edge,)))
        in_middle = renderer.render(Frame(1, 0, (shifted,)))
        np.testing.assert_allclose(at_edge[80:120, :20], in_middle[80:120, 100:120], atol=1e-15)
        layer = optical_layer(edge, c)
        self.assertLess(layer.x, 0)

    def test_exposure_ignored_with_motion_disabled_in_both_modes(self):
        for mode in ("ideal", "realistic"):
            c = config(**{"droplets.render_mode": mode, "appearance.defocus_enabled": True,
                          "background.noise_enabled": True})
            state = state_at()
            state = replace(state, droplet=replace(state.droplet, blur_sigma_px=1, q_left=1.3))
            frame = Frame(1, 0, (state,))
            changed = update_configuration(c, {"camera.exposure_us": 100}).configuration
            np.testing.assert_array_equal(create_renderer(c).render(frame), create_renderer(changed).render(frame))

    def test_motion_length_and_symmetric_mid_exposure(self):
        c = config(**{"motion.motion_blur_enabled": True})
        for vx, vy in ((3, 4), (3, -4), (5, 0), (0, 5)):
            state = state_at(x=100.5, y=100.5, radius=5)
            state = replace(state, droplet=replace(state.droplet, vx_m_s=vx, vy_m_s=vy))
            n, dx, dy = temporal_sampling(state, c)
            self.assertEqual((n, dx, dy), (40, vx * 2, vy * 2))
            layer = optical_layer(state, c)
            ys, xs = np.nonzero(layer.alpha > 0)
            projection = ((xs + layer.x + .5) * vx + (ys + layer.y + .5) * vy) / 5
            self.assertLessEqual(abs(projection.max() - projection.min() - 10 - 10), 1)
            weights = layer.alpha[ys, xs]
            self.assertAlmostEqual(np.average(xs + layer.x + .5, weights=weights), 100.5)
            self.assertAlmostEqual(np.average(ys + layer.y + .5, weights=weights), 100.5)

    def test_full_coverage_brightness_is_preserved_with_11_time_samples(self):
        c = config(**{"motion.motion_blur_enabled": True})
        state = state_at(radius=20)
        state = replace(state, droplet=replace(state.droplet, vx_m_s=1.3))
        self.assertEqual(temporal_sampling(state, c)[0], 11)
        layer = optical_layer(state, c)
        self.assertEqual(layer.alpha[100 - layer.y, 100 - layer.x], 1)
        self.assertEqual(create_renderer(c).render(Frame(1, 0, (state,)))[100, 100], .1)

    def test_time_averages_shape_at_each_position_including_outside(self):
        moving = config(**{"droplets.render_mode": "realistic", "motion.motion_blur_enabled": True})
        static = update_configuration(moving, {"motion.motion_blur_enabled": False}).configuration
        for center in (0, 320, 641):
            state = state_at(x=center, radius=10)
            state = replace(state, droplet=replace(state.droplet, vx_m_s=10, angle_deg=27, q_left=1.35))
            n, dx, dy = temporal_sampling(state, moving)
            self.assertEqual(n, math.ceil(4 * (20 + 10 * .35 * 20 / 640)))
            expected = np.zeros((480, 640))
            for k in range(n):
                x = center + ((k + .5) / n - .5) * dx
                at_time = replace(state, droplet=replace(state.droplet, center_x_px=x))
                expected += create_renderer(static).render(Frame(1, 0, (at_time,))) / n
            actual = create_renderer(moving).render(Frame(1, 0, (state,)))
            np.testing.assert_allclose(actual, expected, atol=4e-15, rtol=0)
            self.assertEqual(state.droplet.angle_deg, 27)
            if center == 641:
                a, b = state.droplet.semi_axes_at(center, 640)
                self.assertEqual(a, b)

    def test_noise_after_composition_and_repeat_random_access(self):
        noisy = config(**{"background.noise_enabled": True})
        quiet = config()
        renderer = create_renderer(noisy)
        empty = Frame(1, 0, ())
        frame = Frame(1, 0, (state_at(radius=20),))
        noise = renderer.render(empty) - .85
        difference = renderer.render(frame) - create_renderer(quiet).render(frame)
        np.testing.assert_allclose(difference, noise, atol=1e-16, rtol=0)
        self.assertLess(abs(noise.mean()), .0002)
        self.assertLess(abs(noise.std() - .02), .0002)
        first = renderer.render(frame)
        second = renderer.render(replace(frame, frame_number=2, time_us=100))
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(first, renderer.render(frame))

    def test_uniform_background_both_tones(self):
        for tone, value in (("light", .85), ("dark", .15)):
            renderer = create_renderer(config(**{"background.tone": tone}))
            for index in (1, 2):
                self.assertTrue(np.all(renderer.render(Frame(index, (index - 1) * 100, ())) == value))

    def test_four_ideal_effect_combinations_and_csv_sigma(self):
        with tempfile.TemporaryDirectory() as directory:
            for motion in (False, True):
                for defocus in (False, True):
                    c = config(**{"motion.motion_blur_enabled": motion, "appearance.defocus_enabled": defocus,
                                  "appearance.blur_sigma_min_px": 1, "appearance.blur_sigma_max_px": 1,
                                  "droplets.count_per_frame": 2, "export.output_directory": directory})
                    result = export_series(c)
                    entries = list(csv.DictReader((result / "droplets.csv").read_text().splitlines()))
                    self.assertTrue(all(float(r["blur_sigma_px"]) == (1 if defocus else 0) for r in entries))
                    saved = load(result / "config.json").configuration
                    self.assertEqual(saved.appearance.blur_sigma_min_px, 1)
                    self.assertEqual(saved.run.status, "completed")

    def test_realistic_replay_all_codecs_noise_independent_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            for fmt, bits in (("png", 8), ("png", 16), ("tiff", 16)):
                c = config(**{"droplets.render_mode": "realistic", "droplets.count_per_frame": 2,
                              "motion.motion_blur_enabled": True, "appearance.defocus_enabled": True,
                              "background.noise_enabled": True, "camera.format": fmt, "camera.bit_depth": bits,
                              "export.output_directory": directory})
                first = export_series(c)
                restored = load(first / "config.json").configuration
                create_renderer(restored).render(next(generate_frames(restored)))
                replay = export_series(restored)
                quiet = export_series(update_configuration(c, {"background.noise_enabled": False}).configuration)
                self.assertEqual((first / "droplets.csv").read_bytes(), (quiet / "droplets.csv").read_bytes())
                self.assertEqual((first / "droplets.csv").read_bytes(), (replay / "droplets.csv").read_bytes())
                for image in (first / "frames").iterdir():
                    self.assertEqual(image.read_bytes(), (replay / "frames" / image.name).read_bytes())
                    self.assertNotEqual(image.read_bytes(), (quiet / "frames" / image.name).read_bytes())


if __name__ == "__main__":
    unittest.main()
