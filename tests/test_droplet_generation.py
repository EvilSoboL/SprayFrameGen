from dataclasses import FrozenInstanceError, replace
import math
from statistics import NormalDist
import unittest
from unittest.mock import Mock

import numpy as np

from sprayframegen.configuration import ConfigurationError, new_configuration, update_configuration
from sprayframegen.generation import generate_initial, prepare
from sprayframegen.model.sampling import DropletSampler, MAX_ATTEMPTS, SamplingError, rejection_sample


class DropletGenerationTests(unittest.TestCase):
    def config(self, **changes):
        return update_configuration(new_configuration(seed=123), changes).configuration

    def test_repeat_seed_preview_noise_and_ids(self):
        config = self.config()
        expected = generate_initial(config)
        generate_initial(config)  # independent preview
        self.assertEqual(expected, generate_initial(config))
        self.assertEqual(expected, generate_initial(replace(config, background=replace(
            config.background, noise_enabled=False, noise_sigma=0.5))))
        fresh, streams = prepare(config)
        sampler = DropletSampler(fresh, streams.model)
        first = sampler.sample_initial()
        streams.noise.normal(size=10000)
        second = sampler.sample_initial()
        self.assertEqual((first, second), expected[:2])
        self.assertEqual([d.droplet_id for d in expected], list(range(1, 101)))
        self.assertNotEqual(expected, generate_initial(replace(config, seed=124)))
        with self.assertRaises(FrozenInstanceError):
            first.speed_m_s = 0

    def test_ranges_components_and_area(self):
        config = self.config(**{"droplets.count_per_frame": 10000})
        for d in generate_initial(config):
            self.assertTrue(30 <= d.diameter_eq_um <= 500)
            self.assertGreaterEqual(d.diameter_eq_px, 3)
            self.assertEqual(d.diameter_eq_px, d.diameter_eq_um / 10)
            self.assertTrue(0 <= d.center_x_px <= 640 and 0 <= d.center_y_px <= 480)
            self.assertTrue(4 <= d.speed_m_s <= 12)
            self.assertTrue(1 <= d.q_left <= 1.35)
            self.assertTrue(0 <= d.angle_deg < 180)
            self.assertTrue(0 <= d.blur_sigma_px <= 2)
            self.assertGreater(d.vx_m_s, 0)
            self.assertAlmostEqual(math.hypot(d.vx_m_s, d.vy_m_s), d.speed_m_s)
            self.assertAlmostEqual(math.degrees(math.atan2(d.vy_m_s, d.vx_m_s)), d.theta_deg)
            self.assertTrue(30 < abs(d.theta_deg) < 90 if d.is_outlier else abs(d.theta_deg) <= 30)
            for x in (-10, 0, d.center_x_px, 320, 640, 650):
                a, b = d.semi_axes_at(x, 640)
                self.assertLessEqual(abs(2 * math.sqrt(a * b) / d.diameter_eq_px - 1), 0.001)
                expected_q = 1 + (d.q_left - 1) * (1 - min(1, max(0, x / 640)))
                self.assertAlmostEqual(a / b, expected_q)

    def test_seed_123_sampling_order_regression(self):
        # Эталон PCG64 / NumPy 2.3.5. Изменение порядка требует новой сборки.
        first = generate_initial(self.config())[0]
        expected = {
            "diameter_eq_um": 100.79249552112384,
            "theta_deg": -1.1145205867535353,
            "speed_m_s": 8.133208555122224,
            "center_x_px": 495.5707088359404,
            "center_y_px": 301.8500403579212,
            "q_left": 1.061771014220805,
            "angle_deg": 60.02209537576024,
            "blur_sigma_px": 1.9778673629765353,
        }
        self.assertFalse(first.is_outlier)
        for field, value in expected.items():
            self.assertAlmostEqual(getattr(first, field), value, places=12)

    def test_fixed_distributions_and_ideal_defocus(self):
        for distribution in ("normal", "lognormal"):
            config = self.config(**{
                "droplets.size_distribution": distribution, "droplets.mean_um": 30,
                "droplets.std_um": 0, "droplets.median_um": 30, "droplets.geometric_std": 1,
                "droplets.max_diameter_um": 30, "motion.speed_mean_m_s": 2,
                "motion.speed_std_m_s": 0, "motion.speed_min_m_s": 2, "motion.speed_max_m_s": 2,
                "motion.direction_std_deg": 0, "droplets.render_mode": "ideal",
                "appearance.blur_sigma_min_px": 1.25, "appearance.blur_sigma_max_px": 1.25,
            })
            population = generate_initial(config)
            self.assertTrue(any(d.is_outlier for d in population))
            for d in population:
                self.assertEqual((d.diameter_eq_px, d.speed_m_s, d.q_left, d.angle_deg,
                                  d.blur_sigma_px), (3, 2, 1, 0, 1.25))
                self.assertEqual(d.semi_axes_at(d.center_x_px, 640), (1.5, 1.5))
                if not d.is_outlier:
                    self.assertEqual((d.theta_deg, d.vx_m_s, d.vy_m_s), (0, 2, 0))
            disabled = replace(config, appearance=replace(config.appearance, defocus_enabled=False))
            self.assertTrue(all(d.blur_sigma_px == 0 for d in generate_initial(disabled)))

    def test_invalid_configuration_before_rng(self):
        base = self.config()
        for bad in (
            replace(base, motion=replace(base.motion, speed_min_m_s=0)),
            replace(base, motion=replace(base.motion, speed_std_m_s=0, speed_mean_m_s=20)),
            replace(base, droplets=replace(base.droplets, geometric_std=1, median_um=20)),
            replace(base, droplets=replace(base.droplets, max_diameter_um=30)),
        ):
            rng = Mock()
            with self.assertRaises(ConfigurationError):
                DropletSampler(bad, rng)
            self.assertEqual(rng.mock_calls, [])

    def test_rejection_limit_last_success_and_no_clipping(self):
        source = Mock(side_effect=[-1.0] * (MAX_ATTEMPTS - 1) + [0.5])
        self.assertEqual(rejection_sample(source, 0, 1, "диаметр", "среднее=2"), 0.5)
        self.assertEqual(source.call_count, MAX_ATTEMPTS)
        source = Mock(return_value=2.0)
        with self.assertRaisesRegex(SamplingError, r"диаметр.*\[0, 1\].*100000.*среднее=2"):
            rejection_sample(source, 0, 1, "диаметр", "среднее=2")
        self.assertEqual(source.call_count, MAX_ATTEMPTS)
        source = Mock(side_effect=[float("nan"), float("inf"), -1, 2, 0.4])
        self.assertEqual(rejection_sample(source, 0, 1, "test", "test"), 0.4)

    def test_sampler_rejection_for_each_distribution(self):
        for changes, method, values in (
            ({"droplets.size_distribution": "normal"}, "normal", [20, 501, 120]),
            ({}, "lognormal", [20, 501, 120]),
        ):
            config = self.config(**changes)
            rng = Mock()
            getattr(rng, method).side_effect = values
            sampler = DropletSampler(config, rng)
            self.assertEqual(sampler._diameter(), 120)
            self.assertEqual(getattr(rng, method).call_count, 3)
        for parameter, minimum, maximum in (("speed", 4, 12), ("theta_deg", -30, 30)):
            rng = Mock()
            rng.normal.side_effect = [minimum - 1, maximum + 1, maximum]
            sampler = DropletSampler(self.config(), rng)
            self.assertEqual(sampler._normal(0, 1, minimum, maximum, parameter), maximum)
            self.assertEqual(rng.normal.call_count, 3)

    def test_unreachable_distribution_reports_error(self):
        for changes, method in (({"droplets.size_distribution": "normal"}, "normal"),
                                ({}, "lognormal")):
            rng = Mock()
            getattr(rng, method).return_value = -1
            sampler = DropletSampler(self.config(**changes), rng)
            with self.assertRaisesRegex(SamplingError, "диаметр.*100000"):
                sampler.sample_initial()
            self.assertEqual(getattr(rng, method).call_count, MAX_ATTEMPTS)
            rng.random.assert_not_called()

    def test_open_outlier_interval(self):
        source = Mock(side_effect=[30, 90, 45])
        self.assertEqual(rejection_sample(source, 30, 90, "theta_deg", "uniform",
                                          open_interval=True), 45)
        self.assertEqual(source.call_count, 3)

    def assert_cdf(self, values, cdf, tolerance=0.02):
        values = np.sort(values)
        theoretical = np.array([cdf(float(x)) for x in values])
        n = len(values)
        distance = max(np.max(np.arange(1, n + 1) / n - theoretical),
                       np.max(theoretical - np.arange(n) / n))
        self.assertLess(distance, tolerance)

    def test_size_speed_and_angles_statistics(self):
        for distribution in ("normal", "lognormal"):
            config = self.config(**{"droplets.count_per_frame": 10000,
                                    "droplets.size_distribution": distribution})
            population = generate_initial(config)
            diameters = np.array([d.diameter_eq_um for d in population])
            if distribution == "normal":
                self.assertLessEqual(abs(diameters.mean() - 120), 0.05 * 20)
                self.assertLessEqual(abs(diameters.std(ddof=1) / 20 - 1), 0.05)
            else:
                self.assertLessEqual(abs(np.median(diameters) / 120 - 1), 0.03)
                self.assertLessEqual(abs(np.exp(np.log(diameters).std(ddof=1)) / 1.35 - 1), 0.05)
            speeds = np.array([d.speed_m_s for d in population])
            self.assertLessEqual(abs(speeds.mean() - 8), 0.05 * 0.8)
            self.assertLessEqual(abs(speeds.std(ddof=1) / 0.8 - 1), 0.05)
            main = [d.theta_deg for d in population if not d.is_outlier]
            outliers = [d.theta_deg for d in population if d.is_outlier]
            self.assertLess(abs(len(outliers) / 10000 - 0.05), 0.01)
            self.assertLess(abs(np.mean(np.array(outliers) > 0) - 0.5), 0.1)
            normal = NormalDist(0, 10)
            lo, hi = normal.cdf(-30), normal.cdf(30)
            self.assert_cdf(main, lambda x: (normal.cdf(x) - lo) / (hi - lo))
            self.assert_cdf(np.abs(outliers), lambda x: (x - 30) / 60, tolerance=0.08)
            self.assert_cdf([d.q_left for d in population], lambda x: (x - 1) / 0.35)
            self.assert_cdf([d.angle_deg for d in population], lambda x: x / 180)
            self.assert_cdf([d.blur_sigma_px for d in population], lambda x: x / 2)
            self.assertLess(abs(np.corrcoef(speeds, [d.theta_deg for d in population])[0, 1]), 0.04)
            self.assertLess(abs(np.corrcoef(speeds, [d.is_outlier for d in population])[0, 1]), 0.04)

    def test_concentration_truncated_normal_and_coordinate_independence(self):
        for level, spread in (("low", 0.45), ("medium", 0.30), ("high", 0.18)):
            population = generate_initial(self.config(**{
                "droplets.count_per_frame": 10000, "droplets.concentration": level,
            }))
            for field, length in (("center_x_px", 640), ("center_y_px", 480)):
                normal = NormalDist(length / 2, spread * length)
                lo, hi = normal.cdf(0), normal.cdf(length)
                self.assert_cdf([getattr(d, field) for d in population],
                                lambda x: (normal.cdf(x) - lo) / (hi - lo))
            self.assertLess(abs(np.corrcoef([d.center_x_px for d in population],
                                           [d.center_y_px for d in population])[0, 1]), 0.04)


if __name__ == "__main__":
    unittest.main()
