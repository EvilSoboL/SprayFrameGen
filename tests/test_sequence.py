from dataclasses import replace, fields
import math
from statistics import NormalDist
import unittest
from unittest.mock import patch

import numpy as np

from sprayframegen.configuration import new_configuration, update_configuration, ConfigurationError
from sprayframegen.generation import generate_frames, generate_initial, prepare
from sprayframegen.geometry import Ellipse, is_active
from sprayframegen.model.sampling import DropletSampler


class SequenceTests(unittest.TestCase):
    def config(self, **changes):
        return update_configuration(new_configuration(seed=321), changes).configuration

    def initial_at(self, positions):
        original = DropletSampler.initial_population

        def population(sampler):
            droplets = original(sampler)
            return tuple(replace(d, center_x_px=x, center_y_px=y,
                                 vx_m_s=vx, vy_m_s=vy, speed_m_s=math.hypot(vx, vy),
                                 theta_deg=math.degrees(math.atan2(vy, vx)))
                         for d, (x, y, vx, vy) in zip(droplets, positions))
        return patch.object(DropletSampler, "initial_population", population)

    def test_reference_motion_and_all_invariant_fields(self):
        config = self.config(**{"droplets.count_per_frame": 1, "camera.frame_count": 3})
        with self.initial_at([(100, 100, 2, -1)]):
            frames = list(generate_frames(config))
        original = frames[0].droplets[0].droplet
        for index, frame in enumerate(frames):
            state = frame.droplets[0]
            self.assertEqual(frame.frame_number, index + 1)
            self.assertEqual(frame.time_us, index * 100)
            self.assertEqual((state.droplet.center_x_px, state.droplet.center_y_px),
                             (100 + index * 20, 100 - index * 10))
            self.assertEqual(state.is_new, index == 0)
            for field in fields(original):
                if field.name not in ("center_x_px", "center_y_px"):
                    self.assertEqual(getattr(state.droplet, field.name), getattr(original, field.name))
            self.assertAlmostEqual(state.semi_major_px * state.semi_minor_px,
                                   (original.diameter_eq_px / 2) ** 2)
        self.assertLess(frames[-1].droplets[0].semi_major_px, frames[0].droplets[0].semi_major_px)

    def test_top_bottom_right_exit_and_survivor_unchanged(self):
        config = self.config(**{"droplets.count_per_frame": 4, "camera.frame_count": 2})
        with self.initial_at([(639, 100, 10, 0), (100, 1, 10, -100),
                              (100, 479, 10, 100), (100, 100, 2, 0)]):
            frames = list(generate_frames(config))
        self.assertEqual([s.droplet.droplet_id for s in frames[1].droplets], [4, 5, 6, 7])
        survivor = frames[1].droplets[0]
        self.assertEqual(survivor.droplet, replace(frames[0].droplets[3].droplet, center_x_px=120))
        self.assertFalse(survivor.is_new)
        for state in frames[1].droplets[1:]:
            self.assertTrue(state.is_new and state.border)
            self.assertEqual(state.droplet.center_x_px, 0)
            self.assertTrue(0 <= state.droplet.center_y_px <= 480)

    def test_partial_survives_then_external_tangent_replaced(self):
        config = self.config(**{
            "droplets.count_per_frame": 1, "camera.frame_count": 3,
            "droplets.render_mode": "ideal", "droplets.size_distribution": "normal",
            "droplets.mean_um": 40, "droplets.std_um": 0,
        })
        with self.initial_at([(640, 100, 0.1, 0)]):
            frames = list(generate_frames(config))
        self.assertEqual([f.droplets[0].droplet.droplet_id for f in frames], [1, 1, 2])
        self.assertEqual([f.droplets[0].is_new for f in frames], [True, False, True])
        self.assertEqual([f.droplets[0].droplet.center_x_px for f in frames], [640, 641, 0])

    def test_coincident_droplets_remain_separate(self):
        config = self.config(**{
            "droplets.count_per_frame": 2, "camera.frame_count": 2,
            "droplets.render_mode": "ideal", "droplets.size_distribution": "normal",
            "droplets.mean_um": 40, "droplets.std_um": 0,
        })
        with self.initial_at([(100, 100, 2, 0), (100, 100, 2, 0)]):
            frames = list(generate_frames(config))
        for frame in frames:
            self.assertEqual([s.droplet.droplet_id for s in frame.droplets], [1, 2])
            self.assertTrue(all(s.overlap for s in frame.droplets))

    def test_mass_exit_maximum_population_ids_and_both_groups(self):
        config = self.config(**{
            "droplets.count_per_frame": 10000, "camera.frame_count": 2,
            "motion.speed_mean_m_s": 100000, "motion.speed_std_m_s": 0,
            "motion.speed_min_m_s": 100000, "motion.speed_max_m_s": 100000,
        })
        frames = list(generate_frames(config))
        for index, frame in enumerate(frames):
            self.assertEqual(len(frame.droplets), 10000)
            self.assertEqual([s.droplet.droplet_id for s in frame.droplets],
                             list(range(1 + index * 10000, 1 + (index + 1) * 10000)))
            self.assertTrue(all(s.is_new for s in frame.droplets))
            self.assertTrue(all(is_active(Ellipse.from_droplet(s.droplet, 640), 640, 480)
                                for s in frame.droplets))
        replacements = frames[1].droplets
        self.assertTrue(all(s.droplet.center_x_px == 0 and s.border for s in replacements))
        self.assertTrue(any(s.droplet.is_outlier and s.droplet.vy_m_s > 0 for s in replacements))
        self.assertTrue(any(s.droplet.is_outlier and s.droplet.vy_m_s < 0 for s in replacements))

    def test_replacement_y_concentrations(self):
        for level, spread in (("low", .45), ("medium", .30), ("high", .18)):
            config, streams = prepare(self.config(**{"droplets.concentration": level}))
            sampler = DropletSampler(config, streams.model)
            droplets = [sampler.sample_replacement() for _ in range(10000)]
            y = np.sort([d.center_y_px for d in droplets])
            self.assertTrue(all(d.center_x_px == 0 for d in droplets))
            self.assertTrue(0 <= y[0] <= y[-1] <= 480)
            normal = NormalDist(240, spread * 480)
            lo, hi = normal.cdf(0), normal.cdf(480)
            theoretical = np.array([(normal.cdf(float(v)) - lo) / (hi - lo) for v in y])
            self.assertLess(max(np.max(np.arange(1, 10001) / 10000 - theoretical),
                                np.max(theoretical - np.arange(10000) / 10000)), .02)

    def test_series_repeat_after_partial_preview_and_noise_switch(self):
        config = self.config()
        expected = list(generate_frames(config))
        preview = generate_frames(config)
        next(preview)
        next(preview)
        self.assertEqual(expected, list(generate_frames(config)))
        quiet = replace(config, background=replace(config.background, noise_enabled=False))
        self.assertEqual(expected, list(generate_frames(quiet)))
        self.assertEqual(tuple(s.droplet for s in expected[0].droplets), generate_initial(config))

    def test_1000_frames_unique_ids_and_coordinate_accuracy(self):
        config = self.config(**{"camera.frame_count": 1000, "droplets.count_per_frame": 1})
        seen = set()
        previous = None
        for frame in generate_frames(config):
            self.assertEqual(len(frame.droplets), 1)
            state = frame.droplets[0]
            d = state.droplet
            if state.is_new:
                self.assertNotIn(d.droplet_id, seen)
                seen.add(d.droplet_id)
            else:
                self.assertEqual(d.droplet_id, previous.droplet_id)
                self.assertLessEqual(abs(d.center_x_px - previous.center_x_px - d.vx_m_s * 10), .01)
                self.assertLessEqual(abs(d.center_y_px - previous.center_y_px - d.vy_m_s * 10), .01)
            previous = d
        self.assertEqual(frame.frame_number, 1000)
        self.assertEqual(frame.time_us, 99900)
        self.assertGreater(len(seen), 1)

    def test_zero_speed_rejected(self):
        config = new_configuration(seed=0)
        invalid = replace(config, motion=replace(config.motion, speed_mean_m_s=0,
                                                speed_std_m_s=0, speed_min_m_s=0))
        with self.assertRaises(ConfigurationError):
            next(generate_frames(invalid))


if __name__ == "__main__":
    unittest.main()
