from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from sprayframegen.model import Droplet
from sprayframegen.geometry import Ellipse, is_active, crosses_border, overlaps, overlap_flags


def circle(x, y, radius=2, **changes):
    d = Droplet(1, x, y, radius * 20, radius * 2, False, 0, 2, 2, 0, 1, 0, 0)
    return replace(d, **changes)


def shape(x, y, radius=2, **changes):
    return Ellipse.from_droplet(circle(x, y, radius, **changes), 640)


class GeometryTests(unittest.TestCase):
    def test_four_sides_inside_partial_and_external_tangency(self):
        for transform in (lambda x: (x, 100), lambda x: (640 - x, 100),
                          lambda x: (100, x), lambda x: (100, 480 - x)):
            for offset, active, border in ((2, True, False), (1, True, True),
                                           (0, True, True), (-1, True, True), (-2, False, True)):
                with self.subTest(offset=offset, transform=transform):
                    e = shape(*transform(offset))
                    self.assertEqual(is_active(e, 640, 480), active)
                    if active:
                        self.assertEqual(crosses_border(e, 640, 480), border)

    def test_corner_bbox_is_not_sufficient(self):
        for xsign, ysign, ox, oy in ((1, 1, 0, 0), (-1, 1, 640, 0),
                                    (1, -1, 0, 480), (-1, -1, 640, 480)):
            for distance, expected in ((1, True), (1.5, False), (math.sqrt(2), False)):
                e = shape(ox - xsign * distance, oy - ysign * distance)
                self.assertEqual(is_active(e, 640, 480), expected)

    def test_circle_overlap_tangency_containment_and_coincidence(self):
        first = shape(100, 100)
        for second, expected in ((shape(103, 100), True), (shape(104, 100), False),
                                 (shape(105, 100), False), (shape(100, 100, 1), True),
                                 (first, True), (shape(103, 103), False)):
            self.assertEqual(overlaps(first, second, 640, 480), expected)
            self.assertEqual(overlaps(second, first, 640, 480), expected)
        self.assertEqual(overlap_flags((first, first, shape(200, 200)), 640, 480),
                         (True, True, False))

    def test_overlap_only_outside_frame(self):
        for first, second in ((shape(641.5, 10), shape(641.5, 13)),
                              (shape(-1.5, 10), shape(-1.5, 13)),
                              (shape(10, -1.5), shape(13, -1.5)),
                              (shape(10, 481.5), shape(13, 481.5))):
            self.assertTrue(is_active(first, 640, 480))
            self.assertTrue(is_active(second, 640, 480))
            self.assertFalse(overlaps(first, second, 640, 480))
        self.assertTrue(overlaps(shape(640.5, 10), shape(640.5, 13), 640, 480))

    def test_rotated_ellipse_tangent_and_small_penetration(self):
        # Одинаковые эллипсы: расстояние вдоль большой оси равно 2a при касании.
        # q_left корректируется так, чтобы текущее q в обеих позициях было 1.2.
        for angle in (0, 17, 45, 89, 120):
            theta = math.radians(angle)
            a = 10 * math.sqrt(1.2)
            def ellipse_at(x, y):
                q_left = 1 + 0.2 / (1 - x / 640)
                return shape(x, y, 10, angle_deg=angle, q_left=q_left)
            first = ellipse_at(100, 100)
            for distance, expected in ((2 * a, False), (2 * a - 1e-6, True),
                                       (2 * a + 1e-6, False)):
                second = ellipse_at(100 + distance * math.cos(theta),
                                    100 + distance * math.sin(theta))
                self.assertEqual(overlaps(first, second, 640, 480), expected)

    def test_rotated_corner_and_containment(self):
        # Большая ось направлена от угла: bbox пересекает кадр, сам эллипс — нет.
        outside = shape(-7, -7, 10, q_left=1.35, angle_deg=135)
        inside = shape(-7, -7, 10, q_left=1.35, angle_deg=45)
        self.assertFalse(is_active(outside, 640, 480))
        self.assertTrue(is_active(inside, 640, 480))
        self.assertTrue(overlaps(shape(100, 100, 20, q_left=1.3, angle_deg=35),
                                 shape(101, 101, 2, q_left=1.3, angle_deg=120), 640, 480))

    def test_rotated_ellipses_common_area_only_outside(self):
        first = shape(-1.8, 10, q_left=1.3, angle_deg=45)
        second = shape(-1.8, 13.5, q_left=1.3, angle_deg=45)
        self.assertTrue(is_active(first, 640, 480))
        self.assertTrue(is_active(second, 640, 480))
        self.assertFalse(overlaps(first, second, 640, 480))
        self.assertTrue(overlaps(replace(first, x=100), replace(second, x=100), 640, 480))

    def test_huge_ellipses_cover_frame(self):
        first = shape(100, 100, 1e150, q_left=1.3, angle_deg=45)
        second = shape(200, 200, 1e150, q_left=1.3, angle_deg=120)
        self.assertTrue(is_active(first, 640, 480))
        self.assertTrue(overlaps(first, second, 640, 480))
        self.assertEqual(overlap_flags((first, second), 640, 480), (True, True))

    def test_random_circles_against_analytic_distance(self):
        rng = np.random.default_rng(19)
        for _ in range(1000):
            r1, r2 = rng.uniform(1.5, 30, size=2)
            x1, y1, x2, y2 = rng.uniform(100, 200, size=4)
            expected = math.hypot(x1 - x2, y1 - y2) < r1 + r2
            self.assertEqual(overlaps(shape(x1, y1, r1), shape(x2, y2, r2), 640, 480), expected)

    def test_current_axes_affect_border_and_overlap(self):
        d = circle(0, 100, 10, q_left=1.3)
        for x, q in ((0, 1.3), (320, 1.15), (640, 1.0)):
            e = Ellipse.from_droplet(replace(d, center_x_px=x), 640)
            self.assertAlmostEqual(e.a / e.b, q)
            self.assertAlmostEqual(e.a * e.b, 100)
        e = Ellipse.from_droplet(replace(d, center_x_px=629.8), 640)
        self.assertFalse(crosses_border(e, 640, 480))
        self.assertFalse(overlaps(e, shape(610, 100, 9), 640, 480))

    def test_grid_matches_all_pairs_for_rotated_clipped_ellipses(self):
        rng = np.random.default_rng(43)
        ellipses = tuple(shape(float(rng.uniform(-20, 660)), float(rng.uniform(-20, 500)),
                               float(rng.uniform(2, 35)), angle_deg=float(rng.uniform(0, 180)),
                               q_left=float(rng.uniform(1, 1.35))) for _ in range(200))
        expected = [False] * len(ellipses)
        for i, first in enumerate(ellipses):
            for j in range(i):
                if overlaps(first, ellipses[j], 640, 480):
                    expected[i] = expected[j] = True
        self.assertEqual(overlap_flags(ellipses, 640, 480), tuple(expected))

    def test_spatial_search_avoids_all_pairs(self):
        ellipses = tuple(shape(x, y) for x in range(10, 640, 20) for y in range(10, 480, 20))
        with patch("sprayframegen.geometry.projections.overlaps", wraps=overlaps) as check:
            self.assertFalse(any(overlap_flags(ellipses, 640, 480)))
        self.assertLess(check.call_count, len(ellipses) * 4)


if __name__ == "__main__":
    unittest.main()
