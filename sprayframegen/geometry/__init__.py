"""Непрерывная геометрия кругов и повёрнутых эллипсов — issue #4."""

from .projections import Ellipse, crosses_border, is_active, overlap_flags, overlaps

__all__ = ["Ellipse", "crosses_border", "is_active", "overlap_flags", "overlaps"]
