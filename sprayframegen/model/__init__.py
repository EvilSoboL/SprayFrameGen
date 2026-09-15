"""Неизменяемое состояние капли; не зависит от GUI и экспорта."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Droplet:
    droplet_id: int
    center_x_px: float
    center_y_px: float
    diameter_eq_um: float
    diameter_eq_px: float
    is_outlier: bool
    theta_deg: float
    speed_m_s: float
    vx_m_s: float
    vy_m_s: float
    q_left: float
    angle_deg: float
    blur_sigma_px: float

    def semi_axes_at(self, center_x_px: float, width_px: int) -> tuple[float, float]:
        """Полуоси в заданной позиции, включая продолжение за границы кадра."""
        if not math.isfinite(center_x_px) or not math.isfinite(width_px) or width_px <= 0:
            raise ValueError("Форма: нужны конечная координата и положительная ширина")
        progress = min(1.0, max(0.0, center_x_px / width_px))
        root_q = math.sqrt(1.0 + (self.q_left - 1.0) * (1.0 - progress))
        radius = self.diameter_eq_px / 2.0
        return radius * root_q, radius / root_q
