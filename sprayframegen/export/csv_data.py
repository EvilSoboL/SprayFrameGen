"""19 столбцов эталона; строка зависит только от модели и масштаба."""

import math

from ..generation.sequence import Frame, FrameDroplet


COLUMNS = (
    "frame_number", "time_us", "droplet_id", "is_new", "center_x_px", "center_y_px",
    "center_x_mm", "center_y_mm", "diameter_eq_px", "diameter_eq_um", "semi_major_px",
    "semi_minor_px", "angle_deg", "vx_m_s", "vy_m_s", "speed_m_s", "blur_sigma_px",
    "border", "overlap",
)


def number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("CSV: недопустимо неконечное число")
    return "0" if value == 0 else format(value, ".17g")


def row(frame: Frame, state: FrameDroplet, scale_um_per_px: float) -> tuple[str, ...]:
    d = state.droplet
    boolean = lambda value: "true" if value else "false"
    return (
        str(frame.frame_number), number(frame.time_us), str(d.droplet_id), boolean(state.is_new),
        number(d.center_x_px), number(d.center_y_px),
        number(d.center_x_px * scale_um_per_px / 1000),
        number(d.center_y_px * scale_um_per_px / 1000),
        number(d.diameter_eq_px), number(d.diameter_eq_um), number(state.semi_major_px),
        number(state.semi_minor_px), number(d.angle_deg), number(d.vx_m_s), number(d.vy_m_s),
        number(d.speed_m_s), number(d.blur_sigma_px), boolean(state.border), boolean(state.overlap),
    )
