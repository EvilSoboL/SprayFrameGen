"""standard-v1: временные сечения, индивидуальная оптика и сенсорный шум."""

from dataclasses import asdict
import math

import numpy as np
from numpy.random import Generator, PCG64, SeedSequence

from ..configuration import Configuration, from_mapping
from ..generation.sequence import Frame, FrameDroplet
from .ideal import Layer, circle_layer, composite


def temporal_sampling(state: FrameDroplet, config: Configuration) -> tuple[int, float, float]:
    if not config.motion.motion_blur_enabled:
        return 1, 0.0, 0.0
    d, camera = state.droplet, config.camera
    dx = d.vx_m_s * camera.exposure_us / camera.scale_um_per_px
    dy = d.vy_m_s * camera.exposure_us / camera.scale_um_per_px
    change = (d.diameter_eq_px / 2 * ((d.q_left - 1) * abs(dx) / camera.width_px)
              if config.droplets.render_mode == "realistic" else 0.0)
    samples = 4 * (math.hypot(dx, dy) + change)
    if not math.isfinite(samples):
        raise ValueError("Размытие движения: переполнение числа временных отсчётов")
    return max(1, math.ceil(samples)), dx, dy


def gaussian_kernel(sigma: float) -> np.ndarray:
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Расфокусировка: sigma должна быть конечной и неотрицательной")
    if sigma == 0:
        return np.ones(1, dtype=np.float64)
    radius = math.ceil(4 * sigma)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    with np.errstate(over="ignore"):
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    return weights / weights.sum()


def gaussian_blur(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if len(kernel) == 1 or values.size == 0:
        return values.copy()
    radius = len(kernel) // 2
    # valid после явных нулей сохраняет размер даже при ядре шире слоя.
    horizontal = np.apply_along_axis(
        lambda row: np.convolve(np.pad(row, radius), kernel, mode="valid"), 1, values)
    return np.apply_along_axis(
        lambda col: np.convolve(np.pad(col, radius), kernel, mode="valid"), 0, horizontal)


def optical_layer(state: FrameDroplet, config: Configuration) -> Layer:
    d, camera = state.droplet, config.camera
    realistic = config.droplets.render_mode == "realistic"
    outer = 0.1 if config.background.tone == "light" else 0.9
    inner = 0.95 if config.background.tone == "light" else 0.05
    sigma = d.blur_sigma_px if config.appearance.defocus_enabled else 0.0
    n, dx, dy = temporal_sampling(state, config)
    if not realistic and n == 1 and sigma == 0:
        return circle_layer(state, camera.width_px, camera.height_px, outer)
    kernel = gaussian_kernel(sigma)
    margin = len(kernel) // 2
    radius = d.diameter_eq_px / 2 * (math.sqrt(d.q_left) if realistic else 1.0)
    # Консервативный охват всего пути и формы, плюс область свёртки.
    # Сохраняем исходную геометрию вне кадра в пределах радиуса ядра.
    def bounds(center, displacement, length):
        low = max(-margin, min(length + margin, center - abs(displacement) / 2 - radius - margin))
        high = max(-margin, min(length + margin, center + abs(displacement) / 2 + radius + margin))
        return math.floor(low), math.ceil(high)
    x0, x1 = bounds(d.center_x_px, dx, camera.width_px)
    y0, y1 = bounds(d.center_y_px, dy, camera.height_px)
    coverage = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)
    interior = np.zeros_like(coverage)
    columns, rows = np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64)
    theta = math.radians(d.angle_deg) if realistic else 0.0
    cosine, sine = math.cos(theta), math.sin(theta)
    for k in range(n):
        fraction = (k + 0.5) / n - 0.5
        x, y = d.center_x_px + fraction * dx, d.center_y_px + fraction * dy
        a, b = d.semi_axes_at(x, camera.width_px) if realistic else (radius, radius)
        for ky in range(4):
            ry = (rows + (ky + 0.5) / 4 - y)[:, None]
            for kx in range(4):
                rx = (columns + (kx + 0.5) / 4 - x)[None, :]
                u = (rx / a) * cosine + (ry / a) * sine
                v = -(rx / b) * sine + (ry / b) * cosine
                rho2 = u * u + v * v
                coverage += rho2 <= 1.0
                if realistic:
                    interior += rho2 <= 0.8 ** 2
    # Усредняем счётчики до умножения на яркости: полностью покрытый пиксель
    # остаётся точно непрозрачным даже при числе отсчётов, не равном степени 2.
    rim = (coverage - interior) / (16 * n)
    interior /= 16 * n
    alpha = rim + 0.35 * interior
    premultiplied = outer * rim + (0.35 * inner) * interior
    return Layer(x0, y0, gaussian_blur(premultiplied, kernel), gaussian_blur(alpha, kernel))


class OpticalRenderer:
    """Повтор кадра не зависит от порядка предпросмотра: шум имеет ключ (1, номер-1)."""

    def __init__(self, config: Configuration):
        self.config = from_mapping(asdict(config), compare_environment=False).configuration

    def render(self, frame: Frame) -> np.ndarray:
        c = self.config
        if type(frame.frame_number) is not int or frame.frame_number < 1:
            raise ValueError("frame_number: ожидается положительное целое")
        image = np.full((c.camera.height_px, c.camera.width_px), c.background.intensity, dtype=np.float64)
        for state in sorted(frame.droplets, key=lambda s: s.droplet.droplet_id):
            composite(image, optical_layer(state, c))
        if c.background.noise_enabled and c.background.noise_sigma > 0:
            noise = Generator(PCG64(SeedSequence(c.seed, spawn_key=(1, frame.frame_number - 1))))
            image += noise.normal(0.0, c.background.noise_sigma, size=image.shape)
        return image
