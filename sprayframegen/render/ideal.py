"""Идеальные непрозрачные круги, float64 и фиксированная сетка 4×4."""

from dataclasses import asdict, dataclass
import math
from typing import Protocol

import numpy as np

from ..configuration import Configuration, ConfigurationError, from_mapping
from ..generation.sequence import Frame, FrameDroplet


class FrameRenderer(Protocol):
    def render(self, frame: Frame) -> np.ndarray:
        """Одноканальная яркость float64 до квантования; без служебных наложений."""
        ...


@dataclass(frozen=True)
class Layer:
    x: int
    y: int
    premultiplied: np.ndarray
    alpha: np.ndarray


def circle_layer(state: FrameDroplet, width: int, height: int, intensity: float) -> Layer:
    d = state.droplet
    radius = d.diameter_eq_px / 2.0
    # Обрезаем непрерывные границы перед округлением/выделением памяти.
    x0 = math.floor(min(width, max(0.0, d.center_x_px - radius)))
    y0 = math.floor(min(height, max(0.0, d.center_y_px - radius)))
    x1 = math.ceil(min(width, max(0.0, d.center_x_px + radius)))
    y1 = math.ceil(min(height, max(0.0, d.center_y_px + radius)))
    alpha = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)
    columns = np.arange(x0, x1, dtype=np.float64)
    rows = np.arange(y0, y1, dtype=np.float64)
    for ky in range(4):
        dy = (rows + (ky + 0.5) / 4.0 - d.center_y_px) / radius
        for kx in range(4):
            dx = (columns + (kx + 0.5) / 4.0 - d.center_x_px) / radius
            alpha += (dy[:, None] ** 2 + dx[None, :] ** 2 <= 1.0)
    alpha /= 16.0
    return Layer(x0, y0, alpha * intensity, alpha)


def composite(image: np.ndarray, layer: Layer) -> None:
    """P + (1-A) I; интерфейс слоя сохраняет возможность эффектов D06."""
    h, w = layer.alpha.shape
    region = image[layer.y:layer.y + h, layer.x:layer.x + w]
    region[:] = layer.premultiplied + (1.0 - layer.alpha) * region


class IdealRenderer:
    def __init__(self, config: Configuration):
        self.config = from_mapping(asdict(config), compare_environment=False).configuration
        c = self.config
        if c.droplets.render_mode != "ideal":
            raise ConfigurationError("droplets.render_mode: рендер D05 поддерживает только ideal; реалистичный режим — issue #6")
        if c.motion.motion_blur_enabled:
            raise ConfigurationError("motion.motion_blur_enabled: отключите размытие движения для D05; эффект — issue #6")
        if c.appearance.defocus_enabled and c.appearance.blur_sigma_max_px > 0:
            raise ConfigurationError("appearance.defocus_enabled: отключите расфокусировку для D05; эффект — issue #6")
        if c.background.noise_enabled and c.background.noise_sigma > 0:
            raise ConfigurationError("background.noise_enabled: отключите шум для D05; эффект — issue #6")

    def render(self, frame: Frame) -> np.ndarray:
        c = self.config
        image = np.full((c.camera.height_px, c.camera.width_px), c.background.intensity,
                        dtype=np.float64)
        intensity = 0.1 if c.background.tone == "light" else 0.9
        for state in sorted(frame.droplets, key=lambda s: s.droplet.droplet_id):
            composite(image, circle_layer(state, c.camera.width_px, c.camera.height_px, intensity))
        return image


def create_renderer(config: Configuration) -> FrameRenderer:
    return IdealRenderer(config)


def encode_intensity(image: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth: допустимо 8 или 16")
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Изображение должно содержать конечную одноканальную яркость")
    encoded = np.floor(np.clip(image, 0.0, 1.0) * ((1 << bit_depth) - 1) + 0.5)
    return encoded.astype(np.uint8 if bit_depth == 8 else np.uint16)
