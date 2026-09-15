"""Скалярная повторная выборка; порядок вызовов является частью сборки."""

from dataclasses import asdict
import math
from collections.abc import Callable

from numpy.random import Generator

from ..configuration import Configuration, from_mapping
from . import Droplet


MAX_ATTEMPTS = 100_000
CONCENTRATION = {"low": 0.45, "medium": 0.30, "high": 0.18}


class SamplingError(ValueError):
    """Лимит выборки исчерпан; запуск должен завершиться со статусом error."""


def rejection_sample(draw: Callable[[], float], minimum: float, maximum: float,
                     parameter: str, settings: str, *, open_interval: bool = False) -> float:
    for _ in range(MAX_ATTEMPTS):
        value = float(draw())
        inside = minimum < value < maximum if open_interval else minimum <= value <= maximum
        if math.isfinite(value) and inside:
            return value
    brackets = ("(", ")") if open_interval else ("[", "]")
    raise SamplingError(
        f"{parameter}: не удалось выбрать значение в диапазоне "
        f"{brackets[0]}{minimum}, {maximum}{brackets[1]} за {MAX_ATTEMPTS} попыток; "
        f"{settings}. Проверьте параметры распределения и границы."
    )


class DropletSampler:
    """Последовательное создание новых ID из проверенной конфигурации и model RNG.

    На каплю: диаметр, группа, угол (у выброса модуль угла, затем знак),
    модуль скорости, X, Y, q_left, angle_deg, blur_sigma_px.
    Отклонённый кандидат расходует только выборку текущего свойства.
    Фиксированные и отключённые свойства RNG не расходуют.
    """

    def __init__(self, config: Configuration, rng: Generator):
        self.config = from_mapping(asdict(config), compare_environment=False).configuration
        self.rng = rng
        self._next_id = 1

    def _normal(self, mean: float, std: float, minimum: float, maximum: float,
                parameter: str) -> float:
        if std == 0:
            return float(mean)
        return rejection_sample(lambda: self.rng.normal(mean, std), minimum, maximum,
                                parameter, f"нормальное: среднее={mean}, разброс={std}")

    def _diameter(self) -> float:
        d = self.config.droplets
        minimum = 3 * self.config.camera.scale_um_per_px
        if d.size_distribution == "normal":
            return self._normal(d.mean_um, d.std_um, minimum, d.max_diameter_um,
                                "droplets: диаметр, мкм")
        if d.geometric_std == 1:
            return float(d.median_um)
        return rejection_sample(
            lambda: self.rng.lognormal(math.log(d.median_um), math.log(d.geometric_std)),
            minimum, d.max_diameter_um, "droplets: диаметр, мкм",
            f"логнормальное: медиана={d.median_um}, геометрический разброс={d.geometric_std}",
        )

    def sample_initial(self) -> Droplet:
        """Одна новая капля первого кадра; счётчик ID увеличивается после успеха."""
        c = self.config
        diameter_um = self._diameter()
        is_outlier = bool(self.rng.random() < 0.05)
        if is_outlier:
            absolute = rejection_sample(lambda: self.rng.uniform(30.0, 90.0), 30.0, 90.0,
                                        "theta_deg", "равномерное отклонение выброса",
                                        open_interval=True)
            theta = absolute * (-1.0 if self.rng.random() < 0.5 else 1.0)
        else:
            theta = self._normal(0.0, c.motion.direction_std_deg, -30.0, 30.0, "theta_deg")
        m = c.motion
        speed = self._normal(m.speed_mean_m_s, m.speed_std_m_s, m.speed_min_m_s,
                             m.speed_max_m_s, "motion: модуль скорости, м/с")
        width, height = c.camera.width_px, c.camera.height_px
        spread = CONCENTRATION[c.droplets.concentration]
        x = self._normal(width / 2, spread * width, 0.0, float(width), "center_x_px")
        y = self._normal(height / 2, spread * height, 0.0, float(height), "center_y_px")
        if c.droplets.render_mode == "realistic":
            q_left = float(self.rng.uniform(1.0, 1.35))
            angle = float(self.rng.uniform(0.0, 180.0))
        else:
            q_left, angle = 1.0, 0.0
        a = c.appearance
        blur = 0.0
        if a.defocus_enabled:
            blur = (float(a.blur_sigma_min_px) if a.blur_sigma_min_px == a.blur_sigma_max_px
                    else float(self.rng.uniform(a.blur_sigma_min_px, a.blur_sigma_max_px)))
        radians = math.radians(theta)
        droplet = Droplet(
            self._next_id, x, y, diameter_um, diameter_um / c.camera.scale_um_per_px,
            is_outlier, theta, speed, speed * math.cos(radians), speed * math.sin(radians),
            q_left, angle, blur,
        )
        self._next_id += 1
        return droplet

    def initial_population(self) -> tuple[Droplet, ...]:
        return tuple(self.sample_initial() for _ in range(self.config.droplets.count_per_frame))
