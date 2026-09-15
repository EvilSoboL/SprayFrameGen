"""Непрерывные эллипсы: границы и положительная общая площадь внутри кадра."""

from dataclasses import dataclass
import math
from statistics import median

from ..model import Droplet


@dataclass(frozen=True)
class Ellipse:
    x: float
    y: float
    a: float
    b: float
    half_width: float
    half_height: float
    slope: float
    slice_radius: float

    @classmethod
    def from_droplet(cls, droplet: Droplet, width: int) -> "Ellipse":
        a, b = droplet.semi_axes_at(droplet.center_x_px, width)
        theta = math.radians(droplet.angle_deg)
        c, s = math.cos(theta), math.sin(theta)
        hx, hy = math.hypot(a * c, b * s), math.hypot(a * s, b * c)
        # Нормированные отношения избегают переполнения a*a и b*b.
        slope = ((a / hx) ** 2 - (b / hx) ** 2) * c * s
        return cls(droplet.center_x_px, droplet.center_y_px, a, b,
                   hx, hy, slope, a * (b / hx))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x - self.half_width, self.y - self.half_height,
                self.x + self.half_width, self.y + self.half_height)

    def vertical_slice(self, x: float) -> tuple[float, float]:
        offset = x - self.x
        u = offset / self.half_width
        radius = self.slice_radius * math.sqrt(max(0.0, (1.0 - u) * (1.0 + u)))
        center = self.y + self.slope * offset
        return center - radius, center + radius


def _positive_area(ellipses: tuple[Ellipse, ...], width: int, height: int) -> bool:
    left = max(0.0, *(e.bounds[0] for e in ellipses))
    right = min(float(width), *(e.bounds[2] for e in ellipses))
    bottom = max(0.0, *(e.bounds[1] for e in ellipses))
    top = min(float(height), *(e.bounds[3] for e in ellipses))
    if left >= right or bottom >= top:
        return False
    # Допуск привязан к кадру: огромная капля не должна скрывать пересечение
    # величиной во весь кадр из-за ULP её собственного диаметра.
    tolerance = 32 * math.ulp(float(max(width, height)))

    def gap(x: float) -> float:
        lower, upper = 0.0, float(height)
        for e in ellipses:
            lo, hi = e.vertical_slice(x)
            lower, upper = max(lower, lo), min(upper, hi)
        return lower - upper

    # Нижние границы выпуклы, верхние вогнуты: gap выпукла.
    # Строго отрицательный минимум означает общую внутреннюю область.
    # Проверяем аналитические сечения, без растровой маски/аппроксимации контура.
    if min(gap(left), gap(right), gap((left + right) / 2)) < -tolerance:
        return True
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1, f2 = gap(x1), gap(x2)
    for _ in range(80):
        if min(f1, f2) < -tolerance:
            return True
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = gap(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = gap(x2)
    return min(f1, f2) < -tolerance


def is_active(ellipse: Ellipse, width: int, height: int) -> bool:
    if 0 <= ellipse.x <= width and 0 <= ellipse.y <= height:
        return True
    return _positive_area((ellipse,), width, height)


def crosses_border(ellipse: Ellipse, width: int, height: int) -> bool:
    """Для активной капли: есть ли площадь снаружи; касание изнутри не считается."""
    left, top, right, bottom = ellipse.bounds
    return left < 0 or top < 0 or right > width or bottom > height


def overlaps(first: Ellipse, second: Ellipse, width: int, height: int) -> bool:
    return _positive_area((first, second), width, height)


def overlap_flags(ellipses: tuple[Ellipse, ...], width: int, height: int) -> tuple[bool, ...]:
    """Равномерная сетка по обрезанным AABB; точный узкий этап для кандидатов.

    Размер ячейки адаптируется к медианному диаметру ограничивающих коробок.
    Уже отмеченную пару можно не проверять: нужны флаги, а не список всех пар.
    """
    if not ellipses:
        return ()
    cell = max(16.0, median(2 * max(e.half_width, e.half_height) for e in ellipses))
    grid: dict[tuple[int, int], list[int]] = {}
    flags = [False] * len(ellipses)
    for i, ellipse in enumerate(ellipses):
        left, top, right, bottom = ellipse.bounds
        left, top, right, bottom = max(0.0, left), max(0.0, top), min(width, right), min(height, bottom)
        if left >= right or top >= bottom:
            continue
        cells = [(x, y) for x in range(math.floor(left / cell), math.floor(right / cell) + 1)
                 for y in range(math.floor(top / cell), math.floor(bottom / cell) + 1)]
        candidates = set()
        for key in cells:
            candidates.update(grid.get(key, ()))
        for j in sorted(candidates):
            if flags[i] and flags[j]:
                continue
            if overlaps(ellipse, ellipses[j], width, height):
                flags[i] = flags[j] = True
        for key in cells:
            grid.setdefault(key, []).append(i)
    return tuple(flags)
