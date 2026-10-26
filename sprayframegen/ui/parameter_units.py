"""Чистые преобразования физических и отображаемых единиц интерфейса."""

import math


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name}: значение должно быть положительным и конечным")
    return value


def interval_us_to_hz(interval_us: float) -> float:
    result = 1_000_000.0 / _finite_positive(interval_us, "Интервал")
    return _finite_positive(result, "Частота")


def hz_to_interval_us(frequency_hz: float) -> float:
    result = 1_000_000.0 / _finite_positive(frequency_hz, "Частота")
    return _finite_positive(result, "Интервал")


def micrometers_to_pixels(diameter_um: float, scale_um_per_px: float) -> float:
    result = float(diameter_um) / _finite_positive(scale_um_per_px, "Масштаб")
    if not math.isfinite(result) or (diameter_um != 0 and result == 0):
        raise ValueError("Диаметр: переполнение или потеря точности при преобразовании")
    return result


def pixels_to_micrometers(diameter_px: float, scale_um_per_px: float) -> float:
    result = float(diameter_px) * _finite_positive(scale_um_per_px, "Масштаб")
    if not math.isfinite(result) or (diameter_px != 0 and result == 0):
        raise ValueError("Диаметр: переполнение или потеря точности при преобразовании")
    return result


def speed_to_pixels_per_frame(speed_m_s: float, interval_us: float, scale_um_per_px: float) -> float:
    speed_m_s = float(speed_m_s)
    if not math.isfinite(speed_m_s):
        raise ValueError("Средняя скорость: значение должно быть конечным")
    result = speed_m_s * _finite_positive(interval_us, "Интервал") / _finite_positive(scale_um_per_px, "Масштаб")
    if not math.isfinite(result) or (speed_m_s != 0 and result == 0):
        raise ValueError("Средняя скорость: переполнение или потеря точности при преобразовании")
    return result


def parse_number(text: str, label: str) -> float:
    value_text = text.strip()
    if not value_text:
        raise ValueError(f"{label}: введите число")
    if "," in value_text:
        raise ValueError(f"{label}: используйте точку как десятичный разделитель")
    try:
        value = float(value_text)
    except ValueError as exc:
        raise ValueError(f"{label}: введите корректное число") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}: значение должно быть конечным")
    return value


def format_editable(value: float) -> str:
    return format(value, ".17g")


def format_reference(value: float) -> str:
    return format(value, ".6g")
