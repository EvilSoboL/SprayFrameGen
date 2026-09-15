"""Единая модель и строгая проверка ручного ввода и JSON (schema 1.x)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
import json
import math
from pathlib import Path, PureWindowsPath
import re
import secrets
from typing import get_type_hints

from . import __version__


class ConfigurationError(ValueError):
    """Ошибка с полным путём поля, пригодная для GUI и CLI."""


@dataclass(frozen=True)
class Camera:
    width_px: int = 640
    height_px: int = 480
    scale_um_per_px: float = 10.0
    frame_count: int = 20
    frame_interval_us: float = 100.0
    exposure_us: float = 20.0
    format: str = "png"
    bit_depth: int = 8


@dataclass(frozen=True)
class Droplets:
    count_per_frame: int = 100
    size_distribution: str = "lognormal"
    mean_um: float = 120.0
    std_um: float = 20.0
    median_um: float = 120.0
    geometric_std: float = 1.35
    max_diameter_um: float = 500.0
    concentration: str = "medium"
    render_mode: str = "realistic"


@dataclass(frozen=True)
class Motion:
    speed_mean_m_s: float = 8.0
    speed_std_m_s: float = 0.8
    speed_min_m_s: float = 4.0
    speed_max_m_s: float = 12.0
    direction_std_deg: float = 10.0
    motion_blur_enabled: bool = True


@dataclass(frozen=True)
class Appearance:
    profile: str = "standard-v1"
    defocus_enabled: bool = True
    blur_sigma_min_px: float = 0.0
    blur_sigma_max_px: float = 2.0


@dataclass(frozen=True)
class Background:
    tone: str = "light"
    noise_enabled: bool = True
    noise_sigma: float = 0.02

    @property
    def intensity(self) -> float:
        return {"dark": 0.15, "light": 0.85}[self.tone]


@dataclass(frozen=True)
class Export:
    output_directory: str | None = None


@dataclass(frozen=True)
class Runtime:
    name: str
    version: str


@dataclass(frozen=True)
class Environment:
    os: str
    architecture: str
    processor: str
    runtime: Runtime
    dependencies: dict[str, str]
    build_id: str
    numerical_profile: str


@dataclass(frozen=True)
class Run:
    status: str
    started_at: str
    completed_frame_count: int
    output_directory: str
    environment: Environment


@dataclass(frozen=True)
class Configuration:
    schema_version: str = "1.0"
    program_version: str = __version__
    seed: int = field(default_factory=lambda: secrets.randbits(32))
    camera: Camera = field(default_factory=Camera)
    droplets: Droplets = field(default_factory=Droplets)
    motion: Motion = field(default_factory=Motion)
    appearance: Appearance = field(default_factory=Appearance)
    background: Background = field(default_factory=Background)
    export: Export = field(default_factory=Export)
    run: Run | None = None


@dataclass(frozen=True)
class LoadResult:
    configuration: Configuration
    warnings: tuple[str, ...]


def _require(condition: bool, path: str, reason: str) -> None:
    if not condition:
        raise ConfigurationError(f"{path}: {reason}")


def _decode(cls, data, path: str, warnings: list[str]):
    _require(type(data) is dict, path or "config", "ожидается объект")
    annotations = get_type_hints(cls)
    for name in data.keys() - annotations.keys():
        warnings.append(f"{path + '.' if path else ''}{name}: неизвестное поле проигнорировано")
    values = {}
    for item in fields(cls):
        name = item.name
        key = f"{path}.{name}" if path else name
        _require(name in data, key, "отсутствует обязательное поле")
        value, kind = data[name], annotations[name]
        if kind == Run | None:
            values[name] = None if value is None else _decode(Run, value, key, warnings)
        elif kind == str | None:
            _require(value is None or (type(value) is str and bool(value.strip())), key,
                     "ожидается непустая строка или null")
            values[name] = value
        elif is_dataclass(kind):
            values[name] = _decode(kind, value, key, warnings)
        elif kind == dict[str, str]:
            _require(type(value) is dict and bool(value), key, "ожидается непустой объект версий")
            for dep, version in value.items():
                _require(type(dep) is str and bool(dep.strip()) and type(version) is str
                         and bool(version.strip()), key, "имена и версии должны быть непустыми строками")
            values[name] = dict(value)
        elif kind is float:
            _require(type(value) in (int, float), key, "ожидается конечное число, не строка и не bool")
            try:
                value = float(value)
            except OverflowError:
                raise ConfigurationError(f"{key}: число не представимо в float64") from None
            _require(math.isfinite(value), key, "число должно быть конечным")
            values[name] = value
        else:
            _require(type(value) is kind, key, f"ожидается тип {kind.__name__}")
            if kind is str:
                _require(bool(value.strip()), key, "строка не должна быть пустой")
            values[name] = value
    return cls(**values)


def _absolute(value: str | None, path: str) -> None:
    if value is not None:
        _require("\x00" not in value and (PureWindowsPath(value).is_absolute()
                 or Path(value).is_absolute()), path, "ожидается абсолютный путь")


def _distribution(center, spread, minimum, maximum, path, zero=0.0):
    if spread == zero:
        _require(minimum <= center <= maximum, path,
                 "при нулевом разбросе фиксированное значение должно лежать в допустимом диапазоне")
    if minimum == maximum:
        _require(spread == zero and center == minimum, path,
                 "при равных границах нужен нулевой разброс и значение, равное границам")


def _validate(c: Configuration, warnings: list[str]) -> None:
    version = re.fullmatch(r"1\.(0|[1-9][0-9]*)", c.schema_version)
    _require(version is not None, "schema_version", "поддерживается совместимая схема 1.x")
    if c.schema_version != "1.0":
        warnings.append("schema_version: более новая минорная схема; проверены известные поля")
    _require(0 <= c.seed <= 4294967295, "seed", "допустимо целое от 0 до 4294967295")
    camera, d, m, a, b = c.camera, c.droplets, c.motion, c.appearance, c.background
    _require((camera.width_px, camera.height_px) in
             {(640, 480), (1024, 768), (1280, 1024), (1920, 1080)},
             "camera.width_px/camera.height_px", "неподдерживаемое разрешение")
    _require(2 <= camera.frame_count <= 1000, "camera.frame_count", "допустимо от 2 до 1000")
    _require((camera.format, camera.bit_depth) in {("png", 8), ("png", 16), ("tiff", 16)},
             "camera.format/camera.bit_depth", "допустимы PNG 8/16 бит и TIFF 16 бит")
    _require(1 <= d.count_per_frame <= 10000, "droplets.count_per_frame", "допустимо от 1 до 10000")
    for path, value in [("camera.scale_um_per_px", camera.scale_um_per_px),
                        ("camera.frame_interval_us", camera.frame_interval_us),
                        ("camera.exposure_us", camera.exposure_us),
                        ("droplets.median_um", d.median_um),
                        ("motion.speed_min_m_s", m.speed_min_m_s)]:
        _require(value > 0, path, "значение должно быть больше нуля")
    for path, value in [("droplets.std_um", d.std_um), ("motion.speed_std_m_s", m.speed_std_m_s),
                        ("motion.direction_std_deg", m.direction_std_deg),
                        ("appearance.blur_sigma_min_px", a.blur_sigma_min_px),
                        ("background.noise_sigma", b.noise_sigma)]:
        _require(value >= 0, path, "значение должно быть неотрицательным")
    _require(camera.exposure_us <= camera.frame_interval_us, "camera.exposure_us",
             "выдержка не должна превышать camera.frame_interval_us")
    _require(d.geometric_std >= 1, "droplets.geometric_std", "значение должно быть не меньше 1")
    _require(d.max_diameter_um >= 3 * camera.scale_um_per_px, "droplets.max_diameter_um",
             "максимум должен быть не меньше 3 × camera.scale_um_per_px")
    _require(m.speed_max_m_s >= m.speed_min_m_s, "motion.speed_max_m_s", "максимум меньше минимума")
    _require(a.blur_sigma_max_px >= a.blur_sigma_min_px, "appearance.blur_sigma_max_px",
             "максимум меньше минимума")
    for path, value, allowed in [
        ("droplets.size_distribution", d.size_distribution, {"normal", "lognormal"}),
        ("droplets.concentration", d.concentration, {"low", "medium", "high"}),
        ("droplets.render_mode", d.render_mode, {"ideal", "realistic"}),
        ("appearance.profile", a.profile, {"standard-v1"}),
        ("background.tone", b.tone, {"dark", "light"})]:
        _require(value in allowed, path, "неизвестное значение; допустимо: " + ", ".join(sorted(allowed)))
    if d.size_distribution == "normal":
        _distribution(d.mean_um, d.std_um, 3 * camera.scale_um_per_px, d.max_diameter_um, "droplets.mean_um/std_um")
    else:
        _distribution(d.median_um, d.geometric_std, 3 * camera.scale_um_per_px,
                      d.max_diameter_um, "droplets.median_um/geometric_std", zero=1.0)
    _distribution(m.speed_mean_m_s, m.speed_std_m_s, m.speed_min_m_s,
                  m.speed_max_m_s, "motion.speed_mean_m_s/speed_std_m_s")
    _absolute(c.export.output_directory, "export.output_directory")
    # Conservative bounds, before any allocation, RNG sampling or calculation by consumers.
    scale = camera.scale_um_per_px
    diameter = d.max_diameter_um / scale
    last_time = (camera.frame_count - 1) * camera.frame_interval_us
    displacement = m.speed_max_m_s * camera.frame_interval_us / scale
    trail = m.speed_max_m_s * camera.exposure_us / scale
    shape_change = diameter / 2 * (0.35 * trail / camera.width_px) if d.render_mode == "realistic" else 0
    derived = {
        "camera.frame_interval_us": last_time + camera.exposure_us / 2,
        "camera.scale_um_per_px": max(camera.width_px, camera.height_px) * scale,
        "droplets.max_diameter_um": diameter * math.sqrt(1.35),
        "motion.speed_max_m_s": max(displacement, m.speed_max_m_s * last_time / scale),
    }
    if m.motion_blur_enabled:
        derived["motion.motion_blur_enabled"] = 4 * (trail + shape_change)
    if a.defocus_enabled:
        derived["appearance.blur_sigma_max_px"] = 4 * a.blur_sigma_max_px
    for path, value in derived.items():
        _require(math.isfinite(value), path, "переполнение производного размера, перемещения или времени")
    if c.run is not None:
        r = c.run
        _require(r.status in {"running", "completed", "cancelled", "error"}, "run.status", "неизвестный статус")
        _require(0 <= r.completed_frame_count <= camera.frame_count, "run.completed_frame_count",
                 "счётчик вне диапазона 0..camera.frame_count")
        _require(r.status != "completed" or r.completed_frame_count == camera.frame_count,
                 "run.completed_frame_count", "для completed нужны все запрошенные кадры")
        try:
            started = datetime.fromisoformat(r.started_at)
            _require(started.utcoffset() is not None, "run.started_at", "нужен часовой пояс ISO 8601")
        except ValueError as exc:
            raise ConfigurationError("run.started_at: ожидается ISO 8601 с часовым поясом") from exc
        _absolute(r.output_directory, "run.output_directory")
        _require(r.environment.numerical_profile == "standard-v1", "run.environment.numerical_profile",
                 "неподдерживаемый численный профиль")


def from_mapping(data: dict, *, compare_environment: bool = True) -> LoadResult:
    """Импорт и ручной ввод проходят один и тот же путь без подстановки defaults."""
    warnings: list[str] = []
    config = _decode(Configuration, data, "", warnings)
    _validate(config, warnings)
    if config.program_version != __version__:
        warnings.append("program_version: версия отличается; точный повтор не гарантируется")
    if compare_environment and config.run is not None:
        from .environment import current_environment
        if config.run.environment != current_environment():
            warnings.append("run.environment: сборка или окружение отличаются; точный повтор не гарантируется")
    return LoadResult(config, tuple(warnings))


def new_configuration(*, seed: int | None = None, render_mode: str = "realistic") -> Configuration:
    config = Configuration(seed=secrets.randbits(32) if seed is None else seed,
                           droplets=Droplets(render_mode=render_mode),
                           appearance=Appearance(defocus_enabled=render_mode != "ideal"))
    return from_mapping(asdict(config)).configuration


def update_configuration(config: Configuration, changes: dict[str, object]) -> LoadResult:
    """Изменить поля по путям вида camera.exposure_us, сохранив неактивные настройки."""
    data = asdict(config)
    for path, value in changes.items():
        parts = path.split(".")
        target = data
        for name in parts[:-1]:
            _require(isinstance(target, dict) and name in target, path, "неизвестное поле")
            target = target[name]
        _require(isinstance(target, dict) and parts[-1] in target, path, "неизвестное поле")
        target[parts[-1]] = value
    return from_mapping(data)


def loads(text: str) -> LoadResult:
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise ConfigurationError(f"JSON: невозможно прочитать документ: {exc}") from exc
    return from_mapping(data)


def load(path: str | Path) -> LoadResult:
    return loads(Path(path).read_text(encoding="utf-8-sig"))


def dumps(config: Configuration) -> str:
    data = asdict(config)
    data["program_version"] = __version__
    checked = from_mapping(data, compare_environment=False).configuration
    return json.dumps(asdict(checked), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def save(config: Configuration, path: str | Path) -> None:
    Path(path).write_text(dumps(config), encoding="utf-8", newline="\n")
