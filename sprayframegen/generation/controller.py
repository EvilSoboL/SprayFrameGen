"""Контроллер генерации: фоновая работа, прогресс, отмена, безопасная запись."""

from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ..configuration import Configuration, ConfigurationError, Run, save
from ..environment import current_environment
from .sequence import generate_frames
from .core import prepare
from ..render import create_renderer, encode_intensity
from .random_streams import RandomStreams
from ..export.csv_data import COLUMNS, row


@dataclass(frozen=True)
class ProgressEvent:
    """Событие прогресса генерации."""
    frame_number: int
    total_frames: int
    completed_frames: int
    elapsed_seconds: float
    status: str


@dataclass(frozen=True)
class CompletionEvent:
    """Событие завершения генерации."""
    directory: Path
    completed_frames: int
    total_frames: int
    status: str
    message: str


class CancellationToken:
    """Токен отмены для безопасной остановки генерации."""

    def __init__(self) -> None:
        self._cancelled = False
        self._lock = threading.Lock()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled


class ExportError(OSError):
    """Ошибка экспорта с путём доступной части серии."""

    def __init__(self, message: str, directory: Optional[Path] = None):
        super().__init__(message)
        self.directory = directory


def _estimate_series_size(config: Configuration) -> int:
    """Оценка размера серии в байтах (изображения + CSV + метаданные)."""
    camera = config.camera
    pixels_per_frame = camera.width_px * camera.height_px
    bytes_per_pixel = 2 if camera.bit_depth == 16 else 1
    # PNG сжатие ~0.5-0.8, TIFF без сжатия. Берём коэффициент 0.7 для PNG.
    compression_factor = 0.7 if camera.format == "png" else 1.0
    image_size = int(pixels_per_frame * bytes_per_pixel * compression_factor * camera.frame_count)
    # CSV: ~200 байт на строку * count_per_frame * frame_count
    csv_size = 200 * config.droplets.count_per_frame * camera.frame_count
    # Метаданные ~5 КБ
    metadata_size = 5000
    return image_size + csv_size + metadata_size


def _check_disk_space(directory: Path, required_bytes: int) -> None:
    """Проверка свободного места. Выбрасывает ConfigurationError при нехватке."""
    try:
        free = shutil.disk_usage(directory).free
    except OSError as exc:
        raise ConfigurationError(f"export.output_directory: невозможно определить свободное место: {exc}") from exc
    # Требуем 20% запаса
    if free < required_bytes * 1.2:
        raise ConfigurationError(
            f"export.output_directory: недостаточно свободного места. "
            f"Требуется ~{required_bytes // 1024 // 1024} МБ, доступно {free // 1024 // 1024} МБ"
        )


def _check_write_permission(directory: Path) -> None:
    """Проверка прав записи в директорию."""
    test_file = directory / ".sprayframegen_write_test"
    try:
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
    except OSError as exc:
        raise ConfigurationError(f"export.output_directory: нет прав записи в папку: {exc}") from exc


def _create_directory(parent: Path, seed: int, started: datetime) -> Path:
    base = f"series_{started:%Y%m%d_%H%M%S}_{seed}"
    suffix = 0
    while True:
        target = parent / (base if suffix == 0 else f"{base}_{suffix:03d}")
        try:
            target.mkdir()
            return target
        except FileExistsError:
            suffix += 1


def _save_run_atomic(config: Configuration, directory: Path) -> None:
    """Атомарное сохранение run метаданных через временный файл."""
    temporary = directory / "config.json.tmp"
    save(config, temporary)
    temporary.replace(directory / "config.json")


def _save_csv_atomic(directory: Path, rows: list, frame_number: int, config: Configuration) -> None:
    """Атомарное сохранение CSV с откатом при ошибке. Заголовок пишется всегда."""
    csv_path = directory / "droplets.csv"
    temp_path = csv_path.with_suffix(".tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as f:
            import csv
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(COLUMNS)
            for r in rows:
                writer.writerow(r)
        temp_path.replace(csv_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def run_generation(
    config: Configuration,
    progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
    completion_callback: Optional[Callable[[CompletionEvent], None]] = None,
    cancellation_token: Optional[CancellationToken] = None,
) -> Path:
    """
    Запуск генерации серии с поддержкой прогресса и отмены.

    Args:
        config: Конфигурация (run должен быть None)
        progress_callback: Функция для получения событий прогресса
        completion_callback: Функция для получения события завершения
        cancellation_token: Токен для отмены генерации

    Returns:
        Путь к папке серии

    Raises:
        ExportError: При ошибке экспорта (directory атрибут содержит папку серии)
        ConfigurationError: При невалидной конфигурации
    """
    token = cancellation_token or CancellationToken()
    fresh, _ = prepare(config)

    # Валидация и проверки перед стартом
    renderer = create_renderer(fresh)
    output = fresh.export.output_directory
    if output is None or not Path(output).is_dir():
        raise ConfigurationError("export.output_directory: выберите существующую папку для серий")
    parent = Path(output).resolve()

    # Проверка диска и прав
    estimated_size = _estimate_series_size(fresh)
    _check_disk_space(parent, estimated_size)
    _check_write_permission(parent)

    started = datetime.now().astimezone()
    environment = current_environment()
    directory = _create_directory(parent, fresh.seed, started)

    run = Run("running", started.isoformat(), 0, str(directory), environment)
    current = replace(fresh, run=run)

    try:
        (directory / "frames").mkdir()
        _save_run_atomic(current, directory)

        csv_rows_buffer = []
        completed = 0
        start_time = time.time()

        for frame in generate_frames(fresh):
            if token.is_cancelled:
                raise ExportError("Генерация отменена пользователем", directory)

            # Рендер кадра
            image = encode_intensity(renderer.render(frame), fresh.camera.bit_depth)
            extension = "png" if fresh.camera.format == "png" else "tif"
            destination = directory / "frames" / f"frame_{frame.frame_number:06d}.{extension}"
            temporary = destination.with_suffix(destination.suffix + ".tmp")

            try:
                options = ({"compress_level": 6} if extension == "png" else {"compression": "raw"})
                from PIL import Image
                with Image.fromarray(image) as picture:
                    picture.save(temporary, format="PNG" if extension == "png" else "TIFF", **options)

                # Подготовка строк CSV для этого кадра
                for state in sorted(frame.droplets, key=lambda s: s.droplet.droplet_id):
                    csv_rows_buffer.append(row(frame, state, fresh.camera.scale_um_per_px))

                temporary.replace(destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise

            # Кадр и CSV строки готовы — фиксируем
            completed = frame.frame_number
            _save_csv_atomic(directory, csv_rows_buffer, completed, fresh)
            current = replace(current, run=replace(current.run, completed_frame_count=completed))
            _save_run_atomic(current, directory)

            del image
            # csv_rows_buffer НЕ очищается — накапливает все строки для атомарной перезаписи

            # Прогресс
            if progress_callback:
                elapsed = time.time() - start_time
                progress_callback(ProgressEvent(
                    frame_number=frame.frame_number,
                    total_frames=fresh.camera.frame_count,
                    completed_frames=completed,
                    elapsed_seconds=elapsed,
                    status="running",
                ))

        # Успешное завершение
        current = replace(current, run=replace(current.run, status="completed"))
        _save_run_atomic(current, directory)

        if completion_callback:
            completion_callback(CompletionEvent(
                directory=directory,
                completed_frames=completed,
                total_frames=fresh.camera.frame_count,
                status="completed",
                message="Генерация успешно завершена",
            ))

        return directory

    except ExportError:
        # Отмена — сохраняем статус cancelled
        try:
            _save_run_atomic(replace(current, run=replace(current.run, status="cancelled")), directory)
        except OSError:
            pass
        if completion_callback:
            completion_callback(CompletionEvent(
                directory=directory,
                completed_frames=completed,
                total_frames=fresh.camera.frame_count,
                status="cancelled",
                message="Генерация отменена пользователем",
            ))
        raise

    except Exception as exc:
        # Ошибка — сохраняем статус error
        try:
            _save_run_atomic(replace(current, run=replace(current.run, status="error")), directory)
        except OSError:
            pass
        message = f"Не удалось завершить серию {directory}: {exc}"
        if completion_callback:
            completion_callback(CompletionEvent(
                directory=directory,
                completed_frames=completed,
                total_frames=fresh.camera.frame_count,
                status="error",
                message=message,
            ))
        raise ExportError(message, directory) from exc


def run_generation_in_thread(
    config: Configuration,
    progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
    completion_callback: Optional[Callable[[CompletionEvent], None]] = None,
    cancellation_token: Optional[CancellationToken] = None,
) -> tuple[threading.Thread, CancellationToken]:
    """
    Запуск генерации в отдельном потоке.

    Возвращает (thread, cancellation_token).
    """
    token = cancellation_token or CancellationToken()

    def target():
        try:
            run_generation(config, progress_callback, completion_callback, token)
        except ExportError:
            # ExportError уже обработан в run_generation, completion_callback вызван
            pass
        except Exception as exc:
            # Непредвиденная ошибка
            if completion_callback:
                completion_callback(CompletionEvent(
                    directory=Path(),
                    completed_frames=0,
                    total_frames=config.camera.frame_count,
                    status="error",
                    message=f"Непредвиденная ошибка: {exc}",
                ))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, token