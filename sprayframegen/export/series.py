"""Синхронный покадровый экспорт D05. Фоновая работа/отмена/восстановление — D07."""

import csv
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PIL import Image

from ..configuration import Configuration, ConfigurationError, Run, save
from ..environment import current_environment
from ..generation import generate_frames
from ..generation.core import prepare
from ..render import create_renderer, encode_intensity
from .csv_data import COLUMNS, row


class ExportError(OSError):
    """Ошибка экспорта с путём доступной части серии."""


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


def _save_run(config: Configuration, directory: Path) -> None:
    temporary = directory / "config.json.tmp"
    save(config, temporary)
    temporary.replace(directory / "config.json")


def export_series(config: Configuration) -> Path:
    fresh, _ = prepare(config)
    renderer = create_renderer(fresh)  # Неподдерживаемый режим отклоняется до записи.
    output = fresh.export.output_directory
    if output is None or not Path(output).is_dir():
        raise ConfigurationError("export.output_directory: выберите существующую папку для серий")
    parent = Path(output).resolve()
    started = datetime.now().astimezone()
    environment = current_environment()
    directory = _create_directory(parent, fresh.seed, started)
    run = Run("running", started.isoformat(), 0, str(directory), environment)
    current = replace(fresh, run=run)
    try:
        (directory / "frames").mkdir()
        _save_run(current, directory)
        with (directory / "droplets.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(COLUMNS)
            stream.flush()
            for frame in generate_frames(fresh):
                image = encode_intensity(renderer.render(frame), fresh.camera.bit_depth)
                extension = "png" if fresh.camera.format == "png" else "tif"
                destination = directory / "frames" / f"frame_{frame.frame_number:06d}.{extension}"
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                checkpoint = stream.tell()
                try:
                    options = ({"compress_level": 6} if extension == "png" else {"compression": "raw"})
                    with Image.fromarray(image) as picture:
                        picture.save(temporary, format="PNG" if extension == "png" else "TIFF", **options)
                    for state in sorted(frame.droplets, key=lambda s: s.droplet.droplet_id):
                        writer.writerow(row(frame, state, fresh.camera.scale_um_per_px))
                    stream.flush()
                    temporary.replace(destination)
                except Exception:
                    stream.seek(checkpoint)
                    stream.truncate()
                    stream.flush()
                    temporary.unlink(missing_ok=True)
                    raise
                # Изображение и CSV готовы; сбой метаданных не отменяет готовый кадр.
                current = replace(current, run=replace(current.run, completed_frame_count=frame.frame_number))
                _save_run(current, directory)
                del image
        current = replace(current, run=replace(current.run, status="completed"))
        _save_run(current, directory)
    except Exception as exc:
        try:
            _save_run(replace(current, run=replace(current.run, status="error")), directory)
        except OSError:
            pass  # При недоступном носителе метаданные также могут быть недоступны.
        raise ExportError(f"Не удалось завершить серию {directory}: {exc}") from exc
    return directory
