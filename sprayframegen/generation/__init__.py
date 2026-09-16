"""Управление генерацией: контроллер с прогрессом/отменой, prepare, семплирование."""

from dataclasses import asdict, replace

from ..configuration import Configuration, from_mapping
from .random_streams import RandomStreams
from .core import prepare
from ..model import Droplet
from ..model.sampling import DropletSampler


def generate_initial(config: Configuration) -> tuple[Droplet, ...]:
    """Модель первого кадра; каждый вызов начинает RNG заново от seed."""
    fresh, streams = prepare(config)
    return DropletSampler(fresh, streams.model).initial_population()


def generate_frames(config: Configuration):
    """Ленивый итератор кадров с траекториями и геометрическими признаками."""
    from .sequence import generate_frames as iterate
    return iterate(config)


# Импорт контроллера в конце, чтобы избежать циклических зависимостей
from .controller import (
    CancellationToken,
    CompletionEvent,
    ExportError,
    ProgressEvent,
    run_generation,
    run_generation_in_thread,
)


__all__ = [
    "prepare",
    "generate_initial",
    "generate_frames",
    "CancellationToken",
    "ProgressEvent",
    "CompletionEvent",
    "ExportError",
    "run_generation",
    "run_generation_in_thread",
]
