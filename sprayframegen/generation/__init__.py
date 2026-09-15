"""Управление генерацией; фиксация кадров и отмена относятся к issue #7."""

from dataclasses import asdict, replace

from ..configuration import Configuration, from_mapping
from .random_streams import RandomStreams
from ..model import Droplet
from ..model.sampling import DropletSampler


def prepare(config: Configuration) -> tuple[Configuration, RandomStreams]:
    """Новый запуск с первого кадра: старые метаданные не являются входом RNG."""
    fresh = from_mapping(asdict(replace(config, run=None))).configuration
    return fresh, RandomStreams.from_seed(fresh.seed)


def generate_initial(config: Configuration) -> tuple[Droplet, ...]:
    """Модель первого кадра; каждый вызов начинает RNG заново от seed."""
    fresh, streams = prepare(config)
    return DropletSampler(fresh, streams.model).initial_population()


def generate_frames(config: Configuration):
    """Ленивый итератор кадров с траекториями и геометрическими признаками."""
    from .sequence import generate_frames as iterate
    return iterate(config)
