"""Подготовка запуска и независимые RNG-потоки."""

from dataclasses import asdict, replace

from ..configuration import Configuration, from_mapping
from .random_streams import RandomStreams


def prepare(config: Configuration) -> tuple[Configuration, RandomStreams]:
    """Новый запуск с первого кадра: старые метаданные не являются входом RNG."""
    fresh = from_mapping(asdict(replace(config, run=None))).configuration
    return fresh, RandomStreams.from_seed(fresh.seed)