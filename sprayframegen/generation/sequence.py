"""Последовательная модель кадров; без рендера, файлов и накопления всей серии."""

from dataclasses import dataclass, replace
from collections.abc import Iterator

from ..configuration import Configuration
from ..model import Droplet
from ..model.sampling import DropletSampler
from ..geometry.projections import Ellipse, crosses_border, is_active, overlap_flags
from .core import prepare


@dataclass(frozen=True)
class FrameDroplet:
    droplet: Droplet
    semi_major_px: float
    semi_minor_px: float
    is_new: bool
    border: bool
    overlap: bool


@dataclass(frozen=True)
class Frame:
    frame_number: int
    time_us: float
    droplets: tuple[FrameDroplet, ...]


def generate_frames(config: Configuration) -> Iterator[Frame]:
    """Каждая серия начинает model RNG заново; замены идут в порядке старых ID."""
    fresh, streams = prepare(config)
    sampler = DropletSampler(fresh, streams.model)
    camera = fresh.camera
    width, height = camera.width_px, camera.height_px
    population = sampler.initial_population()
    for frame_number in range(1, camera.frame_count + 1):
        new_ids = set()
        if frame_number == 1:
            new_ids.update(d.droplet_id for d in population)
        else:
            updated = []
            for droplet in population:
                moved = replace(
                    droplet,
                    center_x_px=droplet.center_x_px + droplet.vx_m_s * camera.frame_interval_us / camera.scale_um_per_px,
                    center_y_px=droplet.center_y_px + droplet.vy_m_s * camera.frame_interval_us / camera.scale_um_per_px,
                )
                if not is_active(Ellipse.from_droplet(moved, width), width, height):
                    moved = sampler.sample_replacement()
                    new_ids.add(moved.droplet_id)
                updated.append(moved)
            population = tuple(sorted(updated, key=lambda d: d.droplet_id))
        ellipses = tuple(Ellipse.from_droplet(d, width) for d in population)
        flags = overlap_flags(ellipses, width, height)
        states = tuple(FrameDroplet(d, e.a, e.b, d.droplet_id in new_ids,
                                   crosses_border(e, width, height), overlap)
                       for d, e, overlap in zip(population, ellipses, flags))
        yield Frame(frame_number, (frame_number - 1) * camera.frame_interval_us, states)
