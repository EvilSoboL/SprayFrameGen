"""PCG64 + SeedSequence: модель (0), сенсорный шум (1)."""

from dataclasses import dataclass
from numpy.random import Generator, PCG64, SeedSequence


@dataclass
class RandomStreams:
    model: Generator
    noise: Generator

    @classmethod
    def from_seed(cls, seed: int) -> "RandomStreams":
        if type(seed) is not int or not 0 <= seed <= 4294967295:
            raise ValueError("seed: допустимо целое от 0 до 4294967295")
        model, noise = SeedSequence(seed).spawn(2)
        return cls(Generator(PCG64(model)), Generator(PCG64(noise)))
