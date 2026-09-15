"""Общий интерфейс рендера и идеальный режим; оптические эффекты — D06."""

from .ideal import FrameRenderer, IdealRenderer, Layer, composite, create_renderer, encode_intensity

__all__ = ["FrameRenderer", "IdealRenderer", "Layer", "composite", "create_renderer", "encode_intensity"]
