from __future__ import annotations

import importlib

from .config import ImageAnalysisConfig, default_config
from . import scoring

__all__ = [
    "ImageAnalysisConfig",
    "default_config",
    "musiq",
    "duplicates",
    "scoring",
]


def __getattr__(name: str):
    """Lazy import so scoring works without Pillow or duplicate CNN deps.

    Use importlib.import_module here, not 'from . import X'. A relative import
    inside __getattr__ for the same name can re-enter __getattr__ and recurse.
    """
    if name == "musiq":
        return importlib.import_module(f"{__name__}.musiq")
    if name == "duplicates":
        return importlib.import_module(f"{__name__}.duplicates")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

