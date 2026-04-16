from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _repo_root() -> Path:
    """Repository root (package lives under image_analysis_lib/)."""
    return Path(__file__).resolve().parent.parent


def _default_musiq_model_handle() -> str:
    """Path string for hub.load: local AVA SavedModel under vendor/musiq_ava/."""
    return str(_repo_root() / "vendor" / "musiq_ava")


@dataclass
class ImageAnalysisConfig:
    """
    Configuration for MUSIQ scoring and duplicate detection.

    This is intentionally simple so it can be used from both headless
    scripts (e.g. on Raspberry Pi 5) and GUI apps (e.g. on macOS).
    """

    # MUSIQ: default loads AVA from vendor/musiq_ava (gitignored). First scoring run downloads
    # from TensorFlow Hub when that directory is missing (needs network once).
    # To load from Hub URL directly instead, set musiq_model_url to the URL below.
    # musiq_model_url: str = "https://tfhub.dev/google/musiq/ava/1"
    musiq_model_url: str = field(default_factory=_default_musiq_model_handle)
    musiq_default_max_sizes: List[int] = field(default_factory=lambda: [0])

    # Duplicate detection
    musiq_csv_prefix: str = "image_evaluation_musiq_results"
    musiq_csv_default_size: int = 0
    poor_quality_threshold: float = 4.0
    min_similarity_threshold: float = 0.65
    gps_radius_meters: Optional[float] = 0.0
    best_score_threshold: float = 6.0
    tbd_best_score_threshold: float = 5.0

    # Performance / environment
    batch_size: int = 1  # simple default; callers can tune
    device: str = "cpu"  # placeholder; TensorFlow will choose appropriate device

    # Paths / caching
    cache_dir: Optional[Path] = None


default_config = ImageAnalysisConfig()

