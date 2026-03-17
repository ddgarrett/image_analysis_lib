from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import ImageAnalysisConfig, default_config

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - import error path
    raise ImportError(
        "Pillow is required for MUSIQ scoring. Install with: pip install Pillow"
    ) from exc

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError as exc:  # pragma: no cover - import error path
    raise ImportError(
        "TensorFlow and TensorFlow Hub are required for MUSIQ. "
        "Install with: pip install tensorflow tensorflow_hub"
    ) from exc


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".tiff",
    ".tif",
}

_MUSIQ_TF = None


def _is_ignored_path(path: Path, root: Path) -> bool:
    """
    Return True if any component of the path (file or directory)
    starts with '.' or '_', relative to the given root.
    """
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        # If the path is not under root, fall back to absolute parts
        parts = path.parts
    return any(part.startswith(".") or part.startswith("_") for part in parts)


def _load_musiq_tf(config: ImageAnalysisConfig = default_config):
    """Lazy-load MUSIQ via TensorFlow Hub."""
    global _MUSIQ_TF
    if _MUSIQ_TF is not None:
        return _MUSIQ_TF
    model = hub.load(config.musiq_model_url)
    _MUSIQ_TF = model.signatures["serving_default"]
    return _MUSIQ_TF


def _resize_image(img: Image.Image, max_size: int) -> Image.Image:
    """Resize so longest side is at most max_size, keeping aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    if w >= h:
        new_w, new_h = max_size, int(round(h * max_size / w))
    else:
        new_w, new_h = int(round(w * max_size / h)), max_size
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def image_to_jpeg_bytes(
    path: Path,
    max_size: Optional[int] = None,
    quality: int = 85,
) -> bytes:
    """Load image, optionally resize, and return as JPEG bytes suitable for MUSIQ."""
    img = Image.open(path).convert("RGB")
    if max_size is not None and max_size > 0:
        img = _resize_image(img, max_size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def score_image(
    image_path: Path,
    *,
    max_size: Optional[int] = None,
    config: ImageAnalysisConfig = default_config,
) -> Dict[str, Optional[float]]:
    """
    Score a single image with MUSIQ.

    Returns a dict with keys:
      - musiq_score: float | None (1–10, higher is better)
      - musiq_error: str | None
    """
    try:
        image_bytes = image_to_jpeg_bytes(image_path, max_size=max_size)
        predict_fn = _load_musiq_tf(config)
        inp = tf.constant(image_bytes)
        out = predict_fn(inp)
        if isinstance(out, dict):
            v = next(iter(out.values()))
        else:
            v = out
        score = float(tf.squeeze(v).numpy())
        return {"musiq_score": round(score, 4), "musiq_error": None}
    except Exception as exc:  # noqa: BLE001
        return {"musiq_score": None, "musiq_error": str(exc)}


def find_jpeg_files(root: Path) -> List[Path]:
    """Return all supported image file paths under root (main and subdirectories)."""
    out: List[Path] = []
    for p in root.rglob("*"):
        if _is_ignored_path(p, root):
            continue
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(p)
    return sorted(out)


def relative_path(path: Path, root: Path) -> str:
    """Path relative to root, using forward slashes."""
    return path.relative_to(root).as_posix()


def collect_file_info(image_path: Path, root: Path) -> Dict[str, object]:
    """Basic file and image metadata as a serialisable dict."""
    from . import musiq as _self  # avoid circular import in some tools

    stat = image_path.stat()
    try:
        with Image.open(image_path) as img:
            w, h = img.size
    except Exception:  # noqa: BLE001
        w, h = None, None
    file_size_mb = round(stat.st_size / (1024 * 1024), 4)
    return {
        "relative_path": relative_path(image_path, root),
        "file_size_bytes": stat.st_size,
        "file_size_mb": file_size_mb,
        "width": w,
        "height": h,
    }


def score_images(
    root: Path,
    images: Iterable[Path],
    *,
    max_size: Optional[int] = None,
    config: ImageAnalysisConfig = default_config,
) -> Dict[str, Dict[str, object]]:
    """
    Score a collection of images and return a mapping keyed by relative path.

    Returns: dict[relative_path] -> {
      "relative_path", "file_size_bytes", "file_size_mb",
      "width", "height", "evaluated_at", "evaluation_time_seconds",
      "max_size", "musiq_score", "musiq_error",
    }
    """
    import time

    results: Dict[str, Dict[str, object]] = {}
    images_list = list(images)
    if not images_list:
        return results

    predict_fn = _load_musiq_tf(config)

    for img_path in images_list:
        info = collect_file_info(img_path, root)
        t0 = time.perf_counter()
        try:
            image_bytes = image_to_jpeg_bytes(img_path, max_size=max_size)
            inp = tf.constant(image_bytes)
            out = predict_fn(inp)
            if isinstance(out, dict):
                v = next(iter(out.values()))
            else:
                v = out
            score = float(tf.squeeze(v).numpy())
            musiq_score = round(score, 4)
            musiq_error: Optional[str] = None
        except Exception as exc:  # noqa: BLE001
            musiq_score = None
            musiq_error = str(exc)
        elapsed = time.perf_counter() - t0
        evaluated_at = datetime.now(timezone.utc).isoformat()
        rel = info["relative_path"]
        results[rel] = {
            **info,
            "evaluated_at": evaluated_at,
            "evaluation_time_seconds": round(elapsed, 4),
            "max_size": max_size if max_size is not None else 0,
            "musiq_score": musiq_score,
            "musiq_error": musiq_error or "",
        }
    return results


def write_scores_csv_for_sizes(
    root: Path,
    images: Iterable[Path],
    max_sizes: List[int],
    output_prefix: str,
    *,
    config: ImageAnalysisConfig = default_config,
) -> List[Path]:
    """
    Evaluate MUSIQ for each image at each requested max-size, writing one CSV per size.

    Returns the list of created CSV Paths.
    """
    import csv

    images_list = list(images)
    if not images_list:
        return []

    csv_paths: List[Path] = []

    for max_size in max_sizes:
        size_label = "full" if max_size == 0 else str(max_size)
        csv_name = f"{output_prefix}_{size_label}.csv"
        csv_path = root / csv_name

        results = score_images(
            root,
            images_list,
            max_size=(max_size if max_size > 0 else None),
            config=config,
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "relative_path",
                    "file_size_bytes",
                    "file_size_mb",
                    "width",
                    "height",
                    "evaluated_at",
                    "evaluation_time_seconds",
                    "max_size",
                    "musiq_score",
                    "musiq_error",
                ]
            )
            for rel, row in sorted(results.items()):
                writer.writerow(
                    [
                        rel,
                        row["file_size_bytes"],
                        f"{row['file_size_mb']:.4f}",
                        row["width"] if row["width"] is not None else "",
                        row["height"] if row["height"] is not None else "",
                        row["evaluated_at"],
                        f"{row['evaluation_time_seconds']:.4f}",
                        row["max_size"],
                        row["musiq_score"] if row["musiq_score"] is not None else "",
                        row["musiq_error"] or "",
                    ]
                )

        csv_paths.append(csv_path)

    return csv_paths

